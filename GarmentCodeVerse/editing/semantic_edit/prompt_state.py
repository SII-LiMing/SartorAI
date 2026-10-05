"""Shared constructor-path, neckline, and rendered-prompt state extraction.

Rule applicability and candidate generation stay in the rule modules. This
module only resolves shared GarmentCode dispatch state and the prompt
vocabulary consumed by more than one rule.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import math
import re
from typing import Any

from editing.canonical_io import (
    canonical_hash,
    design_dotted_get,
    dump_design_yaml,
)
from editing.models import (
    ProtectedStateSchemaError,
    SemanticState,
    StateUnavailableError,
)
from .prompt import PromptBuilder


class ActivePathError(ValueError):
    """Raised when an active constructor branch has an invalid design schema."""


@dataclass(frozen=True)
class ActivePaths:
    """Immutable result of resolving constructor-active design paths."""

    design_hash: str
    paths: frozenset[str]
    concepts: Mapping[str, tuple[str, ...]]
    suppressed_paths: frozenset[str]
    dispatch: Mapping[str, Any]

    def paths_for(self, concept: str) -> tuple[str, ...]:
        return tuple(self.concepts.get(concept, ()))


class _Resolver:
    def __init__(self, design: Mapping[str, Any]):
        if not isinstance(design, Mapping):
            raise ActivePathError("canonical design must be a mapping")
        self.design = design
        self.paths: set[str] = set()
        self.concepts: dict[str, tuple[str, ...]] = {}
        self.suppressed: set[str] = set()
        self.dispatch: dict[str, Any] = {}

    def read(self, path: str) -> Any:
        try:
            return design_dotted_get(self.design, path)
        except (KeyError, TypeError, ValueError) as exc:
            raise ActivePathError(
                f"active constructor path is missing or invalid: {path}"
            ) from exc

    def consume(self, path: str) -> Any:
        value = self.read(path)
        self.paths.add(path)
        return value

    def activate(self, concept: str, *paths: str) -> tuple[str, ...]:
        self.add_active(*paths)
        resolved = tuple(paths)
        self.concepts[concept] = resolved
        return resolved

    def add_active(self, *paths: str) -> None:
        for path in paths:
            self.consume(path)

    def suppress(self, *paths: str) -> None:
        self.suppressed.update(paths)

    def sleeve_branch(self, prefix: str, side: str, suppressed: bool) -> bool:
        if suppressed:
            self.suppress(f"{prefix}.length.v", f"{prefix}.cuff.type.v")
            return False

        core = (
            f"{prefix}.sleeveless.v",
            f"{prefix}.armhole_shape.v",
            f"{prefix}.connecting_width.v",
            f"{prefix}.sleeve_angle.v",
            f"{prefix}.opening_dir_mix.v",
            f"{prefix}.smoothing_coeff.v",
        )
        self.add_active(*core)
        sleeveless = self.read(f"{prefix}.sleeveless.v")
        if not isinstance(sleeveless, bool):
            raise ActivePathError(f"{prefix}.sleeveless.v must be bool")
        self.dispatch[f"{side}_sleeveless"] = sleeveless
        if sleeveless:
            self.suppress(f"{prefix}.length.v", f"{prefix}.cuff.type.v")
            return False

        # SleevePanel consumes these only after the sleeveless early return.
        self.add_active(
            f"{prefix}.length.v",
            f"{prefix}.end_width.v",
            f"{prefix}.standing_shoulder.v",
            f"{prefix}.standing_shoulder_len.v",
            f"{prefix}.connect_ruffle.v",
            f"{prefix}.cuff.type.v",
        )
        self.resolve_cuff(prefix, side)
        return True

    def resolve_cuff(self, prefix: str, side: str) -> None:
        cuff_type_path = f"{prefix}.cuff.type.v"
        cuff_type = self.read(cuff_type_path)
        allowed = {None, "CuffBand", "CuffSkirt", "CuffBandSkirt"}
        if cuff_type not in allowed:
            raise ActivePathError(f"unsupported active cuff class: {cuff_type!r}")
        self.dispatch[f"{side}_cuff"] = cuff_type
        self.concepts[f"sleeve_cuff_type:{side}"] = (cuff_type_path,)
        if cuff_type is None:
            return

        top_ruffle = f"{prefix}.cuff.top_ruffle.v"
        cuff_len = f"{prefix}.cuff.cuff_len.v"
        self.add_active(top_ruffle, cuff_len)
        shape_paths = [top_ruffle]
        if cuff_type == "CuffBandSkirt":
            shape_paths.append(f"{prefix}.cuff.skirt_fraction.v")
        if cuff_type in {"CuffSkirt", "CuffBandSkirt"}:
            shape_paths.extend(
                (
                    f"{prefix}.cuff.skirt_flare.v",
                    f"{prefix}.cuff.skirt_ruffle.v",
                )
            )
        self.add_active(*shape_paths)
        self.concepts[f"sleeve_cuff_length:{side}"] = (cuff_len,)
        self.concepts[f"sleeve_cuff_shape:{side}"] = tuple(shape_paths)

    def collar_branch(self, prefix: str, side: str, suppressed: bool) -> None:
        component_path = f"{prefix}.component.style.v"
        if suppressed:
            self.suppress(component_path)
            return
        paths = [
            f"{prefix}.f_collar.v",
            f"{prefix}.b_collar.v",
            f"{prefix}.width.v",
        ]
        if side == "right":
            paths.extend((f"{prefix}.fc_depth.v", f"{prefix}.bc_depth.v"))
        else:
            # Asymmetric Shirt.eval_dep_params creates these left values from
            # the main branch; they are intentionally absent from source YAML.
            self.suppress(f"{prefix}.fc_depth.v", f"{prefix}.bc_depth.v")
        self.add_active(*paths)
        self.concepts[f"neckline:{side}"] = tuple(paths)

    def resolve_upper(self) -> None:
        upper_path = "design.meta.upper.v"
        upper = self.consume(upper_path)
        if upper not in {None, "Shirt", "FittedShirt"}:
            raise ActivePathError(f"unsupported upper constructor: {upper!r}")
        self.dispatch["upper"] = upper
        if upper is None:
            self.dispatch.update(asymmetric=False, collar_component_suppressed=True)
            return

        asym_path = "design.left.enable_asym.v"
        asymmetric = self.consume(asym_path)
        if not isinstance(asymmetric, bool):
            raise ActivePathError(f"{asym_path} must be bool")
        self.dispatch["asymmetric"] = asymmetric

        fitted = upper == "FittedShirt"
        right_strapless = False
        left_strapless = False
        if fitted:
            right_strapless = self.consume("design.shirt.strapless.v")
            if not isinstance(right_strapless, bool):
                raise ActivePathError("design.shirt.strapless.v must be bool")
            if asymmetric:
                left_strapless = self.consume("design.left.shirt.strapless.v")
                if not isinstance(left_strapless, bool):
                    raise ActivePathError("design.left.shirt.strapless.v must be bool")

        if upper == "Shirt":
            self.activate("shirt_length", "design.shirt.length.v")
            self.add_active("design.shirt.width.v", "design.shirt.flare.v")
            if asymmetric:
                # Shirt.eval_dep_params forces left length from the main branch.
                self.add_active(
                    "design.left.shirt.width.v", "design.left.shirt.flare.v"
                )
                self.suppress("design.left.shirt.length.v")

        # The seam toggle lives in Shirt.__init__, which FittedShirt inherits.
        self.activate("open_front", "design.shirt.openfront.v")

        right_sleeve = self.sleeve_branch("design.sleeve", "right", right_strapless)
        if asymmetric:
            left_sleeve = self.sleeve_branch(
                "design.left.sleeve", "left", left_strapless
            )
            if right_sleeve:
                self.concepts["sleeve_length:right"] = ("design.sleeve.length.v",)
            if left_sleeve:
                self.concepts["sleeve_length:left"] = ("design.left.sleeve.length.v",)
            if right_sleeve and left_sleeve:
                self.concepts["sleeve_length:both"] = (
                    "design.sleeve.length.v",
                    "design.left.sleeve.length.v",
                )
                self._combine_cuff_concepts()
            active_sleeve_lengths = tuple(
                path
                for path in (
                    "design.sleeve.length.v",
                    "design.left.sleeve.length.v",
                )
                if path in self.paths
            )
            if active_sleeve_lengths:
                self.concepts["sleeve_length"] = active_sleeve_lengths
        elif right_sleeve:
            # A shared parameter creates both physical sleeves.
            self.concepts["sleeve_length:both"] = ("design.sleeve.length.v",)
            self.concepts["sleeve_length"] = ("design.sleeve.length.v",)
            for suffix in ("type", "length", "shape"):
                source = self.concepts.get(f"sleeve_cuff_{suffix}:right")
                if source:
                    self.concepts[f"sleeve_cuff_{suffix}:both"] = source

        self.collar_branch("design.collar", "right", right_strapless)
        if asymmetric:
            self.collar_branch("design.left.collar", "left", left_strapless)
            # Shirt.eval_dep_params explicitly replaces both component styles.
            self.suppress(
                "design.collar.component.style.v",
                "design.left.collar.component.style.v",
            )
            self.dispatch["collar_component_suppressed"] = True
        elif not right_strapless:
            component = "design.collar.component.style.v"
            self.activate("hood_style", component)
            self.concepts["collar_component"] = (component,)
            self.dispatch["collar_component_suppressed"] = False
        else:
            self.dispatch["collar_component_suppressed"] = True

    def _combine_cuff_concepts(self) -> None:
        for suffix in ("type", "length", "shape"):
            right = self.concepts.get(f"sleeve_cuff_{suffix}:right", ())
            left = self.concepts.get(f"sleeve_cuff_{suffix}:left", ())
            if right and left:
                self.concepts[f"sleeve_cuff_{suffix}:both"] = right + left

    def resolve_bottom(self) -> None:
        bottom_path = "design.meta.bottom.v"
        waistband_path = "design.meta.wb.v"
        bottom = self.consume(bottom_path)
        waistband = self.consume(waistband_path)
        allowed_bottoms = {
            None,
            "Pants",
            "Skirt2",
            "PencilSkirt",
            "SkirtCircle",
            "AsymmSkirtCircle",
            "SkirtManyPanels",
            "GodetSkirt",
            "SkirtLevels",
        }
        if bottom not in allowed_bottoms:
            raise ActivePathError(f"unsupported bottom constructor: {bottom!r}")
        if waistband not in {None, "StraightWB", "FittedWB"}:
            raise ActivePathError(f"unsupported waistband constructor: {waistband!r}")
        self.dispatch["bottom"] = bottom
        self.dispatch["waistband"] = waistband
        self.dispatch["nested_skirt"] = bottom in {"GodetSkirt", "SkirtLevels"}

        if waistband is not None:
            self.add_active("design.waistband.waist.v", "design.waistband.width.v")

        skirt_base = None
        length_path = None
        if bottom == "Pants":
            self.activate("pants_length", "design.pants.length.v")
        elif bottom == "Skirt2":
            skirt_base = "Skirt2"
            length_path = "design.skirt.length.v"
            self.add_active(
                "design.skirt.ruffle.v",
                "design.skirt.flare.v",
                "design.skirt.bottom_cut.v",
            )
        elif bottom == "PencilSkirt":
            skirt_base = "PencilSkirt"
            length_path = "design.pencil-skirt.length.v"
            slits = tuple(
                f"design.pencil-skirt.{side}_slit.v"
                for side in ("front", "back", "left", "right")
            )
            self.activate("pencil_slits", *slits)
            self.add_active("design.pencil-skirt.flare.v")
        elif bottom in {"SkirtCircle", "AsymmSkirtCircle", "SkirtManyPanels"}:
            skirt_base = bottom
            length_path = "design.flare-skirt.length.v"
            self.add_active("design.flare-skirt.suns.v")
            if bottom in {"SkirtCircle", "AsymmSkirtCircle"}:
                cut_paths = (
                    "design.flare-skirt.cut.add.v",
                    "design.flare-skirt.cut.depth.v",
                    "design.flare-skirt.cut.width.v",
                    "design.flare-skirt.cut.place.v",
                )
                self.activate("flare_cut", *cut_paths)
            if bottom == "AsymmSkirtCircle":
                self.add_active("design.flare-skirt.asymm.front_length.v")
            if bottom == "SkirtManyPanels":
                self.add_active(
                    "design.flare-skirt.skirt-many-panels.n_panels.v",
                    "design.flare-skirt.skirt-many-panels.panel_curve.v",
                )
        elif bottom == "GodetSkirt":
            base_path = "design.godet-skirt.base.v"
            skirt_base = self.consume(base_path)
            if skirt_base not in {"Skirt2", "PencilSkirt"}:
                raise ActivePathError(
                    f"unsupported GodetSkirt base constructor: {skirt_base!r}"
                )
            self.add_active(
                "design.godet-skirt.insert_w.v",
                "design.godet-skirt.insert_depth.v",
                "design.godet-skirt.num_inserts.v",
                "design.godet-skirt.cuts_distance.v",
            )
            if skirt_base == "Skirt2":
                length_path = "design.skirt.length.v"
                self.add_active("design.skirt.ruffle.v", "design.skirt.flare.v")
                self.suppress("design.skirt.bottom_cut.v")
            else:
                length_path = "design.pencil-skirt.length.v"
                self.add_active("design.pencil-skirt.flare.v")
                self.suppress(
                    *(
                        f"design.pencil-skirt.{side}_slit.v"
                        for side in ("front", "back", "left", "right")
                    )
                )
        elif bottom == "SkirtLevels":
            base_path = "design.levels-skirt.base.v"
            level_path = "design.levels-skirt.level.v"
            skirt_base = self.consume(base_path)
            level_class = self.consume(level_path)
            supported_bases = {
                "Skirt2",
                "PencilSkirt",
                "SkirtCircle",
                "AsymmSkirtCircle",
            }
            if skirt_base not in supported_bases:
                raise ActivePathError(
                    f"unsupported SkirtLevels base constructor: {skirt_base!r}"
                )
            if level_class not in {"Skirt2", "SkirtCircle", "AsymmSkirtCircle"}:
                raise ActivePathError(
                    f"unsupported SkirtLevels level constructor: {level_class!r}"
                )
            self.dispatch["skirt_level"] = level_class
            length_path = "design.levels-skirt.length.v"
            self.add_active(
                "design.levels-skirt.num_levels.v",
                "design.levels-skirt.level_ruffle.v",
                "design.levels-skirt.base_length_frac.v",
            )
            if skirt_base == "Skirt2":
                # The base retains top gathering; only its length is forced.
                self.add_active("design.skirt.ruffle.v", "design.skirt.flare.v")
            elif skirt_base == "PencilSkirt":
                self.add_active("design.pencil-skirt.flare.v")
            elif skirt_base in {"SkirtCircle", "AsymmSkirtCircle"}:
                self.add_active("design.flare-skirt.suns.v")
            if level_class == "Skirt2":
                self.add_active("design.skirt.flare.v")
                if skirt_base != "Skirt2":
                    # Level constructors pass top_ruffles=False.
                    self.suppress("design.skirt.ruffle.v")
            elif level_class in {"SkirtCircle", "AsymmSkirtCircle"}:
                self.add_active("design.flare-skirt.suns.v")
            if "AsymmSkirtCircle" in {skirt_base, level_class}:
                self.add_active("design.flare-skirt.asymm.front_length.v")
            # Explicit constructor arguments replace base/level lengths and
            # disable their slits/cuts.  Skirt2 level gathering is also forced.
            self.suppress(
                "design.skirt.length.v",
                "design.pencil-skirt.length.v",
                "design.flare-skirt.length.v",
                "design.skirt.bottom_cut.v",
                "design.flare-skirt.cut.add.v",
                "design.flare-skirt.cut.depth.v",
                "design.flare-skirt.cut.width.v",
                "design.flare-skirt.cut.place.v",
                *(
                    f"design.pencil-skirt.{side}_slit.v"
                    for side in ("front", "back", "left", "right")
                ),
            )

        self.dispatch["skirt_base"] = skirt_base
        if length_path is not None:
            self.activate("skirt_length", length_path)

    def finish(self) -> ActivePaths:
        overlap = self.paths.intersection(self.suppressed)
        if overlap:
            raise ActivePathError(
                "paths resolved as both active and suppressed: "
                + ", ".join(sorted(overlap))
            )
        return ActivePaths(
            design_hash=canonical_hash(self.design),
            paths=frozenset(self.paths),
            concepts=dict(sorted(self.concepts.items())),
            suppressed_paths=frozenset(self.suppressed),
            dispatch=dict(sorted(self.dispatch.items())),
        )


def resolve_active_paths(design: Mapping[str, Any]) -> ActivePaths:
    """Resolve constructor-active paths without constructing a pattern.

    The logic mirrors the dispatch and early-return decisions in
    ``MetaGarment``, ``Shirt``/``FittedShirt``, ``Sleeve``, ``GodetSkirt`` and
    ``SkirtLevels``.  Only branches selected by the canonical design are
    required to exist.
    """

    resolver = _Resolver(design)
    resolver.resolve_upper()
    resolver.resolve_bottom()
    return resolver.finish()

COMPONENT_STYLE_PATH = "design.collar.component.style.v"
FRONT_NECKLINE_PATH = "design.collar.f_collar.v"


def _unavailable(code: str, message: str) -> StateUnavailableError:
    return StateUnavailableError(code, message)


def _number(design: Mapping[str, Any], path: str) -> float:
    try:
        value = design_dotted_get(design, path)
        if isinstance(value, bool):
            raise TypeError
        result = float(value)
    except (KeyError, TypeError, ValueError) as exc:
        raise _unavailable(
            "state_schema_unavailable",
            f"active state path must be finite numeric: {path}",
        ) from exc
    if not math.isfinite(result):
        raise _unavailable(
            "state_schema_unavailable",
            f"active state path must be finite numeric: {path}",
        )
    return result


def neckline_inputs(design: Mapping[str, Any]) -> dict[str, Any]:
    """Read the local parameters used by the neckline phrase contract."""

    try:
        component = design_dotted_get(design, COMPONENT_STYLE_PATH)
        constructor = str(design_dotted_get(design, FRONT_NECKLINE_PATH) or "")
        standing = bool(
            design_dotted_get(
                design, "design.collar.component.lapel_standing.v"
            )
        )
        flip = bool(
            design_dotted_get(design, "design.collar.f_flip_curve.v")
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise _unavailable(
            "state_schema_unavailable",
            "active collar state paths are missing or invalid",
        ) from exc
    if component not in {None, "SimpleLapel", "Hood2Panels", "Turtle"}:
        raise _unavailable(
            "collar_component_style_schema_unavailable",
            f"unsupported collar component style: {component!r}",
        )
    return {
        "constructor": constructor,
        "component": component,
        "depth": _number(design, "design.collar.fc_depth.v"),
        "panel_depth": _number(design, "design.collar.component.depth.v"),
        "standing": standing,
        "width": _number(design, "design.collar.width.v"),
        "bezier_x": _number(design, "design.collar.f_bezier_x.v"),
        "bezier_y": _number(design, "design.collar.f_bezier_y.v"),
        "flip": flip,
    }


def protected_neckline_phrase(inputs: Mapping[str, Any]) -> str:
    """Return the exact protected-prompt phrase implied by local inputs."""

    component = inputs["component"]
    constructor = inputs["constructor"]
    depth = inputs["depth"]
    panel_depth = inputs["panel_depth"]
    standing = inputs["standing"]
    width = inputs["width"]
    bezier_x = inputs["bezier_x"]
    bezier_y = inputs["bezier_y"]
    flip = inputs["flip"]
    if component == "Hood2Panels":
        return "hood"
    if component == "SimpleLapel":
        if (
            standing
            and constructor == "CircleNeckHalf"
            and width < 0.20
            and depth >= 0.68
        ):
            return "split stand V-neck collar"
        if constructor == "CircleNeckHalf" and depth >= 0.58:
            return "fold-over collar framing a deep V-shaped neckline"
        if constructor == "CircleNeckHalf" and panel_depth >= 5.5:
            return "fold-over V-neck collar"
        if not standing and width >= 0.65:
            return "wide fold-over collar framing a shallow V-shaped neckline"
        if panel_depth >= 6.0 or depth >= 0.65:
            return "wide fold-over collar"
        if not standing and panel_depth >= 3.0:
            return "soft fold-over V-neck collar"
        if not standing:
            return "V-neckline"
        return (
            "split stand V-neck collar"
            if constructor == "VNeckHalf"
            else "split stand collar"
        )
    if constructor == "CircleNeckHalf":
        if depth >= 0.58:
            return "deep scoop neckline"
        return "scoop neckline" if depth >= 0.48 else "round neckline"
    if constructor == "CurvyNeckHalf":
        if depth <= 0.38:
            return "notched V-neckline"
        return (
            "V-neckline"
            if flip or bezier_x >= 0.75
            else "softly curved neckline"
        )
    if constructor == "VNeckHalf":
        return "deep V-neck" if depth > 0.60 else "V-neckline"
    if constructor == "SquareNeckHalf":
        return "square neckline"
    if constructor == "CircleArcNeckHalf":
        if width < 0.0:
            return "V-neckline"
        return (
            "rounded neckline featuring a shallow center dip"
            if depth <= 0.36
            else "rounded arc neckline"
        )
    if constructor == "Bezier2NeckHalf":
        if flip and bezier_y >= 0.40:
            return "split neckline featuring a V-shaped notch"
        if depth >= 0.60:
            return "V-neckline"
        return (
            "notched V-neckline"
            if depth >= 0.38 and bezier_y <= 0.35
            else "curved neckline"
        )
    raise _unavailable(
        "neckline_constructor_unavailable",
        f"unsupported neckline constructor: {constructor!r}",
    )


COLLAR_PHRASES = (
    "fold-over collar framing a deep V-shaped neckline",
    "wide fold-over collar framing a shallow V-shaped neckline",
    "soft fold-over V-neck collar",
    "split stand V-neck collar",
    "fold-over V-neck collar",
    "wide fold-over collar",
    "split stand collar",
)
NECKLINE_PHRASES = (
    "rounded neckline featuring a shallow center dip",
    "split neckline featuring a V-shaped notch",
    "deep scoop neckline",
    "softly curved neckline",
    "notched V-neckline",
    "rounded arc neckline",
    "scoop neckline",
    "round neckline",
    "square neckline",
    "curved neckline",
    "deep V-neck",
    "V-neckline",
)
_FIT_ANCHOR_RE = re.compile(
    r"(?<![A-Za-z0-9-])"
    r"(close-fitting|loose)"
    r"\s+(?:underbust-length|cropped|waist-length|high-hip-length|"
    r"hip-length|tunic-length)\s+top"
    r"(?![A-Za-z0-9-])"
)


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return value


def _phrase_matches(text: str, vocabulary: Sequence[str]) -> tuple[str, ...]:
    matches: list[tuple[int, int, str]] = []
    for phrase in vocabulary:
        pattern = re.compile(
            rf"(?<![A-Za-z0-9-]){re.escape(phrase)}(?![A-Za-z0-9-])"
        )
        matches.extend(
            (match.start(), match.end(), phrase)
            for match in pattern.finditer(text)
        )
    maximal = [
        item
        for item in matches
        if not any(
            other[0] <= item[0]
            and other[1] >= item[1]
            and other[1] - other[0] > item[1] - item[0]
            for other in matches
        )
    ]
    return tuple(
        phrase
        for _, _, phrase in sorted(
            set(maximal), key=lambda item: (item[0], item[1], item[2])
        )
    )


def _single_match(
    text: str,
    vocabulary: Sequence[str],
    feature: str,
) -> str | None:
    matches = _phrase_matches(text, vocabulary)
    if len(matches) > 1:
        raise ProtectedStateSchemaError(
            "ambiguous_protected_state",
            f"ambiguous {feature} anchors: {list(matches)}",
        )
    return matches[0] if matches else None


def _selected_upper_prompt(document: Mapping[str, Any]) -> str | None:
    parts = _mapping(_mapping(document, "prompt_document").get("parts"), "parts")
    selected = [name for name in ("upper", "whole") if name in parts]
    if len(selected) > 1:
        raise ProtectedStateSchemaError(
            "ambiguous_upper_representation",
            "protected prompt contains both parts.upper and parts.whole",
        )
    if not selected:
        return None
    part = _mapping(parts[selected[0]], f"parts.{selected[0]}")
    text = part.get("part_prompt")
    if not isinstance(text, str) or not text or "\x00" in text:
        raise ProtectedStateSchemaError(
            "protected_part_prompt_invalid",
            f"parts.{selected[0]}.part_prompt must be non-empty text",
        )
    return text


def _active(design: Mapping[str, Any]) -> Any:
    try:
        return resolve_active_paths(design)
    except (ActivePathError, KeyError, TypeError, ValueError) as exc:
        raise StateUnavailableError(
            "active_path_unavailable",
            f"cannot resolve active constructor paths: {exc}",
        ) from exc


def _strapless(design: Mapping[str, Any], active: Any) -> bool:
    return active.dispatch.get("upper") == "FittedShirt" and bool(
        design_dotted_get(design, "design.shirt.strapless.v")
    )


def extract_prompt_states(
    design: Mapping[str, Any],
    prompt_document: Mapping[str, Any],
) -> Mapping[str, SemanticState]:
    """Parse explicit shared states without invoking any rule classifier."""

    active = _active(design)
    text = _selected_upper_prompt(prompt_document)
    if text is None or active.dispatch.get("upper") is None:
        return {}

    states: dict[str, SemanticState] = {}
    if active.dispatch.get("upper") == "Shirt":
        fits = tuple(match.group(1) for match in _FIT_ANCHOR_RE.finditer(text))
        if len(fits) > 1:
            raise ProtectedStateSchemaError(
                "ambiguous_protected_state",
                f"ambiguous shirt fit anchors: {list(fits)}",
            )
        fit = fits[0] if fits else "unnamed"
        states["shirt_fit"] = SemanticState(
            "shirt_fit",
            fit,
            fit,
            {"protected_phrase_present": bool(fits)},
        )

    if (
        active.dispatch.get("asymmetric") is True
        or _strapless(design, active)
        or active.dispatch.get("collar_component_suppressed") is True
    ):
        return dict(sorted(states.items()))

    hood = _single_match(text, ("hood",), "hood")
    states["hood_presence"] = SemanticState(
        "hood_presence",
        "hooded" if hood else "hoodless",
        "hooded" if hood else "hoodless",
        (
            {"present": True, "protected_phrase": "hood"}
            if hood
            else {"present": False}
        ),
    )

    collar = _single_match(text, COLLAR_PHRASES, "collar")
    neckline = _single_match(text, NECKLINE_PHRASES, "neckline")
    if hood and (collar or neckline):
        raise ProtectedStateSchemaError(
            "protected_state_mismatch",
            "hood prompt conflicts with a collar or neckline anchor",
        )
    if collar and neckline:
        raise ProtectedStateSchemaError(
            "protected_state_mismatch",
            "collar prompt also contains a separate neckline anchor",
        )
    if collar:
        states["collar_presence"] = SemanticState(
            "collar_presence",
            "collared",
            f"having a {collar}",
            {
                "present": True,
                "component_style": "SimpleLapel",
                "subtype_phrase": collar,
            },
        )
    if neckline:
        constructor = design_dotted_get(design, "design.collar.f_collar.v")
        if not isinstance(constructor, str) or not constructor:
            raise ProtectedStateSchemaError(
                "protected_state_mismatch",
                "neckline prompt requires an active front-neckline constructor",
            )
        states["neckline_type"] = SemanticState(
            "neckline_type",
            constructor,
            neckline,
            {"constructor": constructor},
        )
    return dict(sorted(states.items()))


@dataclass(slots=True)
class PromptStateClassifier:
    """Render once and return only states observed in the protected prompt."""

    builder: PromptBuilder
    output_root: Path

    def __post_init__(self) -> None:
        self.output_root = Path(self.output_root)

    def __call__(
        self,
        design: dict[str, Any] | Mapping[str, Any],
        body_yaml: bytes,
    ) -> Mapping[str, SemanticState]:
        artifact = self.builder.render(
            dump_design_yaml(design),
            body_yaml,
            output_root=self.output_root,
        )
        return extract_prompt_states(design, artifact.document)
