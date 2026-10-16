"""Build per-case split-aware GCD garment and body prompts."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any
import yaml

NECKLINE_MAP = {
    "CircleNeckHalf": "round neckline",
    "CurvyNeckHalf": "softly curved neckline",
    "VNeckHalf": "V-neckline",
    "SquareNeckHalf": "square neckline",
    "TrapezoidNeckHalf": "trapezoid neckline",
    "CircleArcNeckHalf": "rounded arc neckline",
    "Bezier2NeckHalf": "curved neckline",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one prompt JSON per GarmentCode case."
    )
    parser.add_argument(
        "--root", required=True, help="random_body directory or batch directory."
    )
    parser.add_argument("--log", required=True)
    return parser.parse_args()


def is_nullish(value: Any) -> bool:
    return value is None or str(value).strip().lower() in {"", "none", "null"}


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if is_nullish(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def get_v(data: dict[str, Any], path: list[str], default: Any = None) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]

    if isinstance(cur, dict) and "v" in cur:
        value = cur["v"]
        return default if is_nullish(value) else value

    return default if is_nullish(cur) else cur


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def body_float(
    body: dict[str, Any],
    key: str,
) -> float:
    if key not in body or is_nullish(body[key]):
        raise KeyError(f"Missing body measurement: {key}")
    return float(body[key])


def with_indefinite_article(text: str) -> str:
    article = "an" if text[:1].lower() in "aeiou" else "a"
    return f"{article} {text}"


def clean_name(value: Any) -> str:
    if is_nullish(value):
        return ""

    text = str(value)
    replacements = {
        "FittedShirt": "top",
        "Shirt": "top",
        "StraightWB": "straight waistband",
        "FittedWB": "fitted waistband",
        "SkirtCircle": "circle skirt",
        "AsymmSkirtCircle": "asymmetric circle skirt",
        "GodetSkirt": "godet skirt",
        "Skirt2": "skirt",
        "SkirtManyPanels": "paneled skirt",
        "PencilSkirt": "pencil skirt",
        "SkirtLevels": "tiered skirt",
        "Pants": "pants",
        "CuffBand": "cuff band",
        "CuffSkirt": "ruffled cuff",
        "CuffBandSkirt": "banded ruffled cuff",
    }
    if text in replacements:
        return replacements[text]

    out = []
    for i, char in enumerate(text):
        if i > 0 and char.isupper() and text[i - 1].islower():
            out.append(" ")
        out.append(char)
    return "".join(out).replace("_", " ").replace("-", " ").lower()


def shirt_length_word(
    value: Any,
    body: dict[str, Any],
) -> str:
    length_ratio = to_float(value, 1.0)
    waist_line = body_float(body, "waist_line")
    hips_line = body_float(body, "hips_line")
    if length_ratio < 0.70:
        return "underbust-length"
    if length_ratio < 0.95:
        return "cropped"
    if length_ratio < 1.15:
        return "waist-length"

    length_cm = length_ratio * waist_line
    below_waist = length_cm - waist_line
    if below_waist <= 0.35 * hips_line:
        return "high-hip-length"
    if below_waist <= 1.05 * hips_line:
        return "hip-length"
    return "tunic-length"


def sleeve_style(
    sleeveless: Any,
    strapless: Any,
    sleeve_length: Any,
    connecting_width: Any = 0.0,
) -> tuple[str, str]:
    if as_bool(strapless):
        return "strapless", ""
    if as_bool(sleeveless):
        return "sleeveless", ""

    length = to_float(sleeve_length, 0.5)
    if length < 0.35:
        length_index = 0
    elif length < 0.72:
        length_index = 1
    elif length < 0.94:
        length_index = 2
    else:
        length_index = 3

    # A deep armhole starts the sleeve farther down the arm. GarmentCode's
    # length parameter only covers the distance remaining below that opening.
    if to_float(connecting_width) >= 1.0:
        length_index = min(length_index + 1, 3)

    return "", ("short", "elbow-length", "three-quarter-length", "long")[length_index]


def sleeve_shape_word(
    end_width: Any,
    connecting_width: Any,
    connect_ruffle: Any,
    cuff_type: Any,
    cuff_top_ruffle: Any,
    cuff_length: Any,
    cuff_skirt_fraction: Any,
    cuff_skirt_flare: Any,
    sleeve_length: Any,
    body: dict[str, Any],
) -> str:
    width = to_float(end_width, 1.0)
    connection = to_float(connecting_width, 0.2)
    armhole_ruffle = to_float(connect_ruffle, 1.0)
    cuff_ruffle = to_float(cuff_top_ruffle, 1.0)
    cuff = str(cuff_type or "")
    cuff_cm = body_float(body, "arm_length") * min(
        max(0.0, to_float(cuff_length)),
        0.70 * max(0.0, to_float(sleeve_length)),
    )
    skirt_fraction = to_float(cuff_skirt_fraction, 0.5)
    band_cm = cuff_cm
    if cuff == "CuffBandSkirt":
        band_cm *= 1.0 - skirt_fraction

    if (
        cuff in {"CuffBand", "CuffBandSkirt"}
        and band_cm >= 1.5
        and cuff_ruffle >= 1.25
        and width >= 0.85
    ):
        return "balloon"

    visible_flared_cuff = (
        cuff == "CuffSkirt"
        and cuff_cm >= 2.0
        and to_float(cuff_skirt_flare, 1.0) >= 1.30
    )
    if (connection >= 1.0 and width >= 0.90) or visible_flared_cuff:
        return "bell"
    if width > 1.20:
        return "puffed wide" if armhole_ruffle >= 1.65 else "wide"
    if width > 1.05:
        return "wide"
    if width < 0.75:
        return "tapered"
    if armhole_ruffle >= 1.65:
        return "puffed"
    return ""


def sleeve_cuff_word(
    cuff_type: Any,
    cuff_top_ruffle: Any,
    cuff_length: Any,
    cuff_skirt_fraction: Any,
    cuff_skirt_flare: Any,
    cuff_skirt_ruffle: Any,
    sleeve_length: Any,
    body: dict[str, Any],
    plural: bool = True,
) -> str:
    cuff = str(cuff_type or "")
    top_ruffle = to_float(cuff_top_ruffle, 1.0)
    length_fraction = max(0.0, to_float(cuff_length, 0.0))
    sleeve_fraction = max(0.0, to_float(sleeve_length, 0.0))
    arm_length = body_float(body, "arm_length")
    cuff_cm = arm_length * min(length_fraction, 0.70 * sleeve_fraction)
    if cuff_cm < 2.0:
        return ""

    noun = "cuffs" if plural else "cuff"
    if cuff == "CuffBandSkirt":
        skirt_fraction = to_float(cuff_skirt_fraction, 0.5)
        skirt_cm = cuff_cm * skirt_fraction
        if skirt_cm >= 1.5 and to_float(cuff_skirt_flare, 1.0) >= 1.30:
            return f"banded flounced {noun}"
        if skirt_cm >= 1.5 and to_float(cuff_skirt_ruffle, 1.0) >= 1.25:
            return f"banded ruffled {noun}"
        if top_ruffle >= 1.25:
            return f"gathered band {noun}"
        return f"banded {noun}"
    if cuff == "CuffSkirt":
        if to_float(cuff_skirt_flare, 1.0) >= 1.30:
            return f"flounced {noun}"
        return ""
    if cuff == "CuffBand":
        if top_ruffle >= 1.25:
            return f"gathered band {noun}"
        return f"banded {noun}"
    return ""


def describe_sleeve(
    sleeve_length: Any,
    end_width: Any,
    connecting_width: Any,
    connect_ruffle: Any,
    cuff_type: Any,
    cuff_top_ruffle: Any,
    cuff_length: Any,
    cuff_skirt_fraction: Any,
    cuff_skirt_flare: Any,
    cuff_skirt_ruffle: Any,
    body: dict[str, Any],
    *,
    plural: bool,
) -> str:
    _, length_detail = sleeve_style(
        False,
        False,
        sleeve_length,
        connecting_width,
    )
    shape = sleeve_shape_word(
        end_width,
        connecting_width,
        connect_ruffle,
        cuff_type,
        cuff_top_ruffle,
        cuff_length,
        cuff_skirt_fraction,
        cuff_skirt_flare,
        sleeve_length,
        body,
    )
    noun = "sleeves" if plural else "sleeve"
    description = " ".join(word for word in (length_detail, shape, noun) if word)
    cuff = sleeve_cuff_word(
        cuff_type,
        cuff_top_ruffle,
        cuff_length,
        cuff_skirt_fraction,
        cuff_skirt_flare,
        cuff_skirt_ruffle,
        sleeve_length,
        body,
        plural=plural,
    )
    if cuff:
        article = "" if plural else "a "
        description += f" with {article}{cuff}"
    return description


def shirt_fit_word(
    width: Any,
    flare: Any,
    length: Any,
    body: dict[str, Any],
) -> str:
    width_value = to_float(width, 1.0)
    flare_value = to_float(flare, 1.0)
    length_value = to_float(length, 1.0)

    bust = body_float(body, "bust")
    waist = body_float(body, "waist")
    hips = body_float(body, "hips")
    waist_line = body_float(body, "waist_line")
    hips_line = body_float(body, "hips_line")
    garment_bust = width_value * bust
    garment_hem = flare_value * garment_bust
    below_waist = max(0.0, length_value * waist_line - waist_line)
    hip_progress = min(1.0, below_waist / hips_line)
    body_at_hem = waist + hip_progress * (hips - waist)
    bust_ease = garment_bust / bust
    hem_ease = garment_hem / body_at_hem

    if length_value < 1.0:
        if bust_ease < 0.88 and flare_value < 1.20:
            return "close-fitting"
        return ""
    if flare_value >= 1.45:
        return "loose"
    if bust_ease < 0.88 and flare_value < 1.25:
        return "close-fitting"
    if bust_ease > 1.15 and hem_ease > 1.10:
        return "loose"
    return ""


def silhouette_word(flare: Any, noun: str) -> str:
    flare = to_float(flare, 1.0)
    if flare > 1.35:
        return f"flared {noun}"
    if flare > 1.15:
        return f"slightly flared {noun}"
    if flare < 0.85:
        return f"tapered {noun}"
    return f"straight-cut {noun}"


def neckline_word(
    front_collar: Any,
    component_style: Any,
    fc_depth: Any,
    component_depth: Any,
    component_standing: Any,
    collar_width: Any,
    front_bezier_x: Any,
    front_bezier_y: Any,
    front_flip_curve: Any,
) -> str:
    component = str(component_style or "")
    collar = str(front_collar or "")
    depth = to_float(fc_depth, 0.0)
    panel_depth = to_float(component_depth, 0.0)
    standing = as_bool(component_standing)
    raw_collar_width = to_float(collar_width, 0.2)
    bezier_x = to_float(front_bezier_x, 0.3)
    bezier_y = to_float(front_bezier_y, 0.5)
    flip_curve = as_bool(front_flip_curve)

    if component == "Hood2Panels":
        return "hood"
    if component == "SimpleLapel":
        if (
            standing
            and collar == "CircleNeckHalf"
            and raw_collar_width < 0.20
            and depth >= 0.68
        ):
            return "split stand V-neck collar"
        if collar == "CircleNeckHalf" and depth >= 0.58:
            return "fold-over collar framing a deep V-shaped neckline"
        if collar == "CircleNeckHalf" and panel_depth >= 5.5:
            return "fold-over V-neck collar"
        if not standing and raw_collar_width >= 0.65:
            return "wide fold-over collar framing a shallow V-shaped neckline"
        if panel_depth >= 6.0 or depth >= 0.65:
            return "wide fold-over collar"
        if not standing:
            if panel_depth >= 3.0:
                return "soft fold-over V-neck collar"
            return "V-neckline"
        if collar == "VNeckHalf":
            return "split stand V-neck collar"
        return "split stand collar"
    # A Turtle component adds a separate collar band, but the underlying
    # neckline cut is still visible when that band folds or drapes down.
    # Fall through so the base neckline is never discarded.

    if collar == "CircleNeckHalf":
        if depth >= 0.58:
            return "deep scoop neckline"
        if depth >= 0.48:
            return "scoop neckline"

    if collar == "CircleArcNeckHalf":
        if raw_collar_width < 0.0:
            return "V-neckline"
        if depth <= 0.36:
            return "rounded neckline featuring a shallow center dip"

    if collar == "VNeckHalf" and depth > 0.6:
        return "deep V-neck"

    if collar == "TrapezoidNeckHalf":
        if raw_collar_width >= 0.85 and bezier_y <= 0.20:
            return "draped cowl neckline"
        if depth >= 0.65:
            return "plunging U-neckline"
        return "square neckline"

    if collar == "Bezier2NeckHalf":
        if flip_curve and bezier_y >= 0.40:
            return "split neckline featuring a V-shaped notch"
        if depth >= 0.60:
            return "V-neckline"
        if depth >= 0.38 and bezier_y <= 0.35:
            return "notched V-neckline"

    if collar == "CurvyNeckHalf":
        if depth <= 0.38:
            return "notched V-neckline"
        if flip_curve or bezier_x >= 0.75:
            return "V-neckline"

    return NECKLINE_MAP.get(collar, "")


def skirt_length_label(effective: float) -> str:
    if effective < 0.45:
        return "mini"
    if effective < 0.55:
        return "above-knee"
    if effective < 0.70:
        return "knee-length"
    if effective < 0.97:
        return "midi-length"
    return "floor-length"


def skirt_length_word(value: Any, rise: Any = 0.75) -> str:
    effective = to_float(value, 0.65) + 0.20 * to_float(rise, 0.75)
    return skirt_length_label(effective)


def body_skirt_length_word(
    length_cm: float,
    body: dict[str, Any],
    *,
    ankle_threshold: float = 0.82,
) -> str:
    waist_level = body_float(body, "_waist_level")
    relative_length = length_cm / waist_level
    if relative_length < 0.40:
        return "mini"
    if relative_length < 0.46:
        return "above-knee"
    if relative_length < 0.60:
        return "knee-length"
    if relative_length < ankle_threshold:
        return "midi-length"
    if relative_length < 0.88:
        return "ankle-length"
    return "floor-length"


def body_bottom_length_ratio(
    value: Any,
    rise: Any,
    body: dict[str, Any],
) -> float:
    length_cm = to_float(rise, 0.75) * body_float(body, "hips_line") + to_float(
        value, 0.65
    ) * body_float(body, "_leg_length")
    return length_cm / body_float(body, "_waist_level")


def asymmetric_skirt_length_words(
    value: Any,
    rise: Any,
    front_ratio: Any,
    body: dict[str, Any],
) -> tuple[str, str]:
    length_value = to_float(value, 0.65)
    rise_value = to_float(rise, 0.75)
    front_value = to_float(front_ratio, 1.0)
    hips_line = body_float(body, "hips_line")
    leg_length = body_float(body, "_leg_length")

    back_length_cm = max(
        rise_value * hips_line + length_value * leg_length,
        5.0,
    )
    front_length_cm = front_value * back_length_cm
    return (
        body_skirt_length_word(
            front_length_cm,
            body,
            ankle_threshold=0.78,
        ),
        body_skirt_length_word(
            back_length_cm,
            body,
            ankle_threshold=0.78,
        ),
    )


def straight_skirt_length_word(
    value: Any,
    rise: Any,
    body: dict[str, Any],
) -> str:
    length_cm = body_bottom_length_ratio(value, rise, body) * body_float(
        body,
        "_waist_level",
    )
    return body_skirt_length_word(length_cm, body)


def pencil_skirt_length_word(
    value: Any,
    rise: Any,
    body: dict[str, Any],
) -> str:
    return straight_skirt_length_word(value, rise, body)


def paneled_skirt_length_word(value: Any, rise: Any, suns: Any) -> str:
    effective = (
        to_float(value, 0.65) + 0.20 * to_float(rise, 0.75) + 0.15 * to_float(suns, 0.5)
    )
    if effective < 0.45:
        return "mini"
    if effective < 0.55:
        return "above-knee"
    if effective < 0.66:
        return "knee-length"
    if effective < 0.97:
        return "midi-length"
    return "floor-length"


def pants_length_word(
    value: Any,
    rise: Any,
    body: dict[str, Any],
) -> str:
    relative_length = body_bottom_length_ratio(value, rise, body)
    if relative_length < 0.48:
        return "shorts"
    if relative_length < 0.58:
        return "knee-length shorts"
    if relative_length < 0.84:
        return "cropped"
    if relative_length < 0.87:
        return "ankle-length"
    return "full-length"


def pants_rise_word(value: Any) -> str:
    rise = to_float(value, 0.75)
    if rise >= 0.85:
        return "high-waisted"
    return ""


def pants_opening_ratio(
    width: Any,
    flare: Any,
    body: dict[str, Any],
) -> float:
    leg_circ = body_float(body, "leg_circ")
    hips = body_float(body, "hips")
    width_value = to_float(width, 1.0)
    flare_value = to_float(flare, 1.0)
    # Reconstruct one leg opening from GarmentCode's front/back PantPanel hems.
    base_extension = leg_circ - hips / 2.0 + 5.0
    opening = (
        hips / 2.0 + base_extension * width_value - 4.0 + leg_circ * (flare_value - 1.0)
    )
    return opening / leg_circ


def pants_cuff_word(
    cuff_type: Any,
    top_ruffle: Any,
    cuff_length: Any,
    skirt_fraction: Any,
    skirt_flare: Any,
    skirt_ruffle: Any,
    pants_length: Any,
    body: dict[str, Any],
    *,
    shorts: bool = False,
) -> str:
    cuff = str(cuff_type or "")
    top_ruffle_value = to_float(top_ruffle, 1.0)
    fraction = to_float(skirt_fraction, 0.5)
    if not cuff or cuff.lower() in {"none", "null"}:
        return ""

    effective_fraction = min(
        max(0.0, to_float(cuff_length, 0.1)),
        max(0.0, to_float(pants_length, 0.65) - 0.20),
    )
    cuff_cm = effective_fraction * body_float(body, "_leg_length")
    if cuff_cm < 2.0:
        return ""

    if shorts:
        if cuff == "CuffBandSkirt":
            flounce_cm = cuff_cm * fraction
            if flounce_cm < 2.0:
                return "banded hems"
            if to_float(skirt_flare, 1.0) >= 1.30:
                return "banded ruffled flounce hems"
            if to_float(skirt_ruffle, 1.0) >= 1.25:
                return "banded ruffled hems"
            return "banded flounce hems"
        if cuff == "CuffSkirt":
            if to_float(skirt_flare, 1.0) >= 1.30:
                return "ruffled flounce hems"
            if to_float(skirt_ruffle, 1.0) >= 1.25 or top_ruffle_value >= 1.25:
                return "ruffled hems"
            return "flared hems"
        if cuff == "CuffBand":
            if top_ruffle_value >= 1.25:
                return "gathered banded hems"
            return "banded hems"
        return ""

    if cuff == "CuffBandSkirt":
        flounce_cm = cuff_cm * fraction
        if flounce_cm >= 2.0 and to_float(skirt_flare, 1.0) >= 1.30:
            return "banded flounced cuffs"
        if flounce_cm >= 2.0 and to_float(skirt_ruffle, 1.0) >= 1.25:
            return "banded ruffled cuffs"
        if top_ruffle_value >= 1.25:
            return "gathered band cuffs"
        return "banded cuffs"
    if cuff == "CuffSkirt":
        if to_float(skirt_flare, 1.0) >= 1.30:
            return "flounced cuffs"
        if to_float(skirt_ruffle, 1.0) >= 1.25:
            return "ruffled cuffs"
        return "flared cuffs"
    if top_ruffle_value >= 1.25:
        return "gathered cuffs"
    return "banded cuffs"


def bottom_type(bottom: Any) -> str:
    name = clean_name(bottom)
    if not name:
        return ""
    if "pants" in name or "shorts" in name:
        return "pants"
    if "skirt" in name:
        return "skirt"
    return name


def skirt_shape_word(flare: Any, ruffle: Any) -> str:
    flare_value = to_float(flare, 1.0)
    ruffle_value = to_float(ruffle, 1.0)

    if flare_value <= 8:
        shape = "fitted straight"
    elif flare_value <= 14:
        shape = "slightly flared A-line"
    else:
        shape = "flared A-line"

    if ruffle_value > 1.25:
        return f"gathered {shape}"
    return shape


def join_details(details: list[str]) -> str:
    if len(details) < 2:
        return "".join(details)
    if len(details) == 2:
        return " and ".join(details)
    return ", ".join(details[:-1]) + f", and {details[-1]}"


def make_upper_part(
    design: dict[str, Any],
    upper: Any,
    body: dict[str, Any],
) -> dict[str, Any]:
    is_fitted_shirt = str(upper) == "FittedShirt"
    strapless = is_fitted_shirt and as_bool(
        get_v(design, ["shirt", "strapless"], False)
    )
    sleeveless = as_bool(get_v(design, ["sleeve", "sleeveless"], False))

    shirt_length = get_v(design, ["shirt", "length"])
    shirt_width = get_v(design, ["shirt", "width"])
    shirt_flare = get_v(design, ["shirt", "flare"])
    open_front = as_bool(get_v(design, ["shirt", "openfront"], False))

    sleeve_length = get_v(design, ["sleeve", "length"])
    sleeve_end_width = get_v(design, ["sleeve", "end_width"])
    sleeve_connecting_width = get_v(design, ["sleeve", "connecting_width"])
    sleeve_connect_ruffle = get_v(design, ["sleeve", "connect_ruffle"])
    cuff_type = get_v(design, ["sleeve", "cuff", "type"])
    cuff_top_ruffle = get_v(design, ["sleeve", "cuff", "top_ruffle"])
    cuff_length = get_v(design, ["sleeve", "cuff", "cuff_len"])
    cuff_skirt_fraction = get_v(
        design,
        ["sleeve", "cuff", "skirt_fraction"],
    )
    cuff_skirt_flare = get_v(design, ["sleeve", "cuff", "skirt_flare"])
    cuff_skirt_ruffle = get_v(design, ["sleeve", "cuff", "skirt_ruffle"])

    front_collar = get_v(design, ["collar", "f_collar"])
    collar_component = get_v(design, ["collar", "component", "style"])
    collar_depth = get_v(design, ["collar", "fc_depth"])
    collar_component_depth = get_v(design, ["collar", "component", "depth"])
    collar_component_standing = get_v(
        design,
        ["collar", "component", "lapel_standing"],
    )
    collar_width = get_v(design, ["collar", "width"])
    collar_front_bezier_x = get_v(design, ["collar", "f_bezier_x"])
    collar_front_bezier_y = get_v(design, ["collar", "f_bezier_y"])
    collar_front_flip_curve = get_v(design, ["collar", "f_flip_curve"])
    asymmetric = as_bool(get_v(design, ["left", "enable_asym"], False))

    left_sleeveless = as_bool(get_v(design, ["left", "sleeve", "sleeveless"], False))
    left_sleeve_length = get_v(design, ["left", "sleeve", "length"])
    left_sleeve_end_width = get_v(design, ["left", "sleeve", "end_width"])
    left_sleeve_connecting_width = get_v(
        design,
        ["left", "sleeve", "connecting_width"],
    )
    left_sleeve_connect_ruffle = get_v(
        design,
        ["left", "sleeve", "connect_ruffle"],
    )
    left_cuff_type = get_v(design, ["left", "sleeve", "cuff", "type"])
    left_cuff_top_ruffle = get_v(
        design,
        ["left", "sleeve", "cuff", "top_ruffle"],
    )
    left_cuff_length = get_v(
        design,
        ["left", "sleeve", "cuff", "cuff_len"],
    )
    left_cuff_skirt_fraction = get_v(
        design,
        ["left", "sleeve", "cuff", "skirt_fraction"],
    )
    left_cuff_skirt_flare = get_v(
        design,
        ["left", "sleeve", "cuff", "skirt_flare"],
    )
    left_cuff_skirt_ruffle = get_v(
        design,
        ["left", "sleeve", "cuff", "skirt_ruffle"],
    )

    one_sleeve = asymmetric and sleeveless != left_sleeveless
    both_sleeveless = asymmetric and sleeveless and left_sleeveless
    both_have_sleeves = asymmetric and not sleeveless and not left_sleeveless

    cape_sleeves = (
        not is_fitted_shirt
        and not asymmetric
        and sleeveless
        and to_float(shirt_width, 1.0) >= 1.20
        and to_float(shirt_flare, 1.0) >= 1.45
    )

    sleeve_prefix, sleeve_length_detail = sleeve_style(
        sleeveless,
        strapless,
        sleeve_length,
        sleeve_connecting_width,
    )
    if one_sleeve:
        sleeve_prefix = "one-sleeve"
    elif both_sleeveless:
        sleeve_prefix = "sleeveless"
    elif both_have_sleeves:
        sleeve_prefix = ""
    if cape_sleeves:
        sleeve_prefix = "cold-shoulder"

    words = []
    if asymmetric:
        words.append("asymmetric")

    words.append(sleeve_prefix)

    if is_fitted_shirt:
        words.append("fitted")
        words.append("cropped")
    else:
        words.append(
            shirt_fit_word(
                shirt_width,
                shirt_flare,
                shirt_length,
                body,
            )
        )
        words.append(shirt_length_word(shirt_length, body))

    words.append(clean_name(upper))
    prompt = " ".join(word for word in words if word)

    details = []
    if not is_fitted_shirt:
        body_shape = silhouette_word(shirt_flare, "silhouette")
        if body_shape:
            details.append(f"a {body_shape}")

    if (
        strapless
        and str(front_collar) == "Bezier2NeckHalf"
        and to_float(collar_depth, 0.4) <= 0.35
    ):
        details.append("a pointed center-front upper edge")
    elif asymmetric:
        details.append("an asymmetric neckline")
    elif not strapless:
        neckline = neckline_word(
            front_collar,
            collar_component,
            collar_depth,
            collar_component_depth,
            collar_component_standing,
            collar_width,
            collar_front_bezier_x,
            collar_front_bezier_y,
            collar_front_flip_curve,
        )
        if neckline:
            details.append(f"a {neckline}")

    if cape_sleeves:
        details.append("draped elbow-length bell sleeves")
    elif one_sleeve:
        if sleeveless:
            active_sleeve = describe_sleeve(
                left_sleeve_length,
                left_sleeve_end_width,
                left_sleeve_connecting_width,
                left_sleeve_connect_ruffle,
                left_cuff_type,
                left_cuff_top_ruffle,
                left_cuff_length,
                left_cuff_skirt_fraction,
                left_cuff_skirt_flare,
                left_cuff_skirt_ruffle,
                body,
                plural=False,
            )
        else:
            active_sleeve = describe_sleeve(
                sleeve_length,
                sleeve_end_width,
                sleeve_connecting_width,
                sleeve_connect_ruffle,
                cuff_type,
                cuff_top_ruffle,
                cuff_length,
                cuff_skirt_fraction,
                cuff_skirt_flare,
                cuff_skirt_ruffle,
                body,
                plural=False,
            )
        details.append(f"one {active_sleeve}")
    elif both_have_sleeves:
        right_sleeve = describe_sleeve(
            sleeve_length,
            sleeve_end_width,
            sleeve_connecting_width,
            sleeve_connect_ruffle,
            cuff_type,
            cuff_top_ruffle,
            cuff_length,
            cuff_skirt_fraction,
            cuff_skirt_flare,
            cuff_skirt_ruffle,
            body,
            plural=False,
        )
        left_sleeve = describe_sleeve(
            left_sleeve_length,
            left_sleeve_end_width,
            left_sleeve_connecting_width,
            left_sleeve_connect_ruffle,
            left_cuff_type,
            left_cuff_top_ruffle,
            left_cuff_length,
            left_cuff_skirt_fraction,
            left_cuff_skirt_flare,
            left_cuff_skirt_ruffle,
            body,
            plural=False,
        )
        if right_sleeve == left_sleeve:
            details.append(
                describe_sleeve(
                    sleeve_length,
                    sleeve_end_width,
                    sleeve_connecting_width,
                    sleeve_connect_ruffle,
                    cuff_type,
                    cuff_top_ruffle,
                    cuff_length,
                    cuff_skirt_fraction,
                    cuff_skirt_flare,
                    cuff_skirt_ruffle,
                    body,
                    plural=True,
                )
            )
        else:
            details.append(f"one {right_sleeve} and one {left_sleeve}")
    elif sleeve_length_detail:
        details.append(
            describe_sleeve(
                sleeve_length,
                sleeve_end_width,
                sleeve_connecting_width,
                sleeve_connect_ruffle,
                cuff_type,
                cuff_top_ruffle,
                cuff_length,
                cuff_skirt_fraction,
                cuff_skirt_flare,
                cuff_skirt_ruffle,
                body,
                plural=True,
            )
        )

    if open_front:
        details.append("an open front")

    if details:
        prompt += " with " + join_details(details)

    return {
        "part_type": "top",
        "part_prompt": prompt,
    }


def make_bottom_part(
    design: dict[str, Any],
    bottom: Any,
    b_type: str,
    body: dict[str, Any],
) -> dict[str, Any]:
    bottom_name = str(bottom)

    if bottom_name == "Skirt2":
        length = straight_skirt_length_word(
            get_v(design, ["skirt", "length"]),
            get_v(design, ["skirt", "rise"]),
            body,
        )
        shape = skirt_shape_word(
            get_v(design, ["skirt", "flare"]),
            get_v(design, ["skirt", "ruffle"]),
        )
        prompt = f"{length} {shape} skirt"

    elif bottom_name == "SkirtCircle":
        length = skirt_length_word(
            get_v(design, ["flare-skirt", "length"]),
            get_v(design, ["flare-skirt", "rise"]),
        )
        suns = to_float(
            get_v(design, ["flare-skirt", "suns"]),
            0.5,
        )
        if suns >= 0.9:
            prompt = f"{length} full-circle skirt"
        else:
            prompt = f"{length} circle skirt"

    elif bottom_name == "AsymmSkirtCircle":
        length_value = get_v(design, ["flare-skirt", "length"])
        rise_value = get_v(design, ["flare-skirt", "rise"])
        front_ratio = get_v(
            design,
            ["flare-skirt", "asymm", "front_length"],
        )
        front_length, back_length = asymmetric_skirt_length_words(
            length_value,
            rise_value,
            front_ratio,
            body,
        )
        if front_length == back_length:
            prompt = f"{back_length} high-low asymmetric circle skirt"
        else:
            prompt = (
                "high-low asymmetric circle skirt with "
                f"{with_indefinite_article(front_length)} front hem and "
                f"{with_indefinite_article(back_length)} back hem"
            )

    elif bottom_name == "SkirtManyPanels":
        suns = to_float(
            get_v(design, ["flare-skirt", "suns"]),
            0.5,
        )
        length = paneled_skirt_length_word(
            get_v(design, ["flare-skirt", "length"]),
            get_v(design, ["flare-skirt", "rise"]),
            suns,
        )
        panel_curve = to_float(
            get_v(
                design,
                ["flare-skirt", "skirt-many-panels", "panel_curve"],
            )
        )

        if suns >= 0.9:
            shape = "full, flared"
        elif suns >= 0.4:
            shape = "flared"
        else:
            shape = "slightly flared"

        prompt = f"{shape} {length} paneled skirt"
        if panel_curve <= -0.40:
            prompt += " with a pointed panel hem"
        elif panel_curve >= 0.40:
            prompt += " with a scalloped decorative hem"

    elif bottom_name == "GodetSkirt":
        base = get_v(design, ["godet-skirt", "base"])
        insert_width = to_float(
            get_v(
                design,
                ["godet-skirt", "insert_w"],
            )
        )
        num_inserts = to_float(get_v(design, ["godet-skirt", "num_inserts"]), 1.0)

        if str(base) == "PencilSkirt":
            length_value = get_v(design, ["pencil-skirt", "length"])
            rise_value = get_v(design, ["pencil-skirt", "rise"])
            base_flare = to_float(
                get_v(design, ["pencil-skirt", "flare"]),
                1.0,
            )
        else:
            length_value = get_v(design, ["skirt", "length"])
            rise_value = get_v(design, ["skirt", "rise"])
            base_flare = to_float(get_v(design, ["skirt", "flare"]))

        if str(base) == "PencilSkirt":
            length = pencil_skirt_length_word(length_value, rise_value, body)
        else:
            length = straight_skirt_length_word(length_value, rise_value, body)
        subtle_inserts = insert_width < 20 and insert_width * num_inserts < 80

        if str(base) == "PencilSkirt" and subtle_inserts:
            prompt = f"{length} fitted godet pencil skirt with subtle godet inserts"
        elif str(base) == "Skirt2" and subtle_inserts and base_flare <= 8:
            prompt = f"{length} fitted straight godet skirt with subtle godet inserts"
        elif str(base) == "PencilSkirt":
            prompt = f"{length} fitted godet pencil skirt with a flared godet hem"
        elif str(base) == "Skirt2" and base_flare <= 8:
            prompt = f"{length} fitted straight godet skirt with a flared godet hem"
        else:
            prompt = f"{length} flared godet skirt"

    elif bottom_name == "PencilSkirt":
        length_value = get_v(design, ["pencil-skirt", "length"])
        rise_value = get_v(design, ["pencil-skirt", "rise"])
        length = pencil_skirt_length_word(length_value, rise_value, body)
        flare = to_float(
            get_v(design, ["pencil-skirt", "flare"]),
            1.0,
        )
        front_slit = to_float(
            get_v(design, ["pencil-skirt", "front_slit"]),
        )
        back_slit = to_float(
            get_v(design, ["pencil-skirt", "back_slit"]),
        )
        divided = (
            to_float(length_value, 0.65) >= 0.65
            and flare <= 0.90
            and front_slit >= 0.70
            and back_slit >= 0.45
        )
        if divided:
            shape = "straight divided skirt"
        elif flare > 1.02:
            shape = "slightly flared A-line skirt"
        elif flare < 0.88:
            shape = "fitted tapered pencil skirt"
        else:
            shape = "fitted pencil skirt"
        prompt = f"{length} {shape}"

        details = []
        low_angle = to_float(
            get_v(design, ["pencil-skirt", "low_angle"]),
        )
        if abs(low_angle) >= 8.0:
            details.append("an asymmetric angled hem")

        if divided:
            details.append("deep front and back slits")
            slit_fields = []
        else:
            slit_fields = [
                ("front_slit", "front slit", 0.08),
                ("back_slit", "back slit", 0.20),
                ("left_slit", "left-side slit", 0.20),
                ("right_slit", "right-side slit", 0.20),
            ]

        visible_slits = []
        for key, label, threshold in slit_fields:
            value = get_v(design, ["pencil-skirt", key])
            if to_float(value) >= threshold:
                visible_slits.append(label)

        if len(visible_slits) >= 3:
            details.append("multiple slits")
        else:
            details.extend(f"a {label}" for label in visible_slits)

        if details:
            prompt += " with " + " and ".join(details)

    elif bottom_name == "SkirtLevels":
        length = skirt_length_word(
            get_v(design, ["levels-skirt", "length"]),
            get_v(design, ["levels-skirt", "rise"]),
        )
        base = get_v(design, ["levels-skirt", "base"])
        level_type = get_v(design, ["levels-skirt", "level"])
        level_ruffle = to_float(
            get_v(design, ["levels-skirt", "level_ruffle"]),
            1.0,
        )
        num_levels = int(
            to_float(
                get_v(design, ["levels-skirt", "num_levels"]),
                2,
            )
        )
        has_high_low = (
            str(base) == "AsymmSkirtCircle" or str(level_type) == "AsymmSkirtCircle"
        )
        length_phrase = f"{length} high-low" if has_high_low else length

        if num_levels <= 2 and level_ruffle <= 1.05:
            if str(base) == "PencilSkirt":
                shape = "fitted straight"
            elif str(base) in {"SkirtCircle", "AsymmSkirtCircle"}:
                shape = "flared"
            else:
                shape = skirt_shape_word(
                    get_v(design, ["skirt", "flare"]),
                    get_v(design, ["skirt", "ruffle"]),
                )
            prompt = f"{length_phrase} {shape} skirt"
        elif str(level_type) == "Skirt2" and level_ruffle <= 1.15:
            if str(base) in {"SkirtCircle", "AsymmSkirtCircle"}:
                shape = "gathered flared"
            elif str(base) == "PencilSkirt":
                shape = "fitted straight"
            else:
                shape = skirt_shape_word(
                    get_v(design, ["skirt", "flare"]),
                    get_v(design, ["skirt", "ruffle"]),
                )
            prompt = f"{length_phrase} {shape} skirt"
        else:
            if str(base) == "PencilSkirt":
                upper_section = "fitted pencil-skirt upper section"
            elif str(base) == "SkirtCircle":
                upper_section = "flared circular upper section"
            elif str(base) == "AsymmSkirtCircle":
                upper_section = "flared upper section"
            else:
                base_flare = to_float(get_v(design, ["skirt", "flare"]))
                if base_flare > 8:
                    upper_section = "smooth flared upper section"
                else:
                    upper_section = "smooth fitted upper section"

            tier_words = []
            if level_ruffle > 1.1:
                tier_words.append("gathered")

            if str(level_type) == "SkirtCircle":
                tier_words.append("circular")
            lower_section = " ".join(tier_words + ["lower section"])

            prompt = (
                f"{length_phrase} tiered skirt with a {upper_section} "
                f"and a {lower_section}"
            )

    elif bottom_name == "Pants" or b_type == "pants":
        length_value = to_float(get_v(design, ["pants", "length"]), 0.65)
        rise_value = get_v(design, ["pants", "rise"])
        length = pants_length_word(length_value, rise_value, body)
        width = to_float(get_v(design, ["pants", "width"]), 1.0)
        flare = to_float(get_v(design, ["pants", "flare"]), 1.0)
        opening_ratio = pants_opening_ratio(width, flare, body)
        is_shorts = length.endswith("shorts")
        rise = pants_rise_word(rise_value)
        cuff_type = get_v(design, ["pants", "cuff", "type"])
        cuff_ruffle = to_float(
            get_v(design, ["pants", "cuff", "top_ruffle"]),
            1.0,
        )
        cuff_length = get_v(design, ["pants", "cuff", "cuff_len"])
        cuff_skirt_fraction = get_v(
            design,
            ["pants", "cuff", "skirt_fraction"],
        )
        cuff_skirt_flare = get_v(design, ["pants", "cuff", "skirt_flare"])
        cuff_skirt_ruffle = get_v(design, ["pants", "cuff", "skirt_ruffle"])
        cuff_description = pants_cuff_word(
            cuff_type,
            cuff_ruffle,
            cuff_length,
            cuff_skirt_fraction,
            cuff_skirt_flare,
            cuff_skirt_ruffle,
            length_value,
            body,
            shorts=is_shorts,
        )
        gathered_cuffs = (
            not is_shorts and str(cuff_type) == "CuffBand" and cuff_ruffle >= 1.25
        )

        prefix = f"{rise} " if rise else ""

        if is_shorts:
            length_prefix = "knee-length " if length.startswith("knee") else ""
            decorated_hem = (
                "flounce" in cuff_description or "ruffled" in cuff_description
            )
            if opening_ratio < 0.72 and not length_prefix:
                prompt = f"{prefix}fitted biker shorts"
            elif opening_ratio < 0.72:
                prompt = f"{prefix}{length_prefix}fitted straight-leg shorts"
            elif decorated_hem and opening_ratio > 1.15:
                prompt = f"{prefix}{length_prefix}loose balloon shorts"
            elif decorated_hem and opening_ratio < 0.88:
                prompt = f"{prefix}{length_prefix}fitted straight-leg shorts"
            elif length_prefix and opening_ratio >= 0.98:
                prompt = f"{prefix}{length_prefix}loose straight-leg shorts"
            elif opening_ratio > 1.15:
                prompt = f"{prefix}{length_prefix}wide-leg shorts"
            else:
                prompt = f"{prefix}{length_prefix}straight-leg shorts"
        elif gathered_cuffs:
            prompt = f"{prefix}{length} loose tapered pants"
        elif length == "cropped":
            if flare >= 0.97:
                prompt = f"{prefix}cropped wide-leg culottes"
            elif flare < 0.80:
                prompt = f"{prefix}cropped tapered-leg pants"
            else:
                prompt = f"{prefix}cropped straight-leg pants"
        else:
            if flare > 1.08:
                leg_shape = "flared-leg"
            elif flare < 0.80:
                leg_shape = "tapered-leg"
            elif opening_ratio >= 0.98:
                leg_shape = "wide-leg"
            elif opening_ratio < 0.75:
                leg_shape = "slim straight-leg"
            else:
                leg_shape = "straight-leg"

            prompt = f"{prefix}{length} {leg_shape} pants"

        if cuff_description:
            prompt += f" with {cuff_description}"

        return {
            "part_type": "shorts" if is_shorts else "pants",
            "part_prompt": prompt,
        }

    else:
        prompt = clean_name(bottom)
        return {
            "part_type": b_type,
            "part_prompt": prompt,
        }

    return {
        "part_type": "skirt",
        "part_prompt": prompt,
    }


def capitalize_first(text: str) -> str:
    return text[:1].upper() + text[1:]


def compose_two_piece_prompt(
    upper_phrase: str,
    lower_phrase: str,
    relation: str,
) -> str:
    return capitalize_first(f"{upper_phrase}, {relation} {lower_phrase}")


def compose_single_piece_prompt(piece_phrase: str) -> str:
    return capitalize_first(piece_phrase)


def part_with_article(part: dict[str, Any]) -> str:
    prompt = part["part_prompt"]
    if part["part_type"] in {"pants", "shorts"}:
        return f"a pair of {prompt}"
    return with_indefinite_article(prompt)


def load_body_measurements(design_path: Path) -> dict[str, Any]:
    case_id = design_path.parent.name
    body_path = design_path.parent / f"{case_id}_body_measurements.yaml"
    if not body_path.is_file():
        raise FileNotFoundError(f"Body measurements not found: {body_path}")

    data = yaml.safe_load(body_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("body"), dict):
        raise ValueError(f"Invalid body measurements file: {body_path}")
    return data["body"]


def make_body_prompt(body: dict[str, Any]) -> str:
    return (
        f"A person {body_float(body, 'height'):.1f} cm tall with a "
        f"{body_float(body, 'bust'):.1f} cm bust, "
        f"{body_float(body, 'waist'):.1f} cm waist, "
        f"{body_float(body, 'hips'):.1f} cm hips, "
        f"{body_float(body, '_leg_length'):.1f} cm leg length, "
        f"and {body_float(body, 'arm_length'):.1f} cm arm length."
    )


def upper_bottom_relation(
    design: dict[str, Any],
    upper: Any,
    body: dict[str, Any],
) -> str:
    if str(upper) != "Shirt":
        return "worn with"

    length_ratio = to_float(get_v(design, ["shirt", "length"]), 1.0)
    waist_line = body_float(body, "waist_line")
    hips_line = body_float(body, "hips_line")
    below_waist = max(0.0, length_ratio * waist_line - waist_line)
    if below_waist >= 0.75 * hips_line:
        return "layered over"
    return "worn with"


def parse_design_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    design = data.get("design", data)
    body = load_body_measurements(path)

    upper = get_v(design, ["meta", "upper"])
    bottom = get_v(design, ["meta", "bottom"])
    waistband = get_v(design, ["meta", "wb"])
    is_separate_set = (
        not is_nullish(upper)
        and not is_nullish(bottom)
        and (str(bottom) == "Pants" or is_nullish(waistband))
    )

    parts: dict[str, dict[str, Any]] = {}

    if not is_nullish(upper):
        parts["upper"] = make_upper_part(design, upper, body)

    if not is_nullish(bottom):
        parts["bottom"] = make_bottom_part(
            design,
            bottom,
            bottom_type(bottom),
            body,
        )

    if "upper" in parts and "bottom" in parts:
        sample_prompt = compose_two_piece_prompt(
            part_with_article(parts["upper"]),
            part_with_article(parts["bottom"]),
            upper_bottom_relation(design, upper, body),
        )
    elif "upper" in parts:
        sample_prompt = compose_single_piece_prompt(
            part_with_article(parts["upper"]),
        )
    elif "bottom" in parts:
        sample_prompt = compose_single_piece_prompt(
            part_with_article(parts["bottom"]),
        )
    else:
        sample_prompt = "a garment"

    if not is_nullish(waistband):
        sample_prompt += f", finished with a {clean_name(waistband)}"

    sample_prompt = sample_prompt.rstrip(".") + "."
    for part in parts.values():
        part["part_prompt"] = part["part_prompt"].rstrip(".") + "."

    whole_part_type = "garment"
    part_values = list(parts.values())

    if len(part_values) == 1:
        whole_part_type = part_values[0]["part_type"]
    elif len(part_values) > 1:
        whole_part_type = "dress"

    parts["whole"] = {
        "part_type": whole_part_type,
        "part_prompt": sample_prompt,
    }

    output_parts: dict[str, dict[str, Any]] = {}

    if is_separate_set and "upper" in parts and "bottom" in parts:
        output_parts["upper"] = parts["upper"]
        output_parts["bottom"] = parts["bottom"]
    else:
        output_parts["whole"] = parts["whole"]

    return {
        "sample_prompt": sample_prompt,
        "body_prompt": make_body_prompt(body),
        "parts": output_parts,
    }


def find_yaml_files(root: Path, yaml_name: str) -> list[Path]:
    if root.stem != "random_body":
        root = root/"random_body"
    return sorted(
        path
        for path in root.rglob(f"*_{yaml_name}")
        if path.is_file() and "design_params" in path.name
    )


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    if not root.is_dir():
        raise NotADirectoryError(f"Prompt root is not a directory: {root}")

    logging.basicConfig(
        filename=args.log, filemode="w", level=logging.INFO, format="%(message)s"
    )
    logger = logging.getLogger(__name__)

    yaml_files = find_yaml_files(root, "design_params.yaml")
    if not yaml_files:
        raise FileNotFoundError(f"No design parameter YAML files found under: {root}")

    counts: dict[str, int] = {}

    for yaml_path in yaml_files:
        gcd_id = yaml_path.parent.name
        sample_dir = yaml_path.parent
        prompt_json = parse_design_yaml(yaml_path)
        out_path = sample_dir / f"{gcd_id}_prompts.json"
        out_path.write_text(
            json.dumps(prompt_json, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        split_status = (
            "split_upper_bottom" if "upper" in prompt_json["parts"] else "whole"
        )
        counts[split_status] = counts.get(split_status, 0) + 1
        logger.info("%s: %s -> %s", gcd_id, split_status, out_path)

    logger.info("parsed YAML files: %d", len(yaml_files))
    logger.info("wrote prompt JSON files: %d", len(yaml_files))
    logger.info("split_status_counts: %s", counts)


if __name__ == "__main__":
    main()
