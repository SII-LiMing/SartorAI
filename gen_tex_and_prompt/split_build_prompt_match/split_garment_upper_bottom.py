from pathlib import Path
import argparse
import logging

import trimesh
from trimesh.exchange.obj import export_obj
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("--log", required=True)
    args = parser.parse_args()

    batch = Path(args.input_dir)
    obj_paths = sorted((batch / "random_body").glob("*/*_sim.obj"))

    logging.basicConfig(filename=args.log, filemode="w", level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    # for obj_path in tqdm(obj_paths):
    for obj_path in obj_paths:
        cur_dir = obj_path.parent
        case_id = cur_dir.name
        base = obj_path.stem[:-4]
        
        mesh = trimesh.load(obj_path, force="mesh", process=False)
        mesh.merge_vertices(merge_tex=True, merge_norm=True)

        parts = sorted(
            mesh.split(only_watertight=False),
            key=lambda m: m.bounds[:, 1].mean(),
            reverse=True,
        )

        logger.info("[%s]", case_id)
        logger.info("mesh: %s", obj_path)
        logger.info("parts: %d", len(parts))
        if len(parts) == 1:
            logger.info("result: single garment, skip\n")
        elif len(parts) == 2:
            for name, part in (("upper", parts[0]), ("bottom", parts[1])):
                path = cur_dir / f"{base}_{name}.obj"
                obj = export_obj(
                    part,
                    include_color=False,
                    include_texture=False,
                    write_texture=False,
                )
                path.write_text(obj, encoding="utf-8")
                center_y = part.bounds[:, 1].mean()
                logger.info(
                    "%s: %s, center_y=%.6f, vertices=%d, faces=%d",
                    name, path, center_y, len(part.vertices), len(part.faces),
                )
            logger.info("result: split\n")
        else:
            logger.info("result: unexpected part count\n")



if __name__ == "__main__":
    main()
