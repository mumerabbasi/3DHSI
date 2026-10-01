#!/usr/bin/env python3
"""Refresh paper GenZI images using the saved camera and shared comparison crop."""

import argparse
import json
import math
import shutil
from pathlib import Path

from PIL import Image

from camera_and_crop import BASE_HEIGHT, BASE_WIDTH, FINAL_SIZE, read_camera
from render_interaction import MODULE_DIR, PROJECT_DIR, pick_gpu, resolve_artifact, run_blender, write_json


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--interaction-name")
    group.add_argument("--all-cameras", action="store_true")
    parser.add_argument("--camera-dir", type=Path, default=MODULE_DIR / "Render_Cameras")
    parser.add_argument("--output-dir", type=Path, default=MODULE_DIR / "Output")
    parser.add_argument("--project-dir", type=Path, default=PROJECT_DIR)
    parser.add_argument(
        "--blender-bin",
        default=shutil.which("blender") or "/home/umer/software/blender-4.2.17-linux-x64/blender",
    )
    parser.add_argument("--gpu-index", default="auto")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--human-z-offset-m", type=float, default=0.0)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def render_one(name, args):
    camera_path = (args.camera_dir / f"{name}.xml").resolve()
    camera = read_camera(camera_path)
    output = (args.output_dir / name).resolve()
    paper_manifest_path = output / "manifest.json"
    paper_camera_path = output / "camera.xml"
    if not paper_manifest_path.is_file() or not paper_camera_path.is_file():
        raise FileNotFoundError(f"Existing paper crop and camera missing for {name}")
    if camera_path.read_bytes() != paper_camera_path.read_bytes():
        raise ValueError(f"Paper camera changed for {name}; existing comparison renders use another camera")
    paper_manifest = json.loads(paper_manifest_path.read_text(encoding="utf-8"))
    crop = [int(value) for value in paper_manifest["crop_base_xyxy"]]
    multiplier = int(paper_manifest["render_multiplier"])
    if len(crop) != 4 or multiplier < 1:
        raise ValueError(f"Invalid saved paper crop for {name}")

    evaluated_config_path = (
        args.project_dir / "09_Run_Genzi" / "evaluation" / "output" / name
        / "semantics" / "assets" / "render_config.json"
    )
    evaluated_config = json.loads(evaluated_config_path.read_text(encoding="utf-8"))
    source_blend = resolve_artifact(evaluated_config["blend_path"], args.project_dir)
    human_mesh = resolve_artifact(evaluated_config["human_mesh_world"], args.project_dir)
    (output / "logs").mkdir(parents=True, exist_ok=True)
    config_path = output / "genzi_render_config.json"
    inspection_path = output / "genzi_inspection.json"
    config = {
        "interaction_name": name,
        "camera_xml": str(camera_path),
        "camera": camera,
        "source_blend": str(source_blend),
        "human_meshes": {"genzi": str(human_mesh)},
        "methods": ["genzi"],
        "output_dir": str(output),
        "inspection_path": str(inspection_path),
        "samples": args.samples,
        "selected_candidate_dir": evaluated_config["selected_candidate_dir"],
        "evaluated_config": str(evaluated_config_path),
        "human_z_offset_m": args.human_z_offset_m,
    }
    write_json(config_path, config)
    gpu = pick_gpu(args.gpu_index)
    run_blender(args.blender_bin, config_path, output, "inspect", gpu, log_prefix="genzi_")
    inspection = json.loads(inspection_path.read_text(encoding="utf-8"))
    bounds = inspection["methods"]["genzi"]["bounds_base_xyxy"]
    cropped = (
        bounds[0] < crop[0] or bounds[1] < crop[1]
        or bounds[2] > crop[2] or bounds[3] > crop[3]
    )
    genzi_manifest_path = output / "genzi_manifest.json"
    genzi_manifest = {
        "status": "prepared",
        "interaction_name": name,
        "source_blend": str(source_blend),
        "human_mesh_world": str(human_mesh),
        "human_z_offset_m": args.human_z_offset_m,
        "selected_candidate_dir": config["selected_candidate_dir"],
        "camera_xml": str(camera_path),
        "crop_base_xyxy": crop,
        "render_multiplier": multiplier,
        "human_bounds_base_xyxy": bounds,
        "crop_clips_human": cropped,
        "style": inspection["style"],
    }
    write_json(genzi_manifest_path, genzi_manifest)
    if cropped:
        print(f"WARNING: {name} GenZI human extends beyond the saved shared crop", flush=True)
    if args.prepare_only:
        return genzi_manifest

    run_blender(args.blender_bin, config_path, output, "render", gpu, 1, log_prefix="genzi_")
    if multiplier > 1:
        run_blender(args.blender_bin, config_path, output, "render", gpu, multiplier, log_prefix="genzi_")
    full = output / "full" / f"{BASE_WIDTH * multiplier}x{BASE_HEIGHT * multiplier}" / "genzi.jpg"
    with Image.open(full) as image:
        if image.size != (BASE_WIDTH * multiplier, BASE_HEIGHT * multiplier):
            raise RuntimeError(f"Unexpected GenZI render dimensions for {name}: {image.size}")
        scaled_crop = [value * multiplier for value in crop]
        final = image.convert("RGB").crop(scaled_crop)
        if final.width < FINAL_SIZE:
            raise RuntimeError(f"GenZI crop would be upscaled for {name}")
        if final.size != (FINAL_SIZE, FINAL_SIZE):
            final = final.resize((FINAL_SIZE, FINAL_SIZE), Image.Resampling.LANCZOS)
        final.save(output / "genzi.jpg", quality=98, subsampling=0)
    genzi_manifest["status"] = "complete"
    write_json(genzi_manifest_path, genzi_manifest)
    print(f"{name}: updated GenZI render with saved paper crop", flush=True)
    return genzi_manifest


def main():
    args = parse_args()
    if args.samples is not None and args.samples < 1:
        raise ValueError("--samples must be positive")
    if not math.isfinite(args.human_z_offset_m):
        raise ValueError("--human-z-offset-m must be finite")
    if args.all_cameras and args.human_z_offset_m != 0.0:
        raise ValueError("Height offsets require a single --interaction-name")
    names = (
        [path.stem for path in sorted(args.camera_dir.glob("interaction_*.xml"))]
        if args.all_cameras else [args.interaction_name]
    )
    if not names:
        raise FileNotFoundError(f"No paper cameras under {args.camera_dir}")
    for name in names:
        render_one(name, args)


if __name__ == "__main__":
    main()
