#!/usr/bin/env python3
"""Render all four methods from a MeshLab camera, then make matched paper crops."""

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from camera_and_crop import (
    BASE_HEIGHT, BASE_WIDTH, FINAL_SIZE, METHODS, read_camera,
    resolution_multiplier, shared_crop,
)

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODULE_DIR.parent
METHOD_MODULES = {
    "ours": "06_Evaluate_Interaction", "physic": "07_Run_PhySIC",
    "prox": "08_Run_Prox", "genzi": "09_Run_Genzi",
}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def resolve_artifact(raw, project_dir):
    """Resolve current paths and the repository's pre-rename 4DHSI metadata."""
    path = Path(raw)
    parts = path.parts
    for root_name in ("3DHSI", "4DHSI"):
        if root_name in parts:
            tail = list(parts[parts.index(root_name) + 1:])
            if root_name == "4DHSI" and tail:
                tail[0] = {"08_Run_PhySIC": "07_Run_PhySIC",
                           "09_Run_Prox": "08_Run_Prox"}.get(tail[0], tail[0])
            candidate = project_dir.joinpath(*tail)
            if candidate.is_file():
                return candidate.resolve()
    if path.is_file():
        return path.resolve()
    if not path.is_absolute() and (project_dir / path).is_file():
        return (project_dir / path).resolve()
    raise FileNotFoundError(f"Cannot resolve artifact: {raw}")


def resolve_inputs(project_dir, interaction):
    paths, configs = {}, {}
    for method, module in METHOD_MODULES.items():
        base = project_dir / module
        if method != "ours":
            base /= "evaluation"
        path = base / "output" / interaction / "semantics/assets/render_config.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {method} evaluation render config: {path}")
        configs[method] = path
        data = json.loads(path.read_text())
        # Use the evaluated GenZI candidate, never pick an arbitrary run.
        paths[method] = str(resolve_artifact(data["human_mesh_world"], project_dir))
    source_blend = configs["ours"].parent / "render_scene.blend"
    if not source_blend.is_file():
        data = json.loads(configs["ours"].read_text())
        source_blend = resolve_artifact(data["blend_path"], project_dir)
    return str(source_blend.resolve()), paths, {k: str(v) for k, v in configs.items()}


def pick_gpu(requested):
    if requested != "auto":
        return requested
    # Respect a caller's device restriction. Otherwise choose the most free GPU.
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return os.environ["CUDA_VISIBLE_DEVICES"]
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        devices = [line.split(",") for line in result.stdout.splitlines()]
        return max(devices, key=lambda row: int(row[1]))[0].strip()
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None


def run_blender(blender, config_path, output, phase, gpu, multiplier=1):
    log_path = output / "logs" / f"{phase}_{multiplier}x.log"
    command = [blender, "--background", "--python-exit-code", "1", "--python",
               str(MODULE_DIR / "blender_render.py"), "--", str(config_path),
               "--phase", phase, "--multiplier", str(multiplier)]
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    # Do not leak a conda Python environment into Blender's bundled Python.
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    print(f"{phase}: log at {log_path}", flush=True)
    with log_path.open("w") as log:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, env=env)
        try:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                if "PAPER:" in line or "Error:" in line or "Traceback" in line:
                    print(line.rstrip(), flush=True)
            code = proc.wait()
        except BaseException:
            proc.terminate()
            proc.wait()
            raise
    if code:
        raise RuntimeError(f"Blender failed ({code}); see {log_path}")


def export_crops(output, crop, multiplier):
    from PIL import Image

    full = output / "full" / f"{BASE_WIDTH * multiplier}x{BASE_HEIGHT * multiplier}"
    scaled = [x * multiplier for x in crop]
    for method in METHODS:
        with Image.open(full / f"{method}.jpg") as image:
            if image.size != (BASE_WIDTH * multiplier, BASE_HEIGHT * multiplier):
                raise RuntimeError(f"Unexpected render dimensions for {method}: {image.size}")
            image = image.convert("RGB").crop(scaled)
            if image.width < FINAL_SIZE:
                raise RuntimeError("Refusing to upscale a crop below 1024 pixels")
            if image.size != (FINAL_SIZE, FINAL_SIZE):
                image = image.resize((FINAL_SIZE, FINAL_SIZE), Image.Resampling.LANCZOS)
            image.save(output / f"{method}.jpg", quality=98, subsampling=0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interaction_name", "--interaction-name", required=True)
    parser.add_argument("--camera-dir", type=Path, default=MODULE_DIR / "Render_Cameras")
    parser.add_argument("--output-dir", type=Path, default=MODULE_DIR / "Output")
    parser.add_argument("--project-dir", type=Path, default=PROJECT_DIR)
    parser.add_argument("--blender-bin", default=shutil.which("blender") or "/my_workspace/blender-4.2.17-linux-x64/blender")
    parser.add_argument("--gpu-index", default="auto", help="GPU index, or auto (most free memory)")
    parser.add_argument("--margin", type=float, default=0.02,
                        help="Padding per side, relative to the longest combined human bound (default .12)")
    parser.add_argument("--context-box", nargs=4, type=float, metavar=("X0", "Y0", "X1", "Y1"),
                        help="Also include this scene region, in 4290x2560 image coordinates")
    parser.add_argument("--crop-box", nargs=3, type=int, metavar=("LEFT", "TOP", "SIDE"),
                        help="Explicit shared square crop, in 4290x2560 image coordinates")
    parser.add_argument("--samples", type=int, default=None, help="Override saved Module 06 sample count")
    parser.add_argument("--prepare-only", action="store_true", help="Validate camera, inputs and crop without rendering")
    return parser.parse_args()


def main():
    args = parse_args()
    if not re.fullmatch(r"interaction_\d+", args.interaction_name):
        raise ValueError("Expected an interaction name such as interaction_18")
    if args.samples is not None and args.samples < 1:
        raise ValueError("--samples must be positive")
    # Fail early if the crop/export dependency is missing.
    from PIL import Image  # noqa: F401

    project = args.project_dir.resolve()
    camera_path = (args.camera_dir / f"{args.interaction_name}.xml").resolve()
    camera = read_camera(camera_path)
    source, meshes, configs = resolve_inputs(project, args.interaction_name)
    output = (args.output_dir / args.interaction_name).resolve()
    (output / "logs").mkdir(parents=True, exist_ok=True)
    config = {
        "interaction_name": args.interaction_name,
        "camera_xml": str(camera_path), "camera": camera,
        "source_blend": source, "human_meshes": meshes, "source_configs": configs,
        "output_dir": str(output), "inspection_path": str(output / "inspection.json"),
        "samples": args.samples,
    }
    config_path = output / "render_config.json"
    write_json(config_path, config)
    shutil.copyfile(camera_path, output / "camera.xml")
    gpu = pick_gpu(args.gpu_index)
    print(f"{args.interaction_name}: all four methods; GPU {gpu or 'default'}", flush=True)
    run_blender(args.blender_bin, config_path, output, "inspect", gpu)
    inspection = json.loads(Path(config["inspection_path"]).read_text())
    bounds = [inspection["methods"][m]["bounds_base_xyxy"] for m in METHODS]
    crop = shared_crop(bounds, args.margin, args.context_box, args.crop_box)
    multiplier = resolution_multiplier(crop)
    side = crop[2] - crop[0]
    warnings = []
    for method, box in zip(METHODS, bounds):
        occupancy = max(box[2] - box[0], box[3] - box[1]) / side
        inspection["methods"][method]["crop_occupancy"] = occupancy
        if occupancy < 0.5:
            warnings.append(f"{method} occupies only {occupancy:.0%} of the shared crop; review framing")
    manifest = {
        "status": "prepared", "base_resolution": [BASE_WIDTH, BASE_HEIGHT],
        "crop_base_xyxy": crop, "crop_render_xyxy": [x * multiplier for x in crop],
        "render_multiplier": multiplier,
        "final_render_resolution": [BASE_WIDTH * multiplier, BASE_HEIGHT * multiplier],
        "native_crop_size": side * multiplier, "output_size": [FINAL_SIZE, FINAL_SIZE],
        "margin": args.margin, "context_box": args.context_box, "crop_override": args.crop_box,
        "methods": inspection["methods"], "style": inspection["style"], "warnings": warnings,
    }
    manifest_path = output / "manifest.json"
    write_json(manifest_path, manifest)
    print(f"Shared crop: {crop} ({side} x {side}); final render multiplier {multiplier}", flush=True)
    for warning in warnings:
        print(f"WARNING: {warning}", flush=True)
    if args.prepare_only:
        return
    # Always produce the requested base-resolution renders first.
    run_blender(args.blender_bin, config_path, output, "render", gpu, 1)
    if multiplier > 1:
        print("Base crop is below 1024 pixels; increasing render resolution.", flush=True)
        run_blender(args.blender_bin, config_path, output, "render", gpu, multiplier)
    export_crops(output, crop, multiplier)
    manifest["status"] = "complete"
    write_json(manifest_path, manifest)
    print(f"Saved four matched 1024 x 1024 JPEGs in {output}", flush=True)


if __name__ == "__main__":
    main()
