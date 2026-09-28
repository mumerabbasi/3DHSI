"""Blender worker. Called by render_interaction.py, not directly."""

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from camera_and_crop import BASE_HEIGHT, BASE_WIDTH, METHODS, intrinsics, project


def import_ply(path):
    before = set(bpy.context.scene.objects)
    bpy.ops.wm.ply_import(filepath=str(path))
    objects = list(set(bpy.context.scene.objects) - before)
    if len(objects) != 1 or objects[0].type != "MESH":
        raise RuntimeError(f"Expected one PLY mesh: {path}")
    return objects[0]


def setup(config):
    bpy.ops.wm.open_mainfile(filepath=config["source_blend"])
    scene = bpy.context.scene
    reference = bpy.data.objects.get("optimized_human")
    if reference is None or reference.type != "MESH":
        raise RuntimeError("Module 06 blend has no optimized_human mesh")
    materials = list(reference.data.materials)
    if not materials:
        raise RuntimeError("Module 06 human material is missing")
    reference.hide_render = True
    humans = {}
    for method in METHODS:
        obj = import_ply(config["human_meshes"][method])
        obj.name = f"paper_{method}"
        obj.matrix_world = reference.matrix_world.copy()
        obj.data.materials.clear()
        for material in materials:
            obj.data.materials.append(material)
        obj.hide_render = True
        humans[method] = obj

    # Preserve Module 06's scene, light transforms, materials, color management,
    # samples, denoising, bounce limits and world. All methods share this scene.
    data = bpy.data.cameras.new("paper_camera")
    camera = bpy.data.objects.new("paper_camera", data)
    scene.collection.objects.link(camera)
    camera.matrix_world = Matrix(config["camera"]["matrix_world"])
    data.type = "PERSP"
    data.sensor_fit = "HORIZONTAL"
    data.sensor_width = 36.0
    k = intrinsics(config["camera"])
    data.lens = k["fx"] * data.sensor_width / BASE_WIDTH
    data.shift_x = (BASE_WIDTH / 2 - k["cx"]) / BASE_WIDTH
    data.shift_y = (k["cy"] - BASE_HEIGHT / 2) / BASE_WIDTH
    # Match Module 06's clipping, not MeshLab's interactive clipping controls.
    data.clip_start = 0.01
    data.clip_end = 100.0
    scene.camera = camera
    scene.render.resolution_x = BASE_WIDTH
    scene.render.resolution_y = BASE_HEIGHT
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.0
    scene.render.use_border = False
    scene.render.use_crop_to_border = False
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.quality = 100
    scene.render.use_file_extension = True
    if config.get("samples") is not None:
        scene.cycles.samples = config["samples"]
    bpy.context.view_layer.update()
    return scene, camera, humans


def world_vertices(obj):
    xyz = np.empty(len(obj.data.vertices) * 3, dtype=np.float64)
    obj.data.vertices.foreach_get("co", xyz)
    xyz = xyz.reshape(-1, 3)
    m = np.asarray(obj.matrix_world, dtype=np.float64)
    return xyz @ m[:3, :3].T + m[:3, 3]


def inspect(config, scene, camera, humans):
    report = {"methods": {}, "warnings": []}
    inverse = np.asarray(camera.matrix_world.inverted(), dtype=np.float64)
    k = intrinsics(config["camera"])
    for method, obj in humans.items():
        xyz = world_vertices(obj)
        local = xyz @ inverse[:3, :3].T + inverse[:3, 3]
        depth = -local[:, 2]
        if np.any(depth <= camera.data.clip_start) or np.any(depth >= camera.data.clip_end):
            raise RuntimeError(f"{method}: human crosses the camera clipping planes; export a wider view")
        uv = np.column_stack((k["fx"] * local[:, 0] / depth + k["cx"],
                              k["cy"] - k["fy"] * local[:, 1] / depth))
        bounds = [*uv.min(axis=0).tolist(), *uv.max(axis=0).tolist()]
        error = 0.0
        # Cross-check the conversion against Blender's own camera projection.
        for point in xyz[::max(1, len(xyz) // 100)]:
            ndc = world_to_camera_view(scene, camera, Vector(point))
            actual = np.array([ndc.x * BASE_WIDTH, (1 - ndc.y) * BASE_HEIGHT])
            expected = np.array(project(config["camera"], point)[:2])
            error = max(error, float(np.max(np.abs(actual - expected))))
        if error > 0.05:
            raise RuntimeError(f"Camera projection mismatch: {error:.4f} pixels")
        report["methods"][method] = {
            "bounds_base_xyxy": bounds,
            "depth_range_m": [float(depth.min()), float(depth.max())],
            "projection_check_max_error_px": error,
            "vertex_count": len(xyz),
        }
    report["style"] = {
        "source_blend": config["source_blend"],
        "engine": scene.render.engine,
        "cycles_samples": scene.cycles.samples,
        "denoising": scene.cycles.use_denoising,
        "view_transform": scene.view_settings.view_transform,
        "look": scene.view_settings.look,
        "exposure": scene.view_settings.exposure,
        "gamma": scene.view_settings.gamma,
        "lights": [{"name": obj.name, "matrix_world": [list(row) for row in obj.matrix_world],
                    "energy": obj.data.energy, "type": obj.data.type}
                   for obj in scene.objects if obj.type == "LIGHT"],
    }
    Path(config["inspection_path"]).write_text(json.dumps(report, indent=2) + "\n")
    print("PAPER: camera and mesh projection checks passed", flush=True)


def enable_gpu():
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
            if any(device.type != "CPU" for device in prefs.devices):
                for device in prefs.devices:
                    device.use = device.type != "CPU"
                bpy.context.scene.cycles.device = "GPU"
                print(f"PAPER: rendering with {backend}", flush=True)
                return
        except Exception:
            continue
    bpy.context.scene.cycles.device = "CPU"
    print("PAPER: no compatible GPU found; using CPU", flush=True)


def render(config, scene, humans, multiplier):
    enable_gpu()
    scene.render.resolution_x = BASE_WIDTH * multiplier
    scene.render.resolution_y = BASE_HEIGHT * multiplier
    out = Path(config["output_dir"]) / "full" / f"{scene.render.resolution_x}x{scene.render.resolution_y}"
    out.mkdir(parents=True, exist_ok=True)
    for method in METHODS:
        for key, obj in humans.items():
            obj.hide_render = key != method
        scene.render.filepath = str(out / f"{method}.jpg")
        print(f"PAPER: rendering {method} at {scene.render.resolution_x} x {scene.render.resolution_y}", flush=True)
        bpy.ops.render.render(write_still=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--phase", choices=("inspect", "render"), required=True)
    parser.add_argument("--multiplier", type=int, default=1)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    config = json.loads(args.config.read_text())
    scene, camera, humans = setup(config)
    if args.phase == "inspect":
        inspect(config, scene, camera, humans)
    else:
        render(config, scene, humans, args.multiplier)


if __name__ == "__main__":
    main()
