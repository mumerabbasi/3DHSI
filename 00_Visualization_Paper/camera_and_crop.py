"""MeshLab camera conversion and shared crop geometry (no Blender dependency)."""

import math
import xml.etree.ElementTree as ET

BASE_WIDTH = 4290
BASE_HEIGHT = 2560
FINAL_SIZE = 1024
METHODS = ("genzi", "prox", "physic", "ours")


def read_camera(path):
    root = ET.parse(path).getroot()
    node = root if root.tag == "VCGCamera" else root.find(".//VCGCamera")
    if node is None:
        raise ValueError("Camera XML must contain VCGCamera")
    if node.get("BinaryData", "0") != "0" or node.get("CameraType", "0") != "0":
        raise ValueError("Export a non-binary perspective MeshLab camera (CameraType=0)")

    def values(key, length):
        result = [float(x) for x in node.attrib[key].split()]
        if len(result) != length or not all(math.isfinite(x) for x in result):
            raise ValueError(f"Invalid {key}: expected {length} finite numbers")
        return result

    translation = values("TranslationVector", 4)
    rotation = values("RotationMatrix", 16)
    viewport = values("ViewportPx", 2)
    center = values("CenterPx", 2)
    pixel = values("PixelSizeMm", 2)
    focal = values("FocalMm", 1)[0]
    if min(*viewport, *pixel, focal) <= 0:
        raise ValueError("Viewport, pixel size, and focal length must be positive")
    if not math.isclose(pixel[0], pixel[1], rel_tol=1e-5):
        raise ValueError("Non-square sensor pixels are not supported; export a standard MeshLab view")
    if any(abs(x) > 1e-10 for x in values("LensDistortion", 2)):
        raise ValueError("Lens distortion is not supported; export an undistorted MeshLab view")
    r = [rotation[i * 4:i * 4 + 3] for i in range(3)]
    for i in range(3):
        for j in range(3):
            dot = sum(r[i][k] * r[j][k] for k in range(3))
            if abs(dot - float(i == j)) > 1e-4:
                raise ValueError("Camera rotation is not orthonormal")
    # VCGLib wrap/qt/shot_qt.h writes TranslationVector = -Extrinsics.Tra().
    # Extrinsics.Rot() is world-to-camera, with OpenGL axes (+Y up, -Z forward).
    # Thus Blender's camera-to-world is [R^T | -TranslationVector]. TrackScale
    # is already absorbed into the exported shot; do NOT scale the pose again.
    pose = [[r[j][i] for j in range(3)] + [-translation[i]] for i in range(3)]
    pose.append([0.0, 0.0, 0.0, 1.0])
    settings = root.find(".//ViewSettings")
    return {
        "matrix_world": pose,
        "viewport": viewport,
        "center_px_bottom_left": center,
        "focal_px": focal / pixel[0],
        "view_settings": dict(settings.attrib) if settings is not None else {},
    }


def intrinsics(camera, width=BASE_WIDTH, height=BASE_HEIGHT):
    """Preserve vertical FOV; a different viewport aspect changes horizontal coverage."""
    vw, vh = camera["viewport"]
    scale = height / vh
    cx, cy_bottom = camera["center_px_bottom_left"]
    f = camera["focal_px"] * scale
    return {
        "fx": f, "fy": f,
        "cx": width / 2 + (cx - vw / 2) * scale,
        "cy": height / 2 - (cy_bottom - vh / 2) * scale,
    }


def project(camera, point, width=BASE_WIDTH, height=BASE_HEIGHT):
    """Independent reference projection, used to validate Blender's camera."""
    pose = camera["matrix_world"]
    delta = [point[i] - pose[i][3] for i in range(3)]
    local = [sum(pose[j][i] * delta[j] for j in range(3)) for i in range(3)]
    depth = -local[2]
    if depth <= 0:
        raise ValueError("Point is behind the camera")
    k = intrinsics(camera, width, height)
    return [k["fx"] * local[0] / depth + k["cx"],
            k["cy"] - k["fy"] * local[1] / depth, depth]


def union_bounds(bounds):
    return [min(b[0] for b in bounds), min(b[1] for b in bounds),
            max(b[2] for b in bounds), max(b[3] for b in bounds)]


def shared_crop(bounds, margin=0.12, context_box=None, override=None):
    """One integer square in BASE-resolution pixels, enclosing every method."""
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("Margin must be finite and non-negative")
    boxes = list(bounds)
    if context_box is not None:
        boxes.append(context_box)
    for box in boxes:
        if len(box) != 4 or not all(math.isfinite(v) for v in box):
            raise ValueError("Bounds must have four finite coordinates")
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("Bounds must have positive width and height")
    x0, y0, x1, y1 = union_bounds(boxes)
    if x0 < 0 or y0 < 0 or x1 > BASE_WIDTH or y1 > BASE_HEIGHT:
        raise ValueError("A human/context region is outside the saved camera frame. "
                         "Zoom out in MeshLab and export the camera again.")
    if override is not None:
        left, top, side = override
        if side <= 0:
            raise ValueError("Crop side must be positive")
    else:
        side = math.ceil(max(x1 - x0, y1 - y0) * (1 + 2 * margin))
        if side > min(BASE_WIDTH, BASE_HEIGHT):
            raise ValueError("The shared square crop plus margin does not fit. "
                             "Use a wider camera view or reduce --margin.")
        left = max(0, min(BASE_WIDTH - side, math.floor((x0 + x1 - side) / 2)))
        top = max(0, min(BASE_HEIGHT - side, math.floor((y0 + y1 - side) / 2)))
    if left < 0 or top < 0 or left + side > BASE_WIDTH or top + side > BASE_HEIGHT:
        raise ValueError("Crop must lie inside the base render")
    if left > x0 or top > y0 or left + side < x1 or top + side < y1:
        raise ValueError("Crop would clip a human or requested context region")
    return [int(left), int(top), int(left + side), int(top + side)]


def resolution_multiplier(crop):
    # Integer multiples maintain the exact aspect and integer crop coordinates.
    return max(1, math.ceil(FINAL_SIZE / (crop[2] - crop[0])))
