"""Geometry checks: run with python -m unittest discover -s <this directory>."""

import tempfile
import unittest
from pathlib import Path

from camera_and_crop import intrinsics, project, read_camera, resolution_multiplier, shared_crop
from render_interaction import resolve_artifact


class CameraTests(unittest.TestCase):
    def test_meshlab_translation_rotation_and_trackscale(self):
        # Camera center is (1,2,3), looking along world +X, with world +Y up.
        # A point one unit right/up and two units ahead must project to (150,30).
        xml = '''<project><VCGCamera TranslationVector="-1 -2 -3 1"
        RotationMatrix="0 0 1 0 0 1 0 0 -1 0 0 0 0 0 0 1"
        ViewportPx="200 160" CenterPx="100 80" FocalMm="10"
        PixelSizeMm="0.1 0.1" LensDistortion="0 0"/>
        <ViewSettings TrackScale="0.2"/></project>'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "camera.xml"
            path.write_text(xml)
            camera = read_camera(path)
        self.assertEqual([r[3] for r in camera["matrix_world"][:3]], [1, 2, 3])
        self.assertEqual(project(camera, [3, 3, 4], 200, 160), [150, 30, 2])
        self.assertEqual(project(camera, [3, 3, 4], 400, 320), [300, 60, 2])
        with self.assertRaisesRegex(ValueError, "behind"):
            project(camera, [0, 2, 3])

    def test_off_center_principal_point_y_flip(self):
        camera = {"viewport": [200, 160], "focal_px": 100,
                  "center_px_bottom_left": [90, 70]}
        self.assertEqual(intrinsics(camera, 400, 320),
                         {"fx": 200, "fy": 200, "cx": 180, "cy": 180})

    def test_provided_camera_position(self):
        camera = read_camera(Path(__file__).parent / "Render_Cameras/interaction_18.xml")
        self.assertEqual([r[3] for r in camera["matrix_world"][:3]],
                         [-0.594888, 0.558376, 3.27771])


class CropTests(unittest.TestCase):
    def test_union_includes_displaced_method_and_context(self):
        boxes = [[1000, 800, 1300, 1400], [1400, 900, 1800, 1500]]
        context = [900, 700, 1900, 1550]
        crop = shared_crop(boxes, margin=.1, context_box=context)
        self.assertEqual(crop[2] - crop[0], 1200)
        self.assertEqual(crop[3] - crop[1], 1200)
        for x0, y0, x1, y1 in boxes + [context]:
            self.assertLessEqual(crop[0], x0)
            self.assertLessEqual(crop[1], y0)
            self.assertGreaterEqual(crop[2], x1)
            self.assertGreaterEqual(crop[3], y1)

    def test_never_silently_clip_a_method(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            shared_crop([[-1, 0, 100, 100]])
        with self.assertRaisesRegex(ValueError, "clip"):
            shared_crop([[100, 100, 400, 400]], override=[150, 100, 300])
        with self.assertRaisesRegex(ValueError, "does not fit"):
            shared_crop([[1, 1, 4200, 2500]])

    def test_resolution_threshold_and_integer_scaling(self):
        for side, expected in [(1334, 1), (1024, 1), (1023, 2), (400, 3)]:
            multiplier = resolution_multiplier([100, 200, 100 + side, 200 + side])
            self.assertEqual(multiplier, expected)
            self.assertGreaterEqual(side * multiplier, 1024)

    def test_migrated_paths_do_not_confuse_genzi_and_prox(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for old, new in [("08_Run_PhySIC", "07_Run_PhySIC"),
                             ("09_Run_Prox", "08_Run_Prox"),
                             ("09_Run_Genzi", "09_Run_Genzi")]:
                target = root / new / "human.ply"
                target.parent.mkdir(exist_ok=True)
                target.touch()
                self.assertEqual(resolve_artifact(f"/old/work/4DHSI/{old}/human.ply", root), target)


if __name__ == "__main__":
    unittest.main()
