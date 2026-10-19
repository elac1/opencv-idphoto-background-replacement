from __future__ import annotations

import unittest

import numpy as np

from idphoto.compose import CropBox, compute_crop_box, render_crop
from idphoto.config import ONE_INCH, TWO_INCH
from idphoto.detector import Face


class ClampTest(unittest.TestCase):
    def test_translates_box_into_image(self) -> None:
        box = CropBox(-10, 20, 100, 140).clamped(1000, 1000)
        self.assertEqual((box.x, box.y, box.width, box.height), (0, 20, 100, 140))
        box = CropBox(950, 900, 100, 140).clamped(1000, 1000)
        self.assertEqual((box.x, box.y), (900, 860))

    def test_scales_oversized_box_preserving_aspect(self) -> None:
        box = CropBox(0, 0, 2000, 2800).clamped(1000, 1000)
        self.assertAlmostEqual(box.height, 1000)
        self.assertAlmostEqual(box.width, 1000 * 2000 / 2800)

    def test_rejects_invalid_input(self) -> None:
        with self.assertRaises(ValueError):
            CropBox(0, 0, 10, 10).clamped(0, 100)
        with self.assertRaises(ValueError):
            CropBox(0, 0, 0, 10).clamped(100, 100)

    def test_to_int_rect_stays_inside_image(self) -> None:
        x0, y0, x1, y1 = CropBox(-20, -30, 80, 120).to_int_rect(50, 60)
        self.assertEqual((x0, y0, x1, y1), (0, 0, 40, 60))


class ComputeCropBoxTest(unittest.TestCase):
    def test_centers_face_and_uses_target_aspect(self) -> None:
        face = Face(x=100, y=80, width=100, height=150, score=0.9)
        box = compute_crop_box(face, 1000, 1000, ONE_INCH)
        self.assertAlmostEqual(box.height, 300)
        self.assertAlmostEqual(box.width, 300 * ONE_INCH.aspect_ratio)
        self.assertAlmostEqual(box.x + box.width / 2, face.center_x)
        self.assertAlmostEqual(box.y, 80 - 150 * 0.45)

    def test_clamps_large_crop_to_image(self) -> None:
        face = Face(x=-20, y=-30, width=200, height=300, score=0.9)
        box = compute_crop_box(face, 400, 400, ONE_INCH)
        self.assertAlmostEqual(box.height, 400)
        self.assertAlmostEqual(box.width / box.height, ONE_INCH.aspect_ratio, places=3)
        self.assertGreaterEqual(box.x, 0)
        self.assertGreaterEqual(box.y, 0)
        self.assertLessEqual(box.right, 400)
        self.assertLessEqual(box.bottom, 400)

    def test_two_inch_uses_its_own_aspect(self) -> None:
        face = Face(x=200, y=200, width=120, height=180, score=0.9)
        box = compute_crop_box(face, 1200, 1200, TWO_INCH)
        self.assertAlmostEqual(box.height, 360)
        self.assertAlmostEqual(box.width / box.height, TWO_INCH.aspect_ratio, places=3)


class RenderCropTest(unittest.TestCase):
    def test_output_size_and_transparent_background(self) -> None:
        rgba = np.zeros((300, 200, 4), dtype=np.uint8)
        rgba[:, :, 3] = 0
        photo = render_crop(
            rgba, CropBox(0, 0, 200, 300), ONE_INCH.pixels, (67, 142, 219)
        )
        self.assertEqual(photo.shape, (413, 295, 3))
        self.assertTrue((photo == np.array([67, 142, 219], dtype=np.uint8)).all())

    def test_opaque_foreground_is_preserved(self) -> None:
        rgba = np.zeros((300, 200, 4), dtype=np.uint8)
        rgba[:, :, :3] = (10, 200, 30)
        rgba[:, :, 3] = 255
        photo = render_crop(rgba, CropBox(0, 0, 200, 300), (100, 140), (255, 255, 255))
        self.assertEqual(photo.shape, (140, 100, 3))
        self.assertTrue((photo == np.array([10, 200, 30], dtype=np.uint8)).all())

    def test_half_alpha_blends_with_background(self) -> None:
        rgba = np.zeros((100, 100, 4), dtype=np.uint8)
        rgba[:, :, :3] = (200, 100, 50)
        rgba[:, :, 3] = 128
        photo = render_crop(rgba, CropBox(0, 0, 100, 100), (100, 100), (0, 0, 0))
        expected = round(200 * 128 / 255)
        self.assertAlmostEqual(int(photo[50, 50, 0]), expected, delta=1)

    def test_crop_box_selects_region(self) -> None:
        rgba = np.zeros((100, 100, 4), dtype=np.uint8)
        rgba[:, :50, :3] = (255, 0, 0)
        rgba[:, :50, 3] = 255
        photo = render_crop(rgba, CropBox(0, 0, 50, 100), (50, 100), (0, 255, 0))
        self.assertTrue((photo == np.array([255, 0, 0], dtype=np.uint8)).all())

    def test_rejects_non_rgba(self) -> None:
        with self.assertRaises(ValueError):
            render_crop(np.zeros((10, 10, 3), dtype=np.uint8), CropBox(0, 0, 10, 10), (5, 5), (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
