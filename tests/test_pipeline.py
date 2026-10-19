from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from idphoto.compose import CropBox
from idphoto.config import (
    BACKGROUND_COLORS,
    ONE_INCH,
    TWO_INCH,
    U2NET_MODEL_DIR,
    YUNET_MODEL_PATH,
)
from idphoto.detector import Face, FaceNotFoundError
from idphoto.io_utils import read_image
from idphoto.pipeline import IDPhotoProcessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRIVATE_PORTRAIT = PROJECT_ROOT / "assets" / "test" / "private" / "portrait_source.png"
INTEGRATION_READY = (
    PRIVATE_PORTRAIT.is_file()
    and YUNET_MODEL_PATH.is_file()
    and (U2NET_MODEL_DIR / "u2net.onnx").is_file()
)


class StubDetector:
    def __init__(self, face: Face | None) -> None:
        self.face = face

    def detect_primary(self, _image: np.ndarray) -> Face | None:
        return self.face

    def detect(self, _image: np.ndarray) -> list[Face]:
        return [self.face] if self.face else []


class StubSegmenter:
    def __init__(self, rgba: np.ndarray) -> None:
        self.rgba = rgba

    def cutout(self, _image: np.ndarray) -> np.ndarray:
        return self.rgba


def make_image() -> np.ndarray:
    return np.full((900, 600, 3), 128, dtype=np.uint8)


def make_cutout() -> np.ndarray:
    rgba = np.zeros((900, 600, 4), dtype=np.uint8)
    rgba[:, :, :3] = (10, 200, 30)
    rgba[200:390, 220:380, 3] = 255
    return rgba


def make_processor() -> tuple[IDPhotoProcessor, Face]:
    face = Face(x=250, y=200, width=100, height=130, score=0.95)
    return (
        IDPhotoProcessor(
            detector=StubDetector(face),
            segmenter=StubSegmenter(make_cutout()),
        ),
        face,
    )


class PipelineTest(unittest.TestCase):
    def test_process_returns_sized_photo_and_metadata(self) -> None:
        processor, face = make_processor()
        result = processor.process(
            make_image(), size=ONE_INCH, background_rgb=BACKGROUND_COLORS["blue"]
        )
        self.assertEqual(result.photo.shape, (413, 295, 3))
        self.assertIs(result.face, face)
        self.assertEqual(result.size, ONE_INCH)
        self.assertEqual(result.background, (67, 142, 219))
        self.assertEqual(result.cutout.shape, (900, 600, 4))
        self.assertAlmostEqual(
            result.crop_box.width / result.crop_box.height,
            ONE_INCH.aspect_ratio,
            places=3,
        )

    def test_transparent_corners_use_background(self) -> None:
        processor, _ = make_processor()
        result = processor.process(
            make_image(), size=ONE_INCH, background_rgb=BACKGROUND_COLORS["red"]
        )
        self.assertEqual(tuple(result.photo[0, 0]), (255, 0, 0))
        center = result.photo[206, 147]
        self.assertEqual(tuple(center), (10, 200, 30))

    def test_manual_crop_box_overrides_auto_framing(self) -> None:
        processor, _ = make_processor()
        result = processor.process(
            make_image(),
            size=ONE_INCH,
            background_rgb=BACKGROUND_COLORS["white"],
            crop_box=CropBox(0, 0, 300, 420),
        )
        self.assertAlmostEqual(result.crop_box.x, 0)
        self.assertAlmostEqual(result.crop_box.y, 0)
        self.assertAlmostEqual(result.crop_box.width, 300)
        self.assertAlmostEqual(result.crop_box.height, 420)
        self.assertEqual(tuple(result.photo[0, 0]), (255, 255, 255))

    def test_manual_crop_box_is_clamped(self) -> None:
        processor, _ = make_processor()
        result = processor.process(
            make_image(), crop_box=CropBox(590, 890, 300, 420)
        )
        self.assertLessEqual(result.crop_box.right, 600)
        self.assertLessEqual(result.crop_box.bottom, 900)

    def test_missing_face_raises(self) -> None:
        processor = IDPhotoProcessor(
            detector=StubDetector(None),
            segmenter=StubSegmenter(make_cutout()),
        )
        with self.assertRaises(FaceNotFoundError):
            processor.process(make_image())

    def test_process_file_writes_jpeg_with_size_and_dpi(self) -> None:
        processor, _ = make_processor()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "photo.jpg"
            processor.process_file(
                _write_temp_input(directory),
                output,
                size=ONE_INCH,
                background_rgb=BACKGROUND_COLORS["red"],
            )
            self.assertTrue(output.is_file())
            with Image.open(output) as image:
                self.assertEqual(image.size, (295, 413))
                dpi = image.info.get("dpi")
                self.assertIsNotNone(dpi)
                self.assertAlmostEqual(dpi[0], 300, delta=1)
                corner = image.convert("RGB").getpixel((0, 0))
                self.assertGreater(corner[0], 245)
                self.assertLess(corner[1], 10)
                self.assertLess(corner[2], 10)


def _write_temp_input(directory: str) -> Path:
    path = Path(directory) / "input.png"
    Image.fromarray(make_image()[:, :, ::-1]).save(path)
    return path


@unittest.skipUnless(INTEGRATION_READY, "缺少真实人像或模型，跳过集成测试")
class RealPortraitIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.image = read_image(PRIVATE_PORTRAIT)
        cls.processor = IDPhotoProcessor()

    def test_white_background_one_inch(self) -> None:
        result = self.processor.process(
            self.image, size=ONE_INCH, background_rgb=BACKGROUND_COLORS["white"]
        )
        self.assertEqual(result.photo.shape, (413, 295, 3))
        self.assertGreater(result.face.score, 0.8)
        coverage = float((result.cutout[:, :, 3] > 127).mean())
        self.assertGreater(coverage, 0.2)
        self.assertGreater(int(result.photo[0, 0].min()), 245)
        self.assertLessEqual(result.crop_box.right, self.image.shape[1])
        self.assertLessEqual(result.crop_box.bottom, self.image.shape[0])

    def test_red_background_two_inch(self) -> None:
        result = self.processor.process(
            self.image, size=TWO_INCH, background_rgb=BACKGROUND_COLORS["red"]
        )
        self.assertEqual(result.photo.shape, (579, 413, 3))
        red, green, blue = (int(channel) for channel in result.photo[0, 0])
        self.assertGreater(red, 245)
        self.assertLess(green, 10)
        self.assertLess(blue, 10)


if __name__ == "__main__":
    unittest.main()
