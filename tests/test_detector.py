from __future__ import annotations

import unittest

import numpy as np

from idphoto.config import DETECT_MAX_SIDE, YUNET_MODEL_PATH
from idphoto.detector import FaceDetector

MODEL_READY = YUNET_MODEL_PATH.is_file()


class FakeYuNet:
    """替身：记录收到的输入尺寸，并返回一行固定的人脸数据。"""

    def __init__(self, row: np.ndarray) -> None:
        self.row = row
        self.sizes: list[tuple[int, int]] = []
        self.seen_shape: tuple[int, ...] | None = None

    def setInputSize(self, size: tuple[int, int]) -> None:
        self.sizes.append(size)

    def detect(self, image: np.ndarray):
        self.seen_shape = image.shape
        return None, self.row.reshape(1, -1)


def make_row(score: float = 0.9) -> np.ndarray:
    # YuNet 的输出：4 个框 + 5 个关键点(10 个数) + 分数
    return np.array(
        [100, 200, 300, 400, *([150, 250] * 5), score], dtype=np.float32
    )


@unittest.skipUnless(MODEL_READY, "缺少 YuNet 模型，跳过检测器测试")
class YunetScaleTest(unittest.TestCase):
    def make_detector(self, row: np.ndarray) -> tuple[FaceDetector, FakeYuNet]:
        detector = FaceDetector(method="yunet", score_threshold=0.5)
        fake = FakeYuNet(row)
        detector._yunet = fake  # 注入替身，避免依赖真实人脸
        return detector, fake

    def test_large_image_is_detected_at_reduced_size_and_mapped_back(self) -> None:
        height, width = 3000, 2000
        image = np.full((height, width, 3), 128, dtype=np.uint8)
        detector, fake = self.make_detector(make_row())

        faces = detector.detect(image)

        # 喂给 YuNet 的是缩小后的图，长边等于 DETECT_MAX_SIDE
        self.assertEqual(fake.sizes[0], (720, DETECT_MAX_SIDE))
        self.assertEqual(fake.seen_shape, (DETECT_MAX_SIDE, 720, 3))

        # 返回的坐标要还原回原图尺度
        back = width / 720
        self.assertEqual(len(faces), 1)
        self.assertAlmostEqual(faces[0].x, 100 * back, places=4)
        self.assertAlmostEqual(faces[0].y, 200 * back, places=4)
        self.assertAlmostEqual(faces[0].width, 300 * back, places=4)
        self.assertAlmostEqual(faces[0].height, 400 * back, places=4)
        self.assertAlmostEqual(
            float(faces[0].landmarks[0, 0]), 150 * back, places=3
        )

    def test_small_image_is_used_as_is(self) -> None:
        height, width = 640, 480
        image = np.full((height, width, 3), 128, dtype=np.uint8)
        detector, fake = self.make_detector(make_row())

        faces = detector.detect(image)

        self.assertEqual(fake.sizes[0], (width, height))
        self.assertEqual(len(faces), 1)
        # 没有缩放，坐标原样保留
        self.assertAlmostEqual(faces[0].x, 100.0, places=4)
        self.assertAlmostEqual(faces[0].width, 300.0, places=4)

    def test_portrait_image_shorter_than_limit_is_used_as_is(self) -> None:
        # 长边恰好等于上限时不缩放
        height, width = DETECT_MAX_SIDE, 700
        image = np.full((height, width, 3), 128, dtype=np.uint8)
        detector, fake = self.make_detector(make_row())

        detector.detect(image)

        self.assertEqual(fake.sizes[0], (width, height))

    def test_low_score_is_filtered_out(self) -> None:
        image = np.full((600, 400, 3), 128, dtype=np.uint8)
        detector, _ = self.make_detector(make_row(score=0.4))

        self.assertEqual(detector.detect(image), [])


if __name__ == "__main__":
    unittest.main()
