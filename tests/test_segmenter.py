from __future__ import annotations

import unittest

import cv2
import numpy as np

from idphoto.segmenter import (
    PortraitSegmenter,
    color_line_alpha,
    decontaminate,
    estimate_background_field,
    fill_holes,
    keep_largest_component,
    refine_alpha,
    trim_alpha,
)


def make_mask(*, hole: bool = False, blob: bool = False) -> np.ndarray:
    mask = np.zeros((100, 100), np.uint8)
    cv2.rectangle(mask, (20, 20), (80, 80), 255, -1)
    if hole:
        cv2.rectangle(mask, (40, 40), (60, 60), 0, -1)
    if blob:
        cv2.rectangle(mask, (88, 88), (95, 95), 255, -1)
    return mask


class FillHolesTest(unittest.TestCase):
    def test_fills_enclosed_hole(self) -> None:
        filled = fill_holes(make_mask(hole=True))
        self.assertEqual(int(filled[50, 50]), 255)

    def test_keeps_exterior_background(self) -> None:
        filled = fill_holes(make_mask(hole=True))
        self.assertEqual(int(filled[5, 5]), 0)
        self.assertEqual(int(filled[50, 50]), 255)

    def test_keeps_subject_touching_border(self) -> None:
        mask = make_mask()
        mask[:, :10] = 255
        filler = fill_holes(mask)
        self.assertEqual(int(filler[50, 5]), 255)
        self.assertEqual(int(filler[98, 98]), 0)


class KeepLargestComponentTest(unittest.TestCase):
    def test_drops_small_blob(self) -> None:
        cleaned = keep_largest_component(make_mask(blob=True))
        self.assertEqual(int(cleaned[50, 50]), 255)
        self.assertEqual(int(cleaned[90, 90]), 0)

    def test_single_component_is_unchanged(self) -> None:
        mask = make_mask()
        self.assertTrue(np.array_equal(keep_largest_component(mask), mask))


class EstimateBackgroundFieldTest(unittest.TestCase):
    @staticmethod
    def scene(background: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        image = background.copy()
        image[30:90, 30:90] = 30
        alpha = np.zeros(image.shape[:2], np.float32)
        alpha[30:90, 30:90] = 1.0
        return image, alpha

    def test_fits_flat_background(self) -> None:
        image, alpha = self.scene(np.full((120, 120, 3), 240, np.uint8))
        field = estimate_background_field(image, alpha)
        self.assertIsNotNone(field)
        np.testing.assert_allclose(field[5, 5], [240, 240, 240], atol=3)

    def test_follows_gradient_background(self) -> None:
        """墙面明暗不均的渐变背景也要能跟上，否则边缘会残留原背景的色边。"""
        ramp = np.linspace(150.0, 250.0, 120, dtype=np.float32)
        background = np.broadcast_to(ramp[:, None, None], (120, 120, 3))
        image, alpha = self.scene(np.rint(background).astype(np.uint8))

        field = estimate_background_field(image, alpha)
        self.assertIsNotNone(field, "渐变背景应当被接受（旧版会因'非纯色'而放弃）")
        self.assertLess(abs(float(field[10, 60].mean()) - background[10, 60, 0]), 8)
        self.assertLess(abs(float(field[110, 60].mean()) - background[110, 60, 0]), 8)

    def test_rejects_busy_background(self) -> None:
        rng = np.random.default_rng(0)
        image = rng.integers(0, 255, (120, 120, 3), dtype=np.uint8)
        alpha = np.zeros((120, 120), np.float32)
        alpha[30:90, 30:90] = 1.0
        self.assertIsNone(estimate_background_field(image, alpha))

    def test_returns_none_when_no_background_pixels(self) -> None:
        image = np.full((20, 20, 3), 128, np.uint8)
        self.assertIsNone(estimate_background_field(image, np.ones((20, 20), np.float32)))


class DecontaminateTest(unittest.TestCase):
    def test_recovers_foreground_from_composited_edge(self) -> None:
        """
        构造已知真值：黑发 F=30 与白背景 B0=240 按 alpha 混合成 C。
        去污染后应当把边缘像素还原回接近黑发，而不是残留白色。
        """
        background = np.array([240.0, 240.0, 240.0])
        foreground = np.array([30.0, 30.0, 30.0])
        alpha = np.full((10, 10), 0.5, np.float32)
        composited = (alpha[:, :, None] * foreground + (1 - alpha[:, :, None]) * background)
        composited = composited.astype(np.uint8)

        restored = decontaminate(composited, alpha, background, floor=0.0)
        np.testing.assert_allclose(restored[0, 0], foreground, atol=6)

    def test_leaves_opaque_pixels_untouched(self) -> None:
        image = np.full((10, 10, 3), 77, np.uint8)
        restored = decontaminate(image, np.ones((10, 10), np.float32), [240, 240, 240])
        np.testing.assert_array_equal(restored[0, 0], [77, 77, 77])

    def test_clips_to_valid_range(self) -> None:
        image = np.full((10, 10, 3), 250, np.uint8)
        restored = decontaminate(image, np.full((10, 10), 0.5, np.float32), [0, 0, 0])
        self.assertTrue((restored <= 255).all())


class RefineAlphaTest(unittest.TestCase):
    def test_zero_radius_is_identity(self) -> None:
        alpha = np.linspace(0, 1, 100, dtype=np.float32).reshape(10, 10)
        self.assertTrue(np.array_equal(refine_alpha(alpha, np.zeros((10, 10, 3), np.uint8), 0), alpha))

    def test_keeps_output_in_unit_range(self) -> None:
        image = np.zeros((60, 60, 3), np.uint8)
        image[:, 30:] = 255
        alpha = np.zeros((60, 60), np.float32)
        alpha[:, 30:] = 1.0
        refined = refine_alpha(alpha, image, 6)
        self.assertGreaterEqual(float(refined.min()), 0.0)
        self.assertLessEqual(float(refined.max()), 1.0)


class ColorLineAlphaTest(unittest.TestCase):
    """轮廓带内按合成方程反解 alpha：模型把柔软发丝判成不透明时靠它救回来。"""

    FOREGROUND = np.array([28.0, 28.0, 30.0], np.float32)
    BACKGROUND = np.array([238.0, 238.0, 238.0], np.float32)

    @classmethod
    def make_scene(cls, inflation: int = 8):
        """白底上的深色圆盘，模型 alpha 比真实轮廓外扩 inflation 像素（经典白边成因）。"""
        size = 120
        true_alpha = np.zeros((size, size), np.float32)
        cv2.circle(true_alpha, (60, 60), 30, 1.0, -1)
        true_alpha = cv2.GaussianBlur(true_alpha, (0, 0), 2.0)

        model_alpha = np.zeros((size, size), np.float32)
        cv2.circle(model_alpha, (60, 60), 30 + inflation, 1.0, -1)
        model_alpha = cv2.GaussianBlur(model_alpha, (0, 0), 3.0)

        composited = (true_alpha[:, :, None] * cls.FOREGROUND
                      + (1 - true_alpha[:, :, None]) * cls.BACKGROUND)
        return composited.astype(np.uint8), model_alpha, true_alpha

    def test_zero_radius_is_identity(self) -> None:
        alpha = np.zeros((20, 20), np.float32)
        alpha[5:15, 5:15] = 1.0
        rgb = np.full((20, 20, 3), 200, np.uint8)
        self.assertTrue(np.array_equal(color_line_alpha(rgb, alpha, 0), alpha))

    def test_pulls_inflated_edge_back_to_real_silhouette(self) -> None:
        rgb, model_alpha, true_alpha = self.make_scene(inflation=8)
        # 外扩环带：模型说是不透明前景，真值其实是透的
        ring = (model_alpha > 0.5) & (true_alpha < 0.2)
        self.assertGreater(int(ring.sum()), 0, "构造的场景本身应当有外扩环带")

        fixed = color_line_alpha(rgb, model_alpha, radius=10)
        self.assertLess(float(fixed[ring].mean()), 0.5, "外扩环带的 alpha 应当被压下来")
        # 画面中心（真·前景）必须仍然不透明
        self.assertGreater(float(fixed[60, 60]), 0.95)

    def test_output_stays_in_unit_range(self) -> None:
        rgb, model_alpha, _ = self.make_scene()
        fixed = color_line_alpha(rgb, model_alpha, radius=8)
        self.assertGreaterEqual(float(fixed.min()), 0.0)
        self.assertLessEqual(float(fixed.max()), 1.0)

    def test_keeps_model_alpha_when_foreground_matches_background(self) -> None:
        """白衬衫配白底时前背景色无法区分，投影会退化，此时应保留模型原值。"""
        rgb = np.full((80, 80, 3), 236, np.uint8)
        alpha = np.zeros((80, 80), np.float32)
        cv2.circle(alpha, (40, 40), 24, 1.0, -1)
        fixed = color_line_alpha(rgb, alpha, radius=6)
        # 轮廓带上不应被压成 0（与纯背景同色时不能凭空判定为透明）
        band = (cv2.dilate((alpha > 0.5).astype(np.uint8), np.ones((13, 13), np.uint8)) > 0) \
            & (alpha > 0.5)
        self.assertGreater(float(fixed[band].min()), 0.4)


class TrimAlphaTest(unittest.TestCase):
    def test_zero_pixels_is_identity(self) -> None:
        alpha = np.ones((10, 10), np.float32)
        self.assertTrue(np.array_equal(trim_alpha(alpha, 0), alpha))

    def test_shrinks_the_silhouette(self) -> None:
        alpha = np.zeros((40, 40), np.float32)
        alpha[10:30, 10:30] = 1.0
        self.assertLess(float(trim_alpha(alpha, 2)[10, 20]), 0.5)


class PostprocessTest(unittest.TestCase):
    """整条后处理的回归测试：核心是"白底换深色底不出白边"。"""

    #: 真值前景色（深色"头发"）
    FOREGROUND = np.array([28.0, 28.0, 30.0])
    #: 拍摄时的原背景色（近白影棚底）
    BACKGROUND = np.array([238.0, 238.0, 238.0])

    def make_scene(self, inflation: int = 6):
        """
        构造一张物理自洽的"白底深色头发"照片，并复现真实白边的两个成因：

        1. 颜色污染：相机成像 C = a·F + (1-a)·B0，半透明边缘像素里夹着原背景色；
        2. alpha 高估：分割模型判出来的轮廓比真实轮廓胖一圈，
           于是"背景像素 + alpha≈1"的区域被当作不透明前景保留下来 ——
           这是换深色底后那圈明显白边的主因。

        inflation 就是模型轮廓比真实轮廓外扩的像素数。
        """
        true_alpha = np.zeros((120, 120), np.float32)
        cv2.circle(true_alpha, (60, 60), 34, 1.0, -1)
        true_alpha = cv2.GaussianBlur(true_alpha, (0, 0), 2.0)

        model_alpha = np.zeros((120, 120), np.float32)
        cv2.circle(model_alpha, (60, 60), 34 + inflation, 1.0, -1)
        model_alpha = cv2.GaussianBlur(model_alpha, (0, 0), 3.0)

        composited = (true_alpha[:, :, None] * self.FOREGROUND
                      + (1 - true_alpha[:, :, None]) * self.BACKGROUND)
        rgb = composited.astype(np.uint8)
        rgba = np.dstack([rgb, np.rint(model_alpha * 255).astype(np.uint8)])
        return rgb, rgba, true_alpha, model_alpha

    @staticmethod
    def count_light_halo(cutout: np.ndarray, background: tuple = (255, 0, 0)) -> int:
        """把抠图结果合成到红底上，统计可见的浅色白边像素数。"""
        alpha = cutout[:, :, 3].astype(np.float32) / 255.0
        red = np.array(background, np.float32)
        photo = cutout[:, :, :3].astype(np.float32) * alpha[:, :, None] + red * (1 - alpha[:, :, None])
        not_background = np.abs(photo - red).sum(axis=2) > 45
        # 纯红底的三通道最小值是 0，只有残留的白边会把最小值顶高
        return int((not_background & (photo.min(axis=2) > 150)).sum())

    def test_returns_rgba_of_same_shape(self) -> None:
        rgb, rgba, _, _ = self.make_scene()
        out = PortraitSegmenter().postprocess(rgb, rgba)
        self.assertEqual(out.shape, (120, 120, 4))

    def test_removes_light_halo_on_red_background(self) -> None:
        rgb, rgba, _, _ = self.make_scene()
        self.assertGreater(
            self.count_light_halo(rgba), 0, "构造的场景本身应当带有白边"
        )
        after = self.count_light_halo(PortraitSegmenter().postprocess(rgb, rgba))
        self.assertEqual(after, 0, "后处理应当消除白边")

    def test_recovers_foreground_colour_where_alpha_is_partial(self) -> None:
        """
        在"alpha 是真的一半"的过渡带上，原图像素 = 0.5*前景 + 0.5*原背景，
        因此发白（约 133）。去污染应当把它还原回接近真值前景色（约 28）。

        注意：这里用 inflation=0（模型轮廓没有外扩）。若模型轮廓比真实轮廓胖一圈，
        那些像素的 alpha 被高估成≈1，单靠去污染无从下手 —— 那一层由
        color_line_alpha 按颜色线模型单独反解（见 ColorLineAlphaTest）。
        """
        rgb, rgba, true_alpha, _ = self.make_scene(inflation=0)
        band = (true_alpha > 0.25) & (true_alpha < 0.75)
        self.assertGreater(int(band.sum()), 0)

        before = float(np.abs(rgb[band].astype(np.float32) - self.FOREGROUND).mean())
        after = float(np.abs(
            PortraitSegmenter().postprocess(rgb, rgba)[band][:, :3].astype(np.float32)
            - self.FOREGROUND
        ).mean())

        self.assertGreater(before, 80.0, "构造的过渡带本身应当明显发白")
        self.assertLess(after, before * 0.25, "去污染后过渡带应当接近真值前景色")

    def test_decontaminates_on_gradient_background(self) -> None:
        """
        原背景只是渐变（非纯色）时也要照常去污染：
        旧版按"纯色/非纯色"一刀切，渐变背景会被判为非纯色而整个跳过，白边复现。
        """
        true_alpha = np.zeros((120, 120), np.float32)
        cv2.circle(true_alpha, (60, 60), 34, 1.0, -1)
        true_alpha = cv2.GaussianBlur(true_alpha, (0, 0), 2.0)

        ramp = np.linspace(190.0, 250.0, 120, dtype=np.float32)
        background = np.broadcast_to(ramp[:, None, None], (120, 120, 3)).astype(np.float32)
        composited = (true_alpha[:, :, None] * self.FOREGROUND
                      + (1 - true_alpha[:, :, None]) * background)
        rgb = np.rint(composited).astype(np.uint8)
        rgba = np.dstack([rgb, np.rint(true_alpha * 255).astype(np.uint8)])

        band = (true_alpha > 0.25) & (true_alpha < 0.75)
        before = float(np.abs(rgb[band].astype(np.float32) - self.FOREGROUND).mean())
        after = float(np.abs(
            PortraitSegmenter().postprocess(rgb, rgba)[band][:, :3].astype(np.float32)
            - self.FOREGROUND
        ).mean())

        self.assertGreater(before, 60.0, "构造的过渡带本身应当被原背景染亮")
        self.assertLess(after, before * 0.3, "渐变背景下也应当把前景色还原回来")

    def test_skips_decontamination_on_busy_background(self) -> None:
        rng = np.random.default_rng(1)
        rgb = rng.integers(60, 200, (120, 120, 3), dtype=np.uint8)
        alpha = np.zeros((120, 120), np.float32)
        alpha[40:80, 40:80] = 1.0
        rgba = np.dstack([rgb, np.rint(alpha * 255).astype(np.uint8)])

        out = PortraitSegmenter().postprocess(rgb, rgba)
        # 背景杂乱时不应改动前景颜色
        np.testing.assert_array_equal(out[60, 60, :3], rgb[60, 60])


if __name__ == "__main__":
    unittest.main()
