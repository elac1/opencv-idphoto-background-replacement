from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import CROP_HEIGHT_FACTOR, CROP_TOP_FACTOR, IDPhotoSize
from .detector import Face


@dataclass
class CropBox:
    x: float
    y: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    def clamped(self, image_width: int, image_height: int) -> CropBox:
        if image_width <= 0 or image_height <= 0:
            raise ValueError("图片尺寸必须为正数")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("裁剪框尺寸必须为正数")

        scale = min(1.0, image_width / self.width, image_height / self.height)
        width = self.width * scale
        height = self.height * scale
        x = min(max(self.x, 0.0), image_width - width)
        y = min(max(self.y, 0.0), image_height - height)
        return CropBox(x=x, y=y, width=width, height=height)

    def to_int_rect(
        self,
        image_width: int,
        image_height: int,
    ) -> tuple[int, int, int, int]:
        box = self.clamped(image_width, image_height)
        x0 = max(0, min(image_width - 1, int(round(box.x))))
        y0 = max(0, min(image_height - 1, int(round(box.y))))
        x1 = max(x0 + 1, min(image_width, int(round(box.right))))
        y1 = max(y0 + 1, min(image_height, int(round(box.bottom))))
        return x0, y0, x1, y1


def compute_crop_box(
    face: Face,
    image_width: int,
    image_height: int,
    size: IDPhotoSize,
) -> CropBox:
    if face.width <= 0 or face.height <= 0:
        raise ValueError("人脸区域尺寸无效")

    crop_height = face.height * CROP_HEIGHT_FACTOR
    crop_width = crop_height * size.aspect_ratio
    x = face.center_x - crop_width / 2
    y = face.y - face.height * CROP_TOP_FACTOR
    return CropBox(x=x, y=y, width=crop_width, height=crop_height).clamped(
        image_width, image_height
    )


def render_crop(
    rgba: np.ndarray,
    crop_box: CropBox,
    output_size: tuple[int, int],
    background_rgb: tuple[int, int, int],
) -> np.ndarray:
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("抠图结果必须是 RGBA 图片")
    output_width, output_height = output_size
    if output_width <= 0 or output_height <= 0:
        raise ValueError("输出尺寸必须为正数")

    x0, y0, x1, y1 = crop_box.to_int_rect(rgba.shape[1], rgba.shape[0])
    crop = rgba[y0:y1, x0:x1]
    if crop.shape[0] < 1 or crop.shape[1] < 1:
        raise ValueError("裁剪框超出图片范围")

    # 预乘 alpha 后再缩放，避免透明区域的颜色在发丝边缘形成暗边。
    alpha = crop[:, :, 3].astype(np.float32) / 255.0
    foreground = crop[:, :, :3].astype(np.float32) * alpha[:, :, None]
    downscaling = crop.shape[1] >= output_width and crop.shape[0] >= output_height
    interpolation = cv2.INTER_AREA if downscaling else cv2.INTER_CUBIC
    alpha = cv2.resize(alpha, (output_width, output_height), interpolation=interpolation)
    foreground = cv2.resize(
        foreground, (output_width, output_height), interpolation=interpolation
    )

    background = np.asarray(background_rgb, dtype=np.float32).reshape(1, 1, 3)
    photo = foreground + background * (1.0 - alpha[:, :, None])
    return np.rint(np.clip(photo, 0, 255)).astype(np.uint8)
