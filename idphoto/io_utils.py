from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


class ImageLoadError(RuntimeError):
    pass


def read_image(path: str | Path) -> np.ndarray:
    image_path = Path(path)
    if not image_path.is_file():
        raise ImageLoadError(f"图片不存在: {image_path}")
    try:
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image)
            if "A" in image.getbands():
                rgba = image.convert("RGBA")
                background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                image = Image.alpha_composite(background, rgba)
            rgb = image.convert("RGB")
            array = np.asarray(rgb)
    except Exception as exc:
        raise ImageLoadError(f"无法读取图片: {image_path}") from exc
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


def write_photo(
    rgb: np.ndarray,
    path: str | Path,
    dpi: int = 300,
    quality: int = 95,
) -> Path:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("导出的图片必须是 uint8 RGB 数组")
    if not 1 <= quality <= 100:
        raise ValueError("JPEG 质量必须在 1-100 之间")

    output_path = Path(path)
    suffix = output_path.suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("仅支持导出 JPG 或 PNG")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.fromarray(rgb)
    if suffix in {".jpg", ".jpeg"}:
        image.save(
            output_path,
            format="JPEG",
            quality=quality,
            subsampling=0,
            optimize=True,
            dpi=(dpi, dpi),
        )
    else:
        image.save(output_path, format="PNG", optimize=True, dpi=(dpi, dpi))
    return output_path
