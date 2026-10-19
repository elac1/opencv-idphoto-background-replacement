from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
YUNET_MODEL_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
U2NET_MODEL_DIR = MODELS_DIR / "rembg"
DEFAULT_DPI = 300

CROP_HEIGHT_FACTOR = 2.0
CROP_TOP_FACTOR = 0.45

BACKGROUND_COLORS: dict[str, tuple[int, int, int]] = {
    "white": (255, 255, 255),
    "blue": (67, 142, 219),
    "red": (255, 0, 0),
}


@dataclass(frozen=True)
class IDPhotoSize:
    key: str
    label: str
    width_px: int
    height_px: int
    dpi: int = DEFAULT_DPI
    width_mm: float | None = None
    height_mm: float | None = None

    @classmethod
    def from_mm(
        cls,
        key: str,
        label: str,
        width_mm: float,
        height_mm: float,
        dpi: int = DEFAULT_DPI,
    ) -> IDPhotoSize:
        return cls(
            key=key,
            label=label,
            width_px=round(width_mm / 25.4 * dpi),
            height_px=round(height_mm / 25.4 * dpi),
            dpi=dpi,
            width_mm=width_mm,
            height_mm=height_mm,
        )

    @property
    def pixels(self) -> tuple[int, int]:
        return self.width_px, self.height_px

    @property
    def aspect_ratio(self) -> float:
        return self.width_px / self.height_px


ONE_INCH = IDPhotoSize.from_mm("one-inch", "一寸", 25, 35)
TWO_INCH = IDPhotoSize.from_mm("two-inch", "二寸", 35, 49)
SIZES = {size.key: size for size in (ONE_INCH, TWO_INCH)}


def custom_size(width_px: int, height_px: int, dpi: int = DEFAULT_DPI) -> IDPhotoSize:
    if width_px <= 0 or height_px <= 0:
        raise ValueError("尺寸必须是正整数")
    return IDPhotoSize("custom", "自定义", width_px, height_px, dpi)


def parse_rgb(value: str) -> tuple[int, int, int]:
    text = value.strip()
    is_hex = text.startswith("#") or ("," not in text and len(text) == 6)
    if is_hex:
        digits = text[1:] if text.startswith("#") else text
        if len(digits) != 6:
            raise ValueError("颜色格式应为 R,G,B 或 #RRGGBB")
        try:
            red, green, blue = (int(digits[index:index + 2], 16) for index in (0, 2, 4))
        except ValueError as exc:
            raise ValueError("颜色格式应为 R,G,B 或 #RRGGBB") from exc
    else:
        parts = [part.strip() for part in text.split(",")] if "," in text else []
        if len(parts) != 3:
            raise ValueError("颜色格式应为 R,G,B 或 #RRGGBB")
        try:
            red, green, blue = (int(part) for part in parts)
        except ValueError as exc:
            raise ValueError("颜色分量必须是 0-255 的整数") from exc
    if any(channel < 0 or channel > 255 for channel in (red, green, blue)):
        raise ValueError("颜色分量必须是 0-255 的整数")
    return red, green, blue
