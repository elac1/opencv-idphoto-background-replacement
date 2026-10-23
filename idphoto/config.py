from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
YUNET_MODEL_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
U2NET_MODEL_DIR = MODELS_DIR / "rembg"
DEFAULT_DPI = 300

# 抠图模型名（rembg session 名）。可选 u2net / u2net_human_seg / isnet-general-use 等。
# 换模型的唯一改动点在这里，setup.sh 会自动下载 `{SEGMENT_MODEL}.onnx`。
SEGMENT_MODEL = "isnet-general-use"

CROP_HEIGHT_FACTOR = 2.0
CROP_TOP_FACTOR = 0.45

# YuNet 检测前把长边缩到这个上限再检测，检测完把坐标按比例还原。
# 大图直接喂进 YuNet 会把置信度压得非常低：实测一张 1760x2464 的照片，
# 原尺寸只有 0.654（并被默认阈值 0.85 拒掉，还多报一张假脸），
# 缩到长边 1080 后升到 0.924。顺带推理也更快。
DETECT_MAX_SIDE = 1080

# --- 抠图后处理 ---------------------------------------------------------------
# 是否用 rembg 的 alpha matting（pymatting 闭式抠图）。
# 开：发丝过渡带最自然、残留白边最轻（实测环带亮度再降约 23%），代价是耗时翻倍。
# 关：只走下面的后处理管线，速度快。追求性价比可以关掉。
ALPHA_MATTING = True
ALPHA_MATTING_FOREGROUND_THRESHOLD = 240
ALPHA_MATTING_BACKGROUND_THRESHOLD = 15
ALPHA_MATTING_ERODE_SIZE = 8

# alpha 引导滤波细化的窗口半径。越大越贴合真实轮廓，过大会把细节磨平。
ALPHA_REFINE_RADIUS = 6

# 是否填补 mask 内部空洞（深色头发/衣服容易被模型挖洞）。
FILL_MASK_HOLES = True

# 去背景色污染（消除白边）的下限：alpha 低于该值时不参与反解，避免除零放大噪点。
DECONTAMINATE_FLOOR = 0.25

# 背景场拟合的残差容差：alpha≈0 区域与拟合背景场的平均偏差超过它，就不做反解。
# 注意判的是"残差"而非背景本身的起伏，所以渐变背景能通过、杂乱背景会被挡下。
BACKGROUND_SOLID_TOLERANCE = 20.0

# 沿轮廓额外内收的像素数，作为去白边之后的兜底（0 表示不额外收缩）。
EDGE_TRIM_PX = 0

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
