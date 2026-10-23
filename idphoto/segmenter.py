from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .config import (
    ALPHA_MATTING,
    ALPHA_MATTING_BACKGROUND_THRESHOLD,
    ALPHA_MATTING_ERODE_SIZE,
    ALPHA_MATTING_FOREGROUND_THRESHOLD,
    ALPHA_REFINE_RADIUS,
    BACKGROUND_SOLID_TOLERANCE,
    COLOR_LINE_RADIUS,
    DECONTAMINATE_FLOOR,
    EDGE_TRIM_PX,
    FILL_MASK_HOLES,
    SEGMENT_MODEL,
    U2NET_MODEL_DIR,
)


# --------------------------------------------------------------------------- #
# 后处理工具
# --------------------------------------------------------------------------- #
def fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    填补 mask 内部的空洞。

    模型经常把深色头发、深色衣服误判成背景，在人物内部挖出一块块洞；
    直接合成会在头发/衣服上留下背景色的斑点。做法是先把 mask 反相，
    从图像外边界灌水，灌不到的区域就是被前景包围的内部空洞，补上即可。
    """
    inverse = cv2.bitwise_not(mask)
    # 补 1px 边框，保证即使人物贴着图像边缘也有可灌水的外边界
    padded = cv2.copyMakeBorder(inverse, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
    flood_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(padded, flood_mask, (0, 0), 0)
    holes = padded[1:-1, 1:-1]
    return cv2.bitwise_or(mask, holes)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """只保留最大的前景连通域，去掉模型误判出来的零散小块。"""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 2:
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def refine_alpha(alpha: np.ndarray, image_rgb: np.ndarray, radius: int) -> np.ndarray:
    """
    用引导滤波（guided filter）把 alpha 的边缘对齐到真实图像边缘。

    模型输出的 alpha 是低分辨率上采样来的，在发丝处是"糊"的：该透的地方不够透，
    该实的地方不够实，于是轮廓外侧残留一层半透明像素 —— 换成深色底就是白边。
    引导滤波以灰度图为引导，能把这一层重新拉回到真实轮廓上。
    """
    if radius <= 0:
        return alpha

    guide = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    src = alpha.astype(np.float32)
    size = (radius * 2 + 1, radius * 2 + 1)
    eps = 1e-4

    mean_guide = cv2.blur(guide, size)
    mean_src = cv2.blur(src, size)
    cov = cv2.blur(guide * src, size) - mean_guide * mean_src
    var = cv2.blur(guide * guide, size) - mean_guide * mean_guide
    a = cov / (var + eps)
    b = mean_src - a * mean_guide
    return np.clip(cv2.blur(a, size) * guide + cv2.blur(b, size), 0.0, 1.0)


def estimate_background_field(
    image_rgb: np.ndarray,
    alpha: np.ndarray,
    *,
    cell: int | None = None,
    tolerance: float = BACKGROUND_SOLID_TOLERANCE,
) -> np.ndarray | None:
    """
    用 alpha≈0 的像素拟合一张**平滑的背景色场** B(x,y)，而不是只取一个全局颜色。

    为什么不能只取一个颜色：真实照片的背景几乎从来不是绝对纯色 —— 墙面明暗不均、
    影棚布光渐变、镜头暗角都会让背景色在画面上漂移几十个灰阶。若只用一个中位数去
    反解，估偏的那部分背景色会原样留在边缘上，深色底上依然是白边。用低频场跟上这种
    缓变，边缘才干净。

    仍然保留闸门：只有背景**足够平滑**（能被低频场拟合）时才返回场，否则返回 None。
    判定看的是"背景像素与拟合场的残差"，而不是背景本身的起伏，因此渐变背景能通过、
    真正杂乱（树叶、碎花、贴纸）的背景会被挡下，避免把前景颜色改坏。
    """
    height, width = alpha.shape
    if cell is None:
        cell = max(8, min(height, width) // 16)

    background_mask = alpha < 0.02
    if int(background_mask.sum()) < 64:
        return None

    # 把画面切成 cell×cell 的小块，只统计每块里的背景像素，得到一张低分辨率背景图。
    rows = (height + cell - 1) // cell
    cols = (width + cell - 1) // cell
    padded_image = cv2.copyMakeBorder(
        image_rgb.astype(np.float32), 0, rows * cell - height, 0, cols * cell - width,
        cv2.BORDER_REPLICATE,
    )
    padded_mask = cv2.copyMakeBorder(
        background_mask.astype(np.uint8), 0, rows * cell - height, 0, cols * cell - width,
        cv2.BORDER_CONSTANT, value=0,
    )

    blocks = padded_image.reshape(rows, cell, cols, cell, 3)
    masks = padded_mask.reshape(rows, cell, cols, cell, 1).astype(np.float32)
    counts = masks.sum(axis=(1, 3))[:, :, 0]
    valid = counts > 0
    if int(valid.sum()) < 4:
        return None

    sums = (blocks * masks).sum(axis=(1, 3))
    coarse = np.zeros((rows, cols, 3), np.float32)
    coarse[valid] = sums[valid] / counts[valid][:, None]

    # 被人物盖住的块没有背景像素，用 inpaint 从相邻背景块外推到画面内部。
    coarse_u8 = np.rint(np.clip(coarse, 0, 255)).astype(np.uint8)
    holes = (~valid).astype(np.uint8) * 255
    if holes.any():
        coarse_u8 = cv2.inpaint(coarse_u8, holes, 3, cv2.INPAINT_TELEA)

    field = cv2.resize(
        coarse_u8.astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC
    )
    field = cv2.GaussianBlur(field, (0, 0), max(1.0, cell / 3.0))

    pixels = image_rgb.astype(np.float32)[background_mask]
    residual = float(np.abs(pixels - field[background_mask]).mean())
    return field if residual <= tolerance else None


def _extend_colour(source: np.ndarray, known: np.ndarray, sigma: float) -> np.ndarray:
    """
    用 known 区域内像素的**加权邻域均值**外推出整幅画面的颜色。

    必须是单边的：前景色只能由轮廓内侧的发丝推出来，背景色只能由外侧推出来。
    用 inpaint 时若待填充环的另一侧就是另一种颜色，填充值会被拉过去，alpha 就解不准。
    归一化卷积既天然单边，又能把高光/噪点平均掉，不会像最近邻取值那样出现亮斑。
    """
    weights = known.astype(np.float32)
    support = cv2.GaussianBlur(weights, (0, 0), sigma)
    colour = cv2.GaussianBlur(source.astype(np.float32) * weights[:, :, None], (0, 0), sigma)
    filled = colour / np.maximum(support, 1e-4)[:, :, None]
    # 离 known 太远的地方没有可信外推值，退回原图像素
    weak = support < 1e-3
    filled[weak] = source.astype(np.float32)[weak]
    return filled


def color_line_alpha(
    image_rgb: np.ndarray,
    alpha: np.ndarray,
    radius: int = COLOR_LINE_RADIUS,
    background: np.ndarray | None = None,
) -> np.ndarray:
    """
    在轮廓带内用**颜色线模型**重新解算 alpha。

    模型的 alpha 是低分辨率上采样来的，在柔软、虚焦的发丝边缘经常给得偏高：
    那片像素的颜色其实就是原背景色（浅灰白），却被打成不透明前景。这种像素靠
    decontaminate 救不回来 —— 减掉背景再除以 a，得到的仍是灰白，贴到蓝底上就是
    一圈白光。真正错的是 a 本身。

    对轮廓两侧 radius 像素的带内像素，用合成方程 C = a·F + (1-a)·B 反解：
        a = (C - B)·(F - B) / ‖F - B‖²
    F 取轮廓内侧（确定前景）外推出的颜色，B 取轮廓外侧（远处背景）外推出的颜色。
    发丝让 a 落在 0~1 之间，纯背景自然得 a≈0，纯头发得 a≈1。

    前背景色过于接近（例如白衬衫配白底）时投影没有意义，保留原 alpha。

    background 可以传入 estimate_background_field 已经算好的背景场，省掉一次
    背景外推；不传则从轮廓外侧自行 inpaint 估计。
    """
    if radius <= 0:
        return alpha

    binary = (alpha > 0.5).astype(np.uint8)
    if not binary.any():
        return alpha

    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8)
    inner = cv2.erode(binary, kernel)
    outer = cv2.dilate(binary, kernel)
    band = (outer > 0) & (inner == 0)
    if not band.any():
        return alpha

    source = np.ascontiguousarray(image_rgb, dtype=np.uint8)
    # F 只由轮廓内侧外推、B 只由外侧外推；sigma 与带宽同量级，保持局部性。
    foreground = _extend_colour(source, (inner > 0), float(max(2, radius)))
    if background is None:
        background = _extend_colour(source, (outer == 0), float(max(2, radius)))
    background = np.asarray(background, np.float32)

    direction = foreground - background
    denom = (direction ** 2).sum(axis=2)
    projection = ((source.astype(np.float32) - background) * direction).sum(axis=2)
    projection = np.clip(projection / np.maximum(denom, 1e-3), 0.0, 1.0)

    # 前背景色差太小的时候投影会退化成噪声，这些像素沿用模型给的 alpha。
    reliable = denom > 40.0 ** 2
    result = np.where(inner > 0, 1.0, 0.0).astype(np.float32)
    result[band] = np.where(reliable[band], projection[band], alpha[band])
    return result


def decontaminate(
    image_rgb: np.ndarray,
    alpha: np.ndarray,
    background_rgb: np.ndarray,
    floor: float = DECONTAMINATE_FLOOR,
) -> np.ndarray:
    """
    反解前景色，消除背景色污染（白边的真正解法）。

    相机/生成图里，半透明边缘像素满足合成方程：
        C = a * F + (1 - a) * B        F=真实前景色  B=原背景色  a=真实不透明度
    直接把 C 当前景色用，等于把 B 一起带到了新背景上，深色底上就是一圈白边。
    这里把 B 减掉、再除以 a，还原出干净的 F：
        F = (C - (1 - a) * B) / a
    a 很小时除法会放大噪点，用 floor 截断；这些像素本来几乎全透明，对结果无影响。

    background_rgb 既可以是单个颜色 (3,)，也可以是 estimate_background_field
    拟合出的逐像素背景场 (H, W, 3)，靠广播同时支持两种输入。
    """
    foreground = image_rgb.astype(np.float32)
    background = np.asarray(background_rgb, np.float32)
    safe_alpha = np.clip(alpha, floor, 1.0)[:, :, None]
    restored = (foreground - (1.0 - safe_alpha) * background) / safe_alpha
    return np.clip(restored, 0, 255).astype(np.uint8)


def trim_alpha(alpha: np.ndarray, pixels: int) -> np.ndarray:
    """沿轮廓内收若干像素，作为去白边的兜底手段（会牺牲一点发丝）。"""
    if pixels <= 0:
        return alpha
    kernel = np.ones((pixels * 2 + 1, pixels * 2 + 1), np.uint8)
    binary = (alpha * 255).astype(np.uint8)
    binary = cv2.erode(binary, kernel)
    binary = cv2.GaussianBlur(binary, (0, 0), 0.8)
    return np.clip(binary.astype(np.float32) / 255.0, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# 分割器
# --------------------------------------------------------------------------- #
class PortraitSegmenter:
    """
    人像抠图：rembg 模型出 alpha，再做后处理打磨边缘。

    输出 RGBA，其中 RGB 是**去污染后的前景色**，而不是原图像素 ——
    这是换底色不出白边的关键。
    """

    def __init__(
        self,
        model_name: str = SEGMENT_MODEL,
        model_dir: str | Path | None = None,
        *,
        alpha_matting: bool = ALPHA_MATTING,
        refine_radius: int = ALPHA_REFINE_RADIUS,
        decontaminate_background: bool = True,
        fill_mask_holes: bool = FILL_MASK_HOLES,
        edge_trim_px: int = EDGE_TRIM_PX,
        color_line_radius: int = COLOR_LINE_RADIUS,
    ) -> None:
        self.model_name = model_name
        self.model_dir = Path(model_dir) if model_dir else U2NET_MODEL_DIR
        self.alpha_matting = alpha_matting
        self.refine_radius = refine_radius
        self.decontaminate_background = decontaminate_background
        self.fill_holes = fill_mask_holes
        self.edge_trim_px = edge_trim_px
        self.color_line_radius = color_line_radius
        self._session = None

    def _load_session(self):
        if self._session is not None:
            return self._session

        model_path = self.model_dir / f"{self.model_name}.onnx"
        if not model_path.is_file():
            raise FileNotFoundError(
                f"缺少抠图模型: {model_path}，请先运行 ./scripts/setup.sh"
            )

        os.environ["U2NET_HOME"] = str(self.model_dir)
        # 系统 TBB 版本过旧时 numba 会告警并禁用线程层，workqueue 始终可用。
        os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
        from rembg import new_session

        self._session = new_session(self.model_name)
        return self._session

    def cutout(self, image_bgr: np.ndarray) -> np.ndarray:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("输入必须是 BGR 三通道图片")

        session = self._load_session()
        from rembg import remove

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = remove(
            Image.fromarray(rgb),
            session=session,
            alpha_matting=self.alpha_matting,
            alpha_matting_foreground_threshold=ALPHA_MATTING_FOREGROUND_THRESHOLD,
            alpha_matting_background_threshold=ALPHA_MATTING_BACKGROUND_THRESHOLD,
            alpha_matting_erode_size=ALPHA_MATTING_ERODE_SIZE,
        )
        rgba = np.asarray(result)
        if rgba.ndim != 3 or rgba.shape[2] != 4:
            raise RuntimeError("抠图模型未返回 RGBA 结果")

        return self.postprocess(rgb, rgba)

    def postprocess(self, image_rgb: np.ndarray, rgba: np.ndarray) -> np.ndarray:
        """对模型原样输出做打磨：补洞 -> 细化 alpha -> 颜色线重解 -> 去背景色污染。"""
        alpha = rgba[:, :, 3].astype(np.float32) / 255.0

        if self.fill_holes:
            solid = (alpha > 0.5).astype(np.uint8) * 255
            filled = keep_largest_component(fill_holes(solid))
            # 只把"新补出来的洞"置为不透明，原有软边保持不动；
            # 否则会把整条过渡带二值化，反而把边缘变硬、白边更明显。
            newly_filled = (filled > 0) & (solid == 0)
            alpha = np.where(newly_filled, 1.0, alpha)

        alpha = refine_alpha(alpha, image_rgb, self.refine_radius)

        foreground = image_rgb
        background = None
        if self.decontaminate_background:
            background = estimate_background_field(image_rgb, alpha)
            # 只有背景足够平滑（背景场可信）时才敢用颜色线重解 alpha，
            # 否则 F/B 都估不准，投影会把边缘改坏。
            if background is not None:
                alpha = color_line_alpha(image_rgb, alpha, self.color_line_radius, background)
                foreground = decontaminate(image_rgb, alpha, background)

        alpha = trim_alpha(alpha, self.edge_trim_px)

        out = np.dstack([foreground, np.rint(alpha * 255.0).astype(np.uint8)])
        return np.ascontiguousarray(out)
