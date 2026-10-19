from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from .config import BACKGROUND_COLORS, SIZES, IDPhotoSize, custom_size, parse_rgb
from .detector import FaceDetector
from .io_utils import ImageLoadError
from .pipeline import IDPhotoProcessor
from .segmenter import PortraitSegmenter


def parse_dimensions(value: str) -> tuple[int, int]:
    parts = re.split(r"[xX*,]", value.strip())
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("尺寸格式应为 宽x高，例如 295x413")
    try:
        width, height = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("宽度和高度必须是整数") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("宽度和高度必须大于 0")
    return width, height


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m idphoto",
        description="自动检测人脸、按证件照构图、抠图并替换纯色背景",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python -m idphoto photo.jpg\n"
            "  python -m idphoto photo.jpg -o outputs/blue.jpg --bg blue --size two-inch\n"
            "  python -m idphoto photo.jpg --pixels 295x413 --rgb 67,142,219\n"
        ),
    )
    parser.add_argument("input", type=Path, help="输入人像图片路径")
    parser.add_argument("-o", "--output", type=Path, help="输出文件或目录，默认 outputs/")
    size_group = parser.add_mutually_exclusive_group()
    size_group.add_argument(
        "--size",
        choices=sorted(SIZES),
        default="one-inch",
        help="证件照规格，默认 one-inch",
    )
    size_group.add_argument(
        "--pixels",
        type=parse_dimensions,
        metavar="WxH",
        help="自定义像素尺寸，例如 295x413",
    )
    background_group = parser.add_mutually_exclusive_group()
    background_group.add_argument(
        "--bg",
        choices=sorted(BACKGROUND_COLORS),
        default="white",
        help="背景颜色，默认 white",
    )
    background_group.add_argument(
        "--rgb",
        type=parse_rgb,
        metavar="R,G,B",
        help="自定义背景颜色，支持 R,G,B 或 #RRGGBB",
    )
    parser.add_argument(
        "--detector",
        choices=("auto", "yunet", "haar"),
        default="auto",
        help="人脸检测器，默认 auto",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        metavar="1-100",
        help="JPEG 导出质量，默认 95",
    )
    return parser


def _size_slug(size: IDPhotoSize) -> str:
    return size.key if size.key != "custom" else f"{size.width_px}x{size.height_px}"


def _resolve_output(output: Path | None, input_path: Path, size: IDPhotoSize, tag: str) -> Path:
    filename = f"{input_path.stem}_{_size_slug(size)}_{tag}.jpg"
    if output is None:
        return Path("outputs") / filename
    if output.is_dir():
        return output / filename
    return output if output.suffix else output.with_suffix(".jpg")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    size = (
        custom_size(*args.pixels)
        if args.pixels
        else SIZES[args.size]
    )
    if args.rgb is not None:
        background_name, background_rgb = "rgb", args.rgb
    else:
        background_name, background_rgb = args.bg, BACKGROUND_COLORS[args.bg]
    if not 1 <= args.quality <= 100:
        parser.error("--quality 必须在 1-100 之间")

    output_path = _resolve_output(args.output, args.input, size, background_name)
    try:
        detector = FaceDetector(method=args.detector)
        processor = IDPhotoProcessor(detector=detector, segmenter=PortraitSegmenter())
        result = processor.process_file(
            args.input,
            output_path,
            size=size,
            background_rgb=background_rgb,
            quality=args.quality,
        )
    except (ImageLoadError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    face, crop = result.face, result.crop_box
    print(
        f"人脸: x={face.x:.1f}, y={face.y:.1f}, "
        f"w={face.width:.1f}, h={face.height:.1f}, score={face.score:.3f}"
    )
    print(
        f"裁剪框: x={crop.x:.1f}, y={crop.y:.1f}, "
        f"w={crop.width:.1f}, h={crop.height:.1f}"
    )
    print(
        f"已导出: {output_path} "
        f"({size.width_px}x{size.height_px} @ {size.dpi}dpi, 背景 {background_name})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
