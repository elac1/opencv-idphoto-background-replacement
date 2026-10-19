from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .compose import CropBox, compute_crop_box, render_crop
from .config import BACKGROUND_COLORS, ONE_INCH, IDPhotoSize
from .detector import Face, FaceDetector, FaceNotFoundError
from .io_utils import read_image, write_photo
from .segmenter import PortraitSegmenter


@dataclass
class PhotoResult:
    photo: np.ndarray
    face: Face
    crop_box: CropBox
    cutout: np.ndarray
    size: IDPhotoSize
    background: tuple[int, int, int]


class IDPhotoProcessor:
    def __init__(
        self,
        detector: FaceDetector | None = None,
        segmenter: PortraitSegmenter | None = None,
    ) -> None:
        self.detector = detector or FaceDetector()
        self.segmenter = segmenter or PortraitSegmenter()

    def detect(self, image_bgr: np.ndarray) -> Face:
        face = self.detector.detect_primary(image_bgr)
        if face is None:
            raise FaceNotFoundError("未检测到人脸，请使用光线充足、正面清晰的照片")
        return face

    def cutout(self, image_bgr: np.ndarray) -> np.ndarray:
        return self.segmenter.cutout(image_bgr)

    def render(
        self,
        cutout: np.ndarray,
        crop_box: CropBox,
        size: IDPhotoSize,
        background_rgb: tuple[int, int, int],
    ) -> np.ndarray:
        return render_crop(cutout, crop_box, size.pixels, background_rgb)

    def process(
        self,
        image_bgr: np.ndarray,
        size: IDPhotoSize = ONE_INCH,
        background_rgb: tuple[int, int, int] = BACKGROUND_COLORS["white"],
        crop_box: CropBox | None = None,
    ) -> PhotoResult:
        image_height, image_width = image_bgr.shape[:2]
        face = self.detect(image_bgr)
        if crop_box is None:
            crop_box = compute_crop_box(face, image_width, image_height, size)
        else:
            crop_box = crop_box.clamped(image_width, image_height)
        cutout = self.cutout(image_bgr)
        photo = self.render(cutout, crop_box, size, background_rgb)
        return PhotoResult(
            photo=photo,
            face=face,
            crop_box=crop_box,
            cutout=cutout,
            size=size,
            background=background_rgb,
        )

    def process_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        size: IDPhotoSize = ONE_INCH,
        background_rgb: tuple[int, int, int] = BACKGROUND_COLORS["white"],
        quality: int = 95,
    ) -> PhotoResult:
        image = read_image(input_path)
        result = self.process(image, size=size, background_rgb=background_rgb)
        write_photo(result.photo, output_path, dpi=size.dpi, quality=quality)
        return result
