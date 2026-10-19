from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import YUNET_MODEL_PATH


class FaceNotFoundError(RuntimeError):
    pass


@dataclass(frozen=True)
class Face:
    x: float
    y: float
    width: float
    height: float
    score: float
    landmarks: np.ndarray | None = None

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2

    @property
    def area(self) -> float:
        return self.width * self.height


class FaceDetector:
    def __init__(
        self,
        method: str = "auto",
        model_path: str | Path | None = None,
        score_threshold: float = 0.85,
    ) -> None:
        if method not in {"auto", "yunet", "haar"}:
            raise ValueError("method 必须是 auto、yunet 或 haar")
        self.model_path = Path(model_path) if model_path else YUNET_MODEL_PATH
        if method == "auto":
            method = "yunet" if self.model_path.is_file() else "haar"
            if method == "haar":
                warnings.warn("YuNet 模型不存在，自动改用 Haar 级联检测", stacklevel=2)
        if method == "yunet" and not self.model_path.is_file():
            raise FileNotFoundError(
                f"缺少 YuNet 模型: {self.model_path}，请先运行 ./scripts/setup.sh"
            )

        self.method = method
        self.score_threshold = score_threshold
        self._yunet: cv2.FaceDetectorYN | None = None
        self._haar: cv2.CascadeClassifier | None = None

    def detect(self, image_bgr: np.ndarray) -> list[Face]:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("输入必须是 BGR 三通道图片")
        height, width = image_bgr.shape[:2]
        if self.method == "yunet":
            return self._detect_yunet(image_bgr, width, height)
        return self._detect_haar(image_bgr)

    def detect_primary(self, image_bgr: np.ndarray) -> Face | None:
        faces = self.detect(image_bgr)
        return max(faces, key=lambda face: face.area, default=None)

    def _detect_yunet(
        self,
        image_bgr: np.ndarray,
        width: int,
        height: int,
    ) -> list[Face]:
        if self._yunet is None:
            self._yunet = cv2.FaceDetectorYN.create(
                str(self.model_path),
                "",
                (width, height),
                self.score_threshold,
                0.3,
                5000,
            )
        else:
            self._yunet.setInputSize((width, height))

        _, raw_faces = self._yunet.detect(image_bgr)
        if raw_faces is None:
            return []

        faces: list[Face] = []
        for row in raw_faces:
            score = float(row[14])
            if score < self.score_threshold:
                continue
            x, y, face_width, face_height = (float(value) for value in row[:4])
            faces.append(
                Face(
                    x=x,
                    y=y,
                    width=face_width,
                    height=face_height,
                    score=score,
                    landmarks=row[4:14].reshape(5, 2).astype(np.float32),
                )
            )
        return faces

    def _detect_haar(self, image_bgr: np.ndarray) -> list[Face]:
        if self._haar is None:
            haar_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            self._haar = cv2.CascadeClassifier(str(haar_path))
            if self._haar.empty():
                raise RuntimeError(f"无法加载 Haar 模型: {haar_path}")

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        min_side = max(30, min(image_bgr.shape[:2]) // 10)
        boxes = self._haar.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(min_side, min_side),
        )
        return [
            Face(
                x=float(x),
                y=float(y),
                width=float(width),
                height=float(height),
                score=1.0,
            )
            for x, y, width, height in boxes
        ]
