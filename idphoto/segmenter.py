from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .config import U2NET_MODEL_DIR


class PortraitSegmenter:
    def __init__(
        self,
        model_name: str = "u2net",
        model_dir: str | Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.model_dir = Path(model_dir) if model_dir else U2NET_MODEL_DIR
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
        result = remove(Image.fromarray(rgb), session=session)
        rgba = np.asarray(result)
        if rgba.ndim != 3 or rgba.shape[2] != 4:
            raise RuntimeError("抠图模型未返回 RGBA 结果")
        return rgba
