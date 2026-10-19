#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ ! -x .venv/bin/python ]; then
  rm -rf .venv
  if ! python3 -m venv .venv 2>/dev/null; then
    # Ubuntu 默认可能缺少 python3-venv；用用户级 virtualenv 绕过，无需 sudo。
    rm -rf .venv
    python3 -m pip install --user --upgrade virtualenv
    python3 -m virtualenv -p python3 .venv
  fi
fi

.venv/bin/python -m pip install --upgrade pip wheel
.venv/bin/pip install -r requirements.txt

# rembg 依赖 opencv-python-headless，会与桌面 GUI 版 cv2 冲突（缺少 imshow）。
# 运行库已写入 requirements.txt，这里只装 rembg 本体，不装它声明的 OpenCV。
.venv/bin/pip install --no-deps rembg==2.0.69

mkdir -p models/rembg
if [ ! -s models/face_detection_yunet_2023mar.onnx ]; then
  echo "Downloading YuNet face model (~232 KB)..."
  curl -fL --retry 3 -o models/face_detection_yunet_2023mar.onnx \
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
fi
if [ "$(stat -c%s models/face_detection_yunet_2023mar.onnx)" -lt 200000 ]; then
  echo "YuNet model download failed (file too small, possibly an LFS pointer)." >&2
  exit 1
fi

echo "Preparing U2Net segmentation model (~176 MB, first run only)..."
U2NET_PATH="$ROOT/models/rembg/u2net.onnx"
U2NET_SHA256="8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491"
if [ ! -s "$U2NET_PATH" ]; then
  # GitHub 直连在国内经常只有 ~100KB/s，优先走 HF 镜像并校验哈希。
  if curl -fL --retry 3 -o "$U2NET_PATH.part" \
      "${U2NET_MIRROR_URL:-https://hf-mirror.com/tomjackson2023/rembg/resolve/main/u2net.onnx}" \
      && [ "$(sha256sum "$U2NET_PATH.part" | cut -d' ' -f1)" = "$U2NET_SHA256" ]; then
    mv "$U2NET_PATH.part" "$U2NET_PATH"
  else
    rm -f "$U2NET_PATH.part"
    echo "Mirror unavailable, falling back to rembg's GitHub downloader..."
    U2NET_HOME="$ROOT/models/rembg" .venv/bin/python - <<'PY'
from rembg import new_session
new_session("u2net")
PY
  fi
fi
if [ "$(sha256sum "$U2NET_PATH" | cut -d' ' -f1)" != "$U2NET_SHA256" ]; then
  echo "U2Net model checksum mismatch." >&2
  exit 1
fi
U2NET_HOME="$ROOT/models/rembg" .venv/bin/python - <<'PY'
from rembg import new_session
new_session("u2net")
print("U2Net model ready")
PY

.venv/bin/python - <<'PY'
import cv2
import numpy
import onnxruntime
import PIL
import rembg

assert hasattr(cv2, "imshow"), "OpenCV GUI build missing (imshow not found)"
assert hasattr(cv2, "FaceDetectorYN"), "YuNet API missing in OpenCV build"
print(f"opencv={cv2.__version__} numpy={numpy.__version__} "
      f"onnxruntime={onnxruntime.__version__} pillow={PIL.__version__} rembg={rembg.__version__ if hasattr(rembg, '__version__') else 'ok'}")
PY

echo "Setup complete."
