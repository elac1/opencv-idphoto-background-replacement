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

# rembg 依赖 opencv-python-headless，会与项目使用的 opencv-python 冲突
# （两个 cv2 发行版互相覆盖）。运行库已写入 requirements.txt，这里只装 rembg 本体。
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

echo "Preparing segmentation model (first run only)..."
# 模型名与校验值以 idphoto/config.py 为准，换模型只需改那里。
SEGMENT_MODEL="$(
  .venv/bin/python -c "from idphoto.config import SEGMENT_MODEL; print(SEGMENT_MODEL)"
)"
SEGMENT_PATH="$ROOT/models/rembg/$SEGMENT_MODEL.onnx"
case "$SEGMENT_MODEL" in
  u2net)
    SEGMENT_SHA256="8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491"
    ;;
  isnet-general-use)
    # 与 rembg 官方 md5 fc16ebd8b0c10d971d3513d564d01e29 对应。
    SEGMENT_SHA256="60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a"
    ;;
  *)
    echo "Unknown SEGMENT_MODEL '$SEGMENT_MODEL'; falling back to rembg's downloader." >&2
    SEGMENT_SHA256=""
    ;;
esac
echo "  model: $SEGMENT_MODEL"

if [ ! -s "$SEGMENT_PATH" ]; then
  # GitHub 直连在国内经常只有 ~100KB/s，优先走 HF 镜像；没有对应镜像时回落官方源。
  if [ -n "${SEGMENT_MIRROR_URL:-}" ] \
      && curl -fL --retry 3 -o "$SEGMENT_PATH.part" "$SEGMENT_MIRROR_URL" \
      && { [ -z "$SEGMENT_SHA256" ] \
           || [ "$(sha256sum "$SEGMENT_PATH.part" | cut -d' ' -f1)" = "$SEGMENT_SHA256" ]; }; then
    mv "$SEGMENT_PATH.part" "$SEGMENT_PATH"
  elif [ "$SEGMENT_MODEL" = "u2net" ] \
      && curl -fL --retry 3 -o "$SEGMENT_PATH.part" \
           "${U2NET_MIRROR_URL:-https://hf-mirror.com/tomjackson2023/rembg/resolve/main/u2net.onnx}" \
      && [ "$(sha256sum "$SEGMENT_PATH.part" | cut -d' ' -f1)" = "$SEGMENT_SHA256" ]; then
    mv "$SEGMENT_PATH.part" "$SEGMENT_PATH"
  else
    rm -f "$SEGMENT_PATH.part"
    echo "Mirror unavailable, using rembg's GitHub downloader..."
    U2NET_HOME="$ROOT/models/rembg" .venv/bin/python - "$SEGMENT_MODEL" <<'PY'
import sys
from rembg import new_session
new_session(sys.argv[1])
PY
  fi
fi
if [ ! -s "$SEGMENT_PATH" ]; then
  echo "$SEGMENT_MODEL 下载后未出现在 $SEGMENT_PATH。" >&2
  echo "若 rembg 版本高于 2.0.69，它会把模型存到 <U2NET_HOME>/models/<name>/ 下，" >&2
  echo "请把该文件移到此路径，或改回 requirements.txt 锁定的版本。" >&2
  exit 1
fi
if [ -n "$SEGMENT_SHA256" ] \
    && [ "$(sha256sum "$SEGMENT_PATH" | cut -d' ' -f1)" != "$SEGMENT_SHA256" ]; then
  echo "$SEGMENT_MODEL checksum mismatch." >&2
  exit 1
fi
U2NET_HOME="$ROOT/models/rembg" .venv/bin/python - "$SEGMENT_MODEL" <<'PY'
import sys
from rembg import new_session
new_session(sys.argv[1])
print(f"{sys.argv[1]} model ready")
PY

.venv/bin/python - <<'PY'
import tkinter
import cv2
import numpy
import onnxruntime
import PIL
import rembg
import tkinterdnd2
from tkinterdnd2 import TkinterDnD

assert hasattr(cv2, "FaceDetectorYN"), "YuNet API missing in OpenCV build"
assert hasattr(TkinterDnD, "Tk"), "tkinterdnd2 is not usable"
print(f"opencv={cv2.__version__} numpy={numpy.__version__} "
      f"onnxruntime={onnxruntime.__version__} pillow={PIL.__version__} "
      f"rembg={rembg.__version__ if hasattr(rembg, '__version__') else 'ok'} "
      f"tkinter={tkinter.TkVersion} tkinterdnd2=ok")
PY

echo "Setup complete."
