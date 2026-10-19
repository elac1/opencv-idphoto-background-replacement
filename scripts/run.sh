#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -x .venv/bin/python ]]; then
  echo "缺少虚拟环境，请先运行 ./scripts/setup.sh" >&2
  exit 1
fi

if [[ ! -f models/face_detection_yunet_2023mar.onnx ]]; then
  echo "缺少人脸检测模型，请先运行 ./scripts/setup.sh" >&2
  exit 1
fi

if [[ ! -f models/rembg/u2net.onnx ]]; then
  echo "缺少抠图模型，请先运行 ./scripts/setup.sh" >&2
  exit 1
fi

if ! .venv/bin/python -c "import tkinter, tkinterdnd2" 2>/dev/null; then
  echo "缺少图形界面依赖（python3-tk 或 tkinterdnd2），请安装 python3-tk 后重新运行 ./scripts/setup.sh" >&2
  exit 1
fi

# 会打开桌面窗口；未指定图片且未加 --no-camera 时会自动调用摄像头。
exec .venv/bin/python -m idphoto.app "$@"
