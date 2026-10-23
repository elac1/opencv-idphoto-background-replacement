# Models

- `face_detection_yunet_2023mar.onnx`: YuNet face detector from
  [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).
  MIT License, Copyright (c) 2020 Shiqi Yu.
- `rembg/`: 人像分割模型，由 `scripts/setup.sh` 按 `idphoto/config.py` 里的
  `SEGMENT_MODEL` 下载（默认 `isnet-general-use.onnx`），Git 忽略。
  Source: [rembg releases](https://github.com/danielgatis/rembg/releases)。
