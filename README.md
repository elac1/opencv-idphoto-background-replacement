# 证件照自动裁剪与背景替换工具

基于 OpenCV + YuNet + rembg（U²-Net）的本地证件照工具：自动识别人脸、按标准比例构图、抠出人像并替换红/蓝/白/自定义纯色背景。提供 OpenCV 桌面界面和命令行两种用法，支持一寸、二寸及自定义像素尺寸。

![原始人像](docs/images/demo_source.jpg)

| 一寸 · 白底 | 一寸 · 蓝底 | 二寸 · 红底 |
| :---: | :---: | :---: |
| ![一寸白底](docs/images/demo_one_inch_white.jpg) | ![一寸蓝底](docs/images/demo_one_inch_blue.jpg) | ![二寸红底](docs/images/demo_two_inch_red.jpg) |
| 295×413 @ 300 DPI | 295×413 @ 300 DPI | 413×579 @ 300 DPI |

> 演示人像由 AI 生成，仅用于展示处理效果，不是真实人物。

## 功能

- **人脸检测**：默认使用 YuNet DNN，模型缺失时自动回退到 Haar 级联。
- **自动构图**：以人脸尺寸和位置计算证件照裁剪框，保证头部比例与留白。
- **人像抠图**：rembg + U²-Net，本地推理，无需上传照片。
- **背景替换**：白、蓝、红三种预设，或自定义 `R,G,B` / `#RRGGBB`。
- **标准规格**：一寸 25×35 mm、二寸 35×49 mm，按 300 DPI 输出。
- **手动微调**：拖动裁剪框移动，拖动四角按 5:7 比例缩放。
- **摄像头拍摄**：实时预览、镜像拍摄，拍完直接进入处理流程。
- **双入口**：桌面界面适合交互调整，命令行适合批量和脚本化。
- **隐私**：处理全程在本机完成；只有首次安装时需要联网下载模型。

## 环境与安装

要求：Python 3.10+、Linux 桌面环境（GUI 需要显示服务）。已在 Ubuntu + Python 3.10.12 上验证。

```bash
git clone https://github.com/elac1/opencv-idphoto-background-replacement.git
cd opencv-idphoto-background-replacement
./scripts/setup.sh
```

`setup.sh` 会：

1. 创建项目内虚拟环境 `.venv/`（系统缺少 `python3-venv` 时自动改用用户级 `virtualenv`，无需 sudo）。
2. 安装 `requirements.txt` 中锁定的依赖，并以 `--no-deps` 安装 rembg，避免它带入与桌面 GUI 冲突的 `opencv-python-headless`。
3. 下载 YuNet（约 232 KB）与 U²-Net（约 176 MB）模型到 `models/`，并校验 U²-Net 的 SHA-256。首次安装耗时取决于网络，之后可离线使用。

也可以手动安装：

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install --no-deps rembg==2.0.69
.venv/bin/python -m idphoto --help
```

模型文件会被自动忽略，不进入 Git；缺失时程序会提示先运行安装脚本。

## 桌面界面

```bash
./scripts/run.sh photo.jpg          # 打开指定图片
./scripts/run.sh                    # 无图片时自动打开 0 号摄像头
./scripts/run.sh --no-camera        # 打开空白编辑器
./scripts/run.sh --camera 1         # 指定摄像头索引
```

界面左侧为原图与裁剪框（框外区域压暗、显示三分线），右侧为实时预览和参数面板。

| 操作 | 按键 / 鼠标 |
| --- | --- |
| 打开图片 | `O` |
| 打开 / 关闭摄像头 | `C` |
| 摄像头拍摄 | `SPACE`（`ESC` 取消） |
| 保存到 `outputs/` | `S` |
| 按人脸重新自动构图 | `A` |
| 切到一寸 / 二寸 | `1` / `2` |
| 白 / 蓝 / 红底 | `W` / `B` / `R` |
| 移动裁剪框 | 在框内拖动 |
| 缩放裁剪框 | 拖动四角（保持输出比例） |
| 退出 | `Q` 或 `ESC` |

保存文件名为 `outputs/idphoto_YYYYmmdd_HHMMSS_{规格}_{底色}.jpg`，例如 `idphoto_20261019_143000_one-inch_white.jpg`。

## 命令行

```bash
# 一寸白底，默认输出到 outputs/
.venv/bin/python -m idphoto photo.jpg

# 二寸蓝底
.venv/bin/python -m idphoto photo.jpg --size two-inch --bg blue

# 自定义背景色（R,G,B 或 #RRGGBB）与输出路径
.venv/bin/python -m idphoto photo.jpg --rgb 67,142,219 -o outputs/custom.jpg
.venv/bin/python -m idphoto photo.jpg --rgb '#438EDB' -o outputs/

# 自定义像素尺寸
.venv/bin/python -m idphoto photo.jpg --pixels 295x413

# 强制使用 Haar 检测器（例如排查 YuNet 问题）
.venv/bin/python -m idphoto photo.jpg --detector haar
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `-o, --output` | 输出文件或目录；省略时为 `outputs/` |
| `--size` | `one-inch`（默认）或 `two-inch` |
| `--pixels WxH` | 自定义像素尺寸，与 `--size` 互斥 |
| `--bg` | `white`（默认）、`blue`、`red` |
| `--rgb` | 自定义背景色，与 `--bg` 互斥 |
| `--detector` | `auto`（默认）、`yunet`、`haar` |
| `--quality` | JPEG 质量 1–100，默认 95 |

默认输出名为 `{原文件名}_{规格}_{底色}.jpg`，例如 `photo_one-inch_white.jpg`。支持 JPG/JPEG/PNG 导出，JPEG 会写入 300 DPI 元数据并采用 4:4:4 色度采样。

## 规格与构图规则

| 规格 | 物理尺寸 | 输出像素（300 DPI） | 比例 |
| --- | --- | --- | --- |
| 一寸 | 25×35 mm | 295×413 | ≈5:7 |
| 二寸 | 35×49 mm | 413×579 | ≈5:7 |

自动构图以检测到的最大人脸为基准：

- 裁剪框高度 = `2.0 × 人脸高度`
- 裁剪框宽度 = `裁剪框高度 × 输出宽高比`
- 裁剪框上边 = `人脸上边 − 0.45 × 人脸高度`
- 人脸水平中心与裁剪框水平中心对齐，超出原图时自动向图内平移并等比调整。

这些参数集中在 `idphoto/config.py`，可按拍摄习惯调整。

## 工作流程

```text
读取图片（EXIF 转正）
      ↓
人脸检测（YuNet DNN / Haar 回退）
      ↓
计算裁剪框（人脸比例 + 输出宽高比）
      ↓
人像分割（rembg U²-Net → RGBA）
      ↓
预乘 Alpha 缩放与纯色背景合成
      ↓
按目标像素输出（JPEG/PNG，写入 DPI）
```

合成阶段使用预乘 Alpha，避免半透明发丝边缘在缩放后出现暗边；背景色使用 RGB 定义，在 OpenCV 的 BGR 边界处显式转换。

## 项目结构

```text
idphoto/
  config.py      规格、构图参数、背景色、颜色解析
  io_utils.py    图片读取（EXIF）与带 DPI 的导出
  detector.py    YuNet / Haar 人脸检测
  compose.py     裁剪框计算与 Alpha 合成
  segmenter.py   rembg U²-Net 人像分割
  pipeline.py    检测 → 抠图 → 合成 的编排
  cli.py         命令行入口
  app.py         OpenCV 桌面界面
scripts/
  setup.sh       环境、依赖与模型安装
  run.sh         一键启动桌面界面
tests/           单元测试与本地集成测试
docs/images/     README 演示图片
models/          模型目录（Git 忽略）
outputs/         默认输出目录（Git 忽略）
```

项目内的 Qoder 技能位于 `.qoder/skills/idphoto/SKILL.md`，汇总了安装、运行、验收和常见问题，便于 AI 编码助手快速接手。

## 测试

```bash
.venv/bin/python -m unittest discover -s tests -v
```

共 29 个测试，覆盖规格换算、颜色解析、构图与边界钳制、预乘合成、流水线编排、JPEG DPI 写入。真实人像集成测试需要私有测试图和两个模型，条件不满足时会自动跳过。

## 已知限制

- 每张图只处理**一个人脸**（取面积最大者），不支持合影分别裁切。
- 复杂背景、发丝细碎或前景与背景颜色接近时，抠图边缘可能出现残影；建议在纯色墙面前拍摄。
- 摄像头实时预览依赖 Qt 软件渲染，会占用较多 CPU（约单核 70%）；静态编辑模式已做脏标记优化，空闲时占用很低。
- 证件照尺寸要求因国家、地区、用途而异；本项目按常见的 25×35 mm / 35×49 mm 生成，正式提交前请以受理机构要求为准。
- 桌面界面需要图形显示；服务器或无显示环境请使用命令行。

## 模型与许可

| 组件 | 用途 | 许可 |
| --- | --- | --- |
| [OpenCV](https://github.com/opencv/opencv) | 图像处理与 GUI | Apache-2.0 |
| [YuNet / OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | 人脸检测 | MIT |
| [U²-Net](https://github.com/xuebinqin/U-2-Net) | 人像分割 | Apache-2.0 |
| [rembg](https://github.com/danielgatis/rembg) | 分割推理封装 | MIT |

模型权重在安装时从上游/镜像下载，不随仓库分发。使用本项目产出证件照时，请同时遵守图片来源、人物肖像权及受理机构的规范。
