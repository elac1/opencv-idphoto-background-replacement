---
name: idphoto
description: Operate and maintain the local ID-photo auto-crop and background-replacement app. Use when setting up the Python environment or models, running the OpenCV desktop editor or CLI, debugging YuNet face detection, crop framing, rembg U2Net cutout or alpha compositing, validating output sizes and DPI, running tests, or preparing Conventional Commits in this repo.
---

# ID Photo Studio

Local-only ID photo tool: YuNet detects the largest face, a 2×face-height crop is composed to a 5:7 frame, rembg U²-Net cuts out the portrait, and premultiplied-alpha compositing replaces the background.

## Commands

Run from the repo root. `.venv/bin/python` is Python 3.10.

### Setup

```bash
./scripts/setup.sh
```

Side effects: creates `.venv/` if missing, installs pinned dependencies, downloads YuNet (~232 KB) and U²-Net (~176 MB) into `models/` (gitignored). No sudo. First run needs network; later runs are offline.

### Desktop editor

```bash
./scripts/run.sh [IMAGE] [--camera N] [--no-camera]
```

Side effects: opens a 1280×800 window and blocks until quit. Without `IMAGE` and without `--no-camera`, it opens camera 0 (camera indicator turns on). `S` writes to `outputs/` (gitignored); camera capture is mirrored.

Controls: `O` open, `C` camera, `SPACE` capture, `ESC` cancel capture, `S` save, `A` auto-frame, `1`/`2` one/two-inch, `W`/`B`/`R` white/blue/red, drag inside to move the crop, drag a corner to resize at the fixed output ratio, `Q`/`ESC` quit.

### CLI

```bash
.venv/bin/python -m idphoto IMAGE \
  [--size one-inch|two-inch] [--pixels WxH] \
  [--bg white|blue|red] [--rgb R,G,B|#RRGGBB] \
  [--detector auto|yunet|haar] [-o PATH] [--quality 1-100]
```

Writes `outputs/{stem}_{size}_{bg}.jpg` by default and prints face box, crop box, and export summary. `--size` and `--pixels` are mutually exclusive; so are `--bg` and `--rgb`.

### Tests and acceptance

```bash
.venv/bin/python -m compileall -q idphoto tests
.venv/bin/python -m unittest discover -s tests -v
```

29 `unittest` tests (not pytest). The real-portrait integration test skips automatically when `assets/test/private/portrait_source.png` or either model is absent.

Acceptance check with the bundled demo:

```bash
.venv/bin/python -m idphoto docs/images/demo_source.jpg --size one-inch --bg white -o /tmp/idphoto_white.jpg
.venv/bin/python -m idphoto docs/images/demo_source.jpg --size two-inch --bg red -o /tmp/idphoto_red.jpg
.venv/bin/python - <<'PY'
from PIL import Image
cases = [("/tmp/idphoto_white.jpg", (295, 413)), ("/tmp/idphoto_red.jpg", (413, 579))]
for path, expected in cases:
    with Image.open(path) as im:
        assert im.size == expected, (path, im.size)
        dpi = tuple(round(v) for v in im.info.get("dpi", (0, 0)))
        assert dpi == (300, 300), (path, dpi)
print("acceptance OK")
PY
```

YuNet should report a face score of about 0.9 on the demo. If no display is available, skip the GUI and validate via CLI.

## Architecture map

| File | Responsibility |
| --- | --- |
| `idphoto/config.py` | `IDPhotoSize`, `SIZES`, crop factors, `BACKGROUND_COLORS`, `parse_rgb` |
| `idphoto/io_utils.py` | EXIF-corrected read, alpha over white, RGB→BGR; JPEG/PNG write with DPI |
| `idphoto/detector.py` | `Face`, YuNet via `cv2.FaceDetectorYN`, Haar fallback |
| `idphoto/compose.py` | `CropBox`, `compute_crop_box`, premultiplied-alpha `render_crop` |
| `idphoto/segmenter.py` | Lazy rembg U²-Net session, RGBA cutout |
| `idphoto/pipeline.py` | `PhotoResult`; detect → cutout → render orchestration |
| `idphoto/cli.py` | Argument parsing and CLI output |
| `idphoto/app.py` | OpenCV desktop editor, camera, drag, save |

Data contracts: `read_image` returns BGR; the detector expects BGR; `cutout` returns RGBA; background colors are RGB; `render_crop` returns RGB `uint8`.

Fixed specs: one-inch 295×413, two-inch 413×579, both 300 DPI. Crop math: height `2.0 × face.height`, width `height × target_aspect`, top `face.y − 0.45 × face.height`, horizontally centered on the face, then clamped and proportionally shrunk to the image bounds.

## Invariants and gotchas

- Install rembg with `--no-deps`. It declares `opencv-python-headless`, which replaces the GUI build and breaks `cv2.imshow`; never install the headless package into this venv.
- YuNet must come from `media.githubusercontent.com/media/opencv/opencv_zoo/...`; the raw GitHub URL returns an LFS pointer. `setup.sh` rejects files under 200 KB.
- U²-Net lives at `models/rembg/u2net.onnx`; set `U2NET_HOME` to that directory before importing rembg. `setup.sh` verifies SHA-256 `8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491`.
- `segmenter.py` sets `NUMBA_THREADING_LAYER=workqueue` before importing rembg to avoid the TBB interface warning; keep it.
- `render_crop` must keep `np.rint` before `uint8` and premultiplied alpha; truncation causes dark halos and breaks compositing tests.
- Still mode redraws only when `_dirty` is set. Do not force continuous redraws; idle CPU was reduced from ~90% to ~1%. Camera preview still costs roughly 70% of one core because Qt5 uses software rendering here.
- Auto detector falls back to Haar only when the YuNet file is missing; a missing U²-Net model is a hard `RuntimeError`.
- Models run locally; no image leaves the machine during processing.
- Known limits: one face per image, segmentation artifacts on busy backgrounds or fine hair, and output sizes are common standards rather than a legal guarantee.

## Repo rules

- Use Conventional Commits with one logical change per commit, e.g. `feat(ui): add camera capture`, `fix(compose): avoid alpha halo`, `docs(readme): document CLI`.
- Never push unless the user explicitly asks. Never modify git config (no `git config user.*`).
- If the identity is unset, pass it per command:
  `git -c user.name=elac1 -c user.email=272543901+elac1@users.noreply.github.com commit -m "..."`
- Keep `.venv/`, `models/rembg/`, `outputs/`, `assets/test/private/`, and `__pycache__/` out of commits. YuNet is small and tracked; U²-Net is not.
- State side effects in reports: windows opened, camera activated, downloads, files written.
