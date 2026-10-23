---
name: idphoto
description: Operate and maintain the local ID-photo auto-crop and background-replacement app. Use when setting up the Python environment or models, running the Tkinter/OpenCV desktop editor (including file drag-and-drop) or CLI, debugging YuNet face detection, crop framing, rembg U2Net cutout or alpha compositing, validating output sizes and DPI, running tests, or preparing Conventional Commits in this repo.
---

# ID Photo Studio

Local-only ID photo tool: YuNet detects the largest face, a 2×face-height crop is composed to a 5:7 frame, rembg U²-Net cuts out the portrait, and premultiplied-alpha compositing replaces the background.

## Commands

Run from the repo root. `.venv/bin/python` is Python 3.10.

### Setup

```bash
./scripts/setup.sh
```

Side effects: creates `.venv/` if missing, installs pinned dependencies (including `tkinterdnd2`), downloads YuNet (~232 KB) and U²-Net (~176 MB) into `models/` (gitignored). No sudo, but the system must provide Tk (`python3-tk` on Ubuntu). First run needs network; later runs are offline.

### Desktop editor

```bash
./scripts/run.sh [IMAGE] [--camera N] [--no-camera]
```

The editor is a Tkinter Canvas + `tkinterdnd2` shell; all frames are still rendered by OpenCV. Side effects: opens a 1280×800 window and blocks until quit. Without `IMAGE` and without `--no-camera`, it opens camera 0 (camera indicator turns on). `S` writes to `outputs/` (gitignored); camera capture is mirrored.

Drop a JPG/PNG/BMP/WEBP file anywhere on the window to load it automatically; dropping while an image or camera is active replaces it. Controls: `O` open, `C` camera, `SPACE` capture, `ESC` cancel capture, `S` save, `A` auto-frame, `1`/`2` one/two-inch, `W`/`B`/`R` white/blue/red, drag inside to move the crop, drag a corner to resize at the fixed output ratio, `Q`/`ESC` quit.

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

YuNet should report a face score of about 0.9 on the demo. If no display is available, skip the GUI and validate via CLI. GUI changes should also be smoke-tested for: initial image launch, immediate shortcuts before any click, real file drop, crop drag, camera open/cancel, save, and quit.

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
| `idphoto/app.py` | Tkinter Canvas shell, DnD, camera loop, crop mouse events, save |

Data contracts: `read_image` returns BGR; the detector expects BGR; `cutout` returns RGBA; background colors are RGB; `render_crop` returns RGB `uint8`.

Fixed specs: one-inch 295×413, two-inch 413×579, both 300 DPI. Crop math: height `2.0 × face.height`, width `height × target_aspect`, top `face.y − 0.45 × face.height`, horizontally centered on the face, then clamped and proportionally shrunk to the image bounds.

## Invariants and gotchas

- Install rembg with `--no-deps`. It declares `opencv-python-headless`, which replaces the GUI build and breaks `cv2.imshow`; never install the headless package into this venv.
- YuNet must come from `media.githubusercontent.com/media/opencv/opencv_zoo/...`; the raw GitHub URL returns an LFS pointer. `setup.sh` rejects files under 200 KB.
- The segmentation model is chosen by `SEGMENT_MODEL` in `config.py` (default `isnet-general-use`) and must sit at `models/rembg/<SEGMENT_MODEL>.onnx`; set `U2NET_HOME` to that directory before importing rembg. `setup.sh` reads the name from `config.py` and verifies SHA-256 (`isnet-general-use`: `60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a`, matching rembg's official md5 `fc16ebd8b0c10d971d3513d564d01e29`; `u2net`: `8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491`).
- rembg ≥2.0.7x moved its cache to a nested `<U2NET_HOME>/models/<name>/<name>.onnx` layout, while this project (pinned `rembg==2.0.69`) expects the flat `<U2NET_HOME>/<name>.onnx`. If rembg is ever upgraded, `PortraitSegmenter._load_session` will stop finding models — re-check the layout before bumping the pin.
- YuNet must not be fed full-resolution large images: confidence collapses (measured 0.654 on a 1760×2464 photo, below the 0.85 threshold, plus one spurious face) and it rose to 0.924 once the long side was capped at `DETECT_MAX_SIDE` (1080). Keep the resize + coordinate scale-back in `_detect_yunet`.
- `segmenter.py` sets `NUMBA_THREADING_LAYER=workqueue` before importing rembg to avoid the TBB interface warning; keep it.
- `render_crop` must keep `np.rint` before `uint8` and premultiplied alpha; truncation causes dark halos and breaks compositing tests.
- Never reintroduce `cv2.imshow` / `cv2.waitKey`; the Tk event loop owns input and display. Frames remain BGR arrays rendered by `_build_*_frame`, then shown through `_show_canvas` (RGB → PIL → `PhotoImage.paste`).
- Keyboard events are bound to the Canvas. Call `canvas.focus_force()` after mapping, not only `root.focus_force()`, or shortcuts do nothing until the first mouse click.
- Keep `canvas.drop_target_register(DND_FILES)` and the four `dnd_bind` handlers. A real OS drag is required to validate the payload: synthetic `event_generate("<<Drop>>")` yields `??` fields. For automation, run the app and a tkdnd drag source inside a nested Xephyr display and drive the pointer with XTEST.
- `_busy` blocks reentrant keys and drops while YuNet/rembg run synchronously. `_processing_frame` flushes with `update_idletasks`, not `update`, so a scheduled `_tick` cannot overwrite the progress frame.
- Dropping a file while the camera is live must release the camera first; keep that behavior.
- Still mode redraws only when `_dirty` is set. Do not force continuous redraws; idle CPU is ~0.7%. Camera preview is heavier because every frame is resized, converted, and pasted into a Tk `PhotoImage`.
- Auto detector falls back to Haar only when the YuNet file is missing; a missing segmentation model raises `FileNotFoundError`.
- Models run locally; no image leaves the machine during processing.
- Known limits: one face per image, segmentation artifacts on busy backgrounds or fine hair, output sizes are common standards rather than a legal guarantee, and drag-and-drop relies on X11/Xdnd (Wayland uses XWayland).

## Repo rules

- Use Conventional Commits with one logical change per commit, e.g. `feat(ui): add camera capture`, `fix(compose): avoid alpha halo`, `docs(readme): document CLI`.
- Never push unless the user explicitly asks. Never modify git config (no `git config user.*`).
- If the identity is unset, pass it per command:
  `git -c user.name=elac1 -c user.email=272543901+elac1@users.noreply.github.com commit -m "..."`
- Keep `.venv/`, `models/rembg/`, `outputs/`, `assets/test/private/`, and `__pycache__/` out of commits. YuNet is small and tracked; U²-Net is not.
- State side effects in reports: windows opened, camera activated, downloads, files written.
