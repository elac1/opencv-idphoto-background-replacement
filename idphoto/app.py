from __future__ import annotations

import argparse
import math
import sys
import time
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import filedialog

import cv2
import numpy as np
from PIL import Image, ImageTk
from tkinterdnd2 import DND_FILES, TkinterDnD

from .compose import CropBox, compute_crop_box, render_crop
from .config import (
    BACKGROUND_COLORS,
    ONE_INCH,
    PROJECT_ROOT,
    TWO_INCH,
    IDPhotoSize,
)
from .detector import Face, FaceNotFoundError
from .io_utils import ImageLoadError, read_image, write_photo
from .pipeline import IDPhotoProcessor

WINDOW_NAME = "ID Photo Studio"
CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 800
SOURCE_BOX = (20, 70, 900, 760)
PANEL_BOX = (920, 70, 1260, 760)
PREVIEW_MAX = (280, 390)
FONT = cv2.FONT_HERSHEY_SIMPLEX
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CAMERA_FPS = 15

COLOR_BG = (22, 22, 26)
COLOR_HEADER = (30, 30, 36)
COLOR_SOURCE_BG = (16, 16, 20)
COLOR_PANEL_BG = (30, 30, 36)
COLOR_TEXT = (238, 238, 242)
COLOR_MUTED = (148, 148, 158)
COLOR_BORDER = (70, 70, 80)
COLOR_ACCENT = (120, 220, 130)
COLOR_ERROR = (90, 90, 245)


@dataclass
class _Drag:
    mode: str
    corner: str | None
    mouse_x: float
    mouse_y: float
    crop: CropBox
    grab_dx: float = 0.0
    grab_dy: float = 0.0


class IDPhotoApp:
    def __init__(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index
        self.processor = IDPhotoProcessor()

        self.image: np.ndarray | None = None
        self.source_label = ""
        self.face: Face | None = None
        self.cutout: np.ndarray | None = None
        self.crop_box: CropBox | None = None
        self.size: IDPhotoSize = ONE_INCH
        self.bg_name = "white"
        self.background = BACKGROUND_COLORS["white"]

        self._preview: np.ndarray | None = None
        self._preview_key = None
        self._display_cache: np.ndarray | None = None
        self._shaded_cache: np.ndarray | None = None
        self._display_rect: tuple[int, int, int, int, float] | None = None
        self._drag: _Drag | None = None
        self._base_cache: np.ndarray | None = None

        self.camera: cv2.VideoCapture | None = None
        self.camera_active = False
        self.camera_frame: np.ndarray | None = None
        self._last_camera_draw = 0.0

        self.status = "Ready"
        self.status_error = False
        self.running = True
        self._dirty = True

        self.root: TkinterDnD.Tk | None = None
        self.canvas: tk.Canvas | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._image_item: int | None = None
        self._drop_hover = False
        self._busy = False

    def run(self, initial_image: Path | None = None, auto_camera: bool = True) -> int:
        try:
            self.root = TkinterDnD.Tk()
        except tk.TclError as exc:
            print(f"无法打开桌面窗口: {exc}", file=sys.stderr)
            return 1
        self.root.title(WINDOW_NAME)
        self.root.resizable(False, False)
        self.canvas = tk.Canvas(
            self.root,
            width=CANVAS_WIDTH,
            height=CANVAS_HEIGHT,
            highlightthickness=0,
            bd=0,
            bg="#1a1616",
        )
        self.canvas.pack()
        self.root.geometry(f"{CANVAS_WIDTH}x{CANVAS_HEIGHT}+40+40")
        self._image_item = self.canvas.create_image(0, 0, anchor="nw")
        self._bind_events()
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

        self._set_status("Drop a photo to start")
        self._show_canvas(self._build_still_frame())
        self.root.update_idletasks()
        self.root.lift()
        try:
            self.root.focus_force()
            self.canvas.focus_force()
        except tk.TclError:
            pass

        try:
            if initial_image is not None:
                self.load_path(initial_image)
            elif auto_camera:
                self.start_camera()
            if not self.running:
                return 0
            self._tick()
            if self.running:
                self.root.mainloop()
        finally:
            self._release_camera()
            if self.root is not None:
                try:
                    self.root.destroy()
                except tk.TclError:
                    pass
                self.root = None
                self.canvas = None
        return 0

    def _bind_events(self) -> None:
        assert self.canvas is not None
        self.canvas.bind(
            "<ButtonPress-1>",
            lambda event: self._on_tk_mouse(cv2.EVENT_LBUTTONDOWN, event),
        )
        self.canvas.bind(
            "<B1-Motion>",
            lambda event: self._on_tk_mouse(cv2.EVENT_MOUSEMOVE, event),
        )
        self.canvas.bind(
            "<ButtonRelease-1>",
            lambda event: self._on_tk_mouse(cv2.EVENT_LBUTTONUP, event),
        )
        self.canvas.bind("<KeyPress>", self._on_key_event)
        self.canvas.drop_target_register(DND_FILES)
        self.canvas.dnd_bind("<<DropEnter>>", self._on_drop_enter)
        self.canvas.dnd_bind("<<DropPosition>>", self._on_drop_position)
        self.canvas.dnd_bind("<<DropLeave>>", self._on_drop_leave)
        self.canvas.dnd_bind("<<Drop>>", self._on_drop)
        self.canvas.focus_set()

    def _tick(self) -> None:
        if not self.running or self.root is None:
            return
        if self.camera_active:
            frame = self._read_camera_frame()
            now = time.monotonic()
            if frame is not None and now - self._last_camera_draw >= 1 / CAMERA_FPS:
                self._show_canvas(self._build_camera_frame(frame))
                self._last_camera_draw = now
                self._dirty = False
        elif self._dirty:
            self._show_canvas(self._build_still_frame())
            self._dirty = False
        self.root.after(30, self._tick)

    def _show_canvas(self, bgr: np.ndarray) -> None:
        if self.canvas is None or self._image_item is None:
            return
        image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if self._photo is None:
            self._photo = ImageTk.PhotoImage(image)
        else:
            self._photo.paste(image)
        self.canvas.itemconfigure(self._image_item, image=self._photo)

    def _quit(self) -> None:
        self.running = False
        if self.root is not None:
            try:
                self.root.destroy()
            except tk.TclError:
                pass

    def _on_key_event(self, event: tk.Event) -> None:
        if self._busy or event.state & (0x4 | 0x8):
            return
        keysym = event.keysym
        if keysym == "Escape":
            key = 27
        elif keysym == "space":
            key = 32
        elif len(keysym) == 1 and keysym.isalnum():
            key = ord(keysym.lower())
        else:
            return
        self._handle_key(key)

    def _on_tk_mouse(self, cv_event: int, event: tk.Event) -> None:
        if self._busy:
            return
        if self.canvas is not None:
            self.canvas.focus_set()
        self._on_mouse(cv_event, event.x, event.y, 0, None)

    def _on_drop_enter(self, event) -> str | None:
        if not self._busy:
            self._drop_hover = True
            self._dirty = True
        return getattr(event, "action", None)

    def _on_drop_position(self, event) -> str | None:
        return getattr(event, "action", None)

    def _on_drop_leave(self, event) -> None:
        if self._drop_hover:
            self._drop_hover = False
            self._dirty = True

    def _on_drop(self, event) -> str | None:
        self._drop_hover = False
        self._dirty = True
        if self._busy or self.root is None:
            return None
        try:
            paths = self.root.tk.splitlist(event.data)
        except tk.TclError:
            paths = [event.data]
        if not paths:
            return None
        path = Path(paths[0])
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            self._set_status("Unsupported file - use JPG/PNG/WEBP", error=True)
            return None
        if self.camera_active:
            self._release_camera()
        if self.canvas is not None:
            self.canvas.focus_set()
        self.load_path(path)
        return getattr(event, "action", None)

    def load_path(self, path: Path) -> None:
        try:
            image = read_image(path)
        except Exception as exc:
            self._fail("Could not read image", exc)
            return
        self._load_image(image, path.name)

    def start_camera(self) -> None:
        if self.camera_active:
            return
        capture = cv2.VideoCapture(self.camera_index)
        if not capture.isOpened():
            capture.release()
            self._set_status(
                f"Camera {self.camera_index} unavailable - press O to open a photo",
                error=True,
            )
            print(f"无法打开摄像头 {self.camera_index}", file=sys.stderr)
            return
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        ok, frame = capture.read()
        if not ok:
            capture.release()
            self._set_status("Camera opened but returned no frame", error=True)
            return
        self.camera = capture
        self.camera_active = True
        self.camera_frame = cv2.flip(frame, 1)
        self._last_camera_draw = 0.0
        self._set_status("Camera live - SPACE to capture, ESC to cancel")

    def save(self) -> None:
        preview = self._get_preview()
        if preview is None:
            self._set_status("No photo to save", error=True)
            return
        filename = (
            f"idphoto_{datetime.now():%Y%m%d_%H%M%S}_"
            f"{self.size.key}_{self.bg_name}.jpg"
        )
        path = PROJECT_ROOT / "outputs" / filename
        try:
            write_photo(preview, path, dpi=self.size.dpi)
        except Exception as exc:
            self._fail("Save failed", exc)
            return
        print(f"已保存: {path}")
        self._set_status(f"Saved outputs/{filename}")

    def auto_frame(self) -> None:
        if self.image is None or self.face is None:
            self._set_status("No face available for auto framing", error=True)
            return
        self.crop_box = compute_crop_box(
            self.face, self.image.shape[1], self.image.shape[0], self.size
        )
        self._invalidate_preview()
        self._set_status("Auto framing restored")

    def set_size(self, size: IDPhotoSize) -> None:
        self.size = size
        if self.image is not None and self.face is not None:
            self.crop_box = compute_crop_box(
                self.face, self.image.shape[1], self.image.shape[0], size
            )
        self._invalidate_preview()
        self._set_status(f"Size: {size.key} ({size.width_px}x{size.height_px})")

    def set_background(self, name: str) -> None:
        self.bg_name = name
        self.background = BACKGROUND_COLORS[name]
        self._invalidate_preview()
        self._set_status(f"Background: {name}")

    def _load_image(
        self,
        image: np.ndarray,
        label: str,
    ) -> None:
        if self._busy:
            return
        self._busy = True
        try:
            self._set_status("Detecting face...")
            self._processing_frame("Detecting face...")
            try:
                face = self.processor.detect(image)
                self._set_status("Removing background...")
                self._processing_frame("Removing background...")
                cutout = self.processor.cutout(image)
            except Exception as exc:
                self._fail(self._error_text(exc), exc)
                return
            if not self.running:
                return

            self.image = image
            self.source_label = label
            self.face = face
            self.cutout = cutout
            self.crop_box = compute_crop_box(
                face, image.shape[1], image.shape[0], self.size
            )
            self._display_cache = None
            self._shaded_cache = None
            self._display_rect = None
            self._drag = None
            self._invalidate_preview()
            self._set_status("Ready - drag crop or drop another photo")
        finally:
            self._busy = False

    def _error_text(self, exc: Exception) -> str:
        if isinstance(exc, FaceNotFoundError):
            return "No face detected - try another photo"
        if isinstance(exc, FileNotFoundError):
            return "Model missing - run ./scripts/setup.sh"
        if isinstance(exc, ImageLoadError):
            return "Could not read image"
        return "Processing failed - see terminal"

    def _fail(self, message: str, exc: Exception) -> None:
        print(f"{message}: {exc}", file=sys.stderr)
        self._set_status(message, error=True)

    def _set_status(self, text: str, error: bool = False) -> None:
        self.status = text
        self.status_error = error
        self._dirty = True
        self._base_cache = None

    def _invalidate_preview(self) -> None:
        self._preview = None
        self._preview_key = None
        self._dirty = True
        self._base_cache = None

    def _get_preview(self) -> np.ndarray | None:
        if self.cutout is None or self.crop_box is None:
            return None
        crop = self.crop_box
        key = (
            crop.x,
            crop.y,
            crop.width,
            crop.height,
            self.size,
            self.background,
        )
        if self._preview is None or self._preview_key != key:
            self._preview = render_crop(
                self.cutout, crop, self.size.pixels, self.background
            )
            self._preview_key = key
        return self._preview

    def _read_camera_frame(self) -> np.ndarray | None:
        if self.camera is None:
            return None
        ok, frame = self.camera.read()
        if not ok:
            self._release_camera()
            self._set_status("Camera read failed", error=True)
            return None
        frame = cv2.flip(frame, 1)
        self.camera_frame = frame
        return frame

    def _release_camera(self) -> None:
        if self.camera is not None:
            self.camera.release()
        self.camera = None
        self.camera_active = False
        self.camera_frame = None
        self._dirty = True
        self._base_cache = None

    def _capture_camera(self) -> None:
        frame = self.camera_frame
        if frame is None:
            self._set_status("No camera frame to capture", error=True)
            return
        self._release_camera()
        self._load_image(frame.copy(), "camera capture")

    def _open_dialog(self) -> None:
        if self.root is None:
            return
        try:
            selected = filedialog.askopenfilename(
                parent=self.root,
                title="Open a portrait photo",
                filetypes=[
                    ("Images", "*.jpg *.jpeg *.png *.bmp *.webp"),
                    ("All files", "*.*"),
                ],
            )
        except tk.TclError as exc:
            self._fail("File dialog failed", exc)
            return
        if selected:
            self.load_path(Path(selected))
        if self.canvas is not None:
            self.canvas.focus_set()

    def _handle_key(self, key: int) -> None:
        if 65 <= key <= 90:
            key += 32
        if key == 27:
            if self.camera_active:
                self._release_camera()
                self._set_status("Camera cancelled")
            else:
                self._quit()
            return
        if key == ord("q"):
            self._quit()
            return
        if self.camera_active:
            if key == 32:
                self._capture_camera()
            return
        if key == ord("o"):
            self._open_dialog()
        elif key == ord("c"):
            self.start_camera()
        elif key == ord("s"):
            self.save()
        elif key == ord("a"):
            self.auto_frame()
        elif key == ord("1"):
            self.set_size(ONE_INCH)
        elif key == ord("2"):
            self.set_size(TWO_INCH)
        elif key == ord("w"):
            self.set_background("white")
        elif key == ord("b"):
            self.set_background("blue")
        elif key == ord("r"):
            self.set_background("red")

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        if self.camera_active or self.image is None or self.crop_box is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            hit = self._hit_test(x, y)
            point = self._display_to_image(x, y)
            if hit is None or point is None:
                return
            mode, corner = hit
            mouse_x, mouse_y = point
            drag = _Drag(
                mode=mode,
                corner=corner,
                mouse_x=mouse_x,
                mouse_y=mouse_y,
                crop=self.crop_box,
            )
            if mode == "move":
                drag.grab_dx = self.crop_box.x - mouse_x
                drag.grab_dy = self.crop_box.y - mouse_y
            self._drag = drag
        elif event == cv2.EVENT_MOUSEMOVE and self._drag is not None:
            point = self._display_to_image(x, y)
            if point is not None:
                self._apply_drag(*point)
        elif event == cv2.EVENT_LBUTTONUP:
            self._drag = None

    def _hit_test(self, x: int, y: int) -> tuple[str, str | None] | None:
        if self._display_rect is None:
            return None
        dx, dy, width, height, scale = self._display_rect
        if not (
            dx - 12 <= x <= dx + width + 12
            and dy - 12 <= y <= dy + height + 12
        ):
            return None
        point = self._display_to_image(x, y)
        if point is None or self.crop_box is None:
            return None
        mouse_x, mouse_y = point
        crop = self.crop_box
        corners = {
            "tl": (crop.x, crop.y),
            "tr": (crop.right, crop.y),
            "bl": (crop.x, crop.bottom),
            "br": (crop.right, crop.bottom),
        }
        nearest = min(
            corners,
            key=lambda name: math.hypot(
                mouse_x - corners[name][0], mouse_y - corners[name][1]
            ),
        )
        corner_x, corner_y = corners[nearest]
        if math.hypot(mouse_x - corner_x, mouse_y - corner_y) <= 16.0 / scale:
            return "resize", nearest
        if crop.x <= mouse_x <= crop.right and crop.y <= mouse_y <= crop.bottom:
            return "move", None
        return None

    def _display_to_image(self, x: int, y: int) -> tuple[float, float] | None:
        if self._display_rect is None or self.image is None:
            return None
        dx, dy, _, _, scale = self._display_rect
        image_height, image_width = self.image.shape[:2]
        image_x = (x - dx) / scale
        image_y = (y - dy) / scale
        image_x = min(max(image_x, 0.0), float(image_width - 1))
        image_y = min(max(image_y, 0.0), float(image_height - 1))
        return image_x, image_y

    def _apply_drag(self, mouse_x: float, mouse_y: float) -> None:
        if self._drag is None or self.image is None:
            return
        drag = self._drag
        crop = drag.crop
        image_height, image_width = self.image.shape[:2]

        if drag.mode == "move":
            x = mouse_x + drag.grab_dx
            y = mouse_y + drag.grab_dy
            x = min(max(x, 0.0), image_width - crop.width)
            y = min(max(y, 0.0), image_height - crop.height)
            self.crop_box = CropBox(x, y, crop.width, crop.height)
        else:
            corner = drag.corner or "br"
            aspect = self.size.aspect_ratio
            anchor_x = crop.right if corner in ("tl", "bl") else crop.x
            anchor_y = crop.bottom if corner in ("tl", "tr") else crop.y
            max_width = anchor_x if corner in ("tl", "bl") else image_width - anchor_x
            max_height = anchor_y if corner in ("tl", "tr") else image_height - anchor_y
            max_width = min(max_width, max_height * aspect)
            width = max(abs(mouse_x - anchor_x), abs(mouse_y - anchor_y) * aspect)
            width = min(max(width, min(80.0, max_width)), max_width)
            height = width / aspect
            x = anchor_x - width if corner in ("tl", "bl") else anchor_x
            y = anchor_y - height if corner in ("tl", "tr") else anchor_y
            self.crop_box = CropBox(x, y, width, height)

        self._invalidate_preview()

    def _get_source_display(self) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int, float]]:
        if (
            self._display_cache is None
            or self._shaded_cache is None
            or self._display_rect is None
        ):
            rect = self._fit_rect(self.image.shape[1], self.image.shape[0], SOURCE_BOX)
            dx, dy, width, height, scale = rect
            interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            display = cv2.resize(
                self.image, (width, height), interpolation=interpolation
            )
            shaded = np.clip(display.astype(np.float32) * 0.32, 0, 255).astype(np.uint8)
            self._display_cache = display
            self._shaded_cache = shaded
            self._display_rect = rect
        return self._display_cache, self._shaded_cache, self._display_rect

    @staticmethod
    def _fit_rect(
        width: int,
        height: int,
        box: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int, float]:
        x0, y0, x1, y1 = box
        max_width = x1 - x0
        max_height = y1 - y0
        scale = min(max_width / width, max_height / height)
        draw_width = max(1, int(round(width * scale)))
        draw_height = max(1, int(round(height * scale)))
        x = x0 + (max_width - draw_width) // 2
        y = y0 + (max_height - draw_height) // 2
        return x, y, draw_width, draw_height, scale

    def _build_still_frame(self) -> np.ndarray:
        canvas = self._get_base_canvas().copy()
        if self.image is None or self.cutout is None or self.crop_box is None:
            self._draw_empty_source(canvas)
        else:
            self._draw_still_source(canvas)
            if self._drop_hover:
                self._draw_drop_overlay(canvas)
        return canvas

    def _build_camera_frame(self, frame: np.ndarray) -> np.ndarray:
        canvas = self._get_base_canvas().copy()
        self._draw_camera_source(canvas, frame)
        if self._drop_hover:
            self._draw_drop_overlay(canvas)
        return canvas

    def _draw_drop_overlay(self, canvas: np.ndarray) -> None:
        cv2.rectangle(
            canvas, SOURCE_BOX[:2], SOURCE_BOX[2:], COLOR_ACCENT, 3, cv2.LINE_AA
        )
        self._put_centered(
            canvas,
            "Release to load photo",
            (SOURCE_BOX[0] + SOURCE_BOX[2]) // 2,
            SOURCE_BOX[3] - 26,
            0.6,
            COLOR_ACCENT,
            2,
            shadow=True,
        )

    def _get_base_canvas(self) -> np.ndarray:
        if self._base_cache is None:
            self._base_cache = self._build_base()
        return self._base_cache

    def _build_base(self) -> np.ndarray:
        canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), COLOR_BG, dtype=np.uint8)
        cv2.rectangle(canvas, (0, 0), (CANVAS_WIDTH, 60), COLOR_HEADER, -1)
        cv2.line(canvas, (0, 60), (CANVAS_WIDTH, 60), COLOR_BORDER, 1, cv2.LINE_AA)
        cv2.rectangle(canvas, SOURCE_BOX[:2], SOURCE_BOX[2:], COLOR_SOURCE_BG, -1)
        cv2.rectangle(
            canvas, PANEL_BOX[:2], PANEL_BOX[2:], COLOR_PANEL_BG, -1, cv2.LINE_AA
        )
        cv2.putText(
            canvas, "ID PHOTO STUDIO", (20, 40), FONT, 0.82, COLOR_TEXT, 2, cv2.LINE_AA
        )
        cv2.putText(
            canvas, "YuNet + U2Net", (278, 39), FONT, 0.48, COLOR_MUTED, 1, cv2.LINE_AA
        )

        status_color = COLOR_ERROR if self.status_error else COLOR_MUTED
        status = self._fit_text(self.status, 520, 0.52)
        status_width = cv2.getTextSize(status, FONT, 0.52, 1)[0][0]
        cv2.putText(
            canvas,
            status,
            (CANVAS_WIDTH - 20 - status_width, 39),
            FONT,
            0.52,
            status_color,
            1,
            cv2.LINE_AA,
        )

        if self.camera_active:
            hint = "SPACE  Capture     ESC  Cancel     Q  Quit"
        else:
            hint = (
                "Drop photo anywhere | Drag crop: move | corners: resize    "
                "O Open   C Camera   S Save   A Auto   1/2 Size   W/B/R BG   Q Quit"
            )
        cv2.putText(canvas, hint, (20, 786), FONT, 0.46, COLOR_MUTED, 1, cv2.LINE_AA)
        self._draw_panel(canvas)
        return canvas

    def _draw_empty_source(self, canvas: np.ndarray) -> None:
        x0 = SOURCE_BOX[0] + 70
        y0 = SOURCE_BOX[1] + 70
        x1 = SOURCE_BOX[2] - 70
        y1 = SOURCE_BOX[3] - 70
        color = COLOR_ACCENT if self._drop_hover else COLOR_BORDER
        if self._drop_hover:
            cv2.rectangle(canvas, (x0 + 1, y0 + 1), (x1 - 1, y1 - 1), (26, 34, 26), -1)
        self._draw_dashed_rect(canvas, (x0, y0), (x1, y1), color, 2)

        center_x = (SOURCE_BOX[0] + SOURCE_BOX[2]) // 2
        center_y = (SOURCE_BOX[1] + SOURCE_BOX[3]) // 2
        icon_width, icon_height = 84, 60
        icon_x = center_x - icon_width // 2
        icon_y = center_y - 132
        cv2.rectangle(
            canvas,
            (icon_x, icon_y),
            (icon_x + icon_width, icon_y + icon_height),
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.circle(
            canvas, (icon_x + icon_width - 20, icon_y + 18), 7, color, 2, cv2.LINE_AA
        )
        cv2.line(
            canvas,
            (icon_x + 8, icon_y + icon_height - 12),
            (icon_x + 32, icon_y + 34),
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.line(
            canvas,
            (icon_x + 32, icon_y + 34),
            (icon_x + icon_width - 8, icon_y + icon_height - 12),
            color,
            2,
            cv2.LINE_AA,
        )
        title = "Drop to load the photo" if self._drop_hover else "Drop a photo here"
        self._put_centered(
            canvas, title, center_x, center_y - 20, 0.82, color, 2
        )
        self._put_centered(
            canvas,
            "JPG / PNG / BMP / WEBP",
            center_x,
            center_y + 18,
            0.5,
            COLOR_MUTED,
        )
        self._put_centered(
            canvas,
            "or press O to open, C to use the camera",
            center_x,
            center_y + 54,
            0.5,
            COLOR_MUTED,
        )

    def _draw_still_source(self, canvas: np.ndarray) -> None:
        display, shaded, rect = self._get_source_display()
        dx, dy, width, height, scale = rect
        canvas[dy:dy + height, dx:dx + width] = shaded

        crop = self.crop_box
        x0 = max(dx, min(dx + width, int(round(dx + crop.x * scale))))
        y0 = max(dy, min(dy + height, int(round(dy + crop.y * scale))))
        x1 = max(dx, min(dx + width, int(round(dx + crop.right * scale))))
        y1 = max(dy, min(dy + height, int(round(dy + crop.bottom * scale))))
        if x1 <= x0 or y1 <= y0:
            return

        canvas[y0:y1, x0:x1] = display[
            y0 - dy:y1 - dy, x0 - dx:x1 - dx
        ]
        cv2.rectangle(canvas, (x0, y0), (x1, y1), COLOR_ACCENT, 2, cv2.LINE_AA)
        for index in (1, 2):
            guide_x = x0 + (x1 - x0) * index // 3
            guide_y = y0 + (y1 - y0) * index // 3
            cv2.line(
                canvas, (guide_x, y0), (guide_x, y1), (185, 185, 190), 1, cv2.LINE_AA
            )
            cv2.line(
                canvas, (x0, guide_y), (x1, guide_y), (185, 185, 190), 1, cv2.LINE_AA
            )
        for handle_x, handle_y in (
            (x0, y0), (x1, y0), (x0, y1), (x1, y1)
        ):
            cv2.circle(canvas, (handle_x, handle_y), 7, (24, 24, 28), -1, cv2.LINE_AA)
            cv2.circle(canvas, (handle_x, handle_y), 5, COLOR_ACCENT, -1, cv2.LINE_AA)

    def _draw_camera_source(self, canvas: np.ndarray, frame: np.ndarray) -> None:
        dx, dy, width, height, scale = self._fit_rect(
            frame.shape[1], frame.shape[0], SOURCE_BOX
        )
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        display = cv2.resize(frame, (width, height), interpolation=interpolation)
        canvas[dy:dy + height, dx:dx + width] = display

        guide_height = int(height * 0.86)
        guide_width = int(guide_height * 5 / 7)
        if guide_width <= width:
            guide_x = dx + (width - guide_width) // 2
            guide_y = dy + (height - guide_height) // 2
            cv2.rectangle(
                canvas,
                (guide_x, guide_y),
                (guide_x + guide_width, guide_y + guide_height),
                (210, 210, 210),
                1,
                cv2.LINE_AA,
            )

        cv2.circle(canvas, (dx + 24, dy + 24), 6, (60, 60, 235), -1, cv2.LINE_AA)
        cv2.putText(
            canvas, "LIVE", (dx + 38, dy + 31), FONT, 0.5, (230, 230, 235), 2, cv2.LINE_AA
        )
        self._put_centered(
            canvas,
            "SPACE capture    ESC cancel",
            dx + width // 2,
            dy + height - 18,
            0.55,
            (240, 240, 245),
            2,
            shadow=True,
        )

    def _draw_panel(self, canvas: np.ndarray) -> None:
        cv2.putText(
            canvas, "OUTPUT", (PANEL_BOX[0] + 20, 98), FONT, 0.52, COLOR_MUTED, 1, cv2.LINE_AA
        )
        area_y0 = 112
        preview = self._get_preview()
        panel_width = PANEL_BOX[2] - PANEL_BOX[0]
        if preview is not None:
            preview_height, preview_width = preview.shape[:2]
            scale = min(
                PREVIEW_MAX[0] / preview_width, PREVIEW_MAX[1] / preview_height
            )
            draw_width = max(1, int(round(preview_width * scale)))
            draw_height = max(1, int(round(preview_height * scale)))
            preview_x = PANEL_BOX[0] + (panel_width - draw_width) // 2
            preview_y = area_y0 + (PREVIEW_MAX[1] - draw_height) // 2
            interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            display = cv2.resize(
                preview, (draw_width, draw_height), interpolation=interpolation
            )
            canvas[
                preview_y:preview_y + draw_height,
                preview_x:preview_x + draw_width,
            ] = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
            cv2.rectangle(
                canvas,
                (preview_x - 1, preview_y - 1),
                (preview_x + draw_width, preview_y + draw_height),
                COLOR_BORDER,
                1,
                cv2.LINE_AA,
            )
        else:
            box_width = PREVIEW_MAX[0]
            box_x = PANEL_BOX[0] + (panel_width - box_width) // 2
            cv2.rectangle(
                canvas,
                (box_x, area_y0),
                (box_x + box_width, area_y0 + PREVIEW_MAX[1]),
                COLOR_BORDER,
                1,
                cv2.LINE_AA,
            )
            self._put_centered(
                canvas,
                "NO PREVIEW",
                PANEL_BOX[0] + panel_width // 2,
                area_y0 + PREVIEW_MAX[1] // 2,
                0.55,
                COLOR_MUTED,
            )

        info_y = area_y0 + PREVIEW_MAX[1] + 28
        if self.face is not None and self.crop_box is not None:
            self._put_text(
                canvas,
                f"{self.size.key.upper().replace('-', ' ')}",
                (PANEL_BOX[0] + 20, info_y),
                0.62,
                COLOR_TEXT,
                2,
            )
            self._put_text(
                canvas,
                f"{self.size.width_px} x {self.size.height_px} px  @ {self.size.dpi} DPI",
                (PANEL_BOX[0] + 20, info_y + 25),
                0.46,
                COLOR_MUTED,
            )
            self._put_text(
                canvas,
                f"BG  {self.bg_name.upper()}   RGB{self.background}",
                (PANEL_BOX[0] + 20, info_y + 50),
                0.46,
                COLOR_TEXT,
            )
            self._put_text(
                canvas,
                f"FACE SCORE  {self.face.score:.2f}",
                (PANEL_BOX[0] + 20, info_y + 75),
                0.46,
                COLOR_TEXT,
            )
            self._put_text(
                canvas,
                f"CROP  {self.crop_box.width:.0f} x {self.crop_box.height:.0f} px",
                (PANEL_BOX[0] + 20, info_y + 100),
                0.46,
                COLOR_TEXT,
            )
            label = self.source_label or "unknown"
            label = self._fit_text(label, panel_width - 60, 0.44)
            self._put_text(
                canvas,
                f"FILE  {label}",
                (PANEL_BOX[0] + 20, info_y + 125),
                0.44,
                COLOR_MUTED,
            )
        else:
            self._put_text(
                canvas,
                f"{self.size.key.upper().replace('-', ' ')}  "
                f"{self.size.width_px}x{self.size.height_px}",
                (PANEL_BOX[0] + 20, info_y),
                0.5,
                COLOR_TEXT,
            )
            self._put_text(
                canvas,
                "Open a photo to start",
                (PANEL_BOX[0] + 20, info_y + 28),
                0.5,
                COLOR_MUTED,
            )

        cv2.putText(
            canvas, "BG", (PANEL_BOX[0] + 20, 708), FONT, 0.46, COLOR_MUTED, 1, cv2.LINE_AA
        )
        swatch_x = PANEL_BOX[0] + 58
        for name, letter in (("white", "W"), ("blue", "B"), ("red", "R")):
            color_bgr = BACKGROUND_COLORS[name][::-1]
            cv2.rectangle(canvas, (swatch_x, 688), (swatch_x + 28, 712), color_bgr, -1)
            selected = name == self.bg_name
            cv2.rectangle(
                canvas,
                (swatch_x, 688),
                (swatch_x + 28, 712),
                COLOR_ACCENT if selected else COLOR_BORDER,
                2 if selected else 1,
                cv2.LINE_AA,
            )
            letter_color = (25, 25, 25) if name == "white" else (250, 250, 250)
            self._put_centered(
                canvas, letter, swatch_x + 14, 706, 0.45, letter_color
            )
            swatch_x += 40

        status_color = COLOR_ERROR if self.status_error else COLOR_ACCENT
        status = self._fit_text(self.status, panel_width - 40, 0.44)
        self._put_text(
            canvas,
            status,
            (PANEL_BOX[0] + 20, 746),
            0.44,
            status_color,
        )

    def _processing_frame(self, text: str) -> None:
        canvas = self._build_still_frame()
        center_x = (SOURCE_BOX[0] + SOURCE_BOX[2]) // 2
        center_y = (SOURCE_BOX[1] + SOURCE_BOX[3]) // 2
        box_width, box_height = 420, 70
        x0 = center_x - box_width // 2
        y0 = center_y - box_height // 2
        cv2.rectangle(
            canvas, (x0, y0), (x0 + box_width, y0 + box_height), (38, 38, 44), -1
        )
        cv2.rectangle(
            canvas,
            (x0, y0),
            (x0 + box_width, y0 + box_height),
            COLOR_BORDER,
            1,
            cv2.LINE_AA,
        )
        self._put_centered(canvas, text, center_x, y0 + 44, 0.62, COLOR_TEXT)
        if self.root is None:
            return
        try:
            self._show_canvas(canvas)
            self.root.update_idletasks()
        except tk.TclError:
            self.running = False

    @staticmethod
    def _draw_dashed_rect(
        canvas: np.ndarray,
        top_left: tuple[int, int],
        bottom_right: tuple[int, int],
        color: tuple[int, int, int],
        thickness: int = 2,
        dash: int = 14,
        gap: int = 10,
    ) -> None:
        x0, y0 = top_left
        x1, y1 = bottom_right
        step = dash + gap
        for x in range(x0, x1, step):
            end = min(x + dash, x1)
            cv2.line(canvas, (x, y0), (end, y0), color, thickness, cv2.LINE_AA)
            cv2.line(canvas, (x, y1), (end, y1), color, thickness, cv2.LINE_AA)
        for y in range(y0, y1, step):
            end = min(y + dash, y1)
            cv2.line(canvas, (x0, y), (x0, end), color, thickness, cv2.LINE_AA)
            cv2.line(canvas, (x1, y), (x1, end), color, thickness, cv2.LINE_AA)

    @staticmethod
    def _put_text(
        canvas: np.ndarray,
        text: str,
        origin: tuple[int, int],
        scale: float,
        color: tuple[int, int, int],
        thickness: int = 1,
    ) -> None:
        cv2.putText(
            canvas, text, origin, FONT, scale, color, thickness, cv2.LINE_AA
        )

    @classmethod
    def _put_centered(
        cls,
        canvas: np.ndarray,
        text: str,
        center_x: int,
        baseline_y: int,
        scale: float,
        color: tuple[int, int, int],
        thickness: int = 1,
        shadow: bool = False,
    ) -> None:
        width = cv2.getTextSize(text, FONT, scale, thickness)[0][0]
        origin = (int(center_x - width / 2), int(baseline_y))
        if shadow:
            cv2.putText(
                canvas, text, origin, FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA
            )
        cv2.putText(
            canvas, text, origin, FONT, scale, color, thickness, cv2.LINE_AA
        )

    @staticmethod
    def _fit_text(text: str, max_width: int, scale: float) -> str:
        if cv2.getTextSize(text, FONT, scale, 1)[0][0] <= max_width:
            return text
        suffix = "..."
        while text and cv2.getTextSize(text + suffix, FONT, scale, 1)[0][0] > max_width:
            text = text[:-1]
        return f"{text}{suffix}" if text else ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m idphoto.app",
        description="证件照桌面工具：自动检测、构图、抠图、换底色",
    )
    parser.add_argument("image", nargs="?", type=Path, help="启动时载入的人像图片")
    parser.add_argument("--camera", type=int, default=0, help="摄像头索引，默认 0")
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="未指定图片时不自动打开摄像头",
    )
    args = parser.parse_args(argv)
    app = IDPhotoApp(camera_index=args.camera)
    return app.run(initial_image=args.image, auto_camera=not args.no_camera)


if __name__ == "__main__":
    raise SystemExit(main())
