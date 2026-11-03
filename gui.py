"""Malody 4K auto-charting - desktop GUI.

Tkinter app: drop an audio file (or use the file picker), analyse BPM /
offset / beat grid, inspect an interactive waveform, and switch between
metrical candidates.  All heavy work runs in a worker thread.
"""

from __future__ import annotations

import ctypes
import os
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

APP_IS_FROZEN = bool(getattr(sys, "frozen", False))
APP_DIR = Path(sys.executable).resolve().parent if APP_IS_FROZEN else Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
_VENDOR = APP_DIR / "vendor"
if _VENDOR.is_dir() and str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

from core.analyze import analyze_samples, save_analysis          # noqa: E402
from core.audio import load_audio                                # noqa: E402
from core.bpm import fixed_bpm_grid                              # noqa: E402
from core.waveform import render_waveform, waveform_peaks        # noqa: E402

AUDIO_EXTS = {".ogg", ".oga", ".opus", ".mp3", ".wav", ".flac", ".m4a", ".aac", ".wma"}

# ---------------------------------------------------------------------------
# palette / typography
# ---------------------------------------------------------------------------
BG = "#0c0e13"
PANEL = "#12161f"
PANEL2 = "#1a1f2b"
BORDER = "#252c3a"
TEXT = "#e9eef6"
DIM = "#8d99ab"
FAINT = "#5d6878"
ACCENT = "#5fb0ea"
ACCENT_DARK = "#2f6f9f"
GOLD = "#ffc457"
GREEN = "#57d19a"
RED = "#ff7b72"

WAVE_FILL = "#24506f"
WAVE_EDGE = "#63b0e8"
RMS_FILL = "#3d7aa8"
CENTER_LINE = "#2c3446"
GRID_LINE = "#212836"
BEAT_LINE = "#34405c"
BEAT_ACCENT = "#4e5f84"


def font(size: int, weight: str = "normal"):
    return ("Microsoft YaHei UI", size, weight)


def mono(size: int, weight: str = "normal"):
    return ("Consolas", size, weight)


def fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m, s = divmod(seconds, 60.0)
    if abs(s - round(s)) < 0.02:
        return f"{int(m)}:{int(round(s)):02d}"
    return f"{int(m)}:{s:04.1f}"


def nice_step(span: float, target: int = 8) -> float:
    raw = span / max(target, 1)
    for step in (0.5, 1, 2, 5, 10, 15, 20, 30, 60, 120, 300, 600):
        if step >= raw:
            return float(step)
    return float(raw)


# ---------------------------------------------------------------------------
# small widgets
# ---------------------------------------------------------------------------

def flat_button(parent, text, command, kind: str = "ghost", size: int = 10, **kw):
    if kind == "accent":
        bg, fg, active = ACCENT, "#08131d", "#7cc4f2"
    elif kind == "gold":
        bg, fg, active = GOLD, "#1a1204", "#ffd27d"
    else:
        bg, fg, active = PANEL2, TEXT, "#232a38"
    return tk.Button(
        parent, text=text, command=command, bg=bg, fg=fg,
        activebackground=active, activeforeground=fg,
        relief="flat", bd=0, cursor="hand2", padx=14, pady=6,
        font=font(size), **kw,
    )


class DropZone(tk.Canvas):
    def __init__(self, master, command, **kw):
        super().__init__(master, bg=PANEL, highlightthickness=0, height=128, **kw)
        self.command = command
        self._hover = False
        self.configure(cursor="hand2")
        self.bind("<Configure>", lambda _e: self.redraw())
        self.bind("<Button-1>", lambda _e: self.command())
        self.bind("<Enter>", lambda _e: self._set_hover(True))
        self.bind("<Leave>", lambda _e: self._set_hover(False))
        self.redraw()

    def _set_hover(self, value: bool) -> None:
        if self._hover != value:
            self._hover = value
            self.redraw()

    def redraw(self) -> None:
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 20 or h < 20:
            return
        color = ACCENT if self._hover else BORDER
        self.create_rectangle(5, 5, w - 5, h - 5, outline=color, dash=(6, 4), width=2)
        self.create_text(w / 2, h / 2 - 20, text="拖入音频文件", fill=TEXT, font=font(13, "bold"))
        self.create_text(w / 2, h / 2 + 8, text="或点击选择  ·  ogg / mp3 / wav / flac",
                         fill=DIM, font=font(9))


class WaveformView(tk.Canvas):
    PAD = 18
    TOP = 20
    BOTTOM = 26

    def __init__(self, master, **kw):
        super().__init__(master, bg=PANEL, highlightthickness=0, **kw)
        self.samples: np.ndarray | None = None
        self.sr = 0
        self.beats: list[float] = []
        self.duration = 0.0
        self.start = 0.0
        self.span = 0.0
        self.full_peak = 1.0
        self._drag: tuple[float, float] | None = None
        self._hover_x: float | None = None
        self._resize_job = None

        self.bind("<Configure>", self._on_configure)
        self.bind("<Motion>", self._on_motion)
        self.bind("<Leave>", self._on_leave)
        self.bind("<MouseWheel>", self._on_wheel)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Double-Button-1>", lambda _e: self.reset_zoom())
        self.draw()

    # -- public ---------------------------------------------------------
    def set_audio(self, samples: np.ndarray, sr: int, beats_ms=None) -> None:
        self.samples = samples
        self.sr = sr
        self.duration = samples.size / float(sr) if sr else 0.0
        self.full_peak = float(np.abs(samples).max()) if samples.size else 1.0
        self.beats = list(beats_ms or [])
        self.start, self.span = 0.0, self.duration
        self.draw()

    def set_beats(self, beats_ms) -> None:
        self.beats = list(beats_ms or [])
        self.draw()

    def reset_zoom(self) -> None:
        self.start, self.span = 0.0, self.duration
        self.draw()

    # -- events ---------------------------------------------------------
    def _on_configure(self, _event=None) -> None:
        if self._resize_job:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(70, self.draw)

    def _plot_span(self) -> tuple[float, float, float, float]:
        w, h = self.winfo_width(), self.winfo_height()
        x0, x1 = self.PAD, w - self.PAD
        y0, y1 = self.TOP, h - self.BOTTOM
        return x0, x1, y0, y1

    def _time_at(self, x: float) -> float:
        x0, x1, _y0, _y1 = self._plot_span()
        frac = (x - x0) / max(x1 - x0, 1)
        return self.start + frac * self.span

    def _x_at(self, seconds: float) -> float:
        x0, x1, _y0, _y1 = self._plot_span()
        return x0 + (seconds - self.start) / max(self.span, 1e-9) * (x1 - x0)

    def _on_motion(self, event) -> None:
        self._hover_x = event.x
        self._draw_hover()

    def _on_leave(self, _event=None) -> None:
        self._hover_x = None
        self.delete("hover")

    def _on_wheel(self, event) -> None:
        if not self.samples or self.span <= 0:
            return
        x0, x1, _y0, _y1 = self._plot_span()
        x = min(max(event.x, x0), x1)
        frac = (x - x0) / max(x1 - x0, 1)
        anchor = self.start + frac * self.span
        factor = 0.82 if event.delta > 0 else 1.22
        new_span = max(1.0, min(self.duration, self.span * factor))
        new_start = max(0.0, min(anchor - frac * new_span, self.duration - new_span))
        self.start, self.span = new_start, new_span
        self._hover_x = event.x
        self.draw()

    def _on_press(self, event) -> None:
        self._drag = (event.x, self.start)
        self.configure(cursor="fleur")

    def _on_drag(self, event) -> None:
        if not self._drag or not self.samples:
            return
        x0, x1, _y0, _y1 = self._plot_span()
        dx = event.x - self._drag[0]
        new_start = self._drag[1] - dx / max(x1 - x0, 1) * self.span
        self.start = max(0.0, min(new_start, self.duration - self.span))
        self.draw()

    def _on_release(self, _event=None) -> None:
        self._drag = None
        self.configure(cursor="")

    # -- painting -------------------------------------------------------
    def draw(self) -> None:
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 40 or h < 60:
            return
        if self.samples is None or not self.sr or self.duration <= 0:
            self.create_text(w / 2, h / 2 - 8, text="拖入音频文件开始分析", fill=DIM, font=font(13))
            self.create_text(w / 2, h / 2 + 20, text="自动识别 BPM / offset，并显示波形与拍点",
                             fill=FAINT, font=font(9))
            return

        x0, x1, y0, y1 = self._plot_span()
        cy = (y0 + y1) / 2.0
        half = max(6.0, (y1 - y0) / 2.0 - 6.0)
        cols = max(20, int(x1 - x0))

        i0 = int(max(0.0, self.start) * self.sr)
        i1 = int(min(self.duration, self.start + self.span) * self.sr)
        segment = self.samples[i0:i1]
        if segment.size < 2:
            return
        peaks = waveform_peaks(segment, cols, peak=self.full_peak)

        self.create_rectangle(x0, y0, x1, y1, fill="#0f131b", outline="")

        xs = np.linspace(x0, x1, cols)
        max_arr = np.asarray(peaks["max"], dtype=np.float64)
        min_arr = np.asarray(peaks["min"], dtype=np.float64)
        rms_arr = np.asarray(peaks["rms"], dtype=np.float64)

        top_pts = np.column_stack([xs, cy - max_arr * half])
        bot_pts = np.column_stack([xs, cy - min_arr * half])
        self.create_polygon(
            top_pts.ravel().tolist() + bot_pts[::-1].ravel().tolist(),
            fill=WAVE_FILL, outline="",
        )
        self.create_line(top_pts.ravel().tolist(), fill=WAVE_EDGE, width=1)
        self.create_line(bot_pts.ravel().tolist(), fill=WAVE_EDGE, width=1)

        rms_top = np.column_stack([xs, cy - rms_arr * half])
        rms_bottom = np.column_stack([xs, cy + rms_arr * half])
        self.create_polygon(
            rms_top.ravel().tolist() + rms_bottom[::-1].ravel().tolist(),
            fill=RMS_FILL, outline="",
        )

        # beat grid: only when there is room, otherwise every 4th beat (bars)
        if self.beats:
            period = (self.beats[1] - self.beats[0]) / 1000.0 if len(self.beats) > 1 else 0.5
            per_beat_px = (x1 - x0) * period / max(self.span, 1e-9)
            if per_beat_px >= 10:
                every = 1
            elif per_beat_px * 4 >= 10:
                every = 4
            else:
                every = 0
            if every:
                for idx, t in enumerate(self.beats):
                    if idx % every:
                        continue
                    tt = t / 1000.0
                    if not (self.start - 0.001 <= tt <= self.start + self.span + 0.001):
                        continue
                    x = self._x_at(tt)
                    accent = (idx % (4 * every) == 0)
                    self.create_line(x, y0, x, y1,
                                     fill=BEAT_ACCENT if accent else BEAT_LINE)

        self.create_line(x0, cy, x1, cy, fill=CENTER_LINE)

        # time ruler
        step = nice_step(self.span)
        t = np.ceil(self.start / step) * step
        while t <= self.start + self.span + 1e-9:
            x = self._x_at(t)
            self.create_line(x, y1 + 2, x, y1 + 5, fill=GRID_LINE)
            self.create_text(x + 4, y1 + 14, text=fmt_time(t), fill=FAINT,
                             font=mono(8), anchor="w")
            t += step

        # zoom badge
        if self.span < self.duration - 0.5:
            self.create_text(x0 + 4, y0 + 10,
                             text=f"缩放 {self.span:0.1f}s / {fmt_time(self.duration)}",
                             fill=FAINT, font=mono(8), anchor="w")

        self._draw_hover()

    def _draw_hover(self) -> None:
        self.delete("hover")
        if self._hover_x is None or not self.samples:
            return
        x0, x1, y0, y1 = self._plot_span()
        x = min(max(self._hover_x, x0), x1)
        self.create_line(x, y0, x, y1, fill="#8fb8d8", dash=(2, 3), tags="hover")
        label = fmt_time(self._time_at(x))
        self.create_text(x, y0 - 6, text=label, fill=TEXT, font=mono(8),
                         anchor="s", tags="hover")


# ---------------------------------------------------------------------------
# application
# ---------------------------------------------------------------------------

class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.queue: queue.Queue = queue.Queue()
        self.busy = False
        self.samples: np.ndarray | None = None
        self.sr = 0
        self.result: dict | None = None
        self.rhythm: dict | None = None
        self.out_dir = APP_DIR / "out"

        self._build_window()
        self._build_header()
        self._build_body()
        self._build_status()
        self._enable_dnd()

        self.root.after(80, self._poll)
        self.root.after(150, self._bring_to_front)
        if len(sys.argv) > 1:
            candidate = Path(sys.argv[1])
            if candidate.is_file():
                self.root.after(300, lambda: self.load_file(candidate))

    def _bring_to_front(self) -> None:
        try:
            self.root.lift()
            self.root.attributes("-topmost", True)
            self.root.after(500, lambda: self.root.attributes("-topmost", False))
            self.root.focus_force()
        except Exception:
            pass

    # -- window ---------------------------------------------------------
    def _build_window(self) -> None:
        self.root.title("Malody 4K 自动写谱 — 音频分析")
        self.root.geometry("1320x860")
        self.root.minsize(1050, 680)
        self.root.configure(bg=BG)
        icon_ico = APP_DIR / "assets" / "icon.ico"
        icon_png = APP_DIR / "assets" / "icon.png"
        try:
            if icon_ico.is_file():
                self.root.iconbitmap(default=str(icon_ico))
        except Exception:
            pass
        try:
            if icon_png.is_file():
                self._icon = tk.PhotoImage(file=str(icon_png))
                self.root.iconphoto(True, self._icon)
        except Exception:
            pass
        self.root.bind("<Control-o>", lambda _e: self.choose_file())

    def _build_header(self) -> None:
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=24, pady=(16, 8))

        left = tk.Frame(header, bg=BG)
        left.pack(side="left")
        tk.Label(left, text="Malody 4K 自动写谱", bg=BG, fg=TEXT,
                 font=font(18, "bold")).pack(side="left")
        tk.Label(left, text="音频分析", bg=BG, fg=DIM,
                 font=font(11)).pack(side="left", padx=(12, 0), pady=(6, 0))

        right = tk.Frame(header, bg=BG)
        right.pack(side="right")
        self.btn_open_dir = flat_button(right, "打开输出目录", self.open_out_dir)
        self.btn_open_dir.pack(side="right", padx=(8, 0))
        self.btn_choose = flat_button(right, "选择音频", self.choose_file, kind="accent")
        self.btn_choose.pack(side="right")

    def _build_body(self) -> None:
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=24, pady=(0, 8))

        self.sidebar = tk.Frame(body, bg=BG, width=352)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        content = tk.Frame(body, bg=BG)
        content.pack(side="left", fill="both", expand=True, padx=(18, 0))
        self.wave = WaveformView(content)
        self.wave.pack(fill="both", expand=True)

        # ---- drop zone -------------------------------------------------
        self.drop = DropZone(self.sidebar, self.choose_file)
        self.drop.pack(fill="x", pady=(0, 12))

        # ---- file card -------------------------------------------------
        self.file_card = tk.Frame(self.sidebar, bg=PANEL, highlightbackground=BORDER,
                                  highlightthickness=1)
        self.file_card.pack(fill="x", pady=(0, 12))
        inner = tk.Frame(self.file_card, bg=PANEL)
        inner.pack(fill="x", padx=14, pady=12)
        tk.Label(inner, text="当前文件", bg=PANEL, fg=FAINT, font=font(8)).pack(anchor="w")
        self.file_name = tk.Label(inner, text="尚未加载", bg=PANEL, fg=TEXT,
                                  font=font(11, "bold"), anchor="w", justify="left",
                                  wraplength=300)
        self.file_name.pack(anchor="w", pady=(3, 2))
        self.file_meta = tk.Label(inner, text="拖入音频或点击选择", bg=PANEL, fg=DIM,
                                  font=font(9), anchor="w")
        self.file_meta.pack(anchor="w")

        # ---- BPM card --------------------------------------------------
        self.bpm_card = tk.Frame(self.sidebar, bg=PANEL, highlightbackground=BORDER,
                                 highlightthickness=1)
        self.bpm_card.pack(fill="x", pady=(0, 12))
        card = tk.Frame(self.bpm_card, bg=PANEL)
        card.pack(fill="x", padx=14, pady=12)

        top = tk.Frame(card, bg=PANEL)
        top.pack(fill="x")
        self.bpm_value = tk.Label(top, text="--", bg=PANEL, fg=ACCENT,
                                  font=mono(30, "bold"))
        self.bpm_value.pack(side="left")
        tk.Label(top, text="BPM", bg=PANEL, fg=DIM,
                 font=font(10)).pack(side="left", padx=(8, 0), pady=(12, 0))
        self.offset_label = tk.Label(card, text="首拍 offset --", bg=PANEL, fg=DIM,
                                     font=font(9))
        self.offset_label.pack(anchor="w", pady=(2, 0))

        conf_row = tk.Frame(card, bg=PANEL)
        conf_row.pack(fill="x", pady=(8, 0))
        tk.Label(conf_row, text="置信度", bg=PANEL, fg=FAINT, font=font(8)).pack(side="left")
        self.conf_bar = tk.Canvas(conf_row, width=150, height=6, bg=PANEL2,
                                  highlightthickness=0)
        self.conf_bar.pack(side="left", padx=(8, 8))
        self.conf_label = tk.Label(conf_row, text="--", bg=PANEL, fg=DIM, font=mono(8))
        self.conf_label.pack(side="left")

        tk.Label(card, text="倍频候选", bg=PANEL, fg=FAINT,
                 font=font(8)).pack(anchor="w", pady=(12, 4))
        self.cand_frame = tk.Frame(card, bg=PANEL)
        self.cand_frame.pack(fill="x", anchor="w")

        manual = tk.Frame(card, bg=PANEL)
        manual.pack(fill="x", pady=(10, 0))
        tk.Label(manual, text="手动 BPM", bg=PANEL, fg=FAINT,
                 font=font(8)).pack(side="left")
        self.bpm_entry = tk.Entry(manual, width=8, bg=PANEL2, fg=TEXT, bd=0,
                                  insertbackground=TEXT, font=mono(10),
                                  highlightthickness=1, highlightbackground=BORDER,
                                  highlightcolor=ACCENT)
        self.bpm_entry.pack(side="left", padx=(8, 6), ipady=3)
        self.bpm_entry.bind("<Return>", lambda _e: self._apply_entry_bpm())
        self.btn_apply_bpm = flat_button(manual, "应用", self._apply_entry_bpm, size=9)
        self.btn_apply_bpm.pack(side="left")
        self.btn_reset_bpm = flat_button(manual, "复位", self._reset_bpm, size=9)
        self.btn_reset_bpm.pack(side="left", padx=(6, 0))

        tip = tk.Label(self.sidebar,
                       text="滚轮缩放  ·  拖动平移  ·  双击复位视图\n"
                            "结果自动保存到 out\\（JSON + 波形 PNG）",
                       bg=BG, fg=FAINT, font=font(8), justify="left", anchor="w")
        tip.pack(fill="x", pady=(2, 0))

    def _build_status(self) -> None:
        bar = tk.Frame(self.root, bg=BG)
        bar.pack(fill="x", padx=24, pady=(0, 10))
        self.status_label = tk.Label(bar, text="就绪", bg=BG, fg=DIM, font=font(9),
                                     anchor="w")
        self.status_label.pack(side="left")
        self.out_label = tk.Label(bar, text=f"输出目录：{self.out_dir}", bg=BG, fg=FAINT,
                                  font=font(8), anchor="e")
        self.out_label.pack(side="right")

    # -- drag & drop ----------------------------------------------------
    def _enable_dnd(self) -> None:
        try:
            from tkinterdnd2 import DND_FILES
        except Exception:
            return
        for widget in (self.root, self.drop):
            try:
                widget.drop_target_register(DND_FILES)
                widget.dnd_bind("<<Drop>>", self._on_drop)
            except Exception:
                pass

    def _on_drop(self, event) -> None:
        try:
            paths = self.root.tk.splitlist(event.data)
        except Exception:
            paths = [event.data]
        for raw in paths:
            path = Path(str(raw).strip("{}"))
            if path.suffix.lower() in AUDIO_EXTS and path.is_file():
                self.load_file(path)
                return
        self.set_status("只支持音频文件：ogg / mp3 / wav / flac 等", error=True)

    # -- actions --------------------------------------------------------
    def choose_file(self) -> None:
        if self.busy:
            return
        path = filedialog.askopenfilename(
            title="选择音频文件",
            filetypes=[("音频文件", "*.ogg *.mp3 *.wav *.flac *.m4a *.opus"),
                       ("所有文件", "*.*")],
        )
        if path:
            self.load_file(Path(path))

    def open_out_dir(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(self.out_dir))
        except Exception as exc:
            messagebox.showerror("打开目录失败", str(exc))

    def load_file(self, path: Path) -> None:
        if self.busy:
            self.set_status("正在处理，请稍候…")
            return
        if not path.is_file():
            self.set_status(f"文件不存在：{path}", error=True)
            return
        if path.suffix.lower() not in AUDIO_EXTS:
            self.set_status("不支持的音频格式", error=True)
            return

        self.busy = True
        self._set_controls(False)
        self.set_status(f"正在解码并分析 {path.name} …")
        self.bpm_value.configure(text="--", fg=ACCENT)
        self.offset_label.configure(text="首拍 offset --")
        self.conf_label.configure(text="--")
        self._set_confidence(None)
        threading.Thread(target=self._worker_analyze, args=(path,), daemon=True).start()

    def _worker_analyze(self, path: Path) -> None:
        try:
            started = time.time()
            samples, sr = load_audio(path)
            result = analyze_samples(path, samples, sr, width=1600)
            rhythm = result["rhythm"]
            out_dir = self.out_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = path.stem
            save_analysis(result, out_dir / f"{stem}.analysis.json")
            meta = {
                "title": path.name,
                "bpm": rhythm["bpm"],
                "offset_ms": rhythm["offset_ms"],
                "confidence": rhythm["confidence"],
                "channels": result["audio"].get("channels"),
            }
            render_waveform(samples, sr, out_dir / f"{stem}.waveform.png", meta=meta,
                            width=2000, beat_times_ms=rhythm["beats_ms"])
            render_waveform(samples, sr, out_dir / f"{stem}.waveform.zoom.png", meta=meta,
                            width=2000, height=360, beat_times_ms=rhythm["beats_ms"],
                            span=min(20.0, max(1.0, samples.size / float(sr))),
                            title=f"{path.name}  (first 20s)")
            self.queue.put(("done", {
                "path": path, "samples": samples, "sr": sr,
                "result": result, "rhythm": rhythm,
                "elapsed": time.time() - started,
            }))
        except Exception as exc:
            self.queue.put(("error", f"{type(exc).__name__}: {exc}"))

    def _apply_entry_bpm(self) -> None:
        text = self.bpm_entry.get().strip()
        try:
            bpm = float(text)
        except ValueError:
            self.set_status("BPM 需要是数字", error=True)
            return
        self.apply_bpm(bpm)

    def apply_bpm(self, bpm: float) -> None:
        if self.busy or self.samples is None:
            return
        if not 20.0 <= bpm <= 400.0:
            self.set_status("BPM 需要在 20–400 之间", error=True)
            return
        self.busy = True
        self._set_controls(False)
        self.set_status(f"按 {bpm:.3f} BPM 重新对拍…")
        threading.Thread(target=self._worker_bpm, args=(float(bpm),), daemon=True).start()

    def _worker_bpm(self, bpm: float) -> None:
        try:
            grid = fixed_bpm_grid(self.samples, self.sr, bpm)
            rhythm = dict(self.rhythm)
            rhythm.update(
                bpm=grid["bpm"],
                offset_ms=grid["offset_ms"],
                beats_ms=grid["beats_ms"],
                coverage=grid["coverage"],
                beat_strength=grid["beat_strength"],
            )
            coverage = max(0.0, min(1.0, grid["coverage"] / 0.6))
            phase = 1.0 - float(np.exp(-max(0.0, grid["beat_strength"] - 1.0) / 2.0))
            rhythm["confidence"] = round(0.5 * coverage + 0.5 * phase, 4)

            self.result["rhythm"] = rhythm
            out_dir = self.out_dir
            stem = self.path.stem
            save_analysis(self.result, out_dir / f"{stem}.analysis.json")
            meta = {
                "title": self.path.name,
                "bpm": rhythm["bpm"],
                "offset_ms": rhythm["offset_ms"],
                "confidence": rhythm["confidence"],
                "channels": self.result["audio"].get("channels"),
            }
            render_waveform(self.samples, self.sr, out_dir / f"{stem}.waveform.png",
                            meta=meta, width=2000, beat_times_ms=rhythm["beats_ms"])
            render_waveform(self.samples, self.sr, out_dir / f"{stem}.waveform.zoom.png",
                            meta=meta, width=2000, height=360,
                            beat_times_ms=rhythm["beats_ms"],
                            span=min(20.0, self.samples.size / float(self.sr)),
                            title=f"{self.path.name}  (first 20s)")
            self.queue.put(("bpm_done", rhythm))
        except Exception as exc:
            self.queue.put(("error", f"{type(exc).__name__}: {exc}"))

    def _reset_bpm(self) -> None:
        if not self.rhythm or self.busy:
            return
        original = self.rhythm.get("_selected_bpm")
        if original:
            self.apply_bpm(float(original))

    # -- queue / UI updates --------------------------------------------
    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "done":
                    self._on_done(payload)
                elif kind == "bpm_done":
                    self._on_bpm_done(payload)
                elif kind == "error":
                    self.busy = False
                    self._set_controls(True)
                    self.set_status(f"分析失败：{payload}", error=True)
                    messagebox.showerror("分析失败", payload)
        except queue.Empty:
            pass
        self.root.after(80, self._poll)

    def _on_done(self, data: dict) -> None:
        self.busy = False
        self._set_controls(True)
        self.path = data["path"]
        self.samples = data["samples"]
        self.sr = data["sr"]
        self.result = data["result"]
        self.rhythm = data["rhythm"]
        self.rhythm["_selected_bpm"] = self.rhythm["bpm"]

        info = self.result["audio"]
        channels = info.get("channels") or 1
        self.file_name.configure(text=self.path.name)
        self.file_meta.configure(
            text=f"{fmt_time(self.rhythm['duration_sec'])}  ·  "
                 f"{self.rhythm['sr'] / 1000:.1f} kHz  ·  "
                 f"{'stereo' if channels > 1 else 'mono'}  ·  "
                 f"{info.get('format', '?')}"
        )
        self.wave.set_audio(self.samples, self.sr, self.rhythm["beats_ms"])
        self._refresh_bpm_panel()
        self.set_status(
            f"分析完成 · 用时 {data['elapsed']:.1f}s · 已保存到 {self.out_dir.name}\\"
        )

    def _on_bpm_done(self, rhythm: dict) -> None:
        self.busy = False
        self._set_controls(True)
        self.rhythm = rhythm
        self.wave.set_beats(rhythm["beats_ms"])
        self._refresh_bpm_panel()
        self.set_status(f"已按 {rhythm['bpm']:.3f} BPM 重新对拍并保存")

    def _refresh_bpm_panel(self) -> None:
        rhythm = self.rhythm
        if not rhythm:
            return
        self.bpm_value.configure(text=f"{rhythm['bpm']:.3f}", fg=ACCENT)
        self.offset_label.configure(text=f"首拍 offset {rhythm['offset_ms']:.1f} ms")
        conf = float(rhythm.get("confidence") or 0.0)
        self.conf_label.configure(text=f"{conf * 100:4.1f}%",
                                  fg=GREEN if conf >= 0.6 else DIM)
        self._set_confidence(conf)

        for child in self.cand_frame.winfo_children():
            child.destroy()
        selected = rhythm["bpm"]
        for cand in rhythm.get("candidates", [])[:5]:
            bpm = float(cand["bpm"])
            is_selected = abs(bpm - selected) < 0.01
            btn = flat_button(
                self.cand_frame, f"{bpm:.2f}",
                lambda b=bpm: self.apply_bpm(b),
                kind="gold" if is_selected else "ghost", size=9,
            )
            btn.configure(padx=9, pady=3)
            btn.pack(side="left", padx=(0, 6), pady=2)

    def _set_confidence(self, conf: float | None) -> None:
        self.conf_bar.delete("all")
        w = int(self.conf_bar["width"])
        self.conf_bar.create_rectangle(0, 0, w, 6, fill=PANEL2, outline="")
        if conf is not None:
            self.conf_bar.create_rectangle(0, 0, max(2, int(w * min(1.0, conf))), 6,
                                           fill=GREEN if conf >= 0.6 else GOLD, outline="")

    def _set_controls(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for btn in (getattr(self, "btn_choose", None),
                    getattr(self, "btn_apply_bpm", None),
                    getattr(self, "btn_reset_bpm", None),
                    getattr(self, "btn_open_dir", None)):
            if btn is not None:
                btn.configure(state=state)

    def set_status(self, text: str, error: bool = False) -> None:
        self.status_label.configure(text=text, fg=RED if error else DIM)


def run_selftest(audio_path: str) -> int:
    """Used by the packaged exe: analyse one file, verify DnD/canvas, write a log."""
    log: list[str] = []
    try:
        from tkinterdnd2 import TkinterDnD
        root = TkinterDnD.Tk()
        dnd_available = True
    except Exception as exc:
        root = tk.Tk()
        dnd_available = False
        log.append(f"dnd import failed: {exc}")
    root.geometry("1200x760")
    app = App(root)
    log.append(f"dnd_available={dnd_available}")
    log.append(f"audio={audio_path}")
    app.load_file(Path(audio_path))
    deadline = time.time() + 180
    while app.busy and time.time() < deadline:
        root.update()
        time.sleep(0.02)
    for _ in range(40):
        root.update()
        time.sleep(0.03)
    ok = bool(app.rhythm and app.rhythm.get("bpm", 0) > 0)
    log.append(f"bpm={app.rhythm.get('bpm') if app.rhythm else None}")
    log.append(f"canvas_items={len(app.wave.find_all())}")
    log.append(f"ok={ok}")
    root.destroy()
    (APP_DIR / "selftest.log").write_text("\n".join(log), encoding="utf-8")
    return 0 if ok else 1


def main() -> int:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    if "--selftest" in sys.argv:
        idx = sys.argv.index("--selftest")
        target = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else ""
        if not target:
            return 2
        return run_selftest(target)
    try:
        from tkinterdnd2 import TkinterDnD
        root = TkinterDnD.Tk()
    except Exception:
        root = tk.Tk()
    try:
        dpi = root.winfo_fpixels("1i")
        root.tk.call("tk", "scaling", max(1.0, min(1.4, dpi / 96.0)))
    except Exception:
        pass
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
