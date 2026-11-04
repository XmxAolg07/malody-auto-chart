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
from core.audio import load_audio, resample_audio                # noqa: E402
from core.bpm import fixed_bpm_grid                              # noqa: E402
from core.tempo_map import detect_tempo_map                      # noqa: E402
from core.waveform import render_waveform, waveform_peaks        # noqa: E402
from audio_player import AudioPlayer                             # noqa: E402

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


def fmt_clock(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m, s = divmod(seconds, 60.0)
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
        super().__init__(master, bg=PANEL, highlightthickness=0, height=104, **kw)
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
        self._moved = False
        self.playhead: float | None = None
        self.markers: list[tuple[float, str]] = []
        self.on_seek = None

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

    def set_playhead(self, seconds: float | None) -> None:
        self.playhead = seconds
        self._draw_playhead()

    def set_markers(self, markers) -> None:
        self.markers = list(markers or [])
        self.draw()

    def follow(self, seconds: float) -> None:
        """Scroll the view when the playhead leaves the visible window."""
        if self.samples is None or self.span <= 0:
            return
        if self.span >= self.duration - 1e-6:
            return
        left = self.start + self.span * 0.04
        right = self.start + self.span * 0.96
        if seconds < left or seconds > right:
            self.start = max(0.0, min(seconds - self.span * 0.3,
                                        self.duration - self.span))
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
        if self.samples is None or self.span <= 0:
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
        self._moved = False
        self.configure(cursor="fleur")

    def _on_drag(self, event) -> None:
        if not self._drag or self.samples is None:
            return
        x0, x1, _y0, _y1 = self._plot_span()
        dx = event.x - self._drag[0]
        if abs(dx) > 3:
            self._moved = True
        new_start = self._drag[1] - dx / max(x1 - x0, 1) * self.span
        self.start = max(0.0, min(new_start, self.duration - self.span))
        self.draw()

    def _on_release(self, event) -> None:
        was_drag = self._moved
        self._drag = None
        self._moved = False
        self.configure(cursor="")
        if not was_drag and self.on_seek is not None and self.samples is not None:
            self.on_seek(self._time_at(event.x))

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

        for t, label in self.markers:
            if self.start - 0.01 <= t <= self.start + self.span + 0.01:
                mx = self._x_at(t)
                self.create_line(mx, y0, mx, y1, fill=GOLD, dash=(3, 4))
                self.create_text(mx + 3, y0 + 2, text=label, fill=GOLD,
                                 font=mono(8), anchor="nw")

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
        self._draw_playhead()

    def _draw_playhead(self) -> None:
        self.delete("playhead")
        if self.playhead is None or self.samples is None:
            return
        x0, x1, y0, y1 = self._plot_span()
        if not (self.start - 1e-6 <= self.playhead <= self.start + self.span + 1e-6):
            return
        x = self._x_at(self.playhead)
        self.create_line(x, y0, x, y1, fill=GOLD, width=2, tags="playhead")
        self.create_text(x + 4, y0 + 10, text=fmt_clock(self.playhead),
                         fill=GOLD, font=mono(8), anchor="w", tags="playhead")

    def _draw_hover(self) -> None:
        self.delete("hover")
        if self._hover_x is None or self.samples is None:
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

        self.player = AudioPlayer()
        self.path: Path | None = None
        self.rate = 1.0
        self._rate_pending = 1.0
        self._rate_job = None
        self._rate_busy = False
        self.base_samples: np.ndarray | None = None
        self.base_sr = 0
        self.base_rhythm: dict | None = None

        self._build_window()
        self._build_header()
        self._build_body()
        self._build_status()
        self._enable_dnd()

        self.global_rhythm = None
        self._prompted_scan = False
        self.scan_precision = "low"
        self.root.after(80, self._poll)
        self.root.after(60, self._tick)
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
        self.root.bind("<Left>", lambda e: self._on_arrow(e, -1.0))
        self.root.bind("<Right>", lambda e: self._on_arrow(e, 1.0))
        self.root.bind("<Shift-Left>", lambda e: self._on_arrow(e, -5.0))
        self.root.bind("<Shift-Right>", lambda e: self._on_arrow(e, 5.0))

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
        self.wave.on_seek = self.seek_playhead
        self.node_bar = tk.Frame(content, bg=BG)
        self.node_bar.pack(fill="x", pady=(6, 0))
        self.node_inner = tk.Frame(self.node_bar, bg=BG)
        self.node_inner.pack(side="left", fill="x", expand=True)
        right = tk.Frame(self.node_bar, bg=BG)
        right.pack(side="right")
        self.seg_now = tk.Label(right, text="", bg=BG, fg=GOLD, font=font(9))
        self.seg_now.pack(side="right", padx=(8, 0))
        self.btn_scan_high = flat_button(right, "高精度检测", lambda: self.run_tempo_scan("high"), size=8)
        self.btn_scan_high.configure(padx=8, pady=3)
        self.btn_scan_high.pack(side="right", padx=(6, 0))
        self.btn_scan_low = flat_button(right, "低精度检测", lambda: self.run_tempo_scan("low"), size=8)
        self.btn_scan_low.configure(padx=8, pady=3)
        self.btn_scan_low.pack(side="right")

        self.scan_hint = tk.Frame(content, bg="#2a2416")
        self._hint_job = None
        self._hint_visible = False
        self._build_transport(content)

        # ---- drop zone -------------------------------------------------
        self.drop = DropZone(self.sidebar, self.choose_file)
        self.drop.pack(fill="x", pady=(0, 8))

        # ---- file card -------------------------------------------------
        self.file_card = tk.Frame(self.sidebar, bg=PANEL, highlightbackground=BORDER,
                                  highlightthickness=1)
        self.file_card.pack(fill="x", pady=(0, 8))
        inner = tk.Frame(self.file_card, bg=PANEL)
        inner.pack(fill="x", padx=14, pady=10)
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
        card.pack(fill="x", padx=14, pady=10)

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

        self.tempo_badge = tk.Label(card, text="", bg=PANEL, fg=GREEN, font=font(8),
                                    anchor="w", justify="left", wraplength=305)
        self.tempo_badge.pack(anchor="w", pady=(8, 0))

        tk.Label(card, text="倍频候选", bg=PANEL, fg=FAINT,
                 font=font(8)).pack(anchor="w", pady=(8, 4))
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

        # ---- rate card -------------------------------------------------
        self.rate_card = tk.Frame(self.sidebar, bg=PANEL, highlightbackground=BORDER,
                                  highlightthickness=1)
        self.rate_card.pack(fill="x", pady=(0, 10))
        rc = tk.Frame(self.rate_card, bg=PANEL)
        rc.pack(fill="x", padx=14, pady=10)
        head = tk.Frame(rc, bg=PANEL)
        head.pack(fill="x")
        tk.Label(head, text="试听倍速", bg=PANEL, fg=FAINT, font=font(8)).pack(side="left")
        self.rate_value = tk.Label(head, text="1.00×", bg=PANEL, fg=GOLD,
                                   font=mono(15, "bold"))
        self.rate_value.pack(side="right")
        self.rate_scale = tk.Scale(
            rc, from_=0.80, to=2.00, resolution=0.05, orient="horizontal",
            showvalue=0, bg=PANEL, fg=DIM, troughcolor="#39445c",
            activebackground=GOLD, highlightthickness=0, bd=0,
            sliderrelief="flat", sliderlength=26, width=14, length=300,
            command=self._on_rate_scale,
        )
        self.rate_scale.set(1.0)
        self.rate_scale.pack(fill="x", pady=(2, 2))

        rate_row = tk.Frame(rc, bg=PANEL)
        rate_row.pack(fill="x", pady=(2, 0))
        tk.Label(rate_row, text="自由填写", bg=PANEL, fg=FAINT,
                 font=font(8)).pack(side="left")
        self.rate_entry = tk.Entry(rate_row, width=5, bg=PANEL2, fg=TEXT, bd=0,
                                   insertbackground=TEXT, font=mono(9),
                                   highlightthickness=1, highlightbackground=BORDER,
                                   highlightcolor=GOLD)
        self.rate_entry.pack(side="left", padx=(6, 4), ipady=2)
        self.rate_entry.bind("<Return>", lambda _e: self._apply_rate_entry())
        self.btn_rate_apply = flat_button(rate_row, "应用", self._apply_rate_entry, size=8)
        self.btn_rate_apply.configure(padx=9, pady=3)
        self.btn_rate_apply.pack(side="left")

        export_row = tk.Frame(rc, bg=PANEL)
        export_row.pack(fill="x", pady=(6, 0))
        self.btn_export_wav = flat_button(export_row, "导出 WAV",
                                          lambda: self.export_audio_file("wav"), size=8)
        self.btn_export_wav.configure(padx=10, pady=3)
        self.btn_export_wav.pack(side="left")
        self.btn_export_ogg = flat_button(export_row, "导出 OGG",
                                          lambda: self.export_audio_file("ogg"), size=8)
        self.btn_export_ogg.configure(padx=10, pady=3)
        self.btn_export_ogg.pack(side="left", padx=(8, 0))
        tk.Label(rc, text="波形与 BPM 按倍速换算，不修改原文件", bg=PANEL,
                 fg=FAINT, font=font(8)).pack(anchor="w", pady=(5, 0))

        tip = tk.Label(self.sidebar,
                       text="滚轮缩放 · 拖动平移 · 单击定位 · ←/→ 跳转",
                       bg=BG, fg=FAINT, font=font(8), justify="left", anchor="w")
        tip.pack(fill="x", pady=(2, 0))

    def _build_transport(self, parent) -> None:
        bar = tk.Frame(parent, bg=BG)
        bar.pack(fill="x", pady=(8, 0))
        self.btn_play = flat_button(bar, "▶ 播放", self.toggle_play, kind="accent", size=10)
        self.btn_play.pack(side="left")
        self.btn_stop = flat_button(bar, "■ 停止", self.stop_playback, size=10)
        self.btn_stop.pack(side="left", padx=(8, 0))
        self.time_label = tk.Label(bar, text="0:00.0 / 0:00.0", bg=BG, fg=DIM,
                                   font=mono(10))
        self.time_label.pack(side="left", padx=(16, 0))

        for text, delta in (("«5s", -5.0), ("‹1s", -1.0), ("1s›", 1.0), ("5s»", 5.0)):
            btn = flat_button(bar, text, lambda d=delta: self.seek_relative(d), size=8)
            btn.configure(padx=7, pady=4)
            btn.pack(side="left", padx=(4, 0))

        vol = tk.Frame(bar, bg=BG)
        vol.pack(side="right")
        tk.Label(vol, text="音量", bg=BG, fg=FAINT, font=font(8)).pack(side="left",
                                                                       padx=(0, 6))
        self.volume_scale = tk.Scale(
            vol, from_=0, to=100, orient="horizontal", showvalue=0, length=110,
            bg=BG, fg=DIM, troughcolor="#39445c", activebackground=ACCENT,
            highlightthickness=0, bd=0, sliderrelief="flat",
            sliderlength=18, width=12, command=self._on_volume,
        )
        self.volume_scale.set(80)
        self.volume_scale.pack(side="left")
        if not self.player.available:
            tk.Label(bar, text="（未检测到音频输出设备）", bg=BG, fg=RED,
                     font=font(8)).pack(side="left", padx=(12, 0))

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

    # -- playback -------------------------------------------------------
    def _on_volume(self, value) -> None:
        try:
            self.player.set_volume(float(value) / 100.0)
        except Exception:
            pass

    def _set_play_button(self, playing: bool) -> None:
        try:
            self.btn_play.configure(text="⏸ 暂停" if playing else "▶ 播放")
        except Exception:
            pass

    def toggle_play(self) -> None:
        if self.samples is None:
            return
        if self.player.playing:
            self.player.pause()
            self._set_play_button(False)
            return
        if self.player.play():
            self._set_play_button(True)
        else:
            self.set_status(f"无法播放：{self.player.error}", error=True)

    def stop_playback(self) -> None:
        self.player.stop()
        self._set_play_button(False)
        self.wave.set_playhead(0.0)
        self.time_label.configure(text=f"0:00.0 / {fmt_clock(self.player.duration)}")

    def seek_playhead(self, seconds: float) -> None:
        if self.samples is None:
            return
        seconds = max(0.0, min(seconds, self.player.duration))
        self.player.seek(seconds)
        self.wave.set_playhead(seconds)
        self.time_label.configure(
            text=f"{fmt_clock(seconds)} / {fmt_clock(self.player.duration)}"
        )

    def seek_relative(self, delta: float) -> None:
        if self.samples is None:
            return
        self.seek_playhead(self.player.position() + float(delta))

    def _on_arrow(self, _event, delta: float) -> None:
        if isinstance(self.root.focus_get(), tk.Entry):
            return
        self.seek_relative(delta)

    def _tick(self) -> None:
        try:
            if self.samples is not None:
                pos = self.player.position()
                self.wave.set_playhead(pos)
                self._update_active_node(pos)
                self.time_label.configure(
                    text=f"{fmt_clock(pos)} / {fmt_clock(self.player.duration)}"
                )
                if self.player.playing:
                    self.wave.follow(pos)
                    if self.player.pos >= self.player.data.size:
                        self.player.pause()
                        self._set_play_button(False)
                elif str(self.btn_play.cget("text")).startswith("⏸"):
                    self._set_play_button(False)
        except Exception:
            pass
        self.root.after(40, self._tick)

    # -- preview rate ---------------------------------------------------
    def _on_rate_scale(self, value) -> None:
        try:
            rate = float(value)
        except Exception:
            return
        rate = max(0.80, min(2.00, round(rate / 0.05) * 0.05))
        self._rate_pending = rate
        self._show_rate(rate)
        if self.base_rhythm is None:
            return
        if self._rate_job is not None:
            self.root.after_cancel(self._rate_job)
        self._rate_job = self.root.after(380, self._commit_rate)

    def _show_rate(self, rate: float) -> None:
        try:
            self.rate_value.configure(text=f"{rate:.2f}×")
        except Exception:
            pass

    def _apply_rate_entry(self) -> None:
        text = self.rate_entry.get().strip().rstrip("xX×")
        try:
            rate = float(text)
        except ValueError:
            self.set_status("倍速需要是数字，例如 1.25", error=True)
            return
        self.set_rate(rate, immediate=True)

    def set_rate(self, rate: float, immediate: bool = True) -> None:
        """Public entry used by tests: apply a preview rate."""
        rate = max(0.25, min(4.0, round(float(rate) / 0.05) * 0.05))
        self._rate_pending = rate
        self._show_rate(rate)
        try:
            self.rate_scale.set(rate)
        except Exception:
            pass
        if self.base_rhythm is None:
            self.rate = rate
            return
        if immediate:
            self._commit_rate()

    def _commit_rate(self) -> None:
        if self.base_rhythm is None or self._rate_busy:
            return
        rate = float(self._rate_pending)
        if abs(rate - self.rate) < 1e-6:
            return
        old_rate = self.rate
        old_pos = self.player.position()
        was_playing = self.player.playing
        self.player.pause()
        self._set_play_button(False)
        self._rate_busy = True
        self.set_status(f"正在生成 ×{rate:.2f} 变速音频…")
        threading.Thread(
            target=self._worker_rate,
            args=(rate, old_pos, was_playing, old_rate),
            daemon=True,
        ).start()

    def _worker_rate(self, rate: float, pos: float, was_playing: bool,
                     old_rate: float) -> None:
        try:
            if abs(rate - 1.0) < 1e-6:
                samples = self.base_samples
            else:
                sr_out = max(8000, int(round(self.base_sr / rate)))
                samples = resample_audio(self.base_samples, self.base_sr, sr_out)
            rhythm = self._scaled_rhythm(self.base_rhythm, rate)
            self.queue.put(("rate_done", {
                "rate": rate,
                "samples": samples,
                "rhythm": rhythm,
                "pos": pos * old_rate / rate,
                "was_playing": was_playing,
            }))
        except Exception as exc:
            self.queue.put(("error", f"{type(exc).__name__}: {exc}"))

    def _scaled_rhythm(self, base: dict, rate: float) -> dict:
        r = {k: v for k, v in base.items() if not k.startswith("_")}
        r["bpm"] = round(base["bpm"] * rate, 3)
        r["offset_ms"] = round(base["offset_ms"] / rate, 2)
        r["beats_ms"] = [round(t / rate, 2) for t in base["beats_ms"]]
        r["duration_sec"] = round(base["duration_sec"] / rate, 3)
        r["candidates"] = [
            {"bpm": round(c["bpm"] * rate, 3),
             "score": c["score"], "confidence": c["confidence"]}
            for c in base["candidates"]
        ]
        segments = base.get("segments") or []
        if segments:
            r["segments"] = [
                {
                    "start_ms": round(s["start_ms"] / rate, 1),
                    "end_ms": round(s["end_ms"] / rate, 1),
                    "bpm": round(s["bpm"] * rate, 3),
                    "confidence": s.get("confidence", 0.0),
                    "beats_ms": [round(t / rate, 1) for t in s.get("beats_ms", [])],
                }
                for s in segments
            ]
        else:
            r["segments"] = []
        r["has_changes"] = bool(base.get("has_changes")) and bool(segments)
        r["confidence"] = base.get("confidence", 0.0)
        r["_rate"] = rate
        r["_selected_bpm"] = round(base.get("_selected_bpm", base["bpm"]) * rate, 3)
        return r

    def _on_rate_done(self, data: dict) -> None:
        self._rate_busy = False
        rate = float(data["rate"])
        self.rate = rate
        self.samples = data["samples"]
        self.rhythm = data["rhythm"]
        self.player.load(self.samples, self.base_sr)
        self.player.seek(data["pos"])
        self.wave.set_audio(self.samples, self.base_sr, self.rhythm["beats_ms"])
        self.wave.set_playhead(data["pos"])
        self._refresh_bpm_panel()
        self.time_label.configure(
            text=f"{fmt_clock(data['pos'])} / {fmt_clock(self.player.duration)}"
        )
        self._set_play_button(False)
        if data["was_playing"] and self.player.play():
            self._set_play_button(True)
        if abs(self._rate_pending - rate) > 1e-6:
            self.root.after(80, self._commit_rate)
        else:
            self.set_status(
                f"试听倍速 ×{rate:.2f}：BPM {self.rhythm['bpm']:.3f}（不修改原文件）"
            )

    def export_audio_file(self, fmt: str) -> None:
        if self.samples is None or self.path is None:
            return
        rate = self.rate
        tag = "" if abs(rate - 1.0) < 1e-6 else f"_x{rate:.2f}"
        out = self.out_dir / f"{self.path.stem}{tag}.{fmt}"
        self.set_status(f"正在导出 {out.name} …")
        threading.Thread(target=self._worker_export_audio,
                         args=(out, fmt, self.samples, self.base_sr),
                         daemon=True).start()

    def _worker_export_audio(self, out, fmt: str, samples, sr: int) -> None:
        try:
            from core.audio import export_audio
            export_audio(out, samples, sr, fmt)
            self.queue.put(("exported", str(out)))
        except Exception as exc:
            self.queue.put(("error", f"{type(exc).__name__}: {exc}"))

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
        if self.busy or self.samples is None or self.base_rhythm is None:
            return
        if not 20.0 <= bpm <= 400.0:
            self.set_status("BPM 需要在 20–400 之间", error=True)
            return
        base_bpm = float(bpm) / max(self.rate, 1e-9)
        self.busy = True
        self._set_controls(False)
        self.set_status(f"按 {bpm:.3f} BPM 重新对拍…")
        threading.Thread(target=self._worker_bpm, args=(base_bpm,), daemon=True).start()

    def _worker_bpm(self, base_bpm: float) -> None:
        try:
            grid = fixed_bpm_grid(self.base_samples, self.base_sr, base_bpm)
            base = dict(self.base_rhythm)
            base.update(
                bpm=grid["bpm"],
                offset_ms=grid["offset_ms"],
                beats_ms=grid["beats_ms"],
                coverage=grid["coverage"],
                beat_strength=grid["beat_strength"],
                _selected_bpm=grid["bpm"],
            )
            coverage = max(0.0, min(1.0, grid["coverage"] / 0.6))
            phase = 1.0 - float(np.exp(-max(0.0, grid["beat_strength"] - 1.0) / 2.0))
            base["confidence"] = round(0.5 * coverage + 0.5 * phase, 4)
            base["segments"] = []
            base["has_changes"] = False
            self.base_rhythm = base

            rhythm = self._scaled_rhythm(base, self.rate)
            saved = abs(self.rate - 1.0) < 1e-6
            if saved:
                self.rhythm = rhythm
                self.result["rhythm"] = {
                    k: v for k, v in rhythm.items() if not k.startswith("_")
                }
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
                render_waveform(self.samples, self.sr,
                                out_dir / f"{stem}.waveform.zoom.png",
                                meta=meta, width=2000, height=360,
                                beat_times_ms=rhythm["beats_ms"],
                                span=min(20.0, self.samples.size / float(self.sr)),
                                title=f"{self.path.name}  (first 20s)")
            self.queue.put(("bpm_done", {"rhythm": rhythm, "saved": saved}))
        except Exception as exc:
            self.queue.put(("error", f"{type(exc).__name__}: {exc}"))

    def _reset_bpm(self) -> None:
        if not self.base_rhythm or self.busy:
            return
        original = self.base_rhythm.get("_original_bpm")
        if original:
            self.apply_bpm(float(original) * self.rate)

    # -- tempo rescan ---------------------------------------------------
    def run_tempo_scan(self, precision: str = "high") -> None:
        if self.busy or self.base_samples is None or self.global_rhythm is None:
            return
        self.busy = True
        self._set_controls(False)
        label = "高精度" if precision == "high" else "低精度"
        self._hide_scan_hint()
        self.set_status(f"{label}变速检测中…（可能需要几秒）")
        threading.Thread(target=self._worker_tempo_scan, args=(precision,),
                         daemon=True).start()

    def _worker_tempo_scan(self, precision: str) -> None:
        try:
            tempo = detect_tempo_map(self.base_samples, self.base_sr,
                                     float(self.global_rhythm["bpm"]),
                                     precision=precision)
            self.queue.put(("tempo_scan", {"precision": precision, "tempo": tempo}))
        except Exception as exc:
            self.queue.put(("error", f"{type(exc).__name__}: {exc}"))

    def _apply_tempo_map(self, tempo: dict, precision: str) -> None:
        base = {k: v for k, v in self.global_rhythm.items()
                if not k.startswith("_") and k not in ("segments", "has_changes")}
        if tempo.get("has_changes"):
            base["segments"] = [s.to_dict() for s in tempo["segments"]]
            base["has_changes"] = True
            base["bpm"] = round(float(tempo["bpm"]), 3)
            beats = []
            for seg in tempo["segments"]:
                beats.extend(seg.beats_ms)
            base["beats_ms"] = sorted(beats)
            base["_selected_bpm"] = base["bpm"]
        else:
            base["segments"] = []
            base["has_changes"] = False
        self.base_rhythm = base
        self.result["rhythm"] = {k: v for k, v in base.items() if not k.startswith("_")}
        if abs(self.rate - 1.0) < 1e-6:
            self.rhythm = dict(base)
        else:
            self.rhythm = self._scaled_rhythm(base, self.rate)
        self.wave.set_beats(self.rhythm["beats_ms"])
        self._refresh_bpm_panel()
        self.scan_precision = precision
        label = "高精度" if precision == "high" else "低精度"
        if base.get("has_changes"):
            self.set_status(f"{label}变速检测：{len(base['segments'])} 段")
        else:
            self.set_status(f"{label}变速检测：未检测到变速")

    def _show_scan_hint(self) -> None:
        """Non-modal suggestion bar; never blocks the UI and auto-hides."""
        if self._hint_visible or not self.rhythm:
            return
        segs = self.rhythm.get("segments") or []
        if not segs:
            return
        self._prompted_scan = True
        for child in self.scan_hint.winfo_children():
            child.destroy()
        tk.Label(self.scan_hint, text=f"检测到 {len(segs)} 段变速，建议高精度重测",
                 bg="#2a2416", fg=TEXT, font=font(8)).pack(side="left", padx=(8, 6), pady=3)
        btn = flat_button(self.scan_hint, "高精度重测", self._rescan_high_from_hint, size=8)
        btn.configure(padx=8, pady=2)
        btn.pack(side="left", padx=(0, 6), pady=2)
        flat_button(self.scan_hint, "忽略", self._hide_scan_hint, size=8).pack(
            side="left", padx=(0, 8), pady=2)
        self.scan_hint.pack(fill="x", pady=(4, 0), before=self.node_bar)
        self._hint_visible = True
        if self._hint_job is not None:
            self.root.after_cancel(self._hint_job)
        self._hint_job = self.root.after(15000, self._hide_scan_hint)

    def _hide_scan_hint(self) -> None:
        if self._hint_job is not None:
            try:
                self.root.after_cancel(self._hint_job)
            except Exception:
                pass
            self._hint_job = None
        if self._hint_visible:
            self.scan_hint.pack_forget()
            self._hint_visible = False

    def _rescan_high_from_hint(self) -> None:
        self._hide_scan_hint()
        self.run_tempo_scan("high")

    # -- queue / UI updates --------------------------------------------
    def _poll(self) -> None:
        try:
            while True:
                try:
                    kind, payload = self.queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    if kind == "done":
                        self._on_done(payload)
                    elif kind == "bpm_done":
                        self._on_bpm_done(payload)
                    elif kind == "rate_done":
                        self._on_rate_done(payload)
                    elif kind == "tempo_scan":
                        self._on_tempo_scan(payload)
                    elif kind == "exported":
                        self.set_status(f"已导出变速音频：{Path(payload).name}")
                    elif kind == "error":
                        self.busy = False
                        self._rate_busy = False
                        self._set_controls(True)
                        self.set_status(f"处理失败：{payload}", error=True)
                except Exception as exc:  # keep the poll loop alive
                    self.busy = False
                    self._rate_busy = False
                    self._set_controls(True)
                    self.set_status(f"界面错误：{type(exc).__name__}: {exc}", error=True)
        finally:
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
        self.rhythm["_original_bpm"] = self.rhythm["bpm"]

        # keep the untouched analysis as the base for preview-rate scaling
        self.base_samples = self.samples
        self.base_sr = self.sr
        self.base_rhythm = dict(self.rhythm)
        self.global_rhythm = {k: v for k, v in self.rhythm.items()
                              if not k.startswith("_") and k not in ("segments", "has_changes")}
        self.scan_precision = "low"
        self._prompted_scan = False
        self.rate = 1.0
        self._rate_pending = 1.0
        self._rate_busy = False
        self._show_rate(1.0)
        try:
            self.rate_scale.set(1.0)
        except Exception:
            pass

        self.player.load(self.samples, self.sr)
        self.player.seek(0.0)
        self.wave.set_playhead(0.0)
        self.time_label.configure(text=f"0:00.0 / {fmt_clock(self.player.duration)}")
        self._set_play_button(False)

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
        if self.rhythm.get("has_changes"):
            self.root.after(400, self._show_scan_hint)
        self.set_status(
            f"分析完成 · 用时 {data['elapsed']:.1f}s · 已保存到 {self.out_dir.name}\\"
        )

    def _on_tempo_scan(self, payload: dict) -> None:
        self.busy = False
        self._set_controls(True)
        self._apply_tempo_map(payload["tempo"], payload["precision"])

    def _on_bpm_done(self, payload: dict) -> None:
        self.busy = False
        self._set_controls(True)
        self.rhythm = payload["rhythm"]
        self.global_rhythm = {k: v for k, v in (self.base_rhythm or {}).items()
                              if not k.startswith("_") and k not in ("segments", "has_changes")}
        self.wave.set_beats(self.rhythm["beats_ms"])
        self._refresh_bpm_panel()
        if payload.get("saved"):
            self.set_status(f"已按 {self.rhythm['bpm']:.3f} BPM 重新对拍并保存")
        else:
            self.set_status(
                f"已按 {self.rhythm['bpm']:.3f} BPM 重新对拍（试听倍速 ×{self.rate:.2f}，未覆盖原文件）"
            )

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

        if rhythm.get("has_changes") and rhythm.get("segments"):
            parts = [f"{fmt_time(s['start_ms'] / 1000)} {s['bpm']:.1f}"
                     for s in rhythm["segments"][:3]]
            self.tempo_badge.configure(text="变速：" + " · ".join(parts), fg=GOLD)
        else:
            self.tempo_badge.configure(text="恒定 BPM（未检测到变速）", fg=GREEN)
        self._refresh_nodes()

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

    def _refresh_nodes(self) -> None:
        for child in self.node_inner.winfo_children():
            child.destroy()
        self._nodes = []
        self._active_node = -1
        segments = (self.rhythm or {}).get("segments") or []
        if len(segments) <= 1:
            tk.Label(self.node_inner, text="恒定 BPM", bg=BG, fg=GREEN,
                     font=font(8)).pack(side="left")
            self.seg_now.configure(text="")
            self.wave.set_markers([])
            return
        for seg in segments[:8]:
            btn = flat_button(
                self.node_inner,
                f"{fmt_time(seg['start_ms'] / 1000)}  {seg['bpm']:.1f}",
                lambda t=seg['start_ms'] / 1000: self.seek_playhead(t),
                size=8,
            )
            btn.configure(padx=8, pady=2)
            btn.pack(side="left", padx=(0, 6))
            self._nodes.append((btn, seg))
        self.wave.set_markers([(s["start_ms"] / 1000, f"{s['bpm']:.0f}")
                               for s in segments[1:]])
        self.seg_now.configure(text="")

    def _update_active_node(self, pos: float) -> None:
        segments = (self.rhythm or {}).get("segments") or []
        if len(segments) <= 1 or not getattr(self, "_nodes", None):
            return
        ms = pos * 1000.0
        active = -1
        for i, (btn, seg) in enumerate(self._nodes):
            if seg["start_ms"] <= ms < seg["end_ms"]:
                active = i
                break
        if active == getattr(self, "_active_node", -1):
            return
        self._active_node = active
        for i, (btn, seg) in enumerate(self._nodes):
            if i == active:
                btn.configure(bg=GOLD, fg="#1a1204")
            else:
                btn.configure(bg=PANEL2, fg=TEXT)
        if active >= 0:
            self.seg_now.configure(text=f"当前 {segments[active]['bpm']:.1f} BPM")

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
                    getattr(self, "btn_open_dir", None),
                    getattr(self, "btn_rate_apply", None),
                    getattr(self, "btn_export_wav", None),
                    getattr(self, "btn_export_ogg", None),
                    getattr(self, "btn_scan_low", None),
                    getattr(self, "btn_scan_high", None),
                    getattr(self, "btn_play", None),
                    getattr(self, "btn_stop", None)):
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
    try:
        root.tk.call("tk", "scaling",
                     max(1.0, min(1.4, root.winfo_fpixels("1i") / 96.0)))
    except Exception:
        pass
    app = App(root)
    log.append(f"dnd_available={dnd_available}")
    log.append(f"player_available={app.player.available}")
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
    log.append(f"segments={len(app.rhythm.get('segments') or []) if app.rhythm else 0}")
    if ok and app.rhythm and app.rhythm.get("has_changes"):
        app._hide_scan_hint()
        app.run_tempo_scan("high")
        deadline = time.time() + 120
        while app.busy and time.time() < deadline:
            root.update()
            time.sleep(0.02)
        for _ in range(20):
            root.update()
            time.sleep(0.02)
        log.append(f"high_segments={len(app.rhythm.get('segments') or [])}")
        log.append(f"scan_precision={app.scan_precision}")

    # preview-rate scaling + playback position (silent, volume 0)
    if ok:
        app.player.set_volume(0.0)
        app.set_rate(1.50, immediate=True)
        deadline = time.time() + 60
        while (app._rate_busy) and time.time() < deadline:
            root.update()
            time.sleep(0.02)
        for _ in range(20):
            root.update()
            time.sleep(0.02)
        log.append(f"rate={app.rate:.2f}")
        log.append(f"rate_bpm={app.rhythm.get('bpm') if app.rhythm else None}")
        try:
            from core.audio import export_audio
            import time as _time
            for fmt in ('wav', 'ogg'):
                out = APP_DIR / 'out' / f'selftest.{fmt}'
                export_audio(out, app.samples, app.base_sr, fmt)
                log.append(f"export_{fmt}={out.stat().st_size}")
        except Exception as exc:
            log.append(f"export_failed={type(exc).__name__}: {exc}")
        if app.player.available:
            started = app.player.play()
            time.sleep(0.5)
            pos = app.player.position()
            app.player.pause()
            log.append(f"play_started={started} pos_after_0.5s={pos:.3f}")
            ok = ok and started and pos > 0.05

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
