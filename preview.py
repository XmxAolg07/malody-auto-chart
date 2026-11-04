"""Falling-notes 4K chart preview window.

Opens a Malody ``.mc`` chart next to the audio player: notes fall towards a
judgement line in real time, with play/pause, a draggable seek bar, a chart
offset slider and a scroll-speed slider.  The audio itself is the app's
existing player, so the main window and the preview stay in sync.
"""

from __future__ import annotations

import bisect
import json
import math
import os
import struct
import tempfile
import tkinter as tk
import wave
from pathlib import Path

try:
    import winsound
except ImportError:  # non-Windows
    winsound = None

BG = "#0a0c11"
PANEL = "#12161f"
TEXT = "#e9eef6"
DIM = "#8d99ab"
FAINT = "#5d6878"
GOLD = "#ffc457"
ACCENT = "#5fb0ea"
LANE_COLORS = ("#5fb0ea", "#e9eef6", "#e9eef6", "#5fb0ea")
LANE_DARK = ("#2f6f9f", "#8d99ab", "#8d99ab", "#2f6f9f")


def _beat_float(beat) -> float:
    if not beat:
        return 0.0
    if len(beat) == 1:
        return float(beat[0])
    return float(beat[0]) + float(beat[1]) / float(beat[2] or 1)


def load_chart(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    times = []
    for item in data.get("time") or []:
        try:
            times.append((_beat_float(item["beat"]), float(item["bpm"])))
        except (KeyError, TypeError, ValueError):
            continue
    times.sort(key=lambda item: item[0])
    if not times:
        times = [(0.0, 180.0)]

    segments = []
    seconds = 0.0
    for index, (beat, bpm) in enumerate(times):
        if index > 0:
            prev_beat, prev_seconds, prev_bpm = segments[-1]
            seconds = prev_seconds + (beat - prev_beat) * 60.0 / prev_bpm
        segments.append((beat, seconds, bpm))

    notes = []
    for note in data.get("note") or []:
        if not isinstance(note, dict):
            continue
        beat = note.get("beat")
        lane = note.get("column")
        if beat is None or lane is None:
            continue
        beat_value = _beat_float(beat)
        segment = segments[0]
        for candidate in segments:
            if candidate[0] <= beat_value + 1e-9:
                segment = candidate
            else:
                break
        sec = segment[1] + (beat_value - segment[0]) * 60.0 / segment[2]
        notes.append((sec, int(lane)))
    notes.sort()
    return {
        "meta": data.get("meta") or {},
        "segments": segments,
        "notes": notes,
        "times": [note[0] for note in notes],
        "data": data,
    }


class ChartPreviewWindow(tk.Toplevel):
    def __init__(self, app, chart_path: str | Path) -> None:
        super().__init__(app.root)
        self.app = app
        self.chart_path = Path(chart_path)
        self.chart = load_chart(chart_path)
        self.title(f"谱面预览 — {self.chart_path.name}")
        self.configure(bg=BG)
        self.geometry("1240x820")
        self.minsize(900, 620)
        self.transient(app.root)
        self.offset_ms = 0.0
        self.speed = 620.0
        self.closed = False
        self._drag_seek = False
        self.sound_on = tk.BooleanVar(value=True)
        self.grid_on = tk.BooleanVar(value=True)
        self.last_pos = app.player.position()
        self.click_path = self._make_click()
        self._build()
        self.bind("<space>", lambda _e: self.toggle())
        self.bind("<Left>", lambda _e: self.seek_by(-1.0))
        self.bind("<Right>", lambda _e: self.seek_by(1.0))
        self.bind("<Up>", lambda _e: self.seek_by(5.0))
        self.bind("<Down>", lambda _e: self.seek_by(-5.0))
        self.bind("<Escape>", lambda _e: self.close())
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.after(20, self._tick)

    # -- layout ---------------------------------------------------------
    def _build(self) -> None:
        top = tk.Frame(self, bg=BG)
        top.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(top, bg=BG, highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True, padx=(12, 6), pady=10)
        self.canvas.bind("<Configure>", lambda _e: self.draw())

        side = tk.Frame(top, bg=PANEL, width=250)
        side.pack(side="right", fill="y", padx=(6, 12), pady=10)
        side.pack_propagate(False)

        tk.Label(side, text="谱面预览", bg=PANEL, fg=TEXT,
                 font=("Microsoft YaHei UI", 12, "bold")).pack(anchor="w", padx=14, pady=(14, 2))
        meta = self.chart.get("meta") or {}
        version = meta.get("version") or self.chart_path.stem
        tk.Label(side, text=version, bg=PANEL, fg=DIM, wraplength=220,
                 justify="left", font=("Microsoft YaHei UI", 8)).pack(anchor="w", padx=14)
        tk.Label(side, text=f"{len(self.chart['notes'])} 音符 · 4K", bg=PANEL, fg=FAINT,
                 font=("Microsoft YaHei UI", 8)).pack(anchor="w", padx=14, pady=(2, 10))

        self.time_label = tk.Label(side, text="00:00.000", bg=PANEL, fg=TEXT,
                                   font=("Consolas", 17, "bold"))
        self.time_label.pack(anchor="w", padx=14)
        self.beat_label = tk.Label(side, text="Beat 0:0/8", bg=PANEL, fg=GOLD,
                                   font=("Consolas", 10))
        self.beat_label.pack(anchor="w", padx=14, pady=(1, 0))
        self.bpm_label = tk.Label(side, text="BPM —", bg=PANEL, fg=DIM,
                                  font=("Consolas", 9))
        self.bpm_label.pack(anchor="w", padx=14)

        self.play_btn = tk.Button(
            side, text="▶ 播放", command=self.toggle, bg=ACCENT, fg="#08131d",
            activebackground="#7cc4f2", activeforeground="#08131d", relief="flat",
            bd=0, cursor="hand2", font=("Microsoft YaHei UI", 10), pady=6,
        )
        self.play_btn.pack(fill="x", padx=14, pady=(12, 6))

        tk.Label(side, text="谱面偏移（ms）", bg=PANEL, fg=DIM,
                 font=("Microsoft YaHei UI", 8)).pack(anchor="w", padx=14, pady=(8, 0))
        self.offset_scale = tk.Scale(
            side, from_=-200, to=200, resolution=5, orient="horizontal",
            bg=PANEL, fg=DIM, troughcolor="#202736", highlightthickness=0,
            bd=0, sliderrelief="flat", command=self._on_offset,
        )
        self.offset_scale.pack(fill="x", padx=10)

        tk.Label(side, text="下落速度（px/s）", bg=PANEL, fg=DIM,
                 font=("Microsoft YaHei UI", 8)).pack(anchor="w", padx=14, pady=(4, 0))
        self.speed_scale = tk.Scale(
            side, from_=280, to=1400, resolution=20, orient="horizontal",
            bg=PANEL, fg=DIM, troughcolor="#202736", highlightthickness=0,
            bd=0, sliderrelief="flat", command=self._on_speed,
        )
        self.speed_scale.set(620)
        self.speed_scale.pack(fill="x", padx=10)

        options = tk.Frame(side, bg=PANEL)
        options.pack(fill="x", padx=10, pady=(6, 0))
        for label, variable in (("按键音", self.sound_on), ("网格", self.grid_on)):
            tk.Checkbutton(
                options, text=label, variable=variable, bg=PANEL, fg=TEXT,
                selectcolor="#202736", activebackground=PANEL,
                activeforeground=GOLD, font=("Microsoft YaHei UI", 8),
            ).pack(side="left", padx=(0, 10))

        tk.Label(side, text="空格 播放/暂停 · ←/→ 1s · ↑/↓ 5s",
                 bg=PANEL, fg=FAINT, font=("Microsoft YaHei UI", 7),
                 wraplength=220, justify="left").pack(anchor="w", padx=14, pady=(10, 0))

        bottom = tk.Frame(self, bg=BG)
        bottom.pack(fill="x", padx=12, pady=(0, 10))
        self.seek = tk.Canvas(bottom, height=28, bg=BG, highlightthickness=0,
                              cursor="hand2")
        self.seek.pack(fill="x")
        self.seek.bind("<Button-1>", self._seek_click)
        self.seek.bind("<B1-Motion>", self._seek_drag)
        self.seek.bind("<ButtonRelease-1>", self._seek_release)
        self.seek.bind("<Configure>", lambda _e: self.draw_seek())

    # -- controls -------------------------------------------------------
    def _on_offset(self, value) -> None:
        self.offset_ms = float(value)
        self.last_pos = self._position()
        self.draw()

    def _on_speed(self, value) -> None:
        self.speed = float(value)
        self.draw()

    def toggle(self) -> None:
        player = self.app.player
        if player.playing:
            player.pause()
        else:
            if player.position() >= player.duration - 0.05:
                player.seek(0.0)
            player.play()
            self.last_pos = player.position()
        self._sync_button()

    def _sync_button(self) -> None:
        playing = self.app.player.playing
        self.play_btn.configure(text="⏸ 暂停" if playing else "▶ 播放")

    def seek_by(self, delta: float) -> None:
        player = self.app.player
        player.seek(max(0.0, min(player.duration, player.position() + delta)))
        self.last_pos = player.position()
        self.draw()

    def _seek_click(self, event) -> None:
        self._drag_seek = True
        self._seek_to(event.x)

    def _seek_drag(self, event) -> None:
        if self._drag_seek:
            self._seek_to(event.x)

    def _seek_release(self, _event) -> None:
        self._drag_seek = False

    def _seek_to(self, x: int) -> None:
        width = max(1, self.seek.winfo_width())
        fraction = min(1.0, max(0.0, x / width))
        self.app.player.seek(fraction * self.app.player.duration)
        self.last_pos = self._position()
        self.draw()

    # -- position helper ------------------------------------------------
    def _position(self) -> float:
        return self.app.player.position()

    def _beat_at(self, seconds: float) -> tuple[float, float]:
        segments = self.chart["segments"]
        segment = segments[0]
        for candidate in segments:
            if candidate[1] <= seconds + 1e-9:
                segment = candidate
            else:
                break
        beat = segment[0] + (seconds - segment[1]) * segment[2] / 60.0
        return beat, segment[2]

    # -- drawing --------------------------------------------------------
    def draw(self) -> None:
        if self.closed:
            return
        canvas = self.canvas
        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())
        canvas.delete("all")
        margin = 12
        lane_w = (width - margin * 2) / 4.0
        judge_y = height * 0.84
        pos = self._position() - self.offset_ms / 1000.0

        for lane in range(4):
            x0 = margin + lane * lane_w
            canvas.create_rectangle(x0, 0, x0 + lane_w, height,
                                    fill="#0d1119" if lane % 2 == 0 else "#0b0e15",
                                    outline="")
            canvas.create_line(x0, 0, x0, height, fill="#202736")
        canvas.create_line(width - margin, 0, width - margin, height, fill="#202736")
        canvas.create_line(margin, judge_y, width - margin, judge_y,
                           fill="#5a4312", width=7)
        canvas.create_line(margin, judge_y, width - margin, judge_y,
                           fill=GOLD, width=2)
        canvas.create_text(margin + 6, judge_y + 12, text="判定线",
                           fill=FAINT, font=("Microsoft YaHei UI", 7), anchor="w")

        # beat grid: 1/4 beat lines, stronger at beats and measures
        start_beat, _bpm = self._beat_at(max(0.0, pos))
        end_beat, _bpm2 = self._beat_at(pos + judge_y / max(1.0, self.speed) + 0.5)
        if self.grid_on.get():
            step = 0.25
            beat_now = math.floor(start_beat / step) * step
            while beat_now <= end_beat:
                seconds = self._time_at(beat_now)
                y = judge_y - (seconds - pos) * self.speed
                if 0 <= y <= height:
                    if abs(beat_now - round(beat_now)) < 1e-6:
                        measure = int(round(beat_now))
                        color = "#3a4560" if measure % 4 == 0 else "#242c3d"
                        thickness = 2 if measure % 4 == 0 else 1
                    else:
                        color, thickness = "#161c28", 1
                    canvas.create_line(margin, y, width - margin, y,
                                       fill=color, width=thickness)
                beat_now += step
        else:
            measure = math.floor(start_beat / 4.0) * 4.0
            while measure <= end_beat:
                seconds = self._time_at(measure)
                y = judge_y - (seconds - pos) * self.speed
                if 0 <= y <= height:
                    canvas.create_line(margin, y, width - margin, y, fill="#1b2230")
                measure += 4.0

        for lane in range(4):
            center = margin + (lane + 0.5) * lane_w
            canvas.create_text(center, 12, text=str(lane + 1), fill=FAINT,
                               font=("Consolas", 9))

        # notes
        lookahead = judge_y / max(1.0, self.speed) + 0.4
        lo = bisect.bisect_left(self.chart["times"], pos - 0.2)
        hi = bisect.bisect_right(self.chart["times"], pos + lookahead)
        note_h = 12
        for index in range(lo, hi):
            when, lane = self.chart["notes"][index]
            y = judge_y - (when - pos) * self.speed
            if y < -note_h or y > height:
                continue
            x0 = margin + lane * lane_w + 3
            x1 = margin + (lane + 1) * lane_w - 3
            canvas.create_rectangle(x0, y - note_h / 2, x1, y + note_h / 2,
                                    fill=LANE_COLORS[lane], outline="")
            canvas.create_rectangle(x0, y - note_h / 2, x1, y - note_h / 2 + 2,
                                    fill=LANE_DARK[lane], outline="")

        self._update_labels(pos)
        self.draw_seek(pos)

    def _time_at(self, beat: float) -> float:
        segments = self.chart["segments"]
        segment = segments[0]
        for candidate in segments:
            if candidate[0] <= beat + 1e-9:
                segment = candidate
            else:
                break
        return segment[1] + (beat - segment[0]) * 60.0 / segment[2]

    def _update_labels(self, pos: float) -> None:
        pos = max(0.0, min(pos, self.app.player.duration))
        minutes = int(pos // 60)
        seconds = pos - minutes * 60
        self.time_label.configure(text=f"{minutes:02d}:{seconds:06.3f}")
        beat, bpm = self._beat_at(pos)
        measure = int(beat)
        num = int(round((beat - measure) * 8))
        if num >= 8:
            measure += 1
            num = 0
        self.beat_label.configure(text=f"Beat {measure}:{num}/8")
        self.bpm_label.configure(text=f"BPM {bpm:.1f}")

    def draw_seek(self, pos: float | None = None) -> None:
        canvas = self.seek
        width = max(1, canvas.winfo_width())
        height = 28
        if pos is None:
            pos = self._position()
        duration = max(0.01, self.app.player.duration)
        canvas.delete("all")
        canvas.create_rectangle(0, height / 2 - 3, width, height / 2 + 3,
                                fill="#202736", outline="")
        filled = width * min(1.0, max(0.0, pos / duration))
        canvas.create_rectangle(0, height / 2 - 3, filled, height / 2 + 3,
                                fill=ACCENT, outline="")
        canvas.create_oval(filled - 6, height / 2 - 6, filled + 6, height / 2 + 6,
                           fill=GOLD, outline="")
        canvas.create_text(8, 4, text=f"{int(pos // 60):02d}:{pos % 60:06.3f}",
                           fill=DIM, font=("Consolas", 8), anchor="nw")
        canvas.create_text(width - 8, 4,
                           text=f"{int(duration // 60):02d}:{duration % 60:06.3f}",
                           fill=FAINT, font=("Consolas", 8), anchor="ne")

    def _make_click(self):
        if winsound is None:
            return None
        path = os.path.join(tempfile.gettempdir(), "malody_key_click.wav")
        try:
            frames = bytearray()
            rate = 44100
            for index in range(int(rate * 0.025)):
                envelope = math.exp(-index / (rate * 0.006))
                value = int(10000 * envelope * math.sin(2 * math.pi * 1600 * index / rate))
                frames += struct.pack("<h", value)
            with wave.open(path, "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(rate)
                handle.writeframes(bytes(frames))
            return path
        except Exception:
            return None

    def _play_click(self) -> None:
        if winsound is None or not self.click_path:
            return
        try:
            winsound.PlaySound(
                self.click_path,
                winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
        except Exception:
            pass

    def _play_hits(self, start: float, end: float) -> None:
        if not self.sound_on.get():
            return
        offset = self.offset_ms / 1000.0
        lo = bisect.bisect_right(self.chart["times"], start - offset)
        hi = bisect.bisect_right(self.chart["times"], end - offset)
        for _ in range(min(6, max(0, hi - lo))):
            self._play_click()

    def _tick(self) -> None:
        if self.closed:
            return
        try:
            position = self._position()
            if self.app.player.playing and position > self.last_pos:
                self._play_hits(self.last_pos, position)
            self.last_pos = position
            self.draw()
            self._sync_button()
        except tk.TclError:
            return
        self.after(16, self._tick)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.app.player.pause()
        except Exception:
            pass
        self.destroy()
