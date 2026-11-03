"""Waveform extraction + PNG rendering.

`waveform_peaks` turns an arbitrarily long signal into a fixed-width
min/max/RMS envelope (what a DAW draws).  `render_waveform` paints it with
beat lines, a time ruler and a small info header, using Pillow only.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BG = (14, 16, 21)
PANEL = (20, 23, 30)
GRID = (38, 43, 56)
CENTER = (52, 58, 72)
WAVE_FILL = (46, 102, 148)
WAVE_EDGE = (102, 178, 235)
RMS_FILL = (140, 205, 255)
BEAT = (44, 54, 82)
BEAT_ACCENT = (86, 108, 165)
TEXT = (233, 238, 246)
DIM = (146, 157, 173)
ACCENT = (255, 196, 87)


def waveform_peaks(y: np.ndarray, width: int, peak: float | None = None) -> dict:
    """Reduce ``y`` to ``width`` columns of (min, max, rms).

    ``peak`` fixes the normalisation gain (pass the whole-file peak when
    drawing a zoomed view so the amplitude does not jump while zooming).
    """
    width = max(1, int(width))
    n = int(y.size)
    if n == 0:
        zeros = [0.0] * width
        return {"min": zeros, "max": zeros, "rms": zeros, "width": width, "peak": 1.0}

    data = y.astype(np.float64, copy=False)
    starts = np.linspace(0, n, width, endpoint=False).astype(np.int64)
    counts = np.diff(np.append(starts, n)).astype(np.float64)

    mn = np.minimum.reduceat(data, starts)
    mx = np.maximum.reduceat(data, starts)
    sumsq = np.add.reduceat(data * data, starts)
    rms = np.sqrt(sumsq / np.maximum(counts, 1.0))

    peak = float(peak) if peak else float(max(np.abs(mn).max(), np.abs(mx).max(), 1e-9))
    peak = max(peak, 1e-9)
    return {
        "min": (mn / peak).round(6).tolist(),
        "max": (mx / peak).round(6).tolist(),
        "rms": (rms / peak).round(6).tolist(),
        "width": width,
        "peak": peak,
    }


@lru_cache(maxsize=32)
def _font(size: int) -> ImageFont.FreeTypeFont:
    windir = Path(os.environ.get("WINDIR", r"C:\Windows"))
    candidates = ["msyh.ttc", "msyhbd.ttc", "simhei.ttf", "segoeui.ttf", "arial.ttf"]
    for name in candidates:
        path = windir / "Fonts" / name
        if path.is_file():
            try:
                return ImageFont.truetype(str(path), size)
            except Exception:
                continue
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m, s = divmod(seconds, 60.0)
    if abs(s - round(s)) < 0.02:
        return f"{int(m)}:{int(round(s)):02d}"
    return f"{int(m)}:{s:04.1f}"


def _nice_step(span: float, target: int = 8) -> float:
    raw = span / max(target, 1)
    for step in (0.5, 1, 2, 5, 10, 15, 20, 30, 60, 120, 300, 600):
        if step >= raw:
            return float(step)
    return float(raw)


def render_waveform(
    y: np.ndarray,
    sr: int,
    out_path: str | Path,
    meta: Optional[dict] = None,
    width: int = 2000,
    height: int = 560,
    beat_times_ms: Optional[Iterable[float]] = None,
    span: Optional[float] = None,
    start: float = 0.0,
    title: str = "",
) -> dict:
    """Render a waveform image and return the peak data that was drawn.

    ``span`` limits the drawing to [start, start+span] seconds (zoom view).
    """
    meta = meta or {}
    y = np.asarray(y, dtype=np.float32)
    total_duration = y.size / float(sr)
    if span is None:
        span = total_duration
    span = min(span, total_duration - start) if total_duration > start else 0.0
    lo = int(round(start * sr))
    hi = int(round((start + span) * sr))
    segment = y[max(0, lo):max(0, min(y.size, hi))]
    if segment.size == 0:
        segment = y[:1]

    peaks = waveform_peaks(segment, width)

    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)

    header_h = 86
    ruler_h = 30
    pad = 26
    top = header_h + 10
    bottom = height - ruler_h - 8
    center_y = (top + bottom) / 2.0
    half = max(8.0, (bottom - top) / 2.0 - 6.0)

    # panel
    draw.rectangle([pad - 8, top - 6, width - pad + 8, bottom + 6], fill=PANEL)

    # time grid + labels (drawn under the waveform, labels above)
    step = _nice_step(span)
    t = np.ceil(start / step) * step
    while t <= start + span + 1e-9:
        x = pad + (t - start) / max(span, 1e-9) * (width - 2 * pad)
        draw.line([(x, top), (x, bottom)], fill=GRID, width=1)
        label = _fmt_time(t)
        draw.text((x + 4, bottom + 8), label, font=_font(14), fill=DIM)
        t += step

    # waveform envelope
    xs = pad + (np.arange(width) + 0.5) / width * (width - 2 * pad)
    top_pts = [(float(xs[i]), float(center_y - peaks["max"][i] * half)) for i in range(width)]
    bot_pts = [(float(xs[i]), float(center_y - peaks["min"][i] * half)) for i in range(width)]
    draw.polygon(top_pts + bot_pts[::-1], fill=WAVE_FILL)
    draw.line(top_pts, fill=WAVE_EDGE, width=1)
    draw.line(bot_pts, fill=WAVE_EDGE, width=1)

    # RMS overlay
    rms_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    rms_draw = ImageDraw.Draw(rms_layer)
    rms_top = [(float(xs[i]), float(center_y - peaks["rms"][i] * half)) for i in range(width)]
    rms_bot = [(float(xs[i]), float(center_y + peaks["rms"][i] * half)) for i in range(width)]
    rms_draw.polygon(rms_top + rms_bot[::-1], fill=RMS_FILL + (70,))
    image = Image.alpha_composite(image.convert("RGBA"), rms_layer).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.line([(pad, center_y), (width - pad, center_y)], fill=CENTER, width=1)

    # beat lines on top, only when they are readable (zoom view or sparse bars)
    beats = list(beat_times_ms or [])
    if beats:
        beat_len = (beats[1] - beats[0]) / 1000.0 if len(beats) > 1 else 0.5
        px_per_beat = beat_len / max(span, 1e-9) * (width - 2 * pad)
        if span <= 45.0:
            every, alpha_minor, alpha_major = 1, 62, 92
        elif px_per_beat * 4.0 >= 10.0:
            every, alpha_minor, alpha_major = 4, 26, 34
        else:
            every = 0
        if every:
            overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            odraw = ImageDraw.Draw(overlay)
            end_ms = (start + span) * 1000.0
            start_ms = start * 1000.0
            for idx, t_ms in enumerate(beats):
                if idx % every:
                    continue
                if t_ms < start_ms or t_ms > end_ms:
                    continue
                x = pad + (t_ms / 1000.0 - start) / max(span, 1e-9) * (width - 2 * pad)
                major = (idx % (4 * every) == 0)
                color = (255, 196, 87, alpha_major) if major else (125, 160, 215, alpha_minor)
                odraw.line([(x, top - 6), (x, bottom + 6)], fill=color, width=1)
            image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
            draw = ImageDraw.Draw(image)

    # header
    draw.rectangle([0, 0, width, header_h], fill=(10, 12, 16))
    title_text = (title or meta.get("title") or "").strip()
    if not title_text:
        title_text = Path(str(meta.get("path", ""))).name
    title_text = _ellipsize(draw, title_text, _font(24), width - 2 * pad)
    draw.text((pad, 14), title_text, font=_font(24), fill=TEXT)

    subtitle_bits = []
    if meta.get("bpm"):
        subtitle_bits.append(f"BPM {float(meta['bpm']):.2f}")
    if meta.get("offset_ms") is not None:
        subtitle_bits.append(f"offset {float(meta['offset_ms']):.1f} ms")
    subtitle_bits.append(f"{_fmt_time(total_duration)} total")
    subtitle_bits.append(f"{sr / 1000.0:.1f} kHz")
    if meta.get("channels"):
        subtitle_bits.append("stereo" if int(meta["channels"]) > 1 else "mono")
    if span < total_duration - 1e-6:
        subtitle_bits.insert(0, f"zoom {_fmt_time(start)} - {_fmt_time(start + span)}")
    draw.text((pad, 50), "   ".join(subtitle_bits), font=_font(16), fill=DIM)

    if meta.get("confidence") is not None:
        conf = float(meta["confidence"])
        color = ACCENT if conf >= 0.6 else DIM
        draw.text((width - pad - 170, 50), f"confidence {conf * 100:4.1f}%",
                  font=_font(16), fill=color)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, optimize=True)
    return {"peaks": peaks, "image": str(out_path), "span": [start, start + span]}


def _ellipsize(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> str:
    if not text:
        return ""
    if draw.textlength(text, font=font) <= max_width:
        return text
    suffix = "..."
    while text and draw.textlength(text + suffix, font=font) > max_width:
        text = text[:-1]
    return text + suffix
