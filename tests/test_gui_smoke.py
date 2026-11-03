"""Headless-ish GUI smoke test: load, switch candidate, preview rate, playback.

Run:
    python tests/test_gui_smoke.py [audio]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

import tkinter as tk  # noqa: E402

try:
    from tkinterdnd2 import TkinterDnD
except Exception:
    TkinterDnD = None

import gui  # noqa: E402


def pump(root: tk.Tk, app, timeout: float = 180.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        root.update()
        if not app.busy:
            time.sleep(0.12)
            root.update()
            return
        time.sleep(0.02)
    raise TimeoutError("analysis did not finish")


def wait_until(root: tk.Tk, condition, timeout: float = 120.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        root.update()
        if condition():
            time.sleep(0.08)
            root.update()
            return True
        time.sleep(0.02)
    return False


def main() -> int:
    audio = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "work" / "样例曲目 Silentroom - Nhelv.ogg"
    if not audio.is_file():
        print("SKIP: no sample audio", audio)
        return 0

    root = TkinterDnD.Tk() if TkinterDnD else tk.Tk()
    root.geometry("1200x760")
    app = gui.App(root)

    class DropEvent:
        data = "{" + str(audio) + "}"

    app._on_drop(DropEvent())
    pump(root, app)
    assert app.rhythm is not None, "no rhythm after drop"
    first_bpm = app.rhythm["bpm"]
    assert first_bpm > 0
    print(f"drop+analyze OK: {first_bpm:.3f} BPM, {len(app.rhythm['beats_ms'])} beats")

    candidates = [c["bpm"] for c in app.rhythm["candidates"]]
    alt = next((b for b in candidates if abs(b - first_bpm) > 1.0), None)
    if alt:
        app.apply_bpm(alt)
        pump(root, app)
        assert abs(app.rhythm["bpm"] - alt) < 0.01, "candidate switch failed"
        print(f"candidate switch OK: {first_bpm:.2f} -> {app.rhythm['bpm']:.2f}")

    # preview rate 1.5x: BPM scales, duration shrinks
    base_bpm = app.base_rhythm["bpm"]
    base_duration = app.base_rhythm["duration_sec"]
    app.player.set_volume(0.0)
    app.set_rate(1.5, immediate=True)
    assert wait_until(root, lambda: not app._rate_busy and abs(app.rate - 1.5) < 1e-6), \
        "rate change timed out"
    assert abs(app.rhythm["bpm"] - base_bpm * 1.5) < 0.6, "BPM did not scale with rate"
    assert abs(app.wave.duration - base_duration / 1.5) < 0.5, "waveform duration not scaled"
    print(f"rate OK: x1.50 -> {app.rhythm['bpm']:.3f} BPM, "
          f"duration {base_duration:.1f}s -> {app.wave.duration:.1f}s")

    # playback position advances (silent)
    if app.player.available:
        assert app.player.play(), app.player.error
        time.sleep(0.45)
        pos = app.player.position()
        app.player.pause()
        assert pos > 0.05, f"playback position did not advance: {pos}"
        print(f"playback OK: position {pos:.2f}s after 0.45s")

    # waveform canvas drew something
    assert len(app.wave.find_all()) > 5, "waveform canvas is empty"
    print("waveform canvas OK:", len(app.wave.find_all()), "items")

    app.set_rate(1.0, immediate=True)
    wait_until(root, lambda: not app._rate_busy and abs(app.rate - 1.0) < 1e-6)

    root.destroy()
    print("GUI smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
