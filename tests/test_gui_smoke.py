"""Headless-ish GUI smoke test: load a file, switch BPM candidate, verify state.

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


def pump(root: tk.Tk, app, timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        root.update()
        if not app.busy:
            # let the queue poll run at least once
            time.sleep(0.12)
            root.update()
            return
        time.sleep(0.02)
    raise TimeoutError("analysis did not finish")


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

    # waveform canvas drew something
    assert len(app.wave.find_all()) > 5, "waveform canvas is empty"
    print("waveform canvas OK:", len(app.wave.find_all()), "items")

    root.destroy()
    print("GUI smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
