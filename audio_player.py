"""Small PortAudio (sounddevice) playback engine with seek + volume.

The GUI resamples the audio for the current preview rate and hands the
resulting mono buffer to this player; positions are therefore in the
preview timeline (same timeline as the waveform).  If no audio device or
PortAudio is available the player degrades to ``available == False`` and
the GUI disables the transport buttons.
"""

from __future__ import annotations

import threading

import numpy as np

try:  # pragma: no cover - depends on the machine
    import sounddevice as _sd
except Exception:  # pragma: no cover
    _sd = None


class AudioPlayer:
    def __init__(self) -> None:
        self.sd = _sd
        self.stream = None
        self.data = np.zeros(1, dtype=np.float32)
        self.sr = 44100
        self.pos = 0
        self.volume = 0.8
        self.playing = False
        self.error = ""
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self.sd is not None

    @property
    def duration(self) -> float:
        return self.data.size / float(self.sr) if self.sr else 0.0

    # -- lifecycle ------------------------------------------------------
    def load(self, data: np.ndarray, sr: int) -> None:
        self.pause()
        with self._lock:
            self.data = np.ascontiguousarray(np.asarray(data, dtype=np.float32).reshape(-1))
            self.sr = int(sr)
            self.pos = 0
            self.playing = False
        self._close_stream()

    def close(self) -> None:
        self.pause()
        self._close_stream()

    def _close_stream(self) -> None:
        if self.stream is not None:
            try:
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def _ensure_stream(self) -> None:
        if self.stream is not None and int(self.stream.samplerate) == self.sr:
            return
        self._close_stream()
        self.stream = self.sd.OutputStream(
            samplerate=self.sr,
            channels=2,
            dtype="float32",
            blocksize=1024,
            callback=self._callback,
        )

    # -- transport ------------------------------------------------------
    def play(self) -> bool:
        if not self.available:
            self.error = "sounddevice / PortAudio 不可用"
            return False
        try:
            self._ensure_stream()
            if self.pos >= self.data.size:
                self.pos = 0
            self.stream.start()
            self.playing = True
            return True
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)
            self.playing = False
            return False

    def pause(self) -> None:
        self.playing = False
        if self.stream is not None:
            try:
                self.stream.stop()
            except Exception:
                pass

    def stop(self) -> None:
        self.pause()
        with self._lock:
            self.pos = 0

    def toggle(self) -> bool:
        if self.playing:
            self.pause()
            return False
        return self.play()

    def seek(self, seconds: float) -> None:
        with self._lock:
            n = self.data.size
            self.pos = int(max(0, min(n, seconds * self.sr)))

    def position(self) -> float:
        with self._lock:
            return self.pos / float(self.sr) if self.sr else 0.0

    def set_volume(self, value: float) -> None:
        self.volume = float(max(0.0, min(1.0, value)))

    # -- audio callback -------------------------------------------------
    def _callback(self, outdata, frames, _time_info, status) -> None:  # noqa: ANN001
        with self._lock:
            pos = self.pos
            chunk = self.data[pos:pos + frames]
            n = int(chunk.size)
            if n:
                outdata[:n, 0] = chunk * self.volume
                outdata[:n, 1] = chunk * self.volume
            if n < frames:
                outdata[n:, 0] = 0.0
                outdata[n:, 1] = 0.0
            self.pos = pos + n
            if self.pos >= self.data.size:
                self.playing = False
