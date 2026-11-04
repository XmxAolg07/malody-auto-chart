"""Audio energy map used by the chart generator.

* smoothed RMS ratio against the song median (0 = silence, 1 = average),
* 2-second section averages, so generators can thin out quiet sections,
* onset peaks (spectral-flux local maxima) so quiet sections can still hit
  the waveform accents instead of a uniform grid.
"""

from __future__ import annotations

import bisect

import numpy as np

from .bpm import onset_envelopes


class EnergyMap:
    def __init__(self, samples, sr: int) -> None:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        self.sr = int(sr)
        self.duration = samples.size / float(sr) if sr else 0.0

        frame = max(1, int(sr * 0.05))
        count = samples.size // frame
        if count < 4:
            self._ratio = np.ones(1)
            self._peak_times = []
            self._peak_strength = []
            self._env = np.zeros(1)
            self._low = np.zeros(1)
            self._env_hop = 0.02
            self._env_t0 = 0.0
            return
        blocks = samples[: count * frame].reshape(count, frame)
        rms = np.sqrt(np.mean(blocks * blocks, axis=1))
        smooth = max(1, int(0.25 / 0.05))
        kernel = np.ones(smooth) / smooth
        rms = np.convolve(rms, kernel, mode="same")
        positive = rms[rms > 1e-6]
        median = float(np.median(positive)) if positive.size else 1.0
        self._ratio = np.clip(rms / max(median, 1e-9), 0.0, 3.0)
        self._frame_sec = frame / float(sr)

        env, low, hop, t0 = onset_envelopes(samples, sr)
        self._env = env
        self._low = low
        self._env_hop = hop
        self._env_t0 = t0
        if env.size > 4:
            width = max(1, int(0.05 / hop))
            env = np.convolve(env, np.ones(width) / width, mode="same")
            threshold = float(np.percentile(env, 68))
            times, strength = [], []
            last = -1.0
            for i in range(1, env.size - 1):
                if env[i] < threshold:
                    continue
                if not (env[i] > env[i - 1] and env[i] >= env[i + 1]):
                    continue
                t = t0 + i * hop
                if t - last < 0.08:
                    continue
                times.append(float(t))
                strength.append(float(env[i]))
                last = t
            self._peak_times = times
            if strength:
                top = float(np.percentile(strength, 75))
                self._peak_strength = [1.0 if s >= top else 0.5 for s in strength]
            else:
                self._peak_strength = []
        else:
            self._peak_times = []
            self._peak_strength = []

    # -- energy ---------------------------------------------------------
    def ratio_at(self, seconds: float) -> float:
        if self._ratio.size == 0:
            return 1.0
        index = int(seconds / max(self._frame_sec, 1e-6))
        return float(self._ratio[min(max(index, 0), self._ratio.size - 1)])

    def section_ratio(self, seconds: float, block: float = 2.0) -> float:
        start = int(max(0.0, seconds - block) / max(self._frame_sec, 1e-6))
        end = int((seconds + block) / max(self._frame_sec, 1e-6)) + 1
        start = min(start, self._ratio.size)
        end = min(max(end, start + 1), self._ratio.size)
        if start >= self._ratio.size:
            return float(self._ratio[-1])
        return float(np.mean(self._ratio[start:end]))

    # -- peaks ----------------------------------------------------------
    def nearest_peak(self, seconds: float, tolerance: float = 0.12):
        if not self._peak_times:
            return None
        index = bisect.bisect_left(self._peak_times, seconds)
        best = None
        best_dist = tolerance + 1e-9
        for candidate in (index - 1, index):
            if 0 <= candidate < len(self._peak_times):
                dist = abs(self._peak_times[candidate] - seconds)
                if dist <= best_dist:
                    best = candidate
                    best_dist = dist
        if best is None:
            return None
        return self._peak_times[best], self._peak_strength[best]

    def onset_at(self, seconds: float) -> float:
        env = getattr(self, "_env", None)
        if env is None or env.size == 0:
            return 0.0
        index = int(round((seconds - self._env_t0) / max(self._env_hop, 1e-6)))
        return float(env[min(max(index, 0), env.size - 1)])

    def low_at(self, seconds: float) -> float:
        env = getattr(self, "_low", None)
        if env is None or env.size == 0:
            return 0.0
        index = int(round((seconds - self._env_t0) / max(self._env_hop, 1e-6)))
        return float(env[min(max(index, 0), env.size - 1)])

    def peak_density(self, seconds: float, window: float = 1.0) -> float:
        return len(self.peaks_between(seconds - window, seconds + window)) / (2.0 * window)

    def peaks_between(self, start: float, end: float) -> list[float]:
        lo = bisect.bisect_left(self._peak_times, start)
        hi = bisect.bisect_right(self._peak_times, end)
        return self._peak_times[lo:hi]
