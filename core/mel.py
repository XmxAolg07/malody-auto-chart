"""Log-mel audio features, inspired by MuG Diffusion's STFT/mel conditioning.

Computes a small mel filterbank over the song and exposes per-time band
energies.  Used as extra conditioning for the row-size model so it can tell
different timbres/registers apart, not just overall loudness.
"""

from __future__ import annotations

import numpy as np


def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(sr: int, n_fft: int, n_bands: int, fmin=30.0, fmax=None):
    fmax = fmax or sr / 2.0
    mel_min, mel_max = _hz_to_mel(fmin), _hz_to_mel(fmax)
    points = np.linspace(mel_min, mel_max, n_bands + 2)
    freqs = _mel_to_hz(points)
    bins = np.floor((n_fft + 1) * freqs / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    bank = np.zeros((n_bands, n_fft // 2 + 1), dtype=np.float32)
    for band in range(n_bands):
        left, center, right = bins[band], bins[band + 1], bins[band + 2]
        if center == left:
            center = left + 1
        if right == center:
            right = center + 1
        for index in range(left, center):
            bank[band, index] = (index - left) / max(1, center - left)
        for index in range(center, right):
            bank[band, index] = (right - index) / max(1, right - center)
    return bank


class MelFeatures:
    def __init__(self, samples, sr: int, n_bands: int = 16,
                 n_fft: int = 1024, hop: int = 256) -> None:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        self.sr = int(sr)
        self.hop_sec = hop / float(sr)
        self.n_bands = n_bands
        if samples.size < n_fft * 2:
            self._bands = np.zeros((1, n_bands), dtype=np.float32)
            return
        count = 1 + (samples.size - n_fft) // hop
        frames = np.lib.stride_tricks.as_strided(
            samples,
            shape=(count, n_fft),
            strides=(samples.strides[0] * hop, samples.strides[0]),
            writeable=False,
        )
        window = np.hanning(n_fft).astype(np.float32)
        spectrum = np.abs(np.fft.rfft(frames * window, axis=1)).astype(np.float32)
        bank = _mel_filterbank(self.sr, n_fft, n_bands)
        mel = spectrum @ bank.T
        mel = np.log1p(mel)
        mean = mel.mean(axis=0, keepdims=True)
        std = mel.std(axis=0, keepdims=True) + 1e-6
        self._bands = ((mel - mean) / std).astype(np.float32)

    def at(self, seconds: float) -> list[float]:
        if self._bands.size == 0:
            return [0.0] * self.n_bands
        index = int(round(max(0.0, seconds) / max(self.hop_sec, 1e-6)))
        index = min(max(index, 0), self._bands.shape[0] - 1)
        return [float(value) for value in self._bands[index]]
