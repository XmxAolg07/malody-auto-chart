"""Tempo / beat-grid estimation without external DSP dependencies.

Pipeline
--------
1. Log-mel spectrogram (numpy FFT, no scipy).
2. Onset strength = rectified log-mel spectral flux (full band + low band).
3. Tempo candidates from the autocorrelation of the onset envelope, scored
   with a harmonic comb (1x, 2x, 3x, 4x lag) and a mild tempo prior.
4. Sub-frame refinement of (BPM, phase) by maximising the onset energy that
   falls on the beat grid; the low-band envelope is used as a tie-breaker so
   the grid lands on kicks rather than off-beat hi-hats.

The result is a constant-tempo approximation: good for the vast majority of
4K charts. Variable-BPM charts need segmented analysis (next milestone).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np

N_FFT = 4096
N_MELS = 40
FMIN_HZ = 30.0
HOP_SECONDS = 0.023219954648526078  # ~1024 samples @ 44.1 kHz
LOW_BAND_HZ = 200.0  # "kick" band for phase disambiguation


@dataclass
class TempoCandidate:
    bpm: float
    score: float
    confidence: float


@dataclass
class RhythmResult:
    bpm: float
    offset_ms: float
    confidence: float
    beat_strength: float
    candidates: List[TempoCandidate] = field(default_factory=list)
    beats_ms: List[float] = field(default_factory=list)
    duration_sec: float = 0.0
    hop_sec: float = HOP_SECONDS
    sr: int = 44100
    coverage: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["candidates"] = [asdict(c) for c in self.candidates]
        return d


# ---------------------------------------------------------------------------
# signal front-end
# ---------------------------------------------------------------------------

def _stft_magnitude(y: np.ndarray, n_fft: int, hop: int, chunk: int = 256) -> np.ndarray:
    """|STFT| for a mono float32 signal, computed in chunks to save memory."""
    if y.size < n_fft:
        y = np.pad(y, (0, n_fft - y.size))
    n_frames = 1 + (y.size - n_fft) // hop
    out = np.empty((n_frames, n_fft // 2 + 1), dtype=np.float32)
    window = np.hanning(n_fft).astype(np.float32)
    base = np.arange(n_fft, dtype=np.int64)
    for start in range(0, n_frames, chunk):
        stop = min(n_frames, start + chunk)
        offsets = np.arange(start, stop, dtype=np.int64)[:, None] * hop
        frames = y[offsets + base[None, :]] * window
        out[start:stop] = np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32)
    return out


def _hz_to_mel(f: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel_to_hz(m: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int = N_MELS,
                    fmin: float = FMIN_HZ, fmax: Optional[float] = None) -> np.ndarray:
    fmax = fmax if fmax is not None else sr / 2.0
    mel_edges = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz_edges = _mel_to_hz(mel_edges)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    fb = np.zeros((n_mels, freqs.size), dtype=np.float32)
    for i in range(n_mels):
        lo, center, hi = hz_edges[i], hz_edges[i + 1], hz_edges[i + 2]
        rising = (freqs - lo) / max(center - lo, 1e-9)
        falling = (hi - freqs) / max(hi - center, 1e-9)
        fb[i] = np.clip(np.minimum(rising, falling), 0.0, None)
    return fb


def _normalize(x: np.ndarray) -> np.ndarray:
    std = float(np.std(x))
    return x.astype(np.float64) / (std + 1e-9)


def onset_envelopes(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return (full-band, low-band) onset strength, hop seconds and first-frame time.

    Frame ``i`` is centred at ``t0 + i * hop_sec``; the centre correction matters
    because a 4096-sample window would otherwise report onsets ~45 ms early.
    """
    hop = max(64, int(round(sr * HOP_SECONDS)))
    hop_sec = hop / float(sr)
    n_fft = N_FFT if sr <= 48000 else 8192

    magnitude = _stft_magnitude(y, n_fft, hop)
    fb = _mel_filterbank(sr, n_fft)
    mel_power = magnitude @ fb.T  # (frames, mels)
    log_mel = np.log(mel_power + 1e-10)

    diff = np.diff(log_mel, axis=0)
    diff = np.maximum(diff, 0.0)
    flux_full = diff.sum(axis=1)
    flux_full = np.concatenate(([0.0], flux_full))

    mel_edges = np.linspace(_hz_to_mel(FMIN_HZ), _hz_to_mel(sr / 2.0), N_MELS + 2)
    band_centers = _mel_to_hz(mel_edges[1:-1])
    low_bands = band_centers < LOW_BAND_HZ
    if low_bands.sum() < 1:
        low_bands[:2] = True
    flux_low = diff[:, low_bands].sum(axis=1)
    flux_low = np.concatenate(([0.0], flux_low))

    def detrend(x: np.ndarray, win_sec: float = 1.0) -> np.ndarray:
        w = max(3, int(round(win_sec / hop_sec)))
        kernel = np.ones(w, dtype=np.float64) / w
        local = np.convolve(x, kernel, mode="same")
        return np.maximum(x - local, 0.0)

    return (_normalize(detrend(flux_full)), _normalize(detrend(flux_low)),
            hop_sec, (n_fft / 2.0) / float(sr))


# ---------------------------------------------------------------------------
# tempo candidates
# ---------------------------------------------------------------------------

def _autocorrelation(x: np.ndarray) -> np.ndarray:
    n = x.size
    nfft = 1 << int(np.ceil(np.log2(max(2 * n, 8))))
    spectrum = np.fft.rfft(x, nfft)
    acf = np.fft.irfft(spectrum * np.conj(spectrum), nfft)[:n]
    if acf[0] <= 1e-12:
        return np.zeros_like(acf)
    return acf / acf[0]


def _acf_at(acf: np.ndarray, lag: float) -> float:
    if lag <= 0 or lag >= acf.size - 1:
        return 0.0
    i = int(np.floor(lag))
    frac = lag - i
    return float(acf[i] * (1.0 - frac) + acf[i + 1] * frac)


def tempo_candidates(
    acf: np.ndarray,
    hop_sec: float,
    bpm_min: float = 50.0,
    bpm_max: float = 280.0,
    top: int = 5,
    prior_center: float = 150.0,
    prior_octaves: float = 1.0,
) -> tuple[List[TempoCandidate], float]:
    bpms = np.arange(bpm_min, bpm_max + 1e-9, 0.1)
    score = np.zeros_like(bpms)
    for harmonic, weight in ((1, 1.0), (2, 0.5), (3, 0.33), (4, 0.25)):
        lags = 60.0 * harmonic / (bpms * hop_sec)
        score += weight * np.array([_acf_at(acf, lag) for lag in lags])
    prior = np.exp(-0.5 * (np.log2(bpms / prior_center) / prior_octaves) ** 2)
    score *= prior

    median = float(np.median(score)) + 1e-12
    ratio = float(np.max(score)) / median
    overall_conf = 1.0 - float(np.exp(-max(0.0, ratio - 1.0) / 1.2))

    order = np.argsort(score)[::-1]
    picked: List[int] = []
    for idx in order:
        bpm = float(bpms[idx])
        if all(abs(bpm - bpms[j]) > 0.03 * bpm for j in picked):
            picked.append(int(idx))
        if len(picked) >= top:
            break

    best = float(np.max(score)) + 1e-12
    return [
        TempoCandidate(
            bpm=round(float(bpms[i]), 3),
            score=round(float(score[i]), 4),
            confidence=round(float(np.clip(score[i] / best, 0.0, 1.0)), 4),
        )
        for i in picked
    ], overall_conf


# ---------------------------------------------------------------------------
# (bpm, phase) refinement
# ---------------------------------------------------------------------------

def _grid_score(
    env: np.ndarray,
    env_low: np.ndarray,
    hop_sec: float,
    bpm: float,
    offsets: np.ndarray,
    duration: float,
    t0: float,
) -> tuple[float, float, float, float]:
    """Return (mean onset, offset, captured energy, density).

    ``density`` is the fraction of beat slots that actually carry an onset.
    A half-tempo grid on an 8th-note pattern still captures every onset, but
    half of its slots are empty - density exposes that.
    """
    period = 60.0 / bpm
    max_beats = int(np.floor(duration / period)) + 2
    if max_beats < 4:
        return -1.0, 0.0, 0.0, 0.0
    k = np.arange(max_beats, dtype=np.float64)
    times = offsets[:, None] + k[None, :] * period  # (offsets, beats)
    pos = (times - t0) / hop_sec
    i0 = np.floor(pos).astype(np.int64)
    frac = pos - i0
    valid = (times < duration) & (i0 >= 0) & (i0 + 1 < env.size)
    i0c = np.clip(i0, 0, env.size - 2)

    def interp(e: np.ndarray) -> np.ndarray:
        v = e[i0c] * (1.0 - frac) + e[i0c + 1] * frac
        return np.where(valid, v, 0.0)

    counts = np.maximum(valid.sum(axis=1), 1)
    combined = interp(env) + 0.35 * interp(env_low)
    means = combined.sum(axis=1) / counts
    captured = combined.sum(axis=1)
    j = int(np.argmax(means))
    vals = combined[j][valid[j]]
    if vals.size:
        threshold = 0.30 * float(vals.max())
        density = float((vals > threshold).sum()) / float(vals.size)
    else:
        density = 0.0
    return float(means[j]), float(offsets[j]), float(captured[j]), density


def refine_beat_grid(
    env: np.ndarray,
    env_low: np.ndarray,
    hop_sec: float,
    bpm0: float,
    duration: float,
    t0: float,
) -> tuple[float, float, float, float, float, float]:
    """Return (bpm, offset, beat_strength, mean score, captured, density)."""
    best = (-1.0, bpm0, 0.0, 0.0, 0.0)

    # coarse: +-1.5 % in BPM, full phase sweep
    for bpm in np.linspace(bpm0 * 0.985, bpm0 * 1.015, 49):
        period = 60.0 / bpm
        offsets = np.linspace(0.0, period, 48, endpoint=False)
        score, offset, captured, density = _grid_score(
            env, env_low, hop_sec, bpm, offsets, duration, t0)
        if score > best[0]:
            best = (score, float(bpm), offset, captured, density)

    # fine: tighten BPM around the coarse winner, then sweep phase again
    bpm_best = best[1]
    period = 60.0 / bpm_best
    for _ in range(2):
        step = bpm_best * 0.0007
        for bpm in np.linspace(bpm_best - step * 3, bpm_best + step * 3, 25):
            phase_center = best[2]
            offsets = np.linspace(phase_center - period / 40,
                                  phase_center + period / 40, 41)
            score, offset, captured, density = _grid_score(
                env, env_low, hop_sec, bpm, offsets, duration, t0)
            if score > best[0]:
                best = (score, float(bpm), offset, captured, density)
        bpm_best = best[1]
        period = 60.0 / bpm_best

    bpm, offset, score, captured, density = best[1], best[2], best[0], best[3], best[4]
    period = 60.0 / bpm
    offset = offset % period
    n_beats = int(np.floor((duration - offset) / period)) + 1
    if n_beats >= 1:
        idx = np.clip(np.round((offset + np.arange(n_beats) * period - t0) / hop_sec).astype(int),
                      0, env.size - 1)
        beat_strength = float(env[idx].mean() / (env.mean() + 1e-9))
    else:
        beat_strength = 0.0
    return bpm, offset, beat_strength, score, captured, density


def _prior(bpm: float, center: float, octaves: float) -> float:
    return float(np.exp(-0.5 * (np.log2(bpm / center) / octaves) ** 2))


def select_tempo(
    env: np.ndarray,
    env_low: np.ndarray,
    hop_sec: float,
    t0: float,
    duration: float,
    acf: np.ndarray,
    bpm_min: float,
    bpm_max: float,
    prior_center: float,
    prior_octaves: float,
    max_candidates: int = 6,
    density_weight: float = 0.5,
) -> tuple[float, float, float, float, float]:
    """Pick the metrical level that best explains the onsets.

    A pure autocorrelation peak often lands on 1/2 or 2/3 of the notated BPM
    (octave and dotted-tempo errors).  The grid coverage of a candidate - the
    fraction of onset energy that actually falls on its beat grid - separates
    those cases: a half-tempo grid covers at most half of the 8th-note events.
    Coverage is combined with a broad tempo prior (rhythm-game charts cluster
    in the 120-250 range).
    """
    candidates, _conf = tempo_candidates(
        acf, hop_sec, bpm_min, bpm_max,
        top=max_candidates,
        prior_center=prior_center, prior_octaves=prior_octaves,
    )
    if not candidates:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    total_energy = float(env.sum() + 0.35 * env_low.sum()) + 1e-9
    best = None
    for cand in candidates:
        bpm, offset, strength, score, captured, density = refine_beat_grid(
            env, env_low, hop_sec, cand.bpm, duration, t0
        )
        coverage = captured / total_energy
        density_factor = (1.0 - density_weight) + density_weight * density
        rank = coverage * density_factor * _prior(bpm, prior_center, max(prior_octaves, 0.35))
        row = (rank, bpm, offset, strength, score, coverage)
        if best is None or row[0] > best[0]:
            best = row

    _rank, bpm, offset, strength, score, coverage = best
    return bpm, offset, strength, score, coverage


def _candidate_table(
    env: np.ndarray,
    env_low: np.ndarray,
    hop_sec: float,
    t0: float,
    duration: float,
    total_energy: float,
    bpm: float,
    bpm_min: float,
    bpm_max: float,
) -> List[TempoCandidate]:
    """Metrical alternatives around the winner, scored by grid coverage.

    Includes 1/2, 2/3, 1, 3/2 and 2 times the selected BPM, so a user who
    disagrees with the octave can switch with one click later.
    """
    rows: List[list] = []

    def add(candidate_bpm: float, cov: float) -> None:
        for row in rows:
            if abs(row[0] - candidate_bpm) <= 0.02 * max(row[0], candidate_bpm):
                if cov > row[1]:
                    row[0], row[1] = candidate_bpm, cov
                return
        rows.append([candidate_bpm, cov])

    for ratio in (0.5, 2.0 / 3.0, 1.0, 1.5, 2.0):
        target = bpm * ratio
        if not (bpm_min <= target <= bpm_max):
            continue
        refined, _off, _strength, _score, captured, _density = refine_beat_grid(
            env, env_low, hop_sec, target, duration, t0
        )
        add(refined, captured / total_energy)

    rows.sort(key=lambda r: -r[1])
    best_cov = rows[0][1] + 1e-9 if rows else 1.0
    return [
        TempoCandidate(
            bpm=round(float(b), 3),
            score=round(float(c), 4),
            confidence=round(float(np.clip(c / best_cov, 0.0, 1.0)), 4),
        )
        for b, c in rows
    ]


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def estimate_rhythm(
    y: np.ndarray,
    sr: int,
    bpm_min: float = 50.0,
    bpm_max: float = 280.0,
    prior_center: float = 150.0,
    prior_octaves: float = 1.0,
    density_weight: float = 0.5,
) -> RhythmResult:
    duration = float(y.size) / float(sr)
    if duration < 1.0 or y.size < 1024:
        return RhythmResult(bpm=0.0, offset_ms=0.0, confidence=0.0,
                            beat_strength=0.0, duration_sec=duration, sr=sr)

    env, env_low, hop_sec, t0 = onset_envelopes(y, sr)
    acf = _autocorrelation(env)
    acf_candidates, acf_conf = tempo_candidates(
        acf, hop_sec, bpm_min, bpm_max,
        prior_center=prior_center, prior_octaves=prior_octaves,
    )
    if not acf_candidates:
        return RhythmResult(bpm=0.0, offset_ms=0.0, confidence=0.0,
                            beat_strength=0.0, duration_sec=duration, sr=sr)

    bpm, offset, beat_strength, _grid, coverage = select_tempo(
        env, env_low, hop_sec, t0, duration, acf,
        bpm_min, bpm_max, prior_center, prior_octaves,
        density_weight=density_weight,
    )
    if not bpm:
        return RhythmResult(bpm=0.0, offset_ms=0.0, confidence=0.0,
                            beat_strength=0.0, duration_sec=duration, sr=sr)

    total_energy = float(env.sum() + 0.35 * env_low.sum()) + 1e-9
    candidates = _candidate_table(
        env, env_low, hop_sec, t0, duration, total_energy,
        bpm, bpm_min, bpm_max,
    )
    # keep the selected tempo first, then alternatives by coverage
    candidates.sort(key=lambda c: (abs(c.bpm - bpm) > 0.01 * bpm, -c.score))
    period = 60.0 / bpm
    beats = offset + np.arange(int(np.floor((duration - offset) / period)) + 1) * period

    phase_conf = 1.0 - float(np.exp(-max(0.0, beat_strength - 1.0) / 2.0))
    cover_conf = float(np.clip(coverage / 0.6, 0.0, 1.0))
    confidence = float(np.clip(0.45 * acf_conf + 0.25 * phase_conf + 0.30 * cover_conf,
                               0.0, 1.0))

    return RhythmResult(
        bpm=round(float(bpm), 3),
        offset_ms=round(float(offset) * 1000.0, 2),
        confidence=round(confidence, 4),
        beat_strength=round(float(beat_strength), 4),
        candidates=candidates,
        beats_ms=[round(float(t) * 1000.0, 2) for t in beats],
        duration_sec=round(duration, 3),
        hop_sec=round(hop_sec, 6),
        sr=int(sr),
        coverage=round(float(coverage), 4),
    )
