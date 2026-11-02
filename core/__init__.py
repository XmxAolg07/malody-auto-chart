"""Audio analysis core for the Malody 4K auto-charting project.

Stage 1 modules:
    audio    - decoding (soundfile / ffmpeg), resampling
    bpm      - BPM, offset and beat grid estimation
    waveform - peak extraction and PNG rendering
    analyze  - one-call pipeline
"""

from .analyze import analyze_file, save_analysis
from .audio import load_audio, probe_audio, resample_audio
from .bpm import estimate_rhythm
from .waveform import render_waveform, waveform_peaks

__all__ = [
    "analyze_file",
    "save_analysis",
    "load_audio",
    "probe_audio",
    "resample_audio",
    "estimate_rhythm",
    "render_waveform",
    "waveform_peaks",
]
