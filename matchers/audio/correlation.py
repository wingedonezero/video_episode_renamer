# ===========================================
# matchers/audio/correlation.py
# ===========================================

import numpy as np
from scipy.signal import correlate
from pathlib import Path
from typing import Tuple, Optional
from core.matcher import BaseMatcher
from utils.media import extract_audio_segment

class CorrelationMatcher(BaseMatcher):
    """Audio correlation matching using SCC/GCC-PHAT"""

    def compare(self, ref_path: Path, remux_path: Path, language: Optional[str] = None) -> Tuple[float, str]:
        # Get stream indices
        ref_idx = self.get_audio_stream_index(ref_path, language)
        remux_idx = self.get_audio_stream_index(remux_path, language)

        if ref_idx is None or remux_idx is None:
            return 0.0, "No audio stream found"

        # Check cache first
        sr = 48000  # Higher sample rate for correlation
        ref_audio = self.cache.get_audio(ref_path, ref_idx, sr)
        remux_audio = self.cache.get_audio(remux_path, remux_idx, sr)

        # Extract if not cached
        if ref_audio is None:
            ref_audio = extract_audio_segment(ref_path, ref_idx, sr)
            if ref_audio is not None:
                self.cache.set_audio(ref_path, ref_idx, sr, ref_audio)

        if remux_audio is None:
            remux_audio = extract_audio_segment(remux_path, remux_idx, sr)
            if remux_audio is not None:
                self.cache.set_audio(remux_path, remux_idx, sr, remux_audio)

        if ref_audio is None or remux_audio is None:
            return 0.0, "Failed to extract audio"

        # Compute correlation on chunks
        return self._chunked_correlation(ref_audio, remux_audio, sr)

    def _chunked_correlation(self, ref_audio: np.ndarray, remux_audio: np.ndarray, sr: int) -> Tuple[float, str]:
        """Compute correlation on multiple chunks"""
        duration = min(len(ref_audio), len(remux_audio)) / sr

        # Settings
        chunk_duration = 15  # seconds
        n_chunks = 10
        min_valid_chunks = 6
        rms_threshold = 0.005

        # Sample chunks throughout the file (skip intro/outro)
        start = duration * 0.1
        end = duration * 0.9
        chunk_starts = np.linspace(start, end, n_chunks)

        valid_correlations = []
        delays_ms = []

        for start_time in chunk_starts:
            if not self._running:
                break

            start_idx = int(start_time * sr)
            end_idx = start_idx + int(chunk_duration * sr)

            if end_idx > len(ref_audio) or end_idx > len(remux_audio):
                continue

            ref_chunk = ref_audio[start_idx:end_idx]
            remux_chunk = remux_audio[start_idx:end_idx]

            # Skip silent chunks
            if self._rms(ref_chunk) < rms_threshold or self._rms(remux_chunk) < rms_threshold:
                continue

            # Compute correlation
            delay_ms, correlation = self._gcc_phat(ref_chunk, remux_chunk, sr)

            if correlation > 0.5:  # Minimum correlation threshold
                valid_correlations.append(correlation)
                delays_ms.append(delay_ms)

        if len(valid_correlations) < min_valid_chunks:
            return 0.0, f"Only {len(valid_correlations)}/{n_chunks} valid chunks"

        # Check delay consistency
        if delays_ms:
            median_delay = np.median(delays_ms)
            mad = np.median(np.abs(np.array(delays_ms) - median_delay))

            if mad > 50:  # Too much variation in delays
                confidence = np.mean(valid_correlations) * 0.5  # Penalize
                return confidence, f"Inconsistent delays (MAD={mad:.0f}ms)"
            else:
                confidence = np.mean(valid_correlations)
                return confidence, f"Delay={median_delay:.0f}ms, {len(valid_correlations)}/{n_chunks} chunks"

        return 0.0, "No valid chunks"

    def _gcc_phat(self, ref: np.ndarray, test: np.ndarray, sr: int) -> Tuple[float, float]:
        """GCC-PHAT cross-correlation"""
        # Zero-mean
        ref = ref - np.mean(ref)
        test = test - np.mean(test)

        # FFT length
        n = int(2 ** np.ceil(np.log2(len(ref) + len(test) - 1)))

        # Cross-correlation via FFT
        REF = np.fft.rfft(ref, n)
        TEST = np.fft.rfft(test, n)

        # Phase transform
        R = REF * np.conj(TEST)
        R_phat = R / (np.abs(R) + 1e-12)

        # Inverse FFT
        cc = np.fft.irfft(R_phat, n)

        # Find peak
        max_shift = int(0.1 * sr)  # Max 100ms shift
        cc = np.concatenate([cc[-max_shift:], cc[:max_shift+1]])
        peak_idx = np.argmax(np.abs(cc))

        # Convert to delay
        delay_samples = peak_idx - max_shift
        delay_ms = (delay_samples / sr) * 1000

        # Normalized correlation at peak
        correlation = np.abs(cc[peak_idx]) / np.sqrt(np.sum(ref**2) * np.sum(test**2) + 1e-12)

        return delay_ms, float(correlation)

    def _rms(self, signal: np.ndarray) -> float:
        """Root mean square of signal"""
        return np.sqrt(np.mean(signal**2))
