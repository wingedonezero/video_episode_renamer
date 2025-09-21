# ===========================================
# matchers/audio/mfcc.py
# ===========================================

import numpy as np
import librosa
from pathlib import Path
from typing import Tuple, Optional
from core.matcher import BaseMatcher
from utils.media import extract_audio_segment

class MFCCMatcher(BaseMatcher):
    """Lightweight MFCC-based audio matching"""

    def compare(self, ref_path: Path, remux_path: Path, language: Optional[str] = None) -> Tuple[float, str]:
        # Get MFCC features
        ref_features = self._get_mfcc_features(ref_path, language)
        remux_features = self._get_mfcc_features(remux_path, language)

        if ref_features is None or remux_features is None:
            return 0.0, "Failed to extract MFCC features"

        # Compare using cosine similarity
        similarity = self._cosine_similarity(ref_features, remux_features)

        return similarity, f"MFCC similarity"

    def _get_mfcc_features(self, path: Path, language: Optional[str]) -> Optional[np.ndarray]:
        """Extract MFCC features from audio"""
        stream_idx = self.get_audio_stream_index(path, language)
        if stream_idx is None:
            return None

        # Check cache
        cached = self.cache.get_mfcc(path, stream_idx)
        if cached is not None:
            return cached

        # Extract audio
        sr = 22050
        audio = self.cache.get_audio(path, stream_idx, sr)
        if audio is None:
            # Extract middle 2 minutes for efficiency
            duration_limit = 120  # seconds
            audio = extract_audio_segment(path, stream_idx, sr, duration_limit=duration_limit)
            if audio is None:
                return None

        # Compute MFCCs
        try:
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            # Average over time to get a single feature vector
            features = np.mean(mfccs, axis=1)

            self.cache.set_mfcc(path, stream_idx, features)
            return features

        except Exception:
            return None

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between vectors"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))
