# ===========================================
# matchers/video/phash.py
# ===========================================

import numpy as np
import imagehash
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, List
from core.matcher import BaseMatcher
from utils.media import extract_frames

class PerceptualHashMatcher(BaseMatcher):
    """Improved perceptual hash video matching"""

    def compare(self, ref_path: Path, remux_path: Path, language: Optional[str] = None) -> Tuple[float, str]:
        # Get video hashes
        ref_hashes = self._get_video_hashes(ref_path)
        remux_hashes = self._get_video_hashes(remux_path)

        if not ref_hashes or not remux_hashes:
            return 0.0, "Failed to extract video frames"

        # Compare sequences allowing for small offsets
        similarity, offset = self._compare_hash_sequences(ref_hashes, remux_hashes)

        info = f"Video pHash, offset={offset}"
        return similarity, info

    def _get_video_hashes(self, path: Path) -> Optional[List]:
        """Extract perceptual hashes from video frames"""
        # Check cache
        cached = self.cache.get_video_hashes(path, "phash")
        if cached:
            return cached

        # Extract frames
        n_frames = 25
        duration = self.cache.get_duration(path)
        if not duration:
            from utils.media import get_media_duration
            duration = get_media_duration(path)
            if duration:
                self.cache.set_duration(path, duration)
            else:
                return None

        # Sample frames from middle 60% of video
        edge_skip = 0.2
        start = duration * edge_skip
        end = duration * (1 - edge_skip)

        timestamps = np.linspace(start, end, n_frames)
        hashes = []

        for ts in timestamps:
            frame = extract_frames(path, [ts])
            if frame:
                try:
                    # Compute both phash and dhash for robustness
                    img = Image.fromarray(frame[0])
                    ph = imagehash.phash(img, hash_size=16)
                    dh = imagehash.dhash(img, hash_size=16)
                    hashes.append((str(ph), str(dh)))
                except Exception:
                    continue

        if hashes:
            self.cache.set_video_hashes(path, "phash", hashes)

        return hashes if len(hashes) > 10 else None

    def _compare_hash_sequences(self, seq1: List, seq2: List) -> Tuple[float, int]:
        """Compare hash sequences with offset tolerance"""
        best_similarity = 0
        best_offset = 0

        # Try different alignments
        for offset in range(-5, 6):
            if offset >= 0:
                s1 = seq1[offset:]
                s2 = seq2
            else:
                s1 = seq1
                s2 = seq2[-offset:]

            # Align to shorter length
            min_len = min(len(s1), len(s2))
            if min_len < 10:
                continue

            s1 = s1[:min_len]
            s2 = s2[:min_len]

            # Compare hashes
            similarities = []
            for (ph1, dh1), (ph2, dh2) in zip(s1, s2):
                # Hamming distance
                ph_sim = 1 - (bin(int(ph1, 16) ^ int(ph2, 16)).count('1') / (len(ph1) * 4))
                dh_sim = 1 - (bin(int(dh1, 16) ^ int(dh2, 16)).count('1') / (len(dh1) * 4))

                # Average of both hash types
                similarities.append((ph_sim + dh_sim) / 2)

            avg_similarity = np.mean(similarities)

            if avg_similarity > best_similarity:
                best_similarity = avg_similarity
                best_offset = offset

        return best_similarity, best_offset
