# ===========================================
# matchers/audio/chromaprint.py
# ===========================================

import subprocess
import json
from pathlib import Path
from typing import Tuple, Optional
from core.matcher import BaseMatcher

class ChromaprintMatcher(BaseMatcher):
    """Audio fingerprinting using Chromaprint/AcoustID"""

    def compare(self, ref_path: Path, remux_path: Path, language: Optional[str] = None) -> Tuple[float, str]:
        # Get fingerprints
        ref_fp = self._get_fingerprint(ref_path, language)
        remux_fp = self._get_fingerprint(remux_path, language)

        if not ref_fp or not remux_fp:
            return 0.0, "Failed to generate fingerprints"

        # Compare fingerprints
        similarity = self._compare_fingerprints(ref_fp, remux_fp)

        info = f"Chromaprint similarity"
        return similarity, info

    def _get_fingerprint(self, path: Path, language: Optional[str]) -> Optional[str]:
        """Get chromaprint fingerprint for file"""
        # Check cache
        stream_idx = self.get_audio_stream_index(path, language)
        if stream_idx is None:
            return None

        cached = self.cache.get_chromaprint(path, stream_idx)
        if cached:
            return cached

        # Generate fingerprint
        try:
            # Use fpcalc tool
            cmd = [
                'fpcalc',
                '-raw',
                '-length', '120',  # Analyze first 2 minutes
                '-format', 's16le',
                '-rate', '16000',
                '-channels', '1',
                str(path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                return None

            # Parse output
            for line in result.stdout.splitlines():
                if line.startswith('FINGERPRINT='):
                    fingerprint = line.split('=', 1)[1]
                    self.cache.set_chromaprint(path, stream_idx, fingerprint)
                    return fingerprint

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return None

    def _compare_fingerprints(self, fp1: str, fp2: str) -> float:
        """Compare two chromaprint fingerprints"""
        # Convert to integer arrays
        arr1 = [int(x) for x in fp1.split(',')]
        arr2 = [int(x) for x in fp2.split(',')]

        # Align lengths
        min_len = min(len(arr1), len(arr2))
        arr1 = arr1[:min_len]
        arr2 = arr2[:min_len]

        if not arr1:
            return 0.0

        # Compute bit similarity
        matches = 0
        total_bits = 0

        for v1, v2 in zip(arr1, arr2):
            xor = v1 ^ v2
            # Count matching bits (32 bits per integer)
            matches += 32 - bin(xor).count('1')
            total_bits += 32

        return matches / total_bits if total_bits > 0 else 0.0
