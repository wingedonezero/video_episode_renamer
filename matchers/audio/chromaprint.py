# ===========================================
# matchers/audio/chromaprint.py
# ===========================================

import subprocess
import json
from pathlib import Path
from typing import Tuple, Optional
from core.matcher import BaseMatcher
from utils.media import get_media_duration

class ChromaprintMatcher(BaseMatcher):
    """Audio fingerprinting using Chromaprint/AcoustID"""

    def __init__(self, cache, config, app_data_dir: Path):
        super().__init__(cache, config, app_data_dir)

    def compare(self, ref_path: Path, remux_path: Path, language: Optional[str] = None) -> Tuple[float, str]:
        ref_fp = self._get_fingerprint(ref_path, language)
        remux_fp = self._get_fingerprint(remux_path, language)

        if not ref_fp or not remux_fp:
            return 0.0, "Failed to generate fingerprint"

        similarity = self._compare_fingerprints(ref_fp, remux_fp)
        info = f"Chromaprint similarity"
        return similarity, info

    def _get_fingerprint(self, path: Path, language: Optional[str]) -> Optional[str]:
        stream_idx = self.get_audio_stream_index(path, language)
        if stream_idx is None:
            return None

        cached = self.cache.get_chromaprint(path, stream_idx)
        if cached:
            return cached

        duration = get_media_duration(path)
        analysis_duration_s = 120
        start_percent = self.config.get('analysis_start_percent', 15) / 100.0
        start_offset_s = (duration * start_percent) if duration else 0

        seek_args = []
        if duration and duration > (start_offset_s + analysis_duration_s):
            seek_args.extend(['-ss', str(start_offset_s)])

        try:
            audio_rate, audio_channels, audio_format = '16000', '1', 's16le'

            ffmpeg_cmd = [
                'ffmpeg', '-nostdin', '-v', 'error',
                *seek_args,
                '-i', str(path),
                '-t', str(analysis_duration_s),
                # --- CORRECTION: Use absolute stream index ---
                '-map', f'0:{stream_idx}',
                '-ac', audio_channels,
                '-ar', audio_rate,
                '-f', audio_format,
                '-'
            ]
            fpcalc_cmd = [
                'fpcalc', '-raw', '-json', '-rate', audio_rate,
                '-channels', audio_channels, '-format', audio_format, '-'
            ]

            p_ffmpeg = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE)
            p_fpcalc = subprocess.Popen(fpcalc_cmd, stdin=p_ffmpeg.stdout, stdout=subprocess.PIPE, text=True)
            p_ffmpeg.stdout.close()

            stdout, _ = p_fpcalc.communicate(timeout=45)

            if p_fpcalc.returncode == 0:
                result = json.loads(stdout)
                fingerprint = result.get('fingerprint')
                if fingerprint:
                    fp_str = ','.join(map(str, fingerprint))
                    self.cache.set_chromaprint(path, stream_idx, fp_str)
                    return fp_str
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
            pass
        return None

    def _compare_fingerprints(self, fp1: str, fp2: str) -> float:
        arr1 = [int(x) for x in fp1.split(',')]
        arr2 = [int(x) for x in fp2.split(',')]
        min_len = min(len(arr1), len(arr2))
        arr1, arr2 = arr1[:min_len], arr2[:min_len]
        if not arr1: return 0.0
        matches, total_bits = 0, 0
        for v1, v2 in zip(arr1, arr2):
            xor = v1 ^ v2
            matches += 32 - bin(xor).count('1')
            total_bits += 32
        return matches / total_bits if total_bits > 0 else 0.0
