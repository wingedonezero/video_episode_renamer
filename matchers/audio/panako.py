# ===========================================
# matchers/audio/panako.py
# ===========================================

import os
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Tuple, Optional, Dict
from core.matcher import BaseMatcher
from utils.media import extract_audio_to_wav

class PanakoMatcher(BaseMatcher):
    """Panako audio fingerprinting integration"""

    def __init__(self, cache, config):
        super().__init__(cache, config)
        self.panako_jar = config.get('panako_jar')
        self.temp_dir = None
        self.db_initialized = False

    def compare(self, ref_path: Path, remux_path: Path, language: Optional[str] = None) -> Tuple[float, str]:
        # Initialize Panako workspace if needed
        if not self._init_panako():
            return 0.0, "Panako not available"

        # Store reference in Panako DB
        ref_wav = self._prepare_wav(ref_path, language, "ref")
        if not ref_wav:
            return 0.0, "Failed to prepare reference audio"

        if not self._store_in_panako(ref_wav):
            return 0.0, "Failed to store reference"

        # Query with remux
        remux_wav = self._prepare_wav(remux_path, language, "query")
        if not remux_wav:
            return 0.0, "Failed to prepare query audio"

        # Run query
        result = self._query_panako(remux_wav)
        if not result:
            return 0.0, "No match found"

        # Calculate confidence
        confidence = self._calculate_confidence(result)
        info = f"Panako: time_factor={result.get('time_factor', 0):.2f}, score={result.get('score', 0)}"

        return confidence, info

    def _init_panako(self) -> bool:
        """Initialize Panako with isolated DB"""
        if self.db_initialized:
            return True

        # Check for Panako
        if not self.panako_jar or not Path(self.panako_jar).exists():
            return False

        # Create temp workspace
        self.temp_dir = Path(tempfile.mkdtemp(prefix="panako_"))
        self.wav_dir = self.temp_dir / "wav"
        self.wav_dir.mkdir()

        # Set Panako home to temp dir for isolated DB
        os.environ['PANAKO_HOME'] = str(self.temp_dir)

        self.db_initialized = True
        return True

    def _prepare_wav(self, path: Path, language: Optional[str], prefix: str) -> Optional[Path]:
        """Extract audio to WAV for Panako"""
        stream_idx = self.get_audio_stream_index(path, language)
        if stream_idx is None:
            return None

        wav_path = self.wav_dir / f"{prefix}_{path.stem}.wav"

        if extract_audio_to_wav(path, stream_idx, wav_path, sample_rate=22050):
            return wav_path

        return None

    def _store_in_panako(self, wav_path: Path) -> bool:
        """Store audio in Panako DB"""
        try:
            cmd = [
                'java',
                f'-Duser.home={self.temp_dir}',
                '--add-opens=java.base/java.nio=ALL-UNNAMED',
                '-jar', str(self.panako_jar),
                'store',
                'STRATEGY=panako',
                str(wav_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.returncode == 0

        except Exception:
            return False

    def _query_panako(self, wav_path: Path) -> Optional[Dict]:
        """Query Panako DB"""
        try:
            cmd = [
                'java',
                f'-Duser.home={self.temp_dir}',
                '--add-opens=java.base/java.nio=ALL-UNNAMED',
                '-jar', str(self.panako_jar),
                'query',
                'STRATEGY=panako',
                str(wav_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                return None

            # Parse output
            return self._parse_panako_output(result.stdout)

        except Exception:
            return None

    def _parse_panako_output(self, output: str) -> Optional[Dict]:
        """Parse Panako query results"""
        for line in output.splitlines():
            if ';' not in line:
                continue

            parts = line.split(';')
            if len(parts) >= 13:
                try:
                    return {
                        'match_path': parts[5],
                        'score': int(float(parts[9])),
                        'time_factor': float(parts[10].replace('%', '')) / 100,
                        'freq_factor': float(parts[11].replace('%', '')) / 100,
                        'seconds_matched': float(parts[12].replace('%', '')) / 100
                    }
                except (ValueError, IndexError):
                    continue

        return None

    def _calculate_confidence(self, result: Dict) -> float:
        """Calculate confidence from Panako results"""
        # Weight different factors
        coverage = result.get('seconds_matched', 0)
        score_norm = min(result.get('score', 0) / 100, 1.0)

        # Penalize time/frequency stretching
        time_penalty = 1.0 - min(abs(1.0 - result.get('time_factor', 1)), 0.5)
        freq_penalty = 1.0 - min(abs(1.0 - result.get('freq_factor', 1)), 0.5)

        confidence = (
            0.4 * coverage +
            0.3 * score_norm +
            0.2 * time_penalty +
            0.1 * freq_penalty
        )

        return min(max(confidence, 0.0), 1.0)

    def __del__(self):
        """Cleanup temp files"""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
