"""
core/matcher.py - Base matcher interface
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Optional

class BaseMatcher(ABC):
    """Base class for all matching implementations"""

    def __init__(self, cache, config):
        self.cache = cache
        self.config = config
        self._running = True

    def stop(self):
        """Stop the matching process"""
        self._running = False

    @abstractmethod
    def compare(self, ref_path: Path, remux_path: Path, language: Optional[str] = None) -> Tuple[float, str]:
        """
        Compare two files and return similarity score
        Returns: (confidence 0-1, info string)
        """
        pass

    def get_audio_stream_index(self, path: Path, language: Optional[str] = None) -> Optional[int]:
        """Helper to find the right audio stream"""
        from utils.media import get_stream_info

        streams = self.cache.get_stream_info(path)
        if streams is None:
            streams = get_stream_info(path)
            self.cache.set_stream_info(path, streams)

        if not streams:
            return None

        # If no language specified, return first audio stream
        if not language:
            return 0

        # Find streams matching the language
        candidates = []
        for idx, stream in enumerate(streams):
            stream_lang = stream.get('properties', {}).get('language', '').lower()
            if stream_lang == language:
                title = stream.get('properties', {}).get('track_name', '').lower()
                candidates.append((idx, 'commentary' not in title))

        if not candidates:
            return 0  # Fallback to first stream

        # Prefer non-commentary tracks
        non_commentary = [idx for idx, non_comm in candidates if non_comm]
        if non_commentary:
            return non_commentary[0]

        return candidates[0][0]
