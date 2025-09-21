"""
core/matcher.py - Base matcher interface
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Optional

class BaseMatcher(ABC):
    """Base class for all matching implementations"""

    def __init__(self, cache, config, app_data_dir: Path):
        self.cache = cache
        self.config = config
        self.app_data_dir = app_data_dir
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
        """Helper to find the right audio stream, now with clearer logic and fallback."""
        from utils.media import get_stream_info

        all_streams = self.cache.get_stream_info(path)
        if all_streams is None:
            all_streams = get_stream_info(path)
            self.cache.set_stream_info(path, all_streams)

        if not all_streams:
            return None

        # Find all available audio streams and their absolute indices
        audio_streams = [
            (i, s) for i, s in enumerate(all_streams) if s.get('codec_type') == 'audio'
        ]

        if not audio_streams:
            print(f"[DEBUG] No audio streams found in {path.name}")
            return None

        # If no language is specified, just use the first audio stream found
        if not language:
            first_audio_stream_index = audio_streams[0][0]
            print(f"[DEBUG] No language specified. Using first audio stream (absolute index: {first_audio_stream_index}) for {path.name}")
            return first_audio_stream_index

        # If language is specified, search for it
        candidates = []
        for abs_index, stream in audio_streams:
            stream_lang = stream.get('tags', {}).get('language', '').lower()
            if stream_lang == language:
                title = stream.get('tags', {}).get('title', '').lower()
                # Prefer non-commentary tracks
                is_non_commentary = 'commentary' not in title
                candidates.append({'index': abs_index, 'non_comm': is_non_commentary})

        # If we found matches for the language
        if candidates:
            # Sort to prioritize non-commentary tracks
            candidates.sort(key=lambda x: x['non_comm'], reverse=True)
            best_match_index = candidates[0]['index']
            print(f"[DEBUG] Found language '{language}'. Using best match (absolute index: {best_match_index}) for {path.name}")
            return best_match_index

        # --- FALLBACK LOGIC ---
        # If no streams matched the language, fall back to the first audio stream
        first_audio_stream_index = audio_streams[0][0]
        print(f"[DEBUG] Language '{language}' not found. FALLING BACK to first audio stream (absolute index: {first_audio_stream_index}) for {path.name}")
        return first_audio_stream_index
