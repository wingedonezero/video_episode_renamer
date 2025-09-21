"""
core/pipeline.py - Main matching pipeline orchestrator
"""

from pathlib import Path
from typing import List, Dict, Generator, Optional
from dataclasses import dataclass

@dataclass
class MatchConfig:
    mode: str = "correlation"
    language: Optional[str] = None
    confidence_threshold: float = 0.75

class MatchingPipeline:
    """Orchestrates the matching process using selected matcher"""

    def __init__(self, cache, config):
        self.cache = cache
        self.config = config
        self._mode = "correlation"
        self._language = None
        self._threshold = 0.75
        self._running = False
        self._matcher = None

    def set_mode(self, mode: str):
        self._mode = mode

    def set_language(self, language: Optional[str]):
        self._language = language.lower() if language else None

    def set_threshold(self, threshold: float):
        self._threshold = threshold

    def stop(self):
        self._running = False
        if self._matcher:
            self._matcher.stop()

    def match(self, references: List[Path], remuxes: List[Path]) -> Generator[Dict, None, None]:
        """
        Main matching logic with elimination strategy
        Yields progress updates and match results
        """
        self._running = True

        # Get the appropriate matcher
        self._matcher = self._get_matcher()
        if not self._matcher:
            yield {'type': 'progress', 'message': 'Invalid matcher mode', 'value': 0}
            return

        # Prepare for matching
        unmatched_remuxes = list(remuxes)
        total_refs = len(references)

        yield {'type': 'progress', 'message': f'Starting {self._mode} matching...', 'value': 0}

        for idx, ref_path in enumerate(references):
            if not self._running:
                break

            progress = int((idx / total_refs) * 90)
            yield {
                'type': 'progress',
                'message': f'Processing {ref_path.name} ({idx+1}/{total_refs})',
                'value': progress
            }

            # Skip if no unmatched files left
            if not unmatched_remuxes:
                continue

            # Find best match from remaining pool
            best_match = None
            best_score = 0
            best_info = ""

            for remux_path in unmatched_remuxes:
                if not self._running:
                    break

                # Quick pre-filter based on duration if available
                if not self._should_compare(ref_path, remux_path):
                    continue

                # Compare using selected matcher
                score, info = self._matcher.compare(ref_path, remux_path, self._language)

                if score > best_score:
                    best_score = score
                    best_match = remux_path
                    best_info = info

            # Process the best match if above threshold
            if best_match and best_score >= self._threshold:
                yield {
                    'type': 'match',
                    'data': {
                        'remux_path': str(best_match),
                        'reference_path': str(ref_path),
                        'confidence': best_score,
                        'info': best_info
                    }
                }
                unmatched_remuxes.remove(best_match)
            else:
                # Report low confidence match but don't consume from pool
                if best_match:
                    yield {
                        'type': 'match',
                        'data': {
                            'remux_path': str(best_match),
                            'reference_path': str(ref_path),
                            'confidence': best_score,
                            'info': f"Low confidence: {best_info}"
                        }
                    }

        # Report remaining unmatched files
        for remux_path in unmatched_remuxes:
            yield {
                'type': 'match',
                'data': {
                    'remux_path': str(remux_path),
                    'reference_path': None,
                    'confidence': 0.0,
                    'info': 'No match found'
                }
            }

        yield {'type': 'progress', 'message': 'Matching complete', 'value': 100}

    def _get_matcher(self):
        """Get the appropriate matcher instance based on mode"""
        if self._mode == "correlation":
            from matchers.audio.correlation import CorrelationMatcher
            return CorrelationMatcher(self.cache, self.config)

        elif self._mode == "chromaprint":
            from matchers.audio.chromaprint import ChromaprintMatcher
            return ChromaprintMatcher(self.cache, self.config)

        elif self._mode == "mfcc":
            from matchers.audio.mfcc import MFCCMatcher
            return MFCCMatcher(self.cache, self.config)

        elif self._mode == "panako":
            from matchers.audio.panako import PanakoMatcher
            return PanakoMatcher(self.cache, self.config)

        elif self._mode == "phash":
            from matchers.video.phash import PerceptualHashMatcher
            return PerceptualHashMatcher(self.cache, self.config)

        elif self._mode == "scene":
            from matchers.video.scene import SceneDetectionMatcher
            return SceneDetectionMatcher(self.cache, self.config)

        return None

    def _should_compare(self, ref_path: Path, remux_path: Path) -> bool:
        """Quick pre-filter based on file properties"""
        # Get durations from cache
        ref_duration = self.cache.get_duration(ref_path)
        remux_duration = self.cache.get_duration(remux_path)

        if ref_duration and remux_duration:
            # Skip if durations differ by more than 5 seconds
            if abs(ref_duration - remux_duration) > 5.0:
                return False

        return True
