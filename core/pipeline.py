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

    def __init__(self, cache, config, app_data_dir: Path):
        self.cache = cache
        self.config = config
        self.app_data_dir = app_data_dir
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
        self._running = True
        self._matcher = self._get_matcher()
        if not self._matcher:
            yield {'type': 'progress', 'message': 'Invalid matcher mode', 'value': 0}
            return

        unmatched_remuxes = list(remuxes)
        total_refs = len(references)

        yield {'type': 'progress', 'message': f'Starting {self._mode} matching...', 'value': 0}

        for idx, ref_path in enumerate(references):
            if not self._running or not unmatched_remuxes:
                break

            progress = int((idx / total_refs) * 95) # Leave room for final step
            yield {
                'type': 'progress',
                'message': f'Processing {ref_path.name} ({idx+1}/{total_refs})',
                'value': progress
            }

            best_match_for_ref = {'remux_path': None, 'score': 0, 'info': ''}

            for remux_path in unmatched_remuxes:
                if not self._running:
                    break

                if not self._should_compare(ref_path, remux_path):
                    continue

                score, info = self._matcher.compare(ref_path, remux_path, self._language)

                if score > best_match_for_ref['score']:
                    best_match_for_ref['score'] = score
                    best_match_for_ref['remux_path'] = remux_path
                    best_match_for_ref['info'] = info

            best_remux = best_match_for_ref['remux_path']
            best_score = best_match_for_ref['score']

            if best_remux and best_score >= self._threshold:
                yield {
                    'type': 'match',
                    'data': {
                        'remux_path': str(best_remux),
                        'reference_path': str(ref_path),
                        'confidence': best_score,
                        'info': best_match_for_ref['info']
                    }
                }
                unmatched_remuxes.remove(best_remux)
            elif best_remux:
                yield {
                    'type': 'match',
                    'data': {
                        'remux_path': str(best_remux),
                        'reference_path': str(ref_path),
                        'confidence': best_score,
                        'info': f"Low confidence: {best_match_for_ref['info']}"
                    }
                }

        for remux_path in unmatched_remuxes:
            yield {
                'type': 'match',
                'data': {
                    'remux_path': str(remux_path),
                    'reference_path': None,
                    'confidence': 0.0,
                    'info': 'No suitable match found'
                }
            }
        yield {'type': 'progress', 'message': 'Matching complete', 'value': 100}

    def _get_matcher(self):
        if self._mode == "correlation":
            from matchers.audio.correlation import CorrelationMatcher
            return CorrelationMatcher(self.cache, self.config, self.app_data_dir)
        elif self._mode == "chromaprint":
            from matchers.audio.chromaprint import ChromaprintMatcher
            return ChromaprintMatcher(self.cache, self.config, self.app_data_dir)
        elif self._mode == "mfcc":
            from matchers.audio.mfcc import MFCCMatcher
            return MFCCMatcher(self.cache, self.config, self.app_data_dir)
        elif self._mode == "panako":
            from matchers.audio.panako import PanakoMatcher
            return PanakoMatcher(self.cache, self.config, self.app_data_dir)
        elif self._mode == "phash":
            from matchers.video.phash import PerceptualHashMatcher
            return PerceptualHashMatcher(self.cache, self.config, self.app_data_dir)
        elif self._mode == "scene":
            from matchers.video.scene import SceneDetectionMatcher
            return SceneDetectionMatcher(self.cache, self.config, self.app_data_dir)
        return None

    def _should_compare(self, ref_path: Path, remux_path: Path) -> bool:
        ref_duration = self.cache.get_duration(ref_path)
        remux_duration = self.cache.get_duration(remux_path)
        if ref_duration and remux_duration:
            if abs(ref_duration - remux_duration) > 5.0:
                return False
        return True
