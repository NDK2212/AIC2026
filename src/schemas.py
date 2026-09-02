"""Data types shared across the whole pipeline.

Everything that crosses a module boundary is one of these dataclasses, so the
retrieval paths, the fusion stage and the three task runners all speak the same
language.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Canonical path names.  ``image`` is what the LLM calls the visual modality;
# ``visual`` is what the fusion weights call it.  MODALITY_TO_PATH bridges them.
PATH_OCR = "ocr"
PATH_ASR = "asr"
PATH_DESCRIPTION = "description"
PATH_VISUAL = "visual"
ALL_PATHS = (PATH_OCR, PATH_ASR, PATH_DESCRIPTION, PATH_VISUAL)

MODALITY_TO_PATH = {
    "ocr": PATH_OCR,
    "asr": PATH_ASR,
    "description": PATH_DESCRIPTION,
    "image": PATH_VISUAL,
}
PATH_TO_MODALITY = {v: k for k, v in MODALITY_TO_PATH.items()}


@dataclass(frozen=True)
class Candidate:
    """One (video, frame) hit produced by a retrieval path or by fusion."""

    video_id: str
    frame_id: int
    score: float
    source: str = "fused"        # "ocr" | "asr" | "description" | "visual" | "fused" | "expanded"
    rank: int = 0                # 1-based rank inside the originating path
    extra: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def key(self) -> tuple[str, int]:
        """Identity used for de-duplication and fusion."""
        return (self.video_id, self.frame_id)

    def replace(self, **kwargs: Any) -> "Candidate":
        """Return a copy with the given fields overridden."""
        data: dict[str, Any] = {
            "video_id": self.video_id,
            "frame_id": self.frame_id,
            "score": self.score,
            "source": self.source,
            "rank": self.rank,
            "extra": self.extra,
        }
        data.update(kwargs)
        return Candidate(**data)


@dataclass
class DecomposeResult:
    """Output of the query decomposition LLM call across all modalities."""

    original_query: str
    modalities: list[str] = field(default_factory=list)
    ocr_query: str | None = None
    asr_query: str | None = None
    description_query: str | None = None
    image_query: str | None = None
    ocr_terms: list[str] = field(default_factory=list)
    asr_terms: list[str] = field(default_factory=list)
    description_terms: list[str] = field(default_factory=list)
    image_terms: list[str] = field(default_factory=list)
    modality_weights: dict[str, float] = field(default_factory=dict)
    exact_text: bool = False     # True when the user asked for a literal string

    def query_for(self, path: str) -> str | None:
        """Return the sub-query belonging to a fusion path name."""
        return {
            PATH_OCR: self.ocr_query,
            PATH_ASR: self.asr_query,
            PATH_DESCRIPTION: self.description_query,
            PATH_VISUAL: self.image_query,
        }.get(path)

    def terms_for(self, path: str) -> list[str]:
        """Return the retrieval terms belonging to a fusion path name."""
        return {
            PATH_OCR: self.ocr_terms,
            PATH_ASR: self.asr_terms,
            PATH_DESCRIPTION: self.description_terms,
            PATH_VISUAL: self.image_terms,
        }.get(path, [])

    def path_weights(self) -> dict[str, float]:
        """Modality weights re-keyed from ``image`` to ``visual``."""
        return {
            MODALITY_TO_PATH[m]: w
            for m, w in self.modality_weights.items()
            if m in MODALITY_TO_PATH
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "modalities": self.modalities,
            "ocr_query": self.ocr_query,
            "asr_query": self.asr_query,
            "description_query": self.description_query,
            "image_query": self.image_query,
            "ocr_terms": self.ocr_terms,
            "asr_terms": self.asr_terms,
            "description_terms": self.description_terms,
            "image_terms": self.image_terms,
            "modality_weights": self.modality_weights,
            "exact_text": self.exact_text,
        }


@dataclass
class TrakeStep:
    """One key moment of a TRAKE event sequence."""

    index: int                       # 0-based
    description: str                 # English, visually groundable
    description_local: str | None = None
    decompose: DecomposeResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "description": self.description,
            "description_local": self.description_local,
            "decompose": self.decompose.to_dict() if self.decompose else None,
        }


@dataclass
class TrakePlan:
    """The full temporal decomposition of a TRAKE query."""

    original_query: str
    num_events: int
    steps: list[TrakeStep]
    anchor_index: int = 1
    action: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "action": self.action,
            "num_events": self.num_events,
            "anchor_index": self.anchor_index,
            "steps": [s.to_dict() for s in self.steps],
        }


@dataclass
class TrakeSequence:
    """One candidate answer row for TRAKE: a video plus one frame per step."""

    video_id: str
    frame_ids: list[int]
    total_score: float
    per_step_score: list[float] = field(default_factory=list)
    filled_steps: list[int] = field(default_factory=list)
    anchor_frame: int | None = None

    @property
    def key(self) -> tuple[str, tuple[int, ...]]:
        return (self.video_id, tuple(self.frame_ids))

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "frame_ids": self.frame_ids,
            "total_score": self.total_score,
            "per_step_score": self.per_step_score,
            "filled_steps": self.filled_steps,
            "anchor_frame": self.anchor_frame,
        }


@dataclass
class VQASplit:
    """The scene / question split of a Q&A query."""

    scene_description: str
    question: str
    question_en: str = ""
    expected_answer_type: str = "other"
    evidence_query: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_description": self.scene_description,
            "question": self.question,
            "question_en": self.question_en,
            "expected_answer_type": self.expected_answer_type,
            "evidence_query": self.evidence_query,
        }


class PipelineError(RuntimeError):
    """Base class for every error this codebase raises on purpose."""


class ConfigError(PipelineError):
    """Configuration is missing, malformed or inconsistent with the backend."""


class LLMParseError(PipelineError):
    """The LLM produced something that is not the requested JSON object."""


class EncoderUnavailable(PipelineError):
    """A text encoder could not be loaded; the caller may degrade gracefully."""


class SubmissionError(PipelineError):
    """A submission file violates the competition format."""
