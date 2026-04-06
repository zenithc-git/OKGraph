from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMConfig:
    """Configuration for the optional LLM attribute-generation stage."""

    enabled: bool = True
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 150
    api_key: Optional[str] = None  # if None, read from env OPENAI_API_KEY
    base_url: Optional[str] = None  # if None, OpenAI default


@dataclass
class EvalConfig:
    """Top-level evaluation configuration."""

    entropy_threshold_class: float = 99.0
    alpha: float = 0.4

    consolidate_every: int = 160
    novelty_theta: float = 0.3
    soft_jaccard_tau: float = 0.7
    max_all_attrs_per_class: int = 17

    topk_per_class: int = 5
    record_topk_activations: bool = True
    prune_every: int = 240
    min_hits_to_keep: int = 80

    # entropy guard for global merge
    enable_entropy_guard: bool = True
    entropy_guard_delta_mode: str = "absolute"  # "absolute" | "relative"
    entropy_guard_delta_threshold: float = 0.0

    # CLIP softmax temperature used by entropy (matches original script default)
    entropy_temperature: float = 0.04574
