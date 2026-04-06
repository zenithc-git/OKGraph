from __future__ import annotations

import os
import time
from dataclasses import asdict
from typing import Callable, Optional

from .config import LLMConfig


def make_request_descriptor(cfg: LLMConfig) -> Callable[[str, str], str]:
    """Return a function(system_prompt, user_prompt)->str for chat completion.

    - No API keys are stored in code.
    - Reads OPENAI_API_KEY / OPENAI_BASE_URL if not provided in cfg.
    """
    if not cfg.enabled:
        raise ValueError("LLM is disabled; request function should not be created.")

    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "openai package is required for LLM stage. Install with: pip install 'openvocab-eval[llm]'"
        ) from e

    api_key = cfg.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing OpenAI API key. Set OPENAI_API_KEY env var or pass LLMConfig.api_key."
        )
    base_url = cfg.base_url or os.getenv("OPENAI_BASE_URL")

    client = OpenAI(api_key=api_key, base_url=base_url)

    def _request(system_prompt: str, user_prompt: str) -> str:
        resp = client.chat.completions.create(
            model=cfg.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
        return resp.choices[0].message.content.strip()

    return _request
