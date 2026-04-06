from __future__ import annotations

import json
import logging
from typing import Dict, List, Sequence, Tuple

logger = logging.getLogger(__name__)


def _clean_name(name: str) -> str:
    """Remove leading English articles a/an/the (case-insensitive)."""
    name_strip = str(name).strip()
    lower = name_strip.lower()
    if lower.startswith("a "):
        return name_strip[2:].strip()
    if lower.startswith("an "):
        return name_strip[3:].strip()
    if lower.startswith("the "):
        return name_strip[4:].strip()
    return name_strip


def load_prompts(
    json_path: str,
) -> Tuple[
    List[str],
    List[int],
    List[List[str]],
    List[List[int]],
    List[str],
]:
    """Load prompt definitions from a JSON file.

    Returns
    -------
    prompts:
        Flat list of full prompts (class prompt + optional attribute prompts).
    line_to_class:
        Same length as prompts; maps each prompt line to its class index.
    gt_attrs_by_class:
        Per-class list of raw attribute phrases, with '__CLS__' inserted at index 0.
    gt_attr_prompt_indices_by_class:
        Per-class list of indices into `prompts` corresponding to that class's
        (class prompt + attribute prompts).
    classnames:
        List of cleaned class names.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    prompts: List[str] = []
    line_to_class: List[int] = []
    gt_attrs_by_class: List[List[str]] = []
    gt_attr_prompt_indices_by_class: List[List[int]] = []
    classnames: List[str] = []

    def _to_str(x) -> str:
        return str(x).strip()

    for class_idx, row in enumerate(raw_data):
        if isinstance(row, str):
            class_name = _to_str(row)
            attrs_raw: List[str] = []
        elif isinstance(row, list):
            if len(row) == 0:
                class_name = f"class_{class_idx}"
                attrs_raw = []
            else:
                class_name = _to_str(row[0])
                attrs_raw = [_to_str(a) for a in row[1:] if isinstance(a, str) and _to_str(a)]
        elif isinstance(row, dict):
            class_name = _to_str(row.get("name", f"class_{class_idx}"))
            attrs = row.get("attributes", [])
            if isinstance(attrs, list):
                attrs_raw = [_to_str(a) for a in attrs if isinstance(a, str) and _to_str(a)]
            else:
                attrs_raw = []
        else:
            class_name = f"class_{class_idx}"
            attrs_raw = []

        class_name = _clean_name(class_name)
        classnames.append(class_name)

        # ground-truth attribute phrases include sentinel
        gt_attrs_by_class.append(["__CLS__"] + attrs_raw[:])

        # class prompt
        cls_prompt_idx = len(prompts)
        prompts.append(f"a photo of a {class_name}")
        line_to_class.append(class_idx)

        attr_prompt_indices = [cls_prompt_idx]
        if attrs_raw:
            # first attr uses "is", rest use "has"
            first_attr_idx = len(prompts)
            prompts.append(f"a photo of a {class_name} which is {attrs_raw[0]}")
            line_to_class.append(class_idx)
            attr_prompt_indices.append(first_attr_idx)

            for a in attrs_raw[1:]:
                this_idx = len(prompts)
                prompts.append(f"a photo of a {class_name} which has {a}")
                line_to_class.append(class_idx)
                attr_prompt_indices.append(this_idx)

        gt_attr_prompt_indices_by_class.append(attr_prompt_indices)

    logger.info("Loaded %d prompts for %d classes.", len(prompts), len(classnames))
    return prompts, line_to_class, gt_attrs_by_class, gt_attr_prompt_indices_by_class, classnames
