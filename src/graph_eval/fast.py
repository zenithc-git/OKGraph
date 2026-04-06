from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import torch

from .metrics import compute_entropy


def run_fast_thinking_entropy_twostep(
    *,
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    class_prompt_map: Dict[int, List[int]],
    entropy_threshold_class: float,
    entropy_temperature: float = 0.04574,
    topk_per_class: int = 3,
    # activation recording
    all_buffer: Optional[List[List[str]]] = None,  # raw phrases, aligned to classes
    record_topk_activations: bool = True,
    activation_counters: Optional[List[defaultdict]] = None,
    num_classes: Optional[int] = None,
):
    """Fast thinking stage.

    - Compute image-to-text similarity for all prompts.
    - Aggregate per-class score by mean of Top-K prompt scores within each class.
    - Compute class-level entropy and gate predictions: if entropy >= threshold => -1.

    Returns:
      predictions [B], stage_record [B], entropy_prompt [B]=0, entropy_class [B], logits_all [B,P], class_logits [B,C]
    """
    device = image_features.device
    bsz = image_features.size(0)

    logits_all = image_features @ text_features.T  # [B, P]

    if num_classes is None:
        num_classes = (max(class_prompt_map.keys()) + 1) if class_prompt_map else 0

    class_logits = torch.zeros((bsz, num_classes), device=device)

    # store selected positions within each class's raw phrase list (for counters)
    topk_attr_pos_per_sample: List[Dict[int, List[int]]] = [dict() for _ in range(bsz)]

    for c in range(num_classes):
        idxs = class_prompt_map.get(c, [])
        if not idxs:
            continue

        s_c = logits_all[:, idxs]  # [B, |A_c|]
        k = min(int(topk_per_class), int(s_c.size(1)))
        if k <= 0:
            continue

        topk_vals, topk_pos = torch.topk(s_c, k=k, dim=1)  # [B,k]
        class_logits[:, c] = topk_vals.mean(dim=1)

        if record_topk_activations:
            for bi in range(bsz):
                topk_attr_pos_per_sample[bi][c] = topk_pos[bi].tolist()

    entropy_class = compute_entropy(class_logits, temperature=entropy_temperature)
    predictions = torch.full((bsz,), -1, dtype=torch.long, device=device)
    stage_record = torch.full((bsz,), 2, dtype=torch.long, device=device)  # 2 = undecided/high entropy

    low_entropy = entropy_class < float(entropy_threshold_class)
    if low_entropy.any():
        preds = torch.argmax(class_logits[low_entropy], dim=1)
        predictions[low_entropy] = preds
        stage_record[low_entropy] = 1

    # activation counting: count top-k positions for all classes, regardless of entropy
    if record_topk_activations and activation_counters is not None and all_buffer is not None:
        num_to_count = min(num_classes, len(all_buffer), len(activation_counters))
        for bi in range(bsz):
            for c in range(num_to_count):
                local_positions = topk_attr_pos_per_sample[bi].get(c)
                if not local_positions:
                    continue
                phrases_c = all_buffer[c]
                for pos in local_positions:
                    if 0 <= pos < len(phrases_c):
                        phrase = phrases_c[pos]
                        activation_counters[c][phrase] += 1

    entropy_prompt = torch.zeros(bsz, device=device)
    return predictions, stage_record, entropy_prompt, entropy_class, logits_all, class_logits
