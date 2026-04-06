from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import compute_entropy


def flatten_prompt_list(prompts_by_class_id: Sequence[Sequence[str]]) -> Tuple[List[str], Dict[int, List[int]]]:
    """Flatten per-class prompt lists into a single list and map class->indices."""
    all_prompts: List[str] = []
    class_prompt_map: Dict[int, List[int]] = {}

    for class_id, prompts in enumerate(prompts_by_class_id):
        if not prompts:
            continue
        start_idx = len(all_prompts)
        all_prompts.extend(list(prompts))
        class_prompt_map[class_id] = list(range(start_idx, start_idx + len(prompts)))

    return all_prompts, class_prompt_map


def build_zeroshot_classname_prompts(
    classnames: Sequence[str],
    model,
    device: str,
):
    """Build zero-shot prompts using only class names (no attributes)."""
    import clip  # type: ignore

    prompts = list(map(str, classnames))
    tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens).to(device)
        feats = feats / feats.norm(dim=-1, keepdim=True)

    line_to_class = list(range(len(classnames)))
    class_prompt_map = {i: [i] for i in range(len(classnames))}
    return prompts, line_to_class, class_prompt_map, feats


def _make_attr_prompts_for_class(class_name: str, attrs: Sequence[str]) -> List[str]:
    """Convert raw attribute phrases into full CLIP prompts."""
    prompts: List[str] = []

    # sentinel -> class prompt
    for a in attrs:
        if isinstance(a, str) and a.strip() == "__CLS__":
            prompts.append(f"a photo of a {class_name}")

    # attributes
    attr_pos = 0
    for attr in attrs:
        if not isinstance(attr, str):
            continue
        s = attr.strip()
        if s == "" or s == "__CLS__":
            continue
        if attr_pos == 0:
            prompts.append(f"a photo of a {class_name} which is {s}")
        else:
            prompts.append(f"a photo of a {class_name} which has {s}")
        attr_pos += 1

    return prompts


def refresh_all_buffer_cache(
    all_buffer: List[List[str]] | None,
    classnames: Sequence[str],
    model,
    device: str,
    text_dim: int,
):
    """Flatten + encode `all_buffer` into (flat_prompts, class_prompt_map, text_features)."""
    import clip  # type: ignore

    if not all_buffer:
        return [], {}, torch.zeros((0, text_dim), device=device)

    if len(classnames) < len(all_buffer):
        classnames = list(classnames) + [f"class_{i}" for i in range(len(classnames), len(all_buffer))]

    all_flat_prompts: List[str] = []
    all_class_prompt_map: Dict[int, List[int]] = {}

    for c, attrs in enumerate(all_buffer):
        class_name = classnames[c] if c < len(classnames) else f"class_{c}"
        if not isinstance(attrs, (list, tuple)):
            attrs = []
        prompts_c = _make_attr_prompts_for_class(class_name, list(attrs))
        if not prompts_c:
            all_class_prompt_map[c] = []
            continue
        start = len(all_flat_prompts)
        all_flat_prompts.extend(prompts_c)
        all_class_prompt_map[c] = list(range(start, start + len(prompts_c)))

    if not all_flat_prompts:
        all_text_features = torch.zeros((0, text_dim), device=device)
    else:
        tokens = clip.tokenize(all_flat_prompts).to(device)
        with torch.no_grad():
            feats = model.encode_text(tokens).to(device)
            feats /= feats.norm(dim=-1, keepdim=True)
        all_text_features = feats

    return all_flat_prompts, all_class_prompt_map, all_text_features


def update_attribute_buffer(
    shared_buffer: List[List[str]],
    best_prompts: List[List[str]],
    image_features: torch.Tensor,
    model,
    device: str,
    classnames: Sequence[str],
    max_per_class: int = 9,
    entropy_temperature: float = 0.04574,
):
    """Update per-class attribute buffer using entropy + KL heuristic (ported from original script)."""
    import clip  # type: ignore

    all_prompts, class_prompt_map = flatten_prompt_list(best_prompts)
    total_classes = len(shared_buffer)
    classnames = list(classnames)

    for cls_id, prompt_indices in class_prompt_map.items():
        if not prompt_indices:
            continue

        for prompt_idx in prompt_indices:
            new_attr = all_prompts[prompt_idx]
            old_attr = shared_buffer[cls_id]
            attr_list = old_attr.copy()

            if len(attr_list) == 0:
                shared_buffer[cls_id] = [new_attr]
                continue

            # Step 1: entropy per existing attribute
            entropy_vals = []
            for attr in attr_list:
                prompt_texts = [attr if i == cls_id else classnames[i] for i in range(total_classes)]
                tokens = clip.tokenize(prompt_texts).to(device)
                with torch.no_grad():
                    text_feats = model.encode_text(tokens)
                    text_feats = F.normalize(text_feats, dim=-1)
                    logits = image_features @ text_feats.T
                    ent = compute_entropy(logits, temperature=entropy_temperature)
                entropy_vals.append(ent.mean().item())
            p_ent = np.array(entropy_vals)

            # Step 2: KL divergence against baseline distribution Q (baseline uses attr_list[0])
            with torch.no_grad():
                baseline_prompt = [attr_list[0] if i == cls_id else classnames[i] for i in range(total_classes)]
                tokens = clip.tokenize(baseline_prompt).to(device)
                text_feats = model.encode_text(tokens)
                text_feats = F.normalize(text_feats, dim=-1)
                logits_baseline = image_features @ text_feats.T
                q = torch.softmax(logits_baseline, dim=-1)

            kl_vals = []
            eps = 1e-8
            for attr in attr_list:
                prompt_texts = [attr if i == cls_id else classnames[i] for i in range(total_classes)]
                tokens = clip.tokenize(prompt_texts).to(device)
                with torch.no_grad():
                    text_feats = model.encode_text(tokens)
                    text_feats = F.normalize(text_feats, dim=-1)
                    logits = image_features @ text_feats.T
                    p = torch.softmax(logits, dim=-1)
                    kl = torch.sum(p * torch.log((p + eps) / (q + eps)), dim=1)
                kl_vals.append(kl.mean().item())
            kl_vals = np.array(kl_vals)

            # Step 3: keep low-entropy and high-KL prompts
            ent_order = np.argsort(p_ent)  # low -> high
            kl_order = np.argsort(-kl_vals)  # high -> low

            baseline_rank = int(np.where(ent_order == 0)[0][0]) if len(ent_order) else 0
            ent_candidates = ent_order[:baseline_rank]
            kl_candidates = kl_order[:baseline_rank]
            selected_idxs = np.intersect1d(ent_candidates, kl_candidates).tolist()
            if 0 not in selected_idxs:
                selected_idxs.insert(0, 0)

            # Step 4: update buffer
            kept_attrs = [attr_list[i] for i in range(len(attr_list)) if i not in selected_idxs]
            selected_attrs = [attr_list[i] for i in selected_idxs]
            reordered = kept_attrs + selected_attrs

            if len(old_attr) < max_per_class:
                new_buffer = [new_attr] + reordered[: max_per_class - 1]
            else:
                new_buffer = [new_attr] + reordered[1:max_per_class]
            shared_buffer[cls_id] = new_buffer

    return shared_buffer
