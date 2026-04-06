from __future__ import annotations

import logging
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .buffers import flatten_prompt_list, update_attribute_buffer
from .metrics import compute_entropy

logger = logging.getLogger(__name__)


def call_gpt_generate_discriminative_attributes_with_context(
    *,
    topk_class_names_per_image: Sequence[Sequence[str]],
    topk_class_ids_per_image: Sequence[Sequence[int]],
    total_classes: int,
    request_func: Callable[[str, str], str],
    shared_buffer: Optional[List[List[str]]] = None,
    classnames: Optional[Sequence[str]] = None,
    n_attrs_per_class: int = 2,
    context_max_per_class: int = 6,
    max_retry: int = 6,
    sleep_time: float = 2.0,
) -> List[List[str]]:
    """Generate discriminative attribute prompts for candidate classes using an LLM."""
    import clip  # type: ignore

    max_tokens = 77
    candidate_class_ids = sorted({cid for img_ids in topk_class_ids_per_image for cid in img_ids})

    known_by_class: Dict[int, List[str]] = {}
    if shared_buffer is not None:
        for cid in candidate_class_ids:
            if 0 <= cid < len(shared_buffer):
                existing = list(dict.fromkeys(shared_buffer[cid]))
                known_by_class[cid] = existing[:context_max_per_class] if context_max_per_class > 0 else existing
            else:
                known_by_class[cid] = []
    else:
        for cid in candidate_class_ids:
            known_by_class[cid] = []

    def _name_of(cid: int) -> str:
        if classnames is not None and 0 <= cid < len(classnames):
            return str(classnames[cid])
        return f"class_{cid}"

    system_prompt = f"""
You are a texture visual recognition expert specializing in fine-grained texture classification.
Your task is to generate highly discriminative and observable visual attributes for specific candidate classes,
given the confusion set per image AND the currently known attributes (context) for those candidates.

Hard constraints:
- Output EXACTLY {n_attrs_per_class} attributes for the requested class.
- Each attribute MUST start with: "a photo of a <CLASS_NAME> which has ...".
- Each attribute must be concise (<= one short sentence), visually observable in typical photos, and stable across instances.
- Do NOT repeat or paraphrase any "known attributes" listed in the context (for the target class or other confusion classes).
- Avoid numbers, measurements, or vague qualifiers (e.g., "distinctive", "unique").
- Avoid dataset/annotation language; describe visible traits only.

Prioritize:
1) Unique anatomical structures or proportions
2) Distinctive color patterns/markings/textures
3) Characteristic shapes/sizes of body parts
4) Surface patterns or texture
5) Stable postural/structural adaptations
""".strip()

    description_by_class_id: List[List[str]] = [[] for _ in range(total_classes)]

    for names, ids in zip(topk_class_names_per_image, topk_class_ids_per_image):
        context_lines = []
        for cid in ids:
            cname = _name_of(cid)
            known_list = known_by_class.get(cid, [])
            if known_list:
                context_lines.append(f"- {cname}: {'; '.join(known_list)}")
            else:
                context_lines.append(f"- {cname}: (no known attributes)")
        context_block = "I already know these attributes for involved candidate classes:\n" + "\n".join(context_lines)

        for idx, (cls_name, cls_id) in enumerate(zip(names, ids)):
            other_names = [n for i, n in enumerate(names) if i != idx]
            other_names_str = ", ".join(other_names) if other_names else "(none)"

            prompt = (
                f"{context_block}\n\n"
                f"Now focus on the target class: {cls_name} (id={cls_id}).\n"
                f"Confusing categories in this image: {other_names_str}.\n\n"
                f"Please propose EXACTLY {n_attrs_per_class} novel, maximally discriminative, visually observable attributes\n"
                f"for '{cls_name}' against the confusion set above, NOT redundant with any known attributes.\n"
                f"Format: one attribute per line, each line starts with \\\"a photo of a {cls_name} which has ...\\\".\n"
            )

            retries = 0
            while retries < max_retry:
                try:
                    content = request_func(system_prompt, prompt)
                    if not content:
                        retries += 1
                        time.sleep(sleep_time)
                        continue

                    lines = [line.strip().strip('",') for line in content.split("\n") if line.strip()]
                    lines = lines[:n_attrs_per_class]

                    valid_lines: List[str] = []
                    for line in lines:
                        if not line.lower().startswith("a photo of a"):
                            line = f"a photo of a {cls_name} which has {line}"

                        # CLIP token limit guard
                        ok = False
                        for _ in range(3):
                            tokens = clip.tokenize([line])[0]
                            eos_pos = (tokens == 49407).nonzero(as_tuple=True)[0]
                            token_len = int(eos_pos[0].item() + 1) if len(eos_pos) > 0 else max_tokens
                            if token_len <= max_tokens:
                                valid_lines.append(line)
                                ok = True
                                break
                            # ask for shorter
                            shorter_prompt = prompt + "\n(Hint: last attempt too long; make each line shorter.)"
                            content2 = request_func(system_prompt, shorter_prompt)
                            cand = (content2 or line).split("\n")[0].strip().strip('",')
                            if not cand.lower().startswith("a photo of a"):
                                cand = f"a photo of a {cls_name} which has {cand}"
                            line = cand

                        if not ok:
                            # final fallback: truncate by chars until under token limit
                            for cut in range(len(line), 0, -1):
                                tok = clip.tokenize([line[:cut]])[0]
                                eos_pos = (tok == 49407).nonzero(as_tuple=True)[0]
                                token_len = int(eos_pos[0].item() + 1) if len(eos_pos) > 0 else max_tokens
                                if token_len <= max_tokens:
                                    valid_lines.append(line[:cut])
                                    break

                    description_by_class_id[cls_id].extend(valid_lines)
                    break

                except Exception as e:  # pragma: no cover
                    retries += 1
                    logger.warning("LLM error for class %s: %s (retry %d/%d)", cls_id, e, retries, max_retry)
                    time.sleep(sleep_time)

            if retries == max_retry:
                fallback = f"a photo of a {cls_name} which has visually discernible pattern differences"
                description_by_class_id[cls_id].append(fallback)

    return description_by_class_id


def run_slow_thinking(
    *,
    image_features: torch.Tensor,
    topk_class_names_per_image: Sequence[Sequence[str]],
    topk_class_ids_per_image: Sequence[Sequence[int]],
    model,
    device: str,
    classnames: Sequence[str],
    request_func: Callable[[str, str], str],
    shared_buffer: List[List[str]],
    total_classes: int,
    max_rounds: int = 1,
    n_attrs_per_class: int = 2,
    entropy_temperature: float = 0.04574,
) -> List[List[str]]:
    """Run LLM slow-thinking for uncertain samples and update shared buffer."""
    import clip  # type: ignore

    logits_all_rounds: List[torch.Tensor] = []
    entropy_all_rounds: List[torch.Tensor] = []
    all_gpt_prompts: List[List[List[str]]] = []

    for round_i in range(max_rounds):
        logger.info("LLM slow-thinking round %d/%d", round_i + 1, max_rounds)

        new_attr_candidates = call_gpt_generate_discriminative_attributes_with_context(
            topk_class_names_per_image=topk_class_names_per_image,
            topk_class_ids_per_image=topk_class_ids_per_image,
            total_classes=total_classes,
            request_func=request_func,
            shared_buffer=shared_buffer,
            classnames=classnames,
            n_attrs_per_class=n_attrs_per_class,
            context_max_per_class=5,
        )

        all_gpt_prompts.append(new_attr_candidates)

        all_prompts, new_class_prompt_map = flatten_prompt_list(new_attr_candidates)

        text_tokens = clip.tokenize(all_prompts).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_tokens).to(device)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        bsz, cnum = image_features.size(0), total_classes
        class_logits = torch.zeros((bsz, cnum), device=device)

        for cls_id, indices in new_class_prompt_map.items():
            feats_c = text_features[indices]  # [A_c, D]
            sims = image_features @ feats_c.T  # [B, A_c]
            class_logits[:, cls_id] = sims.mean(dim=1)

        entropy_val = compute_entropy(class_logits, temperature=entropy_temperature)
        logits_all_rounds.append(class_logits)
        entropy_all_rounds.append(entropy_val)

    mean_entropies = [float(e.mean().item()) for e in entropy_all_rounds]
    best_idx = int(mean_entropies.index(min(mean_entropies)))
    best_prompts = all_gpt_prompts[best_idx]
    logger.info("Selected best slow-thinking round %d with mean entropy %.4f", best_idx + 1, mean_entropies[best_idx])

    shared_buffer = update_attribute_buffer(
        shared_buffer=shared_buffer,
        best_prompts=best_prompts,
        image_features=image_features,
        model=model,
        device=device,
        classnames=classnames,
        max_per_class=9,
        entropy_temperature=entropy_temperature,
    )
    return shared_buffer
