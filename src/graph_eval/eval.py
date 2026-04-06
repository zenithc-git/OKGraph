from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .buffers import flatten_prompt_list, refresh_all_buffer_cache
from .fast import run_fast_thinking_entropy_twostep
from .metrics import compute_entropy, compute_topology_similarity_from_buffers
from .slow import run_slow_thinking
from .config import EvalConfig, LLMConfig
from .llm import make_request_descriptor

logger = logging.getLogger(__name__)


def set_seed(seed: int = 0) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _is_empty_buffer(buf: Optional[List[List[str]]]) -> bool:
    return (buf is None) or (len(buf) == 0) or all(len(x) == 0 for x in buf)


@torch.no_grad()
def _encode_text_prompts(model, device: str, prompts: Sequence[str]) -> torch.Tensor:
    import clip  # type: ignore

    tokens = clip.tokenize(list(prompts)).to(device)
    feats = model.encode_text(tokens).to(device)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


def evaluate_openvocab(
    *,
    model,
    image_loader,
    dataset,
    classnames: Sequence[str],
    # base prompts (from JSON) are only needed to build initial caches; keep separate
    text_features_base: torch.Tensor,
    line_to_class: Sequence[int],
    all_buffer: Optional[List[List[str]]],
    cfg: EvalConfig,
    llm_cfg: LLMConfig,
    device: str = "cuda",
) -> Tuple[float, float]:
    """Main evaluation loop."""
    model.eval()
    num_classes = len(classnames)

    # activation counters per class
    activation_counters = [defaultdict(int) for _ in range(num_classes)]

    # shared buffer (slow-thinking produces here, then optionally merges into all_buffer)
    shared_attribute_buffer: List[List[str]] = [[] for _ in range(num_classes)]

    # build / refresh all_buffer cache
    if not _is_empty_buffer(all_buffer):
        all_flat_prompts, all_class_prompt_map, all_text_features = refresh_all_buffer_cache(
            all_buffer, classnames, model, device, text_features_base.shape[1]
        )
    else:
        # fallback: class-name-only prompts
        fallback_prompts = [f"a photo of a {name}" for name in classnames]
        all_text_features = _encode_text_prompts(model, device, fallback_prompts)
        all_class_prompt_map = {c: [c] for c in range(num_classes)}
        all_flat_prompts = list(fallback_prompts)

    # LLM request function (optional)
    request_func = None
    if llm_cfg.enabled and not llm_cfg.api_key:
        # allow env var
        pass
    if llm_cfg.enabled:
        request_func = make_request_descriptor(llm_cfg)

    top1_correct = 0
    top5_correct = 0
    total = 0

    entropy_log: List[float] = []
    prediction_log: List[int] = []

    last_prune_at = 0

    pbar = tqdm(image_loader, desc="Evaluating", total=len(image_loader))
    for batch_i, ((images, _raw_pils), labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        preds_fast, stage_record, entropy_prompt, entropy_class, logits_all, class_logits = (
            run_fast_thinking_entropy_twostep(
                image_features=image_features,
                text_features=all_text_features,
                class_prompt_map=all_class_prompt_map,
                entropy_threshold_class=cfg.entropy_threshold_class,
                entropy_temperature=cfg.entropy_temperature,
                topk_per_class=cfg.topk_per_class,
                all_buffer=all_buffer,
                record_topk_activations=cfg.record_topk_activations and (all_buffer is not None),
                activation_counters=activation_counters,
                num_classes=num_classes,
            )
        )

        final_logits = class_logits.clone()

        used_slow = (preds_fast == -1).any().item()
        if used_slow and llm_cfg.enabled and request_func is not None:
            slow_mask = preds_fast == -1
            class_logits_slow = class_logits[slow_mask]  # [B_slow, C]
            image_features_slow = image_features[slow_mask]  # [B_slow, D]

            if image_features_slow.numel() > 0:
                topk_ids = torch.topk(class_logits_slow, k=min(5, num_classes), dim=1).indices.tolist()
                topk_names = [[classnames[cid] for cid in ids] for ids in topk_ids]

                shared_attribute_buffer = run_slow_thinking(
                    image_features=image_features_slow,
                    topk_class_names_per_image=topk_names,
                    topk_class_ids_per_image=topk_ids,
                    model=model,
                    device=device,
                    classnames=classnames,
                    request_func=request_func,
                    shared_buffer=shared_attribute_buffer,
                    total_classes=num_classes,
                    max_rounds=1,
                    n_attrs_per_class=2,
                    entropy_temperature=cfg.entropy_temperature,
                )

                flat_prompts_shared, class_prompt_map_shared = flatten_prompt_list(shared_attribute_buffer)
                if flat_prompts_shared:
                    text_features_buffer = _encode_text_prompts(model, device, flat_prompts_shared)
                    b_slow = image_features_slow.size(0)
                    logits_slow = torch.zeros((b_slow, num_classes), device=device)

                    for cls_id, idxs in class_prompt_map_shared.items():
                        feats_c = text_features_buffer[idxs]
                        sims = image_features_slow @ feats_c.T
                        logits_slow[:, cls_id] = sims.mean(dim=1)

                    slow_indices = slow_mask.nonzero(as_tuple=True)[0]
                    for k, bi in enumerate(slow_indices.tolist()):
                        final_logits[bi, :] = cfg.alpha * class_logits[bi, :] + (1.0 - cfg.alpha) * logits_slow[k, :]

        # accuracy
        preds = torch.argmax(final_logits, dim=1)
        top5_preds = torch.topk(final_logits, k=min(5, num_classes), dim=1).indices

        for j in range(labels.size(0)):
            total += 1
            gt = int(labels[j].item())
            pred = int(preds[j].item())
            if pred == gt:
                top1_correct += 1
            if gt in top5_preds[j].tolist():
                top5_correct += 1

            entropy_log.append(float(entropy_prompt[j].item()))
            prediction_log.append(pred)

        # periodic prune
        if all_buffer is not None and cfg.record_topk_activations and cfg.prune_every > 0:
            batch_count = batch_i + 1
            if (batch_count - last_prune_at) >= cfg.prune_every:
                removed = 0
                for c in range(num_classes):
                    buf = all_buffer[c]
                    if not buf:
                        continue
                    counter_c = activation_counters[c]
                    keep = [a for a in buf if counter_c.get(a, 0) >= cfg.min_hits_to_keep]
                    removed += (len(buf) - len(keep))
                    all_buffer[c] = keep
                    # reset counters
                    for k in list(counter_c.keys()):
                        counter_c[k] = 0

                # refresh caches
                all_flat_prompts, all_class_prompt_map, all_text_features = refresh_all_buffer_cache(
                    all_buffer, classnames, model, device, text_features_base.shape[1]
                )
                if all_text_features.shape[0] == 0:
                    fallback_prompts = [f"a photo of a {name}" for name in classnames]
                    all_text_features = _encode_text_prompts(model, device, fallback_prompts)
                    all_class_prompt_map = {c: [c] for c in range(num_classes)}
                    all_flat_prompts = list(fallback_prompts)

                logger.info("Pruned %d attributes at batch %d", removed, batch_count)
                last_prune_at = batch_count

        # periodic consolidate (shared -> all)
        if cfg.consolidate_every > 0 and (batch_i + 1) % cfg.consolidate_every == 0:
            if all_buffer is None:
                all_buffer = [[] for _ in range(num_classes)]

            flat_shared, map_shared = flatten_prompt_list(shared_attribute_buffer)
            nonempty_classes = len(map_shared)
            if flat_shared and nonempty_classes > (2 * num_classes) / 3.0:
                # all_buffer=None mergeall
                is_all_empty = (sum(len(a) for a in all_buffer) == 0)
                if is_all_empty:
                    merged_cnt = 0
                    for c in map_shared.keys():
                        incoming = shared_attribute_buffer[c]
                        if not incoming:
                            continue
                        existing_set = set(all_buffer[c])
                        new_list = [p for p in incoming if p not in existing_set]
                        if new_list:
                            all_buffer[c] = (all_buffer[c] + new_list)[: cfg.max_all_attrs_per_class]
                            merged_cnt += len(new_list)
                        shared_attribute_buffer[c] = []
                    logger.info("All buffer empty; merged %d attributes directly.", merged_cnt)

                    # refresh caches after direct merge
                    all_flat_prompts, all_class_prompt_map, all_text_features = refresh_all_buffer_cache(
                        all_buffer, classnames, model, device, text_features_base.shape[1]
                    )
                    if all_text_features.shape[0] == 0:
                        fallback_prompts = [f"a photo of a {name}" for name in classnames]
                        all_text_features = _encode_text_prompts(model, device, fallback_prompts)
                        all_class_prompt_map = {c: [c] for c in range(num_classes)}
                        all_flat_prompts = list(fallback_prompts)
                    continue
                    
                # encode shared prompts
                text_features_buffer = _encode_text_prompts(model, device, flat_shared)

                # topology sim
                s_t = compute_topology_similarity_from_buffers(all_buffer, shared_attribute_buffer)

                # edge sim (global mean similarity between shared and existing attrs per class)
                sims_list = []
                for c in range(num_classes):
                    idxs_g = map_shared.get(c, [])
                    if not idxs_g:
                        continue
                    ag = text_features_buffer[idxs_g]
                    idxs_t = all_class_prompt_map.get(c, [])
                    if idxs_t:
                        at = all_text_features[idxs_t]
                        sims_list.append((ag @ at.T).flatten())
                    else:
                        sims_list.append(torch.tensor([0.0], device=device))
                s_e = float(torch.cat(sims_list).mean().item()) if sims_list else 0.0

                s_global = 0.5 * (s_t + s_e)
                novelty = 1.0 - s_global

                merge_all = novelty >= cfg.novelty_theta
                rollback = False

                if merge_all and cfg.enable_entropy_guard:
                    # compute entropy before
                    n = image_features.size(0)
                    logits_before = torch.full((n, num_classes), -1e9, device=device)
                    for c in range(num_classes):
                        idxs = all_class_prompt_map.get(c, [])
                        if not idxs:
                            continue
                        p = all_text_features[idxs]
                        logits_before[:, c] = (image_features @ p.T).mean(dim=1)
                    h_before = float(compute_entropy(logits_before, temperature=cfg.entropy_temperature).mean().item())

                    # compute entropy after (simulate merge)
                    logits_after = logits_before.clone()
                    limit = int(cfg.max_all_attrs_per_class)
                    for c, idxs_g in map_shared.items():
                        ag = text_features_buffer[idxs_g]
                        idxs_t = all_class_prompt_map.get(c, [])
                        if idxs_t:
                            at = all_text_features[idxs_t]
                            if at.shape[0] >= limit:
                                p_after = at[:limit]
                            else:
                                remain = limit - at.shape[0]
                                p_after = torch.cat([at, ag[:remain]], dim=0) if ag.shape[0] else at
                        else:
                            p_after = ag[:limit] if ag.shape[0] > limit else ag
                        logits_after[:, c] = (image_features @ p_after.T).mean(dim=1)

                    h_after = float(compute_entropy(logits_after, temperature=cfg.entropy_temperature).mean().item())
                    delta = (h_after - h_before) if cfg.entropy_guard_delta_mode == "absolute" else (h_after - h_before) / max(1e-8, abs(h_before))
                    rollback = delta > cfg.entropy_guard_delta_threshold
                    logger.info("Entropy guard: before=%.4f after=%.4f delta=%+.4f rollback=%s", h_before, h_after, delta, rollback)

                if merge_all and not rollback:
                    merged_cnt = 0
                    for c in map_shared.keys():
                        incoming = shared_attribute_buffer[c]
                        if not incoming:
                            continue
                        existing_set = set(all_buffer[c])
                        new_list = [p for p in incoming if p not in existing_set]
                        if new_list:
                            all_buffer[c] = (all_buffer[c] + new_list)[: cfg.max_all_attrs_per_class]
                            merged_cnt += len(new_list)
                        shared_attribute_buffer[c] = []
                    logger.info("Merged %d attributes into all_buffer", merged_cnt)
                else:
                    if rollback:
                        for c in map_shared.keys():
                            shared_attribute_buffer[c] = []
                        logger.info("Merge rolled back; cleared shared buffer.")
                    else:
                        logger.info("Novelty below threshold; kept shared buffer.")

                # refresh caches after merge decision
                all_flat_prompts, all_class_prompt_map, all_text_features = refresh_all_buffer_cache(
                    all_buffer, classnames, model, device, text_features_base.shape[1]
                )
                if all_text_features.shape[0] == 0:
                    fallback_prompts = [f"a photo of a {name}" for name in classnames]
                    all_text_features = _encode_text_prompts(model, device, fallback_prompts)
                    all_class_prompt_map = {c: [c] for c in range(num_classes)}
                    all_flat_prompts = list(fallback_prompts)

    top1_acc = (top1_correct / total * 100) if total else 0.0
    top5_acc = (top5_correct / total * 100) if total else 0.0

    # save logs (optional; keep compatibility)
    with open("entropy_log.json", "w", encoding="utf-8") as f:
        json.dump(entropy_log, f, indent=2)
    with open("prediction_log.json", "w", encoding="utf-8") as f:
        json.dump(prediction_log, f, indent=2)

    return top1_acc, top5_acc
