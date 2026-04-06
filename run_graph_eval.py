from __future__ import annotations

import argparse
import logging
import os
import re
from datetime import datetime
from typing import Optional

import torch

from src.graph_eval.config import EvalConfig, LLMConfig
from src.graph_eval.data import build_image_loader
from src.graph_eval.eval import evaluate_openvocab, set_seed
from src.graph_eval.knowledge_base import load_prompts



def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Open-vocabulary evaluation with CLIP + optional LLM slow-thinking.")
    p.add_argument("--model-path", required=True, help="Path to CLIP .pt (or a CLIP identifier supported by clip.load).")
    p.add_argument("--json-path", required=True, help="JSON prompt definition file.")
    p.add_argument("--image-root", required=True, help="ImageFolder root directory (class subfolders).")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)

    # eval cfg
    p.add_argument("--entropy-threshold-class", type=float, default=99.0)
    p.add_argument("--alpha", type=float, default=0.4)
    p.add_argument("--consolidate-every", type=int, default=160)
    p.add_argument("--novelty-theta", type=float, default=0.5)
    p.add_argument("--max-all-attrs-per-class", type=int, default=17)
    p.add_argument("--topk-per-class", type=int, default=5)
    p.add_argument("--prune-every", type=int, default=240)
    p.add_argument("--min-hits-to-keep", type=int, default=80)

    # llm
    p.add_argument("--disable-llm", action="store_true", help="Disable LLM slow-thinking stage.")
    p.add_argument("--llm-model", default="gpt-3.5-turbo")
    p.add_argument("--openai-api-key", default=None, help="If set, overrides OPENAI_API_KEY env var.")
    p.add_argument("--openai-base-url", default=None, help="If set, overrides OPENAI_BASE_URL env var.")

    # logging
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--log-dir", default="logs")
    return p

def _sanitize_name(s: str) -> str:
    s = os.path.basename(s)
    s = os.path.splitext(s)[0]
    s = re.sub(r"[^a-zA-Z0-9_-]", "", s)
    return s

def _extract_dataset_name(image_root: str) -> str:
    # Expect: .../<dataset>/<split>
    parent = os.path.dirname(image_root.rstrip("/"))
    return _sanitize_name(parent)

def _setup_logging_with_args(args, log_dir: str = "logs", level: str = "INFO") -> str:
    os.makedirs(log_dir, exist_ok=True)

    model_name = _sanitize_name(args.model_path)
    dataset_name = _extract_dataset_name(args.image_root)
    ent = args.entropy_threshold_class

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    log_filename = f"{model_name}_{dataset_name}_ent{ent}_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    return log_path

def main(argv: Optional[list[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    log_path = _setup_logging_with_args(args, args.log_dir, args.log_level)
    logger = logging.getLogger("openvocab_eval")
    logger.info("Logging to %s", log_path)

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # Load CLIP
    import clip  # type: ignore

    model, preprocess = clip.load(args.model_path, device=device)
    model.eval()

    # Load prompts
    prompts, line_to_class, gt_attrs_by_class, gt_attr_prompt_indices_by_class, classnames = load_prompts(args.json_path)

    with torch.no_grad():
        text_tokens = clip.tokenize(prompts).to(device)
        text_features = model.encode_text(text_tokens).to(device)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Initialize all_buffer from GT attrs (raw phrases)
    all_buffer = [attrs[:] for attrs in gt_attrs_by_class]

    # Data loader
    image_loader, dataset = build_image_loader(
        args.image_root,
        preprocess,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )

    cfg = EvalConfig(
        entropy_threshold_class=args.entropy_threshold_class,
        alpha=args.alpha,
        consolidate_every=args.consolidate_every,
        novelty_theta=args.novelty_theta,
        max_all_attrs_per_class=args.max_all_attrs_per_class,
        topk_per_class=args.topk_per_class,
        prune_every=args.prune_every,
        min_hits_to_keep=args.min_hits_to_keep,
    )

    llm_cfg = LLMConfig(
        enabled=not args.disable_llm,
        model=args.llm_model,
        api_key=args.openai_api_key,
        base_url=args.openai_base_url,
    )

    top1, top5 = evaluate_openvocab(
        model=model,
        image_loader=image_loader,
        dataset=dataset,
        classnames=classnames,
        text_features_base=text_features,
        line_to_class=line_to_class,
        all_buffer=all_buffer,
        cfg=cfg,
        llm_cfg=llm_cfg,
        device=device,
    )

    logger.info("Top-1 Accuracy: %.2f%% | Top-5 Accuracy: %.2f%%", top1, top5)


if __name__ == "__main__":  # pragma: no cover
    main()
