from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn.functional as F


def compute_entropy(logits: torch.Tensor, temperature: float = 0.04574) -> torch.Tensor:
    """Row-wise softmax entropy for logits [B, C]."""
    scaled = logits / float(temperature)
    probs = F.softmax(scaled, dim=1)
    log_probs = torch.log(probs + 1e-10)
    return -(probs * log_probs).sum(dim=1)


def compute_topology_similarity_from_buffers(
    all_buffer: List[List[str]],
    shared_attribute_buffer: List[List[str]],
) -> float:
    """Approximate topology similarity via normalized node/edge differences."""
    nodes_all = len(all_buffer)
    edges_all = sum(len(attrs) for attrs in all_buffer)

    nodes_shared = len(shared_attribute_buffer)
    edges_shared = sum(len(attrs) for attrs in shared_attribute_buffer)

    ged_raw = abs(nodes_all - nodes_shared) + abs(edges_all - edges_shared)
    denom = max(nodes_all + edges_all, nodes_shared + edges_shared, 1)
    d_norm = ged_raw / denom
    return 1.0 - d_norm
