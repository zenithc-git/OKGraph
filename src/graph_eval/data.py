from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


class ReturnBoth:
    """Transform wrapper that returns (tensor, raw_pil_copy)."""

    def __init__(self, preprocess):
        self.preprocess = preprocess

    def __call__(self, img: Image.Image):
        img = img.convert("RGB")
        return self.preprocess(img), img.copy()


def collate_with_pils(batch):
    """Collate function that keeps raw PILs for optional debugging/visualization.

    Each batch element is: ((tensor, raw_pil), label)

    Supports:
    - single-view tensor: [3, H, W]
    - multi-view tensor:  [V, 3, H, W] (will be concatenated along batch dim)
    """
    tensors = [item[0][0] for item in batch]
    pils = [item[0][1] for item in batch]
    labels = [item[1] for item in batch]

    if tensors[0].ndim == 4:
        # multi-view: concatenate all views
        images = torch.cat(tensors, dim=0)  # [sum V, 3, H, W]
        expanded_labels: List[int] = []
        expanded_pils: List[Image.Image] = []
        for (tensor, pil), lab in batch:
            v = int(tensor.shape[0])
            expanded_labels.extend([int(lab)] * v)
            expanded_pils.extend([pil] * v)
        labels_t = torch.as_tensor(expanded_labels)
        raw_pils = expanded_pils
    else:
        images = torch.stack(tensors, dim=0)  # [B, 3, H, W]
        labels_t = torch.as_tensor(labels)
        raw_pils = pils

    return (images, raw_pils), labels_t


def build_image_loader(
    image_root: str,
    preprocess,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
) -> Tuple[DataLoader, ImageFolder]:
    dataset = ImageFolder(image_root, transform=ReturnBoth(preprocess))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_with_pils,
    )
    return loader, dataset
