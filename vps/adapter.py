"""Residual visual adapter (CLIP-Adapter) trained with an ArcFace loss for retrieval.

Reference: Gao et al., "CLIP-Adapter: Better Vision-Language Models with Feature Adapters"
(arXiv 2110.04544v2) and the authors' code
https://github.com/gaopengcuhk/CLIP-Adapter/blob/main/clip_adapter.py

What is reused: the bottleneck MLP (Linear -> ReLU -> Linear -> ReLU, no bias, reduction 4)
and residual blending  f' = ratio * A(f) + (1 - ratio) * f  on top of a frozen CLIP encoder.
What is adapted: the paper classifies with text prompts of known classes; DeepFashion test
items are unseen, so here the adapted feature is used as a retrieval embedding and trained
with ArcFace over train item_ids. The ArcFace classifier is thrown away after training.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Adapter(nn.Module):
    """Bottleneck MLP exactly as in the authors' code (both ReLUs, no bias)."""

    def __init__(self, dim: int = 512, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class AdaptedEmbedder(nn.Module):
    """Frozen CLIP feature -> residual adapter -> L2-normalised retrieval embedding."""

    def __init__(self, dim: int = 512, reduction: int = 4, ratio: float = 0.2):
        super().__init__()
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("ratio must be in [0, 1]")
        self.adapter = Adapter(dim, reduction)
        self.ratio = ratio

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        f = F.normalize(f, dim=-1)
        out = self.ratio * self.adapter(f) + (1.0 - self.ratio) * f
        return F.normalize(out, dim=-1)


class ArcFace(nn.Module):
    """Additive angular margin softmax (Deng et al., CVPR 2019).

    logit_j = s * cos(theta_j), except the true class uses s * cos(theta_y + m).
    Pushes each embedding towards its class centre by at least an angle m.
    """

    def __init__(self, dim: int, n_classes: int, s: float = 30.0, m: float = 0.3):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_classes, dim))
        nn.init.normal_(self.weight, std=0.01)
        self.s, self.m = s, m

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cos = F.linear(F.normalize(z, dim=-1), F.normalize(self.weight, dim=-1)).clamp(-1 + 1e-7, 1 - 1e-7)
        theta_y = torch.acos(cos.gather(1, y[:, None]))
        # Only add the margin while theta + m stays below pi (keeps the target logit monotonic).
        target = torch.where(theta_y + self.m < math.pi, torch.cos(theta_y + self.m), cos.gather(1, y[:, None]))
        logits = cos.scatter(1, y[:, None], target)
        return F.cross_entropy(self.s * logits, y)


@torch.no_grad()
def embed(model: AdaptedEmbedder, feats: np.ndarray, device: str = "cpu", batch: int = 4096) -> np.ndarray:
    model.eval()
    out = [model(torch.as_tensor(feats[i:i + batch], device=device)).cpu().numpy()
           for i in range(0, len(feats), batch)]
    return np.concatenate(out).astype(np.float32)


def train_adapter(feats: np.ndarray, labels: np.ndarray, *, ratio: float = 0.2, reduction: int = 4,
                  epochs: int = 20, batch_size: int = 256, lr: float = 1e-3, head_lr: float = 1e-2,
                  weight_decay: float = 1e-4, s: float = 30.0, m: float = 0.3, seed: int = 0,
                  device: str = "cpu", on_epoch=None) -> tuple[AdaptedEmbedder, list[dict]]:
    """Train adapter + ArcFace on fixed (cached) CLIP features.

    feats: (N, 512) frozen CLIP features, or (N, V, 512) with V cached views per image
        (E2: original + augmented views); each step then uses one random view per image.
    labels: (N,) int class index 0..C-1.
    on_epoch(epoch, model) -> dict of extra metrics (e.g. validation Recall@K), logged per epoch.
    Returns the model (classifier discarded) and the per-epoch log.
    """
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    x = torch.as_tensor(feats, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.long)
    n_classes = int(y.max()) + 1
    dim = x.shape[-1]
    model = AdaptedEmbedder(dim, reduction, ratio).to(device)
    head = ArcFace(dim, n_classes, s, m).to(device)
    opt = torch.optim.AdamW([{"params": model.parameters(), "lr": lr},
                             {"params": head.parameters(), "lr": head_lr}], weight_decay=weight_decay)
    log = []
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(x), generator=g)
        total, seen = 0.0, 0
        for i in range(0, len(x), batch_size):
            idx = perm[i:i + batch_size]
            xb = x[idx]
            if xb.ndim == 3:  # pick one cached view per image
                xb = xb[torch.arange(len(idx)), torch.randint(xb.shape[1], (len(idx),), generator=g)]
            xb, yb = xb.to(device), y[idx].to(device)
            loss = head(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
            seen += len(idx)
        row = {"epoch": epoch, "train_loss": total / seen}
        if on_epoch is not None:
            row.update(on_epoch(epoch, model))
        log.append(row)
    return model, log
