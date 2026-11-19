"""Inference: one image -> top-k gallery products, with frozen CLIP and an optional trained adapter.

Used by notebooks/task2_demo.ipynb (Task 2 inference demonstration). The gallery side is read
from the cached CLIP embeddings (scripts/pilot.py, scripts/encode_variant.py); only the new image
is encoded here, so a search costs one CLIP forward pass plus a 512 x 12,612 dot product.

Configurations match scripts/eval_test.py:
  E0_full  frozen CLIP, full image                E1  + adapter (ratio 0.2), full image
  E0_crop  frozen CLIP, bbox crop + white pad     E3  + adapter (ratio 0.2), bbox crop + white pad
A crop configuration needs the image's DeepFashion bbox. For an image outside the dataset there is
no bbox, so it is only padded to a white square (no crop) and the result says so.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .adapter import AdaptedEmbedder, embed
from .annotations import parse_partition
from .metrics import l2_normalize
from .preprocess import BBoxCropPad, pad_to_square, parse_bbox

WHITE = (255, 255, 255)
CONFIGS = {  # name: (gallery cache suffix, crop?, checkpoint prefix or None)
    "E0_full": ("", False, None),
    "E0_crop": ("_A1_white_pad", True, None),
    "E1": ("", False, "e1_ratio0.2"),
    "E3": ("_A1_white_pad", True, "e3_ratio0.2"),
}


@dataclass
class Hit:
    rank: int
    path: str
    item_id: str
    score: float


class ProductSearch:
    """Search the official test gallery (12,612 images) with one configuration."""

    def __init__(self, root: Path, config: str, clip_model, clip_preprocess, *, seed: int = 0,
                 data_root: Path | None = None, ckpt_dir: Path | None = None, device: str = "cpu"):
        suffix, self.crop, ck = CONFIGS[config]
        self.config, self.device = config, device
        self.data_root = data_root or root.parent
        self.clip_model, self.clip_preprocess = clip_model, clip_preprocess
        records = parse_partition((self.data_root / "Eval/list_eval_partition.txt").read_text())
        self.gallery = [r for r in records if r.split == "gallery"]
        cache = root / "cache" / "embeddings"
        paths = (cache / f"gallery{suffix}.paths.txt").read_text().splitlines()
        assert paths == [r.path for r in self.gallery], "gallery cache order differs from the partition"
        feats = np.load(cache / f"gallery{suffix}.npy")
        self.adapter = None
        if ck:
            state = torch.load((ckpt_dir or root / "cache/adapter/ep40") / f"{ck}_seed{seed}.pt",
                               map_location=device)
            cfg = state["config"]
            self.adapter = AdaptedEmbedder(512, cfg["reduction"], cfg["ratio"]).to(device).eval()
            self.adapter.load_state_dict(state["state_dict"])
            self.G = embed(self.adapter, feats, device)
        else:
            self.G = l2_normalize(feats)
        self.boxes = parse_bbox((self.data_root / "list_bbox_inshop.txt").read_text()) if self.crop else {}
        self._crop = BBoxCropPad(self.boxes, fill=WHITE) if self.crop else None

    def prepare(self, image, path: str | None = None):
        """PIL image as the encoder will see it, and a note on how it was prepared."""
        image = image.convert("RGB")
        if not self.crop:
            return image, "full image"
        if path in self.boxes:
            return self._crop(image, path), "bbox crop + white pad"
        return pad_to_square(image, WHITE), "no bbox known: white pad only (no crop)"

    @torch.inference_mode()
    def encode(self, image, path: str | None = None) -> np.ndarray:
        x = self.clip_preprocess(self.prepare(image, path)[0]).unsqueeze(0).to(self.device)
        f = self.clip_model.encode_image(x).float().cpu().numpy()
        return embed(self.adapter, f, self.device)[0] if self.adapter else l2_normalize(f)[0]

    def search(self, image, path: str | None = None, k: int = 5, exclude_path: bool = True) -> list[Hit]:
        """Top-k gallery images by cosine. The query file itself is skipped if it is in the gallery."""
        scores = self.G @ self.encode(image, path)
        order = np.argsort(-scores, kind="stable")
        hits = []
        for j in order:
            r = self.gallery[int(j)]
            if exclude_path and r.path == path:
                continue
            hits.append(Hit(len(hits) + 1, r.path, r.item_id, float(scores[j])))
            if len(hits) == k:
                break
        return hits
