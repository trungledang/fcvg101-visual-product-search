"""Encode all official-train images (adapter-train + validation) with a Task 1 preprocessing
variant, deterministically (no augmentation), for E3-style adapter experiments.

Variants (same definitions as scripts/ablate_crop_val.py):
  A1_white_pad : bbox crop (margin 0.1) + pad to square with white   (Task 1 A1)
--split train encodes official train (adapter-train + validation); --split query/gallery
encodes the official test side, used only for the single final evaluation (scripts/eval_test.py).

Usage: python scripts/encode_variant.py --variant A1_white_pad [--device cuda]
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from vps.annotations import parse_partition
from vps.preprocess import make_transform, parse_bbox

VARIANTS = {"A1_white_pad": ("C1_bbox_pad", {"fill": (255, 255, 255)})}


class _Images:
    """Builds the transform lazily per worker (closures cannot be pickled by Windows spawn)."""

    def __init__(self, data_root, paths, config, kwargs, clip_preprocess, boxes):
        self.data_root, self.paths, self.config, self.kwargs = data_root, paths, config, kwargs
        self.clip_preprocess, self.boxes, self.transform = clip_preprocess, boxes, None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        if self.transform is None:
            self.transform = make_transform(self.config, self.clip_preprocess, self.boxes, **self.kwargs)
        with Image.open(self.data_root / self.paths[i]) as im:
            return self.transform(im.convert("RGB"), self.paths[i])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=sorted(VARIANTS), default="A1_white_pad")
    p.add_argument("--split", choices=("train", "query", "gallery"), default="train")
    p.add_argument("--data-root", type=Path, default=ROOT.parent)
    p.add_argument("--weights-dir", type=Path, default=ROOT / ".cache" / "clip")
    p.add_argument("--cache-dir", type=Path, default=ROOT / "cache" / "embeddings")
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()

    from vps.embed import load_model
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    records = parse_partition((a.data_root / "Eval/list_eval_partition.txt").read_text())
    paths = [r.path for r in records if r.split == a.split]
    boxes = parse_bbox((a.data_root / "list_bbox_inshop.txt").read_text())
    config, kwargs = VARIANTS[a.variant]
    model, pre, meta = load_model(a.weights_dir, device)
    dl = torch.utils.data.DataLoader(_Images(a.data_root, paths, config, kwargs, pre, boxes),
                                     batch_size=a.batch_size, num_workers=a.workers)
    out, t0 = [], time.perf_counter()
    with torch.inference_mode():
        for x in dl:
            e = model.encode_image(x.to(device)).float()
            out.append((e / e.norm(dim=-1, keepdim=True)).cpu().numpy())
    emb = np.concatenate(out).astype(np.float32)
    name = f"{a.split}_{a.variant}"
    np.save(a.cache_dir / f"{name}.npy", emb)
    (a.cache_dir / f"{name}.paths.txt").write_text("\n".join(paths) + "\n")
    (a.cache_dir / f"{name}.meta.json").write_text(json.dumps({
        "variant": a.variant, "config": config, "kwargs": {k: list(v) if isinstance(v, tuple) else v
                                                           for k, v in kwargs.items()},
        "images": len(paths), "order_sha256": hashlib.sha256("\n".join(paths).encode()).hexdigest(),
        "checkpoint_sha256": meta["checkpoint_sha256"], "seconds": round(time.perf_counter() - t0, 1)}, indent=2))
    print(f"wrote {name}.npy {emb.shape} in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
