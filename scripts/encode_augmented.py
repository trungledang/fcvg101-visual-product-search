"""T2: encode several augmented views of every adapter-train image with frozen CLIP.

Why: E1 trains on one fixed vector per image, so augmentation has no effect. E2 samples a
random view per image per epoch from these cached views (plus the original).

Each image is read from disk once and turned into V augmented views (I/O was the measured
bottleneck: 25,882 images took ~1h45 on the local machine). The augmentation is the Task 1
training augmentation (vps.preprocess.train_augment + RandomErasing) applied to the full image.
Each view's randomness is seeded from (seed, view, path), so the run is reproducible.
Work is saved in chunks, so an interrupted run resumes where it stopped.
Only adapter-train images are encoded: validation and the official test stay un-augmented.

Usage: python scripts/encode_augmented.py [--views 3] [--device cuda]
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
from vps.preprocess import make_transform, validation_split


class _Views:
    """Builds the transform lazily inside each worker: make_transform returns a closure,
    which Windows (spawn) DataLoader workers cannot pickle."""

    def __init__(self, data_root, paths, config, clip_preprocess, views, seed):
        self.data_root, self.paths, self.views, self.seed = data_root, paths, views, seed
        self.config, self.clip_preprocess, self.transform = config, clip_preprocess, None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        if self.transform is None:
            self.transform = make_transform(self.config, self.clip_preprocess, train=True)
        path = self.paths[i]
        with Image.open(self.data_root / path) as im:
            im = im.convert("RGB")
            out = []
            for v in range(self.views):
                key = hashlib.sha256(f"{self.seed}:{v}:{path}".encode()).digest()
                torch.manual_seed(int.from_bytes(key[:4], "little"))
                out.append(self.transform(im, path))
        return torch.stack(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=ROOT.parent)
    p.add_argument("--weights-dir", type=Path, default=ROOT / ".cache" / "clip")
    p.add_argument("--cache-dir", type=Path, default=ROOT / "cache" / "embeddings")
    p.add_argument("--config", default="C0_full_aug")
    p.add_argument("--views", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=32, help="images per batch (x views on GPU)")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--limit", type=int, default=None, help="smoke test: first N images, separate cache name")
    p.add_argument("--chunk", type=int, default=2048, help="images per saved chunk")
    a = p.parse_args()

    from vps.embed import load_model
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    records = parse_partition((a.data_root / "Eval/list_eval_partition.txt").read_text())
    paths = [r.path for r in validation_split(records)["train"]][:a.limit]
    model, pre, meta = load_model(a.weights_dir, device)

    name = f"train_aug_{a.config}_v{a.views}_s{a.seed}" + (f"_limit{a.limit}" if a.limit else "")
    part_dir = a.cache_dir / f"{name}.parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(paths)} adapter-train images x {a.views} views on {device} -> {name}")
    t_all = time.perf_counter()
    for c, start in enumerate(range(0, len(paths), a.chunk)):
        f = part_dir / f"{c:03d}.npy"
        if f.exists():
            continue
        sub = paths[start:start + a.chunk]
        dl = torch.utils.data.DataLoader(_Views(a.data_root, sub, a.config, pre, a.views, a.seed),
                                         batch_size=a.batch_size, num_workers=a.workers)
        out, t0 = [], time.perf_counter()
        with torch.inference_mode():
            for x in dl:
                b, v = x.shape[:2]
                e = model.encode_image(x.flatten(0, 1).to(device)).float()
                out.append((e / e.norm(dim=-1, keepdim=True)).view(b, v, -1).cpu().numpy())
        np.save(f, np.concatenate(out).astype(np.float32))
        dt = time.perf_counter() - t0
        print(f"  chunk {c} ({start + len(sub)}/{len(paths)}) {len(sub) / dt:.1f} img/s", flush=True)

    parts = sorted(part_dir.glob("*.npy"))
    emb = np.concatenate([np.load(f) for f in parts])
    assert emb.shape == (len(paths), a.views, 512), emb.shape
    np.save(a.cache_dir / f"{name}.npy", emb)
    (a.cache_dir / f"{name}.paths.txt").write_text("\n".join(paths) + "\n")
    (a.cache_dir / f"{name}.meta.json").write_text(json.dumps({
        "config": a.config, "views": a.views, "seed": a.seed, "images": len(paths),
        "order_sha256": hashlib.sha256("\n".join(paths).encode()).hexdigest(),
        "checkpoint_sha256": meta["checkpoint_sha256"],         "augmentation": "vps.preprocess.train_augment + RandomErasing(p=0.25) then CLIP preprocess",
        "seconds_this_run": round(time.perf_counter() - t_all, 1)}, indent=2))
    print(f"wrote {name}.npy {emb.shape}")


if __name__ == "__main__":
    main()
