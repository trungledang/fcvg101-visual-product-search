"""Compare Task 1 configurations C0/C1/C3 on the train-identity validation split.

Frozen CLIP ViT-B/32 (or HSV histogram for C3), exact cosine retrieval, Recall@1/5.
The official query/gallery split is never read here: this is the model-selection set.
Usage: python scripts/eval_val.py [--data-root ..] [--device cuda]
"""
import argparse
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from vps.annotations import file_sha256, parse_partition
from vps.metrics import recall_at, retrieve
from vps.preprocess import CONFIGS, POSE_TYPES, BBoxCropPad, hsv_histogram, make_transform, parse_bbox, validation_split


class _Split:
    def __init__(self, recs, data_root, transform):
        self.recs, self.data_root, self.transform = recs, data_root, transform

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        from PIL import Image
        r = self.recs[i]
        with Image.open(self.data_root / r.path) as im:
            return self.transform(im.convert("RGB"), r.path)


def encode(model, ds, device, batch_size=32):
    import torch
    out = []
    with torch.inference_mode():
        for x in torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False):
            f = model.encode_image(x.to(device)).float()
            out.append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy())
    return np.concatenate(out)


def bootstrap(hits: dict, groups=None, n_boot=2000, seed=0):
    """Paired 95% CI (the same resample for every config). With `groups` (item_id per query)
    whole items are resampled, since queries of one item are not independent."""
    rng = np.random.default_rng(seed)
    n = len(next(iter(hits.values())))
    if groups is None:
        idx = rng.integers(0, n, size=(n_boot, n))
    else:
        members = {}
        for i, g in enumerate(groups):
            members.setdefault(g, []).append(i)
        clusters = list(members.values())
        picks = rng.integers(0, len(clusters), size=(n_boot, len(clusters)))
        idx = [np.concatenate([clusters[c] for c in row]) for row in picks]
    mean = lambda h: np.array([h[i].mean() for i in idx])  # noqa: E731
    return {name: [float(np.percentile(mean(h), q)) for q in (2.5, 97.5)] for name, h in hits.items()}, idx


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=ROOT.parent)
    p.add_argument("--bbox", type=Path, default=None, help="default: <data-root>/list_bbox_inshop.txt")
    p.add_argument("--weights-dir", type=Path, default=ROOT / ".cache" / "clip")
    p.add_argument("--device", default=None)
    p.add_argument("--out", type=Path, default=ROOT / "reports" / "val_results.json")
    a = p.parse_args()

    import torch
    from vps.embed import load_model
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    partition_file = a.data_root / "Eval/list_eval_partition.txt"
    bbox_file = a.bbox or a.data_root / "list_bbox_inshop.txt"
    records = parse_partition(partition_file.read_text())
    boxes = parse_bbox(bbox_file.read_text())
    val = validation_split(records)
    vq, vg = val["val_query"], val["val_gallery"]
    q_ids, g_ids = [r.item_id for r in vq], [r.item_id for r in vg]
    print(f"val: {len(vq)} query / {len(vg)} gallery images, {len(set(q_ids))} items; device {device}")

    model, preprocess, meta = load_model(a.weights_dir, device)
    crop = BBoxCropPad(boxes)
    hsv = lambda im, path: hsv_histogram(crop(im, path))  # noqa: E731
    results, ranks, timing = {}, {}, {}
    for name in ("C0_full", "C1_bbox_pad", "C3_hsv_hist"):
        t0 = time.perf_counter()
        if CONFIGS[name]["representation"] == "clip":
            tf = make_transform(name, preprocess, boxes)
            Q, G = (encode(model, _Split(rs, a.data_root, tf), device) for rs in (vq, vg))
        else:
            Q, G = (np.stack([x for x in _Split(rs, a.data_root, hsv)]) for rs in (vq, vg))
        timing[name] = time.perf_counter() - t0
        r = retrieve(Q, G, q_ids, g_ids, k=5)
        ranks[name] = r["first_pos_rank"]
        results[name] = {**recall_at(r["first_pos_rank"]), "dim": int(Q.shape[1]),
                         "median_first_positive_rank": float(np.median(r["first_pos_rank"])),
                         "mrr": float(np.mean(1 / r["first_pos_rank"])), "seconds": round(timing[name], 2)}
        print(f"  {name:12s} R@1 {results[name]['R@1']:.2%}  R@5 {results[name]['R@5']:.2%}  ({timing[name]:.1f}s)")

    hits1 = {n: (rk == 1).astype(float) for n, rk in ranks.items()}
    ci, idx = bootstrap(hits1, groups=q_ids)
    d = hits1["C1_bbox_pad"] - hits1["C0_full"]
    diff = np.array([d[i].mean() for i in idx])
    poses = [boxes[r.path].pose_type for r in vq]
    by_pose = {POSE_TYPES[pt]: {"queries": n, **{name: float(np.mean([ranks[name][i] == 1 for i in range(len(vq))
                                                                    if poses[i] == pt])) for name in ranks}}
               for pt, n in sorted(Counter(poses).items())}
    out = {
        "split": "validation: 10% of official train item_ids (vps.preprocess.validation_split, seed fcvg101-val)",
        "official_test_used": False,
        "partition_sha256": file_sha256(partition_file), "bbox_sha256": file_sha256(bbox_file),
        "val_query_images": len(vq), "val_gallery_images": len(vg), "val_items": len(set(q_ids)),
        "model": meta["model"], "checkpoint_sha256": meta["checkpoint_sha256"], "encoder_dtype": meta["encoder_dtype"],
        "device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "torch": torch.__version__, "python": platform.python_version(),
        "configs": {n: CONFIGS[n] for n in results},
        "results": results,
        "R@1_bootstrap95_by_item": ci,
        "C1_minus_C0_R@1": {"mean": float(diff.mean()), "ci95": [float(np.percentile(diff, 2.5)),
                                                                 float(np.percentile(diff, 97.5))]},
        "R@1_by_query_pose": by_pose,
    }
    a.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
