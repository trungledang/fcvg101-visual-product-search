"""T1 / E1: train a residual CLIP-Adapter with ArcFace on cached frozen CLIP features.

Steps
  1. Encode every official-train image once with frozen CLIP ViT-B/32 (C0: full image,
     CLIP preprocessing) and cache the vectors (cache/embeddings/train_full.*).
  2. Split official train by item_id into adapter-train / val_query / val_gallery
     (vps.preprocess.validation_split, the same split as the Task 1 study).
  3. E0 = frozen CLIP on val. E1 = adapter trained on adapter-train ids, evaluated on val
     every epoch; the epoch with the best val R@1 is kept.
The official query/gallery (test) split is never read here.

Usage: python scripts/train_adapter.py [--ratio 0.2] [--epochs 20] [--device cuda]
"""
import argparse
import copy
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from vps.adapter import ArcFace, AdaptedEmbedder, embed, train_adapter
from vps.annotations import file_sha256, parse_partition
from vps.metrics import l2_normalize, recall_at, retrieve
from vps.preprocess import validation_split


def encode_train(a, records, device):
    """Frozen CLIP features for all official-train images, cached and validated like the test cache."""
    from vps.embed import cache_meta, encode_images, load_cache, load_model, save_cache, source_fingerprint
    paths = [r.path for r in records]
    model, pre, meta = load_model(a.weights_dir, device)
    cm = cache_meta(paths, file_sha256(a.data_root / "Eval/list_eval_partition.txt"), meta,
                    source_fingerprint(a.data_root, paths))
    emb = load_cache(a.cache_dir, "train_full", paths, cm)
    if emb is None:
        print(f"encoding {len(paths)} train images on {device} ...")
        emb, timing = encode_images(model, pre, [a.data_root / p for p in paths], device, a.batch_size, a.workers)
        save_cache(a.cache_dir, "train_full", emb, paths, cm)
        print(f"  done in {timing['seconds']:.0f}s")
    else:
        print("train_full cache valid; not re-encoding")
    del model
    return dict(zip(paths, emb)), meta


def evaluate(q, g, q_ids, g_ids):
    r = retrieve(q, g, q_ids, g_ids, k=5)
    return r["first_pos_rank"], recall_at(r["first_pos_rank"])


def item_bootstrap(diff, groups, n_boot=2000, seed=0):
    """95% CI of a per-query difference, resampling whole items (queries of one item are correlated)."""
    members = {}
    for i, gid in enumerate(groups):
        members.setdefault(gid, []).append(i)
    clusters = list(members.values())
    rng = np.random.default_rng(seed)
    means = [diff[np.concatenate([clusters[c] for c in rng.integers(0, len(clusters), len(clusters))])].mean()
             for _ in range(n_boot)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=ROOT.parent)
    p.add_argument("--weights-dir", type=Path, default=ROOT / ".cache" / "clip")
    p.add_argument("--cache-dir", type=Path, default=ROOT / "cache" / "embeddings")
    p.add_argument("--ckpt-dir", type=Path, default=ROOT / "cache" / "adapter")
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=64, help="CLIP encoding batch size")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--ratio", type=float, default=0.2, help="residual ratio (CLIP-Adapter alpha)")
    p.add_argument("--reduction", type=int, default=4, help="bottleneck 512 -> 512/reduction")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--head-lr", type=float, default=1e-2)
    p.add_argument("--s", type=float, default=30.0, help="ArcFace scale")
    p.add_argument("--m", type=float, default=0.3, help="ArcFace angular margin (radians)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    records = parse_partition((a.data_root / "Eval/list_eval_partition.txt").read_text())
    split = validation_split(records)
    tr, vq, vg = split["train"], split["val_query"], split["val_gallery"]
    print(f"adapter-train {len(tr)} imgs / {len({r.item_id for r in tr})} ids; "
          f"val {len(vq)} query / {len(vg)} gallery / {len({r.item_id for r in vq})} ids")
    assert not {r.item_id for r in tr} & {r.item_id for r in vq + vg}, "train/val item_id overlap"

    feats, clip_meta = encode_train(a, [r for r in records if r.split == "train"], device)
    X = np.stack([feats[r.path] for r in tr])
    classes = sorted({r.item_id for r in tr})
    class_index = {c: i for i, c in enumerate(classes)}
    y = np.array([class_index[r.item_id] for r in tr])
    Q0, G0 = (np.stack([feats[r.path] for r in rs]) for rs in (vq, vg))
    q_ids, g_ids = [r.item_id for r in vq], [r.item_id for r in vg]

    # E0: frozen CLIP, no training. Should reproduce the Task 1 C0 val number (71.1% R@1).
    rank0, e0 = evaluate(l2_normalize(Q0), l2_normalize(G0), q_ids, g_ids)
    print(f"E0 frozen CLIP     val R@1 {e0['R@1']:.2%}  R@5 {e0['R@5']:.2%}")

    # Sanity check: random class weights give random cosines (~N(0, 1/512)), scaled by s,
    # so the initial loss should be a little above ln(C), not far above and not below.
    torch.manual_seed(a.seed)
    with torch.no_grad():
        init_loss = ArcFace(512, len(classes), a.s, 0.0)(AdaptedEmbedder(512, a.reduction, a.ratio)(
            torch.as_tensor(X[:2048])), torch.as_tensor(y[:2048])).item()
    print(f"init loss {init_loss:.3f} (ln C = {np.log(len(classes)):.3f})")

    best = {"R@1": -1.0}

    def on_epoch(epoch, model):
        Q, G = embed(model, Q0, device), embed(model, G0, device)
        rank, m = evaluate(Q, G, q_ids, g_ids)
        if m["R@1"] > best["R@1"]:
            best.update({"R@1": m["R@1"], "epoch": epoch, "state": copy.deepcopy(model.state_dict()), "rank": rank})
        print(f"  epoch {epoch:2d}  val R@1 {m['R@1']:.2%}  R@5 {m['R@5']:.2%}", flush=True)
        return {"val_R@1": m["R@1"], "val_R@5": m["R@5"]}

    t0 = time.perf_counter()
    model, log = train_adapter(X, y, ratio=a.ratio, reduction=a.reduction, epochs=a.epochs, lr=a.lr,
                               head_lr=a.head_lr, s=a.s, m=a.m, seed=a.seed, device=device, on_epoch=on_epoch)
    secs = time.perf_counter() - t0
    print(f"train loss {log[0]['train_loss']:.3f} (epoch 1) -> {log[-1]['train_loss']:.3f} (epoch {len(log)})")

    model.load_state_dict(best["state"])
    a.ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = a.ckpt_dir / f"e1_ratio{a.ratio}_seed{a.seed}.pt"
    cfg = {k: getattr(a, k) for k in ("ratio", "reduction", "epochs", "lr", "head_lr", "s", "m", "seed")}
    torch.save({"state_dict": best["state"], "config": cfg, "best_epoch": best["epoch"]}, ckpt)

    _, e1 = evaluate(embed(model, Q0, device), embed(model, G0, device), q_ids, g_ids)
    d = (best["rank"] == 1).astype(float) - (rank0 == 1).astype(float)
    out = {
        "experiment": "E1: frozen CLIP ViT-B/32 + residual adapter (CLIP-Adapter) + ArcFace, cached C0 features",
        "official_test_used": False,
        "selection": "best epoch by val R@1 (val is the model-selection set)",
        "split": {"adapter_train_images": len(tr), "adapter_train_ids": len(classes),
                  "val_query": len(vq), "val_gallery": len(vg), "val_ids": len(set(q_ids))},
        "clip": {k: clip_meta[k] for k in ("model", "checkpoint_sha256", "preprocess")},
        "config": cfg, "optimizer": "AdamW, weight_decay 1e-4, batch 256, constant LR",
        "device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "torch": torch.__version__, "python": platform.python_version(),
        "train_seconds": round(secs, 1), "init_loss": init_loss, "ln_num_classes": float(np.log(len(classes))),
        "adapter_params": sum(p.numel() for p in model.parameters()),
        "E0_frozen": e0, "E1_adapter": e1, "best_epoch": best["epoch"],
        "E1_minus_E0_R@1": {"mean": float(d.mean()), "ci95_by_item": item_bootstrap(d, q_ids)},
        "log": log, "checkpoint": str(ckpt.relative_to(ROOT)),
    }
    out_path = a.out or ROOT / "reports" / f"adapter_e1_val_ratio{a.ratio}_seed{a.seed}.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nE0 R@1 {e0['R@1']:.2%} -> E1 R@1 {e1['R@1']:.2%} (best epoch {best['epoch']}); "
          f"diff CI95 {out['E1_minus_E0_R@1']['ci95_by_item']}\nwrote {out_path}")


if __name__ == "__main__":
    main()
