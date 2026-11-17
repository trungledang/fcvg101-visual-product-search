"""T5: single final evaluation of the configurations fixed on validation, on the official test.

Configurations were fixed BEFORE this run (docs/T1_ADAPTER_GIAI_THICH.md, 5c):
  E0_full  frozen CLIP, full image             E0_crop  frozen CLIP, crop + white pad
  E1       adapter (ratio 0.2), full image     E2       E1 trained with cached augmented views
  E3       adapter (ratio 0.2), crop + white pad
Each adapter uses the 40-epoch checkpoint (best val epoch) of seeds 0, 1, 2; the mean is reported.
Nothing here feeds back into a choice: no parameter is selected on test.
Full official gallery (12,612 images) as distractors; all 14,218 queries.

Usage: python scripts/eval_test.py [--device cuda]
"""
import argparse
import json
import platform
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from vps.adapter import AdaptedEmbedder, embed
from vps.annotations import parse_partition
from vps.metrics import l2_normalize, recall_at, retrieve

CONFIGS = {  # name: (feature cache prefix for query/gallery, checkpoint prefix or None)
    "E0_full": ("", None),
    "E0_crop": ("_A1_white_pad", None),
    "E1": ("", "e1_ratio0.2"),
    "E2": ("", "e2_ratio0.2"),
    "E3": ("_A1_white_pad", "e3_ratio0.2"),
}
SEEDS = (0, 1, 2)


def load_feats(cache_dir, split, suffix, paths):
    got = (cache_dir / f"{split}{suffix}.paths.txt").read_text().splitlines()
    assert got == paths, f"{split}{suffix} order differs from the partition"
    return np.load(cache_dir / f"{split}{suffix}.npy")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=ROOT.parent)
    p.add_argument("--cache-dir", type=Path, default=ROOT / "cache" / "embeddings")
    p.add_argument("--ckpt-dir", type=Path, default=ROOT / "cache" / "adapter" / "ep40")
    p.add_argument("--device", default=None)
    p.add_argument("--out", type=Path, default=ROOT / "reports" / "test_results.json")
    a = p.parse_args()
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

    records = parse_partition((a.data_root / "Eval/list_eval_partition.txt").read_text())
    q = [r for r in records if r.split == "query"]
    g = [r for r in records if r.split == "gallery"]
    q_ids, g_ids = [r.item_id for r in q], [r.item_id for r in g]
    cats = [r.category for r in q]
    print(f"test: {len(q)} queries / {len(g)} gallery / {len(set(q_ids))} items")

    results, ranks = {}, {}
    for name, (suffix, ck) in CONFIGS.items():
        Q0 = load_feats(a.cache_dir, "query", suffix, [r.path for r in q])
        G0 = load_feats(a.cache_dir, "gallery", suffix, [r.path for r in g])
        runs = []
        for seed in (SEEDS if ck else (None,)):
            if ck:
                state = torch.load(a.ckpt_dir / f"{ck}_seed{seed}.pt", map_location=device)
                cfg = state["config"]
                m = AdaptedEmbedder(512, cfg["reduction"], cfg["ratio"]).to(device)
                m.load_state_dict(state["state_dict"])
                Q, G = embed(m, Q0, device), embed(m, G0, device)
            else:
                Q, G = l2_normalize(Q0), l2_normalize(G0)
            r = retrieve(Q, G, q_ids, g_ids, k=10)
            rk = r["first_pos_rank"]
            met = recall_at(rk, ks=(1, 5, 10))
            met["MRR"] = float(np.mean(np.where(rk > 0, 1 / np.maximum(rk, 1), 0)))
            runs.append({"seed": seed, **met})
            ranks[f"{name}_seed{seed}"] = rk
        agg = {k: {"mean": float(np.mean([x[k] for x in runs])), "min": float(min(x[k] for x in runs)),
                   "max": float(max(x[k] for x in runs))} for k in ("R@1", "R@5", "R@10", "MRR")}
        results[name] = {"runs": runs, "aggregate": agg}
        print(f"{name:8s} R@1 {agg['R@1']['mean']:.2%} [{agg['R@1']['min']:.2%}-{agg['R@1']['max']:.2%}]  "
              f"R@5 {agg['R@5']['mean']:.2%}  R@10 {agg['R@10']['mean']:.2%}")

    # R@1 per category (seed 0 for adapters), to show where each configuration helps.
    by_cat = defaultdict(dict)
    for name in CONFIGS:
        rk = ranks[f"{name}_seed{0 if CONFIGS[name][1] else None}"]
        for c in sorted(set(cats)):
            idx = [i for i, x in enumerate(cats) if x == c]
            by_cat[c][name] = float(np.mean(rk[idx] == 1))
            by_cat[c]["queries"] = len(idx)

    out = {
        "what": "single final evaluation on DeepFashion In-shop official test (query/gallery)",
        "configs_fixed_before_test": True, "selection_on_test": False,
        "queries": len(q), "gallery": len(g), "items": len(set(q_ids)),
        "seeds": list(SEEDS), "checkpoints": str(a.ckpt_dir.relative_to(ROOT)),
        "device": device, "torch": torch.__version__, "python": platform.python_version(),
        "results": results, "R@1_by_category_seed0": by_cat,
    }
    a.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(a.cache_dir.parent / "test_first_pos_ranks.npz", **ranks)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
