"""Deterministic diagnostic example selection and montage rendering."""
from __future__ import annotations

import hashlib
from collections import defaultdict


def _key(path: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{path}".encode()).hexdigest()


def select_examples(rows: list[dict], per_group: int, seed: str = "fcvg101") -> dict[str, list[dict]]:
    """rows: per-query dicts with keys path, item_id, category, first_pos_rank.

    Groups: success@1, fail@1 but hit@5, fail@5. Within a group, round-robin over
    categories (sorted) and take the hash-ordered first query per category per round,
    never repeating an item id. Selection uses only outcome group + path hash, not scores.
    """
    groups = {"success_top1": lambda r: r["first_pos_rank"] == 1,
              "fail_top1_recovered_top5": lambda r: 2 <= r["first_pos_rank"] <= 5,
              "fail_top5": lambda r: r["first_pos_rank"] > 5}
    out = {}
    for name, pred in groups.items():
        by_cat = defaultdict(list)
        for r in rows:
            if pred(r):
                by_cat[r["category"]].append(r)
        for c in by_cat:
            by_cat[c].sort(key=lambda r: _key(r["path"], seed))
        chosen, used_ids = [], set()
        cats = sorted(by_cat)
        while len(chosen) < per_group and any(by_cat[c] for c in cats):
            for c in cats:
                while by_cat[c]:
                    r = by_cat[c].pop(0)
                    if r["item_id"] not in used_ids:
                        chosen.append(r)
                        used_ids.add(r["item_id"])
                        break
                if len(chosen) >= per_group:
                    break
        out[name] = chosen
    return out


def render(example: dict, image_root, out_path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    panels = [("QUERY\n" + example["item_id"], example["path"], "black")]
    for rk, t in enumerate(example["top5"], 1):
        ok = t["item_id"] == example["item_id"]
        panels.append((f"#{rk} {'OK' if ok else 'X'} {t['item_id']}\ncos={t['score']:.3f}", t["path"], "green" if ok else "red"))
    if example.get("known_positive"):
        kp = example["known_positive"]
        panels.append((f"positive (rank {example['first_pos_rank']})\ncos={kp['score']:.3f}", kp["path"], "blue"))
    fig, axes = plt.subplots(1, len(panels), figsize=(2.2 * len(panels), 3.4))
    for ax, (lab, p, col) in zip(axes, panels):
        with Image.open(image_root / p) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(lab, fontsize=7, color=col)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(col); s.set_linewidth(2)
    fig.suptitle(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
