"""Error analysis of the single final test run (plan items P3/P5).

Recomputes the E3 top-1 of every official test query from the cached features and the
seed-0 adapter, then groups the failures two ways:

  - outcome groups (they partition the 3,973 E3 failures, so the counts add up):
      adapter broke it (E0 right -> E3 wrong), both wrong but the right item is in #2-5
      or beyond, and the wrong top-1 is the same product category or not;
  - query groups (defined without looking at the outcome, so E0/E3 can be compared):
      close-up vs full-body bbox, view type from the file name, narrow bbox, Graphic_Tees.

Writes:
  reports/error_groups.json                      numbers used by the notebook and report
  docs/images/report/test_error_analysis.png     charts only (no dataset pixels) -> committed
  docs/images/dataset/error-groups.png           real examples -> NOT committed (DeepFashion
                                                 images are research/education only)

Usage: python scripts/analyze_error_groups.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vps.adapter import AdaptedEmbedder, embed  # noqa: E402
from vps.annotations import parse_partition  # noqa: E402

DATA_ROOT = ROOT.parent
CACHE = ROOT / "cache" / "embeddings"
CKPT = ROOT / "cache" / "adapter" / "ep40" / "e3_ratio0.2_seed0.pt"
OUT_JSON = ROOT / "reports" / "error_groups.json"
FIG_CHART = ROOT / "docs" / "images" / "report" / "test_error_analysis.png"
FIG_EXAMPLES = ROOT / "docs" / "images" / "dataset" / "error-groups.png"


def load_split():
    records = parse_partition((DATA_ROOT / "Eval" / "list_eval_partition.txt").read_text())
    q = [r for r in records if r.split == "query"]
    g = [r for r in records if r.split == "gallery"]
    return q, g


def load_features(paths: list[str], split: str) -> np.ndarray:
    got = (CACHE / f"{split}_A1_white_pad.paths.txt").read_text().splitlines()
    assert got == paths, f"{split} cache order differs from the partition"
    return np.load(CACHE / f"{split}_A1_white_pad.npy")


def load_bboxes() -> dict[str, tuple[int, int, int, int]]:
    out = {}
    with open(DATA_ROOT / "list_bbox_inshop.txt", encoding="utf-8") as f:
        next(f)
        next(f)
        for line in f:
            p = line.split()
            if len(p) == 7:
                out[p[0]] = tuple(int(v) for v in p[3:7])
    return out


def bbox_fraction(paths: list[str], bboxes) -> tuple[np.ndarray, np.ndarray]:
    """Fraction of the image area covered by the item bbox, and bbox width/height."""
    from PIL import Image

    frac, aspect = np.zeros(len(paths)), np.zeros(len(paths))
    for i, p in enumerate(paths):
        with Image.open(DATA_ROOT / p) as im:
            w, h = im.size
        x1, y1, x2, y2 = bboxes[p]
        frac[i] = (x2 - x1) * (y2 - y1) / (w * h)
        aspect[i] = (x2 - x1) / max(1, y2 - y1)
    return frac, aspect


def e3_top1(q_feats, g_feats, q_ids, g_ids, batch=1024):
    """Exact E3 top-1 and the best positive, chunked so the similarity matrix stays small."""
    n = len(q_feats)
    top1 = np.zeros(n, dtype=np.int64)
    top1_score = np.zeros(n, dtype=np.float32)
    pos_score = np.zeros(n, dtype=np.float32)
    for s in range(0, n, batch):
        e = min(s + batch, n)
        sims = q_feats[s:e] @ g_feats.T
        top1[s:e] = sims.argmax(axis=1)
        top1_score[s:e] = sims.max(axis=1)
        pos_score[s:e] = np.max(np.where(g_ids[None, :] == q_ids[s:e, None], sims, -np.inf), axis=1)
    assert np.isfinite(pos_score).all(), "a test query has no positive in the gallery"
    return top1, top1_score, pos_score


def main():
    q, g = load_split()
    q_paths, g_paths = [r.path for r in q], [r.path for r in g]
    q_ids = np.array([r.item_id for r in q])
    g_ids = np.array([r.item_id for r in g])
    q_cat = np.array([r.category for r in q])
    g_cat = np.array([r.category for r in g])
    q_view = np.array([r.path.rsplit("_", 1)[-1].split(".")[0] for r in q])

    q0, g0 = load_features(q_paths, "query"), load_features(g_paths, "gallery")
    state = torch.load(CKPT, map_location="cpu")
    cfg = state["config"]
    model = AdaptedEmbedder(512, cfg["reduction"], cfg["ratio"])
    model.load_state_dict(state["state_dict"])
    qf, gf = embed(model, q0), embed(model, g0)

    top1, top1_score, pos_score = e3_top1(qf, gf, q_ids, g_ids)

    ranks = np.load(ROOT / "reports" / "test_first_pos_ranks_E0_E3.npz")
    r0, r3 = ranks["E0_full_seedNone"], ranks["E3_seed0"]
    ok0, ok3 = r0 == 1, r3 == 1
    same_cat = g_cat[top1] == q_cat
    fail = ~ok3
    broken = ok0 & fail
    both = ~ok0 & fail
    close = r3 <= 5
    gap = top1_score - pos_score

    frac, aspect = bbox_fraction(q_paths, load_bboxes())

    buckets = [
        ("adapter_broke", "Adapter làm hỏng", "Adapter\nlàm hỏng",
         "E0 đúng ở #1, E3 sai", int(broken.sum())),
        ("close_same_cat", "Gần trùng, sát nút", "Gần trùng\nsát nút",
         "cả hai sai, top-1 cùng nhóm hàng, món đúng ở #2–5", int((both & close & same_cat).sum())),
        ("close_other_cat", "Sát nút nhưng khác nhóm hàng", "Sát nút\nkhác nhóm",
         "cả hai sai, top-1 khác nhóm hàng, món đúng ở #2–5", int((both & close & ~same_cat).sum())),
        ("far_same_cat", "Cùng nhóm hàng nhưng xếp hạng xa", "Cùng nhóm\nhạng xa",
         "cả hai sai, top-1 cùng nhóm hàng, món đúng ngoài #5", int((both & ~close & same_cat).sum())),
        ("far_other_cat", "Lạc hẳn (khác nhóm, hạng xa)", "Lạc hẳn",
         "cả hai sai, top-1 khác nhóm hàng, món đúng ngoài #5", int((both & ~close & ~same_cat).sum())),
    ]
    bucket_rows = [{"key": k, "label": lbl, "short": short, "definition": d, "queries": n,
                    "share_of_failures": n / int(fail.sum())} for k, lbl, short, d, n in buckets]
    assert sum(b["queries"] for b in bucket_rows) == int(fail.sum())

    def query_group(key, label, definition, mask):
        return {"key": key, "label": label, "definition": definition, "queries": int(mask.sum()),
                "e0_r1": float(ok0[mask].mean()), "e3_r1": float(ok3[mask].mean())}

    groups = [
        query_group("all", "Toàn bộ test", "14.218 query", np.ones(len(q), bool)),
        query_group("close_up", "Cận cảnh", "bbox chiếm ≥ 55% diện tích ảnh", frac >= 0.55),
        query_group("full_body", "Toàn thân", "bbox chiếm < 15% diện tích ảnh", frac < 0.15),
        query_group("flat", "Ảnh phẳng (không người mẫu)", "view = flat", q_view == "flat"),
        query_group("side", "Chụp nghiêng", "view = side", q_view == "side"),
        query_group("additional", "Ảnh chi tiết bổ sung", "view = additional", q_view == "additional"),
        query_group("narrow_bbox", "Bbox hẹp", "bbox rộng/cao < 0,45", aspect < 0.45),
        query_group("graphic_tees", "WOMEN/Graphic_Tees", "nhóm hàng khó nhất", q_cat == "WOMEN/Graphic_Tees"),
    ]

    by_view = []
    for v in ("front", "side", "back", "full", "flat", "additional"):
        m = q_view == v
        by_view.append({"view": v, "queries": int(m.sum()),
                        "e0_r1": float(ok0[m].mean()), "e3_r1": float(ok3[m].mean())})

    area_edges = [0, 0.10, 0.20, 0.35, 0.55, 1.01]
    by_area = []
    for lo, hi in zip(area_edges[:-1], area_edges[1:]):
        m = (frac >= lo) & (frac < hi)
        by_area.append({"from": lo, "to": hi, "queries": int(m.sum()),
                        "e0_r1": float(ok0[m].mean()), "e3_r1": float(ok3[m].mean())})

    # One typical example per bucket: the failure whose score gap is closest to the bucket median.
    gid_to_idx: dict[str, list[int]] = {}
    for i, iid in enumerate(g_ids):
        gid_to_idx.setdefault(iid, []).append(i)
    examples = []
    masks = {"adapter_broke": broken}
    for key, mask in (("close_same_cat", both & close & same_cat), ("close_other_cat", both & close & ~same_cat),
                      ("far_same_cat", both & ~close & same_cat), ("far_other_cat", both & ~close & ~same_cat)):
        masks[key] = mask
    for key, _lbl, _short, _d, _n in buckets:
        idx = np.flatnonzero(masks[key])
        med = np.median(gap[idx])
        pick = int(idx[np.argmin(np.abs(gap[idx] - med))])
        positives = gid_to_idx[q_ids[pick]]
        sims = qf[pick] @ gf[positives].T
        true_idx = int(positives[int(sims.argmax())])
        examples.append({
            "bucket": key, "query": q_paths[pick], "query_item": q_ids[pick], "category": q_cat[pick],
            "top1": g_paths[top1[pick]], "top1_item": g_ids[top1[pick]],
            "top1_score": float(top1_score[pick]),
            "true": g_paths[true_idx], "true_item": g_ids[true_idx],
            "score_gap": float(gap[pick]), "rank_e0": int(r0[pick]), "rank_e3": int(r3[pick]),
        })

    bands = [("2–5", 2, 5), ("6–10", 6, 10), ("ngoài #10", 11, len(g))]
    gap_bands = [{"band": name, "queries": int((fail & (r3 >= lo) & (r3 <= hi)).sum()),
                  "median_gap": float(np.median(gap[fail & (r3 >= lo) & (r3 <= hi)]))}
                 for name, lo, hi in bands]

    out = {
        "what": "error analysis of the single final test run (E3 seed 0), DeepFashion In-shop",
        "config": "E3: CLIP ViT-B/32 frozen + adapter ratio 0.2 (40 epochs, seed 0), crop + white pad",
        "queries": len(q), "gallery": len(g),
        "e0_r1": float(ok0.mean()), "e3_r1": float(ok3.mean()),
        "failures": int(fail.sum()),
        "adapter_fixes": int((~ok0 & ok3).sum()), "adapter_breaks": int(broken.sum()),
        "fail_same_category_top1": float(same_cat[fail].mean()),
        "fail_rank_in_top5": float((r3[fail] <= 5).mean()),
        "fail_gap_median_by_rank": gap_bands,
        "buckets": bucket_rows, "query_groups": groups,
        "by_view": by_view, "by_bbox_area": by_area,
        "top_failing_categories": [{"category": c, "failures": n} for c, n in
                                   Counter(q_cat[fail]).most_common(8)],
        "examples": examples,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", OUT_JSON)
    print(f"E3 R@1 {ok3.mean():.4f}  failures {fail.sum()}  adapter fixes {out['adapter_fixes']} / breaks {broken.sum()}")

    chart(out)
    example_figure(out)


def chart(out: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    ax = axes[0]
    labels = [b["short"] for b in out["buckets"]]
    vals = [b["queries"] for b in out["buckets"]]
    ax.barh(range(len(vals)), vals, color=["#c0504d", "#2f6fb0", "#6f9ecb", "#4f81bd", "#9db9d8"])
    ax.set_yticks(range(len(vals)), labels, fontsize=8)
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v + 20, i, f"{v:,}".replace(",", "."), va="center", fontsize=8)
    ax.set_xlabel("so query"); ax.set_title(f"Nhóm lỗi của E3 (tổng {out['failures']:,})".replace(",", "."), fontsize=10)

    ax = axes[1]
    views = out["by_view"]
    x = np.arange(len(views))
    ax.bar(x - 0.2, [v["e0_r1"] * 100 for v in views], 0.4, label="E0", color="#9aa5b1")
    ax.bar(x + 0.2, [v["e3_r1"] * 100 for v in views], 0.4, label="E3", color="#2f6fb0")
    ax.set_xticks(x, [f"{v['view']}\n(n={v['queries']:,})".replace(",", ".") for v in views], fontsize=7)
    ax.set_ylabel("R@1 (%)"); ax.set_ylim(0, 90); ax.legend(fontsize=8)
    ax.set_title("R@1 theo kiểu ảnh", fontsize=10); ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    area = out["by_bbox_area"]
    x = np.arange(len(area))
    ax.plot(x, [a["e3_r1"] * 100 for a in area], "o-", color="#2f6fb0", label="E3")
    ax.plot(x, [a["e0_r1"] * 100 for a in area], "o--", color="#9aa5b1", label="E0")
    ax.set_xticks(x, [f"{int(a['from'] * 100)}–{int(min(a['to'], 1) * 100)}%" for a in area], fontsize=8)
    ax.set_xlabel("bbox chiếm bao nhiêu diện tích ảnh"); ax.set_ylabel("R@1 (%)")
    ax.set_ylim(0, 90); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("Cận cảnh (bên phải) khó hơn", fontsize=10)
    fig.tight_layout()
    FIG_CHART.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_CHART, dpi=130, bbox_inches="tight")
    print("saved", FIG_CHART)


def example_figure(out: dict) -> None:
    """5 rows (one per outcome bucket) x 3 columns: query | wrong top-1 | true item."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from generate_docs_images import BLUE, GREEN, RED, font, open_img

    from PIL import Image, ImageDraw

    tile, label_h, pad = 190, 26, 8
    cols, w = 3, 3 * (tile + pad) + pad
    rows = len(out["examples"])
    h = label_h + tile + pad
    canvas = Image.new("RGB", (cols * w + pad, rows * h + pad), "white")
    d = ImageDraw.Draw(canvas)
    for r, ex in enumerate(out["examples"]):
        x0, y0 = pad, pad + r * h
        d.text((x0, y0 + 4), f"{ex['bucket']}  |  query #{ex['query_item']}  {ex['category']}"
                              f"  (E0 hang {ex['rank_e0']} -> E3 hang {ex['rank_e3']}, gap {ex['score_gap']:.3f})",
               fill="black", font=font(11))
        for c, (rel, lab, color) in enumerate((
                (ex["query"], "QUERY", BLUE),
                (ex["top1"], f"E3 #1 - SAI - {ex['top1_item']}", RED),
                (ex["true"], f"DUNG - {ex['true_item']}", GREEN))):
            im = open_img(rel)
            im.thumbnail((tile, tile), Image.LANCZOS)
            tx = x0 + c * (tile + pad)
            canvas.paste(im, (tx + (tile - im.width) // 2, y0 + label_h))
            d.rectangle([tx, y0 + label_h, tx + tile - 1, y0 + label_h + tile - 1], outline=color, width=2)
            d.text((tx + 3, y0 + label_h + tile - 14), lab, fill=color, font=font(10))
    FIG_EXAMPLES.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(FIG_EXAMPLES, optimize=True)
    print("saved", FIG_EXAMPLES)


if __name__ == "__main__":
    main()
