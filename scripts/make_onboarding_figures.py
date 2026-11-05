"""Onboarding figures + stats for docs/HIEU_DU_AN.md.

Every number comes from files already in the project:
- Eval/list_eval_partition.txt, list_bbox_inshop.txt (labels; next to the repo)
- cache/embeddings/{query,gallery}.npy (frozen CLIP ViT-B/32 baseline embeddings)
- reports/per_query_results.csv, reports/metrics.json, reports/val_crop_ablation.json

Outputs contain only aggregate statistics (no dataset pixels), so they can be
committed to the public repo:
- docs/images/onboarding/*.png
- reports/onboarding_stats.json

Run: .venv/Scripts/python.exe scripts/make_onboarding_figures.py
"""
from __future__ import annotations

import collections
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT.parent
EMB = ROOT / "cache" / "embeddings"
OUT = ROOT / "docs" / "images" / "onboarding"
STATS = ROOT / "reports" / "onboarding_stats.json"

# Reference palette (dataviz skill, light mode; first three slots validate all-pairs)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
})

item_of = lambda p: Path(p).parts[-2]
cat_of = lambda p: "/".join(Path(p).parts[-4:-2])
colour_of = lambda p: Path(p).stem.split("_")[0]
view_of = lambda p: Path(p).stem.split("_", 2)[2]

VIEW_VI = {"front": "front (trước)", "side": "side (nghiêng)", "back": "back (sau)",
           "full": "full (toàn thân)", "additional": "additional (ảnh bổ sung)",
           "flat": "flat (trải phẳng)"}


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / name)


def dataset_stats() -> dict:
    lines = (DATA / "Eval" / "list_eval_partition.txt").read_text().splitlines()[2:]
    part = [l.split() for l in lines]
    out = {}
    for s in ("train", "query", "gallery"):
        rows = [(p, i) for p, i, st in part if st == s]
        per_item = collections.Counter(i for _, i in rows)
        out[s] = {
            "images": len(rows), "items": len(per_item),
            "images_per_item_median": float(np.median(list(per_item.values()))),
            "images_per_item_max": max(per_item.values()),
            "views": dict(collections.Counter(view_of(p) for p, _ in rows).most_common()),
        }
    colours = collections.defaultdict(set)
    for p, i, _ in part:
        colours[i].add(colour_of(p))
    n = collections.Counter(len(v) for v in colours.values())
    out["items_total"] = len(colours)
    out["items_with_multiple_colour_variants"] = sum(v for k, v in n.items() if k > 1)
    out["categories"] = len({cat_of(p) for p, _, _ in part})
    return out


def similarity_stats() -> dict:
    q = np.load(EMB / "query.npy").astype(np.float32)
    g = np.load(EMB / "gallery.npy").astype(np.float32)
    qp = (EMB / "query.paths.txt").read_text().split()
    gp = (EMB / "gallery.paths.txt").read_text().split()
    gi = np.array([item_of(p) for p in gp])
    gc = np.array([cat_of(p) for p in gp])
    gcol = np.array([colour_of(p) for p in gp])
    rng = np.random.default_rng(0)
    best_pos, hard_neg, rand_neg, all_pos = [], [], [], []
    top1_ok, hard_same_cat, same_colour, views = [], [], [], []
    for s in range(0, len(qp), 2000):
        sims = q[s:s + 2000] @ g.T
        for j, row in enumerate(sims):
            p = qp[s + j]
            pos = gi == item_of(p)
            neg_idx = np.where(~pos)[0]
            k = neg_idx[row[neg_idx].argmax()]
            best_pos.append(row[pos].max())
            hard_neg.append(row[k])
            rand_neg.append(row[rng.choice(neg_idx)])
            all_pos.extend(row[pos].tolist())
            top1_ok.append(bool(pos[row.argmax()]))
            hard_same_cat.append(gc[k] == cat_of(p))
            same_colour.append(bool((gcol[pos] == colour_of(p)).any()))
            views.append(view_of(p))
    best_pos, hard_neg, rand_neg = map(np.array, (best_pos, hard_neg, rand_neg))
    all_pos, top1_ok = np.array(all_pos), np.array(top1_ok)
    same_colour, views = np.array(same_colour), np.array(views)

    # Fig: three similarity distributions
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.linspace(0.5, 1.0, 101)
    for data, c, lab in ((rand_neg, S2, "Ảnh khác sản phẩm, chọn ngẫu nhiên"),
                         (all_pos, S1, "Ảnh cùng sản phẩm (mọi cặp query–gallery)"),
                         (hard_neg, S3, "Ảnh khác sản phẩm GIỐNG NHẤT với query")):
        ax.hist(data, bins=bins, density=True, histtype="step", linewidth=2, color=c, label=lab)
    ax.set_xlabel("Cosine similarity giữa embedding CLIP của query và ảnh gallery")
    ax.set_ylabel("Mật độ")
    ax.set_title("Cosine luôn cao; cùng sản phẩm chỉ cao hơn một chút")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "similarity-distributions.png")

    # Fig: best positive vs hardest negative per query
    fig, ax = plt.subplots(figsize=(6, 5.6))
    h = ax.hist2d(hard_neg, best_pos, bins=120, range=[[0.8, 1.0], [0.7, 1.0]], cmap="Blues", cmin=1)
    ax.plot([0.8, 1.0], [0.8, 1.0], color=INK, linewidth=1.5)
    ax.text(0.81, 0.985, f"Trên đường chéo: đúng ở hạng 1\n({top1_ok.mean():.1%} query)",
            va="top", fontsize=9, color=INK)
    ax.text(0.99, 0.72, f"Dưới đường chéo: có ảnh sai\nxếp trên ảnh đúng ({1 - top1_ok.mean():.1%})",
            ha="right", va="bottom", fontsize=9, color=INK)
    ax.set_xlabel("Cosine của ảnh SAI giống nhất")
    ax.set_ylabel("Cosine của ảnh ĐÚNG giống nhất")
    ax.set_title("Mỗi điểm là một query (14.218)")
    ax.grid(False)
    fig.colorbar(h[3], ax=ax, label="Số query")
    save(fig, "best-positive-vs-hardest-negative.png")

    # Fig: R@1 by query view
    order = [v for v in ("front", "side", "full", "additional", "back", "flat") if (views == v).any()]
    r1 = [top1_ok[views == v].mean() for v in order]
    cnt = [(views == v).sum() for v in order]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    y = np.arange(len(order))[::-1]
    ax.barh(y, r1, color=S1, height=0.6)
    for yy, r, n in zip(y, r1, cnt):
        ax.text(r + 0.01, yy, f"{r:.1%}  (n={n:,})", va="center", fontsize=9, color=INK2)
    ax.set_yticks(y, [VIEW_VI[v] for v in order])
    ax.set_xlim(0, 0.75)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("Recall@1 theo kiểu ảnh của query (official test)")
    ax.grid(axis="y", visible=False)
    save(fig, "recall1-by-query-view.png")

    pct = lambda a: [round(float(x), 4) for x in np.percentile(a, [5, 50, 95])]
    return {
        "check_R@1_recomputed": float(top1_ok.mean()),
        "cosine_all_positive_pairs": {"mean": float(all_pos.mean()), "p5_p50_p95": pct(all_pos)},
        "cosine_random_negative": {"mean": float(rand_neg.mean()), "p5_p50_p95": pct(rand_neg)},
        "cosine_best_positive_per_query": {"mean": float(best_pos.mean()), "p5_p50_p95": pct(best_pos)},
        "cosine_hardest_negative_per_query": {"mean": float(hard_neg.mean()), "p5_p50_p95": pct(hard_neg)},
        "hardest_negative_same_category_share": float(np.mean(hard_same_cat)),
        "queries_with_same_colour_positive_in_gallery": int(same_colour.sum()),
        "R@1_when_same_colour_positive_exists": float(top1_ok[same_colour].mean()),
        "queries_only_other_colour_positives": int((~same_colour).sum()),
        "R@1_when_only_other_colour_positives": float(top1_ok[~same_colour].mean()),
        "R@1_by_query_view": {v: {"queries": int(n), "R@1": round(float(r), 4)} for v, n, r in zip(order, cnt, r1)},
    }


def recall_curve() -> dict:
    with open(ROOT / "reports" / "per_query_results.csv", encoding="utf-8") as f:
        ranks = np.array([int(r["first_pos_rank"]) for r in csv.DictReader(f)])
    ks = np.arange(1, 51)
    rec = np.array([(ranks <= k).mean() for k in ks])
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(ks, rec, color=S1, linewidth=2)
    for k in (1, 5, 10, 20, 50):
        ax.plot(k, rec[k - 1], "o", color=S1, markersize=7, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(f"R@{k} = {rec[k - 1]:.1%}", (k, rec[k - 1]), textcoords="offset points",
                    xytext=(6, -14), fontsize=9, color=INK2)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("K (số kết quả đầu được xem)")
    ax.set_ylabel("Recall@K")
    ax.set_title("Ảnh đúng thường ở gần đầu, nhưng không phải hạng 1")
    save(fig, "recall-at-k.png")
    return {f"R@{k}": round(float(rec[k - 1]), 4) for k in (1, 5, 10, 20, 50)} | {
        "median_first_positive_rank": float(np.median(ranks)), "max_first_positive_rank": int(ranks.max())}


def category_chart() -> None:
    m = json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))["per_category"]
    items = sorted(m.items(), key=lambda kv: kv[1]["R@1"])
    fig, ax = plt.subplots(figsize=(7, 7))
    y = np.arange(len(items))
    ax.barh(y, [v["R@1"] for _, v in items], color=S1, height=0.65)
    for yy, (_, v) in zip(y, items):
        ax.text(v["R@1"] + 0.008, yy, f"{v['R@1']:.0%} (n={v['queries']})", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, [k for k, _ in items], fontsize=8)
    ax.set_xlim(0, 0.85)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("Recall@1 theo category (official test)")
    ax.grid(axis="y", visible=False)
    save(fig, "recall1-by-category.png")


def ablation_chart() -> None:
    r = json.loads((ROOT / "reports" / "val_crop_ablation.json").read_text(encoding="utf-8"))["results"]
    labels = {"C0_full": "C0 ảnh đầy đủ", "A1_white_pad": "Cắt bbox + viền trắng",
              "C1_bbox_pad": "Cắt bbox + viền màu trung bình", "X_context_only": "Che sản phẩm, chỉ giữ nền"}
    keep = [(labels[k], r[k]["R@1"]) for k in labels if k in r]
    fig, ax = plt.subplots(figsize=(7, 2.8))
    y = np.arange(len(keep))[::-1]
    ax.barh(y, [v for _, v in keep], color=[S1] + [INK2] * (len(keep) - 1), height=0.6)
    for yy, (_, v) in zip(y, keep):
        ax.text(v + 0.01, yy, f"{v:.1%}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(y, [k for k, _ in keep])
    ax.set_xlim(0, 0.85)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("Thí nghiệm xử lý ảnh trên validation (R@1)")
    ax.grid(axis="y", visible=False)
    save(fig, "validation-crop-ablation.png")


if __name__ == "__main__":
    stats = {"dataset": dataset_stats(), "similarity": similarity_stats(), "recall_curve": recall_curve()}
    category_chart()
    ablation_chart()
    STATS.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved", STATS)
