#!/usr/bin/env python
"""Visual Product Search feasibility pilot runner.

Subcommands (run in order):
  preflight  record hardware/software, host reachability, official checkpoint hash
  benchmark  measure CPU/GPU throughput of the ViT-B/32 image tower (random init; compute only)
  audit      parse + validate official partition, hash every image, duplicate audit
  embed      encode query+gallery with frozen OpenAI CLIP ViT-B/32 (cached)
  evaluate   exact cosine retrieval vs the full gallery, Recall@1/5, sensitivity eval
  examples   deterministic success/failure selection + montages (private artifacts)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vps import annotations as ann  # noqa: E402
from vps import env_info  # noqa: E402

REPORTS = ROOT / "reports"
ART = ROOT / "artifacts"  # private: may reference/contain dataset-derived images


def dump(name: str, obj) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    p = REPORTS / name
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n")
    print(f"wrote {p}")
    return p


def partition(data_root: Path):
    f = data_root / "Eval" / "list_eval_partition.txt"
    if not f.is_file():
        sys.exit(f"missing {f}. Download the official In-shop release first (see README).")
    return ann.parse_partition(f.read_text()), ann.file_sha256(f)


# ---------------------------------------------------------------- preflight
HOSTS = {
    "official_project_page": "https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html",
    "google_drive (official In-shop files)": "https://drive.google.com/",
    "google_drive_download": "https://drive.usercontent.google.com/",
    "openai_clip_checkpoint_cdn": "https://openaipublic.azureedge.net/",
    "huggingface": "https://huggingface.co/",
    "pypi": "https://pypi.org/simple/",
    "github": "https://github.com/",
}


def cmd_preflight(a):
    import urllib.request
    from vps.embed import official_checkpoint
    reach = {}
    for name, url in HOSTS.items():
        try:
            with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=15) as r:
                reach[name] = {"url": url, "status": r.status}
        except Exception as e:  # noqa: BLE001
            code = getattr(e, "code", None)
            reach[name] = {"url": url, "status": code, "error": f"{e.__class__.__name__}: {str(e)[:120]}"}
    url, sha = official_checkpoint()
    dump("preflight.json", {"measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "env": env_info.collect(str(ROOT)),
                            "host_reachability": reach,
                            "clip_checkpoint": {"url": url, "official_sha256": sha,
                                                "present_locally": (a.weights_dir / Path(url).name).exists()}})


# ---------------------------------------------------------------- benchmark
def cmd_benchmark(a):
    """Throughput of the exact ViT-B/32 image tower with random weights.

    FLOPs do not depend on weight values, so this measures compute cost without the
    checkpoint. It says nothing about retrieval quality.
    """
    import torch
    from clip.model import CLIP
    from PIL import Image
    from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
    from torchvision.transforms import InterpolationMode

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIP(embed_dim=512, image_resolution=224, vision_layers=12, vision_width=768, vision_patch_size=32,
                 context_length=77, vocab_size=49408, transformer_width=512, transformer_heads=8, transformer_layers=12)
    model.visual.to(device).eval()
    nparams = sum(p.numel() for p in model.visual.parameters())
    pre = Compose([Resize(224, interpolation=InterpolationMode.BICUBIC), CenterCrop(224),
                   lambda im: im.convert("RGB"), ToTensor(),
                   Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
    rng = np.random.default_rng(0)
    imgs = [Image.fromarray(rng.integers(0, 255, (256, 176, 3), dtype=np.uint8)) for _ in range(a.n)]
    t0 = time.perf_counter()
    x = torch.stack([pre(im) for im in imgs]).to(device)
    t_pre = time.perf_counter() - t0
    res = {}
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    batch_sizes = (8, 16) if device == "cuda" else (32, 64)
    with torch.inference_mode():
        model.visual(x[:4])  # warm-up
        for bs in batch_sizes:
            t0 = time.perf_counter()
            for s in range(0, a.n, bs):
                model.visual(x[s:s + bs])
            dt = time.perf_counter() - t0
            res[f"batch_{bs}"] = {"images": a.n, "seconds": round(dt, 3), "images_per_second": round(a.n / dt, 2)}
    best = max(v["images_per_second"] for v in res.values())
    pre_ips = a.n / t_pre
    eff = 1 / (1 / best + 1 / (pre_ips * min(4, torch.get_num_threads())))  # rough: preprocessing in 4 workers
    dump("benchmark_local.json", {
        "what": "ViT-B/32 image tower, random weights, synthetic 256x176 RGB inputs (compute cost only, NOT a quality result)",
        "env": env_info.collect(str(ROOT)), "device": device, "visual_params": nparams,
        "preprocess_single_thread_images_per_second": round(pre_ips, 2),
        "forward": res, "peak_rss_mb": round(env_info.peak_rss_mb(), 1),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if device == "cuda" else None,
        "extrapolation_note": "seconds_for_N = N / images_per_second; an extrapolation, not a measurement",
        "estimated_end_to_end_images_per_second": round(eff, 2),
    })


# ---------------------------------------------------------------- audit
def cmd_audit(a):
    from vps.audit import exact_groups, hash_images, near_duplicate_pairs
    recs, ann_sha = partition(a.data_root)
    summary = ann.split_summary(recs)
    facts = ann.validate_splits(recs)
    evalrecs = [r for r in recs if r.split in ("query", "gallery")] if not a.include_train else recs
    t0 = time.perf_counter()
    rows = hash_images([str(a.data_root / r.path) for r in evalrecs], workers=a.workers)
    t_hash = time.perf_counter() - t0
    errors = [(evalrecs[i].path, r["error"]) for i, r in enumerate(rows) if "error" in r]
    split_of = [r.split for r in evalrecs]

    def cross(groups):
        out = []
        for g in groups:
            sp = {split_of[i] for i in g}
            if "query" in sp and "gallery" in sp:
                out.append([{"path": evalrecs[i].path, "split": split_of[i], "item_id": evalrecs[i].item_id} for i in g])
        return out

    file_dups, pix_dups = exact_groups(rows, "file_sha256"), exact_groups(rows, "pixel_sha256")
    q_idx = [i for i, s in enumerate(split_of) if s == "query" and "phash" in rows[i]]
    g_idx = [i for i, s in enumerate(split_of) if s == "gallery" and "phash" in rows[i]]
    pairs = near_duplicate_pairs([rows[i]["phash"] for i in q_idx], [rows[i]["phash"] for i in g_idx], a.phash_threshold)
    near = [{"query": evalrecs[q_idx[i]].path, "gallery": evalrecs[g_idx[j]].path, "hamming": d,
             "same_item": evalrecs[q_idx[i]].item_id == evalrecs[g_idx[j]].item_id} for i, j, d in pairs]
    sizes = Counter((r["w"], r["h"]) for r in rows if "w" in r)
    hashes_path = ART / "image_hashes.csv"
    hashes_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hashes_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "split", "item_id", "file_sha256", "pixel_sha256", "phash"])
        for r, row in zip(evalrecs, rows):
            w.writerow([r.path, r.split, r.item_id, row.get("file_sha256"), row.get("pixel_sha256"), row.get("phash")])
    cats = defaultdict(lambda: Counter())
    for r in recs:
        cats[r.category][r.split] += 1
    dump("audit.json", {
        "annotation_file": "Eval/list_eval_partition.txt", "annotation_sha256": ann_sha,
        "parsed_rows": len(recs), "split_summary": summary, "split_checks": facts,
        "per_category_images": {c: dict(v) for c, v in sorted(cats.items())},
        "images_checked": len(rows), "hash_seconds": round(t_hash, 1), "missing_or_corrupt": len(errors),
        "missing_or_corrupt_examples": errors[:50], "image_sizes_top": [[list(k), v] for k, v in sizes.most_common(5)],
        "exact_file_duplicate_groups_total": len(file_dups),
        "exact_file_duplicate_groups_cross_query_gallery": cross(file_dups),
        "exact_pixel_duplicate_groups_total": len(pix_dups),
        "exact_pixel_duplicate_groups_cross_query_gallery": cross(pix_dups),
        "phash_threshold_hamming": a.phash_threshold,
        "phash_near_duplicate_query_gallery_pairs": len(near),
        "phash_near_duplicate_pairs_same_item": sum(p["same_item"] for p in near),
        "phash_near_duplicate_pairs_sample": near[:100],
        "note": "pHash pairs are candidates for manual review only; same-item pairs across query/gallery are intended positives unless pixel-identical.",
    })


# ---------------------------------------------------------------- query sampling
def sample_queries(qrecs, fraction: float, seed: str):
    """Deterministic, outcome-independent: keep >=1 query per item id, then hash-ordered fill per category."""
    if fraction >= 1:
        return qrecs
    key = lambda r: hashlib.sha256(f"{seed}:{r.path}".encode()).hexdigest()  # noqa: E731
    by_item = defaultdict(list)
    for r in qrecs:
        by_item[r.item_id].append(r)
    chosen = {min(v, key=key).path for v in by_item.values()}
    target = int(round(fraction * len(qrecs)))
    rest = sorted((r for r in qrecs if r.path not in chosen), key=key)
    for r in rest[:max(0, target - len(chosen))]:
        chosen.add(r.path)
    return [r for r in qrecs if r.path in chosen]


# ---------------------------------------------------------------- embed
def cmd_embed(a):
    import torch
    from vps.embed import cache_meta, encode_images, load_cache, load_model, save_cache
    recs, ann_sha = partition(a.data_root)
    torch.set_num_threads(a.threads or torch.get_num_threads())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    model, pre, meta = load_model(a.weights_dir, device)
    t_load = time.perf_counter() - t0
    timings = {"model_load_seconds": round(t_load, 2)}
    for split in ("gallery", "query"):
        paths = [r.path for r in recs if r.split == split]
        cm = cache_meta(paths, ann_sha, meta)
        if load_cache(a.cache_dir, split, paths, cm) is not None:
            print(f"{split}: cache valid, skipping")
            continue
        emb, t = encode_images(model, pre, [a.data_root / p for p in paths], device, a.batch_size, a.workers)
        save_cache(a.cache_dir, split, emb, paths, cm)
        timings[split] = t
    dump("embed_timing.json", {"model": meta, "device": device, "torch_threads": torch.get_num_threads(),
                               "timings": timings, "peak_rss_mb": round(env_info.peak_rss_mb(), 1),
                               "peak_vram_mb": (torch.cuda.max_memory_allocated() / 2**20) if device == "cuda" else None,
                               "cache_bytes": sum(f.stat().st_size for f in a.cache_dir.glob("*") if f.is_file())})


# ---------------------------------------------------------------- evaluate
def cmd_evaluate(a):
    from vps.embed import cache_meta, load_cache, official_checkpoint
    from vps.metrics import recall_at, retrieve
    recs, ann_sha = partition(a.data_root)
    meta = json.loads((a.cache_dir / "gallery.meta.json").read_text())
    model_meta = {"checkpoint_sha256": official_checkpoint()[1], "preprocess": meta["preprocess"]}
    q_all = [r for r in recs if r.split == "query"]
    g = [r for r in recs if r.split == "gallery"]
    G = load_cache(a.cache_dir, "gallery", [r.path for r in g], cache_meta([r.path for r in g], ann_sha, model_meta))
    Qall = load_cache(a.cache_dir, "query", [r.path for r in q_all], cache_meta([r.path for r in q_all], ann_sha, model_meta))
    if G is None or Qall is None:
        sys.exit("embedding cache missing or incompatible; run `embed` first")
    q = sample_queries(q_all, a.query_fraction, a.seed)
    qi = {r.path: i for i, r in enumerate(q_all)}
    Q = Qall[[qi[r.path] for r in q]]
    q_ids, g_ids = [r.item_id for r in q], [r.item_id for r in g]

    t0 = time.perf_counter()
    res = retrieve(Q, G, q_ids, g_ids, k=5)
    t_ret = time.perf_counter() - t0
    main = recall_at(res["first_pos_rank"])

    # sensitivity: mask gallery images byte- or pixel-identical to the query
    hashes = {}
    hp = ART / "image_hashes.csv"
    if hp.exists():
        with open(hp) as f:
            for row in csv.DictReader(f):
                hashes[row["path"]] = (row["file_sha256"], row["pixel_sha256"])
    sens = None
    if hashes:
        g_by_hash = defaultdict(set)
        for j, r in enumerate(g):
            for h in hashes.get(r.path, ()):
                g_by_hash[h].add(j)
        mask = {}
        for i, r in enumerate(q):
            m = set().union(*(g_by_hash.get(h, set()) for h in hashes.get(r.path, ())))
            if m:
                mask[i] = m
        if mask:
            r2 = retrieve(Q, G, q_ids, g_ids, k=5, mask=mask)
            sens = recall_at(r2["first_pos_rank"])
            sens["queries_with_masked_identical_gallery_images"] = len(mask)
            sens["masked_gallery_matches"] = sum(len(v) for v in mask.values())
            sens["queries_losing_all_positives"] = int(((res["first_pos_rank"] > 0) & (r2["first_pos_rank"] == 0)).sum())
            sens["treatment"] = "queries left with no positive are excluded from this sensitivity denominator and counted above"
        else:
            sens = {"note": "no query had a byte- or pixel-identical gallery image; sensitivity eval equals main eval"}

    # per-query CSV
    REPORTS.mkdir(parents=True, exist_ok=True)
    per_q = []
    with open(REPORTS / "per_query_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_path", "item_id", "category", "first_pos_rank", "n_gallery_positives"]
                   + [f"top{k}_{c}" for k in range(1, 6) for c in ("path", "item_id", "score")])
        gpos = Counter(g_ids)
        for i, r in enumerate(q):
            top = []
            for k in range(5):
                j = res["top_idx"][i, k]
                top.append({"path": g[j].path, "item_id": g[j].item_id, "score": float(res["top_scores"][i, k])})
            fr = int(res["first_pos_rank"][i])
            per_q.append({"path": r.path, "item_id": r.item_id, "category": r.category, "first_pos_rank": fr, "top5": top})
            w.writerow([r.path, r.item_id, r.category, fr, gpos[r.item_id]]
                       + [x for t in top for x in (t["path"], t["item_id"], f"{t['score']:.5f}")])
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "per_query.json").write_text(json.dumps(per_q))

    # per-category + macro over items
    by_cat = defaultdict(list)
    by_item = defaultdict(list)
    for p in per_q:
        by_cat[p["category"]].append(p["first_pos_rank"])
        by_item[p["item_id"]].append(p["first_pos_rank"])
    per_cat = {c: {"queries": len(v), **{f"R@{k}": round(float(np.mean([0 < x <= k for x in v])), 4) for k in (1, 5)}}
               for c, v in sorted(by_cat.items())}
    macro_item = {f"R@{k}": float(np.mean([np.mean([0 < x <= k for x in v]) for v in by_item.values()])) for k in (1, 5)}
    ranks = res["first_pos_rank"][res["first_pos_rank"] > 0]
    dump("metrics.json", {
        "status": "full official query set" if a.query_fraction >= 1 else f"PARTIAL estimate: {a.query_fraction:.0%} seeded query sample",
        "query_sampling": {"fraction": a.query_fraction, "seed": a.seed, "queries": len(q), "of_official_queries": len(q_all),
                           "items_covered": len(set(q_ids)), "of_query_items": len({r.item_id for r in q_all}),
                           "categories_covered": len(by_cat)},
        "gallery_images": len(g), "gallery_items": len(set(g_ids)),
        "main_official_split": main, "sensitivity_identical_images_masked": sens,
        "median_first_positive_rank": float(np.median(ranks)) if len(ranks) else None,
        "mean_reciprocal_rank": float(np.mean(1 / ranks)) if len(ranks) else None,
        "macro_over_items": macro_item, "per_category": per_cat,
        "retrieval_seconds": round(t_ret, 2), "retrieval": "exact brute-force cosine (dot product of L2-normalised float32 embeddings)",
    })


# ---------------------------------------------------------------- examples
def cmd_examples(a):
    from vps.examples import render, select_examples
    per_q = json.loads((ART / "per_query.json").read_text())
    recs, _ = partition(a.data_root)
    sel = select_examples(per_q, a.per_group, a.seed)
    G = np.load(a.cache_dir / "gallery.npy")
    Q = np.load(a.cache_dir / "query.npy")
    qpaths = (a.cache_dir / "query.paths.txt").read_text().splitlines()
    gpaths = (a.cache_dir / "gallery.paths.txt").read_text().splitlines()
    qi = {p: i for i, p in enumerate(qpaths)}
    g_by_item = defaultdict(list)
    for j, p in enumerate(gpaths):
        g_by_item[p.split("/")[3]].append(j)
    outdir = ART / "examples"
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = {"selection_policy": "deterministic: outcome group, then round-robin over sorted categories, "
                "sha256(seed:path) order within category, one query per item id; diagnostic only, not prevalence",
                "seed": a.seed, "per_group_target": a.per_group, "groups": {}}
    for grp, items in sel.items():
        manifest["groups"][grp] = []
        for n, ex in enumerate(items, 1):
            if ex["first_pos_rank"] > 1:
                sims = G[g_by_item[ex["item_id"]]] @ Q[qi[ex["path"]]]
                j = g_by_item[ex["item_id"]][int(np.argmax(sims))]
                ex = {**ex, "known_positive": {"path": gpaths[j], "score": float(sims.max())}}
            fn = f"{grp}_{n:02d}.png"
            render(ex, a.data_root, outdir / fn, f"{grp} | {ex['category']} | first positive rank {ex['first_pos_rank']}")
            manifest["groups"][grp].append({"file": fn, **ex})
        if len(items) < a.per_group:
            manifest["groups"][grp + "_shortfall"] = a.per_group - len(items)
    dump("examples_manifest.json", manifest)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=ROOT / "data" / "inshop")
    p.add_argument("--weights-dir", type=Path, default=ROOT / ".cache" / "clip")
    p.add_argument("--cache-dir", type=Path, default=ROOT / "cache" / "embeddings")
    p.add_argument("--seed", default="fcvg101")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("preflight")
    b = sub.add_parser("benchmark"); b.add_argument("--n", type=int, default=256)
    au = sub.add_parser("audit"); au.add_argument("--workers", type=int, default=4)
    au.add_argument("--phash-threshold", type=int, default=4); au.add_argument("--include-train", action="store_true")
    e = sub.add_parser("embed"); e.add_argument("--batch-size", type=int, default=64)
    e.add_argument("--workers", type=int, default=4); e.add_argument("--threads", type=int, default=0)
    ev = sub.add_parser("evaluate"); ev.add_argument("--query-fraction", type=float, default=1.0)
    ex = sub.add_parser("examples"); ex.add_argument("--per-group", type=int, default=10)
    p.add_argument("--reports-dir", type=Path, default=REPORTS)
    p.add_argument("--artifacts-dir", type=Path, default=ART)
    a = p.parse_args()
    globals().update(REPORTS=a.reports_dir, ART=a.artifacts_dir)
    {"preflight": cmd_preflight, "benchmark": cmd_benchmark, "audit": cmd_audit, "embed": cmd_embed,
     "evaluate": cmd_evaluate, "examples": cmd_examples}[a.cmd](a)


if __name__ == "__main__":
    main()
