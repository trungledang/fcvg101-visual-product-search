"""Val-only ablations explaining why bbox crop (C1) loses to the full image (C0).

Each variant changes one factor of C1, plus a context-only probe that hides the
product bbox and keeps everything else. Official query/gallery are never read.
Usage: python scripts/ablate_crop_val.py [--data-root ..]
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np

from eval_val import _Split, bootstrap, encode
from vps.annotations import file_sha256, parse_partition
from vps.metrics import recall_at, retrieve
from vps.preprocess import CLIP_MEAN_RGB, POSE_TYPES, crop_box, make_transform, parse_bbox, validation_split

WHITE = (255, 255, 255)
VARIANTS = {  # name -> (config, BBoxCropPad args, what changes vs C1)
    "C0_full": ("C0_full", {}, "reference: full image"),
    "C1_bbox_pad": ("C1_bbox_pad", {}, "reference: margin 0.1, mean-colour pad"),
    "A1_white_pad": ("C1_bbox_pad", {"fill": WHITE}, "pad colour white"),
    "A2_margin_0.3": ("C1_bbox_pad", {"margin": 0.3}, "margin 0.3"),
    "A2_margin_0.5": ("C1_bbox_pad", {"margin": 0.5}, "margin 0.5"),
    "A3_full_if_narrow32": ("C1_bbox_pad", {"full_if_narrower": 32}, "full image when bbox width < 32 px"),
}


def context_only(boxes, preprocess):
    """Hide the (10%-expanded) product bbox with the CLIP mean colour; keep the rest."""
    from PIL import ImageDraw

    def apply(image, path):
        image = image.copy()
        ImageDraw.Draw(image).rectangle(crop_box(boxes[path], *image.size), fill=CLIP_MEAN_RGB)
        return preprocess(image)
    return apply


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=ROOT.parent)
    p.add_argument("--weights-dir", type=Path, default=ROOT / ".cache" / "clip")
    p.add_argument("--out", type=Path, default=ROOT / "reports" / "val_crop_ablation.json")
    p.add_argument("--batch-size", type=int, default=32, help="small enough for a shared 4 GB GPU")
    a = p.parse_args()

    import torch
    from vps.embed import load_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    partition_file, bbox_file = a.data_root / "Eval/list_eval_partition.txt", a.data_root / "list_bbox_inshop.txt"
    records, boxes = parse_partition(partition_file.read_text()), parse_bbox(bbox_file.read_text())
    val = validation_split(records)
    vq, vg = val["val_query"], val["val_gallery"]
    q_ids, g_ids = [r.item_id for r in vq], [r.item_id for r in vg]
    model, preprocess, meta = load_model(a.weights_dir, device)

    transforms = {n: (make_transform(cfg, preprocess, boxes, **kw), why) for n, (cfg, kw, why) in VARIANTS.items()}
    transforms["X_context_only"] = (context_only(boxes, preprocess), "probe: product bbox hidden, context kept")
    results, hits1 = {}, {}
    for name, (tf, why) in transforms.items():
        t0 = time.perf_counter()
        Q, G = (encode(model, _Split(rs, a.data_root, tf), device, a.batch_size) for rs in (vq, vg))
        rank = retrieve(Q, G, q_ids, g_ids, k=5)["first_pos_rank"]
        hits1[name] = (rank == 1).astype(float)
        results[name] = {"change": why, **recall_at(rank), "mrr": float(np.mean(1 / rank)),
                         "side_view_R@1": float(np.mean([rank[i] == 1 for i, r in enumerate(vq)
                                                          if boxes[r.path].pose_type == "2"])),
                         "seconds": round(time.perf_counter() - t0, 1)}
        print(f"  {name:20s} R@1 {results[name]['R@1']:.2%}  R@5 {results[name]['R@5']:.2%}  "
              f"side R@1 {results[name]['side_view_R@1']:.2%}  ({why})", flush=True)

    ci, idx = bootstrap(hits1)
    vs_c0 = {n: [float(np.percentile((h - hits1["C0_full"])[idx].mean(1), q)) for q in (2.5, 50, 97.5)]
             for n, h in hits1.items() if n != "C0_full"}
    out = {"split": "validation (train item_ids); official test not used", "official_test_used": False,
           "partition_sha256": file_sha256(partition_file), "bbox_sha256": file_sha256(bbox_file),
           "val_query_images": len(vq), "val_gallery_images": len(vg),
           "model": meta["model"], "checkpoint_sha256": meta["checkpoint_sha256"], "device": device,
           "narrow_box_share_val_query": float(np.mean([boxes[r.path].width < 32 for r in vq])),
           "side_view_queries": sum(boxes[r.path].pose_type == "2" for r in vq),
           "results": results, "R@1_bootstrap95": ci,
           "R@1_minus_C0_bootstrap_2.5_50_97.5": vs_c0, "pose_names": POSE_TYPES}
    a.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
