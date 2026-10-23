"""End-to-end plumbing test on a synthetic fixture. Tests correctness only; not a dataset result."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from vps.annotations import file_sha256, parse_partition
from vps.embed import cache_meta, official_checkpoint, save_cache

ROOT = Path(__file__).resolve().parents[1]


def test_audit_evaluate_examples(tmp_path):
    data = tmp_path / "data"
    rng = np.random.default_rng(0)
    rows = []
    for item in range(6):
        iid = f"id_{item:08d}"
        cat = ["MEN/Denim", "WOMEN/Dresses"][item % 2]
        split_imgs = [("train", 1)] if item == 5 else [("query", 2), ("gallery", 2)]
        k = 0
        for split, n in split_imgs:
            for _ in range(n):
                rel = f"img/{cat}/{iid}/{k:02d}_1_front.jpg"
                (data / rel).parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rng.integers(0, 255, (64, 48, 3), dtype=np.uint8)).save(data / rel)
                rows.append(f"{rel} {iid} {split}")
                k += 1
    # one gallery image byte-identical to a query image -> exercised by sensitivity eval
    q0 = data / "img/MEN/Denim/id_00000000/00_1_front.jpg"
    (data / "img/MEN/Denim/id_00000000/02_1_front.jpg").write_bytes(q0.read_bytes())
    (data / "Eval").mkdir()
    (data / "Eval/list_eval_partition.txt").write_text(f"{len(rows)}\nimage_name item_id evaluation_status\n" + "\n".join(rows) + "\n")

    recs = parse_partition((data / "Eval/list_eval_partition.txt").read_text())
    ann = file_sha256(data / "Eval/list_eval_partition.txt")
    mm = {"checkpoint_sha256": official_checkpoint()[1], "preprocess": "fake"}
    cache = tmp_path / "cache"
    for split in ("query", "gallery"):
        ps = [r.path for r in recs if r.split == split]
        ids = [int(r.item_id[-2:]) for r in recs if r.split == split]
        emb = np.eye(8, 512, dtype=np.float32)[ids] + 0.01 * rng.normal(size=(len(ps), 512)).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        save_cache(cache, split, emb, ps, cache_meta(ps, ann, mm))

    common = [sys.executable, str(ROOT / "scripts/pilot.py"), "--data-root", str(data), "--cache-dir", str(cache),
              "--reports-dir", str(tmp_path / "rep"), "--artifacts-dir", str(tmp_path / "art")]
    for cmd in (["audit", "--workers", "1"], ["evaluate"], ["examples", "--per-group", "3"]):
        subprocess.run(common + cmd, check=True, cwd=tmp_path)
    audit = json.loads((tmp_path / "rep/audit.json").read_text())
    assert audit["split_summary"]["query"] == {"images": 10, "unique_items": 5}
    assert len(audit["exact_file_duplicate_groups_cross_query_gallery"]) == 1
    m = json.loads((tmp_path / "rep/metrics.json").read_text())
    assert m["main_official_split"]["R@1"] == 1.0 and m["main_official_split"]["eligible_queries"] == 10
    s = m["sensitivity_identical_images_masked"]
    assert s["queries_with_masked_identical_gallery_images"] == 1 and s["queries_losing_all_positives"] == 0
    man = json.loads((tmp_path / "rep/examples_manifest.json").read_text())
    assert len(man["groups"]["success_top1"]) == 3 and man["groups"]["fail_top5_shortfall"] == 3
    assert (tmp_path / "art/examples/success_top1_01.png").exists()
