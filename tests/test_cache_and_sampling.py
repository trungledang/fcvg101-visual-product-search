import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pilot import sample_queries  # noqa: E402
from vps.annotations import Record  # noqa: E402
from vps.embed import cache_meta, load_cache, save_cache  # noqa: E402
from vps.examples import select_examples  # noqa: E402

MM = {"checkpoint_sha256": "abc", "preprocess": "Compose(...)"}


def test_cache_roundtrip_and_invalidation(tmp_path):
    paths = ["img/a.jpg", "img/b.jpg"]
    emb = np.eye(2, 512, dtype=np.float32)
    meta = cache_meta(paths, "annsha", MM)
    save_cache(tmp_path, "gallery", emb, paths, meta)
    np.testing.assert_array_equal(load_cache(tmp_path, "gallery", paths, meta), emb)
    assert load_cache(tmp_path, "gallery", paths[::-1], cache_meta(paths[::-1], "annsha", MM)) is None  # order
    assert load_cache(tmp_path, "gallery", paths, cache_meta(paths, "other", MM)) is None  # annotation
    assert load_cache(tmp_path, "gallery", paths, cache_meta(paths, "annsha", {**MM, "checkpoint_sha256": "x"})) is None


def test_query_sampling_keeps_every_identity_and_is_deterministic():
    recs = [Record(f"img/MEN/C{i % 3}/id_{i // 4:08d}/{i}.jpg", f"id_{i // 4:08d}", "query") for i in range(40)]
    s1, s2 = sample_queries(recs, 0.3, "s"), sample_queries(recs, 0.3, "s")
    assert s1 == s2
    assert {r.item_id for r in s1} == {r.item_id for r in recs}
    assert sample_queries(recs, 1.0, "s") == recs


def test_example_selection_groups_and_unique_items():
    rows = [{"path": f"p{i}", "item_id": f"id{i % 7}", "category": f"c{i % 3}", "first_pos_rank": [1, 3, 9][i % 3]}
            for i in range(60)]
    sel = select_examples(rows, 5)
    assert all(r["first_pos_rank"] == 1 for r in sel["success_top1"])
    assert all(2 <= r["first_pos_rank"] <= 5 for r in sel["fail_top1_recovered_top5"])
    assert all(r["first_pos_rank"] > 5 for r in sel["fail_top5"])
    for g in sel.values():
        assert len({r["item_id"] for r in g}) == len(g)
    assert select_examples(rows, 5) == sel
