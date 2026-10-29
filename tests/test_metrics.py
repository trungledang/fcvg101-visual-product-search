import numpy as np
import pytest

from vps.metrics import l2_normalize, recall_at, retrieve

# gallery: 5 items on unit axes / mixtures
G = l2_normalize(np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0, 0, 1], [0.7, 0.7, 0]], float))
G_IDS = ["a", "b", "a", "c", "d"]


def test_top5_and_ranks_with_multiple_positives_and_absent():
    Q = l2_normalize(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0.05, 0]], float))
    q_ids = ["a", "b", "x", "c"]
    r = retrieve(Q, G, q_ids, G_IDS, k=5, batch=2)
    # q0 "a": nearest is gallery 0 (a) -> rank 1
    assert r["first_pos_rank"][0] == 1
    # q1 "b": order g2(a), g4(d), g1(b)... -> first positive at rank 3
    assert list(r["top_idx"][1][:3]) == [2, 4, 1]
    assert r["first_pos_rank"][1] == 3
    # q2 "x": no positive in gallery
    assert r["first_pos_rank"][2] == 0
    # q3 "c": g3 is orthogonal -> last
    assert r["first_pos_rank"][3] == 5
    np.testing.assert_allclose(r["top_scores"][0][0], 1.0, rtol=1e-6)
    assert np.all(np.diff(r["top_scores"], axis=1) <= 1e-7)
    m = recall_at(r["first_pos_rank"])
    assert m["eligible_queries"] == 3 and m["excluded_no_positive"] == 1
    assert m["R@1"] == pytest.approx(1 / 3) and m["R@5"] == pytest.approx(1.0)


def test_mask_removes_identical_image_and_can_remove_all_positives():
    Q = l2_normalize(np.array([[1, 0, 0], [0, 0, 1]], float))
    r = retrieve(Q, G, ["a", "c"], G_IDS, k=5, mask={0: {0}, 1: {3}})
    assert 0 not in r["top_idx"][0][:4]
    # remaining order: g1(b,.99), g4(d,.71), g2(a,0) -> rank 3; masked g0 sinks to -inf
    assert r["first_pos_rank"][0] == 3
    assert r["top_idx"][0][4] == 0 and np.isneginf(r["top_scores"][0][4])
    assert r["first_pos_rank"][1] == 0  # only positive was masked -> no eligible positive
    m = recall_at(r["first_pos_rank"])
    assert m["eligible_queries"] == 1


def test_cosine_equals_dot_after_normalisation():
    x = l2_normalize(np.random.default_rng(0).normal(size=(4, 8)))
    np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1, rtol=1e-6)
    with pytest.raises(ValueError):
        l2_normalize(np.zeros((1, 3)))


def test_rejects_unnormalised_or_nonfinite_embeddings():
    with pytest.raises(ValueError, match='L2-normalised'):
        retrieve(np.array([[2, 0, 0]], dtype=np.float32), G, ['a'], G_IDS)
    with pytest.raises(ValueError, match='L2-normalised'):
        retrieve(np.array([[np.nan, 0, 0]], dtype=np.float32), G, ['a'], G_IDS)
