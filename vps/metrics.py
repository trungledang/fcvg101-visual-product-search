"""Exact cosine retrieval and Recall@K with optional identical-image masking."""
from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(n == 0):
        raise ValueError("zero-norm embedding")
    return x / n


def retrieve(q: np.ndarray, g: np.ndarray, q_ids, g_ids, k: int = 5, batch: int = 1024,
             mask: dict[int, set[int]] | None = None) -> dict:
    """Exact brute-force retrieval over the full gallery.

    q, g: L2-normalised embeddings. mask: query index -> gallery indices to exclude
    (used only for the identical-image sensitivity evaluation).
    Returns top-k indices/scores and the exact 1-based rank of the first positive
    (0 if the query has no eligible positive).
    """
    q_ids = np.asarray(q_ids)
    g_ids = np.asarray(g_ids)
    nq = len(q)
    k = min(k, len(g))
    top_idx = np.zeros((nq, k), dtype=np.int64)
    top_sc = np.zeros((nq, k), dtype=np.float32)
    first_rank = np.zeros(nq, dtype=np.int64)
    for s in range(0, nq, batch):
        e = min(s + batch, nq)
        sims = q[s:e] @ g.T  # cosine = dot product of unit vectors
        if mask:
            for qi in range(s, e):
                for gi in mask.get(qi, ()):
                    sims[qi - s, gi] = -np.inf
        # stable ordering: score desc, then gallery index asc (deterministic tie-break)
        order = np.argsort(-sims, axis=1, kind="stable")
        top_idx[s:e] = order[:, :k]
        top_sc[s:e] = np.take_along_axis(sims, order[:, :k], axis=1)
        pos = (g_ids[order] == q_ids[s:e, None]) & np.isfinite(np.take_along_axis(sims, order, axis=1))
        has = pos.any(axis=1)
        first_rank[s:e] = np.where(has, pos.argmax(axis=1) + 1, 0)
    return {"top_idx": top_idx, "top_scores": top_sc, "first_pos_rank": first_rank}


def recall_at(first_pos_rank: np.ndarray, ks=(1, 5)) -> dict:
    r = np.asarray(first_pos_rank)
    eligible = r > 0
    n = int(eligible.sum())
    out = {"eligible_queries": n, "excluded_no_positive": int((~eligible).sum())}
    for k in ks:
        hits = int(((r > 0) & (r <= k)).sum())
        out[f"R@{k}"] = hits / n if n else float("nan")
        out[f"hits@{k}"] = hits
    return out
