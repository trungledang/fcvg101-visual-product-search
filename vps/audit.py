"""Image-level audit: existence, decoding, exact file/pixel duplicates, perceptual near-duplicates."""
from __future__ import annotations

import hashlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def _hash_one(path: str) -> dict:
    import imagehash
    from PIL import Image
    p = Path(path)
    if not p.is_file():
        return {"path": path, "error": "missing"}
    try:
        data = p.read_bytes()
        with Image.open(p) as im:
            im.load()
            rgb = im.convert("RGB")
        arr = np.asarray(rgb)
        return {"path": path, "file_sha256": hashlib.sha256(data).hexdigest(),
                "pixel_sha256": hashlib.sha256(arr.tobytes() + str(arr.shape).encode()).hexdigest(),
                "phash": str(imagehash.phash(rgb)), "w": rgb.width, "h": rgb.height, "bytes": len(data)}
    except Exception as e:  # noqa: BLE001
        return {"path": path, "error": f"decode: {e.__class__.__name__}: {e}"}


def hash_images(paths: list[str], workers: int = 4) -> list[dict]:
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(_hash_one, paths, chunksize=64))


def exact_groups(rows: list[dict], key: str) -> list[list[int]]:
    d = defaultdict(list)
    for i, r in enumerate(rows):
        if key in r:
            d[r[key]].append(i)
    return [v for v in d.values() if len(v) > 1]


def phash_bits(hexes: list[str]) -> np.ndarray:
    return np.array([[int(c, 16) for c in h] for h in hexes], dtype=np.uint8)


_POP = np.array([bin(i).count("1") for i in range(16)], dtype=np.uint8)


def near_duplicate_pairs(a_hex: list[str], b_hex: list[str], threshold: int, chunk: int = 512) -> list[tuple[int, int, int]]:
    """All (i, j, hamming) with hamming(phash_a[i], phash_b[j]) <= threshold (64-bit pHash)."""
    A, B = phash_bits(a_hex), phash_bits(b_hex)
    out = []
    for s in range(0, len(A), chunk):
        x = A[s:s + chunk, None, :] ^ B[None, :, :]
        d = _POP[x].sum(axis=2, dtype=np.int32)
        ii, jj = np.nonzero(d <= threshold)
        out.extend((int(i + s), int(j), int(d[i, j])) for i, j in zip(ii, jj))
    return out
