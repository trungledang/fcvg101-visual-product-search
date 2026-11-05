"""Frozen OpenAI CLIP ViT-B/32 image embeddings with a validated on-disk cache."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

MODEL_NAME = "ViT-B/32"
CACHE_VERSION = 2


def source_fingerprint(root: Path, paths: list[str]) -> str:
    """Invalidate after replacing an input file; the audit holds its content hashes."""
    import hashlib
    h = hashlib.sha256()
    for path in paths:
        stat = (root / path).stat()
        h.update(f"{path}\t{stat.st_size}\t{stat.st_mtime_ns}\n".encode())
    return h.hexdigest()


class _ImageDataset:
    """Top-level so PyTorch DataLoader workers can pickle it on Windows."""

    def __init__(self, paths: list[Path], preprocess):
        self.paths = paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        with Image.open(self.paths[i]) as im:
            return self.preprocess(im.convert("RGB"))


def official_checkpoint() -> tuple[str, str]:
    """(url, sha256) for ViT-B/32 as published in the pinned openai/CLIP package."""
    import clip
    url = clip.clip._MODELS[MODEL_NAME]
    return url, url.split("/")[-2]


def load_model(weights_dir: Path, device: str = "cpu"):
    """Load the official OpenAI checkpoint (download if reachable) and verify its SHA-256.

    If the file was placed manually in weights_dir (e.g. copied from another machine),
    it is accepted only when its SHA-256 equals the official hash embedded in the
    openai/CLIP download URL.
    """
    import clip
    import torch
    from .annotations import file_sha256

    url, sha = official_checkpoint()
    weights_dir.mkdir(parents=True, exist_ok=True)
    local = weights_dir / Path(url).name
    download_seconds = 0.0
    if local.exists():
        got = file_sha256(local)
        if got != sha:
            raise RuntimeError(f"{local} sha256 {got} != official {sha}; refusing to use it")
        model, preprocess = clip.load(str(local), device=device, jit=False)
    else:
        # clip._download verifies the SHA-256 itself
        t_download = time.perf_counter()
        clip.clip._download(url, str(weights_dir))
        download_seconds = time.perf_counter() - t_download
        model, preprocess = clip.load(str(local), device=device, jit=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    import re
    import PIL
    import torchvision
    # Compose repr includes a Python function address, which changes each process.
    transform = re.sub(r" at 0x[0-9a-fA-F]+", "", repr(preprocess))
    return model, preprocess, {"model": f"openai/CLIP {MODEL_NAME}", "checkpoint_url": url,
                               "checkpoint_sha256": sha, "checkpoint_file": str(local),
                               "checkpoint_download_seconds": download_seconds,
                               "preprocess": transform,
                               "preprocess_versions": {"torchvision": torchvision.__version__, "pillow": PIL.__version__},
                               "encoder_dtype": str(model.dtype),
                               "frozen": all(not p.requires_grad for p in model.parameters()),
                               "training": model.training}


def encode_images(model, preprocess, paths: list[Path], device="cpu", batch_size=64, workers=4,
                  log_every=20) -> tuple[np.ndarray, dict]:
    import torch

    dl = torch.utils.data.DataLoader(_ImageDataset(paths, preprocess), batch_size=batch_size,
                                     num_workers=workers, shuffle=False)
    out = []
    t0 = time.perf_counter()
    with torch.inference_mode():
        for bi, x in enumerate(dl):
            f = model.encode_image(x.to(device)).float()
            f = f / f.norm(dim=-1, keepdim=True)
            out.append(f.cpu().numpy())
            if log_every and bi % log_every == 0:
                done = min((bi + 1) * batch_size, len(paths))
                el = time.perf_counter() - t0
                print(f"  encoded {done}/{len(paths)} images, {done / el:.1f} img/s", flush=True)
    secs = time.perf_counter() - t0
    emb = np.concatenate(out) if out else np.zeros((0, 512), np.float32)
    return emb, {"images": len(paths), "seconds": secs, "images_per_second": len(paths) / secs if secs else None,
                 "batch_size": batch_size, "workers": workers, "device": device}


def cache_meta(paths: list[str], annotation_sha256: str, model_meta: dict, source_sha256=None) -> dict:
    import hashlib
    order = hashlib.sha256("\n".join(paths).encode()).hexdigest()
    return {"cache_version": CACHE_VERSION, "n": len(paths), "order_sha256": order,
            "annotation_sha256": annotation_sha256, "checkpoint_sha256": model_meta["checkpoint_sha256"],
            "preprocess": model_meta["preprocess"],
            "preprocess_versions": model_meta.get("preprocess_versions"), "source_fingerprint": source_sha256,
            "encoder_dtype": model_meta.get("encoder_dtype")}


def save_cache(cache_dir: Path, name: str, emb: np.ndarray, paths: list[str], meta: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / f"{name}.npy", emb.astype(np.float32))
    from .annotations import file_sha256
    (cache_dir / f"{name}.npy.sha256").write_text(file_sha256(cache_dir / f"{name}.npy"))
    (cache_dir / f"{name}.paths.txt").write_text("\n".join(paths) + "\n")
    (cache_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=2))


def load_cache(cache_dir: Path, name: str, paths: list[str], meta: dict) -> np.ndarray | None:
    """Return cached embeddings only if order, annotation, model and preprocessing all match."""
    f, m, p = (cache_dir / f"{name}.npy", cache_dir / f"{name}.meta.json", cache_dir / f"{name}.paths.txt")
    if not (f.exists() and m.exists() and p.exists()):
        return None
    if json.loads(m.read_text()) != meta or p.read_text().splitlines() != paths:
        print(f"cache {name} incompatible; recomputing")
        return None
    from .annotations import file_sha256
    checksum = cache_dir / f"{name}.npy.sha256"
    if not checksum.exists() or checksum.read_text().strip() != file_sha256(f):
        return None
    emb = np.load(f, allow_pickle=False)
    if emb.shape != (len(paths), 512) or not np.isfinite(emb).all():
        return None
    if len(emb) and not np.allclose(np.linalg.norm(emb, axis=1), 1, atol=1e-4):
        return None
    return emb
