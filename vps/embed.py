"""Frozen OpenAI CLIP ViT-B/32 image embeddings with a validated on-disk cache."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

MODEL_NAME = "ViT-B/32"
CACHE_VERSION = 1


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
    if local.exists():
        got = file_sha256(local)
        if got != sha:
            raise RuntimeError(f"{local} sha256 {got} != official {sha}; refusing to use it")
        model, preprocess = clip.load(str(local), device=device, jit=False)
    else:
        # clip._download verifies the SHA-256 itself
        model, preprocess = clip.load(MODEL_NAME, device=device, jit=False, download_root=str(weights_dir))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, preprocess, {"model": f"openai/CLIP {MODEL_NAME}", "checkpoint_url": url,
                               "checkpoint_sha256": sha, "checkpoint_file": str(local),
                               "preprocess": repr(preprocess)}


def encode_images(model, preprocess, paths: list[Path], device="cpu", batch_size=64, workers=4,
                  log_every=20) -> tuple[np.ndarray, dict]:
    import torch
    from PIL import Image

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            with Image.open(paths[i]) as im:
                return preprocess(im.convert("RGB"))

    dl = torch.utils.data.DataLoader(DS(), batch_size=batch_size, num_workers=workers, shuffle=False)
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


def cache_meta(paths: list[str], annotation_sha256: str, model_meta: dict) -> dict:
    import hashlib
    order = hashlib.sha256("\n".join(paths).encode()).hexdigest()
    return {"cache_version": CACHE_VERSION, "n": len(paths), "order_sha256": order,
            "annotation_sha256": annotation_sha256, "checkpoint_sha256": model_meta["checkpoint_sha256"],
            "preprocess": model_meta["preprocess"]}


def save_cache(cache_dir: Path, name: str, emb: np.ndarray, paths: list[str], meta: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / f"{name}.npy", emb.astype(np.float32))
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
    emb = np.load(f)
    if emb.shape[0] != len(paths):
        return None
    return emb
