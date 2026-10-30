"""Task 1 data pipeline: bbox parsing, crop/pad transforms, train-identity validation split, HSV features."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .annotations import AnnotationError, Record

BBOX_HEADER = ("image_name", "clothes_type", "pose_type", "x_1", "y_1", "x_2", "y_2")
CLOTHES_TYPES = {"1": "upper-body", "2": "lower-body", "3": "full-body"}
POSE_TYPES = {"1": "front", "2": "side", "3": "back", "4": "zoom-out", "5": "zoom-in", "6": "stand-alone"}
# CLIP normalize mean as 0-255 RGB: padding with it becomes ~0 after normalisation.
CLIP_MEAN_RGB = (123, 117, 104)

# Configurations compared in Task 2. Augmentation applies to training only; evaluation
# always uses the deterministic transform of the same configuration.
CONFIGS = {
    "C0_full": {"crop": False, "train_augment": False, "representation": "clip"},
    "C1_bbox_pad": {"crop": True, "train_augment": False, "representation": "clip"},
    "C2_bbox_pad_aug": {"crop": True, "train_augment": True, "representation": "clip"},
    "C3_hsv_hist": {"crop": True, "train_augment": False, "representation": "hsv"},
}


@dataclass(frozen=True)
class BBox:
    clothes_type: str
    pose_type: str
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


def parse_bbox(text: str, image_size: int = 256) -> dict[str, BBox]:
    """Parse list_bbox_inshop.txt (line 1: count, line 2: header) into path -> BBox."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2 or tuple(lines[1].split()) != BBOX_HEADER:
        raise AnnotationError("unexpected bbox header; expected list_bbox_inshop.txt")
    out: dict[str, BBox] = {}
    for i, ln in enumerate(lines[2:], start=3):
        cols = ln.split()
        if len(cols) != 7:
            raise AnnotationError(f"line {i}: expected 7 columns, got {len(cols)}")
        path, ct, pt, *xy = cols
        if ct not in CLOTHES_TYPES or pt not in POSE_TYPES:
            raise AnnotationError(f"line {i}: invalid clothes/pose type {ct!r}/{pt!r}")
        x1, y1, x2, y2 = map(int, xy)
        if not (0 <= x1 < x2 <= image_size and 0 <= y1 < y2 <= image_size):
            raise AnnotationError(f"line {i}: box {xy} outside {image_size}x{image_size} or empty")
        if path in out:
            raise AnnotationError(f"line {i}: duplicate path {path!r}")
        out[path] = BBox(ct, pt, x1, y1, x2, y2)
    if len(out) != int(lines[0]):
        raise AnnotationError(f"declared {lines[0]} rows but parsed {len(out)}")
    return out


def crop_box(box: BBox, width: int, height: int, margin: float = 0.1,
             min_side: int = 64) -> tuple[int, int, int, int]:
    """Expand the bbox by `margin` per side, grow each side to at least `min_side`
    around its centre (very narrow side-view boxes), then clamp to the image."""
    bw, bh = box.width * (1 + 2 * margin), box.height * (1 + 2 * margin)
    bw, bh = max(bw, min(min_side, width)), max(bh, min(min_side, height))
    cx, cy = (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2
    x1, y1 = cx - bw / 2, cy - bh / 2
    # shift (not shrink) a box that crosses the border, then clamp
    x1 = min(max(x1, 0), width - bw)
    y1 = min(max(y1, 0), height - bh)
    x1, y1 = max(0, round(x1)), max(0, round(y1))
    return x1, y1, min(width, round(x1 + bw)), min(height, round(y1 + bh))


def pad_to_square(image, fill=CLIP_MEAN_RGB):
    """Centre the image on a square canvas so the later centre crop drops nothing."""
    from PIL import Image
    side = max(image.size)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return canvas


class BBoxCropPad:
    """PIL transform: crop to the (expanded) product bbox, then pad to square."""

    def __init__(self, boxes: dict[str, BBox], margin: float = 0.1, min_side: int = 64,
                 fill=CLIP_MEAN_RGB, full_if_narrower: int = 0):
        self.boxes, self.margin, self.min_side = boxes, margin, min_side
        self.fill, self.full_if_narrower = fill, full_if_narrower

    def __call__(self, image, path: str):
        if self.boxes[path].width < self.full_if_narrower:
            return image  # unreliable (side-view) box: keep the full image
        box = crop_box(self.boxes[path], image.width, image.height, self.margin, self.min_side)
        return pad_to_square(image.crop(box), self.fill)


def train_augment(size: int = 224):
    """Training-only augmentation. Hue/saturation stay fixed: colour identifies the product."""
    from torchvision import transforms as T
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.75, 1.0), ratio=(0.9, 1.1),
                            interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2),
    ])


def make_transform(config: str, clip_preprocess, boxes: dict[str, BBox] | None = None,
                   train: bool = False, **crop_args):
    """Return f(PIL image, path) -> CLIP input tensor for a named configuration.
    crop_args (margin, min_side, fill, full_if_narrower) go to BBoxCropPad for val ablations."""
    from torchvision import transforms as T
    cfg = CONFIGS[config]
    if cfg["representation"] != "clip":
        raise ValueError(f"{config} is not a CLIP-input configuration")
    crop = BBoxCropPad(boxes, **crop_args) if cfg["crop"] else None
    if cfg["crop"] and boxes is None:
        raise ValueError(f"{config} needs bbox annotations")
    # RandomErasing works on tensors, so it is appended after CLIP's ToTensor/Normalize.
    after = (T.Compose([clip_preprocess, T.RandomErasing(p=0.25, scale=(0.02, 0.1), value=0)])
             if train and cfg["train_augment"] else clip_preprocess)
    before = train_augment() if train and cfg["train_augment"] else None

    def apply(image, path: str):
        image = image.convert("RGB")
        if crop is not None:
            image = crop(image, path)
        if before is not None:
            image = before(image)
        return after(image)
    return apply


def _rank(seed: str, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def validation_split(records: list[Record], fraction: float = 0.1, seed: str = "fcvg101-val") -> dict:
    """Hold out whole train identities as a val query/gallery set; official test is untouched.

    Only identities with >= 2 images are eligible so each keeps a query and a gallery image.
    Within an identity, images are ordered by hash; the first half (at least one) are queries.
    """
    by_id: dict[str, list[Record]] = {}
    for r in records:
        if r.split == "train":
            by_id.setdefault(r.item_id, []).append(r)
    eligible = sorted((i for i, rs in by_id.items() if len(rs) >= 2), key=lambda i: _rank(seed, i))
    val_ids = set(eligible[:round(len(by_id) * fraction)])
    out = {"train": [], "val_query": [], "val_gallery": []}
    for item_id, rs in by_id.items():
        if item_id not in val_ids:
            out["train"].extend(rs)
            continue
        rs = sorted(rs, key=lambda r: _rank(seed, r.path))
        n_query = max(1, len(rs) // 2)
        out["val_query"].extend(rs[:n_query])
        out["val_gallery"].extend(rs[n_query:])
    return out


def hsv_histogram(image, bins=(8, 4, 4)):
    """Handcrafted colour representation: L2-normalised joint HSV histogram."""
    import numpy as np
    hsv = np.asarray(image.convert("RGB").convert("HSV")).reshape(-1, 3)
    hist, _ = np.histogramdd(hsv, bins=bins, range=((0, 256),) * 3)
    hist = hist.ravel().astype(np.float32)
    return hist / max(np.linalg.norm(hist), 1e-12)
