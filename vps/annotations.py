"""Parse and validate the official In-shop Eval/list_eval_partition.txt."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

ALLOWED_SPLITS = ("train", "query", "gallery")
ITEM_ID_RE = re.compile(r"^id_\d{8}$")
EXPECTED_HEADER = ("image_name", "item_id", "evaluation_status")


class AnnotationError(ValueError):
    pass


@dataclass(frozen=True)
class Record:
    path: str          # relative POSIX path as written in the annotation, e.g. img/WOMEN/Dresses/id_00000002/02_1_front.jpg
    item_id: str
    split: str

    @property
    def gender(self) -> str:
        return PurePosixPath(self.path).parts[1]

    @property
    def category(self) -> str:
        parts = PurePosixPath(self.path).parts
        return f"{parts[1]}/{parts[2]}"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_path(p: str, lineno: int) -> None:
    pp = PurePosixPath(p)
    if pp.is_absolute() or "\\" in p or any(part in ("..", ".", "") for part in pp.parts):
        raise AnnotationError(f"line {lineno}: unsafe path {p!r}")
    if len(pp.parts) != 5 or pp.parts[0] != "img" or pp.parts[1] not in ("MEN", "WOMEN"):
        raise AnnotationError(f"line {lineno}: unexpected path layout {p!r} (expected img/<MEN|WOMEN>/<cat>/<id>/<file>)")
    if pp.suffix.lower() != ".jpg":
        raise AnnotationError(f"line {lineno}: non-jpg path {p!r}")


def parse_partition(text: str) -> list[Record]:
    """Parse list_eval_partition.txt. Line 1: declared count; line 2: header; then rows."""
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    if len(lines) < 2:
        raise AnnotationError("file too short")
    try:
        declared = int(lines[0])
    except ValueError as e:
        raise AnnotationError(f"first line is not a count: {lines[0]!r}") from e
    header = tuple(lines[1].split())
    if header != EXPECTED_HEADER:
        raise AnnotationError(f"unexpected header {header!r}; this may be a different DeepFashion benchmark")
    records: list[Record] = []
    seen: set[str] = set()
    for i, ln in enumerate(lines[2:], start=3):
        cols = ln.split()
        if len(cols) != 3:
            raise AnnotationError(f"line {i}: expected 3 columns, got {len(cols)}")
        path, item_id, split = cols
        _check_path(path, i)
        if not ITEM_ID_RE.match(item_id):
            raise AnnotationError(f"line {i}: invalid item id {item_id!r}")
        if PurePosixPath(path).parts[3] != item_id:
            raise AnnotationError(f"line {i}: path folder {PurePosixPath(path).parts[3]!r} != item id {item_id!r}")
        if split not in ALLOWED_SPLITS:
            raise AnnotationError(f"line {i}: invalid split {split!r}")
        if path in seen:
            raise AnnotationError(f"line {i}: duplicate path {path!r}")
        seen.add(path)
        records.append(Record(path, item_id, split))
    if len(records) != declared:
        raise AnnotationError(f"declared {declared} rows but parsed {len(records)}")
    return records


def split_summary(records: list[Record]) -> dict:
    out = {}
    for s in ALLOWED_SPLITS:
        rs = [r for r in records if r.split == s]
        out[s] = {"images": len(rs), "unique_items": len({r.item_id for r in rs})}
    return out


def validate_splits(records: list[Record]) -> dict:
    """Structural split checks. Raises AnnotationError on hard violations; returns facts."""
    by = {s: [r for r in records if r.split == s] for s in ALLOWED_SPLITS}
    qpaths, gpaths = {r.path for r in by["query"]}, {r.path for r in by["gallery"]}
    if qpaths & gpaths:
        raise AnnotationError(f"{len(qpaths & gpaths)} paths appear in both query and gallery")
    train_ids = {r.item_id for r in by["train"]}
    eval_ids = {r.item_id for r in by["query"]} | {r.item_id for r in by["gallery"]}
    overlap = train_ids & eval_ids
    if overlap:
        raise AnnotationError(f"{len(overlap)} item ids in both train and evaluation splits")
    gallery_ids = {r.item_id for r in by["gallery"]}
    query_ids = {r.item_id for r in by["query"]}
    no_pos = sorted({r.item_id for r in by["query"] if r.item_id not in gallery_ids})
    return {
        "query_gallery_path_overlap": 0,
        "train_eval_identity_overlap": 0,
        "query_items_without_gallery_positive": len(no_pos),
        "query_items_without_gallery_positive_examples": no_pos[:20],
        "gallery_items_without_query": len(gallery_ids - query_ids),
    }
