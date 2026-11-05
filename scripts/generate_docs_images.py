"""Generate documentation figures.

- docs/images/pipeline.png: diagram only (no dataset pixels) -> committed.
- docs/images/dataset/*.png: built from DeepFashion images -> NOT committed
  (DeepFashion images are restricted to non-commercial research/education and the
  repo is public). Regenerate locally with:

      .venv/Scripts/python.exe scripts/generate_docs_images.py

Retrieval examples come from the real baseline output reports/per_query_results.csv
(frozen CLIP ViT-B/32, full official query set); nothing is hand-ranked.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT.parent  # folder holding img/, Eval/, list_bbox_inshop.txt
sys.path.insert(0, str(ROOT))
from vps.examples import select_examples  # noqa: E402

OUT_DIR = ROOT / "docs" / "images"
DATASET_OUT = OUT_DIR / "dataset"
TILE = 200
LABEL_H = 40
WHITE, BLACK = (255, 255, 255), (0, 0, 0)
RED, GREEN, BLUE, GRAY = (200, 40, 40), (30, 140, 50), (40, 90, 190), (110, 110, 110)


def font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def load_bboxes() -> dict[str, tuple[int, int, int, int]]:
    out = {}
    with open(DATA_ROOT / "list_bbox_inshop.txt", encoding="utf-8") as f:
        next(f); next(f)
        for line in f:
            p = line.split()
            if len(p) == 7:
                out[p[0]] = tuple(int(v) for v in p[3:7])
    return out


def tile(img: Image.Image, label: str, color=BLACK) -> Image.Image:
    img = img.copy()
    img.thumbnail((TILE, TILE), Image.LANCZOS)
    t = Image.new("RGB", (TILE, TILE + LABEL_H), WHITE)
    t.paste(img, ((TILE - img.width) // 2, 0))
    d = ImageDraw.Draw(t)
    d.rectangle([0, 0, TILE - 1, TILE + LABEL_H - 1], outline=color, width=3)
    for i, line in enumerate(label.split("\n")[:2]):
        d.text((6, TILE + 3 + i * 17), line, fill=color, font=font(13))
    return t


def open_img(rel: str) -> Image.Image:
    with Image.open(DATA_ROOT / rel) as im:
        return im.convert("RGB")


def grid(tiles: list[Image.Image], cols: int, name: str, title: str = "") -> None:
    rows = (len(tiles) + cols - 1) // cols
    top = 30 if title else 0
    W, H = TILE + LABEL_H, TILE + LABEL_H
    out = Image.new("RGB", (cols * (TILE + 8) + 8, top + rows * (H + 8) + 8), WHITE)
    if title:
        ImageDraw.Draw(out).text((8, 6), title, fill=BLACK, font=font(15))
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        out.paste(t, (8 + c * (TILE + 8), top + 8 + r * (H + 8)))
    DATASET_OUT.mkdir(parents=True, exist_ok=True)
    out.save(DATASET_OUT / name, optimize=True)
    print("saved", DATASET_OUT / name)


def read_rows() -> list[dict]:
    rows = []
    with open(ROOT / "reports" / "per_query_results.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["path"] = r["query_path"]
            r["first_pos_rank"] = int(r["first_pos_rank"])
            rows.append(r)
    return rows


def retrieval_figure(row: dict, name: str, title: str) -> None:
    tiles = [tile(open_img(row["path"]), f"QUERY\n{row['item_id']}", BLUE)]
    for k in range(1, 6):
        ok = row[f"top{k}_item_id"] == row["item_id"]
        tiles.append(tile(open_img(row[f"top{k}_path"]),
                          f"#{k} {'DUNG' if ok else 'SAI'} {row[f'top{k}_item_id']}\ncos={float(row[f'top{k}_score']):.3f}",
                          GREEN if ok else RED))
    grid(tiles, 6, name, title)


def dataset_figures() -> None:
    bboxes = load_bboxes()
    rows = read_rows()
    picks = select_examples(rows, per_group=1)

    # 1. Same product, different photos (positives share item_id)
    pid = "img/WOMEN/Blouses_Shirts/id_00000001/"
    views = sorted(p for p in bboxes if p.startswith(pid))[:4]
    grid([tile(open_img(p), f"id_00000001\n{Path(p).stem}") for p in views], 4,
         "same-product-example.png", "Cung 1 san pham (cung item_id), anh chup khac nhau")

    # 2. Different products that look alike: a real top-1 confusion from the baseline
    fail = picks["fail_top5"][0]
    grid([tile(open_img(fail["path"]), f"Query\n{fail['item_id']}", BLUE),
          tile(open_img(fail["top1_path"]), f"Top-1 (khac san pham)\n{fail['top1_item_id']}", RED)],
         2, "similar-products.png", "San pham khac nhau nhung CLIP thay giong")

    # 3. Full image vs bbox crop (bbox from list_bbox_inshop.txt)
    rel = pid + "02_1_front.jpg"
    full = open_img(rel)
    x1, y1, x2, y2 = bboxes[rel]
    boxed = full.copy()
    ImageDraw.Draw(boxed).rectangle([x1, y1, x2, y2], outline=RED, width=3)
    grid([tile(boxed, "Anh goc + bbox"), tile(full.crop((x1, y1, x2, y2)), "Vung cat theo bbox")],
         2, "crop-comparison.png", "C0 (anh goc) vs C1 (cat bbox)")

    # 4-5. Real retrieval outcomes (deterministic selection, see vps/examples.py)
    s = picks["success_top1"][0]
    retrieval_figure(s, "retrieval-success.png", f"Thanh cong: dung o hang 1 ({s['category']})")
    retrieval_figure(fail, "retrieval-failure.png",
                     f"That bai: anh dung dau tien o hang {fail['first_pos_rank']} ({fail['category']})")


def arrow(d: ImageDraw.ImageDraw, a, b) -> None:
    d.line([a, b], fill=BLACK, width=2)
    import math
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    l = 10
    p1 = (b[0] - l * math.cos(ang - 0.4), b[1] - l * math.sin(ang - 0.4))
    p2 = (b[0] - l * math.cos(ang + 0.4), b[1] - l * math.sin(ang + 0.4))
    d.polygon([b, p1, p2], fill=BLACK)


def box(d, xy, title, lines, color, width=2):
    d.rectangle(xy, outline=color, width=width)
    d.text((xy[0] + 10, xy[1] + 8), title, fill=color, font=font(16))
    for i, s in enumerate(lines):
        d.text((xy[0] + 10, xy[1] + 34 + i * 20), s, fill=BLACK, font=font(13))


def pipeline_figure() -> None:
    m = json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))
    main = m["main_official_split"]
    img = Image.new("RGB", (820, 500), WHITE)
    d = ImageDraw.Draw(img)
    d.text((250, 18), "Visual Product Retrieval Pipeline", fill=BLACK, font=font(20))
    box(d, (20, 70, 160, 190), "Query", ["1 anh san pham", f"({main['eligible_queries']:,} query)"], BLUE)
    box(d, (20, 240, 160, 360), "Gallery", [f"{m['gallery_images']:,} anh", f"{m['gallery_items']:,} san pham"], BLUE)
    box(d, (210, 70, 400, 190), "CLIP encoder", ["ViT-B/32, frozen", "-> vector 512-d", "+ L2 normalize"], GREEN, 3)
    box(d, (210, 240, 400, 360), "Cung encoder", ["Chay 1 lan cho", "toan bo gallery,", "cache embedding"], GREEN)
    box(d, (450, 140, 630, 290), "Cosine similarity", ["query vs moi anh", "gallery,", "sap xep giam dan"], RED, 3)
    box(d, (670, 140, 805, 290), "Top-5", ["5 anh giong nhat", "", "Dung neu cung", "item_id"], GREEN, 3)
    arrow(d, (160, 130), (210, 130))
    arrow(d, (160, 300), (210, 300))
    arrow(d, (400, 130), (450, 185))
    arrow(d, (400, 300), (450, 245))
    arrow(d, (630, 215), (670, 215))
    d.text((20, 385), "item_id chi dung de cham diem, khong dung khi tim kiem.", fill=GRAY, font=font(13))
    d.text((20, 405), "Recall@K = ti le query co it nhat 1 anh cung item_id trong top-K.", fill=GRAY, font=font(13))
    d.rectangle((20, 440, 805, 480), fill=(250, 248, 215), outline=(200, 190, 110))
    d.text((32, 452),
           f"Ket qua baseline (full official query set): R@1 = {main['R@1']:.1%}, R@5 = {main['R@5']:.1%}",
           fill=(90, 70, 0), font=font(14))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img.save(OUT_DIR / "pipeline.png", optimize=True)
    print("saved", OUT_DIR / "pipeline.png")


if __name__ == "__main__":
    pipeline_figure()
    if (DATA_ROOT / "img").is_dir() and (DATA_ROOT / "list_bbox_inshop.txt").is_file():
        dataset_figures()
    else:
        print("DeepFashion img/ or list_bbox_inshop.txt not found next to repo; skipped dataset figures")
