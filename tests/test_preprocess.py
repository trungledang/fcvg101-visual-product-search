import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from PIL import Image

from vps.annotations import AnnotationError, Record
from vps.preprocess import (BBox, BBoxCropPad, CLIP_MEAN_RGB, crop_box, hsv_histogram, pad_to_square,
                            parse_bbox, validation_split)

HEADER = "image_name clothes_type pose_type x_1 y_1 x_2 y_2"
P = "img/WOMEN/Dresses/id_00000002/02_1_front.jpg"


def test_parse_bbox_ok():
    boxes = parse_bbox(f"1\n{HEADER}\n{P}   3 1 050 049 208 235\n")
    assert boxes[P] == BBox("3", "1", 50, 49, 208, 235)
    assert boxes[P].width == 158 and boxes[P].height == 186


@pytest.mark.parametrize("row,msg", [
    (f"{P} 3 1 50 49 50 235", "outside"),        # empty width
    (f"{P} 3 1 50 49 208 300", "outside"),       # beyond 256
    (f"{P} 4 1 50 49 208 235", "invalid"),       # unknown clothes type
    (f"{P} 3 1 50 49 208", "7 columns"),
])
def test_parse_bbox_rejects(row, msg):
    with pytest.raises(AnnotationError, match=msg):
        parse_bbox(f"1\n{HEADER}\n{row}\n")


def test_parse_bbox_rejects_wrong_header_and_count():
    with pytest.raises(AnnotationError, match="header"):
        parse_bbox(f"1\nimage_name item_id evaluation_status\n{P} 3 1 1 1 2 2\n")
    with pytest.raises(AnnotationError, match="declared 2"):
        parse_bbox(f"2\n{HEADER}\n{P} 3 1 1 1 2 2\n")


def test_crop_box_margin_and_clamp():
    # 100x100 box at (100,100) with 10% margin per side -> 120x120 centred
    assert crop_box(BBox("1", "1", 100, 100, 200, 200), 256, 256, margin=0.1, min_side=0) == (90, 90, 210, 210)
    # box at the border is shifted inside, keeping its size
    x1, y1, x2, y2 = crop_box(BBox("1", "1", 0, 0, 100, 100), 256, 256, margin=0.1, min_side=0)
    assert (x1, y1) == (0, 0) and (x2 - x1, y2 - y1) == (120, 120)


def test_crop_box_grows_narrow_side_view_box():
    x1, y1, x2, y2 = crop_box(BBox("1", "2", 119, 48, 136, 234), 256, 256, margin=0.1, min_side=64)
    assert x2 - x1 == 64 and x1 <= 119 and x2 >= 136
    assert 0 <= x1 and x2 <= 256 and 0 <= y1 and y2 <= 256


def test_pad_to_square_keeps_content_centred():
    img = Image.new("RGB", (40, 100), (255, 0, 0))
    sq = pad_to_square(img)
    assert sq.size == (100, 100)
    assert sq.getpixel((50, 50)) == (255, 0, 0) and sq.getpixel((0, 50)) == CLIP_MEAN_RGB


def test_bbox_crop_pad_transform():
    img = Image.new("RGB", (256, 256), (0, 0, 255))
    out = BBoxCropPad({P: BBox("3", "1", 50, 20, 150, 220)}, margin=0.0, min_side=0)(img, P)
    assert out.size == (200, 200)
    white = BBoxCropPad({P: BBox("3", "1", 50, 20, 150, 220)}, margin=0.0, min_side=0, fill=(255, 255, 255))(img, P)
    assert white.getpixel((0, 100)) == (255, 255, 255)


def test_bbox_crop_pad_falls_back_to_full_image_for_narrow_box():
    img = Image.new("RGB", (256, 256))
    assert BBoxCropPad({P: BBox("3", "2", 120, 40, 136, 230)}, full_if_narrower=32)(img, P) is img


def _records():
    rs = []
    for i in range(40):
        item = f"id_{i:08d}"
        for j in range(1 if i < 4 else 2 + i % 5):
            rs.append(Record(f"img/MEN/Denim/{item}/{j:02d}_1_front.jpg", item, "train"))
    rs.append(Record("img/MEN/Denim/id_00009999/01_1_front.jpg", "id_00009999", "query"))
    return rs


def test_validation_split_holds_out_whole_train_identities():
    rs = _records()
    s = validation_split(rs, fraction=0.25, seed="t")
    ids = {k: {r.item_id for r in v} for k, v in s.items()}
    assert ids["val_query"] == ids["val_gallery"] and len(ids["val_query"]) == 10
    assert not ids["train"] & ids["val_query"]
    assert "id_00009999" not in set().union(*ids.values())        # official eval split untouched
    assert all(i >= "id_00000004" for i in ids["val_query"])      # single-image ids stay in train
    assert not {r.path for r in s["val_query"]} & {r.path for r in s["val_gallery"]}
    assert sum(map(len, s.values())) == len(rs) - 1
    assert validation_split(rs, fraction=0.25, seed="t") == s     # deterministic


def test_hsv_histogram_is_unit_and_colour_sensitive():
    red = hsv_histogram(Image.new("RGB", (8, 8), (255, 0, 0)))
    blue = hsv_histogram(Image.new("RGB", (8, 8), (0, 0, 255)))
    assert red.shape == (128,) and np.isclose(np.linalg.norm(red), 1)
    assert float(red @ blue) == 0.0
