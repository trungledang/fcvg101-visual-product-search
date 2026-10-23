import pytest

from vps.annotations import AnnotationError, parse_partition, split_summary, validate_splits

HDR = "image_name item_id evaluation_status"


def make(rows, declared=None):
    return "\n".join([str(len(rows) if declared is None else declared), HDR] + [" ".join(r) for r in rows]) + "\n"


GOOD = [
    ("img/WOMEN/Dresses/id_00000002/02_1_front.jpg", "id_00000002", "train"),
    ("img/WOMEN/Dresses/id_00000003/02_1_front.jpg", "id_00000003", "query"),
    ("img/WOMEN/Dresses/id_00000003/02_2_side.jpg", "id_00000003", "gallery"),
    ("img/MEN/Denim/id_00000004/01_1_front.jpg", "id_00000004", "query"),
]


def test_parse_ok_and_summary():
    recs = parse_partition(make(GOOD))
    assert len(recs) == 4
    assert recs[1].category == "WOMEN/Dresses"
    s = split_summary(recs)
    assert s["query"] == {"images": 2, "unique_items": 2}
    facts = validate_splits(recs)
    assert facts["query_items_without_gallery_positive"] == 1  # id_00000004 has no gallery image


@pytest.mark.parametrize("rows,declared,msg", [
    (GOOD, 5, "declared"),
    (GOOD + [GOOD[0]], None, "duplicate path"),
    ([("img/WOMEN/Dresses/id_00000002/x.jpg", "id_00000002", "val")], None, "invalid split"),
    ([("img/WOMEN/../id_00000002/x/x.jpg", "id_00000002", "train")], None, "unsafe"),
    ([("/img/WOMEN/Dresses/id_00000002/x.jpg", "id_00000002", "train")], None, "unsafe"),
    ([("img/WOMEN/Dresses/id_00000002/x.jpg", "id_2", "train")], None, "invalid item id"),
    ([("img/WOMEN/Dresses/id_00000002/x.jpg", "id_00000009", "train")], None, "!= item id"),
])
def test_parse_rejects(rows, declared, msg):
    with pytest.raises(AnnotationError, match=msg):
        parse_partition(make(rows, declared))


def test_rejects_other_benchmark_header():
    with pytest.raises(AnnotationError, match="header"):
        parse_partition("1\nimage_name evaluation_status\nimg/a.jpg train\n")


def test_split_violations():
    with pytest.raises(AnnotationError, match="train and evaluation"):
        validate_splits(parse_partition(make(GOOD + [("img/WOMEN/Dresses/id_00000002/09_1_front.jpg", "id_00000002", "gallery")])))
