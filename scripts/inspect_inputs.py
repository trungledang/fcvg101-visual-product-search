"""Inspect the user's original archive, extracted images, partition and bbox file."""
import argparse
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vps.annotations import file_sha256, parse_partition, split_summary, validate_splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--archive', type=Path, required=True)
    ap.add_argument('--bbox', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    part = a.data_root / 'Eval/list_eval_partition.txt'
    records = parse_partition(part.read_text())
    paths = {r.path for r in records}
    bbox = a.bbox.read_text().splitlines()
    assert bbox[1].split() == ['image_name', 'clothes_type', 'pose_type', 'x_1', 'y_1', 'x_2', 'y_2']
    boxes = {}
    clothes, poses = Counter(), Counter()
    for line in bbox[2:]:
        name, ct, pt, *coords = line.split()
        assert name not in boxes and len(coords) == 4
        x1, y1, x2, y2 = map(int, coords)
        assert ct in ('1', '2', '3') and pt in ('1', '2', '3', '4', '5', '6')
        assert 0 <= x1 < x2 <= 256 and 0 <= y1 < y2 <= 256, (name, coords)
        boxes[name] = coords
        clothes[ct] += 1
        poses[pt] += 1
    assert len(boxes) == int(bbox[0]) and set(boxes) == paths
    with zipfile.ZipFile(a.archive) as z:
        members = [x for x in z.infolist() if not x.is_dir()]
        assert len({x.filename for x in members}) == len(members)
        assert {x.filename for x in members} == paths
        assert not any(x.flag_bits & 1 for x in members), 'encrypted archive'
        import zlib
        mismatches = []
        for member in members:
            p = a.data_root / member.filename
            if not p.is_file() or zlib.crc32(p.read_bytes()) != member.CRC:
                mismatches.append(member.filename)
        assert not mismatches, mismatches[:10]
        archive = {'entries': len(z.infolist()), 'images': len(members), 'encrypted': False,
                   'uncompressed_bytes': sum(x.file_size for x in members),
                   'extracted_files_matching_archive_crc32': len(members),
                   'readme_in_archive': any('readme' in x.filename.lower() for x in members)}
    extracted = {p.relative_to(a.data_root).as_posix() for p in (a.data_root / 'img').rglob('*.jpg')}
    assert extracted == paths
    result = {'archive': {**archive, 'source': str(a.archive), 'sha256': file_sha256(a.archive)},
              'partition': {'source': str(part), 'sha256': file_sha256(part), 'splits': split_summary(records),
                            'checks': validate_splits(records)},
              'bbox': {'source': str(a.bbox), 'sha256': file_sha256(a.bbox), 'rows': len(boxes),
                       'paths_equal_partition': True, 'clothes_types': clothes, 'pose_types': poses,
                       'used_for_baseline_cropping': False},
              'readme': {'url': 'https://drive.google.com/file/d/0B7EVK8r0v71pSXp5b2R0UUg3U0E/view',
                         'local_snapshot': 'reports/sources/README.inshop.official.txt',
                         'access': 'direct download from official project-linked Drive release',
                         'retrieved': '2026-10-23'}}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
