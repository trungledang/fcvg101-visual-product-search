"""Build the coursework code ZIP (limit 5 MB): artifacts/submission/FCVG101_code.zip

Contents: the main notebook (outputs stripped, so it re-runs to full output), the Task 3 notebook if present
(notebooks/task3*.ipynb, executed copies excluded), requirements.txt and a short README_RUN.md.
No dataset image, CLIP weight or embedding cache goes in. Usage: python scripts/build_submission.py
"""
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "submission"
LIMIT = 5 * 1024 * 1024

README = """FCVG101 - Tim san pham thoi trang bang anh (DeepFashion In-shop)

FCVG101.ipynb          notebook chinh (Task 1 + Task 2), chay tu tren xuong
task3*.ipynb           Task 3 (neu co), thi nghiem doc lap
requirements.txt       thu vien

Chay: pip install -r requirements.txt, dat img/, Eval/, list_bbox_inshop.txt (DeepFashion In-shop) o thu muc cha,
mo FCVG101.ipynb va Run all. CLIP ViT-B/32 tu tai ve. Chi tiet Colab, link checkpoint: MODEL_LINKS.md.
Checkpoint adapter: https://drive.google.com/file/d/1XWptbtHuBulXwOthm_7qVEHB8puaAw7c/view
"""


def strip_outputs(raw: bytes) -> bytes:
    nb = json.loads(raw)
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    return (json.dumps(nb, ensure_ascii=False, indent=1) + "\n").encode("utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    notebooks = [ROOT / "FCVG101.ipynb"] + sorted(p for p in (ROOT / "notebooks").glob("task3*.ipynb")
                                                  if ".executed" not in p.name)
    path = OUT / "FCVG101_code.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for nb in notebooks:
            z.writestr(f"FCVG101/{nb.name}", strip_outputs(nb.read_bytes()))
        z.write(ROOT / "requirements.txt", "FCVG101/requirements.txt")
        z.write(ROOT / "MODEL_LINKS.md", "FCVG101/MODEL_LINKS.md")
        z.writestr("FCVG101/README_RUN.md", README)
    size = path.stat().st_size
    print(f"{path.name}: {[n.name for n in notebooks]} + requirements + links, {size / 1e6:.2f} MB")
    assert size <= LIMIT, f"{size} bytes is over the 5 MB limit"


if __name__ == "__main__":
    main()
