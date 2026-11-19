"""Build the two coursework archives in artifacts/submission/ (gitignored):

  FCVG101_code.zip            the CMS code ZIP (limit 5 MB): tracked .py/.ipynb (notebook outputs
                              stripped: they hold dataset images), configs, requirements, docs (text
                              only), aggregate reports, MODEL_LINKS.md and the two seed-0 adapters
                              used in the demo (E1, E3; ~0.5 MB each).
  adapter_checkpoints.zip     every 40-epoch adapter used in the test table (E1/E2/E3 x seeds 0-2),
                              for the Drive link in MODEL_LINKS.md.

No dataset image, CLIP weight or embedding cache goes in either archive.
Usage: python scripts/build_submission.py
"""
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "submission"
CKPT = ROOT / "cache" / "adapter" / "ep40"
LIMIT = 5 * 1024 * 1024
SKIP = ("docs/images/", "reports/per_query_results.csv", "reports/history/", "reports/sources/",
        "CLAUDE.md", "CONTRIBUTING.md", "notebooks/kernel-metadata.json",
        "notebooks/learn_clip_retrieval.ipynb", "notebooks/deepfashion_clip_baseline.ipynb",  # học / pilot cũ
        "scripts/build_cloud_notebook.py")
DEMO_CKPTS = ("e1_ratio0.2_seed0.pt", "e3_ratio0.2_seed0.pt")


def strip_outputs(raw: bytes) -> bytes:
    nb = json.loads(raw)
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    return (json.dumps(nb, ensure_ascii=False, indent=1) + "\n").encode("utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True,
                           check=True).stdout.splitlines()
    files = [f for f in files if not f.startswith(SKIP)]

    code_zip = OUT / "FCVG101_code.zip"
    with zipfile.ZipFile(code_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in files:
            data = (ROOT / f).read_bytes()
            z.writestr(f"FCVG101/{f}", strip_outputs(data) if f.endswith(".ipynb") else data)
        for name in DEMO_CKPTS:
            z.write(CKPT / name, f"FCVG101/cache/adapter/ep40/{name}")
    size = code_zip.stat().st_size
    print(f"{code_zip.name}: {len(files)} files + {len(DEMO_CKPTS)} checkpoints, {size / 1e6:.2f} MB")
    assert size <= LIMIT, f"code ZIP is {size} bytes, over the 5 MB limit"

    ck_zip = OUT / "adapter_checkpoints.zip"
    names = sorted(p.name for p in CKPT.glob("e[123]_ratio0.2_seed[012].pt"))
    assert len(names) == 9, names
    with zipfile.ZipFile(ck_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.write(CKPT / n, f"ep40/{n}")
    print(f"{ck_zip.name}: {len(names)} checkpoints, {ck_zip.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
