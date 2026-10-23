# FCVG101: Visual Product Search feasibility pilot

**Status (2026-10-23): local GPU preflight complete; retrieval has NOT run.** The Windows
machine reaches CUHK and the official CLIP checkpoint, but the encrypted image release still
requires the dataset owner's approved agreement/password. No Recall@K exists yet. See
`reports/BAO_CAO_PILOT.md` (Vietnamese report).

Purpose: decide whether Visual Product Search is worth pursuing for FCVG101 — not the coursework itself.
Baseline: frozen OpenAI CLIP ViT-B/32, image→image cosine retrieval, DeepFashion In-shop official splits.
No fine-tuning, no extra model, no web UI.

## Setup

### Windows (the local FCVG101 machine)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The PowerShell runner uses batch size 8 for a 4 GB GPU:

```powershell
scripts\run_all.ps1
```

Keep the dataset and all generated images, embeddings, and caches out of Git. The Windows `DataLoader` uses a top-level, picklable dataset class for worker processes.

### Linux/macOS

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt     # resolved versions: requirements.lock.txt
.venv/bin/python -m pytest -q                 # synthetic fixtures: correctness only
```

## Data (not committed — redistribution not permitted)

Official benchmark: https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html
(terms: https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html). Use the original
**In-shop Clothes Retrieval** release (not DeepFashion2 / Category-Attribute / Consumer-to-shop).
From the official download folder take `Eval/list_eval_partition.txt` and `Img/img.zip`, and lay out:

```
data/inshop/Eval/list_eval_partition.txt
data/inshop/img/<MEN|WOMEN>/<category>/id_XXXXXXXX/*.jpg     # unzip img.zip here
```

The parser rejects any other layout/header (e.g. an alternate DeepFashion benchmark).

CLIP checkpoint: downloaded automatically from `openaipublic.azureedge.net` by the pinned
`openai/CLIP` package, which verifies SHA-256 `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`.
If that host is blocked, copy `ViT-B-32.pt` into `.cache/clip/`; the loader refuses it unless the hash matches.

## Run

```bash
scripts/run_all.sh          # tests, preflight, benchmark, audit, embed, evaluate, examples
# or individually:
.venv/bin/python scripts/pilot.py audit
.venv/bin/python scripts/pilot.py embed
.venv/bin/python scripts/pilot.py evaluate [--query-fraction 0.25]   # <1 = labelled PARTIAL estimate
.venv/bin/python scripts/pilot.py examples --per-group 10
```

Google Colab/Kaggle: clone this repo, `pip install -r requirements.txt` (skip the torch pins if the
runtime already has torch), put the data under `data/inshop/` (e.g. unzip from your Drive), then `scripts/run_all.sh`.

## Outputs

| Path | Committed | Content |
|---|---|---|
| `reports/preflight.json`, `reports/benchmark_cpu.json`, `reports/benchmark_local.json` | yes | hardware/software, host reachability, measured throughput |
| `reports/audit.json` | yes | counts, split checks, exact/pixel/pHash duplicate audit |
| `reports/metrics.json`, `reports/per_query_results.csv` | yes | Recall@1/5, sensitivity eval, per-query top-5 paths/IDs/scores/rank |
| `reports/examples_manifest.json` | yes | selected example query paths + selection policy |
| `artifacts/examples/*.png`, `artifacts/image_hashes.csv` | **no** (private) | montages with dataset images; regenerate with `pilot.py examples` |
| `cache/embeddings/` | no | embeddings + order/annotation/model/preprocess manifest |

## Code map

- `vps/annotations.py` – partition parser + split validation
- `vps/audit.py` – file/pixel SHA-256, pHash near-duplicates
- `vps/embed.py` – checksum-verified frozen CLIP loading, `inference_mode` encoding, validated cache
- `vps/metrics.py` – exact batched cosine retrieval, first-positive rank, Recall@K, identical-image masking
- `vps/examples.py` – deterministic success/failure selection and montages
- `scripts/pilot.py` – CLI runner
