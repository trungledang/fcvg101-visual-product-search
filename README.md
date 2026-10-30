# FCVG101 — frozen CLIP product retrieval feasibility baseline

**Completed on Kaggle GPU, 2026-10-23. Recommendation: Go.**
Frozen OpenAI CLIP ViT-B/32, full images, exact cosine retrieval, full official In-shop query/gallery split.

| Metric | Measured result |
|---|---:|
| Recall@1 | **45.5057%** — 6,470 / 14,218 |
| Recall@5 | **67.2457%** — 9,561 / 14,218 |
| Gallery | 12,612 images / 3,985 products |
| Embedding extraction | 56.19 s for all 26,830 evaluation images |
| Exact retrieval | 31.51 s |

All 14,218 queries and all 23 categories were evaluated; gallery distractors were never subsampled.
No query was excluded. Independent recomputation reproduced both metrics and every query's hit/miss
classification at K=1 and K=5. See [report](reports/BAO_CAO_PILOT.md), [metrics](reports/metrics.json),
[verification](reports/independent_verification.json), and [30 inspected examples](artifacts/examples/index.html).

## Notebook học bằng tiếng Việt — bắt đầu tại đây

Mở [notebooks/learn_clip_retrieval.ipynb](notebooks/learn_clip_retrieval.ipynb) trong VS Code/Jupyter,
chọn kernel Python `.venv/Scripts/python.exe` của dự án. Đọc/chạy **8 ô theo thứ tự từ trên xuống**:
ô 1 giới thiệu; ô 2 đọc partition/cache; ô 3 giải thích tensor; ô 4 xử lý và encode một ảnh;
ô 5 hiển thị cosine/Top-5; ô 6 giải thích Recall; ô 7 xem hai ca thật; ô 8 bàn về crop/metric learning.
Không cần chạy lại `scripts/pilot.py` hay notebook benchmark. `vps/learning.py` chỉ là tiện ích đọc cache/vẽ hình,
được import tự động; các module baseline giữ nguyên. Bộ chạy `ipykernel` đã được cài trong `.venv` hiện tại.
Nếu dựng lại môi trường cho VS Code, cài thêm `.venv/Scripts/python.exe -m pip install ipykernel`.
Notebook tự nhận thư mục dự án khi mở từ project, `notebooks/` hoặc thư mục môn học;
`DATA = ROOT.parent` trỏ đến `../img/` và `../Eval/list_eval_partition.txt`. Chỉ sửa `DATA` nếu chuyển dữ liệu.
Cache và checkpoint được dùng lại, không ghi đè. [Bản có output chạy thử](artifacts/learning/learn_clip_retrieval.executed.ipynb)
được lưu riêng trong `artifacts/` (chứa ảnh dataset, không đưa vào Git).

## Task 1 coursework — pipeline dữ liệu và biểu diễn ảnh

[notebooks/task1_data_pipeline.ipynb](notebooks/task1_data_pipeline.ipynb), cùng kernel `.venv`, chạy từ trên xuống (~2 phút).
Gồm phân bố dữ liệu, chất lượng bbox, split train/val theo item_id (official test không bị chạm), pipeline
crop bbox → pad vuông → CLIP, augmentation chỉ khi train, HSV histogram so với CLIP, và 4 cấu hình C0–C3 cho Task 2.
Code ở `vps/preprocess.py` (test: `tests/test_preprocess.py`); notebook được sinh bởi `scripts/build_task1_notebook.py`.
Bản có output ảnh nằm ở `artifacts/task1/` (không đưa vào Git).

## Run or inspect on Kaggle

The completed **private** [Kaggle notebook](https://www.kaggle.com/code/dangtrung2005/fcvg101-deepfashion-frozen-clip-baseline)
contains the executed experiment and all 30 displayed retrieval cases.
[Portable notebook](notebooks/deepfashion_clip_baseline.ipynb) embeds the Python source/tests, not images or weights.
Attach your [private input dataset](https://www.kaggle.com/datasets/dangtrung2005/fcvg101-inshop-private-inputs)
version 2, enable Internet and a free GPU, then Run All.

The input contains `img_archive.bin` (the **unchanged bytes** of the inspected `img.zip`),
`list_eval_partition.txt`, `list_bbox_inshop.txt`, and the official README.
Kaggle automatically expands `.zip` uploads into remote files; preserving the archive as `.bin`
lets the notebook extract images onto its local scratch disk and avoids slow remote directory walks.
Checksums enforce the same inputs. Keep the dataset and generated image examples private.

The measured runtime was Python 3.13.15, torch 2.11.0+cu128, torchvision 0.26.0+cu128,
NumPy 2.1.3 and Pillow 13.0.0. The notebook preserves the runtime's torch installation.
The resolved environment is in [environment.freeze.txt](reports/environment.freeze.txt);
the exact Kaggle container image is recorded in [kernel metadata](reports/kernel-metadata.executed.json).
The old local dependency lock is retained as the separate Windows environment, not the cloud lock.

For Colab fallback, open the portable notebook, select a GPU, mount Drive, and set `INPUT_ROOT`
to the directory holding your three files. This path is prepared but **Colab was not executed**.

## Existing Windows workspace

The extracted files are at `../img/`, `../Eval/list_eval_partition.txt`, and
`../list_bbox_inshop.txt` relative to this project. The partition was copied from Downloads without modification.
The original archive remains in `E:/Downloads/Chua phan loai/img.zip`.
To re-run locally from this directory with the existing `.venv`:

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe scripts/pilot.py --data-root .. audit --include-train --workers 2
.venv/Scripts/python.exe scripts/pilot.py --data-root .. embed --batch-size 8 --workers 2 --threads 2
.venv/Scripts/python.exe scripts/pilot.py --data-root .. evaluate
.venv/Scripts/python.exe scripts/pilot.py --data-root .. examples --per-group 10
.venv/Scripts/python.exe scripts/make_example_index.py
```

Dataset audit and checkpoint loading were run on Windows; full image extraction/evaluation was run on Kaggle.
Kaggle caches carry source file size/mtime fingerprints, so a local `embed` may conservatively recompute them.
The independent metric check uses the downloaded embeddings directly:

```powershell
.venv/Scripts/python.exe scripts/verify_results.py
```

## Protocol and files

- [Official benchmark](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html),
  [official README snapshot](reports/sources/README.inshop.official.txt),
  [partition provenance](reports/sources/partition_provenance.json).
- [Input inspection](reports/input_inspection.json): archive paths/counts/checksum, CRC match for every
  extracted image, bbox schema/count/path agreement. Full images are used; no bbox crop.
- [Audit](reports/audit.json): train/query/gallery counts, identity separation, complete decoding,
  byte/pixel duplicates, pHash candidates. No exact duplicates were found in any split.
- [Embedding timing](reports/embed_timing.json): official checkpoint SHA-256, exact pretrained preprocessing,
  frozen/eval state, precision, throughput, memory and cache size.
- [Per-query CSV](reports/per_query_results.csv): product ID, positive count/rank, top-five paths/IDs/cosine scores.
- [Example manifest](reports/examples_manifest.json): deterministic outcome/category/identity selection.
  Balanced groups are diagnostic; the selected 10 categories are not all 23 benchmark categories.
- `artifacts/` and `cache/embeddings/` are private and Git-ignored. `reports/history/` preserves the superseded
  preflight-only report and local checks; the old random-weight throughput figures are not baseline measurements.

The cache checks annotation, image order, source fingerprint, checkpoint, preprocessing versions,
encoder precision, payload checksum, dimensions, finiteness and unit norms.
Evaluation requires a matching completed image audit. Exact score ties use gallery order.
Tiny float32 ranking differences can occur across BLAS implementations; all hit/miss labels at K=1/5
matched in the independent Windows check.

This benchmark measures in-shop studio retrieval. It does not establish consumer phone-photo performance,
and overlap with CLIP's unknown original pretraining images cannot be ruled out.
