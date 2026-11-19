"""Ghép notebooks/FCVG101_main.ipynb — MỘT notebook cho cả Task 1 và Task 2, chạy trên Colab hoặc máy cá nhân.

Ô 0 là cấu hình môi trường; phần A (Task 1) lấy từ scripts/nb_cells_task1.py, phần B (Task 2) từ
scripts/nb_cells_task2.py; phụ lục cuối chạy lại toàn bộ từ đầu nếu bật RUN_FULL.
Chạy: python scripts/build_main_notebook.py   (chỉ ghép file, không chạy thí nghiệm)
"""
import hashlib
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "scripts"))
import nb_cells_task1 as t1  # noqa: E402
import nb_cells_task2 as t2  # noqa: E402

path = root / "notebooks/FCVG101_main.ipynb"


def cell(kind, text):
    text = text.strip() + "\n"
    result = {"cell_type": kind, "id": hashlib.sha256(text.encode()).hexdigest()[:12],
              "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


head = [
    cell("markdown", """
# FCVG101 — Tìm sản phẩm thời trang bằng ảnh (DeepFashion In-shop)

**Một notebook cho cả Task 1 (xử lý dữ liệu, biểu diễn ảnh) và Task 2 (mô hình, kết quả, demo, phân tích lỗi).**
Task 3 (hiệu chỉnh camera, lọc/FFT/gradient/điểm đặc trưng, optical flow) là thí nghiệm độc lập, nằm ở notebook riêng.

```text
 Dữ liệu ─► Tiền xử lý ─► CLIP ViT-B/32 (đóng băng) ─► Adapter 512→128→512 ─► vector 512 ─► cosine với gallery ─► Top-K
 (Phần A)   (Phần A)        (Phần B)                      (học bằng ArcFace)                   (Phần B: bảng, demo, lỗi)
```

| Phần | Nội dung | Ô |
|---|---|---|
| 0 | Cấu hình môi trường: Colab hoặc máy cá nhân, đường dẫn dữ liệu, checkpoint | 0 |
| A | Task 1: dataset, chia train/val/test, pipeline, augmentation, biểu diễn ảnh, các cấu hình | `# 2` … `# 10` |
| B | Task 2: bảng kết quả test, demo trên ảnh chưa thấy, phân tích lỗi | `# T2-1` … `# T2-6` |
| Phụ lục | Chạy lại toàn bộ từ đầu (mã hóa, train, đánh giá) — mặc định tắt | cuối |

**Hai chế độ:** mặc định chạy nhanh — dùng checkpoint có sẵn, chỉ mã hóa 12.612 ảnh gallery nếu chưa có.
Muốn tái tạo mọi thứ (vài giờ) thì đặt `RUN_FULL = True` ở ô 0.

**Dữ liệu:** ảnh DeepFashion không được phân phối lại (chỉ dùng nghiên cứu/giáo dục phi thương mại). Cần thư mục có
`img/`, `Eval/list_eval_partition.txt`, `list_bbox_inshop.txt` — tải từ
[trang CUHK](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html).
Trên Colab: đưa thư mục (hoặc `dataset.zip` chứa ba mục trên) vào Google Drive rồi sửa `DRIVE_DATA_DIR`.
"""),
    cell("code", """
# 0. Cấu hình môi trường — Colab hoặc máy cá nhân
import os, subprocess, sys, zipfile
from pathlib import Path

# ---- chỉnh ở đây -------------------------------------------------------------------------------
RUN_FULL = False                                        # True: chạy lại mã hóa + train + test ở phụ lục (vài giờ)
DATA_DIR = None                                         # máy cá nhân: thư mục chứa img/, Eval/, list_bbox_inshop.txt (None = thư mục cha của repo)
DRIVE_DATA_DIR = "/content/drive/MyDrive/FCVG101_data"  # Colab: thư mục trên Drive (img/ + Eval/ + bbox, hoặc dataset.zip)
REPO_URL = "https://github.com/trungledang/fcvg101-visual-product-search"
CKPT_DRIVE_ID = "1XWptbtHuBulXwOthm_7qVEHB8puaAw7c"     # 9 adapter đã train (4,4 MB), xem MODEL_LINKS.md
# -------------------------------------------------------------------------------------------------

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    ROOT = Path("/content/repo")
    if not (ROOT / "vps").is_dir():
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(ROOT)], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ftfy", "regex", "ImageHash", "gdown",
                        "git+https://github.com/openai/CLIP.git"], check=True)
    DATA = Path(DRIVE_DATA_DIR)
    if (DATA / "dataset.zip").exists() and not Path("/content/data/img").is_dir():   # giải nén ra đĩa cục bộ cho nhanh
        zipfile.ZipFile(DATA / "dataset.zip").extractall("/content/data")
    if Path("/content/data/img").is_dir():
        DATA = Path("/content/data")
else:
    ROOT = Path.cwd().resolve()
    if not (ROOT / "vps").is_dir():
        ROOT = ROOT.parent if ROOT.name == "notebooks" else ROOT / "visual-product-search-pilot"
    DATA = Path(DATA_DIR) if DATA_DIR else ROOT.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

need = [DATA / "Eval/list_eval_partition.txt", DATA / "list_bbox_inshop.txt", DATA / "img"]
missing = [str(p) for p in need if not p.exists()]
assert not missing, f"Thiếu dữ liệu DeepFashion: {missing}. Xem hướng dẫn ở ô markdown đầu notebook."

# Checkpoint adapter: dùng bản có sẵn, nếu không có thì tải từ Drive
if not (ROOT / "cache/adapter/ep40/e3_ratio0.2_seed0.pt").exists():
    try:
        import gdown
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
        import gdown
    (ROOT / "cache/adapter").mkdir(parents=True, exist_ok=True)
    gdown.download(id=CKPT_DRIVE_ID, output=str(ROOT / "cache/adapter/ckpt.zip"), quiet=True)
    zipfile.ZipFile(ROOT / "cache/adapter/ckpt.zip").extractall(ROOT / "cache/adapter")

import torch
from vps.embed import load_model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_pre, clip_meta = load_model(ROOT / ".cache/clip", device)   # tự tải CLIP chính thức, kiểm tra SHA-256
print("môi trường:", "Colab" if IN_COLAB else "máy cá nhân", "| thiết bị:", device, "| dữ liệu:", DATA)
"""),
    cell("markdown", """
---
# Phần A — Task 1: xử lý dữ liệu và biểu diễn ảnh
"""),
]

tail = [
    cell("markdown", """
---
# Phụ lục — chạy lại toàn bộ từ đầu (mặc định tắt, `RUN_FULL = False`)

Thứ tự đúng với lúc làm thí nghiệm. Mỗi bước ghi vào `cache/` hoặc `reports/`. Mã hóa ảnh lần đầu có thể mất hàng giờ
nếu đọc ảnh từ ổ chậm (6.305 s trên ổ E: của máy gốc, 96 s khi ảnh đã nằm trong bộ nhớ đệm). Train 40 epoch: ~45 s mỗi lần.
**[Các lệnh dưới đây là các lệnh đã dùng thật; riêng việc chạy chúng từ trong notebook này chưa được chạy trọn vẹn.]**
"""),
    cell("code", """
# Phụ lục. Tái tạo: mã hóa -> train E1/E2/E3 x 3 seed -> đánh giá test một lần
d = ["--data-root", str(DATA)]
steps = [
    ["scripts/pilot.py", *d, "embed", "--batch-size", "64"],                               # query/gallery, ảnh đầy đủ (E0)
    *[["scripts/encode_variant.py", "--variant", "A1_white_pad", "--split", sp, *d] for sp in ("train", "query", "gallery")],
    ["scripts/encode_augmented.py", *d],                                                    # 3 bản augmented / ảnh train (E2)
]
for seed in (0, 1, 2):
    base = ["scripts/train_adapter.py", *d, "--epochs", "40", "--seed", str(seed), "--ckpt-dir", "cache/adapter/ep40"]
    steps += [base,                                                                         # E1
              base + ["--aug-cache", "train_aug_C0_full_aug_v3_s0"],                        # E2
              base + ["--base-cache", "train_A1_white_pad"]]                                # E3
steps.append(["scripts/eval_test.py", *d])                                                   # chạy test MỘT lần
if RUN_FULL:
    for st in steps:
        print(">", " ".join(st))
        subprocess.run([sys.executable, *st], cwd=ROOT, check=True)
else:
    print(f"RUN_FULL = False: bỏ qua {len(steps)} lệnh. Danh sách:")
    for st in steps:
        print("  python", " ".join(st))
"""),
]

cells = head + t1.cells + [cell("markdown", "---")] + t2.cells + tail
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "colab": {"name": "FCVG101_main.ipynb"}},
      "nbformat": 4, "nbformat_minor": 5}
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("wrote", path, len(cells), "cells")
