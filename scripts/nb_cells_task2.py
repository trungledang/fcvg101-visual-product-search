"""Các ô của phần Task 2 trong notebooks/FCVG101_main.ipynb: kết quả test, demo suy luận, phân tích lỗi.

Chỉ ĐỌC số đã chốt (reports/test_results.json, cache/test_first_pos_ranks.npz);
không chọn cấu hình nào dựa trên test. Ghép bởi scripts/build_main_notebook.py.
"""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]


def cell(kind, text):
    text = text.strip() + "\n"
    result = {"cell_type": kind, "id": hashlib.sha256(text.encode()).hexdigest()[:12],
              "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


cells = [
    cell("markdown", """
# Phần B — Task 2: kết quả, demo suy luận và phân tích lỗi

**Hệ thống:** ảnh → CLIP ViT-B/32 (đóng băng) → adapter 512→128→512 (residual, ratio 0.2, học bằng ArcFace) → so cosine với 12.612 ảnh gallery → Top-K.

| Yêu cầu Task 2 | Ô |
|---|---|
| Bảng so sánh trên test + hình | T2-2 |
| Demo suy luận trên mẫu chưa thấy | T2-3, T2-4 |
| Phân tích lỗi ngắn | T2-5, T2-6 |

**Nguyên tắc:** cấu hình đã chốt trên validation trước khi chạy test (`docs/T1_ADAPTER_GIAI_THICH.md`, mục 5c).
Notebook này chỉ đọc lại số test (`reports/test_results.json`), không chọn gì dựa trên test.
Mọi query test thuộc 3.985 sản phẩm **không có trong tập train** → mỗi query là một mẫu chưa thấy.

**Dữ liệu cần có:** cấu hình ở ô 0 (đầu notebook). Chạy được trên CPU, nhanh hơn trên GPU.
"""),
    cell("code", """
# T2-1. Dựng 2 bộ tìm kiếm: E0 (CLIP gốc) và E3 (adapter, crop + pad trắng)
# Lần đầu nếu chưa có vector gallery thì tự mã hóa 12.612 ảnh (vài phút trên GPU) và lưu vào cache/embeddings/.
%matplotlib inline
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from vps.search import ProductSearch
from vps.annotations import parse_partition

search = {name: ProductSearch(ROOT, name, clip_model, clip_pre, seed=0, data_root=DATA, device=device)
          for name in ("E0_full", "E3")}
records = parse_partition((DATA / "Eval/list_eval_partition.txt").read_text())
queries = [r for r in records if r.split == "query"]
train_ids = {r.item_id for r in records if r.split == "train"}
print(device, "| gallery:", len(search["E3"].gallery), "ảnh | query test:", len(queries))
"""),
    cell("code", """
# T2-2. Kết quả test (một lần duy nhất) + R@1 theo nhóm sản phẩm
res = json.loads((ROOT / "reports/test_results.json").read_text(encoding="utf-8"))
names = {"E0_full": "E0 CLIP gốc, toàn ảnh", "E0_crop": "E0 CLIP gốc, crop+pad trắng",
         "E1": "E1 + adapter", "E2": "E2 + adapter + augmentation", "E3": "E3 + adapter, crop+pad trắng"}
print(f"{'Cấu hình':32s} {'R@1':>7s} {'R@5':>7s} {'R@10':>7s} {'MRR':>7s}   (adapter: trung bình 3 seed)")
for k, v in res["results"].items():
    a = v["aggregate"]
    print(f"{names[k]:32s} " + " ".join(f"{a[m]['mean']*100:7.2f}" for m in ("R@1", "R@5", "R@10", "MRR")))

cat = res["R@1_by_category_seed0"]
order = sorted(cat, key=lambda c: cat[c]["E0_full"])
y = np.arange(len(order))
fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(y - 0.2, [cat[c]["E0_full"] * 100 for c in order], 0.4, label="E0 CLIP gốc", color="#9aa5b1")
ax.barh(y + 0.2, [cat[c]["E3"] * 100 for c in order], 0.4, label="E3 adapter", color="#2f6fb0")
ax.set_yticks(y, [f"{c} ({cat[c]['queries']})" for c in order], fontsize=8)
ax.set_xlabel("R@1 trên test (%)"); ax.legend(loc="lower right"); ax.grid(axis="x", alpha=0.3)
ax.set_title("R@1 theo nhóm sản phẩm (số query trong ngoặc), seed 0")
plt.tight_layout(); plt.show()
"""),
    cell("markdown", """
**Đọc bảng:** adapter tăng R@1 từ 45,5% lên 71–72% trên cả 23 nhóm. Crop làm CLIP gốc *kém đi* (39,6%)
nhưng lại nhỉnh nhất khi có adapter (E3). E2 (thêm augmentation) không hơn E1 — augmentation nhẹ gần như không đổi
embedding của CLIP đóng băng (cosine giữa các bản ≈ 0,93), nên adapter không học thêm được gì.
"""),
    cell("code", """
# T2-3. Demo suy luận: query test (sản phẩm chưa từng thấy khi train) → Top-5, E0 và E3 cạnh nhau
def show_row(axes, img, title, hits, true_id):
    axes[0].imshow(img); axes[0].set_title(title, fontsize=8)
    for ax, h in zip(axes[1:], hits):
        ok = h.item_id == true_id
        with Image.open(DATA / h.path) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(f"#{h.rank} {'ĐÚNG' if ok else 'sai'}\\ncos {h.score:.3f}", fontsize=8,
                     color="green" if ok else "red")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

def compare(q, k=5):
    assert q.item_id not in train_ids  # đúng là sản phẩm chưa thấy
    fig, axes = plt.subplots(2, k + 1, figsize=(2.1 * (k + 1), 5.2))
    with Image.open(DATA / q.path) as im:
        for row, name in enumerate(("E0_full", "E3")):
            s = search[name]
            shown, how = s.prepare(im, q.path)
            show_row(axes[row], shown, f"{name}\\n{how}", s.search(im, q.path, k=k), q.item_id)
    fig.suptitle(f"Query {q.path}  (item {q.item_id}, {q.category})", fontsize=9)
    plt.tight_layout(); plt.show()

rng = np.random.default_rng(2026)          # chọn ngẫu nhiên có seed, không chọn tay
for i in rng.choice(len(queries), 3, replace=False):
    compare(queries[int(i)])
"""),
    cell("markdown", """
**Cách đọc:** hàng trên là CLIP gốc, hàng dưới là adapter. Viền chữ xanh = cùng `item_id` với query.
Các mẫu ở trên được chọn ngẫu nhiên có seed — có thể cả hai cùng đúng hoặc cùng sai; đó là hình ảnh thật của hệ thống,
không phải ví dụ được chọn cho đẹp. Muốn xem ví dụ khác: đổi seed `2026`.
"""),
    cell("code", """
# T2-4. Ảnh bất kỳ (ví dụ ảnh tự chụp): đặt đường dẫn vào MY_IMAGE.
# Ảnh ngoài dataset không có bbox → E3 chỉ pad trắng, không crop (đầu ra ghi rõ điều này).
MY_IMAGE = None   # ví dụ: r"E:\\Downloads\\ao_cua_toi.jpg"
path = Path(MY_IMAGE) if MY_IMAGE else DATA / queries[0].path   # mặc định: 1 ảnh query, coi như ảnh lạ
fig, axes = plt.subplots(2, 6, figsize=(12.6, 5.2))
with Image.open(path) as im:
    for row, name in enumerate(("E0_full", "E3")):
        s = search[name]
        shown, how = s.prepare(im, None)     # path=None: hệ thống không biết bbox hay item_id
        hits = s.search(im, None, k=5)
        show_row(axes[row], shown, f"{name}\\n{how}", hits, true_id=None)
        print(name, [(h.item_id, round(h.score, 3)) for h in hits])
plt.tight_layout(); plt.show()
"""),
    cell("code", """
# T2-5. Phân tích lỗi: query nào adapter sửa được, query nào làm hỏng (seed 0, toàn bộ 14.218 query)
ranks = np.load(ROOT / "reports/test_first_pos_ranks_E0_E3.npz")  # hạng đã lưu từ scripts/eval_test.py
r0, r3 = ranks["E0_full_seedNone"], ranks["E3_seed0"]
ok0, ok3 = r0 == 1, r3 == 1
print(f"Cả hai đúng     : {np.sum(ok0 & ok3):6d}")
print(f"E3 sửa được     : {np.sum(~ok0 & ok3):6d}   (E0 sai, E3 đúng)")
print(f"E3 làm hỏng     : {np.sum(ok0 & ~ok3):6d}   (E0 đúng, E3 sai)")
print(f"Cả hai sai      : {np.sum(~ok0 & ~ok3):6d}")
fail = r3[~ok3]
print(f"Khi E3 sai ở #1: món đúng nằm ở #2–5 trong {np.mean(fail <= 5):.0%} trường hợp, "
      f"#6–10 {np.mean((fail > 5) & (fail <= 10)):.0%}, ngoài top-10 {np.mean(fail > 10):.0%}")
fig, ax = plt.subplots(figsize=(7, 3))
bins = [1, 2, 3, 4, 5, 6, 11, 21, 51, 101, 13000]
lab = ["1", "2", "3", "4", "5", "6–10", "11–20", "21–50", "51–100", ">100"]
for r, n, c in ((r0, "E0", "#9aa5b1"), (r3, "E3", "#2f6fb0")):
    h, _ = np.histogram(r, bins=bins)
    ax.plot(lab, h / len(r) * 100, "o-", label=n, color=c)
ax.set_ylabel("% query"); ax.set_xlabel("hạng của ảnh đúng đầu tiên"); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
"""),
    cell("code", """
# T2-6. Sáu lỗi của E3 (chọn ngẫu nhiên có seed) để xếp nhóm lỗi — Phong điền bảng ở ô dưới
fail_idx = np.flatnonzero(~ok3)
for i in np.random.default_rng(7).choice(fail_idx, 6, replace=False):
    print(f"query #{i}: E0 hạng {r0[i]}, E3 hạng {r3[i]}")
    compare(queries[int(i)])
"""),
    cell("markdown", """
**Nhóm lỗi (Phong điền sau khi xem ô 6 và thêm ví dụ nếu cần):**

| Nhóm lỗi | Ví dụ (query #) | Giải thích ngắn | Adapter có giúp không? |
|---|---|---|---|
| Cùng kiểu dáng/màu, khác món (gần như trùng) | | | |
| Query cận cảnh chi tiết, gallery toàn thân | | | |
| Họa tiết/in hình nhỏ (Graphic_Tees) | | | |
| Góc chụp nghiêng / bbox hẹp | | | |
| Khác (ghi rõ) | | | |

Gợi ý: nhóm khó nhất trên test là WOMEN/Graphic_Tees (E0 18,9% → E3 46,0%) — CLIP ở 224×224 khó đọc hình in nhỏ.
"""),
]
