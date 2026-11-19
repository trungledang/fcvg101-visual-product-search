# Đọc dự án qua các đường ống (pipeline) — từng bước, từng file

Viết 2026-11-19 cho Trung và Phong. Mỗi sơ đồ trả lời **một câu hỏi**, đọc theo thứ tự từ 1 đến 8.
Số liệu lấy từ `reports/` (đã chạy thật). Chi tiết lý do: [HIEU_VA_PHAN_BIEN.md](HIEU_VA_PHAN_BIEN.md); chi tiết kết quả: [T1_ADAPTER_GIAI_THICH.md](T1_ADAPTER_GIAI_THICH.md).

| # | Câu hỏi | Sơ đồ |
|---|---|---|
| 1 | Bài toán là gì, đầu vào và đầu ra? | Bài toán trong một hình |
| 2 | Dữ liệu được chia thế nào để không gian lận? | Chia dữ liệu |
| 3 | Một ảnh đi qua những bước nào trước khi vào CLIP? | Tiền xử lý (Task 1) |
| 4 | Adapter nằm ở đâu và học gì? | Mô hình và lúc train |
| 5 | Lúc tìm kiếm thật, chuyện gì xảy ra? | Suy luận (demo) |
| 6 | Các cấu hình E0–E3 khác nhau ở đâu? | Bản đồ thí nghiệm |
| 7 | Chấm điểm thế nào, vì sao tin được? | Đánh giá + các chốt chống gian lận |
| 8 | File nào làm bước nào, ai làm gì? | Bản đồ file và phân công |

---

## 1. Bài toán trong một hình

```text
   ảnh query (khách chụp / ảnh cận cảnh)                kho ảnh gallery (12.612 ảnh, 3.985 món)
              │                                                       │
              ▼                                                       ▼
      ┌───────────────┐        "vector đặc trưng"          ┌───────────────┐
      │ biến ảnh thành │  ── 512 con số mô tả ảnh ──►      │ cùng một cách  │
      │ vector         │                                    │ biến thành vector│
      └───────┬───────┘                                    └───────┬───────┘
              └──────────────► so độ giống (cosine) ◄──────────────┘
                                        │
                                        ▼
                  xếp gallery từ giống nhất → ít giống nhất  ──►  Top-K
                                        │
              Recall@1 = "ảnh xếp đầu có ĐÚNG món của query không?"  (đúng = cùng item_id)
```

- Đây là **tìm kiếm**, không phải phân loại: món ở test chưa từng có khi train, nên không thể "đoán tên món".
- Một món có nhiều ảnh (nhiều góc). Chỉ cần xếp **một** ảnh của đúng món lên đầu là tính đúng.

## 2. Chia dữ liệu — vì sao con số test đáng tin

```text
 DeepFashion In-shop: 52.712 ảnh, 7.982 món
 │
 ├── official TRAIN  25.882 ảnh / 3.997 món ───────────────┐
 │        │ chia theo MÓN (không theo ảnh), seed cố định      │
 │        ├── adapter-train  23.266 ảnh / 3.597 món  ──► DẠY adapter
 │        └── validation     1.232 query + 1.384 gallery / 400 món ──► CHỌN cấu hình, epoch, ratio
 │
 └── official TEST   (3.985 món KHÁC hẳn train) ──────── KHÓA, chỉ mở 1 lần ở cuối
          query 14.218 ảnh ─┐
          gallery 12.612 ảnh ┴─► đủ cả gallery làm nhiễu, không lấy mẫu con
```

Các chốt tự động kiểm tra (nếu vi phạm là script dừng): món train ∩ món val = ∅ · món train ∩ món test = ∅ · không ảnh nào vừa là query vừa là gallery.
**Vì sao chia val theo món:** nếu chia theo ảnh, adapter đã thấy món đó khi train, val sẽ dễ ảo. Chia theo món thì val cũng là "món lạ" như test.

## 3. Một ảnh đi qua tiền xử lý (Task 1)

```text
 file JPEG ──► đọc + ép RGB ──┬────────────────────────────────────────────┐
                              │  ĐƯỜNG A — ảnh đầy đủ (C0, baseline)       │
                              │      giữ nguyên ảnh                        │
                              ├────────────────────────────────────────────┤
                              │  ĐƯỜNG B — crop + pad (C1 / A1)            │
                              │      cắt theo bbox, nới lề 10%, cạnh ≥ 64px│
                              │      pad cho vuông (A1 = nền TRẮNG,         │
                              │      C1 = màu trung bình)                  │
                              └──────────────┬─────────────────────────────┘
                                             │   [chỉ khi TRAIN, chỉ cấu hình có augmentation]
                                             │   crop ngẫu nhiên 75–100% · lật ngang
                                             │   độ sáng/tương phản ±20% · xóa một mảng nhỏ
                                             ▼
                          resize 224×224 ► chuẩn hóa (mean/std của CLIP) ► tensor ► batch 64
```

Ba điều cần nhớ: (1) **không đổi màu sắc** khi augment vì màu là đặc điểm nhận dạng món; (2) crop **làm CLIP gốc kém đi** (val: 71,1 → 65,8 với pad trắng, → 53,6 với pad màu trung bình); (3) biểu diễn tay "HSV histogram" chỉ đạt 33,4% — kém xa CLIP.

## 4. Mô hình và lúc huấn luyện

```text
                       ┌───────────── ĐÓNG BĂNG (không học) ─────────────┐
 ảnh đã tiền xử lý ──► │ CLIP ViT-B/32 (88 triệu tham số)                │──► f: 512 số, độ dài = 1
                       └─────────────────────────────────────────────────┘
                                   │
          ┌────────────────────────┴─ chỉ khối này được HỌC: 131.072 tham số ─┐
          │  f ─► Linear 512→128 ─► ReLU ─► Linear 128→512 ─► ReLU ─► A(f)    │
          │  z = 0,2 × A(f) + 0,8 × f          ← residual: luôn giữ 80% CLIP gốc│
          │  z ← z / |z|                       ← lại chuẩn hóa độ dài = 1      │
          └────────────────────────┬──────────────────────────────────────────┘
                                   │ z (vector cuối cùng dùng để tìm kiếm)
                                   ▼
                  CHỈ LÚC TRAIN: ArcFace (s=30, m=0,3) trên 3.597 món train
                  "kéo z về tâm đúng của món, đẩy ra khỏi tâm món khác cộng thêm lề góc"
                                   │ loss ► AdamW (lr 1e-3 adapter, 1e-2 ArcFace), batch 256, 40 epoch
                                   ▼
                  hết train: VỨT lớp ArcFace, giữ adapter ► mỗi epoch chấm R@1 trên val ► giữ epoch tốt nhất
```

**Tại sao nhanh:** CLIP đóng băng nên vector f của mỗi ảnh chỉ tính **một lần** rồi lưu (`cache/embeddings/*.npy`); train adapter chỉ làm việc trên các vector 512 số (40 epoch ≈ 45 giây). Hệ quả: augmentation chỉ có tác dụng nếu lưu thêm vector của các bản augmented (đó là E2).
**Kiểm tra an toàn:** `ratio = 0` cho đúng CLIP gốc (unit test) — nên adapter hỏng thì kết quả cũng chỉ về mức CLIP gốc.

## 5. Suy luận — chuyện gì xảy ra khi tìm một ảnh (demo)

```text
 MỘT ảnh mới ──► (E3: crop + pad trắng nếu biết bbox, ngược lại chỉ pad trắng) ──► CLIP ──► f
                                                                                       │
                                                                                       ▼ adapter
 gallery 12.612 vector (ĐÃ tính sẵn, cũng qua adapter) ─────────────►  12.612 điểm cosine = G · z
                                                                                       │
                                                                          sắp giảm dần ──► lấy Top-K
                                                                                       │
                                            hiển thị: ảnh, item_id, điểm; xanh = đúng món, đỏ = sai
```

Chi phí mỗi lần hỏi: 1 lần chạy CLIP + 1 phép nhân ma trận 12.612×512. Code: `vps/search.py`; notebook: `notebooks/FCVG101_main.ipynb` (Phần B).
Kiểm tra đã làm: mã hóa lại từ file ra vector trùng cache (cosine 1,0000) và cùng hạng như lúc chấm test.

## 6. Bản đồ thí nghiệm E0–E3 (mỗi bước đổi **một** yếu tố)

```text
 E0  CLIP gốc, ảnh đầy đủ ───────────────────────────────  baseline (đối chứng)
  │   + adapter
  ▼
 E1  adapter, ảnh đầy đủ ─────────────────────────────────  thêm MỘT thứ: adapter
  ├── + 3 bản ảnh augmented / ảnh ──►  E2  (đổi: augmentation)        kết quả: HÒA với E1
  └── thay ảnh đầy đủ bằng crop+pad trắng ► E3 (đổi: tiền xử lý)      kết quả: nhỉnh nhất
 (đối chứng thêm: E0 trên ảnh crop+pad trắng — để biết crop có hại CLIP gốc không)
```

| | Test R@1 | Val R@1 | Ý nghĩa |
|---|---|---|---|
| E0 ảnh đầy đủ | 45,5 | 71,0 | baseline |
| E0 crop+pad trắng | 39,6 | 65,8 | crop **hại** CLIP gốc |
| E1 | 71,4 | 90,9 | adapter **+26 điểm** |
| E2 | 71,1 | 91,0 | augmentation **không giúp** (encoder đóng băng ít đổi vector: cosine 0,93) |
| E3 | **72,2** | 91,7 | crop **đảo chiều**: có adapter thì lại nhỉnh hơn |

Adapter: trung bình 3 seed; chênh giữa seed ≈ 1 điểm trên val và ≈ 0,3–1,5 trên test, nhỏ hơn nhiều so với mức tăng 26 điểm.

## 7. Đánh giá và các chốt chống gian lận

```text
 mọi query test ──► vector z ─┐
                             ├─► cosine với toàn bộ 12.612 ảnh gallery ─► sắp xếp
 mọi ảnh gallery ──► vector ─┘                                            │
        hạng của ảnh ĐÚNG đầu tiên = r                                    ▼
        Recall@K = tỉ lệ query có r ≤ K       MRR = trung bình của 1/r
```

```text
 THỨ TỰ BẮT BUỘC:   chọn cấu hình trên VAL ──► ghi lại (commit) ──► chạy TEST đúng 1 lần ──► báo cáo
                                                         │
 Kiểm tra pipeline chấm điểm: E0 tính lại trên test ra 45,51% = đúng con số baseline cũ ✔
 Kiểm tra nghi ngờ khi kết quả "tốt bất ngờ": không trùng món giữa các tập ✔ · hạng tính lại từ file ảnh khớp cache ✔
```

Khi bị hỏi "có gian lận không?": test chỉ mở một lần; không tham số nào chọn trên test; val chia theo món; gallery đủ 12.612 ảnh.
**Hạn chế cần thú nhận:** 400 món val dễ hơn 3.985 món test (90% so với 71%), nên chỉ dùng val để *so sánh* cấu hình, không để báo "độ chính xác".

## 8. Bản đồ file và phân công

```text
 DỮ LIỆU          vps/annotations.py   đọc partition, kiểm tra rò rỉ
                  vps/preprocess.py    bbox, crop, pad, augmentation, chia val theo món
 BIỂU DIỄN        vps/embed.py         nạp CLIP, mã hóa, lưu cache      scripts/encode_variant.py, encode_augmented.py
 MÔ HÌNH          vps/adapter.py       Adapter, ArcFace, vòng train     scripts/train_adapter.py   (E1/E2/E3)
 ĐÁNH GIÁ         vps/metrics.py       cosine, Recall@K                 scripts/eval_val.py, eval_test.py
 SUY LUẬN / DEMO  vps/search.py        1 ảnh → Top-K                    notebooks/FCVG101_main.ipynb (NOTEBOOK CHÍNH: Task 1 + 2)
 TÀI LIỆU         docs/BAO_CAO_NHAP.md  báo cáo nháp       MODEL_LINKS.md  link checkpoint + lệnh chạy lại
 NỘP BÀI          scripts/build_submission.py  → artifacts/submission/FCVG101_code.zip (≤ 5 MB)
```

| Người | Phụ trách | Trạng thái |
|---|---|---|
| Trung | Task 1 (pipeline, cấu hình), Task 2 (mô hình, E0–E3, test, demo), báo cáo mục 1–7 và 9–11 | xong phần chạy; còn xuất PDF + tập trình bày |
| Phong | Task 3A (hiệu chỉnh camera, lọc, FFT, gradient, điểm đặc trưng), 3B (optical flow), bảng nhóm lỗi, đối chiếu Task 1 với đề | chưa bắt đầu |

**Cách Phong đọc nhanh:** sơ đồ 1 → 2 → 4 → 6 là đủ để trả lời "dự án làm gì và vì sao"; sơ đồ 7 để trả lời "có gian lận không". Phần của Phong (Task 3) độc lập, không cần hiểu adapter.
