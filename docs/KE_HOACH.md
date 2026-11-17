# Kế hoạch nộp bài FCVG101

Lập 2026-11-14 (thứ Bảy). Deadline ước tính 2–3 tuần (khoảng 28/11 – 05/12, **chưa chính thức**). Kế hoạch tính theo 2 tuần làm + 1 tuần dự phòng.
Lý do và phản biện: [HIEU_VA_PHAN_BIEN.md](HIEU_VA_PHAN_BIEN.md). Đánh dấu `[x]` khi xong và ghi đường dẫn đầu ra.

## Mốc

| Mốc | Ngày | Phải có |
|---|---|---|
| M1 | CN 16/11 | Cả hai đã đọc `HIEU_VA_PHAN_BIEN.md`; Phong nhận Task 3 |
| M2 | CN 23/11 | Có số E1 trên val; Task 3A xong → **họp 30 phút quyết định** (xem quy tắc dưới) |
| M3 | CN 30/11 | Test chạy 1 lần cho các cấu hình đã chốt; Task 3 xong; bản nháp báo cáo đủ mục |
| M4 | trước deadline 2–3 ngày | PDF + ZIP ≤ 5 MB nộp thử; tập trình bày |

**Quy tắc tại M2:** E1 tăng R@1 val rõ (≥ ~5 điểm) → làm tiếp E2 (và E3 nếu kịp). Tăng ít hoặc giảm → kiểm tra lỗi code trong 1 buổi; vẫn vậy thì giữ kết quả "adapter không giúp nhiều" làm phát hiện, làm E2 rồi dừng. Không mở thêm hướng mới sau M2.

## Tuần 1 (15–23/11)

### Trung
- [x] **T1 — Adapter E1** — xong 2026-11-14: val R@1 71,0% → 89,9% (1 seed), xem `docs/T1_ADAPTER_GIAI_THICH.md` mục 5. 3 seed (17/11): trung bình 90,0%, dao động ~1 điểm.
- [x] **T2 — Cache ảnh augmented** — xong 17/11, 3 bản/ảnh, 256 s (`scripts/encode_augmented.py`).
  - Xong khi: có file embedding augmented + lệnh tạo lại.
- [x] **T3 — Thử residual ratio** — xong 17/11: 0,5 ≈ 0,2 (91,3 vs 90,9, gần hòa). Mục 5c.

### Phong
- [ ] **P1 — Hiểu đề tài** (~1 buổi). Đọc `HIEU_VA_PHAN_BIEN.md`, chạy `notebooks/learn_clip_retrieval.ipynb`; tự giải thích được pipeline và Recall@K cho Trung.
- [ ] **P2 — Task 3A** (~2 buổi). Hiệu chỉnh camera bằng ảnh bàn cờ (tự chụp 10–15 ảnh hoặc ảnh mẫu OpenCV); lọc tuyến tính, FFT, gradient (Sobel), điểm đặc trưng (Harris/ORB/SIFT).
  - Xong khi: notebook chạy từ đầu tới cuối, mỗi kỹ thuật có hình + 2–3 câu giải thích.
- [ ] **P3 — Phân tích lỗi baseline** (~1 buổi). Lấy ~20 query val sai của E0, xếp nhóm (cùng màu khác món, góc nghiêng, họa tiết nhỏ, …).
  - Xong khi: bảng nhóm lỗi + ảnh ví dụ, cố định danh sách 20 query để tuần sau so với adapter.

## Tuần 2 (24–30/11)

### Trung
- [x] **T4a — E2** — xong 17/11: hòa với E1 (val R@1 ~91,0% ở 40 epoch, 3 seed); xem `T1_ADAPTER_GIAI_THICH.md` mục 5b.
- [x] **T4b — E3** — xong 17/11: crop + pad trắng + adapter 91,7% val, nhỉnh nhất (đảo chiều so với CLIP đóng băng). Mục 5c.
- [ ] **T5 — Chốt cấu hình trên val, chạy test 1 lần** cho E0–E2(E3). Bảng kết quả + R@1 theo category.
- [ ] **T6 — Notebook demo**: 1 ảnh chưa thấy → Top-5, so E0 và adapter cạnh nhau.
- [ ] **T7 — Lưu checkpoint adapter** lên Drive/Kaggle, lấy link (ZIP nộp giới hạn 5 MB).

### Phong
- [ ] **P4 — Task 3B** (~1 buổi). Optical flow (Farneback) giữa 2 khung hình video tự quay; hình mũi tên/màu + giải thích.
- [ ] **P5 — Phân tích lỗi so sánh**: 20 query của P3 với adapter tốt nhất — sửa được nhóm nào, hỏng thêm nhóm nào.
- [ ] **P6 — Đối chiếu Task 1 với đề**: tick từng ý của đề (mô tả dataset, pipeline, augmentation có hình minh họa, ≥ 2 cấu hình) — thiếu gì báo Trung.

## Tuần 3 (dự phòng + hoàn thiện)

- [ ] **Báo cáo PDF** theo đúng thứ tự đề: dataset → pipeline → Task 1 → mô hình + siêu tham số → kết quả + bảng + lỗi → Task 3 → thảo luận/giới hạn → hướng dẫn chạy → tài liệu tham khảo.
  - Trung: mô hình, kết quả, thảo luận. Phong: Task 3, phân tích lỗi, kiểm tra Task 1. Đọc chéo phần của nhau.
- [ ] **ZIP code** ≤ 5 MB: `.py`/`.ipynb`, config, `requirements.txt`, link checkpoint; không kèm ảnh dataset hay `.venv`.
- [ ] **Tập trình bày**: mỗi người giải thích được phần của người kia; luyện 8 câu hỏi ở mục 6 của `HIEU_VA_PHAN_BIEN.md`.

## Ngoài kế hoạch (chỉ khi xong hết mà còn thời gian)

Fine-tune vài block cuối của CLIP; reranking. Không bắt đầu trước M3.

## Rủi ro

| Rủi ro | Dấu hiệu | Ứng phó |
|---|---|---|
| ~~Encode chậm~~ — đã hết: lần 2 đạt ~95 ảnh/s (ảnh đã trong bộ nhớ đệm của Windows), T2 chỉ mất 256 s | — | — |
| GPU/Kaggle không chạy được cho T2 | Lỗi môi trường quá 1 buổi | Bỏ E2 augmented, dùng E3 (crop chỉ cần encode lại 1 lần) hoặc giảm số bản augmented |
| Deadline sớm hơn dự kiến | Thầy công bố < 2 tuần | Bỏ E3 và P5, giữ E0–E2 + Task 3 |
| Phong kẹt Task 3 | Chưa có notebook ở M2 | Trung hỗ trợ 1 buổi; dùng ảnh mẫu có sẵn của OpenCV |
