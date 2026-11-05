# Hiện trạng dự án FCVG101 Visual Product Retrieval — báo cáo cho reviewer độc lập

**Ngày 2026-10-30.** Tổng hợp từ source code, file kết quả JSON, notebook đã chạy và các báo cáo trong repo. Trong nhiệm vụ này không sửa code và không chạy thí nghiệm mới.

Nhãn bằng chứng dùng trong báo cáo:
- **[Đã chạy]**: có file kết quả trong repo.
- **[Nguồn nói]**: lấy từ paper hoặc bài viết, có link.
- **[Suy luận]**: lập luận dựa trên bằng chứng.
- **[Chưa xác minh]**: chưa đủ bằng chứng.

Đường dẫn file tính từ gốc repo `visual-product-search-pilot/`. Branch `claude/zealous-euler-g8hncc`, commit cuối `468e028`.

## 0. Tóm tắt cho reviewer

| # | Kết luận | Mức bằng chứng |
|---|---|---|
| 1 | Baseline frozen CLIP ViT-B/32 trên **official test** (14.218 query / 12.612 gallery) đạt **R@1 45,51%, R@5 67,25%**. | [Đã chạy] `reports/metrics.json` |
| 2 | Paper Unicom báo 45,4 R@1 cho CLIP B/32 frozen trên In-shop. Cùng model, cùng dataset, cùng metric, cùng kiểu feature frozen. | [Nguồn nói] Unicom Bảng 2 |
| 3 | Con số **71,1% là R@1 của C0 (ảnh đầy đủ, KHÔNG crop) trên tập validation**. Không phải số của crop, cũng không phải số test. Crop C1 trên cùng val đạt 53,6%. | [Đã chạy] `reports/val_results.json` |
| 4 | Crop bbox + pad màu mean làm giảm R@1 trên val 17,6 điểm, CI 95% theo item [−21,1; −14,1]. Đổi pad sang trắng thu hẹp còn −5,5 điểm, CI [−8,4; −2,4]. | [Đã chạy] `reports/val_crop_ablation.json` |
| 5 | "Màu pad là nguyên nhân chính" được ủng hộ **ở mức định lượng**: pad giải thích khoảng 12 trên 17,6 điểm. **Cơ chế chưa xác minh.** Câu "không phải do mất bối cảnh" từng nói khi trình bày là **phát biểu quá mức**: phần −5,5 còn lại chưa tách được nguyên nhân. | xem mục 2 |
| 6 | Các kết quả SOTA 92–95% đều **train trên In-shop train**. Không so trực tiếp được với baseline zero-training, chỉ dùng làm "trần tham khảo". | [Nguồn nói] / [Suy luận] |
| 7 | Hướng ưu tiên: (a) head nhẹ + ArcFace trên embedding cache, rồi (b) LP-FT vài block cuối trên Kaggle T4. Crop với CLIP frozen nên dừng. | [Suy luận] |

## 1. Các thí nghiệm đã chạy

### 1.1 Dữ liệu và split [Đã chạy]

Dataset: DeepFashion In-shop. Gồm 52.712 ảnh, 7.982 item_id, 23 category, tất cả ảnh 256×256 JPEG.
- Partition khớp từng byte bản chính thức: `reports/sources/partition_provenance.json`.
- SHA-256 partition `cfada103…94c`, bbox `b1a67bf2…f440`.

| Split | Ảnh | item_id | Dùng cho |
|---|---:|---:|---|
| official train | 25.882 | 3.997 | Nguồn tạo val; sau này dùng để train |
| → train còn lại sau khi tách val | 23.266 | 3.597 | Train Task 2 (chưa dùng) |
| → val_query / val_gallery | 1.232 / 1.384 | 400 / 400 | **Chọn cấu hình** |
| official query | 14.218 | 3.985 | **Chỉ để đánh giá cuối** |
| official gallery | 12.612 | 3.985 | Distractor đầy đủ, không lọc |

Nguồn: `reports/audit.json`, notebook `artifacts/task1/task1_data_pipeline.executed.ipynb` (gitignored; bản không output là `notebooks/task1_data_pipeline.ipynb`).

Các kiểm tra leakage [Đã chạy]:
- Query/gallery không trùng đường dẫn.
- Item_id của train và eval tách biệt.
- Mọi query có positive.
- Không có ảnh trùng byte hoặc trùng pixel trong cùng split hay giữa các split (`reports/audit.json`).
- pHash ≤4 tìm được 1.357 cặp query–gallery gần giống, trong đó 134 cặp cùng ID. Mới xem thủ công 12 cặp và không loại cặp nào (`reports/BAO_CAO_PILOT.md`).

Tạo val (`vps/preprocess.py`, hàm `validation_split`):
- Chọn 10% item_id của train theo hash, seed `fcvg101-val`. Chỉ xét item có từ 2 ảnh trở lên.
- Trong mỗi item, sắp ảnh theo hash; nửa đầu (ít nhất 1 ảnh) làm query, phần còn lại làm gallery.
- Có test đơn vị: `tests/test_preprocess.py::test_validation_split_holds_out_whole_train_identities`.

Pipeline chung của mọi thí nghiệm:
- **Model:** OpenAI CLIP ViT-B/32 frozen, checkpoint SHA-256 `40d36571…a950af`.
- **Preprocess:** `clip.load` (Resize 224 bicubic → CenterCrop 224 → Normalize).
- **Embedding:** encoder chạy fp16, đổi embedding sang float32 rồi L2-normalize.
- **Retrieval:** cosine brute-force trên toàn gallery (`vps/metrics.py::retrieve`), tie-break ổn định.
- **Metric:** R@K. Một hit là có ảnh cùng `item_id` trong top-K. Hàm `recall_at`.

### 1.2 Official test — một lần chạy duy nhất [Đã chạy]

| Mục | Giá trị |
|---|---|
| Môi trường | Kaggle Tesla T4 (1 GPU), Python 3.13.15, torch 2.11.0+cu128 (`reports/cloud_run.json`, `reports/preflight.json`) |
| Cấu hình | C0: ảnh đầy đủ, preprocess CLIP gốc, không bbox, không train |
| Query / gallery | 14.218 / 12.612 (3.985 item, 23/23 category, 0 query bị loại) |
| **R@1 / R@5** | **45,506% (6.470 hit) / 67,246% (9.561 hit)** |
| MRR / median rank | 0,555 / 2 |
| Macro theo item R@1 / R@5 | 42,70% / 63,03% |
| Kiểm tra độc lập | Thuật toán đếm khác, chạy trên Windows: 0 thay đổi hit/miss ở K=1 và K=5; 14 query lệch một hạng (`reports/independent_verification.json`) |

R@1 theo category nằm trong `reports/metrics.json`:
- Thấp nhất: WOMEN/Graphic_Tees **18,9%** (365 query).
- Cao nhất: WOMEN/Skirts 63,2% (614 query).

Chi tiết từng query: `reports/per_query_results.csv`.

### 1.3 Validation: C0 / C1 / C3 [Đã chạy]

- Script: `scripts/eval_val.py`. Kết quả: `reports/val_results.json`.
- Môi trường: RTX 3050 Ti Laptop 4 GB, torch 2.14.1+cu130, Python 3.12.14, batch 32.
- `official_test_used: false`.

| Cấu hình | Tiền xử lý | Biểu diễn | R@1 | CI 95% (bootstrap theo item) | R@5 | MRR |
|---|---|---|---:|---|---:|---:|
| **C0_full** | Ảnh đầy đủ | CLIP 512-d | **71,10%** (876/1.232) | [68,0; 74,4] | 87,26% | 0,783 |
| **C1_bbox_pad** | Crop bbox +10% margin, cạnh ≥64 px, pad vuông màu (123,117,104) | CLIP 512-d | 53,57% (660) | [49,8; 57,3] | 69,48% | 0,611 |
| **C3_hsv_hist** | Crop + pad như C1 | HSV 8×4×4 = 128-d | 33,36% (411) | [30,2; 36,4] | 48,78% | 0,409 |

Chênh lệch R@1 của C1 so với C0: **−17,6 điểm**, CI [−21,1; −14,1]. Bootstrap ghép cặp, resample theo item_id, 2.000 lần.

Crop áp dụng **cho cả query lẫn gallery**. Augmentation (C2) **chưa chạy**: chỉ có ý nghĩa khi train.

### 1.4 Validation: ablation crop [Đã chạy]

- Script: `scripts/ablate_crop_val.py`. Kết quả: `reports/val_crop_ablation.json`.
- Cùng môi trường và cùng val như 1.3. Mỗi biến thể chỉ đổi một yếu tố so với C1.

| Biến thể | Thay đổi | R@1 | R@5 | R@1 ảnh nghiêng (264 query) | R@1 − C0, median [CI 95%] |
|---|---|---:|---:|---:|---|
| C0_full | Tham chiếu | 71,10 | 87,26 | 76,14 | — |
| C1_bbox_pad | Tham chiếu crop | 53,57 | 69,48 | 37,88 | −17,6 [−21,1; −14,1] |
| A1_white_pad | Pad trắng | **65,75** | 84,09 | 58,33 | **−5,5 [−8,4; −2,4]** |
| A2_margin_0.3 | Margin 0,3 | 54,87 | 70,29 | 36,36 | −16,3 [−19,9; −13,0] |
| A2_margin_0.5 | Margin 0,5 | 54,71 | 71,43 | 37,88 | −16,4 [−19,9; −13,1] |
| A3_full_if_narrow32 | Dùng ảnh đầy đủ nếu bbox rộng <32 px | 54,22 | 70,70 | 40,15 | −16,9 [−20,4; −13,5] |
| X_context_only | Che bbox (+10%) bằng màu mean, giữ phần còn lại | 22,73 | 38,31 | 20,08 | −48,5 [−52,3; −44,4] |

Trong val query có 6,3% bbox rộng dưới 32 px.

### 1.5 Làm rõ con số 71,1%

- **Metric:** Recall@1, tức tỉ lệ query có ảnh cùng item_id ở hạng 1.
- **Cấu hình:** C0, ảnh đầy đủ, **không crop**.
- **Split:** validation, 1.232 query, gallery 1.384 ảnh của 400 item tách từ train.

Số này **không so được với 45,5% của test**. Gallery val nhỏ hơn khoảng 9 lần (1.384 so với 12.612) và chỉ có 400 so với 3.985 item, nên ít distractor hơn nhiều và bài toán dễ hơn.

Nó chỉ dùng làm mốc để so các cấu hình với nhau trên cùng val. "Crop giảm hiệu quả" nghĩa là: thay C0 bằng C1 thì R@1 trên **cùng** val tụt từ 71,1% xuống 53,6%, và CI của chênh lệch không chứa 0.

Lưu ý: phần tóm tắt trình bày trước đó viết "71,1% xuống 53,6%". Cách viết này đúng nhưng dễ bị đọc nhầm thành số test.

## 2. Phân tích nguyên nhân

### 2.1 Crop có thực sự làm giảm độ chính xác không?

**Có, trong phạm vi đã đo** [Đã chạy]:
- Áp dụng cho CLIP B/32 frozen, In-shop val, crop cả hai phía query và gallery.
- Giảm ở mọi pose (`val_results.json → R@1_by_query_pose`), trừ nhóm stand-alone chỉ có 13 query.

| Pose | Số query | C0 | C1 | C3 |
|---|---:|---:|---:|---:|
| front | 298 | 80,5 | 65,1 | 44,3 |
| side | 264 | 76,1 | **37,9** | 11,7 |
| back | 236 | 65,3 | 57,2 | 46,6 |
| zoom-out | 140 | 66,4 | 52,1 | 49,3 |
| zoom-in | 281 | 63,7 | 53,0 | 23,8 |

Chưa kiểm chứng:
- Crop với **embedding đã được train trên crop**. Đây là cách industry dùng.
- Crop **chỉ ở phía query**.
- Crop trên official test.

Kết luận hiện tại **không** khẳng định "crop luôn hại".

### 2.2 Bằng chứng về màu pad, nền và mất ngữ cảnh

| Giả thuyết | Bằng chứng | Trạng thái |
|---|---|---|
| **Màu pad** | Chỉ đổi màu pad từ mean sang trắng: R@1 tăng 53,6 → 65,8. Khoảng cách tới C0 giảm từ −17,6 còn −5,5, hai CI không chồng nhau. Ảnh nghiêng tăng 37,9 → 58,3. | **Ủng hộ định lượng** [Đã chạy] |
| Cơ chế của hiệu ứng pad | (a) Ảnh nghiêng có bbox hẹp, nên sau khi pad vuông phần pad chiếm phần lớn khung, và embedding có thể bị khung xám chi phối. (b) Nền studio In-shop sáng, nên pad trắng giống phân bố ảnh gốc hơn. | **Chưa xác minh.** Chưa đo tỉ lệ diện tích pad theo hit/miss. Chưa đo màu nền thực của In-shop. [Suy luận] |
| Bbox hẹp không đáng tin | Chuyển sang ảnh đầy đủ khi bbox <32 px (A3) chỉ +0,7 điểm, ảnh nghiêng 37,9 → 40,2. Notebook Task 1 cho thấy 5,5% bbox toàn dataset rộng <32 px, hẹp nhất 1 px, chủ yếu là ảnh side (2.368 ảnh). | Bbox hẹp không phải nguồn chính [Đã chạy] |
| Thiếu margin | Margin 0,3 hoặc 0,5 chỉ +1,1 đến +1,3 điểm, CI chồng với C1. | Không phải nguồn chính [Đã chạy] |
| **Mất ngữ cảnh** (người mẫu, tóc, phụ kiện, nền cùng buổi chụp) | X_context_only đạt 22,7% R@1. Mức ngẫu nhiên khoảng 0,25%: trung bình 3,46 positive trên 1.384 ảnh gallery. Vậy ngữ cảnh **có mang tín hiệu nhận dạng**. Nhưng X không đo được bao nhiêu điểm mất đi khi crop. | **Chưa tách được.** Phần −5,5 còn lại của A1 có thể do ngữ cảnh, do tỉ lệ hoặc độ phân giải, hoặc do pad trắng vẫn khác nền thật. |
| Giới hạn của probe X | Vùng che hình chữ nhật có thể để sót vải ngoài box và tạo một vật thể lạ trong ảnh (Codex đã chỉ ra, xem `reports/NGHIEN_CUU_HUONG_DI.md`). | Ghi nhận là giới hạn |

**Đính chính:** câu "nguyên nhân chính là màu viền pad, không phải do mất bối cảnh" (nói khi trình bày trên lớp) nên sửa thành: "phần lớn mức giảm, khoảng 12 trên 17,6 điểm, gắn với màu pad; vai trò của ngữ cảnh trong phần còn lại chưa được đo".

### 2.3 Lỗi retrieval thường gặp

Phần đã đo được [Đã chạy]:
- Test có 3.091 query sai top-1 nhưng đúng top-5, và 4.657 query sai cả top-5 (`BAO_CAO_PILOT.md`).
- Category yếu nhất: Graphic Tees nữ (18,9%), Sweatshirts/Hoodies nữ (36,8%), Denim (38–39%).
- Theo pose trên val (C0): back (65,3%) và zoom-in (63,7%) thấp nhất; front cao nhất (80,5%).
- Phân bố zoom-in lệch giữa các split: query test 31,3%, gallery test 8,6%, train 20,7% (notebook Task 1). Query zoom-in phải khớp với gallery chủ yếu không zoom.

Phần định tính, chỉ từ 30 ca đã chọn trên test (`artifacts/examples/index.html`, `reports/examples_manifest.json`):
1. Ảnh toàn thân bị khớp sang món đồ khác trên cùng người mẫu. Ví dụ `fail_top5_01`: positive ở hạng 186.
2. Cùng màu và góc chụp nhưng khác sản phẩm. Ví dụ `fail_top1_recovered_top5_06`: cosine top-1 đạt 0,9716.
3. Người mẫu và tư thế giống nhau chi phối kết quả.

**Chưa xác minh tần suất:**
- Chưa có phân loại lỗi đếm trên toàn bộ dữ liệu.
- R@1 theo pose trên **test** chưa tính, dù dữ liệu đã có trong `per_query_results.csv` kết hợp file bbox.
- Câu "lỗi thường gặp: nhầm màu, nhầm tư thế, ảnh nghiêng" khi trình bày là tóm tắt định tính, không phải số đếm.

### 2.4 Kết luận đã chứng minh và giả thuyết chưa kiểm chứng

| Đã chứng minh (trong phạm vi đã nêu) | Giả thuyết hoặc chưa kiểm chứng |
|---|---|
| Baseline test 45,5 / 67,2, tái lập khớp Unicom | CLIP có từng thấy DeepFashion khi pretrain không (không loại trừ được) |
| C1 thấp hơn C0 trên val, CI không chứa 0 | Crop có lợi khi embedding được train trên crop |
| Pad trắng thu hẹp phần lớn khoảng cách | Cơ chế của hiệu ứng pad (tỉ lệ diện tích, phân bố nền) |
| Margin và fallback cho bbox hẹp gần như không giúp | Vai trò của ngữ cảnh trong 5,5 điểm còn lại |
| Ngữ cảnh một mình đạt 22,7% (rất xa mức ngẫu nhiên) | Head hoặc fine-tune sẽ tăng bao nhiêu trên In-shop với tài nguyên này |
| Kết quả C0 tất định khi giữ cùng batch size (fp16; đổi batch 64↔32 lệch 1 hit) | Tần suất từng loại lỗi |

Một giới hạn của C3 cần ghi rõ: HSV histogram được tính **trên ảnh đã pad bằng màu mean** (`eval_val.py`, dòng 89). Pixel pad lọt vào histogram, nên C3 bị bất lợi nhân tạo. Con số 33,4% vì vậy là mốc dưới của handcrafted, chưa phải giá trị công bằng.

## 3. Đối chiếu nghiên cứu

### 3.1 Unicom: CLIP B/32 đạt 45,4

[Nguồn nói] Unicom, ICLR 2023 ([arXiv 2304.05884](https://arxiv.org/abs/2304.05884)), đọc qua bản HTML ar5iv ngày 2026-10-30.
- **Bảng 2**, caption: "Performance of unsupervised image retrieval on 7 image retrieval datasets".
- R@1 In-shop của CLIP: B/32 **45,4**, B/16 56,1, L/14 62,9, L/14-336 66,3.
- R@1 In-shop của Unicom (pretrain trên LAION-400M phân cụm, **không train trên In-shop**): B/32 72,8, B/16 74,6, L/14 83,6, L/14-336 86,7.
- Setup: backbone frozen, dùng embedding trực tiếp để test. Metric In-shop là Recall@1.
- Bảng 11 của paper ghi In-shop có 25.882 ảnh / 3.997 lớp (train) và 26.830 ảnh / 3.985 lớp. Số 26.830 đúng bằng 14.218 + 12.612, tức query + gallery của split chính thức.

Đánh giá:
- **Khớp:** model, dataset, split, metric, và feature frozen không train.
- **Chưa xác minh:**
  - Resolution và preprocess lúc test (caption không nêu).
  - Unicom dùng chung một tập 26.830 ảnh cho cả query lẫn gallery hay tách query/gallery như chúng ta. Bảng 11 gộp hai tập, nên chưa rõ.
- 45,5 so với 45,4 là **khớp tốt nhưng chưa phải bằng chứng protocol giống hệt**.
- Lần đọc này dùng bản HTML qua công cụ tóm tắt, không phải đọc trực tiếp PDF. Thứ tự cột In-shop do công cụ suy ra. Reviewer nên tự mở Bảng 2.

### 3.2 Phương pháp đạt 92–95%

| Phương pháp | R@1 In-shop | Backbone / chiến lược | Tài nguyên | Mức kiểm tra |
|---|---:|---|---|---|
| Unicom fine-tune B/32 · B/16 · L/14 · L/14-336 | 94,8 · 95,5 · 96,0 · 96,7 | Pretrain Unicom, rồi fine-tune trên In-shop train (Bảng 4) | Paper không nêu batch hoặc GPU cho bước fine-tune. Bảng 7 (128 V100) là của pretrain | [Nguồn nói], đã đọc |
| Hyp-ViT (CVPR 2022) | 92,5 | ViT-S fine-tune, embedding hyperbolic, batch 900 | A100 | Lane nghiên cứu báo cáo, **chưa tự kiểm tra** |
| VPTSP (ICLR 2024) | 88,4–91,2 | PEFT visual prompt, ViT-S, batch 64 | 1 GPU | Lane báo cáo, **chưa tự kiểm tra** |
| UnED (ICCV 2023), CLIP B/16 full FT | 86,2 | Index 8 miền, không riêng In-shop | — | Lane báo cáo, **chưa tự kiểm tra** |

Nguồn đầy đủ: `reports/NGHIEN_CUU_HUONG_DI.md`.

### 3.3 Có so trực tiếp được không?

- **Unicom 45,4 so với baseline 45,5:** so được (cùng điều kiện chính, xem 3.1).
- **92–95% so với 45,5:** **không** so trực tiếp được.
  - Các phương pháp này train có giám sát trên 25.882 ảnh In-shop train.
  - Dùng backbone hoặc pretrain khác (Unicom pretrain riêng, ViT-S ImageNet), batch lớn, augmentation và loss chuyên cho metric learning.
  - Baseline của dự án là zero-training.
- Dùng được làm **trần tham khảo**: nếu dự án train trên cùng train split và test một lần trên cùng split, R@1 thu được nằm trên cùng thang đo với các paper. Tuy vậy vẫn phải ghi rõ khác biệt về backbone và compute.

## 4. Đánh giá các hướng cải tiến

| Hướng | Bằng chứng hiện có | Khả thi (máy 4 GB + Kaggle T4 miễn phí) | Lợi ích tiềm năng | Giải thích được |
|---|---|---|---|---|
| **Frozen CLIP + crop / preprocessing** | Âm trên val: biến thể tốt nhất (pad trắng) vẫn −5,5 [−8,4; −2,4] | Rất cao, mỗi biến thể khoảng 1 phút | **Thấp**: chưa biến thể nào vượt C0 | Cao (đã có câu chuyện về pad) |
| **Head nhẹ (Linear/MLP, có thể kèm BN) + ArcFace** trên embedding đã cache | Chưa chạy. GUIE hạng 5 dùng CLIP frozen + BN + Linear + ArcFace [Nguồn nói, abstract]. Trên CUB, linear probe gần bằng full FT (lane báo cáo). Chưa có số In-shop trong nguồn đã đọc | **Rất cao**: encode khoảng 23k ảnh một lần, train vài phút ngay cả trên CPU/GPU 4 GB | **Chưa rõ.** [Suy luận] Có thể tăng vừa phải. Bị giới hạn vì không đổi được đặc trưng gốc, và không dùng được augmentation ảnh nếu chỉ cache một view | **Cao**: một yếu tố, ablation rõ ràng |
| **Partial fine-tune CLIP (LP-FT, mở 2–4 block cuối)** | Chưa chạy. Fine-tune toàn bộ đạt 86–96% trong các paper (điều kiện khác) | Trung bình: ước tính 2–5 giờ Kaggle T4 [Suy luận, chưa đo]. Train trên 4 GB local chưa chắc đủ bộ nhớ [Chưa xác minh] | **Cao nhất** trong các hướng [Suy luận] | Trung bình: nhiều siêu tham số (lr, số block, sampler P×K) |
| **Data augmentation** | Mới có code (`train_augment`, C2), chưa chạy | Chỉ có ý nghĩa khi đi kèm fine-tune | Phụ thuộc hướng fine-tune | Cao nếu ablate một yếu tố |

Lưu ý cho reviewer về config C2: C2 hiện được định nghĩa là **crop + augmentation**. Vì crop đã thua trên val, ablation augmentation cho Task 2 nên làm trên **ảnh đầy đủ**. Cấu hình "C0 + aug" **chưa có** trong `CONFIGS` (`vps/preprocess.py`). Augmentation cố ý giữ nguyên hue và saturation vì màu là đặc trưng nhận dạng sản phẩm.

**Ưu tiên** cho sinh viên làm cá nhân, cần kết quả thực, dễ giải thích, GPU rẻ [Suy luận]:
1. Head + ArcFace trên embedding cache: rẻ nhất, cô lập được lợi ích của "adaptation nhẹ".
2. LP-FT vài block cuối trên Kaggle T4: kỳ vọng tăng nhiều nhất, vẫn miễn phí.
3. Augmentation: làm ablation một yếu tố bên trong bước 2.

Crop với CLIP frozen nên dừng. Giữ lại làm phân tích lỗi (câu chuyện về pad), không làm hướng cải tiến.

## 5. Đánh giá độc lập

### 5.1 Không gian đóng góp

SOTA In-shop đã bão hoà ở mức khoảng 92–97%, nên không có đóng góp kiểu "vượt SOTA". Những đóng góp thực tế có thể làm [Suy luận]:
- **Đo có kiểm soát** xem mỗi bước rẻ đóng được bao nhiêu phần khoảng cách 45,5 → ~95, dưới ngân sách GPU miễn phí. Các bước: head → LP-FT → augmentation, mỗi lần đổi một yếu tố, có CI.
- Một con số **linear-probe hoặc head trên CLIP cho In-shop**. Trong các nguồn đã đọc, chưa thấy số này.
- **Phân tích lỗi theo pose và category**, cùng phát hiện về hiệu ứng pad khi crop cho encoder frozen. Ghi đúng phạm vi: CLIP B/32 trên In-shop val.

### 5.2 Rủi ro còn tồn tại

| Rủi ro | Hiện trạng | Mức |
|---|---|---|
| Test-set tuning | Test chạy một lần cho baseline, mọi lựa chọn cấu hình làm trên val. **Nhưng** 30 ca lỗi và số theo category của **test** đã được xem, và giả thuyết crop nảy ra từ ca test `fail_top5_01`. Đây là rò rỉ nhẹ vào **việc hình thành giả thuyết**. Giả thuyết sau đó được kiểm chứng trên val nên tác động nhỏ, nhưng nên ghi trong báo cáo cuối | Thấp–trung bình |
| Overfit val | Đã thử 7 cấu hình trên cùng 400 item. CI rộng khoảng ±3 điểm | Trung bình. Cần chốt số lần thử trước Task 2 |
| Val dễ hơn test | Gallery nhỏ hơn khoảng 9 lần. Phân bố pose của val do hash quyết định, không theo kiểu query/gallery chính thức (test: query zoom-in 31,3%, gallery 8,6%) | Trung bình. Thứ hạng cấu hình trên val có thể không giữ nguyên trên test |
| Leakage khi train | `validation_split` đã loại item val khỏi train và có test đơn vị. Script train **chưa tồn tại**, nên cần assert lại trong script đó | Thấp nếu giữ assert |
| Pretrain đã thấy dữ liệu | CLIP, LAION hoặc Marqo-FashionCLIP có thể từng thấy ảnh DeepFashion | Không loại trừ được. Ghi là giới hạn |
| Ảnh gần trùng | 134 cặp query–gallery cùng ID có pHash ≤4 vẫn được giữ trong benchmark (theo protocol chính thức) | Thấp, cùng điều kiện với các paper |
| Tái lập | fp16 lệch ±1 hit khi đổi batch. Test chạy trên môi trường khác val (Kaggle torch 2.11 so với local 2.14.1) | Thấp. So cấu hình luôn trong cùng một lần chạy |
| Bootstrap | CI theo item phản ánh độ bất định của **val**, không phải của test | Ghi rõ khi báo cáo |
| Phạm vi repo | `CLAUDE.md` của repo vẫn ghi phạm vi pilot: "No fine-tuning, additional trained model". Phải cập nhật trước khi làm E1/E2 | Quy trình, không phải lỗi kỹ thuật |

### 5.3 Kế hoạch thí nghiệm tiếp theo (2–3 cái có giá trị nhất)

**Bước chuẩn bị (không tính là thí nghiệm).**
- Chốt protocol chọn mô hình trước khi chạy:
  - Giới hạn số cấu hình.
  - Tiêu chí chọn là R@1 trên val, thêm CI.
  - Test chỉ chạy một lần cho cấu hình cuối cùng và cho các hàng tham chiếu.
- Cân nhắc làm val khó hơn: tăng tỉ lệ val, hoặc báo cáo kèm kích thước gallery.

**E1. Head + ArcFace trên embedding C0 đã cache.**
- So sánh: T0 (frozen) và T1 (Linear 512→512 + ArcFace; một biến thể Proxy Anchor nếu còn thời gian).
- Train trên 23.266 ảnh train còn lại.
- Đo R@1/R@5 trên val với CI ghép cặp.
- Chi phí: vài phút, chạy local được.
- Trả lời câu hỏi: adaptation nhẹ đóng được bao nhiêu phần khoảng cách?

**E2. LP-FT mở 2–4 block cuối của CLIP B/32 trên Kaggle T4.**
- Khởi tạo head từ E1, fp16, sampler P×K. Thêm một ablation: **có hoặc không** augmentation trên ảnh đầy đủ (cần thêm config "C0 + aug").
- Trả lời câu hỏi: mở backbone tăng thêm bao nhiêu so với E1, và augmentation có giúp không?

**E3 (rẻ, tăng độ tin cậy cho phần phân tích).**
- Chạy CLIP B/16 frozen làm hàng tham chiếu cho backbone. Paper báo 56,1, nên đây cũng là một lần kiểm tra tái lập nữa.
- Phân tích lỗi trên dữ liệu đã có: R@1 theo pose trên **test** từ `per_query_results.csv`, và tỉ lệ diện tích pad so với hit/miss của C1 trên val. Phần này chỉ để diễn giải, không dùng để chọn cấu hình.

## 6. Bằng chứng và file tham chiếu

| Nội dung | File |
|---|---|
| Metric test | `reports/metrics.json`, `reports/per_query_results.csv`, `reports/independent_verification.json` |
| Môi trường và lần chạy test | `reports/cloud_run.json`, `reports/preflight.json`, `reports/embed_timing.json`, `reports/environment.freeze.txt` |
| Audit dữ liệu và leakage | `reports/audit.json`, `reports/input_inspection.json`, `reports/sources/` |
| Val C0/C1/C3 | `reports/val_results.json`, `scripts/eval_val.py` |
| Ablation crop | `reports/val_crop_ablation.json`, `scripts/ablate_crop_val.py` |
| Pipeline Task 1 | `vps/preprocess.py`, `tests/test_preprocess.py`, `notebooks/task1_data_pipeline.ipynb`, `artifacts/task1/task1_data_pipeline.executed.ipynb` (local) |
| Retrieval và metric | `vps/metrics.py`, `vps/embed.py` |
| Báo cáo feasibility | `reports/BAO_CAO_PILOT.md` |
| Nghiên cứu và phản biện Codex | `reports/NGHIEN_CUU_HUONG_DI.md` |

Bộ test (32 test) pass ở lần chạy gần nhất trước nhiệm vụ này. **Trong nhiệm vụ này không chạy lại.**
