# T1 — Adapter E1: dự án tổng thể và giải thích từng phần

Viết 2026-11-14, cập nhật 2026-11-19 (đã có kết quả test). Cho Trung và Phong. **Muốn xem từng bước bằng sơ đồ: [PIPELINE_TUNG_BUOC.md](PIPELINE_TUNG_BUOC.md).** Đọc trước: [HIEU_VA_PHAN_BIEN.md](HIEU_VA_PHAN_BIEN.md) (tư duy ngược), [KE_HOACH.md](KE_HOACH.md) (task).

---

## 1. Toàn cảnh dự án — mình đang ở ô nào

```text
┌──────────────────────── DỮ LIỆU ─────────────────────────┐
│ DeepFashion In-shop: 52.712 ảnh, 7.982 mặt hàng          │
│ official train 3.997 món │ official test 3.985 món KHÁC │
└───────────────┬──────────────────────────┬───────────────┘
                │ chia tiếp theo item_id    │ (khóa lại, chỉ mở 1 lần cuối)
                ▼                           ▼
   adapter-train 3.597 món        test: query 14.218 / gallery 12.612
   val 400 món (query/gallery)
                │
┌───────────────▼──── TASK 1: XỬ LÝ ẢNH (xong) ───────────────┐
│ C0 ảnh đầy đủ │ C1/A1 crop+pad │ C3 HSV │ augmentation (xong)│
└───────────────┬─────────────────────────────────────────────┘
                ▼
┌──────────────── TASK 2: MÔ HÌNH ────────────────────────────┐
│ E0 CLIP đóng băng → vector 512 → cosine     ✔ xong          │
│ E1 CLIP đóng băng → ADAPTER → vector 512    ✔ xong (file này)│
│ E2 như E1 + ảnh augmented cache sẵn          ✔ xong (hòa E1) │
│ E3 như E1 trên ảnh crop + pad trắng          ✔ xong (nhỉnh)  │
│ → chọn trên val ✔ → test 1 lần ✔ → demo ✔ → lỗi: Phong ghi  │
└─────────────────────────────────────────────────────────────┘
┌──────────────── TASK 3 (Phong, độc lập) ────────────────────┐
│ 3A hiệu chỉnh camera + lọc/FFT/gradient/keypoint            │
│ 3B optical flow                                              │
└─────────────────────────────────────────────────────────────┘
```

## 2. Pipeline của T1

```text
 ảnh train (25.882)                    ┌─── chỉ chạy 1 lần, ~2–3 phút GPU ───┐
      │                                │                                     │
      ▼                                │                                     │
 CLIP ViT-B/32 (đóng băng)  ───────────┘  lưu vector vào cache/embeddings/   │
      │ vector f (512 số)                 train_full.npy                     │
      ▼                                                                      
 ┌──────────── ADAPTER (phần duy nhất được học) ────────────┐
 │ f ─► Linear 512→128 ─► ReLU ─► Linear 128→512 ─► ReLU ─► A(f)
 │ f' = 0,2 × A(f) + 0,8 × f      (residual: giữ 80% CLIP gốc)
 │ chuẩn hóa độ dài f' = 1                                   │
 └──────────────┬───────────────────────────────────────────┘
                ▼   chỉ lúc TRAIN
 ArcFace: 3.597 "tâm" (mỗi món train một tâm)
 kéo f' về tâm đúng, đẩy xa tâm khác ít nhất một góc m = 0,3 rad
                │   loss giảm → adapter học được cách uốn vector
                ▼
 Hết train: VỨT ArcFace, giữ adapter
                ▼
 Val: query + gallery đều đi qua CLIP → adapter → cosine → Top-5 → Recall@1/5
 Chọn epoch có R@1 val cao nhất, lưu checkpoint
```

## 3. Giải thích từng phần

### 3.1 Vì sao lưu sẵn vector (cache)?
CLIP đóng băng nên cùng một ảnh luôn ra cùng vector. Tính một lần (~2–3 phút trên RTX 3050 Ti) rồi train adapter trên 23.266 vector chỉ mất vài giây mỗi epoch, chạy được cả trên CPU.
**Đánh đổi:** mỗi ảnh chỉ có một vector cố định → augmentation không có tác dụng. E2 đã thử cách sửa: cache thêm 3 bản augmented mỗi ảnh (mục 5b) — kết quả là hòa với E1.

### 3.2 Adapter (`vps/adapter.py`, lớp `Adapter`, `AdaptedEmbedder`)
- **Nút cổ chai 512→128→512:** ép thông tin qua 128 chiều nên adapter chỉ học được những thay đổi "đơn giản", khó học thuộc lòng. Tham số: 512×128 + 128×512 = **131.072**, so với ~88 triệu của bộ mã hóa ảnh CLIP.
- **Residual ratio 0,2:** kết quả = 20% vector mới + 80% vector gốc. Nếu adapter học sai, kết quả vẫn gần CLIP gốc nên không sập. `ratio = 0` thì đúng bằng CLIP đóng băng (có test kiểm tra).
- **Hai ReLU, không bias:** giữ đúng code tác giả. Paper viết công thức rút gọn hơn; nhóm chọn theo code và ghi rõ.

### 3.3 Vì sao ArcFace, và vì sao vứt đi sau khi train?
Mặt hàng ở test chưa từng thấy, nên không thể dùng một bộ phân loại để đoán tên món. Nhưng **khi train**, coi mỗi món train là một lớp thì có tín hiệu học: ảnh cùng món phải gần nhau, khác món phải xa. ArcFace thêm **lề góc m**: chưa đủ gần tâm đúng thêm 0,3 rad thì vẫn bị phạt → các món tách nhau rõ hơn. Học xong thì cái được giữ lại là **cách uốn vector**, áp dụng được cho món mới.
- `s = 30`: phóng đại cosine để softmax đủ "sắc" (giá trị thường dùng trong paper ArcFace).
- Hạng 5 cuộc thi Google Universal Image Embedding 2022 cũng dùng CLIP đóng băng + head + ArcFace — đó là lý do chọn làm điểm xuất phát.

### 3.4 Chọn epoch thế nào cho công bằng?
Mỗi epoch chấm R@1 trên **val** (400 món không có trong train), giữ epoch tốt nhất. **Test không được đọc** trong script — đề cấm chọn tham số trên test.

### 3.5 Kiểm tra code có đúng không (trước khi tin con số)
| Kiểm tra | Ý nghĩa | Ở đâu |
|---|---|---|
| `ratio=0` ⇒ đúng bằng CLIP gốc | adapter gắn đúng chỗ | `tests/test_adapter.py` |
| Dữ liệu giả dễ tách ⇒ loss giảm > 50% | vòng train hoạt động | `tests/test_adapter.py` |
| Loss ban đầu ≈ ln(3.597) ≈ 8,2 hoặc hơi cao hơn | ArcFace nối đúng nhãn | in ra khi chạy |
| E0 val trên cache mới ≈ 71,1% (số Task 1) | cache mới khớp thí nghiệm cũ | in ra khi chạy |
| item_id train ∩ val = ∅ | không rò rỉ | `assert` trong script |

## 4. Cách chạy

```bash
cd visual-product-search-pilot
.venv/Scripts/python.exe -m pytest tests/test_adapter.py -q
.venv/Scripts/python.exe scripts/train_adapter.py --device cuda            # ratio 0,2
.venv/Scripts/python.exe scripts/train_adapter.py --device cuda --ratio 0.5 # T3
```
Kết quả: `reports/adapter_e1_val_ratio<r>_seed<s>.json`. Checkpoint: `cache/adapter/` (không commit); bản 40 epoch dùng ở test có trên Google Drive, link trong `MODEL_LINKS.md`. Demo và phân tích lỗi: `notebooks/FCVG101_main.ipynb`.

## 5. Kết quả

Chạy 2026-11-14 trên RTX 3050 Ti Laptop (4 GB), seed 0, ratio 0,2, 20 epoch. File: `reports/adapter_e1_val_ratio0.2_seed0.json`. **[Đã chạy, chỉ trên val; lúc đó test chưa mở, test đã chạy một lần ở mục 5d.]**

| | R@1 val | R@5 val |
|---|---|---|
| E0 CLIP đóng băng | 71,02% | 87,34% |
| **E1 + adapter (epoch 19)** | **89,94%** | **97,24%** |
| Chênh R@1 | **+18,9 điểm**, CI 95% theo item [+16,0; +21,9] | |

Kiểm tra đã qua:
- E0 trên cache mới 71,02% so với 71,1% ở Task 1: lệch 1 query, đã biết là do fp16 + batch size khác.
- Loss ban đầu 9,10 so với ln(3.597) = 8,19: hơi cao như dự kiến với s = 30. Loss train epoch 1 cao hơn (16,0) vì lúc đó đã có lề m; giảm còn 3,2 ở epoch 20.
- R@1 tăng đều qua các epoch, không nhảy đột ngột.

Đọc kết quả một cách khách quan (tư duy ngược):
- **Tăng lớn hơn dự kiến** (reviewer đoán adapter chỉ đạt khoảng 58–72% trên test). Kết quả tốt bất ngờ thì nghi lỗi trước: đã kiểm tra item_id train và val không trùng; val chỉ dùng để chọn epoch. Chưa thấy lỗi.
- **Val dễ hơn test nhiều** (400 món so với 3.985 món). Không được nói "adapter đạt 90%". Chỉ nói được: **trên val, adapter hơn CLIP đóng băng khoảng 19 điểm**. Con số test thấp hơn thật: E1 đạt 71,4% trên test (mục 5d).
- **Đã chạy 3 seed** (2026-11-17): R@1 val 89,94 / 90,58 / 89,53% → trung bình **90,0%**, dao động ~1 điểm, nhỏ hơn rất nhiều so với mức tăng 19 điểm. R@5 97,24 / 97,32 / 96,67%. Epoch tốt nhất 18–19. File: `reports/adapter_e1_val_ratio0.2_seed{0,1,2}.json`.
- **Epoch 20 vẫn còn tăng nhẹ** → có thể train thêm. Chọn số epoch trên val, không chọn trên test.
- Theo quy tắc M2 trong kế hoạch (tăng ≥ 5 điểm trên val) → **làm tiếp E2**.

**Chi phí:** encode 25.882 ảnh mất **6.305 s (~1 giờ 45 phút)**, chậm hơn ước tính nhiều, có lẽ do đọc ảnh từ ổ E:. Train adapter chỉ mất vài giây mỗi epoch.

## 5b. T2 + E2: thêm ảnh augmented — kết quả (2026-11-17)

**Cách làm:** mỗi ảnh train được đọc một lần và tạo **3 bản augmented**: crop ngẫu nhiên 75–100%, lật ngang, đổi sáng/tương phản ±20%, xóa ngẫu nhiên một mảng nhỏ. Màu sắc giữ nguyên vì màu là đặc điểm nhận dạng mặt hàng. Đây đúng là augmentation đã định nghĩa ở Task 1 (`train_augment`), áp lên ảnh đầy đủ (cấu hình `C0_full_aug`). Mỗi epoch, mỗi ảnh lấy ngẫu nhiên 1 trong 4 bản (bản gốc + 3 augmented). Encode mất 256 giây (`scripts/encode_augmented.py`, có seed, chạy lại ra đúng từng số).

**Kết quả trên val, 3 seed, cùng số epoch** [Đã chạy]:

| | Epoch | R@1 TB [min–max] | R@5 TB [min–max] |
|---|---|---|---|
| E0 CLIP đóng băng | — | 71,02 | 87,34 |
| E1 adapter | 20 | 90,02 [89,53–90,58] | 97,08 [96,67–97,32] |
| E2 adapter + augmented | 20 | 90,15 [90,02–90,26] | 96,73 [96,51–97,08] |
| E1 adapter | 40 | 90,94 [90,50–91,31] | 97,24 [96,75–97,48] |
| E2 adapter + augmented | 40 | 90,99 [90,42–91,31] | 96,97 [96,59–97,40] |

File: `reports/adapter_{e1,e2}_val_ratio0.2_seed{0,1,2}[_ep40].json`. Trong JSON của E2, kết quả vẫn nằm ở khóa `E1_adapter` (tên khóa cũ).

**Đọc kết quả:**
- **E2 ≈ E1:** chênh 0,05–0,13 điểm R@1, nhỏ hơn nhiều so với dao động giữa các seed (~1 điểm). Kết luận đúng là **hòa**, không nói "augmentation giúp".
- **Vì sao augmentation không giúp ở đây** [Phán đoán, có một số đo ủng hộ]: CLIP đóng băng vốn ít bị ảnh hưởng bởi crop nhẹ, lật hay đổi sáng. Cosine giữa hai bản augmented của cùng một ảnh trung bình là **0,93** (đo trên 64 ảnh), nên adapter thấy gần như cùng một vector. Augmentation thường có tác dụng khi **chính encoder** được học (fine-tune); với encoder đóng băng thì tác dụng nhỏ.
- **40 epoch nhỉnh hơn 20 epoch khoảng 0,9 điểm**, và epoch tốt nhất rơi vào 26–37. Số epoch được chọn trên val, đúng quy tắc.
- Đây là một **kết quả âm có giải thích**, đáng viết vào báo cáo: nó nối Task 1 với Task 2 và cho thấy nhóm không chỉ báo những gì "thắng".

## 5c. T3 (tỉ lệ residual) và E3 (ảnh crop + pad trắng) — kết quả (2026-11-17)

Cả hai đều chạy 40 epoch, 3 seed, chỉ trên val [Đã chạy].
- **E3:** encode lại ảnh train và val theo cấu hình A1 của Task 1 (crop theo bbox, lề 10%, pad vuông màu trắng), mất 96 giây (`scripts/encode_variant.py`). Sau đó train adapter giống hệt E1 (`--base-cache train_A1_white_pad`).

| Cấu hình | Ảnh đầu vào | CLIP đóng băng R@1 | + adapter R@1 TB [min–max] | R@5 TB |
|---|---|---|---|---|
| E1, ratio 0,2 | đầy đủ | 71,02 | 90,94 [90,50–91,31] | 97,24 |
| E1, ratio 0,5 (T3) | đầy đủ | 71,02 | 91,34 [91,15–91,56] | 97,38 |
| E2, ratio 0,2 | đầy đủ + augmented | 71,02 | 90,99 [90,42–91,31] | 96,97 |
| **E3, ratio 0,2** | **crop + pad trắng** | **65,75** | **91,69 [91,40–92,05]** | **98,00** |

**Đọc kết quả:**
- **Phát hiện đáng kể nhất là kết quả bị đảo chiều.** Với CLIP đóng băng, crop + pad trắng **kém** ảnh đầy đủ 5,3 điểm (đúng như Task 1). Sau khi có adapter, crop lại **nhỉnh hơn** khoảng 0,75 điểm R@1 và 0,8 điểm R@5. Một cách giải thích [Phán đoán, chưa kiểm chứng riêng]: crop bỏ bớt người mẫu và nền, nên vector tập trung vào món đồ hơn. CLIP gốc không quen ảnh có viền trắng nên bị giảm điểm; adapter được train trên đúng loại ảnh đó nên "học quen" được viền trắng và tận dụng được phần tập trung vào món đồ.
- **Mức chênh nhỏ, cần nói thận trọng.** Khoảng min–max của E3 và E1 (ratio 0,2) không chồng nhau, nhưng chỉ cách nhau dưới 1 điểm, trên val 400 món, và nhóm đã so khá nhiều cấu hình trên cùng tập val. Chỉ nên nói: *"trên val, E3 nhỉnh hơn một chút và ổn định qua 3 seed"*.
- **Ratio 0,5 nhỉnh hơn 0,2 khoảng 0,4 điểm**, và hai khoảng gần như chạm nhau, nên coi là **gần hòa**. Ratio 0,5 hội tụ sớm hơn (epoch tốt nhất 26–29, so với 34–37 của ratio 0,2). Chưa thử E3 với ratio 0,5. Dừng tinh chỉnh ở đây để tránh "chọn đi chọn lại" trên cùng tập val.

## 5d. T5 — kết quả TEST chính thức, chạy một lần (2026-11-17)

Năm cấu hình được chốt trên val **trước** khi mở test: tỉ lệ residual 0,2, checkpoint 40 epoch của 3 seed. Test gồm 14.218 query và toàn bộ 12.612 ảnh gallery (3.985 món chưa từng thấy khi train). Không tham số nào được chọn dựa trên test. Script: `scripts/eval_test.py`. File kết quả: `reports/test_results.json` [Đã chạy].

| Cấu hình | R@1 TB [min–max] | R@5 | R@10 |
|---|---|---|---|
| E0 CLIP đóng băng, ảnh đầy đủ | 45,51 | 67,25 | 74,36 |
| E0 CLIP đóng băng, crop + pad trắng | 39,58 | 60,85 | 68,54 |
| E1 + adapter, ảnh đầy đủ | 71,42 [71,25–71,56] | 88,93 | 92,72 |
| E2 + adapter + augmented | 71,06 [70,24–71,77] | 88,59 | 92,54 |
| **E3 + adapter, crop + pad trắng** | **72,20 [72,04–72,51]** | **89,29** | **93,25** |

**Đọc kết quả:**
- **E0 trên test ra 45,51%**, khớp đúng baseline cũ. Đây là dấu hiệu pipeline chấm điểm không bị thay đổi.
- **Adapter tăng khoảng 26 điểm R@1 trên test** (45,5 → 71,4), lớn hơn mức tăng trên val (~20 điểm). Trên test, tất cả 23 nhóm sản phẩm đều tăng, ít nhất 8 điểm (MEN/Sweatshirts) và nhiều nhất 33 điểm (MEN/Shirts_Polos). Nhóm khó nhất vẫn là WOMEN/Graphic_Tees: 18,9 → 46,0%.
- **Thứ tự các cấu hình trên test giống trên val:** E3 > E1 ≈ E2, và crop kém đi khi CLIP đóng băng. Vì vậy các kết luận chọn trên val vẫn đứng được trên test.
- **So với tài liệu** [Nguồn, `reports/NGHIEN_CUU_HUONG_DI.md`]: CLIP B/32 đóng băng 45,4%; Unicom B/32 đóng băng 72,8%; các phương pháp fine-tune toàn bộ mạng đạt 86–95%. Adapter của nhóm chỉ thêm 131 nghìn tham số và train vài chục giây, đưa CLIP B/32 lên ngang mức Unicom đóng băng, nhưng **vẫn kém fine-tune khoảng 15–20 điểm**. Đây là giới hạn đúng như phần tư duy ngược đã dự đoán: adapter chỉ sắp xếp lại thông tin mà CLIP đã nhìn thấy.
- **Không nói:** "vượt state-of-the-art" hay "phương pháp mới". **Nên nói:** "tái sử dụng CLIP-Adapter cho bài toán retrieval, rẻ, tăng 26 điểm; muốn tiến xa hơn thì phải fine-tune encoder".

## 6. Phong cần nắm gì từ file này

- Đọc mục 1–2 là đủ để giải thích dự án; mục 3 khi bị hỏi "adapter là gì / sao ArcFace".
- Việc của Phong không cần sửa code này. Phân tích lỗi (P3/P5) dùng cùng val query/gallery; danh sách query sai lấy từ thứ hạng trong script.
- Câu hỏi vặn nên tự trả lời được: *"Vì sao không dùng ArcFace để đoán món ở test?"* · *"Ratio 0 thì sao?"* · *"Sao không cho augmentation vào E1?"*
