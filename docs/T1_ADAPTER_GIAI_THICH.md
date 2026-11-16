# T1 — Adapter E1: dự án tổng thể và giải thích từng phần

Viết 2026-11-14 cho Trung và Phong. Đọc trước: [HIEU_VA_PHAN_BIEN.md](HIEU_VA_PHAN_BIEN.md) (tư duy ngược), [KE_HOACH.md](KE_HOACH.md) (task).

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
│ C0 ảnh đầy đủ ✔ chọn │ C1 crop+pad ✘ │ C3 HSV ✘ │ augment   │
└───────────────┬─────────────────────────────────────────────┘
                ▼
┌──────────────── TASK 2: MÔ HÌNH ────────────────────────────┐
│ E0 CLIP đóng băng → vector 512 → cosine     ✔ xong          │
│ E1 CLIP đóng băng → ADAPTER → vector 512    ◀── T1 (file này)│
│ E2 như E1 + ảnh augmented cache sẵn          tuần 2          │
│ E3 như E1 trên ảnh crop + pad trắng          nếu kịp         │
│ → chọn trên val → chạy test 1 lần → demo + phân tích lỗi     │
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
**Đánh đổi:** mỗi ảnh chỉ có một vector cố định → augmentation không có tác dụng. E2 sẽ sửa bằng cách cache thêm vài bản augmented.

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
Kết quả: `reports/adapter_e1_val_ratio<r>_seed<s>.json`. Checkpoint: `cache/adapter/` (không commit; tuần 2 đưa lên Drive).

## 5. Kết quả

Chạy 2026-11-14 trên RTX 3050 Ti Laptop (4 GB), seed 0, ratio 0,2, 20 epoch. File: `reports/adapter_e1_val_ratio0.2_seed0.json`. **[Đã chạy, chỉ trên val — test chưa mở.]**

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
- **Val dễ hơn test nhiều** (400 món so với 3.985 món). Không được nói "adapter đạt 90%". Chỉ nói được: **trên val, adapter hơn CLIP đóng băng khoảng 19 điểm**. Con số test sẽ thấp hơn, chỉ biết khi chạy T5.
- **Mới chạy 1 seed.** Chênh 19 điểm lớn hơn nhiều so với độ dao động thường gặp giữa các seed, nhưng báo cáo nên có ≥ 3 seed.
- **Epoch 20 vẫn còn tăng nhẹ** → có thể train thêm. Chọn số epoch trên val, không chọn trên test.
- Theo quy tắc M2 trong kế hoạch (tăng ≥ 5 điểm trên val) → **làm tiếp E2**.

**Chi phí:** encode 25.882 ảnh mất **6.305 s (~1 giờ 45 phút)**, chậm hơn ước tính nhiều, có lẽ do đọc ảnh từ ổ E:. Train adapter chỉ mất vài giây mỗi epoch.

## 6. Phong cần nắm gì từ file này

- Đọc mục 1–2 là đủ để giải thích dự án; mục 3 khi bị hỏi "adapter là gì / sao ArcFace".
- Việc của Phong không cần sửa code này. Phân tích lỗi (P3/P5) dùng cùng val query/gallery; danh sách query sai lấy từ thứ hạng trong script.
- Câu hỏi vặn nên tự trả lời được: *"Vì sao không dùng ArcFace để đoán món ở test?"* · *"Ratio 0 thì sao?"* · *"Sao không cho augmentation vào E1?"*
