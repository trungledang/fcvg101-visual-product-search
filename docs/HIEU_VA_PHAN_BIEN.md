# Hiểu và phản biện đề tài — bản đọc trước khi trình bày hoặc giao việc

Viết 2026-11-14, cập nhật trạng thái 2026-11-19 (đã có kết quả test; các mục dự đoán giữ nguyên để đối chiếu với kết quả thật). Mục đích: đọc một lần là nhớ lại được đề tài, giải thích được pipeline cho thầy, giao việc cho Phong và trả lời được khi bị hỏi vặn. Môn tự chọn nên mục tiêu là **hiểu và làm đủ, trung thực**, không phải đạt kết quả cao nhất.

Nhãn: **[Đã chạy]** có số đo trong repo · **[Nguồn]** từ paper, đã ghi trong `reports/NGHIEN_CUU_HUONG_DI.md` · **[Phán đoán]** lập luận, chưa đo.

Hướng chi tiết vẫn ở [HUONG_DI_HIEN_TAI.md](HUONG_DI_HIEN_TAI.md). File này không thay hướng, chỉ thêm góc nhìn ngược và phân công.

---

## 1. Đề tài trong ba câu

1. **Bài toán:** cho một ảnh quần áo, tìm trong kho ảnh những ảnh **cùng đúng mặt hàng** đó, chụp ở góc khác (DeepFashion In-shop).
2. **Cách làm:** CLIP biến ảnh thành vector 512 số; ảnh nào có vector gần nhất (cosine) thì xếp lên đầu. Ta thêm một **adapter** nhỏ học lại cách "uốn" vector cho phù hợp với quần áo.
3. **Đo bằng gì:** Recall@K — trong K ảnh đầu tiên có ảnh nào đúng mặt hàng không. Hiện CLIP để nguyên đạt **R@1 45,5%, R@5 67,2%** trên test [Đã chạy].

Thuộc mục **E (Foundation Models — efficient adaptation)** và một phần **D (embedding, Recall@K)** của đề.

## 2. Pipeline

```text
 DeepFashion In-shop: 52.712 ảnh, 7.982 mặt hàng (item_id)
        │  chia theo item_id: train 3.997 món │ test 3.985 món KHÁC hẳn
        ▼
 [Task 1] Xử lý ảnh ── ảnh đầy đủ (C0) │ crop bbox + pad (C1, A1) │ HSV histogram (C3)
        │             augmentation: lật, crop ngẫu nhiên, đổi màu nhẹ
        ▼
 CLIP ViT-B/32 (đóng băng) ── ảnh → vector 512
        ▼
 [Task 2] Adapter: 512 → 128 → 512, cộng dư (residual) với vector gốc
        │  chỉ adapter được train, trên ảnh của train item_id
        ▼
 Chuẩn hóa L2 → cosine với toàn bộ gallery → Top-K
        ▼
 Recall@1/5, phân tích ảnh sai, demo 1 ảnh mới → Top-5
```

Ý chính để nói với thầy: **phần nhóm thay đổi là bộ biến ảnh thành vector; bước so sánh cosine giữ nguyên.**

## 3. Tư duy ngược từng bước

Mỗi lựa chọn: vì sao chọn, nếu sai thì sao, và làm sao biết.

| Bước | Vì sao chọn | Nếu sai / rủi ro | Cách biết |
|---|---|---|---|
| Retrieval thay vì classification | Mặt hàng ở test **không có** trong train, nên không thể phân loại theo tên món | — đây là bắt buộc theo cách chia của dataset | Kiểm tra: item_id train ∩ test = ∅ [Đã chạy, `reports/audit.json`] |
| CLIP | Học trên ~400 triệu cặp ảnh–chữ; đề cho phép model pretrained | CLIP giỏi **ngữ nghĩa** ("váy đỏ") hơn **từng món cụ thể** ("đúng cái váy đỏ này") → điểm thấp | Baseline 45,5%, khớp paper Unicom 45,4% [Đã chạy + Nguồn] |
| Ảnh đầy đủ thay vì crop | Thử rồi: crop + pad màu trung bình làm R@1 val giảm 17,6 điểm; pad trắng chỉ giảm 5,5 | Phần −5,5 còn lại chưa rõ nguyên nhân; đừng nói chắc "do mất bối cảnh" | `reports/val_crop_ablation.json` [Đã chạy] |
| Adapter, CLIP đóng băng | Rẻ (vài trăm nghìn tham số), train nhanh trên vector đã lưu; học được cơ chế của bài CLIP-Adapter | **Trần thấp:** adapter chỉ sắp lại thông tin CLIP đã thấy, không tạo ra chi tiết họa tiết mới. Backbone đóng băng tốt nhất từng thấy ~72,8%, fine-tune 86–95% [Nguồn] | So R@1 adapter với baseline trên cùng test |
| Train trên vector đã lưu (cache) | Nhanh, chạy được trên máy yếu | Mỗi ảnh 1 vector cố định → **augmentation không có tác dụng** → Task 2 tách khỏi Task 1 | Cách sửa ở mục 4: cache sẵn vài bản augmented |
| Chọn cấu hình trên validation | Không được chọn trên test (đề cấm) | Val chỉ có 400 món nên dễ hơn test nhiều (71% vs 45%) → so **giữa các cấu hình trên val**, không so val với test | Thứ tự các cấu hình trên val có giữ trên test không |

### Pre-mortem: nếu cuối môn bị điểm kém, lý do có thể là

1. **Task 2 không nối với Task 1** — đề yêu cầu cấu hình Task 2 dựa trên nghiên cứu xử lý ảnh ở Task 1. Đây là rủi ro lớn nhất [Phán đoán từ đề bài].
2. **Mất thời gian định hướng, thiếu thí nghiệm** — đến 2026-11-14 chưa có code train, Task 3 chưa làm [Đã kiểm tra repo].
3. **Nói quá kết quả** — gọi adapter là "phương pháp mới", hoặc so val 71% với test 45%. Thầy hỏi một câu là lộ.

## 4. Kế hoạch đủ dùng (không cầu toàn)

| Mã | Cấu hình | Nối với Task 1 | Trạng thái |
|---|---|---|---|
| E0 | CLIP đóng băng, ảnh đầy đủ | C0 | **Xong** — R@1 45,5% |
| E1 | + adapter, ảnh đầy đủ, không augmentation | C0 | **Xong** — test R@1 71,4% |
| E2 | + adapter, **thêm vài bản augmented** mỗi ảnh train (lật, crop ngẫu nhiên, đổi màu) được encode sẵn | augmentation Task 1 | **Xong** — hòa với E1 (71,1%) |
| E3 | + adapter, train trên ảnh **crop + pad trắng** (A1) | crop Task 1 | **Xong** — nhỉnh nhất, test R@1 72,2% |

- **Loss đề xuất: ArcFace** — coi mỗi item_id train là một lớp, học xong thì bỏ lớp phân loại, chỉ giữ vector. Dễ viết, không cần sampler đặc biệt; hạng 5 cuộc thi Google Universal Image Embedding 2022 dùng đúng CLIP đóng băng + head + ArcFace [Nguồn]. Đổi sang SupCon nếu ArcFace không hội tụ.
- **Adapter 512→128→512, tỉ lệ residual 0,2** — lấy từ CLIP-Adapter, chưa tối ưu cho bài này; thử thêm 1–2 giá trị tỉ lệ trên val là đủ.
- **Mở rộng chỉ khi thừa thời gian:** fine-tune vài block cuối của CLIP. Không bắt buộc.

**Đổi kế hoạch nếu:**
- E1 tăng **dưới ~5 điểm** R@1 trên val → kiểm tra lỗi code trước (loss có giảm không, adapter có đang ở tỉ lệ 0 không), rồi mới kết luận "adapter không giúp" — kết luận đó vẫn là kết quả hợp lệ để viết báo cáo.
- Deadline gần → bỏ E3, giữ E0–E2 + Task 3.

## 5. Phân công

| Người | Phần | Đầu ra cụ thể |
|---|---|---|
| **Trung** | Adapter + training (E1–E3), tích hợp, phần mô hình trong báo cáo | Code adapter, bảng kết quả, giải thích kiến trúc/loss |
| **Phong** | **Task 3 trọn gói** (20đ, độc lập): hiệu chỉnh camera bằng ảnh bàn cờ, lọc ảnh, FFT, gradient, keypoint; optical flow hoặc stereo | Notebook OpenCV + hình + giải thích ngắn |
| **Phong** | Phân tích lỗi: xem ~20 query sai của baseline trên val, xếp vào nhóm (cùng màu khác món, góc nghiêng, họa tiết nhỏ…) | Bảng nhóm lỗi + ảnh ví dụ; sau này so với adapter |
| Cả hai | Demo query → Top-5, báo cáo, đọc chéo | Notebook demo, PDF |

Giao cho Phong Task 3 trước vì nó **không phụ thuộc** phần của Trung và dễ có điểm.

## 6. Câu hỏi có thể bị hỏi (từ thầy hoặc Phong)

**"Sao không phân loại luôn mà phải tìm kiếm?"**
Món ở test chưa từng xuất hiện khi train. Bộ phân loại chỉ biết các món đã thấy; tìm kiếm theo độ giống thì áp dụng được cho món mới.

**"Sao CLIP chỉ được 45%?"**
CLIP học để hiểu "đây là váy đỏ", không học để phân biệt hai chiếc váy đỏ gần giống nhau. Kho test có ~12.600 ảnh của gần 4.000 món, nhiều món rất giống nhau. Paper Unicom báo cùng con số 45,4%.

**"Validation 71% mà test 45%, overfit à?"**
Không phải. Val chỉ có 400 món nên ít đối thủ gây nhầm hơn ~9 lần. Chỉ dùng val để so các cấu hình với nhau.

**"Adapter là gì, khác fine-tune thế nào?"**
Adapter là mạng nhỏ đặt sau CLIP, chỉ nó được học; CLIP giữ nguyên. Kết quả = 0,8 × vector gốc + 0,2 × vector đã qua adapter, nên không làm hỏng kiến thức sẵn có. Fine-tune thì sửa chính CLIP — mạnh hơn nhưng tốn GPU và dễ làm hỏng đặc trưng gốc.

**"Đây có phải phương pháp của nhóm không?"**
Không phải phương pháp mới. Nhóm **tái sử dụng** CLIP-Adapter (gốc dùng cho phân loại) và **điều chỉnh** sang tìm kiếm với loss metric learning, rồi đo xem có lợi trên In-shop không. Phần của nhóm: pipeline dữ liệu, các thí nghiệm crop/augmentation, chuyển adapter sang retrieval, đánh giá và phân tích lỗi.

**"Sao crop lại làm tệ hơn?"**
Phần lớn do màu nền pad: pad màu trung bình CLIP giảm 17,6 điểm, pad trắng chỉ giảm 5,5. Ảnh In-shop vốn nền studio sáng, nên ảnh đầy đủ đã sạch. Phần giảm còn lại chưa tách được nguyên nhân.

**"Nếu adapter không giúp thì sao?"**
Vẫn là kết quả: cho thấy giới hạn nằm ở đặc trưng của CLIP đóng băng, muốn tăng nhiều phải fine-tune (paper đạt 86–95%). Báo cáo nói rõ điều đó thay vì giấu.

**"Có rò rỉ dữ liệu không?"**
Train và test khác item_id; không có ảnh trùng byte/pixel giữa các tập; test chỉ chạy một lần cuối [Đã chạy, `reports/audit.json`].

## 7. Chưa biết

- Deadline nộp bài (ô "Due date" trong đề để trống; kế hoạch giả định 2–3 tuần kể từ 14/11).
- ~~GPU máy nhà~~ — đã biết: RTX 3050 Ti Laptop 4 GB; đủ cho adapter, chật nếu fine-tune toàn bộ.
- ~~Adapter tăng bao nhiêu~~ — đã biết: +26 điểm R@1 trên test (45,5 → 71,4), xem [T1_ADAPTER_GIAI_THICH.md](T1_ADAPTER_GIAI_THICH.md) mục 5d.
