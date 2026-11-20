# Kịch bản show cho thầy — 15 phút

Viết 2026-11-20. Dùng cùng [PIPELINE_TUNG_BUOC.md](PIPELINE_TUNG_BUOC.md) (sơ đồ) và [HIEU_VA_PHAN_BIEN.md](HIEU_VA_PHAN_BIEN.md) (câu hỏi vặn).
Mục đích: lên nói được **mình làm gì, vì sao làm thế, kết quả bao nhiêu, sai ở đâu** — không phải đọc lại notebook.

---

## 0. Đọc cái này trước: vì sao nó "không giống việc mình tự viết tay"

Đồ án dùng foundation model thì 80% code là keo dán: đọc dữ liệu, cắt ảnh, gọi CLIP, lưu vector, tính Recall. Nhìn vào file thì đúng là không thấy "thuật toán của mình" đâu cả.

Nhưng thầy không chấm "ai gõ từng dòng". Thầy chấm ba thứ:

1. **Hiểu đề** — biết đây là retrieval chứ không phải classification, và biết vì sao.
2. **Tự ra quyết định** — chọn gì, bỏ gì, và có đo lại không.
3. **Trung thực** — con số nào chắc, con số nào không, chỗ nào chưa biết.

Cái làm nên "việc của mình" trong dự án này **không phải** đoạn code CLIP, mà là: chia split thế nào, thử crop rồi phát hiện nó **hại**, làm E2 và **giữ lại kết quả thất bại**, khoá test đến cuối, và ngồi bóc 3.973 lỗi ra 5 nhóm. Đó là những thứ một người hiểu việc mới làm được — và đó là thứ phải lên show.

**Luật khi nói:** nói "bọn em tái sử dụng CLIP + CLIP-Adapter + ArcFace" trước, rồi mới nói "phần bọn em làm là...". Khai phần mượn trước sẽ khiến phần tự làm đáng tin hơn, không phải kém đi.

---

## 1. Câu mở 30 giây (thuộc lòng)

> "Đề cho một ảnh quần áo, tìm trong kho những ảnh **cùng đúng mặt hàng** đó. Bọn em không phân loại, vì mặt hàng ở tập test chưa từng xuất hiện khi train — chỉ có thể so độ giống.
> Cách làm: CLIP biến ảnh thành vector 512 số, thêm một adapter nhỏ 131 nghìn tham số học cách uốn vector cho hợp quần áo, rồi so cosine.
> Kết quả trên test: **Recall@1 từ 45,5% lên 72,2%**, Recall@5 là 89,2%. Toàn bộ test 14.218 câu hỏi, thầy chỉ chạy một lần."

Ba con số phải nhớ: **45,5 → 72,2** (R@1) · **89,2** (R@5) · **131 nghìn** (tham số adapter).

---

## 2. Bộ slide tối thiểu — 8 slide, mỗi slide một hình

Không cần đẹp. Mỗi slide một hình + một câu. Thứ tự này đi từ "đề bài" đến "chỗ sai", đúng mạch thầy muốn nghe.

| # | Slide | Hình / thứ mở | Phút |
|---|---|---|---|
| 1 | Bài toán: 1 ảnh query → kho 12.612 ảnh → Top-K | sơ đồ 1 trong `docs/PIPELINE_TUNG_BUOC.md` | 1,5 |
| 2 | Chia dữ liệu: vì sao test đáng tin | sơ đồ 2 | 2 |
| 3 | Task 1: một ảnh đi qua những gì | sơ đồ 3 + `docs/images/dataset/crop-comparison.png` | 2 |
| 4 | Adapter nằm ở đâu, học gì | sơ đồ 4 | 2 |
| 5 | Bốn cấu hình E0–E3, mỗi bước đổi một thứ | sơ đồ 6 + bảng kết quả | 2 |
| 6 | **Demo live** | notebook phần 7: ô Demo, rồi ô Ảnh bất kỳ với ảnh tự chụp | 3 |
| 7 | Lỗi: 3.973 câu sai được chia thế nào | `docs/images/report/test_error_analysis.png` | 2,5 |
| 8 | Kết luận: làm được gì, chưa làm được gì | 3 dòng chữ | 1 |

Tổng 15 phút. Nếu bị cắt còn 8 phút: bỏ slide 3 và 5, giữ 1-2-4-6-7-8.

---

## 3. Lời thoại từng đoạn

Mỗi đoạn là câu nói thật, không phải gạch đầu dòng. Đọc lên miệng 2 lần rồi nói theo ý, đừng học nguyên văn.

### 3.1 Bài toán (slide 1)

> "Đầu vào là một ảnh chụp sản phẩm. Đầu ra là danh sách ảnh trong kho được xếp từ giống nhất đến ít giống nhất.
> Điểm mấu chốt: **đây là tìm kiếm, không phải phân loại**. Mỗi mặt hàng trong DeepFashion có một `item_id`, và tập test có 3.985 mặt hàng **không hề có** trong train. Nên không thể train một bộ phân loại rồi đoán tên món — phải so độ giống.
> Đo bằng Recall@1: ảnh xếp đầu tiên có đúng mặt hàng của query không. Đúng nghĩa là **cùng `item_id`**, không cần trùng đúng bức ảnh — vì một món có nhiều ảnh ở nhiều góc."

Nếu thầy hỏi "sao không dùng caption/text": trả lời ngay ở đây — "đề là tìm bằng ảnh, nên bọn em giữ đúng một đầu vào là ảnh để so sánh công bằng với baseline; thêm text là hướng mở rộng, bọn em ghi trong phần giới hạn."

### 3.2 Dữ liệu — chỗ ăn điểm nhất (slide 2)

> "Dữ liệu 52.712 ảnh, 7.982 món. Bọn em chia **theo món, không theo ảnh**. Nếu chia theo ảnh thì adapter đã nhìn thấy món đó lúc train, điểm validation sẽ ảo.
> Cụ thể: 3.997 món train, trong đó 3.597 món để dạy adapter và 400 món làm validation. Validation có 1.232 câu hỏi và 1.384 ảnh gallery — cũng là **món lạ** như test, chỉ ít món hơn.
> Tập test 3.985 món còn lại thì **khoá**, chỉ mở đúng một lần ở cuối. Mọi thứ — số epoch, tỉ lệ residual, chọn cấu hình — đều chọn trên validation.
> Có script kiểm tra tự động: món train giao món val phải rỗng, train giao test phải rỗng, không ảnh nào vừa là query vừa là gallery. Vi phạm là script dừng."

Câu chốt cho đoạn này: **"Con số 72,2% là con số duy nhất chưa từng bị dùng để chọn gì cả."**

### 3.3 Task 1 — và chỗ mình sai (slide 3)

> "Task 1 là xử lý ảnh. Có hai đường: giữ nguyên ảnh, hoặc cắt theo bounding box rồi pad cho vuông.
> Bọn em tưởng cắt sẽ tốt hơn — bỏ được nền, chỉ giữ quần áo. Đo trên validation thì **ngược lại**: CLIP gốc từ 71,1% tụt còn 65,8% khi pad trắng, và 53,6% khi pad bằng màu trung bình.
> Truy ra được phần lớn là do **màu nền pad**: ảnh In-shop vốn đã là nền studio sáng, cắt ra rồi bôi màu trung bình vào là tự tạo nhiễu. Phần giảm 5,4 điểm còn lại thì bọn em **chưa tách được nguyên nhân** — không dám nói chắc là do mất bối cảnh.
> Cũng trong Task 1: biểu diễn tay bằng HSV histogram chỉ đạt 33,4%, nên bọn em bỏ hướng đó. Con số này còn là **mốc dưới** vì histogram tính cả pixel nền pad — bọn em ghi rõ trong báo cáo."

Đây là chỗ để lộ rằng mình có đo và dám nói "chưa biết". Đừng bịa nguyên nhân.

### 3.4 Adapter (slide 4)

> "CLIP ViT-B/32 đóng băng, không sửa một tham số nào. Sau nó là một adapter nhỏ: 512 → 128 → 512, khoảng 131 nghìn tham số.
> Đầu ra không thay CLIP mà **cộng dư**: `z = 0,8 × vector CLIP + 0,2 × vector adapter`. Nghĩa là luôn giữ 80% kiến thức gốc. Nếu adapter hỏng hoàn toàn, kết quả tụt về đúng mức CLIP gốc chứ không sập — bọn em có unit test cho trường hợp tỉ lệ bằng 0.
> Lúc train thì thêm ArcFace: coi 3.597 món train là 3.597 lớp, kéo vector về tâm đúng món và đẩy ra khỏi tâm món khác. Train xong thì **vứt lớp ArcFace**, chỉ giữ adapter.
> Vì CLIP đóng băng nên vector của mỗi ảnh chỉ tính một lần rồi lưu. 40 epoch train adapter mất khoảng 45 giây trên laptop."

Nếu thầy hỏi "khác fine-tune thế nào": "fine-tune là sửa chính CLIP — mạnh hơn nhưng tốn GPU và dễ phá đặc trưng gốc; adapter chỉ sắp xếp lại thứ CLIP đã nhìn thấy. Đó cũng là **trần** của phương pháp này."

### 3.5 Kết quả — nói đúng mức (slide 5)

> "Bốn cấu hình, mỗi bước đổi đúng một thứ.
> E0 là CLIP gốc, ảnh đầy đủ: **45,5%**.
> E1 thêm adapter: **71,4%** — tăng 26 điểm, đây là phần lớn kết quả.
> E2 thêm ảnh augmented: **71,1%** — **không giúp gì**. Lý do: CLIP đóng băng gần như cho ra cùng một vector cho ảnh gốc và ảnh lật/cắt nhẹ, cosine khoảng 0,93, nên adapter không có thêm thông tin. Bọn em giữ nguyên kết quả thất bại này trong báo cáo.
> E3 quay lại dùng ảnh crop + pad trắng, lần này **có adapter**: **72,2%**. Crop từ chỗ hại CLIP gốc lại thành nhỉnh nhất.
> Nhưng phải nói rõ: E3 hơn E1 có **0,8 điểm**, qua 3 seed dao động 72,04–72,51. Nên câu đúng là 'nhỉnh hơn và ổn định', **không** phải 'tốt hơn rõ rệt'."

Con số phụ nếu thầy muốn: R@5 E0 67,2 → E3 89,2; MRR 0,555 → 0,798.

### 3.6 Lỗi — đoạn chứng minh mình hiểu (slide 7)

> "E3 sai ở vị trí số 1 với 3.973 câu trên 14.218. Bọn em chia số lỗi đó thành 5 nhóm, mỗi câu chỉ vào một nhóm, cộng lại đúng 3.973.
> Kết quả: **63,4% lỗi có ảnh xếp đầu tiên vẫn cùng nhóm hàng** — tức là nhầm giữa hai cái áo rất giống nhau, không phải nhầm áo với quần.
> Và **61% trường hợp món đúng vẫn nằm trong top 2–5**. Chênh lệch cosine giữa cái sai và cái đúng chỉ 0,028 khi món đúng ở hạng 2–5. Nên R@1 phải đọc là 'đúng ngay vị trí đầu', không phải 'tìm được hay không' — R@5 là 89,2%.
> Nhóm khó nhất là áo thun in hình: 18,9% lên 46,0%. Giả thuyết là hình in nhỏ, ở 224×224 gần như không đọc được, còn màu và dáng thì giống hệt nhau.
> Có cả lỗi do **nhãn**: một món có 8 ảnh nhưng gồm 2 màu, query màu navy mà ảnh đúng chỉ có màu trắng — nhìn ảnh thì không thể phân biệt, nhưng chấm theo `item_id` vẫn tính là sai.
> Cuối cùng: adapter sửa được 4.573 câu và làm hỏng 798 câu — tỉ lệ gần 6 sửa 1 hỏng. Không phải miễn phí, và bọn em nói cả phần hỏng."

### 3.7 Kết (slide 8)

> "Tóm lại: adapter nhỏ, đóng băng CLIP, tăng 26 điểm Recall@1, và bọn em biết chính xác nó tăng nhờ đâu và hỏng ở đâu.
> Giới hạn: vẫn kém fine-tune 15–20 điểm; validation 90% không đại diện cho test 72% vì ít món hơn; E3 cần bounding box nên ảnh ngoài dataset phải có bộ phát hiện đối tượng.
> Hướng tiếp theo: fine-tune vài block cuối của CLIP, hoặc xếp hạng lại top-10."

---

## 4. Demo live — bấm đúng thứ tự

### Trước giờ 15 phút (làm một lần, đừng để lên mới làm)

Mở `FCVG101.ipynb` bằng **VS Code** (extension Jupyter) hoặc PyCharm, chọn kernel `.venv` — trong venv không cài JupyterLab nên đừng gõ `jupyter lab`.

Rồi chạy sẵn **toàn bộ phần 1–6** (Run All đến hết ô Test, ~2 phút nhờ cache vector và file adapter đã lưu `cache/adapter/nb_runs.pt`); lần đầu không có file này thì train mất ~12 phút. Chạy trước để lúc show không phải chờ.

**Lưới an toàn (nên làm):** bản HTML của notebook đã chạy, mở bằng trình duyệt, không cần Python:
`artifacts\main\FCVG101.executed.html`. Nếu máy chiếu chỉ cho mở file, dùng cái này thay notebook.

Chuẩn bị **một ảnh tự chụp** (áo/quần/mũ của bạn), để ở đường dẫn không dấu, ví dụ `E:\Downloads\ao_cua_toi.jpg`. Không cần ảnh đẹp — ảnh hơi xấu lại càng thật.

Mở sẵn 4 tab theo thứ tự: notebook → `docs/images/report/test_error_analysis.png` → `docs/BAO_CAO_NHAP.md` mục 7 → terminal.

### Khi lên show

1. **Ô Demo (phần 7)** — chạy, ra 3 query ngẫu nhiên (seed 2026), mỗi query 2 hàng: trên E0, dưới E3.
   Nói: *"Hàng trên là CLIP gốc, hàng dưới là adapter. Chữ xanh là đúng mặt hàng. Mấy query này chọn ngẫu nhiên có seed, không phải chọn cho đẹp — có câu cả hai cùng đúng, có câu cả hai cùng sai."*
   Nếu muốn thầy thấy đúng chỗ hay: đổi seed thành một số khác, chạy lại, nói *"đổi seed là ra bộ khác"*. Đổi seed **không** phải chọn kết quả, vì mình không chọn câu nào để báo cáo.

2. **Ô Ảnh bất kỳ** — sửa `MY_IMAGE = r"E:\Downloads\ao_cua_toi.jpg"`, chạy.
   Nói: *"Đây là ảnh con chụp bằng điện thoại, không có trong dataset. Hệ thống không biết bounding box nên E3 chỉ pad trắng, không cắt — ô này in ra nó đã làm gì."*
   Đây là khoảnh khắc thuyết phục nhất: ảnh của mình, không phải ảnh mẫu.

   **Nếu kết quả dở** (khá thường gặp): đừng chữa cháy, nói luôn — *"Ảnh điện thoại khác miền với ảnh studio: nền lộn xộn, ánh sáng khác, không có bbox. Đây đúng là giới hạn bọn em ghi trong báo cáo: muốn dùng ngoài dataset phải thêm bộ phát hiện sản phẩm."* Nói được câu này còn tốt hơn một demo đẹp.

3. **Ô Phân tích lỗi** — chạy, chỉ vào 4 con số theo đúng thứ tự in ra (5.679 / 4.531 / 791 / 3.217) và dòng "61% món đúng nằm ở #2–5".

4. **Bảng 5 nhóm lỗi trong báo cáo** (mục 7) — không cần chạy. Rồi chuyển sang `test_error_analysis.png` để nói phần 3.6.

### Nếu demo chết (mạng, GPU, kernel)

Không sửa tại chỗ. Mở `artifacts/main/FCVG101.executed.html` trong trình duyệt — bản đã chạy, hình y hệt, không cần Python. Rồi mở `docs/images/dataset/error-groups.png` để nói phần lỗi.

**Đừng bao giờ** ngồi debug trước mặt thầy quá 60 giây.

---

## 5. Bảng tra: thầy hỏi gì → mở đâu → trả lời một câu

| Câu hỏi | Mở | Trả lời gọn |
|---|---|---|
| Sao không phân loại? | sơ đồ 2 | Món test chưa từng có trong train; bộ phân loại chỉ biết món đã thấy. |
| Sao CLIP chỉ 45%? | slide 1 | CLIP giỏi "váy đỏ", không giỏi "đúng chiếc váy đỏ này"; kho có ~4.000 món rất giống nhau. Paper Unicom báo cùng 45,4%. |
| Val 90% mà test 72%, overfit? | sơ đồ 2 | Không. Val chỉ 400 món nên ít đối thủ gây nhầm hơn ~10 lần. Val chỉ để **so cấu hình**. |
| Adapter khác fine-tune? | sơ đồ 4 | Adapter là mạng nhỏ sau CLIP, chỉ nó học; fine-tune sửa chính CLIP. |
| Đây có phải phương pháp mới? | mục 9 báo cáo | Không. Tái sử dụng CLIP-Adapter + ArcFace, phần nhóm làm là pipeline, 4 thí nghiệm, chuyển sang retrieval, đánh giá và phân tích lỗi. |
| Sao crop lúc hại lúc lợi? | sơ đồ 3 + 6 | Pad màu trung bình phá CLIP gốc (−17,5 val). Có adapter thì crop+pad trắng nhỉnh 0,8 điểm — nhỏ, chỉ nói "nhỉnh". |
| Có rò rỉ test không? | `reports/audit.json` | Train ∩ test = ∅; gallery đủ 12.612 ảnh; test chạy một lần; E0 tái lập đúng 45,51%. |
| 72,2% là cao hay thấp? | bảng E0–E3 | So với CLIP gốc là +26,7 điểm. So với fine-tune (86–95%) thì còn kém — đó là trần của backbone đóng băng. |
| E2 thất bại, sao vẫn làm? | sơ đồ 6 | Vì đề yêu cầu nối Task 1 với Task 2; kết quả âm là kết quả, và nó giải thích luôn vì sao cache vector làm augmentation mất tác dụng. |
| Adapter làm hỏng bao nhiêu? | ô Phân tích lỗi | 791 câu, nhưng sửa 4.531; 79% ca hỏng vẫn còn món đúng ở #2–5. |

---

## 6. Nếu bị hỏi thẳng: "Cái này mày tự làm hay AI làm?"

Đừng chống đỡ, đừng nhận vơ. Chia ba tầng, nói trong 40 giây:

> "Ba tầng ạ.
> **Tầng em dùng sẵn:** CLIP, cấu trúc CLIP-Adapter, loss ArcFace, dataset DeepFashion. Em ghi rõ trong báo cáo mục đóng góp.
> **Tầng em làm:** pipeline dữ liệu và chia validation theo món, bốn cấu hình E0–E3, vòng chọn trên validation rồi khoá test, script đánh giá, demo, và toàn bộ phân tích lỗi.
> **Tầng em ra quyết định ngược:** em tưởng crop tốt hơn, đo ra thì tệ hơn, nên em bỏ. Em làm E2, nó không giúp, em giữ kết quả đó và giải thích vì sao. Em tưởng kết quả 72% là bất thường nên chạy lại E0 và kiểm tra trùng món trước khi tin.
> Em có dùng AI như công cụ viết code và chạy thí nghiệm. Nhưng thầy hỏi bất kỳ con số nào trong báo cáo, em mở được file sinh ra nó và giải thích được vì sao nó ra như thế."

Nếu thầy hỏi tiếp "vậy AI viết hết code à": **"Phần lớn code là do AI viết theo yêu cầu của em, giống như dùng thư viện. Em chịu trách nhiệm phần thiết kế thí nghiệm, kiểm tra số và giải thích."** Trung thực ở đây an toàn hơn nhiều so với bị bắt bài — và đúng là như vậy.

Bằng chứng nếu cần mở: `git log --oneline` (25 commit theo ngày), `docs/KE_HOACH.md` (kế hoạch có tick + ghi chú đổi kế hoạch), `reports/val_crop_ablation.json` (lần đo crop), `reports/test_results.json` (`configs_fixed_before_test: true`, `selection_on_test: false`), `scripts/analyze_error_groups.py` (chạy lại ra đúng 3.973).

---

## 7. Năm thứ tuyệt đối không nói

1. **"Phương pháp của nhóm em là..."** → nói "bọn em dùng CLIP-Adapter, điều chỉnh sang retrieval".
2. **"Validation 90%"** như thành tích → chỉ nói khi so cấu hình, kèm câu "val dễ hơn test".
3. **"Augmentation giúp"** → nó hoà, 71,06 vs 71,42.
4. **"E3 tốt hơn hẳn"** → hơn 0,8 điểm, trong khoảng 3 seed. Nói "nhỉnh và ổn định".
5. **"Crop làm mất bối cảnh nên tệ"** → chưa tách được nguyên nhân; phần lớn là màu pad, phần còn lại chưa rõ.

Và một điều không nên làm: **đừng cuộn notebook từ ô đầu**. Thầy sẽ thấy toàn code nạp thư viện và mất kiên nhẫn. Chỉ mở đúng 4 ô ở mục 4.

---

## 8. Bản 3 phút (nếu bị cắt hoặc thầy hỏi nhanh)

> "Đề: tìm ảnh cùng mặt hàng trong kho, mặt hàng test chưa từng thấy khi train.
> Bọn em đóng băng CLIP, thêm adapter 131 nghìn tham số + ArcFace, train trên vector đã cache, chọn cấu hình trên validation chia theo món, khoá test đến cuối.
> Test 14.218 câu: Recall@1 45,5% → 72,2%, Recall@5 89,3%. Bước nhảy chủ yếu từ adapter (E1 đã 71,6%); augmentation không giúp; crop đảo chiều khi có adapter nhưng chỉ hơn 0,8 điểm.
> Bọn em bóc 3.973 câu sai: 63% là nhầm giữa hai món gần như giống nhau, 61% món đúng vẫn trong top 5, và có một phần lỗi do nhãn. Giới hạn: còn kém fine-tune 15–20 điểm."

---

## 9. Tập một mình — 3 vòng, mỗi vòng 10 phút

1. **Vòng 1:** nói 3.1 → 3.7, nhìn giấy. Ghi lại chỗ nào bí (thường là chỗ con số).
2. **Vòng 2:** nói lại không nhìn giấy, chỉ nhìn slide. Chỗ nào quên thì mở đúng file đó ra đọc — đó là chỗ cần học.
3. **Vòng 3:** tự hỏi 8 câu ở mục 6 của `HIEU_VA_PHAN_BIEN.md`, trả lời thành tiếng. Câu nào ậm ừ là câu thầy sẽ hỏi.

Thêm một bài tập hiệu quả: nhờ Phong (hoặc bạn nào không biết dự án) hỏi vặn 5 phút. Người không biết sẽ hỏi đúng mấy câu ngây thơ mà mình không chuẩn bị.

---

## 10. Checklist 15 phút trước khi vào

- [ ] Mở được notebook bằng VS Code/PyCharm với kernel `.venv` (thử mở trước, đừng tin là nó nhớ kernel).
- [ ] Phần 1–6 của notebook đã chạy xong, không còn lỗi đỏ.
- [ ] `MY_IMAGE` đã trỏ đúng ảnh tự chụp, **đã chạy thử một lần**.
- [ ] Mở sẵn: `artifacts/main/FCVG101.executed.html`, `docs/images/report/test_error_analysis.png`, `docs/BAO_CAO_NHAP.md` mục 7.
- [ ] Tắt thông báo, tắt tab lạ, phóng to chữ notebook (Ctrl +).
- [ ] In thẻ nhớ ở mục 11 hoặc để trên điện thoại.
- [ ] Nước. Nói 15 phút khô miệng hơn tưởng.

---

## 11. Thẻ nhớ — in ra hoặc để trên điện thoại

```text
BÀI TOÁN   retrieval, không classification — item test chưa từng thấy khi train
DỮ LIỆU    52.712 ảnh / 7.982 món · train 3.997 món · val 400 món · test 3.985 món (KHOÁ)
           chia THEO MÓN, không theo ảnh → val cũng là món lạ
MÔ HÌNH     CLIP ViT-B/32 đóng băng + adapter 512→128→512 (~131k) · z = 0,8·f + 0,2·A(f)
           ArcFace lúc train, bỏ sau khi train · 40 epoch ≈ 45 giây
KẾT QUẢ    E0 45,5 → E1 71,6 → E2 71,4 (hoà) → E3 72,2   |   R@5 89,2
           E3 hơn E1 0,8 điểm — nói "nhỉnh", không nói "tốt hơn hẳn"
LỖI        3.973 sai #1 · 63,4% top-1 cùng nhóm hàng · 61% món đúng ở #2–5
           sửa 4.573 / làm hỏng 798 · khó nhất: Graphic_Tees 18,9 → 46,0
GIỚI HẠN   kém fine-tune 15–20 điểm · val 90% ≠ test 72% · E3 cần bbox
MƯỢN        CLIP · CLIP-Adapter · ArcFace · DeepFashion
TỰ LÀM      split · 4 cấu hình · chọn trên val + khoá test · đánh giá · lỗi · demo
CÂU CHỐT    72,2% là con số duy nhất chưa từng dùng để chọn gì
```

---

## 12. Việc còn thiếu, nếu thầy hỏi "còn gì nữa không"

Nói thật, đừng giấu:

- **Task 3B** (optical flow) đang dùng khung hình tổng hợp, kế hoạch là quay video thật — chưa quay.
- **Task 1 đối chiếu đề** từng gạch đầu dòng chưa tick xong (P6 trong `KE_HOACH.md`).
- **Báo cáo PDF** chưa xuất; bản nháp đã đủ mục.
- **Tập trình bày** mới ở vòng 1.

Câu nói: *"Phần chạy và số liệu xong; còn Task 3B cần quay video thật, và bọn em đang đối chiếu lại từng yêu cầu của đề."* Có việc còn lại và biết rõ nó là gì — đó cũng là dấu hiệu của người làm thật.

---

## 13. Phụ lục: bốn khái niệm, giải thích bằng lời thường

Dùng khi thầy hỏi sâu hơn "nó là gì". Không cần đọc thuộc, chỉ cần hiểu để trả lời không ấp úng.

**1. "Vector 512 số" là gì?**
Là toạ độ của ảnh trong một không gian 512 chiều. Mỗi chiều không có tên gọi ("chiều 37 = độ giống váy" là sai), nhưng ảnh giống nhau thì toạ độ gần nhau. Không đọc được từng số, chỉ dùng được khoảng cách giữa các bộ số.

**2. Vì sao dùng cosine mà không dùng khoảng cách thường?**
Cosine đo **góc** giữa hai vector, không quan tâm vector dài hay ngắn. Ảnh chụp gần/ánh sáng mạnh có thể làm vector "dài" ra, nhưng hướng thì vẫn đúng. Vì đã chuẩn hoá độ dài về 1, cosine chỉ còn là tích vô hướng — nên tìm kiếm chỉ là một phép nhân ma trận `G · z` (12.612 × 512), rất nhanh.

**3. ArcFace `s=30, m=0,3` là gì?**
`m` là **margin góc**: trước khi tính softmax, góc giữa vector và tâm lớp đúng bị cộng thêm 0,3 radian (≈17°). Nghĩa là muốn được tính đúng thì ảnh phải nằm **gần tâm lớp hơn mức bình thường** — buộc các món phải tách xa nhau, không chấp nhận "gần đúng". `s=30` chỉ là hệ số phóng to góc thành logit để softmax không bị bẹt. Train xong bỏ toàn bộ lớp này: lúc tìm kiếm chỉ cần vector, không cần biết món nào là lớp nào — nhờ vậy mới tìm được món **chưa từng thấy**.

**4. Recall@K khác accuracy thế nào?**
Accuracy hỏi "đoán đúng nhãn không". Recall@K hỏi "trong K kết quả đầu, có ít nhất một cái đúng không". Ở đây mỗi món có nhiều ảnh, nên chỉ cần **một** ảnh đúng lọt vào top-K là tính đúng — đúng với cách người dùng thật sự dùng: liếc 5 cái đầu. R@1 là thước nghiêm nhất, R@5 là thước thực dụng; chênh giữa hai con số (72,2 vs 89,3) cho biết hệ thống "gần đúng" bao nhiêu lần.

**5. Nếu bị hỏi "sao chọn 0,2 và 40 epoch"?**
Trả lời thật: **lấy từ CLIP-Adapter, không tối ưu cho bài này**. Đã thử tỉ lệ 0,5 trên val: 91,3% so với 90,9% — gần như hoà, nên không đáng tinh chỉnh thêm. Batch 256 và learning rate cũng theo cấu hình gốc. Đây là chỗ em ghi là "chưa tối ưu" trong phần giới hạn, không phải chỗ để nhận là đã làm kỹ.
