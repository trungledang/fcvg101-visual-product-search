# BẮT ĐẦU TỪ ĐÂY — Hiểu Visual Product Search từ con số 0

**Dành cho cả hai thành viên FCVG101.** Không cần biết CLIP, embedding hay Recall trước khi đọc. Mục tiêu không phải thuộc lòng thuật ngữ mà là **tự kiểm tra được một truy vấn bằng ảnh và nhãn thật**.

> Trạng thái: dự án có baseline frozen CLIP đã chạy trên official test; có thử nghiệm crop/padding/HSV trên validation; **chưa có bằng chứng trong repo về một model ArcFace/fine-tuned đã huấn luyện**. Các ví dụ tên ảnh/ID giả định dưới đây để dạy cách đọc; khi cần bằng chứng cho báo cáo phải sử dụng ảnh, dữ liệu và kết quả thật.

## 1. Hãy quên CLIP đi: bài toán là gì?

Hình dung một shop có **một kho ảnh sản phẩm**. Khách gửi ảnh chiếc áo. Shop muốn đưa ra tối đa 5 ảnh từ kho, ưu tiên những ảnh **thuộc cùng sản phẩm**. Không phải chỉ cùng loại “áo thun” hay cùng màu “xanh”.

- **Đầu vào (input):** ảnh người dùng muốn tra cứu. Trong benchmark nó được gọi là **query image**.
- **Kho tìm kiếm:** ảnh thuộc **gallery** (đã lưu tại kho, không phải tập train).
- **Đầu ra (output):** danh sách các đường dẫn ảnh gallery được xếp hạng, thường xem Top-5, và điểm giống nhau do mô hình tính.
- **Đáp án đúng:** ảnh gallery có cùng `item_id` với ảnh query.

Điểm quan trọng: *cùng sản phẩm* **không** đồng nghĩa *cùng file ảnh*. Một chiếc váy có thể được chụp từ trước, sau, hoặc khi người mẫu thay tư thế. Một chiếc váy khác trông gần giống có thể đánh lừa hệ thống.

**Kiểm tra hiểu bài 1:** Nếu query là ảnh chiếc áo A chụp trước mặt, gallery có ảnh A chụp sau lưng và áo B gần giống A, kết quả nào đúng? Đáp án: ảnh của **A**, bất kể góc chụp; B sai vì khác sản phẩm.

## 2. Mở dữ liệu thật — chúng ta có những gì?

Dataset: **DeepFashion In-shop Clothes Retrieval Benchmark**. Trong pilot, 52.712 ảnh, 7.982 IDs, chia như sau:

| Nhóm | Ảnh | Product IDs | Dùng để làm gì? |
|---|---:|---:|---|
| Train | 25.882 | 3.997 | Huấn luyện sau này, hoặc tách validation từ train |
| Query | 14.218 | 3.985 | Những ảnh dùng để đặt câu hỏi tìm kiếm |
| Gallery | 12.612 | 3.985 | Kho để tìm câu trả lời |

**Query và gallery có chung các ID sản phẩm nhưng không dùng chung đường dẫn ảnh**. Train và tập đánh giá có product IDs khác nhau. Kiểm tra nguồn nhãn ở `Eval/list_eval_partition.txt`; ảnh thực ở `img/`; bbox/pose ở `list_bbox_inshop.txt` nếu có trong bản tải.

Không nhầm: `id_...` **không phải category**. Hai ảnh có category Denim vẫn có thể thuộc hai sản phẩm khác nhau và được tính là sai nếu trả nhầm.

### Thực hành A — xem ảnh trước khi xem model

1. Mở `Eval/list_eval_partition.txt`: đọc dòng tiêu đề và chọn **một `item_id` có ít nhất hai ảnh**. Lọc các dòng có ID đó; nhìn cột `evaluation_status` để biết ảnh nào là query, ảnh nào là gallery.
2. Mở các đường dẫn ảnh trong thư mục `img/`. So sánh góc chụp, tỷ lệ quần áo, người mẫu, nền và vị trí sản phẩm. Ghi **những gì thật sự nhìn thấy**, không tự đoán nguyên nhân.
3. Chọn thêm một ảnh gallery **khác `item_id` nhưng nhìn rất giống**. Hãy giải thích vì sao dù giống thị giác, nó vẫn là kết quả sai theo benchmark.
4. Đối chiếu bảng tổng hợp trong `reports/audit.json` và file nhãn; không chép số mà không biết nguồn.

**Cần nộp trong nhóm:** 1 bảng nhỏ gồm query path, query item ID, gallery positive path, gallery distractor path; 1 ảnh/chụp màn hình cho thấy ba ảnh đó (giữ riêng tư theo điều kiện dataset). Nếu chưa có dataset local, bước này **chưa hoàn thành** — đọc README để biết layout được sử dụng và cách lấy dữ liệu hợp lệ.

## 3. Vì sao có đáp án trong gallery nhưng Recall@1 không phải 100%?

Ví dụ **giả định**: query có `item_id=ID_100`. Gallery có ảnh đúng của ID_100, nhưng có cả ảnh áo ID_200 trông gần như y hệt.

| Rank | Gallery `item_id` | Cosine | Có đúng ID_100? |
|---|---|---:|---|
| 1 | ID_200 | 0,96 | Không |
| 2 | ID_100 | 0,93 | Có |
| 3 | ID_300 | 0,89 | Không |

Mô hình xếp sai ảnh ID_200 ở Top-1, dù ảnh đúng có trong gallery ở vị trí 2. **Recall@1=0 và Recall@5=1 đối với truy vấn này**. Bài toán không phải tìm file trùng byte/pixel; nó đòi hỏi nhận ra **cùng product identity** qua những ảnh khác nhau.

Nếu đưa đúng **cùng một ảnh** vào cả query và gallery, phép so sánh của một encoder xác định thường trả ra score tự khớp rất cao; đó là bài toán khác và có thể là đánh giá bị rò rỉ (*leakage*). Vì vậy phải kiểm tra split.

## 4. Một query được biến đổi ra sao?

**Bước 0 — chuẩn bị kho (một lần):** từng ảnh gallery → tiền xử lý theo chuẩn CLIP (resize/crop/normalize) → image encoder CLIP ViT-B/32 → **embedding 512 số** → chuẩn hóa độ dài vector → lưu lại.

**Bước 1 — lúc tìm:** ảnh query → cùng quy trình tiền xử lý → cùng encoder → **một vector 512 số**.

**Bước 2 — so sánh:** lấy vector query so với **tất cả** vector gallery bằng cosine similarity. Vectơ càng cùng hướng thì cosine càng cao. Đây là điểm xếp hạng, **không phải phần trăm xác suất hai ảnh cùng ID**.

**Bước 3 — ranking:** sắp xếp các điểm giảm dần và lấy Top-5. Baseline đang dùng **exact retrieval** chứ chưa có model reranking, FAISS hay database vector bắt buộc.

**Bước 4 — chấm điểm, sau khi đã tìm xong:** tra `item_id` trong annotation và so với query. Nhãn `item_id` **không đi vào bước tính similarity**. Chỉ lúc evaluation (và training có giám sát trong tương lai) mới dùng nhãn.

**Không có decoder ảnh trong pipeline hiện tại**; không sử dụng text encoder của CLIP để tìm bằng văn bản.

### Thực hành B — đọc một kết quả truy hồi thật

Sử dụng `reports/per_query_results.csv` cùng các ảnh dataset tương ứng (hoặc chạy `notebooks/learn_clip_retrieval.ipynb` nếu đã có data/cache):

1. Chọn một query có `first_positive_rank = 1`; ghi đường dẫn ảnh query và Top-5 cùng IDs/scores.
2. Chọn một query có `first_positive_rank > 5`; quan sát Top-5 ảnh sai và **ít nhất một ảnh gallery đúng ID**.
3. Tự trả lời: tại sao cosine cao nhất lại có thể là ID sai? Tại sao chỉ biết cosine chưa đủ để kết luận đúng?
4. Tự tính Recall@1, Recall@5 của **hai query này** rồi đối chiếu với code `vps/metrics.py`.

**Không kết luận “do background/pose” khi chưa có đối chứng**: nếu thấy có sự khác biệt, ghi đó là *giả thuyết từ quan sát*.

## 5. Recall@K đo chính xác cái gì?

Với một query có ảnh đúng trong gallery, đánh dấu **1** nếu ít nhất một ảnh cùng `item_id` nằm trong K kết quả đầu, ngược lại **0**. Lấy trung bình trên tất cả query hợp lệ.

- **Recall@1 = 45,51%:** 6.470/14.218 query có kết quả hạng đầu đúng ID.
- **Recall@5 = 67,25%:** 9.561/14.218 query có ít nhất một kết quả đúng trong năm ảnh đầu.

Số liệu trên là **official test**, frozen CLIP, **không training với DeepFashion**. Bằng chứng ở `reports/metrics.json`, `reports/independent_verification.json`, và `reports/per_query_results.csv`. Không gọi đây là accuracy 23 loại quần áo.

Đừng nhầm với validation: full-image CLIP đạt **71,10% R@1** trên validation tự tách từ train với **1.384 ảnh gallery và 400 IDs**. Gallery dễ hơn/nhỏ hơn; hai con số này không so trực tiếp để nói mô hình đã cải thiện.

## 6. Chúng ta đã thử gì? Phân biệt dữ liệu với giả thuyết

**Đã thực hiện:** frozen CLIP baseline; cùng validation thử bbox crop và padding, HSV histogram. Full-image validation 71,10% R@1; crop + mean pad 53,57%; crop + white pad 65,75%. Kết luận hợp lệ: *các biến thể crop đã thử không vượt full-image trên validation*. Không được nói rằng tất cả crop đều vô dụng.

**Chưa xác nhận đã làm:** ArcFace, projection head được huấn luyện, partial CLIP fine-tuning. Chúng là **ý tưởng thí nghiệm tương lai**, không phải thành tựu.

**Câu hỏi nghiên cứu đề xuất:** liệu một biểu diễn ảnh được thích nghi cho *item identity* có tìm cùng sản phẩm tốt hơn frozen CLIP, khi train/validation/test và chi phí tính toán được kiểm soát không?

## 7. Hai người học và làm cùng nhau như thế nào?

**Buổi 1 — cùng khám phá dữ liệu:** mỗi người độc lập chọn 2 query thật, tìm một positive và một distractor, so nhãn và giải thích. Sau đó đối chiếu chéo xem có hiểu giống nhau không.

**Buổi 2 — cùng hiểu baseline:** mỗi người thực hành truy hồi 1 query, giải thích 512-d embedding, cosine, Top-5, tính Recall@1/5 bằng tay. Người còn lại hỏi phản biện.

**Buổi 3 — mới chia công việc:** phân chia code/experiments/report dựa trên mục tiêu và lịch nhóm; mọi thay đổi cần review chéo theo `CONTRIBUTING.md`. Không phân công kiểu “một người làm hết model, một người làm hết báo cáo” nếu dẫn đến chỉ một người hiểu kết quả.

### 8 câu tự kiểm tra (không tra đáp án trước)

1. Một `item_id` là sản phẩm hay loại quần áo?
2. Query và gallery khác nhau ở đâu? Có thể chung ID mà khác file ảnh không?
3. Vì sao train IDs phải khác test IDs trong giao thức này?
4. CLIP nhận ảnh và trả ra gì? Có decoder sinh ảnh không?
5. Similarity 0,95 có nghĩa 95% xác suất cùng sản phẩm không?
6. Trong Top-5 có ID đúng ở vị trí 3, Recall@1 và Recall@5 là bao nhiêu?
7. Vì sao 71,10% validation không chứng minh tốt hơn 45,51% official test?
8. Em có thể mở một ảnh query thật và **chỉ tận mắt** ảnh positive, distractor và file nhãn tương ứng không?

**Đáp án ngắn:** 1 sản phẩm; 2 hai nhóm ảnh khác nhau, có; 3 đo khả năng tổng quát tới IDs chưa học; 4 embedding 512 chiều, không; 5 không; 6 R@1=0, R@5=1; 7 gallery/IDs khác quy mô; 8 phải thực hành với ảnh thực, không thể trả lời bằng lý thuyết.

## 8. Nguồn cần mở khi phản biện

- `Eval/list_eval_partition.txt`: nhãn và split.
- `img/`: ảnh nguyên gốc (không commit công khai).
- `vps/annotations.py`, `vps/embed.py`, `vps/metrics.py`: đọc nhãn, tạo embedding, tính ranking/metric.
- `reports/audit.json`, `reports/metrics.json`, `reports/per_query_results.csv`: bằng chứng thực nghiệm.
- `notebooks/learn_clip_retrieval.ipynb`: notebook minh họa đã có trên repo.
- `docs/DATASET_AND_PIPELINE.md`: diễn giải dài hơn về data và pipeline.
- `docs/HIEU_DU_AN.md`: biểu đồ thật về độ tương đồng, lỗi theo góc chụp/category, bẫy nhãn và đóng góp của dự án.

**Tiêu chí hoàn thành phần nhập môn:** mỗi thành viên có thể dùng **một query thật** để tự giải thích cả chuỗi `ảnh → embedding → cosine → ranking → ID matching → Recall`, chỉ ra ít nhất một lỗi thực tế và phân biệt đâu là quan sát, đâu là suy đoán.
