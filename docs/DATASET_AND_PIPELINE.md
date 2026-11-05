# Hiểu dữ liệu và pipeline trước khi cải tiến model

## 1. Câu hỏi bài toán

Với **ảnh query Q**, trong tập **gallery G**, hãy xếp hạng những ảnh sao cho ít nhất một ảnh **cùng `item_id`** với Q nằm càng gần đầu danh sách càng tốt. Cùng category hoặc cùng màu nhưng khác `item_id` là **sai** theo ground truth của benchmark.

Dataset In-shop có 52.712 ảnh, thuộc 7.982 IDs. Mỗi ID có thể xuất hiện dưới nhiều góc chụp. Đây là lý do không thể đánh đồng 'sản phẩm có trong gallery' với 'ảnh query được sao chép vào gallery'.

## 2. Mở data lên xem gì?

Trên workspace ban đầu, dữ liệu ở ngoài repo:

```text
workspace-parent/
├── visual-product-search-pilot/  # repo
├── img/                          # ảnh giải nén
├── Eval/list_eval_partition.txt  # nhãn/split
└── list_bbox_inshop.txt          # bbox và pose
```

Trong ảnh: `img/MEN/<category>/<item_id>/<photo>.jpg`, `img/WOMEN/...`.

File `list_eval_partition.txt` có header, mỗi dòng gồm `image_name`, `item_id`, `evaluation_status` (`train`, `query`, `gallery`). Không dùng tên folder hay mã ID để đo similarity; ID chỉ là **ground truth**.

Cách tự xác minh bằng mắt:

1. Chọn 1 `item_id` từ file partition, tìm ít nhất 1 ảnh query và 1 ảnh gallery có cùng ID.
2. Mở cả hai: ảnh có phải cùng một file không? Góc nhìn/pose khác nhau gì?
3. Chọn ảnh khác ID nhưng cùng category. Vì sao chúng có thể trông giống nhau hơn cặp cùng ID?
4. Với 1 query đã chạy, đọc `reports/per_query_results.csv`: xem Top-5 ID, cosine và rank của positive đầu tiên.
5. Đối chiếu ảnh thật trong `artifacts/examples/` ở máy hoặc notebook Kaggle riêng tư. Các artifact ảnh **không có trong Git công khai**.

Một trường hợp được báo cáo: item `id_00007524` có query `01_1_front.jpg` cho first positive rank 1, còn query `01_4_full.jpg` cho rank 186. Đây là hai ảnh **cùng ID nhưng khác query**, minh họa tại sao kết quả có thể thay đổi theo đầu vào. Muốn kết luận nguyên nhân phải kiểm tra ảnh thật.

## 3. Pipeline có hai giai đoạn khác nhau

### Giai đoạn A — Lập chỉ mục gallery (làm trước)

1. Đọc các ảnh thuộc `gallery` từ partition.
2. Resize/crop/normalize bằng preprocessing của CLIP ViT-B/32.
3. Đưa ảnh qua **image encoder** của CLIP, lấy vector 512 chiều.
4. L2-normalize và lưu tất cả gallery embeddings cùng đường dẫn ảnh theo đúng thứ tự.

### Giai đoạn B — Truy vấn

1. Đọc **1 ảnh query**; dùng preprocessing và encoder giống gallery.
2. Lấy 1 embedding 512 chiều và normalize.
3. Tính cosine similarity với **toàn bộ** 12.612 gallery embeddings (đối với official test).
4. Sort điểm giảm dần để lấy Top-1 hoặc Top-5. Đây là ranking trực tiếp, **không có reranker riêng**.
5. Trả về ảnh gallery và similarity; kết quả không phải ảnh do model sinh ra.

Không có decoder trong pipeline này. CLIP cũng có text encoder, nhưng baseline chỉ sử dụng image encoder.

### Giai đoạn C — Đánh giá (chỉ sau khi tìm xong)

Với mỗi query, mở `item_id` của query và Top-K qua partition:

- Đúng Top-1 nếu ảnh đứng đầu có cùng `item_id`.
- Đúng Top-5 nếu **ít nhất một** ảnh trong năm kết quả đầu có cùng `item_id`.
- Recall@K = số query đúng Top-K / tổng query hợp lệ.

Kết quả official test là R@1 **6.470/14.218 = 45,51%**, R@5 **9.561/14.218 = 67,25%**. Nhãn không tham gia tìm kiếm, nên không có chuyện 'nhìn đáp án'.

## 4. Train, validation và official test không được lẫn nhau

- **Official train**: 25.882 ảnh của 3.997 IDs; baseline không dùng để cập nhật trọng số.
- **Validation**: nhóm tách 400 IDs từ official train thành 1.232 query + 1.384 gallery để thử crop, padding và chọn phương pháp. Validation dễ hơn official test vì gallery ít distractors hơn.
- **Official query/gallery**: 14.218/12.612 ảnh của 3.985 IDs không trùng train; dùng để đánh giá cuối.

Điểm **71,10%** của full-image CLIP trên validation **không thể so trực tiếp** với **45,51%** trên official test. Cần so các cấu hình trên cùng tập và cùng quy mô gallery.

## 5. Tại sao ảnh đúng có trong gallery mà Recall@1 thấp?

Vì **top-1 theo cosine similarity** không nhất thiết là **top-1 theo cùng `item_id`**. Hai sản phẩm khác nhau nhưng cùng màu, kiểu hoặc người mẫu có thể có embeddings gần hơn hai góc chụp của một sản phẩm. Khi đó ảnh đúng vẫn ở gallery nhưng nằm hạng 2, 10 hoặc 186. Mục tiêu cải tiến là học representation làm tăng xác suất xếp ảnh **cùng sản phẩm** lên đầu, không đơn thuần tăng similarity mọi ảnh trông tương tự.

## 6. Bằng chứng và giới hạn

- `reports/metrics.json`: R@1/R@5 official test.
- `reports/per_query_results.csv`: ID, cosine, thứ hạng và kết quả của từng query.
- `reports/val_results.json` và `reports/val_crop_ablation.json`: so sánh crop/padding trên validation.
- `vps/annotations.py`, `vps/embed.py`, `vps/metrics.py`: đọc nhãn, tạo embedding và tính retrieval/metrics.
- `reports/BAO_CAO_PILOT.md`: kết quả, môi trường, kiểm tra trùng ảnh và giới hạn của benchmark.

Thí nghiệm crop cho thấy màu padding ảnh hưởng đáng kể. Không thể kết luận mọi crop đều không tốt, cũng chưa có kết quả huấn luyện ArcFace hoặc fine-tuning được ghi nhận trong repo. Kết quả In-shop không chứng minh hiệu quả trên ảnh chụp điện thoại ngoài đời.
