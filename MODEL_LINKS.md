# Chạy lại và tài nguyên

`FCVG101.ipynb` là notebook chính, tự đủ: chạy từ trên xuống là có dữ liệu, tiền xử lý, baseline, train adapter, bảng kết quả, test, demo, phân tích lỗi.

| Thứ cần | Lấy ở đâu |
|---|---|
| DeepFashion In-shop (`img/`, `Eval/list_eval_partition.txt`, `list_bbox_inshop.txt`) | https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html (chỉ dùng nghiên cứu/giáo dục phi thương mại; không phân phối lại) |
| CLIP ViT-B/32 (OpenAI) | notebook tự tải khi chạy lần đầu (`clip.load`, kiểm tra SHA-256) |
| Thư viện | `pip install -r requirements.txt` |
| Adapter đã train (9 file, 4,4 MB) — dùng nếu không muốn train lại | https://drive.google.com/file/d/1XWptbtHuBulXwOthm_7qVEHB8puaAw7c/view (checkpoint của lần chạy bằng script trước, cùng cấu hình; notebook tự train và lưu adapter của riêng nó vào `cache/adapter/nb_runs.pt`) |

## Máy cá nhân
Đặt `img/`, `Eval/`, `list_bbox_inshop.txt` ở thư mục chứa repo (hoặc thư mục cha bất kỳ phía trên; notebook tự tìm), mở `FCVG101.ipynb`, chạy từ trên xuống.

## Colab
1. Nén `img/`, `Eval/`, `list_bbox_inshop.txt` thành `dataset.zip`, upload lên Google Drive, mount Drive và giải nén ra `/content/data`.
2. Ô thứ hai của notebook có dòng gợi ý cài `clip` và dòng đặt `DATA`; bỏ dấu `#` rồi sửa đường dẫn.
3. Chọn GPU, Run all. Lần đầu mã hóa ~52.000 ảnh nên chậm (vài chục phút đến hơn một giờ tùy tốc độ đọc Drive); vector được lưu vào `cache/embeddings/` nên lần sau nhanh.

Chưa kiểm chứng trên Colab thật; đã chạy trọn vẹn trên máy cá nhân (Windows, RTX 3050 Ti 4 GB).

## Thời gian trên máy gốc
Mã hóa ảnh (lần đầu): từ ~2 phút đến hơn một giờ tùy bộ nhớ đệm đĩa · train 9 adapter: ~12 phút · đánh giá test: dưới 1 phút · cả notebook khi chưa có cache adapter: ~15 phút (có cache vector).
