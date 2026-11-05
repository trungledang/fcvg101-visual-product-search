# Làm việc nhóm — FCVG101 Visual Product Retrieval

Dự án do **hai thành viên** cùng thực hiện. Mục tiêu là để cả hai hiểu được dataset, phương pháp và có thể giải thích thí nghiệm; không chỉ ghép code do agent tạo.

## Trước khi sửa code

Cả hai thành viên đọc [`docs/START_HERE.md`](docs/START_HERE.md), cùng mở một `item_id` thật và tự giải thích một kết quả Top-5. Khi không có dữ liệu local thì ghi rõ chưa hoàn thành bước này. Không giao training cho người chưa hiểu query/gallery, nhãn và metric.

## Cách phối hợp

1. Trước mỗi thay đổi, mô tả mục tiêu ngắn gọn: giả thuyết nào, thay đổi biến nào, phép đo nào, dự kiến ảnh hưởng đến đâu.
2. Tạo branch theo công việc, ví dụ `docs/readme-team`, `experiment/linear-head`, `analysis/pose-errors`.
3. Không sửa trực tiếp `main` trong lúc thí nghiệm. Tạo pull request; người còn lại đọc và chạy kiểm thử phù hợp trước khi merge.
4. Khi review, kiểm tra **dữ liệu và nhãn đúng**, code không dùng test để chọn hyperparameters, kết quả chạy thật có artifact/log và tài liệu khớp code.
5. Ghi đóng góp trong lịch sử commits/PR hoặc bảng phân công nhóm; không tự gán công việc cho một thành viên khi chưa thống nhất.

## Checklist cho mỗi thí nghiệm

- [ ] Viết câu hỏi nghiên cứu / giả thuyết có thể sai.
- [ ] Ghi train/validation/test split, số query và số gallery.
- [ ] Ghi checkpoint, preprocessing, loss, optimizer/hyperparameters (nếu training).
- [ ] Giữ cố định các biến không liên quan và có baseline đối chứng.
- [ ] Chạy thử trên validation trước; không lặp lại việc chọn cấu hình từ official test.
- [ ] Lưu kết quả thật (`metrics.json`/CSV), log, seed, thời gian và hardware.
- [ ] Đối chiếu lỗi bằng ảnh thật và `item_id` thay vì chỉ xem cosine score.
- [ ] Viết kết luận đúng phạm vi; nếu cải tiến giảm điểm, vẫn ghi nhận.

## Dữ liệu và bảo mật

Không commit ảnh DeepFashion, checkpoint, cache embeddings, credentials, API tokens hoặc file riêng tư. Dataset có giới hạn sử dụng phi thương mại theo README của nhà phát hành. Chỉ commit mã nguồn, cấu hình, báo cáo tổng hợp và tài liệu được phép phân phối.

## Quy ước tài liệu

- `README.md`: Người mới đọc để biết dự án giải quyết gì, dùng dữ liệu nào và kết quả hiện tại.
- `docs/DATASET_AND_PIPELINE.md`: Giải thích dữ liệu, nhãn và quy trình đo.
- `reports/`: Bằng chứng thí nghiệm, kết quả chạy thật và phân tích.
- Notebook học tập: ưu tiên ngắn, dùng ảnh thật, có đầu vào/đầu ra từng bước.
- Module `vps/`: code thực thi tái sử dụng; notebook không nên sao chép lại nguyên cả pipeline dài.

