# Kết quả feasibility baseline — Visual Product Retrieval

**2026-10-23 · [Đã chạy] Kaggle GPU, full official split · Đề xuất: GO cho đề tài cá nhân FCVG101.**

Dữ liệu truy cập được, baseline chạy nhanh trên GPU miễn phí, phép đo kiểm tra được và có lỗi đủ rõ để nghiên cứu cải thiện. Chưa có cơ sở suy rộng kết quả sang ảnh điện thoại của người dùng.

## Kết quả thực nghiệm

| Phép đo | Kết quả thực tế |
|---|---:|
| Recall@1 | **45,5057%** = 6.470 / 14.218 query |
| Recall@5 | **67,2457%** = 9.561 / 14.218 query |
| Query / gallery | 14.218 / 12.612 ảnh; cùng 3.985 product ID |
| Query bị loại | **0** |
| Số category đánh giá | **23 / 23** |
| Median first-positive rank | 2 |
| Macro theo product ID, R@1 / R@5 | 42,70% / 63,03% |

Một hit yêu cầu đúng **item_id**, không chỉ cùng loại/màu quần áo. Recall@K là tỷ lệ query có ít nhất một ảnh đúng sản phẩm trong K ảnh đầu. Toàn bộ gallery được giữ làm distractor; không có tập con dễ, không lọc gallery theo category hay ID của query.

Có 3.091 query sai top-1 nhưng tìm được trong top-5; 4.657 query chưa tìm được trong top-5. Chi tiết từng query, số positive, hạng positive đầu tiên, top-5 đường dẫn/ID/cosine: [per_query_results.csv](per_query_results.csv). Điểm cosine không phải xác suất đúng.

[metrics.json](metrics.json) là kết quả chính. [Kiểm tra độc lập](independent_verification.json) trên Windows dùng cùng embedding đã tải về nhưng thuật toán khác: đếm số ảnh có similarity cao hơn positive tốt nhất, không dùng hàm sort/retrieval gốc. **Mọi nhãn hit/miss ở K=1 và K=5 đều khớp.** Có 14 query lệch một hạng khi tính float32 bằng runtime/batch khác (kể cả một ca 4↔5), không đổi kết luận ở K=1/5. Đây không phải một lượt trích xuất embedding độc lập lần hai.

## Dữ liệu và chống leakage

[Đã chạy] Đọc ba file thực, đối chiếu [README chính thức](sources/README.inshop.official.txt), giải mã toàn bộ 52.712 ảnh; các ảnh đều 256×256. Archive có 60.820 entry gồm thư mục, trong đó đúng 52.712 JPG, không mã hóa, không chứa README. Cả 52.712 ảnh giải nén khớp CRC với archive. File bbox có đúng 52.712 dòng và cùng tập đường dẫn; schema, loại pose/clothes và tọa độ được kiểm tra. Bbox **không được dùng crop** trong baseline này. [Input inspection](input_inspection.json).

File partition của bạn **khớp từng byte** bản tải trực tiếp từ thư mục phát hành chính thức; [provenance/checksum](sources/partition_provenance.json). SHA-256 archive trong báo cáo là checksum bản bạn cung cấp, không phải checksum do nhà phát hành công bố.

| Split | Ảnh | Product ID |
|---|---:|---:|
| train | 25.882 | 3.997 |
| query | 14.218 | 3.985 |
| gallery | 12.612 | 3.985 |

[Đã chạy] Không trùng đường dẫn query/gallery; train ID tách biệt evaluation ID; mọi query đều có gallery positive. Query/gallery dùng chung product ID là yêu cầu của retrieval. Không thiếu/hỏng ảnh và **không có nhóm trùng byte hoặc decoded pixel trong bất kỳ split nào**. Vì vậy sensitivity khi mask ảnh trùng chính xác bằng kết quả chính, không query nào mất positive. [audit.json](audit.json).

pHash 64 bit, khoảng cách Hamming ≤4, phát hiện 1.357 cặp query/gallery ứng viên gần giống; chỉ 134 cặp cùng ID. Đã xem 12 cặp đầu trong [ảnh rà soát](../artifacts/near_duplicate_review.png): nhiều cặp khác trang phục nhưng giống tư thế/nền sáng, có cả cặp cùng sản phẩm gần như cùng ảnh. Đây là kiểm tra thủ công giới hạn, không phải rà soát toàn bộ 1.357 cặp; không tự loại chúng khỏi benchmark.

## Mô hình và tài nguyên

[Đã chạy] OpenAI CLIP **ViT-B/32**, commit package `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`; checkpoint OpenAI có SHA-256 `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`. Checksum đã được kiểm tra. Model ở eval mode, mọi parameter frozen, dùng inference mode, không fine-tune.

Preprocessing lấy trực tiếp từ `clip.load`: resize bicubic 224, center crop 224, RGB, tensor, normalize mean `(0.48145466, 0.4578275, 0.40821073)`, std `(0.26862954, 0.26130258, 0.27577711)`. Vì ảnh nguồn vuông, bước crop chuẩn này không chứng minh mất vùng ảnh. Image encoder chạy float16 trên CUDA; embedding được chuyển float32 và L2-normalize. Cosine là dot product, tìm kiếm exhaustive theo batch; tie chính xác ưu tiên thứ tự gallery.

| Tài nguyên / bước | Đo được trên Kaggle |
|---|---:|
| GPU | T4×2 được cấp; code dùng **1 GPU** |
| Download checkpoint | 4,40 s |
| Load model, gồm download | 9,81 s |
| Hash/decode 52.712 ảnh | 68,3 s |
| Encode gallery | 27,30 s; 462,0 ảnh/s |
| Encode query | 28,90 s; 492,0 ảnh/s |
| Encode tổng | **56,19 s** |
| Retrieval | **31,51 s** |
| Peak RAM process encode / retrieval | 2.059,8 / 1.506,3 MiB |
| Peak VRAM PyTorch allocated, GPU dùng | 482,7 MiB; không phải toàn bộ CUDA context/reserved memory |
| Cache embedding và manifest | 56.238.261 byte, khoảng 53,6 MiB |
| Lượt chạy thành công v3 đến log cuối | 346,56 s, khoảng **5,78 phút** |

Không provision compute trả phí; dùng quota GPU miễn phí hiện có của tài khoản. Mốc 5,78 phút không gồm upload, chờ cấp tài nguyên hay sửa lỗi setup trước đó; không đo tổng quota bị trừ và không quy đổi lượt dùng trợ lý thành tiền. Hai lượt chuẩn bị trước không đóng góp metrics: v1 chưa gắn dataset đang xử lý; v2 dừng vì dò hàng chục nghìn file trên remote mount quá chậm. Bản cuối giữ nguyên ZIP dưới đuôi `.bin`, giải nén vào scratch disk và dò metadata có giới hạn; đây là thay đổi I/O, không đổi tập ảnh.

Python 3.13.15, torch 2.11.0+cu128, torchvision 0.26.0+cu128, NumPy 2.1.3, Pillow 13.0.0. [Preflight](preflight.json), [timing/model metadata](embed_timing.json), [full environment](environment.freeze.txt), [container/metadata chính xác](kernel-metadata.executed.json), [run events](cloud_run.json). Cảnh báo pip liên quan package có sẵn không dùng trong baseline; toàn bộ **19 test** và pipeline thực đã hoàn tất. Các số throughput trọng số ngẫu nhiên cũ được giữ như lịch sử, không dùng làm số đo này.

## 30 ca đã xem và cơ hội cải thiện

[Trang xem toàn bộ 30 ca](../artifacts/examples/index.html) · [manifest/selection policy](examples_manifest.json). Gồm 10 đúng top-1, 10 sai top-1 nhưng đúng top-5, 10 sai top-5. Mỗi ảnh có query, top-5, ID/điểm cosine, đánh dấu đúng/sai; ca lỗi có thêm positive tốt nhất. Chọn deterministically bằng seed/path hash, category và ID khác nhau trong từng nhóm. Do lấy 10 category đầu theo thứ tự, bộ hình này không đại diện đủ 23 category hay tỷ lệ lỗi; headline metrics vẫn dùng toàn bộ dữ liệu.

Các quan sát từ ảnh, không phải kết luận nhân quả:

- **Sai đối tượng trong ảnh toàn thân:** `fail_top5_01`, ID `id_00007524`, positive hạng 186. CLIP ưu tiên ảnh tank-top/phối đồ tổng thể; ảnh positive chủ yếu thể hiện quần. Cùng ID này ở ảnh tập trung vào quần lại đúng top-1 (`success_top1_01`). Giả thuyết đáng kiểm tra: crop theo bbox giúp tách sản phẩm khỏi người/phụ kiện.
- **Màu/góc gần giống nhưng khác sản phẩm:** `fail_top1_recovered_top5_01`, jeans ID `id_00000353`, positive hạng 4; `fail_top1_recovered_top5_06`, quần ID `id_00000802`, positive hạng 2 dù top-1 cosine 0,9716. Điểm cao không thay thế so khớp product ID.
- **Người mẫu/tư thế có thể chi phối:** `fail_top1_recovered_top5_07` (ID `id_00000396`) và `fail_top1_recovered_top5_10` (ID `id_00005500`) trả về áo khác có người mẫu/góc chụp gần giống. Đây là giả thuyết shortcut cần ablation, chưa chứng minh.
- **Nhận được biến thể cùng ID:** ví dụ `success_top1_02` và `success_top1_07` tìm đúng cả ảnh khác màu. Nhãn item của benchmark không nên bị hiểu thành một SKU màu duy nhất.

[Đã đo] Graphic Tees nữ thấp hơn nhiều category: R@1 18,90%, R@5 32,60% trên 365 query; Skirts nữ đạt 63,19% / 81,43% trên 614 query. Các số này định hướng phân tích lỗi, không đủ chứng minh nguyên nhân hay chọn hyperparameter trên test. Nhóm Suiting nam chỉ 14 query nên không kết luận từ tỷ lệ của riêng nhóm này.

**[Đề xuất, chưa chạy] Bước tiếp theo:** tạo validation từ train identities; so sánh full-image với bbox crop trước. Sau đó mới cân nhắc encoder khác hoặc reranking, cùng protocol và chi phí đo được. Không dùng official test này để dò hyperparameter. Nếu muốn mục tiêu ảnh chụp điện thoại, cần phép đo consumer-to-shop riêng. Không loại trừ được việc CLIP từng thấy một phần DeepFashion khi pretrain.

## Tái lập và nguồn

[README/commands](../README.md), [portable Kaggle/Colab notebook](../notebooks/deepfashion_clip_baseline.ipynb), [notebook Kaggle riêng tư đã chạy](https://www.kaggle.com/code/dangtrung2005/fcvg101-deepfashion-frozen-clip-baseline). Kaggle v3 hoàn tất; fallback Colab mới chuẩn bị, chưa chạy. Dataset, weights, embedding và ảnh minh họa không được đưa vào Git; chúng vẫn có ở local/private Kaggle.

Nguồn chính: [benchmark CUHK](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html), [README trong release chính thức](https://drive.google.com/file/d/0B7EVK8r0v71pSXp5b2R0UUg3U0E/view), [mã CLIP chính thức](https://github.com/openai/CLIP/blob/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6/clip/clip.py). README và partition đã tải trực tiếp từ thư mục Drive được trang CUHK liên kết; source CLIP, loader và output thực được kiểm tra. Các claim kết quả trong báo cáo đến từ lượt chạy, không lấy benchmark của nguồn khác.

Báo cáo preflight từng ghi chưa chạy/thiếu quyền tải đã được thay thế; bản cũ giữ trong [history](history/BAO_CAO_PILOT_PREFLIGHT.md).
