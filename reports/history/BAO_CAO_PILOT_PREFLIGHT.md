# Báo cáo pilot khả thi: Visual Product Search (FCVG101)

Ngày: 2026-10-23 · Nhánh: `claude/zealous-euler-g8hncc`

## Kết luận ngắn

**Chưa xác lập được tính khả thi.** Thí nghiệm truy hồi chưa chạy, nên **chưa có Recall@1/Recall@5**,
chưa có kiểm toán bộ dữ liệu thật và chưa có ví dụ thành công/thất bại nào. Môi trường cloud bị chặn mạng,
nhưng máy local truy cập được CUHK và checkpoint CLIP; blocker hiện tại là cần quyền/mật khẩu tải ảnh.

Khuyến nghị có điều kiện: **Go (có điều kiện)** — nếu lấy được dữ liệu hợp lệ thì phần tính toán
đã chạy thử được trên GPU local. Nếu không lấy được DeepFashion In-shop bằng đường hợp lệ nào → **Change Dataset**,
không phải Change Topic. Không có bằng chứng nào hiện tại ủng hộ việc bỏ chủ đề.

### Cập nhật preflight trên máy local (2026-10-23)

Máy Windows 11 truy cập được trang CUHK, Google Drive và tải được checkpoint OpenAI CLIP chính thức.
Checkpoint đã được loader kiểm SHA-256 (`40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`)
và nạp lên RTX 3050 Ti Laptop GPU (4 GB). Môi trường local: Core i7-12700H, 15.68 GB RAM,
Python 3.12, còn 161 GB trên ổ E. Bộ kiểm tra chạy trên Windows: 17/17 pass.

Đo với tháp ảnh ViT-B/32 và **trọng số ngẫu nhiên** trên ảnh tổng hợp: forward 202.55 ảnh/giây ở batch 8
(cấu hình dự kiến chạy) và 444.43 ảnh/giây ở batch 16; peak VRAM ghi nhận 530 MB, peak RSS 1.68 GB.
Đây chỉ là đo chi phí tính toán, không phải kết quả retrieval. Phép chia thô 26.8 nghìn ảnh / 202.55
cho khoảng 2 phút 13 giây forward; thời gian đọc/giải mã ảnh và kiểm toán dữ liệu chưa đo. Chi tiết ở
`reports/benchmark_local.json`; preflight host/máy ở `reports/preflight.json`.

**Vẫn chưa có dataset ảnh.** CUHK ghi rõ In-shop dành cho nghiên cứu phi thương mại và cấm phân phối lại;
file ảnh được mã hóa, cần tải, hoàn tất/ký form thỏa thuận rồi gửi bằng email cơ quan để xin mật khẩu giải nén
([hướng dẫn và điều khoản chính thức](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)). Đến khi có ảnh đã giải nén
vào `data/inshop/`, chưa thể chạy audit split, Recall hay chọn ví dụ.

## 1. Việc bạn cần làm (chính xác)

1. Nếu chưa có quyền, tải và đọc thỏa thuận DeepFashion trên trang chính thức, hoàn tất/ký rồi gửi bằng email cơ quan theo hướng dẫn CUHK. Chia sẻ mật khẩu giải nén với phiên local khi CUHK cấp.
2. Đặt annotation tại `data/inshop/Eval/list_eval_partition.txt` và giải nén `Img/img.zip` vào `data/inshop/img/`.
3. Chạy `scripts/run_all.ps1` trên Windows để audit, embed, evaluate và tạo ví dụ. Không đẩy ảnh dataset lên GitHub.

## 2. Kết quả đo khác

### Snapshot môi trường Claude Cloud ban đầu

Các kết quả cloud bên dưới chỉ mô tả lần chạy Claude Cloud đầu tiên. Chúng không mô tả máy local;
`reports/preflight.json` hiện được cập nhật từ máy local.

### Cloud: truy cập lúc chạy ban đầu (snapshot trong báo cáo Claude)

| Máy chủ | Kết quả |
|---|---|
| mmlab.ie.cuhk.edu.hk (trang chính thức) | bị chặn (proxy 403) |
| drive.google.com / drive.usercontent.google.com (file chính thức) | bị chặn (proxy 403) |
| openaipublic.azureedge.net (checkpoint CLIP chính thức) | bị chặn (proxy 403) |
| huggingface.co, kaggle.com, zenodo, dropbox… | bị chặn |
| pypi.org, github.com | truy cập được |

Do đó **chưa xác minh** được: bản phát hành hiện tại, điều khoản (không tải được trang), file
`list_eval_partition.txt`, số ảnh/số sản phẩm theo split. Không dùng mirror không chính thức: điều khoản
DeepFashion (theo hiểu biết của tôi, chưa kiểm lại được trang) không cho phép phân phối lại ảnh, nên một
mirror công khai không thể coi là nguồn hợp lệ. **Không commit ảnh dataset nào.**

Checkpoint: SHA-256 chính thức của ViT-B/32 lấy từ package `openai/CLIP` (commit `d05afc4`):
`40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`. Loader từ chối mọi file không khớp.

### Cloud: phần cứng / phần mềm

- CPU Intel Xeon 2.10 GHz, **4 nhân**, RAM **15.7 GB**, **không có GPU** (`torch.cuda.is_available() == False`, không có `nvidia-smi`).
- Đĩa trống ~21 GB sau khi cài môi trường (venv 5.7 GB vì wheel torch trên PyPI kèm thư viện CUDA).
- Python 3.13.16, torch 2.14.1, torchvision 0.29.1, numpy 2.5.4, Pillow 13.0.0 (đầy đủ: `requirements.lock.txt`).
- Cài đặt môi trường: 2 phút 19 giây (đo).

### Cloud: chi phí tính toán (`reports/benchmark_cpu.json`)

Đo trên **đúng kiến trúc tháp ảnh ViT-B/32 (87.8 triệu tham số) với trọng số ngẫu nhiên** và ảnh tổng hợp —
FLOPs không phụ thuộc giá trị trọng số nên đây là phép đo chi phí hợp lệ, **không nói gì về chất lượng truy hồi**.

| Đo | Giá trị |
|---|---|
| Forward, batch 64 | 34.8 ảnh/giây |
| Forward, batch 32 | 32.8 ảnh/giây |
| Tiền xử lý CLIP (1 luồng) | 522 ảnh/giây |
| RSS đỉnh | 1.96 GB |

**Ngoại suy (không phải đo):** bài báo gốc công bố tập query+gallery khoảng 26.8 nghìn ảnh (con số
chưa xác minh trên file thật). Với ~34 ảnh/giây → mã hoá toàn bộ ≈ 13 phút trên CPU này; truy hồi chính
xác 14k × 12.6k bằng tích vô hướng cỡ vài giây. ⇒ **Chạy toàn bộ query với toàn bộ gallery là khả thi trên CPU**,
không cần lấy mẫu query, không cần thuê compute. Chi phí tiền: không có hoá đơn riêng; chỉ dùng thời gian phiên
Claude Code trong gói thuê bao (không định giá được bằng đô la).

## 3. Đã sẵn sàng nhưng chưa chạy trên dữ liệu thật

Code (`scripts/pilot.py`, `vps/`) đã cài đặt đầy đủ giao thức PILOT_TASK.md:

- Parser `list_eval_partition.txt`: kiểm số dòng khai báo vs. thực tế, header (từ chối benchmark khác),
  split hợp lệ, đường dẫn trùng/không an toàn (path traversal), ID hợp lệ, thư mục khớp ID.
- Kiểm split: query/gallery không trùng đường dẫn, train không trùng ID với tập đánh giá, đếm query không có positive.
- Kiểm toán ảnh: file thiếu/hỏng, SHA-256 file, SHA-256 điểm ảnh, pHash gần trùng (ngưỡng Hamming 4, chỉ là ứng viên để xem tay).
- CLIP đóng băng: `eval()`, `requires_grad=False`, `inference_mode`, tiền xử lý chuẩn của OpenAI, chuẩn hoá L2.
- Cache embedding kèm manifest (thứ tự ảnh, checksum annotation, checksum model, tiền xử lý) — tự huỷ nếu không khớp.
- Truy hồi chính xác trên **toàn bộ gallery**; Recall@1/5 với mẫu số rõ ràng; thứ hạng positive đầu tiên chính xác;
  đánh giá độ nhạy loại các ảnh gallery trùng byte/điểm ảnh với query, báo số query mất hết positive.
- Lấy mẫu query (chỉ khi cần): tất định, không phụ thuộc kết quả, giữ mọi ID; kết quả tự gắn nhãn PARTIAL.
- Chọn ví dụ tất định theo nhóm (đúng top-1 / sai top-1 nhưng đúng top-5 / sai cả top-5), xoay vòng theo
  category, mỗi ID một lần; hình montage lưu riêng trong `artifacts/` (không commit).

Kiểm thử: **17/17 test pass** (`pytest`) trên dữ liệu giả lập — chỉ chứng minh tính đúng của code
(metric với nhiều positive, không có positive, mask ảnh trùng, top-5, parser, cache, lấy mẫu, chạy end-to-end).
**Đây không phải kết quả thí nghiệm.**

## 4. Chưa làm (phụ thuộc dữ liệu)

- Recall@1/Recall@5, kích thước query/gallery thực, đánh giá độ nhạy trùng lặp.
- Kiểm toán split/trùng lặp trên file thật.
- ≥20 ví dụ thành công/thất bại và phân tích lỗi (chưa xem ảnh nào nên không nêu mẫu hình lỗi nào).
- Thời gian mã hoá/truy hồi thực tế với checkpoint thật.

## 5. Hạn chế đã biết trước

- Không loại trừ được việc CLIP đã thấy ảnh này khi pretrain; chống rò rỉ trong pipeline không chứng minh điều đó.
- In-shop là ảnh studio, nền sạch, người mẫu; kết quả không chuyển trực tiếp sang ảnh chụp điện thoại của người dùng
  (consumer-to-shop: ánh sáng, nền, góc, che khuất khác hẳn).
- Ước tính thời gian ở mục 2 là ngoại suy từ trọng số ngẫu nhiên và số ảnh công bố, chưa đo trên dữ liệu thật.

## 6. Cơ hội học tập (cho dự án một người) — đề xuất, chưa chạy

Khi có baseline: phân tích lỗi theo category; thử crop theo bbox (thí nghiệm riêng, gắn nhãn rõ);
so sánh query-expansion/re-ranking không cần huấn luyện; đánh giá trên một tập consumer-to-shop nhỏ để đo
khoảng cách miền. Mọi lựa chọn phương pháp phải dựa trên tập không phải test.
