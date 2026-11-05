# Nghiên cứu hướng đi — visual product search: thế giới làm gì, ta nên làm gì

**2026-10-30.** Nguồn: hai lượt tra cứu song song (industry; học thuật/CLIP adaptation), một lượt phản biện độc lập bằng Codex,
và các thí nghiệm validation của dự án. Nhãn: **[Đã chạy]** đo trong dự án · **[Nguồn nói]** có link · **[Suy luận]** lập luận từ bằng chứng.

## Kết luận ngắn

1. **Baseline đúng và không cao:** frozen CLIP ViT-B/32 đạt R@1 45,5% trên test [Đã chạy];
   paper Unicom (ICLR 2023, Bảng 2) báo **45,4%** cho cùng model trên In-shop — tái lập độc lập. [Nguồn nói](https://arxiv.org/abs/2304.05884)
   Fine-tune đưa ViT lên 86–95% R@1 → khoảng cách ~45 điểm là không gian thật cho Task 2.
2. **Crop theo bbox tụt chủ yếu vì màu pad, không phải vì mất bối cảnh.** Pad màu trung bình CLIP: 53,6%; pad trắng: 65,8%; ảnh đầy đủ: 71,1%;
   chỉ giữ bối cảnh (che sản phẩm): 22,7% [Đã chạy, val, CI theo item]. Bối cảnh có tín hiệu nhưng không phải nguồn chính.
   Giả thuyết "đường tắt bối cảnh" ban đầu của chúng ta bị **bác bỏ một phần**; phần chênh còn lại (~5,5 điểm) chưa tách được nguyên nhân.
3. **Industry crop khi query khác miền gallery** (ảnh điện thoại/cảnh nhiều vật thể → ảnh catalog), và luôn crop **cùng với embedding được train trên crop**;
   không nơi nào trong các nguồn đã đọc đưa crop vào CLIP frozen rồi kỳ vọng tăng. In-shop là shop-to-shop cùng buổi chụp nên ảnh đầy đủ thắng là hợp lý. [Suy luận từ nguồn bên dưới]
4. **Task 2 nên làm:** giữ ảnh đầy đủ (C0); (A) head tuyến tính/MLP + ArcFace hoặc Proxy Anchor trên embedding CLIP đã cache;
   (B) fine-tune vài block cuối CLIP theo LP-FT trên Kaggle T4; mốc backbone frozen mạnh hơn (CLIP B/16, Unicom) làm hàng tham chiếu.

## Thế giới làm visual product search thế nào (industry)

| Hệ thống | Crop/detect hay ảnh đầy đủ | Học embedding | Nguồn |
|---|---|---|---|
| Alibaba Pailitao (KDD 2018) | Detect chủ thể trên query (ảnh người mua nhiễu), học chung với feature từ click | Triplet từ click; Identical R@1 0,415 vs 0,067 GoogLeNet generic | [arXiv 2102.04674](https://arxiv.org/abs/2102.04674) |
| Pinterest Shop The Look (KDD 2020) | Detect từng vật thể trong cảnh | Dữ liệu matching thời trang tăng P@1 0,323→0,622 — lợi ích chủ yếu từ **dữ liệu miền**, không phải crop | [arXiv 2006.10866](https://arxiv.org/abs/2006.10866) |
| Pinterest Unified Embedding (KDD 2019) | Chủ yếu ảnh đầy đủ khi train | Proxy/softmax đa nhiệm trên nhãn click + nhãn người | [arXiv 1908.01707](https://arxiv.org/abs/1908.01707) |
| Amazon StyleSnap | Detect + crop; dán vật thể lên nền ngẫu nhiên để thu hẹp khoảng cách miền | Không công bố số | [Amazon Science](https://www.amazon.science/latest-news/the-science-behind-amazons-new-stylesnap-for-home-feature) |
| Wayfair (WACVW 2026) | YOLOX detect query; catalog không nêu crop | Exact-product R@1: CLIP B/16 27,7 · CLIP H/14 41,8 · H/14 fine-tune **56,4** | [arXiv 2601.11769](https://arxiv.org/abs/2601.11769) (đã kiểm tra bảng 4) |
| eBay (KDD 2017) | Ảnh đầy đủ (ảnh listing đã tập trung vào sản phẩm) | Phân loại category + hash | [arXiv 1706.03154](https://arxiv.org/abs/1706.03154) |
| Zalando Studio2Shop | Ảnh studio đầy đủ, không box; khi query là ảnh đường phố thì tách người, làm trắng nền | Đánh giá cả trên In-shop | [arXiv 1807.00556](https://arxiv.org/abs/1807.00556) |
| Shopee MIEM (IAAI 2024) | — | Embedding **mức item** từ nhiều ảnh + text | [arXiv 2311.17954](https://arxiv.org/abs/2311.17954) |
| GUIE Kaggle 2022, hạng 5 | Center crop nhẹ giúp; crop mạnh hại khi bối cảnh có ích | **CLIP frozen + head BN+Linear + ArcFace** | [arXiv 2210.09495](https://arxiv.org/abs/2210.09495) (đã kiểm tra abstract) |

Mẫu chung: dự đoán category → định vị vật thể → embedding học bằng metric/proxy loss trên nhãn yếu → ANN + lọc category → rerank.

## Học thuật: trần và cách adapt rẻ

| Kết quả | R@1 In-shop | Nguồn |
|---|---|---|
| CLIP frozen B/32 · B/16 · L/14 | 45,4 · 56,1 · 62,9 | Unicom Bảng 2 (đã kiểm tra) |
| Unicom frozen B/32 · fine-tune B/32 | 72,8 · 94,8 | Unicom Bảng 2/4 |
| CLIP B/16 fine-tune toàn bộ (index 8 miền) · DINOv2 off-the-shelf | 86,2 · 33,4 | [UnED, ICCV 2023](https://arxiv.org/abs/2309.01858) — lane báo cáo |
| ViT-S frozen · Hyp-ViT fine-tune (batch 900, A100) | 43,2 · 92,5 | [Hyp-ViT, CVPR 2022](https://arxiv.org/abs/2203.10833) — lane báo cáo |
| PEFT visual prompt, ViT-S batch 64, 1 GPU | 88,4–91,2 | [VPTSP, ICLR 2024](https://arxiv.org/abs/2402.02340) — lane báo cáo |

- Trên CUB, linear probe gần bằng full fine-tune (84,9 vs 85,5) — chưa có số In-shop, nên kết quả của dự án sẽ lấp khoảng trống này. (VPTSP Bảng 1, lane báo cáo)
- LP-FT: train head trước rồi mới mở backbone, ít làm hỏng đặc trưng pretrain. [arXiv 2202.10054](https://arxiv.org/abs/2202.10054)
- Background bias trong metric learning: SOP MAP@R 39,65 → 6,51 khi thay nền. [arXiv 2210.01615](https://arxiv.org/abs/2210.01615)
- CLIP-Adapter/Tip-Adapter dựa trên prompt lớp — không hợp với item test chưa thấy; Marqo-FashionCLIP/SigLIP có thể đã dùng In-shop khi đánh giá và không công bố dữ liệu train → cẩn thận rò rỉ nếu dùng.

## Thí nghiệm validation của dự án [Đã chạy]

1.232 query / 1.384 gallery (400 item tách từ train); CI 95% bootstrap theo item_id; official test không dùng.

| Cấu hình | R@1 | R@5 | R@1 ảnh nghiêng | R@1 − C0 (CI 95%) |
|---|---|---|---|---|
| C0 ảnh đầy đủ | **71,1** | **87,3** | 76,1 | — |
| C1 crop + pad màu mean | 53,6 | 69,5 | 37,9 | −21,1 … −14,1 |
| A1 crop + pad trắng | 65,8 | 84,1 | 58,3 | −8,4 … −2,4 |
| A2 margin 0,3 / 0,5 | 54,9 / 54,7 | 70,3 / 71,4 | 36,4 / 37,9 | ≈ −16 |
| A3 ảnh đầy đủ khi box < 32 px | 54,2 | 70,7 | 40,2 | −20,4 … −13,5 |
| X chỉ bối cảnh | 22,7 | 38,3 | 20,1 | −52,3 … −44,4 |
| C3 HSV histogram | 33,4 | 48,8 | 11,7 | — |

## Phản biện độc lập (Codex) và cách xử lý

| Điểm | Loại | Xử lý |
|---|---|---|
| Bootstrap theo ảnh làm CI quá hẹp | Sự thật | **Đã sửa**: bootstrap theo item; kết luận giữ nguyên |
| Val gallery nhỏ gấp ~9 lần test → val 71% không so được với test 45,5% | Sự thật | Chỉ so cấu hình với nhau trên val. Đề xuất: thêm ảnh train-ID khác làm distractor để val khó như test |
| Thử nhiều biến thể trên cùng val → dễ overfit val | Phán đoán | Chốt ít cấu hình, ghi rõ số lần thử; test chạy một lần |
| Probe "chỉ bối cảnh" có thể sót vải ngoài box và tạo hình chữ nhật lạ | Sự thật (giới hạn) | Ghi là giới hạn; phương án tốt hơn: tách nền bằng segmentation (rembg/U2-Net) |
| C0 lệch 1 hit giữa hai lần chạy | Sự thật | Đã kiểm tra: lệch do đổi batch size dưới fp16; cùng batch cho kết quả giống hệt |
| Gợi ý DINOv2 làm backbone đối chứng | Bất đồng về sự thật | Nguồn cho thấy DINOv2 yếu trên In-shop (UnED 33,4) → ưu tiên CLIP B/16 hoặc Unicom |
| Không thấy lỗi code đủ giải thích mức giảm | — | Khớp kiểm tra của chúng ta |

## Kế hoạch Task 2 đề xuất (chọn trên val, test một lần)

| Cấu hình | Thay đổi một yếu tố | Chi phí ước tính [Suy luận] |
|---|---|---|
| T0 = C0 frozen | Baseline | Đã có |
| T1 head tuyến tính/MLP + ArcFace (hoặc Proxy Anchor) trên embedding cache | Thêm adaptation nhẹ | Encode 23k ảnh train ~2 phút GPU laptop; train vài phút |
| T2 = T1 + mở 2–4 block cuối CLIP (LP-FT), fp16, P×K sampler | Mức adaptation | 2–5 giờ Kaggle T4 |
| T2 + augmentation C2 | Augmentation | Như T2 |
| Hàng tham chiếu: CLIP B/16, Unicom B/32 frozen | Backbone | Encode ~10–20 phút mỗi model |

Phân tích lỗi thêm: thay nền bằng segmentation để đo mức phụ thuộc bối cảnh; gom embedding theo item phía gallery; lọc theo category.

## Giới hạn

- Không đọc được toàn văn: JD (1908.07389), GUIE hạng 1, Virtual ID, V²L, MIEM (chỉ abstract); Google Lens, Mercari, ZOZO, Naver/Kakao không có nguồn kỹ thuật công khai tìm thấy.
- Số của UnED, Hyp-ViT, VPTSP, Kobs do lane báo cáo; đã tự kiểm tra Unicom, Wayfair, GUIE-5.
- CLIP có thể đã thấy ảnh DeepFashion khi pretrain; chưa có cách loại trừ.
