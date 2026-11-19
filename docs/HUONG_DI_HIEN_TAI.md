# Hướng hiện tại — CLIP adaptation cho fashion retrieval

Cập nhật 2026-11-07 sau khi rà Git, source và tính lại retrieval từ cache. Đây là nơi ghi hướng hiện hành; báo cáo pilot và bài học cũ được giữ làm lịch sử. Thiết kế có thể thay khi nghiên cứu hoặc thực nghiệm cho thấy cách phù hợp hơn.

Mục tiêu người dùng nêu 2026-11-14: môn tự chọn, cần hiểu pipeline, trình bày và phản biện được với thầy/Phong, không cầu toàn. Bản giải thích + tư duy ngược + phân công: [HIEU_VA_PHAN_BIEN.md](HIEU_VA_PHAN_BIEN.md). Kế hoạch task theo tuần (deadline ước 2–3 tuần): [KE_HOACH.md](KE_HOACH.md). Bản nháp báo cáo: [BAO_CAO_NHAP.md](BAO_CAO_NHAP.md); demo: `notebooks/task2_demo.ipynb`; link model: [MODEL_LINKS.md](../MODEL_LINKS.md). Đọc từng bước bằng sơ đồ: [PIPELINE_TUNG_BUOC.md](PIPELINE_TUNG_BUOC.md).

## Bài toán

**Topic:** Fashion Product Retrieval Across Views Using CLIP Adaptation.

**Dataset:** [DeepFashion In-shop (CUHK)](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html).

**Input:** một ảnh quần áo. **Output:** Top-K ảnh trong gallery, xếp hạng để tìm cùng mặt hàng. `item_id` dùng để học/chấm, không đưa đáp án query vào tìm kiếm. Chưa có bước sinh hoặc bảo đảm chọn ảnh chính diện; benchmark không chứng minh hiệu quả với mọi ảnh ngoài đời.

**Câu hỏi:** Thích nghi CLIP như thế nào để phân biệt từng mặt hàng qua các góc chụp, và lợi ích có xứng với công sức/compute không?

## Bài tham khảo chính trước mắt

Theo thống nhất với người dùng ngày 2026-11-07, lấy **[CLIP-Adapter: Better Vision-Language Models with Feature Adapters, v2](https://arxiv.org/html/2110.04544v2)** làm bài phương pháp chính để học và triển khai trước mắt, kết hợp code tác giả. Tập trung hiểu visual adapter, bottleneck, residual blending và cách đóng băng encoder; áp dụng cơ chế đó cho retrieval DeepFashion với mục tiêu huấn luyện phù hợp.

Đây là bài tham khảo chính hiện tại, không phải tuyên bố chỉ có một nghiên cứu liên quan. Có thể đọc thêm hoặc mở rộng khi cần giải quyết lựa chọn loss, cách train hoặc một hướng cải thiện đáng làm. Partial fine-tuning, local features và reranking là các khả năng, chưa phải phần nhóm cam kết triển khai. Việc chuyển từ classification trong bài gốc sang retrieval cần được mô tả rõ.

## Sơ đồ tổng thể

```text
Ảnh + annotation DeepFashion
          ↓
Hiểu item_id/split; chọn preprocessing và protocol
          ↓
E0: frozen CLIP → embedding → cosine → Top-K → Recall@1/5
          ↓
E1: frozen CLIP + residual visual adapter
    train bằng train IDs; chọn cấu hình bằng validation IDs
          ↓
Query và gallery dùng cùng encoder + adapter đã học
          ↓
So sánh Recall, chi phí và các trường hợp đúng/sai
          ↓
Có thể mở rộng nếu cần: partial FT / local features / reranking
          ↓
Demo query → Top-K; báo cáo và giải thích kết quả
```

## Bằng chứng hiện có

- Đã chạy: frozen OpenAI CLIP ViT-B/32, ảnh đầy đủ, chuẩn hóa L2 và exact cosine; crop/padding/HSV trên validation.
- Rà 2026-11-07: tính lại trên **14.218 query / 12.612 gallery** từ cache, **R@1 45,5057%, R@5 67,2457%**, khớp số cũ. Cache checksum/thứ tự ảnh, annotation, source manifest và split đã kiểm tra. Không train, mã hóa lại ảnh hay hash lại toàn bộ pixel. Chi tiết: [baseline_review.json](../reports/baseline_review.json).
- 14 query khác first-positive rank so với CSV lần chạy cloud, không đổi hit@1/hit@5. Không gọi là tái lập bit-for-bit mọi thứ hạng.
- Chưa có training loop/checkpoint/kết quả cho adapter, retrieval head hoặc partial fine-tuning.
- Validation: 400 ID, 1.232 query / 1.384 gallery từ official train; train còn 23.266 ảnh. Val nhỏ hơn và cách chia khác test, không so trực tiếp val 71,1% với test 45,5%.

## Thiết kế khởi đầu, có thể sửa

1. **E0:** dùng baseline cũ nếu khớp protocol. Đổi model, preprocessing hoặc điều kiện so sánh thì chạy baseline tương ứng. Không giữ code chỉ vì đã làm.
2. **E1 — hướng chính:** residual visual adapter sau encoder, backbone frozen. Kế thừa [CLIP-Adapter](https://arxiv.org/html/2110.04544v2) và [code tác giả](https://github.com/gaopengcuhk/CLIP-Adapter/blob/main/clip_adapter.py). Code gốc classification không phải pipeline retrieval hoàn chỉnh. Code có ReLU sau linear thứ hai, khác công thức rút gọn trong paper; chọn và ghi rõ phiên bản khi triển khai.
3. ViT-B/32 có thể bắt đầu với bottleneck **512→128→512**, residual ratio **0,2**, suy từ tỉ lệ giảm chiều/code tác giả. Đây chưa phải cấu hình tối ưu cho In-shop. Không áp nguyên chiều 1024 của code RN50 sang ViT-B/32.
4. **Loss/sampler chưa chốt:** supervised contrastive, triplet hoặc proxy/margin classifier như ArcFace/Proxy Anchor là các ứng viên. Classification trên train IDs rồi bỏ classifier lấy embedding cũng hợp lệ; không dùng classifier ấy dự đoán trực tiếp IDs test chưa thấy. Chọn dựa trên nguồn/triển khai phù hợp, không cần thử mọi loss.
5. **Nhánh đào sâu:** partial FT để thích nghi backbone; local features/crop/reranking để phân biệt chi tiết. Linear head có thể làm đối chứng. Không bắt buộc làm mọi nhánh hoặc qua head trước adapter.
6. Dùng nghiên cứu và kinh nghiệm để chọn recipe khởi đầu; thực nghiệm xác nhận hiệu quả của nhóm. Khi cần giải thích tác dụng riêng của thành phần thì làm đối chứng tương ứng. Không bắt mọi thử nghiệm chỉ được thay đúng một biến.
7. Dùng cùng query/gallery và metric khi so sánh. Cache sau adapter phải gắn với checkpoint adapter; không so query đã adapt với gallery frozen cũ. Cache một embedding cố định mỗi ảnh không cung cấp augmentation ảnh mới qua từng epoch; muốn augmentation cần encode online hoặc cache nhiều view.

## Phân công đề xuất

Chưa phải ghi nhận ai đã làm; nhóm có thể điều chỉnh.

| Phần | Phụ trách chính | Đầu ra |
|---|---|---|
| Research mô hình, lựa chọn recipe và training | Trung | Giải thích kiến trúc/loss; code adapter, cấu hình và kết quả |
| Logic dữ liệu và đánh giá | Phong | Ví dụ item_id/split; tự chấm Top-K và đối chiếu metric |
| Phân tích lỗi và so sánh | Phong, Trung hỗ trợ tích hợp | Kiểu nhầm, ví dụ validation; biến thể sửa/hỏng gì |
| Demo, báo cáo và phản biện | Cùng làm | Query→Top-K; viết phần mình và đọc chéo |
| Task 3 coursework độc lập | Có thể giao Phong sau khi đối chiếu đề | Notebook/đầu ra OpenCV và giải thích |

Gói đầu tiên cho người phụ trách logic: chọn vài item nhiều góc, chỉ rõ nhãn/đáp án; xem khoảng 20 query validation baseline và phân nhóm lỗi. Phần việc hữu hạn, ít phụ thuộc kỹ năng điều phối AI nhưng tác động trực tiếp tới quyết định mô hình. Cả hai cần hiểu pipeline chung.

## Việc tiếp theo

- Dùng bản này làm đề cương trao đổi với đồng đội/thầy, kết hợp ảnh thật. Không cần đợi nắm hết lý thuyết.
- Trước mắt đọc kỹ CLIP-Adapter v2 và code để chọn recipe E1, hiểu cơ chế và phần cần điều chỉnh cho retrieval. Đọc thêm nguồn khi cần; chưa bắt buộc mở nhiều nhánh nghiên cứu. Số paper là tham khảo theo điều kiện, không phải trần hoặc cam kết điểm.
- Triển khai E1, chạy thử vòng train/eval nhỏ rồi làm thí nghiệm có ý nghĩa. Chọn nhánh đào sâu theo hướng học và bằng chứng; không lấy baseline/demo làm giới hạn dự án.
- Giá trị coursework: hiểu/reuse/adapt phương pháp mạnh, làm hệ thống và giải thích kết quả. Chưa đủ cơ sở nói có thuật toán mới hoặc chưa ai làm thiết lập này.
