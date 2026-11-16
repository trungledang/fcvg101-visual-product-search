# Bài học nền tảng và lộ trình Visual Product Retrieval

> **Cập nhật 2026-11-07:** [HUONG_DI_HIEN_TAI.md](HUONG_DI_HIEN_TAI.md) ghi hướng hiện hành. Các lộ trình head-first/T0–T3 bên dưới được giữ làm lịch sử học tập, không phải nghĩa vụ làm đủ. Visual adapter là hướng chính; loss/sampler chưa chốt. Code, baseline và backbone có thể sửa hoặc thay khi phù hợp. Classification trên train IDs có thể là mục tiêu huấn luyện, còn đầu ra và đánh giá của đề tài vẫn là retrieval.

Tài liệu này gom các khái niệm cần nắm để giải thích, phản biện và bàn hướng làm với bạn cùng nhóm hoặc giảng viên. Học theo thứ tự; chưa cần thuộc thuật ngữ trước khi hiểu ví dụ.

## 1. Bài toán của nhóm bằng một câu

Cho một ảnh quần áo làm truy vấn, xếp hạng ảnh trong kho theo khả năng chúng thuộc **cùng một sản phẩm**. Đầu ra hiện tại là danh sách ảnh Top-K trong gallery, kèm điểm tương đồng và thứ hạng. Hệ thống chưa sinh ảnh mới, chưa tự chọn ảnh chính diện, và baseline hiện tại chưa có bước reranking riêng.

Ví dụ: query là ảnh mặt sau của chiếc váy A. Gallery có ảnh mặt trước của A và ảnh của chiếc váy B trông gần giống. Ảnh của A mới là đáp án đúng theo nhãn; B vẫn sai dù có cosine similarity cao.

## 2. Đọc nhãn và chia tập

Trong `Eval/list_eval_partition.txt`, mỗi dòng có ba trường:

```text
image_name                                      item_id       evaluation_status
img/WOMEN/Dresses/id_00000002/02_1_front.jpg   id_00000002   train
```

- `image_name`: đường dẫn ảnh.
- `item_id`: mã nhóm sản phẩm theo annotation. Hai ảnh cùng ID được benchmark coi là cùng sản phẩm.
- `evaluation_status`: vai trò ảnh trong thí nghiệm — `train`, `query` hoặc `gallery`.

`item_id` không được đưa vào bộ tính độ giống lúc tìm kiếm. Nó dùng để dạy mô hình có giám sát ở train (nếu nhóm chọn huấn luyện) và để chấm dự đoán sau khi retrieval hoàn tất.

Dataset hiện có 52.712 ảnh: 25.882 train, 14.218 query và 12.612 gallery. Query và gallery có chung ID để query có đáp án, nhưng không trùng đường dẫn ảnh theo audit của dự án. Các ID train được tách khỏi ID query/gallery để đánh giá trên sản phẩm chưa dùng khi huấn luyện. Đây là đánh giá khả năng tổng quát hóa sang ID mới trong cùng loại dữ liệu in-shop; nó không tự động chứng minh mô hình dùng tốt với ảnh điện thoại ngoài đời.

File `list_bbox_inshop.txt` ghi `image_name`, loại quần áo, tư thế/góc và tọa độ hộp sản phẩm. Baseline hiện dùng ảnh đầy đủ; bbox được dùng trong các thí nghiệm crop.

Lưu ý: `item_id` là ground truth của benchmark, không nhất thiết đồng nhất với SKU bán hàng. Dataset có thể gom ảnh khác màu vào cùng ID. Nếu ứng dụng cần phân biệt chính xác màu/SKU, phải kiểm tra annotation và định nghĩa nhãn phù hợp.

## 3. Đặc trưng: từ pixel đến vector

Máy nhận ảnh dưới dạng các con số pixel, không nhìn ảnh như con người. Ta cần một encoder biến ảnh thành biểu diễn số.

- Đặc trưng thấp có thể phản ánh màu, cạnh, gradient và kết cấu nhỏ.
- Đặc trưng trung gian có thể biểu diễn đường viền, hình dạng và bộ phận.
- Đặc trưng cao là biểu diễn đã được học cho một mục tiêu, chẳng hạn nhận dạng hoặc truy hồi.

Các mức này là cách giải thích khái niệm, không phải nhãn chắc chắn gắn với từng tầng. Mục tiêu huấn luyện quyết định biểu diễn sẽ giữ thông tin nào.

**CNN / ResNet:** bộ lọc tích chập nhìn các vùng cục bộ và dùng chung trọng số khi trượt trên ảnh. Qua các tầng, mạng thường ghép mẫu cục bộ thành hình dạng và biểu diễn cấp cao. ResNet là một kiến trúc CNN có kết nối tắt.

**Vision Transformer (ViT):** chia ảnh thành patch, biến patch thành token rồi dùng self-attention để các token trao đổi thông tin. CLIP ViT-B/32 xử lý ảnh 224×224 thành 49 patch 32×32 trước các bước tiếp theo của encoder. Không nên hiểu patch là đối tượng hoàn chỉnh; đó là các đơn vị biểu diễn ban đầu.

**CLIP trong dự án:** dùng image encoder để biến ảnh thành embedding 512 chiều. Baseline dùng vector ảnh cuối; không tạo caption, không dùng text encoder, và không so sánh từng patch riêng.

## 4. Retrieval, cosine và Recall@K

Cosine similarity giữa hai vector đã chuẩn hóa là tích vô hướng của chúng. Điểm cao hơn nghĩa là hai vector gần hướng hơn; nó **không phải xác suất** hai ảnh là cùng sản phẩm.

```text
Chuẩn bị gallery một lần:
gallery images → CLIP → normalized embeddings → lưu theo đường dẫn

Mỗi lần truy vấn:
query image → cùng preprocessing + CLIP → query embedding
            → cosine với mọi gallery embedding → sắp hạng → trả Top-5

Chấm benchmark sau khi xếp hạng:
tra item_id của query và ảnh kết quả
```

- R@1: tỷ lệ query có ít nhất một ảnh cùng `item_id` ở vị trí đầu.
- R@5: tỷ lệ query có ít nhất một ảnh cùng `item_id` trong năm vị trí đầu.
- Một query đúng Top-1 cũng đúng Top-5.

Nếu query và gallery chứa cùng tệp ảnh, tìm ảnh tự khớp có thể làm bài toán trở nên tầm thường. Trong audit hiện tại, không có đường dẫn query/gallery chồng nhau và không có ảnh trùng pixel. Bài toán có ý nghĩa vì phải tìm **ảnh khác của cùng sản phẩm** giữa nhiều ảnh gây nhiễu.

Baseline chính thức hiện có: frozen CLIP ViT-B/32 đạt R@1 45,51% và R@5 67,25% trên 14.218 query, 12.612 gallery. Kết quả validation 71,10% được tính trên gallery nhỏ hơn đáng kể, nên không so trực tiếp với test. Crop với CLIP đóng băng chưa vượt ảnh đầy đủ trong các thí nghiệm validation đã ghi nhận.

## 5. Backbone, head và ba cách thích nghi

- **Backbone:** phần encoder tạo biểu diễn ảnh; ở đây là CLIP image encoder.
- **Frozen feature extractor / feature extraction:** backbone pretrained được đóng băng và dùng để tạo đặc trưng; chỉ lớp mới phía sau được huấn luyện.
- **Head:** phần phía sau backbone dùng embedding cho mục tiêu cụ thể. Linear head là một phép chiếu tuyến tính; classifier head có thể xuất điểm các lớp. Retrieval head/loss cần phù hợp với việc học biểu diễn phân biệt sản phẩm.
- **Fine-tuning:** cập nhật tiếp trọng số model pretrained trên dữ liệu và mục tiêu mới.
- **Adapter:** mô-đun nhỏ chèn vào một số khối backbone, được học trong khi nhiều trọng số gốc giữ nguyên. Adapter khác head vì nó can thiệp bên trong encoder.

| Chiến lược | Phần được cập nhật | Câu hỏi nó trả lời | Đặc điểm |
|---|---|---|---|
| Frozen backbone + head | Head | Vector CLIP sẵn có đã đủ hữu ích chưa? | Rẻ, đơn giản, dễ phân tích |
| Partial fine-tuning | Head + một số block cuối | Biểu diễn sâu có cần thích nghi thêm không? | Cân bằng khả năng và chi phí |
| Full fine-tuning | Toàn bộ backbone + head | Cập nhật toàn mạng có giúp đáng kể không? | Tốn tài nguyên, nhạy với cấu hình và overfit |
| Frozen backbone + adapter | Adapter + head theo thiết kế | Điều chỉnh nhẹ bên trong encoder có hiệu quả không? | Thêm độ phức tạp; cần đối chứng phù hợp |

Chiến lược hai giai đoạn thường là: (1) đóng băng backbone để head học ổn định; (2) mở một số tầng cuối và tiếp tục huấn luyện với learning rate thấp hơn cho backbone. Có thể so với head-only. Không bắt buộc mở từng tầng một; cần chọn mức phức tạp vừa với câu hỏi và tài nguyên.

## 6. Learning rate và tối ưu hóa

Learning rate quyết định độ lớn bước cập nhật trọng số. Learning rate quá lớn có thể làm loss dao động hoặc làm hỏng biểu diễn pretrained; quá nhỏ có thể khiến học rất chậm hoặc chưa đủ thích nghi trong số epoch đang chạy. Learning rate nhỏ **không đảm bảo** đạt tối ưu toàn cục, và loss giảm trên train không đảm bảo Recall@K tăng trên dữ liệu chưa thấy.

Một điểm bắt đầu thường dùng khi mở backbone là learning rate của backbone thấp hơn learning rate của head mới. Đây là cấu hình cần kiểm tra qua validation, không phải công thức bảo đảm. Các phần bị đóng băng không được cập nhật nên không có learning rate hữu hiệu.

Phân biệt:

- **Loss:** tín hiệu tối ưu hóa mà thuật toán dùng để cập nhật trọng số.
- **R@1/R@5:** metric đánh giá retrieval sau khi xếp hạng.
- Giảm loss không đồng nghĩa chắc chắn R@1 tăng. Mục tiêu huấn luyện, nhãn, cách lấy batch và metric cần phù hợp.

Fine-tuning khởi đầu từ trọng số pretrained, không phải khởi tạo ngẫu nhiên toàn bộ model. Tuy vậy head mới có thể bắt đầu ngẫu nhiên. Cần theo dõi train và validation để nhận ra học không đủ, overfit hay cấu hình mất ổn định.

## 7. Đọc kết quả và điểm nghiên cứu

Các phương pháp retrieval trên In-shop đã đạt khoảng 95% R@1 trong những thiết lập có huấn luyện chuyên biệt. Đây là mốc tham khảo, không phải yêu cầu coursework và không so trực tiếp với baseline zero-training nếu khác backbone, pretraining, split hoặc protocol.

Vì có phương pháp mạnh đã tồn tại, nhóm không nên tuyên bố “phát minh retrieval mới” chỉ vì thêm một head/adapter. Giá trị phù hợp cho coursework là:

1. Tái lập baseline và hiểu dữ liệu/nhãn.
2. So sánh có kiểm soát một hoặc hai mức thích nghi.
3. Xem ví dụ lỗi để giải thích khi nào cải thiện và khi nào không.
4. Nêu giới hạn: In-shop khác ảnh consumer-to-shop; nhãn item_id có thể không tương ứng SKU/màu bán hàng.

## 8. Lộ trình khuyến nghị cho nhóm hai người

**Bước 1 — Cùng học dữ liệu trước.** Hai thành viên chọn 5–10 query; với mỗi query mở một gallery positive cùng `item_id`, một ảnh khác ID trông giống, và xem kết quả Top-5. Ghi điều gì thật sự thấy, tránh đoán nguyên nhân khi chưa có đối chứng.

**Bước 2 — Chốt câu hỏi.** Hướng khả thi nhất với trạng thái hiện tại:

> Khi train trên các `item_id` ở train, head nhẹ có cải thiện retrieval các sản phẩm chưa thấy so với CLIP đóng băng không?

Đây là nghiên cứu ứng dụng/đánh giá trong coursework, chưa phải tuyên bố phương pháp mới.

**Bước 3 — Chốt protocol trước khi train.** Dùng validation tách theo item_id để chọn cấu hình. Giới hạn số cấu hình và ghi lại seed, loss, optimizer, learning rate, batch/sampler và số epoch. Không chọn tham số theo official test.

**Bước 4 — Thí nghiệm vừa sức.** Giữ baseline frozen CLIP làm T0. Làm một head retrieval làm T1. Nếu T1 chạy ổn và còn thời gian, cân nhắc mở một nhóm block cuối làm T2. Đổi từng yếu tố có thể diễn giải; không đồng thời đổi crop, augmentation, loss, số tầng và learning rate.

**Bước 5 — Đánh giá và trình bày.** Chọn mô hình trên validation; chạy official test cuối theo protocol đã chốt; báo R@1/R@5, ví dụ tốt/xấu và giới hạn. Hoàn thành hai phần độc lập Task 3 của đề: camera/geometric correction cùng filtering/gradient/keypoints; và chọn optical flow hoặc stereo.

Phân chia gợi ý: một thành viên phụ trách audit/hiểu dataset và phân tích ví dụ; thành viên còn lại phụ trách baseline/head và ghi cấu hình. Cả hai cùng xem lỗi, quyết định protocol, đối chiếu kết quả và giải thích báo cáo. Phân chia thực tế theo năng lực/lịch của nhóm.

## 9. Câu hỏi đem trao đổi với giảng viên

1. Với coursework, có phù hợp nếu câu hỏi là “head-only so với partial fine-tuning CLIP cho retrieval theo item_id” trên In-shop không?
2. Mức nào được xem là đủ cho Task 2 khi pretrained/frozen model được phép, nhưng đề cũng yêu cầu baseline và ít nhất hai cấu hình đối chứng?
3. Nên giữ In-shop để làm thí nghiệm có thể tái lập, hay chuyển sang Consumer-to-shop để gần tình huống ảnh người dùng chụp hơn dù cần làm lại pipeline?
4. Đầu ra mong muốn của nhóm là danh sách ảnh cùng sản phẩm (retrieval benchmark) hay một ảnh đại diện chính diện? Nếu là ảnh đại diện, cần định nghĩa và đánh giá bước chọn ảnh đó riêng.

## 10. Tóm tắt 30 giây để nói với bạn cùng nhóm

“Mình đưa ảnh quần áo vào CLIP để tạo vector, so vector đó với ảnh trong gallery rồi xếp hạng. `item_id` là nhãn để biết ảnh nào thuộc cùng sản phẩm và chấm kết quả, không phải thông tin model được cho lúc tìm. Baseline frozen CLIP hiện đạt 45,5% R@1. Mình đề xuất trước hết hiểu vài query bằng ảnh thật, rồi thử head trên train và so với frozen CLIP trên validation. Nếu còn thời gian mới mở một số block cuối. Mục tiêu là hiểu và đánh giá adaptation, không tuyên bố vượt phương pháp nghiên cứu đã có.”

## 11. Chỉnh mô tả đề tài và phân công nhóm

### Mô tả hiện tại

> Visual Product Retrieval Using Pretrained CLIP and Image Representation Techniques. Dataset: DeepFashion In-shop Clothes Retrieval. Develop a fashion product search system using pretrained CLIP. Compare three image representation methods and evaluate retrieval accuracy using Recall@1 and Recall@5.

Mô tả này đúng hướng nhưng cần cụ thể hơn. “Three image representation methods” chưa nêu rõ ba phương pháp là gì; người đọc có thể hiểu nhầm chúng đều là các encoder/model retrieval được huấn luyện. Thực tế hiện có: (1) CLIP frozen trên ảnh đầy đủ, (2) CLIP frozen trên ảnh crop bbox, và (3) HSV histogram trên ảnh crop/pad. Hai lựa chọn crop và HSV đã thấp hơn ảnh đầy đủ trên validation; đây là kết quả âm có giá trị phân tích, không nên hứa rằng chúng sẽ cải thiện hệ thống. Chưa có kết quả huấn luyện head hoặc fine-tuning.

### Bản mô tả tiếng Anh đề xuất — trung thực với trạng thái hiện có

**Title**

> **Visual Product Retrieval on DeepFashion In-Shop: Evaluating Pretrained CLIP and Image Representations**

**Dataset**

> **DeepFashion In-shop Clothes Retrieval Benchmark (CUHK)** — official train/query/gallery annotations, with product identities held out between training and evaluation.

**Aim / project description**

> This project builds an image-to-image fashion retrieval pipeline using a frozen pretrained CLIP ViT-B/32 encoder. It compares full-image CLIP embeddings with bbox-cropped CLIP embeddings and a handcrafted HSV-histogram representation, using a consistent gallery and identity-based Recall@1 and Recall@5 evaluation. It audits the official product-ID splits, analyses retrieval errors, and discusses the limitations of in-shop evaluation for consumer-to-shop search. A supervised retrieval head may be evaluated as a separately reported extension if completed.

**Research question**

> How do full-image CLIP, bbox-cropped CLIP and a handcrafted HSV representation compare for retrieving images of the same clothing item across different views in the DeepFashion In-shop benchmark?

This wording describes experiments already present in the repository. If the team completes and verifies a trained head, revise the aim and question to include that experiment explicitly, for example: “Does a lightweight identity-trained retrieval head improve over frozen CLIP on held-out product identities?” Do not describe the head as completed before it has actually been trained and evaluated.

### Đóng góp nên trình bày

Với mô tả hiện tại, đóng góp phù hợp là một **so sánh thực nghiệm có kiểm soát và phân tích**: audit dữ liệu/split theo identity, dựng baseline CLIP có thể tái lập, đo các biểu diễn khác nhau trên cùng validation/test protocol, và giải thích bằng ví dụ vì sao preprocessing crop/HSV có thể không giúp. Đây là đóng góp coursework về triển khai, đánh giá và hiểu giới hạn; chưa phải tuyên bố phát minh thuật toán mới hoặc vượt state of the art.

Nếu thêm retrieval head đã train, đóng góp thực nghiệm trở nên cụ thể hơn: đo mức cải thiện so với frozen CLIP khi học từ nhãn train và đánh giá trên product IDs chưa thấy. Vẫn cần gọi đó là đánh giá adaptation đã biết trong phạm vi dataset, trừ khi nhóm có phương pháp mới được chứng minh.

### Phân công gợi ý cho hai người

| Thành viên | Phần chính | Bằng chứng bàn giao |
|---|---|---|
| A — dữ liệu, protocol, phân tích | Mô tả dataset và annotation; xác nhận train/query/gallery và identity separation; kiểm tra positive/duplicate; tạo bảng ví dụ query–positive–hard negative; phân tích lỗi theo pose/category khi nhãn hỗ trợ | Sơ đồ pipeline dữ liệu, bảng audit, các ví dụ có đường dẫn/ID, phần giới hạn dữ liệu |
| B — biểu diễn, retrieval, thực nghiệm | Giải thích preprocessing/CLIP embedding; chạy hoặc tái lập các cấu hình C0 full-image, C1 bbox crop, C3 HSV; tạo bảng Recall@1/5 đúng split và log cấu hình/tài nguyên | Mã/config, metrics theo split, bảng so sánh và biểu đồ |
| Cùng làm | Chốt câu hỏi và tiêu chí so sánh; xem ảnh lỗi; viết kết luận/giới hạn; hoàn thành Task 3A và 3B; kiểm tra hai người đều giải thích được nhãn và metric | Báo cáo tích hợp, demo inference, phần phản biện và trình bày |

Phân công này là điểm xuất phát, không phải ranh giới cứng. Hai người cần cùng review pipeline và kết luận để khi thuyết trình cả hai đều trả lời được vì sao cùng `item_id` được xem là đúng và vì sao gallery không thể được lọc theo đáp án.

Nếu còn thời gian cho một thí nghiệm Task 2 có huấn luyện, ưu tiên thêm **frozen CLIP + retrieval head** làm cấu hình mới; giao cho người phụ trách mô hình, còn người phụ trách dữ liệu độc lập rà soát item-level validation và leakage. Chốt head/loss/protocol trước khi chạy. Chỉ cân nhắc partial fine-tuning sau khi head-only đã chạy ổn và có thời gian/compute; không thêm adapter, crop mới, augmentation và full fine-tuning cùng lúc.

## 12. Ghi chú paper CLIP-Adapter (arXiv:2110.04544v2)

Paper tham khảo: [CLIP-Adapter: Better Vision-Language Models with Feature Adapters](https://arxiv.org/html/2110.04544v2).

### Paper thực sự làm gì?

- Giữ nguyên, đóng băng backbone pretrained CLIP.
- Lấy biểu diễn ảnh cuối từ image encoder; paper đặt adapter nhỏ dạng bottleneck **sau encoder**, thay vì chèn adapter vào từng Transformer block.
- Adapter gồm hai phép chiếu tuyến tính với bottleneck và ReLU để tạo đặc trưng điều chỉnh.
- Trộn đặc trưng điều chỉnh với đặc trưng CLIP gốc bằng residual ratio. Mục đích là vừa học thêm thông tin cho tác vụ mới vừa giữ lại thông tin pretrained.
- Phần adapter được học; backbone CLIP được giữ cố định.
- Thí nghiệm paper chủ yếu là few-shot **classification**. Classifier dự đoán các lớp đã định nghĩa trong tập train; đó không phải cùng mục tiêu với truy hồi sản phẩm chưa thấy.

Đây là một cách parameter-efficient adaptation: không fine-tune toàn backbone, nhưng vẫn có bước train cho adapter. Adapter khởi tạo chưa có khả năng điều chỉnh hữu ích cho tác vụ mới cho đến khi được huấn luyện.

### Áp dụng ý tưởng vào retrieval của nhóm

Mục tiêu dự án là tạo embedding để ảnh cùng sản phẩm xếp gần nhau, rồi truy hồi trên gallery. Có thể giữ thiết kế adapter sau image encoder và CLIP frozen, nhưng phải dùng huấn luyện phù hợp với retrieval. Hai khả năng cần phân biệt:

1. **Adapter + classifier theo train `item_id`:** huấn luyện phân loại các ID train rồi dùng embedding adapter để truy hồi các ID held-out. Đây là baseline adaptation dễ hiểu, nhưng train classifier accuracy không phải kết quả retrieval; vẫn phải đánh giá Recall@K riêng trên IDs chưa thấy.
2. **Adapter + metric-learning objective:** huấn luyện trực tiếp đặc trưng ảnh theo quan hệ cùng/khác `item_id` (ví dụ ArcFace hoặc một metric-learning loss phù hợp), sau đó truy hồi bằng cosine. Cách này gần mục tiêu retrieval hơn, nhưng cần chốt loss, batch/sampler và khả năng triển khai.

Không được đưa `item_id` query/test vào lúc tìm. ID chỉ dùng trên train cho supervision và sau ranking để chấm; validation chọn cấu hình trên các IDs không tham gia huấn luyện; official test chỉ đánh giá cuối.

### Bảng thí nghiệm gợi ý nếu CLIP-Adapter là trọng tâm

| Cấu hình | CLIP backbone | Phần học | Vai trò |
|---|---|---|---|
| T0: frozen CLIP | Đóng băng | Không | Baseline đã chạy |
| T1: frozen CLIP + head | Đóng băng | Retrieval head | Đối chứng đơn giản để tách lợi ích của head |
| T2: CLIP-Adapter | Đóng băng | Bottleneck adapter/residual mapping | Thí nghiệm chính: adaptation nhẹ kiểu paper |
| T3: partial fine-tuning (tùy thời gian) | Mở vài block cuối | Các block mở và head | Mốc adaptation sâu hơn; không bắt buộc nếu quá tải |

Giữ cùng split, ảnh query/gallery, preprocessing và Recall@1/R@5. T0/T1/T2 đã đủ một câu hỏi coursework rõ; chỉ thêm T3 khi T2 chạy được và còn thời gian. Không đồng thời thay adapter, crop, augmentation, loss và sampler vì sẽ không biết nguyên nhân kết quả đổi.

### Câu hỏi phương pháp cần chốt trước khi code

- Adapter học đầu ra nào: projection embedding cho metric learning hay classifier theo train IDs rồi bỏ classifier khi truy hồi?
- Dùng loss nào để embedding tổng quát hóa sang IDs test chưa thấy?
- Làm sao tạo batch có nhiều ảnh mỗi item để loss nhìn thấy positive pairs?
- Tiêu chí chọn checkpoint trên item-disjoint validation là gì?
- Có đủ compute/thời gian để train adapter và mã hóa lại toàn bộ gallery không?

Khuyến nghị học/triển khai: bắt đầu bằng T0 đã có và T1 head-only; sau đó triển khai T2 theo cơ chế adapter của paper (backbone đóng băng, bottleneck sau encoder, residual blend), nhưng chọn objective phù hợp retrieval. Ghi rõ đây là **adaptation từ CLIP-Adapter sang retrieval**, không phải tái lập nguyên xi paper classification.

### Trạng thái tại thời điểm handoff

- Đã có: frozen CLIP ViT-B/32 baseline trên official test (R@1 45,51%, R@5 67,25%); crop/padding/HSV validation experiments.
- Chưa có: retrieval head được train; CLIP-Adapter được train; partial fine-tuning được train.
- Hướng đang cân nhắc: giữ CLIP frozen và thử retrieval head cùng CLIP-Adapter; ưu tiên adapter nếu đây là câu hỏi nhóm muốn học từ paper. Cần chốt loss/sampler và xác nhận với đồng đội/giảng viên rằng mức phạm vi phù hợp coursework.
- Không tuyên bố adapter tăng Recall trước khi có kết quả thực nghiệm.

### Mẫu mô tả ngắn để gửi cho bạn cùng nhóm

> “Tên đề tài hiện tại là truy hồi ảnh sản phẩm thời trang trên DeepFashion In-shop bằng CLIP và các biểu diễn ảnh. Ta đã có CLIP ảnh đầy đủ làm baseline, cùng thí nghiệm crop bbox và HSV trên validation. Đóng góp mình có thể bảo vệ là audit nhãn/split, so sánh công bằng các biểu diễn, và phân tích kết quả thật — kể cả khi crop không tốt hơn. Nếu cùng muốn có phần học mô hình cho Task 2, bước tiếp theo là thêm head retrieval được train bằng item_id trên train rồi so với CLIP frozen trên validation; đây hiện là đề xuất, chưa phải kết quả.”

## Tài liệu và hiện trạng dự án

- `README.md`: cách chạy, số liệu chính và trạng thái thí nghiệm.
- `docs/START_HERE.md`: bài thực hành đọc ảnh, nhãn và truy vấn.
- `docs/DATASET_AND_PIPELINE.md`: chi tiết annotation, split và pipeline.
- `reports/metrics.json`: kết quả baseline test.
- `reports/val_results.json` và `reports/val_crop_ablation.json`: kết quả validation.
- `QDH_Coursework_FCVG101_.docx` ở thư mục môn: yêu cầu coursework gốc.

Trong trạng thái repository được đọc khi viết tài liệu này, chưa có bằng chứng về model head ArcFace hoặc partial/full fine-tuning đã được train và đánh giá.
