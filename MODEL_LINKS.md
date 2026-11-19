# Model, dữ liệu và tài nguyên cần để chạy lại

| Thứ cần | Lấy ở đâu | Đặt vào |
|---|---|---|
| Adapter đã train (E1/E2/E3 × seed 0–2, 9 file, mỗi file ~0,5 MB) | **Google Drive:** https://drive.google.com/file/d/1XWptbtHuBulXwOthm_7qVEHB8puaAw7c/view (`FCVG101_adapter_checkpoints.zip`, 4,4 MB, SHA-256 `28807174…421a69`) | giải nén vào `cache/adapter/` → `cache/adapter/ep40/*.pt` |
| Adapter dùng trong demo (E1, E3 seed 0) | có sẵn trong ZIP nộp: `cache/adapter/ep40/` | — |
| CLIP ViT-B/32 (OpenAI, đóng băng) | tự tải khi chạy lần đầu từ URL chính thức trong gói `openai/CLIP`; SHA-256 được kiểm tra (`vps/embed.py`) | `.cache/clip/ViT-B-32.pt` |
| DeepFashion In-shop (ảnh, `Eval/list_eval_partition.txt`, `list_bbox_inshop.txt`) | https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html (chỉ dùng nghiên cứu/giáo dục phi thương mại) | thư mục **cha** của repo: `../img/`, `../Eval/`, `../list_bbox_inshop.txt` |

Mỗi checkpoint là `torch.save({"state_dict", "config", ...})`; `config` ghi `ratio`, `reduction`, seed và epoch tốt nhất trên validation.
Nạp: `vps.search.ProductSearch` hoặc `scripts/eval_test.py` (xem `docs/T1_ADAPTER_GIAI_THICH.md`, mục 4).

## Thứ tự chạy lại từ đầu (GPU 4 GB đủ; CPU chạy được nhưng chậm)

```text
pip install -r requirements.txt
python scripts/pilot.py embed --batch-size 64 --workers 6     # embedding CLIP cho query/gallery (E0)
python scripts/encode_variant.py --split train|query|gallery  # crop + pad trắng (E3)
python scripts/encode_augmented.py                            # 3 bản augmented / ảnh train (E2)
python scripts/train_adapter.py --epochs 40 --seed 0 --ckpt-dir cache/adapter/ep40   # E1; thêm --aug-cache (E2) hoặc --base-cache (E3), lặp seed 1, 2
python scripts/eval_test.py                                   # đánh giá test một lần → reports/test_results.json
notebooks/task2_demo.ipynb                                    # bảng, demo suy luận, phân tích lỗi
```
