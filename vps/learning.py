"""Tiện ích đọc cache và vẽ hình cho notebook học; baseline không gọi module này."""
from __future__ import annotations

import json
from pathlib import Path

from .annotations import file_sha256
from .embed import load_cache, official_checkpoint


def read_embeddings(project_root: Path, split: str, records, annotation_file: Path):
    """Đọc embedding đã đo trên Kaggle, giữ nguyên thứ tự trong partition.

    Dùng kiểm tra checksum/order của baseline. Không so mtime giữa máy Windows
    và Kaggle: hai bản sao dữ liệu có thời gian ghi file khác nhau. Notebook này
    chỉ đọc cache, không sửa cache và không thay thế bước audit của benchmark.
    """
    folder = project_root / "cache/embeddings"
    meta = json.loads((folder / f"{split}.meta.json").read_text(encoding="utf-8"))
    if meta["annotation_sha256"] != file_sha256(annotation_file):
        raise ValueError("Partition khác với cache; dùng dữ liệu đã chạy baseline.")
    if meta["checkpoint_sha256"] != official_checkpoint()[1]:
        raise ValueError("Cache không dùng checkpoint OpenAI CLIP ViT-B/32.")
    paths = [r.path for r in records]
    embeddings = load_cache(folder, split, paths, meta)
    if embeddings is None:
        raise ValueError(f"Cache {split} thiếu hoặc không khớp; xem README để lấy lại artifact đã có.")
    return embeddings


def show_preprocessing(image, batch, preprocess, embedding):
    """Hoàn tác normalize chỉ để xem màu ảnh; tensor đưa vào CLIP không bị đổi."""
    import matplotlib.pyplot as plt
    import torch

    normalize = preprocess.transforms[-1]
    mean = torch.tensor(normalize.mean).view(3, 1, 1)
    std = torch.tensor(normalize.std).view(3, 1, 1)
    visible = (batch[0].cpu() * std + mean).clamp(0, 1).permute(1, 2, 0)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    axes[0].imshow(image)
    axes[0].set_title(f"Query gốc: {image.width} × {image.height}")
    axes[1].imshow(visible.numpy())
    axes[1].set_title("Resize/crop: 224 × 224\nHoàn tác normalize để xem màu")
    axes[0].axis("off")
    axes[1].axis("off")
    axes[2].plot(embedding[0], linewidth=0.7)
    axes[2].set(title="Embedding: 512 số", xlabel="Vị trí trong vector", ylabel="Giá trị")
    fig.tight_layout()
    plt.show()


def show_top5(query, gallery, result, data_root: Path, *, row=0, title="", positive=None):
    """Vẽ query, top-5 và positive đã lưu (nếu có); không tính lại retrieval."""
    import matplotlib.pyplot as plt
    from PIL import Image

    panels = [(query.path, f"Query\n{query.item_id}", "black")]
    for rank, (j, score) in enumerate(zip(result["top_idx"][row], result["top_scores"][row]), 1):
        candidate = gallery[int(j)]
        correct = candidate.item_id == query.item_id
        label = f"#{rank} · {'ĐÚNG' if correct else 'SAI'}\n{candidate.item_id}\ncosine = {score:.4f}"
        panels.append((candidate.path, label, "green" if correct else "red"))
    if positive is not None:
        rank = int(result["first_pos_rank"][row])
        panels.append((positive["path"], f"Sản phẩm đúng, hạng {rank}\n{query.item_id}\ncosine = {positive['score']:.4f}", "blue"))
    fig, axes = plt.subplots(1, len(panels), figsize=(2.25 * len(panels), 3.6))
    for ax, (path, label, color) in zip(axes, panels):
        with Image.open(data_root / path) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_title(label, fontsize=8, color=color)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(2)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    plt.show()
