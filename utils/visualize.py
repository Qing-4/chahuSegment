"""可视化：训练曲线、预测叠加与三联图。"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image


def plot_training_curve(csv_path: Path, png_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过训练曲线绘制。")
        return

    epochs, train_loss, val_loss, val_dice = [], [], [], []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            val_dice.append(float(row["val_dice"]))

    if not epochs:
        return

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, train_loss, label="train_loss", color="tab:blue")
    ax1.plot(epochs, val_loss, label="val_loss", color="tab:orange")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_dice, label="val_dice", color="tab:green")
    ax2.set_ylabel("dice")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def overlay_mask(original_pil: Image.Image, pred_binary: np.ndarray, alpha: float = 0.5,
                 color=(255, 0, 0)) -> Image.Image:
    """在 letterbox 后的原图上叠加预测 mask 的红色半透明蒙版。"""
    if original_pil.mode != "RGB":
        original_pil = original_pil.convert("RGB")
    base = np.array(original_pil).astype(np.float32)
    mask = (pred_binary > 0.5).astype(np.float32)[..., None]
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    overlayed = base * (1.0 - alpha * mask) + color_arr * (alpha * mask)
    return Image.fromarray(overlayed.clip(0, 255).astype(np.uint8))


def save_triplet(
    image_pil: Image.Image,
    gt_mask_pil: Image.Image,
    pred_binary: np.ndarray,
    save_path: Path,
) -> None:
    """把「原图 | GT | 预测叠加」并排拼成一张三联图保存。"""
    w, h = image_pil.size
    overlay = overlay_mask(image_pil, pred_binary)
    gt_rgb = gt_mask_pil.convert("RGB")

    canvas = Image.new("RGB", (w * 3, h), color=(0, 0, 0))
    canvas.paste(image_pil, (0, 0))
    canvas.paste(gt_rgb, (w, 0))
    canvas.paste(overlay, (w * 2, 0))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path)
