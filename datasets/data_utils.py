"""数据准备工具。

本模块负责三件事：
1. audit_dataset  ：一次性遍历 HuggingFace 数据集，剔除异常样本，产出 valid_ids.txt；
2. make_splits    ：基于固定随机种子，把 valid ids 划分为 train/val/test 并落盘；
3. letterbox 工具 ：等比缩放 + padding，保持 512x512 输入而不破坏长宽比。

直接运行本文件即可触发审计 + 划分：
    python datasets/data_utils.py
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# ---------- 输出文件 ----------
SPLIT_DIR = Path("splits")
VALID_IDS_FILE = SPLIT_DIR / "valid_ids.txt"
TRAIN_IDS_FILE = SPLIT_DIR / "train.txt"
VAL_IDS_FILE = SPLIT_DIR / "val.txt"
TEST_IDS_FILE = SPLIT_DIR / "test.txt"

# ---------- 划分参数 ----------
DEFAULT_SEED = 42
DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.15

# ---------- 审计参数 ----------
MASK_FOREGROUND_THRESHOLD = 128   # mask 前景判定阈值（灰度 ≥128 视为前景）
MAX_FOREGROUND_RATIO = 0.95       # 前景占比超过该值视为极端特写，剔除


def audit_dataset(hf_dataset, force: bool = False) -> list[str]:
    """遍历整个 HF 数据集，剔除异常样本，把保留的 id 写入 valid_ids.txt。

    剔除规则：
    - image 与 mask 分辨率不一致（缩放会引入几何错位）；
    - 二值化后前景占比 > MAX_FOREGROUND_RATIO（极端特写，主导 Dice loss）。
    """
    # ---------- 已审计过：直接读缓存文件，避免重复扫描 ----------
    if not force and VALID_IDS_FILE.exists():
        text = VALID_IDS_FILE.read_text(encoding="utf-8")
        return [line.strip() for line in text.splitlines() if line.strip()]

    # ---------- 第一次运行：逐张检查并统计剔除原因 ----------
    print("第一次运行：审计数据集（约需几分钟）...")
    valid_ids: list[str] = []
    number_size_mismatch = 0
    number_extreme_foreground = 0

    for index in tqdm(range(len(hf_dataset)), desc="审计"):
        example = hf_dataset[index]
        image = example["image"]
        mask = example["mask"]

        # 检查 1：image 与 mask 分辨率必须一致，否则缩放对齐会引入几何错位
        if image.size != mask.size:
            number_size_mismatch += 1
            continue

        # 检查 2：前景占比不能过高。mask 转灰度后按阈值判前景，
        # 占比 >95% 基本是壶体特写，会主导 Dice 损失，剔除。
        mask_gray = np.array(mask.convert("L"))
        foreground_ratio = float((mask_gray >= MASK_FOREGROUND_THRESHOLD).mean())
        if foreground_ratio > MAX_FOREGROUND_RATIO:
            number_extreme_foreground += 1
            continue

        valid_ids.append(example["id"])

    # ---------- 落盘，后续运行直接复用 ----------
    VALID_IDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    VALID_IDS_FILE.write_text("\n".join(valid_ids) + "\n", encoding="utf-8")
    print(
        f"审计完成：保留 {len(valid_ids)} / {len(hf_dataset)} 张 | "
        f"剔除尺寸不一致 {number_size_mismatch} 张 | 剔除极端特写 {number_extreme_foreground} 张"
    )
    return valid_ids


def make_splits(
    valid_ids: list[str],
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    seed: int = DEFAULT_SEED,
    force: bool = False,
) -> tuple[list[str], list[str], list[str]]:
    """把 valid_ids 划分为 train / val / test 并写入 splits/*.txt。

    第二次及之后调用：直接从磁盘读，保证 train.py 与 test.py 看到完全一致的划分。
    """
    # ---------- 已划分过：直接读三个 id 文件 ----------
    split_files_exist = (
        TRAIN_IDS_FILE.exists() and VAL_IDS_FILE.exists() and TEST_IDS_FILE.exists()
    )
    if not force and split_files_exist:
        splits = []
        for path in (TRAIN_IDS_FILE, VAL_IDS_FILE, TEST_IDS_FILE):
            text = path.read_text(encoding="utf-8")
            splits.append([line.strip() for line in text.splitlines() if line.strip()])
        return splits[0], splits[1], splits[2]

    # ---------- 第一次划分：固定种子打乱后按比例切分 ----------
    ids = list(valid_ids)
    random.Random(seed).shuffle(ids)

    number_train = int(len(ids) * train_ratio)
    number_val = int(len(ids) * val_ratio)
    train_ids = ids[:number_train]
    val_ids = ids[number_train : number_train + number_val]
    test_ids = ids[number_train + number_val :]   # 剩余全部归 test

    # ---------- 落盘，保证不同脚本看到同一份划分 ----------
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_IDS_FILE.write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    VAL_IDS_FILE.write_text("\n".join(val_ids) + "\n", encoding="utf-8")
    TEST_IDS_FILE.write_text("\n".join(test_ids) + "\n", encoding="utf-8")
    print(f"划分完成：train={len(train_ids)} | val={len(val_ids)} | test={len(test_ids)}")
    return train_ids, val_ids, test_ids


def prepare_splits(hf_dataset, seed: int = DEFAULT_SEED, force: bool = False):
    """一次性准备 valid_ids 与三份 split，返回 (train_ids, val_ids, test_ids)。"""
    valid_ids = audit_dataset(hf_dataset, force=force)
    return make_splits(valid_ids, seed=seed, force=force)


def build_id_to_index(hf_dataset) -> dict[str, int]:
    """从 HF 数据集构建 id -> 全局索引的映射。"""
    return {sample_id: index for index, sample_id in enumerate(hf_dataset["id"])}


def letterbox_image(
    img: Image.Image, target_size: int, fill: int = 0, resample: int = Image.BILINEAR
) -> Image.Image:
    """等比缩放后再 padding 到 target_size×target_size，避免长宽比畸变。"""
    # 按长边等比缩放到 target_size
    width, height = img.size
    scale = target_size / max(width, height)
    new_width, new_height = int(round(width * scale)), int(round(height * scale))
    resized = img.resize((new_width, new_height), resample=resample)

    # 贴到 target_size×target_size 的画布中央，四周对称补 fill 值
    canvas = Image.new(img.mode, (target_size, target_size), color=fill)
    padding_x = (target_size - new_width) // 2
    padding_y = (target_size - new_height) // 2
    canvas.paste(resized, (padding_x, padding_y))
    return canvas



def letterbox_pair(
    image: Image.Image, mask: Image.Image, target_size: int
) -> tuple[Image.Image, Image.Image]:
    """对 image 与 mask 同步做 letterbox。image 用 BILINEAR，mask 用 NEAREST。"""
    image_out = letterbox_image(image, target_size, fill=0, resample=Image.BILINEAR)
    mask_out = letterbox_image(mask, target_size, fill=0, resample=Image.NEAREST)
    return image_out, mask_out


if __name__ == "__main__":
    from datasets import load_dataset

    print("加载数据集 AGI-FBHC/ChaHu (split=CN) ...")
    dataset = load_dataset("AGI-FBHC/ChaHu", split="CN")
    prepare_splits(dataset, force=True)
    print(f"输出目录：{SPLIT_DIR.resolve()}")
