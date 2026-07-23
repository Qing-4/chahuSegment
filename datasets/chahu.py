"""ChaHu 数据集封装。

包含：
- ChaHuBinaryDataset：训练 / 验证用，返回 (image_tensor, mask_tensor)，训练侧带增强；
- ChaHuTestDataset：测试用，额外返回 letterbox 后的 PIL 原图、GT mask 与元数据；
- build_dataloader / collate：DataLoader 构建工具。

两个 Dataset 的预处理流程一致：
    增强（仅训练）→ letterbox 到 img_size → ToTensor + ImageNet 归一化
    mask 按阈值（默认 ≥128）二值化，排除抗锯齿软边。
"""
from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from data_utils import letterbox_pair
from utils.common import IMAGENET_MEAN, IMAGENET_STD

IMAGE_SIZE = 512
MASK_BIN_THRESHOLD = 128

# 训练增强参数
HORIZONTAL_FLIP_PROBABILITY = 0.5
ROTATE_MAX_DEGREE = 15.0
COLOR_JITTER_BRIGHTNESS = 0.2
COLOR_JITTER_CONTRAST = 0.2


class ChaHuBinaryDataset(Dataset):
    """训练 / 验证数据集：基于 id 列表 + HF 数据集索引访问。

    train=True 时启用增强（翻转 / 旋转 / 颜色抖动），验证侧不增强。
    """

    def __init__(
        self,
        hf_dataset,
        ids: list[str],
        id_to_index: dict[str, int],
        img_size: int = IMAGE_SIZE,
        train: bool = False,
        mask_bin_threshold: int = MASK_BIN_THRESHOLD,
    ) -> None:
        self.dataset = hf_dataset
        self.ids = list(ids)
        self.id_to_index = id_to_index
        self.img_size = img_size
        self.train = train
        self.mask_bin_threshold = mask_bin_threshold
        self.color_jitter = transforms.ColorJitter(
            brightness=COLOR_JITTER_BRIGHTNESS, contrast=COLOR_JITTER_CONTRAST
        )
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # ---------- 第 1 步：按 id 取出原始样本 ----------
        sample_id = self.ids[idx]
        example = self.dataset[self.id_to_index[sample_id]]
        image = example["image"].convert("RGB")   # 统一为 3 通道
        mask = example["mask"].convert("L")       # 统一为单通道灰度

        # ---------- 第 2 步：几何增强（仅训练集） ----------
        # image 与 mask 必须做完全相同的几何变换，否则像素对不上。
        if self.train:
            # 水平翻转 p=0.5
            if random.random() < HORIZONTAL_FLIP_PROBABILITY:
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)
            # 随机旋转 ±15°：image 用 BILINEAR 保持平滑；
            # mask 用 NEAREST，避免插值产生新的软边灰度值。
            angle = random.uniform(-ROTATE_MAX_DEGREE, ROTATE_MAX_DEGREE)
            image = transforms.functional.rotate(
                image, angle, interpolation=transforms.InterpolationMode.BILINEAR, fill=[0, 0, 0]
            )
            mask = transforms.functional.rotate(
                mask, angle, interpolation=transforms.InterpolationMode.NEAREST, fill=[0]
            )

        # ---------- 第 3 步：letterbox 等比缩放 + 补边到 img_size×img_size ----------
        # 不直接 Resize 是为了保持长宽比，避免壶体被压扁。
        image, mask = letterbox_pair(image, mask, self.img_size)

        # ---------- 第 4 步：颜色抖动（仅训练集，只作用于 image） ----------
        if self.train:
            image = self.color_jitter(image)

        # ---------- 第 5 步：转 tensor ----------
        # image：ToTensor 后按 ImageNet 均值方差归一化。
        image_tensor = self.normalize(self.to_tensor(image))
        # mask：按阈值（≥128）二值化为 0/1。数据集所有 mask 都带抗锯齿
        # 软边（0-255 渐变像素），不能用 >0 判前景。
        mask_binary = (np.array(mask, dtype=np.uint8) >= self.mask_bin_threshold).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0)  # (H, W) -> (1, H, W)

        return image_tensor, mask_tensor


class ChaHuTestDataset(Dataset):
    """测试数据集：除 tensor 外，同时返回 letterbox 后的 PIL 原图 / GT mask / 元数据，
    供三联图可视化与分桶分析使用。"""

    def __init__(
        self,
        hf_dataset,
        ids: list[str],
        id_to_index: dict[str, int],
        img_size: int = IMAGE_SIZE,
        mask_bin_threshold: int = MASK_BIN_THRESHOLD,
    ) -> None:
        self.dataset = hf_dataset
        self.ids = list(ids)
        self.id_to_index = id_to_index
        self.img_size = img_size
        self.mask_bin_threshold = mask_bin_threshold
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict:
        # ---------- 第 1 步：按 id 取出原始样本 ----------
        sample_id = self.ids[idx]
        example = self.dataset[self.id_to_index[sample_id]]
        image = example["image"].convert("RGB")
        mask = example["mask"].convert("L")

        # ---------- 第 2 步：letterbox（测试不做任何增强） ----------
        image_letterboxed, mask_letterboxed = letterbox_pair(image, mask, self.img_size)

        # ---------- 第 3 步：转 tensor（与训练集相同的归一化 / 二值化） ----------
        image_tensor = self.normalize(self.to_tensor(image_letterboxed))
        mask_binary = (
            np.array(mask_letterboxed, dtype=np.uint8) >= self.mask_bin_threshold
        ).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0)

        # PIL 原图 / GT 一并返回，供三联图可视化；viewpoint 供分桶统计。
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_lb_pil": image_letterboxed,
            "mask_lb_pil": mask_letterboxed,
            "id": sample_id,
            "viewpoint": example.get("viewpoint", "") or "",
        }


def collate(batch: list[dict]):
    """把 ChaHuTestDataset 的 dict 样本整理为 (images, masks, metas)。

    PIL 图像无法被默认 collate 堆叠，所以 tensor 与元数据分开返回。
    """
    images = torch.stack([item["image"] for item in batch])
    masks = torch.stack([item["mask"] for item in batch])
    metas = [
        {
            "id": item["id"],
            "viewpoint": item["viewpoint"],
            "image_lb_pil": item["image_lb_pil"],
            "mask_lb_pil": item["mask_lb_pil"],
        }
        for item in batch
    ]
    return images, masks, metas


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    pin_memory: bool = False,
    collate_fn=None,
    num_workers: int = 8,
) -> DataLoader:
    # 多进程加载 + 预取，避免 GPU 空等数据；num_workers=0 时退回单进程。
    extra = {}
    if num_workers > 0:
        extra = {"persistent_workers": True, "prefetch_factor": 4}
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn,
        **extra,
    )
