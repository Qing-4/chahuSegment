"""二元分割评估指标：Dice / IoU / Precision / Recall / Pixel Accuracy。

每个指标函数都接受 pred / target（torch.Tensor 或 numpy 数组），
内部先统一转成 0/1 的二维 numpy 数组再计算，函数之间互不依赖。
"""

from __future__ import annotations

import numpy as np
import torch

# 极小平滑项，避免分母为 0（全背景样本）
EPS = 1e-6


def dice_score(pred, target) -> float:
    """Dice = 2*|P∩T| / (|P|+|T|)，衡量预测与 GT 的重叠程度。"""
    # 统一转为 0/1 二维 numpy 数组：tensor→numpy，去掉 batch/通道维，0.5 阈值二值化
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    pred = np.asarray(pred)
    target = np.asarray(target)
    while pred.ndim > 2:
        pred = pred[0]
    while target.ndim > 2:
        target = target[0]
    pred = (pred > 0.5).astype(np.uint8)
    target = (target > 0.5).astype(np.uint8)

    intersection = (pred * target).sum()
    return float((2.0 * intersection + EPS) / (pred.sum() + target.sum() + EPS))


def iou_score(pred, target) -> float:
    """IoU = |P∩T| / |P∪T|，比 Dice 对错误更敏感。"""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    pred = np.asarray(pred)
    target = np.asarray(target)
    while pred.ndim > 2:
        pred = pred[0]
    while target.ndim > 2:
        target = target[0]
    pred = (pred > 0.5).astype(np.uint8)
    target = (target > 0.5).astype(np.uint8)

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return float((intersection + EPS) / (union + EPS))


def precision_score(pred, target) -> float:
    """Precision = TP / (TP+FP)，预测为前景的像素里有多少是对的。"""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    pred = np.asarray(pred)
    target = np.asarray(target)
    while pred.ndim > 2:
        pred = pred[0]
    while target.ndim > 2:
        target = target[0]
    pred = (pred > 0.5).astype(np.uint8)
    target = (target > 0.5).astype(np.uint8)

    true_positive = (pred * target).sum()
    false_positive = (pred * (1 - target)).sum()
    return float((true_positive + EPS) / (true_positive + false_positive + EPS))


def recall_score(pred, target) -> float:
    """Recall = TP / (TP+FN)，GT 前景里有多少被找回来了。"""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    pred = np.asarray(pred)
    target = np.asarray(target)
    while pred.ndim > 2:
        pred = pred[0]
    while target.ndim > 2:
        target = target[0]
    pred = (pred > 0.5).astype(np.uint8)
    target = (target > 0.5).astype(np.uint8)

    true_positive = (pred * target).sum()
    false_negative = ((1 - pred) * target).sum()
    return float((true_positive + EPS) / (true_positive + false_negative + EPS))


def pixel_accuracy(pred, target) -> float:
    """全图像素分类正确率（前景占比小时数值会偏高，仅作参考）。"""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    pred = np.asarray(pred)
    target = np.asarray(target)
    while pred.ndim > 2:
        pred = pred[0]
    while target.ndim > 2:
        target = target[0]
    pred = (pred > 0.5).astype(np.uint8)
    target = (target > 0.5).astype(np.uint8)

    return float((pred == target).mean())


def batch_dice_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """按 batch 计算平均 Dice（训练验证阶段用，直接在 GPU 张量上算，不落 numpy）。"""
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities > threshold).float()
    # 对每张图分别在 H、W 两个维度上求和，得到逐图的 Dice，再对 batch 取平均
    intersection = (predictions * targets).sum(dim=(2, 3))
    union = predictions.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice_per_image = (2.0 * intersection + 1e-6) / (union + 1e-6)
    return float(dice_per_image.mean().item())


def compute_all(pred, target) -> dict[str, float]:
    """一次性计算所有指标。"""
    return {
        "dice": dice_score(pred, target),
        "iou": iou_score(pred, target),
        "precision": precision_score(pred, target),
        "recall": recall_score(pred, target),
        "pixel_accuracy": pixel_accuracy(pred, target),
    }


# ===================== 聚合 =====================
METRIC_KEYS: tuple[str, ...] = ("dice", "iou", "precision", "recall", "pixel_accuracy")


def aggregate(records: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """对一组指标记录返回每个指标的 mean / std / median。"""
    result: dict[str, dict[str, float]] = {}
    for key in METRIC_KEYS:
        values = np.array([record[key] for record in records if key in record], dtype=np.float64)
        if values.size == 0:
            result[key] = {"mean": float("nan"), "std": float("nan"), "median": float("nan")}
        else:
            result[key] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "median": float(np.median(values)),
            }
    return result


def bucket_aggregate(
    records: list[dict],
    bucket_key: str,
) -> dict[str, dict[str, dict[str, float]]]:
    """按 record[bucket_key] 分桶后聚合每个桶的指标。"""
    buckets: dict[str, list[dict]] = {}
    for record in records:
        key = record.get(bucket_key, "未知")
        buckets.setdefault(str(key), []).append(record)
    return {bucket_name: aggregate(items) for bucket_name, items in buckets.items()}


def fg_ratio_bucket(fg_ratio: float) -> str:
    """根据前景占比分三桶：small / medium / large。"""
    if fg_ratio < 0.10:
        return "small (<10%)"
    if fg_ratio < 0.40:
        return "medium (10-40%)"
    return "large (>=40%)"


def viewpoint_bucket(viewpoint: str | None) -> str:
    """正面平视 vs 其它两桶，缓解视角长尾。"""
    if viewpoint and "正面平视" in viewpoint:
        return "frontal_level"
    return "other_views"
