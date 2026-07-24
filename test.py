"""测试评估入口：python test.py [--config configs/train.yaml]

evaluate() 内是完整的评估流程，按阶段顺序写在一个函数里：
    加载数据 → 加载模型 → 逐样本推理（算指标 + 存三联图）→ 写四份报告 → 复制好例/错例
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# datasets/ 目录与 HuggingFace datasets 库同名，故把该目录单独加入 sys.path，
# 本地数据集模块以 `from chahu import ...` 方式导入。
sys.path.insert(0, str(ROOT / "datasets"))

import torch
from datasets import load_dataset
from tqdm import tqdm

from chahu import ChaHuTestDataset, build_dataloader, collate
from data_utils import build_id_to_index, prepare_splits
from metrics import (
    aggregate,
    bucket_aggregate,
    compute_all,
    fg_ratio_bucket,
    viewpoint_bucket,
)
from unet import UNet, ResNetUNet
from utils.checkpoint import load_checkpoint
from utils.common import get_device, load_config
from utils.visualize import save_triplet

# 好例 / 错例各挑几张
NUMBER_EXTREME_CASES = 5

METRIC_KEYS = ("dice", "iou", "precision", "recall", "pixel_accuracy")


def evaluate(cfg: dict) -> None:
    dataset_config = cfg["dataset"]
    path_config = cfg["paths"]
    model_config = cfg.get("model", {})
    use_attention = model_config.get("attention", False)
    backbone = model_config.get("backbone", "unet")
    pretrained = model_config.get("pretrained", True)

    # ---------- 数据集参数 ----------
    dataset_name = dataset_config["name"]
    dataset_split = dataset_config["split"]
    seed = dataset_config["seed"]
    image_size = dataset_config["image_size"]
    mask_bin_threshold = dataset_config["mask_bin_threshold"]

    # ---------- 输入 / 输出路径 ----------
    checkpoint_path = Path(path_config["checkpoint_dir"]) / "best.pth"
    test_results_dir = Path(path_config["output_dir"]) / "test_results"
    triplet_dir = test_results_dir / "triplet"
    best_cases_dir = test_results_dir / "best_cases"
    worst_cases_dir = test_results_dir / "worst_cases"
    report_dir = Path(path_config["output_dir"]) / "reports"

    print("启动 U-Net 二元分割测试...")

    # ========== 阶段 1：加载数据集，取出测试集划分 ==========
    print(f"加载数据集 {dataset_name} (split={dataset_split}) ...")
    dataset = load_dataset(dataset_name, split=dataset_split)
    print(f"数据集加载完成，共 {len(dataset)} 张图片")

    # 划分文件已在训练时生成，这里直接读盘，保证与训练看到同一份 test 集
    _, _, test_ids = prepare_splits(dataset, seed=seed)
    id_to_index = build_id_to_index(dataset)

    test_set = ChaHuTestDataset(
        dataset, test_ids, id_to_index,
        img_size=image_size, mask_bin_threshold=mask_bin_threshold,
    )
    # batch_size=1 + 自定义 collate：因为要同时取回 PIL 原图做可视化
    test_loader = build_dataloader(test_set, batch_size=1, shuffle=False, collate_fn=collate)
    print(f"测试集 {len(test_set)} 张")

    # ========== 阶段 2：构建模型并加载 best.pth ==========
    device = get_device()
    print(f"使用设备: {device}")

    if backbone == "resnet34":
        model = ResNetUNet(n_channels=3, n_classes=1, attention=use_attention, pretrained=pretrained).to(device)
        print(f"模型: ResNetUNet (encoder=resnet34, pretrained={pretrained}, attention={use_attention})")
    else:
        model = UNet(n_channels=3, n_classes=1, bilinear=True, attention=use_attention).to(device)
        print(f"模型: UNet (attention={use_attention})")
    checkpoint = load_checkpoint(model, checkpoint_path, device)
    if checkpoint:
        epoch = checkpoint.get("epoch")
        best_dice = checkpoint.get("best_dice")
        if epoch is not None and best_dice is not None:
            print(f"已加载模型: {checkpoint_path} | epoch={epoch} | best_dice={best_dice:.4f}")
        else:
            print(f"已加载模型: {checkpoint_path}")

    triplet_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # ========== 阶段 3：逐样本推理，计算指标并保存三联图 ==========
    model.eval()
    records: list[dict] = []   # 每张图一条记录：指标 + 元数据

    with torch.no_grad():
        for images, masks, metas in tqdm(test_loader, desc="测试中"):
            images = images.to(device)
            # 前向 → sigmoid 概率 → 0.5 阈值二值化
            logits = model(images)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > 0.5).float()

            for batch_index in range(predictions.size(0)):
                meta = metas[batch_index]
                prediction_np = predictions[batch_index, 0].detach().cpu().numpy()
                target_np = masks[batch_index, 0].detach().cpu().numpy()

                # 一次算全 Dice / IoU / Precision / Recall / PixelAcc
                metric_dict = compute_all(prediction_np, target_np)
                # 前景占比与视角，用于后面的分桶报告
                foreground_ratio = float(target_np.mean())
                record = {
                    "id": meta["id"],
                    "viewpoint": meta["viewpoint"],
                    "viewpoint_bucket": viewpoint_bucket(meta["viewpoint"]),
                    "fg_bucket": fg_ratio_bucket(foreground_ratio),
                    "fg_ratio": round(foreground_ratio, 6),
                    **{k: round(v, 6) for k, v in metric_dict.items()},
                }
                records.append(record)

                # 「原图 | GT | 预测叠加」三联图，供人工检查
                save_triplet(
                    image_pil=meta["image_lb_pil"],
                    gt_mask_pil=meta["mask_lb_pil"],
                    pred_binary=prediction_np,
                    save_path=triplet_dir / f"{meta['id']}_triplet.png",
                )

    if not records:
        print("测试集为空，跳过报告生成。")
        return

    # ========== 阶段 4：写四份指标报告 ==========
    # 4.1 整体报告（json）：每个指标的 mean / std / median
    overall = aggregate(records)
    overall_path = report_dir / "metrics_overall.json"
    with overall_path.open("w", encoding="utf-8") as f:
        json.dump({"n_samples": len(records), "metrics": overall}, f, ensure_ascii=False, indent=2)

    # 4.2 / 4.3 分桶报告（csv）：按视角、按前景占比各一份
    for bucket_key, report_name in (
        ("viewpoint_bucket", "metrics_by_viewpoint.csv"),
        ("fg_bucket", "metrics_by_fg.csv"),
    ):
        bucket_stats = bucket_aggregate(records, bucket_key)
        with (report_dir / report_name).open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            # 表头：bucket, n_samples, <指标>_mean, <指标>_std, ...
            header = ["bucket", "n_samples"]
            for key in METRIC_KEYS:
                header.extend([f"{key}_mean", f"{key}_std"])
            writer.writerow(header)
            for bucket_name, stats in sorted(bucket_stats.items()):
                number_samples = sum(
                    1 for r in records if str(r.get(bucket_key, "未知")) == bucket_name
                )
                row = [bucket_name, number_samples]
                for key in METRIC_KEYS:
                    row.append(f"{stats[key]['mean']:.4f}")
                    row.append(f"{stats[key]['std']:.4f}")
                writer.writerow(row)

    # 4.4 逐样本明细（csv）：方便后续用 pandas 等做错例分析
    per_sample_path = report_dir / "metrics_per_sample.csv"
    fieldnames = [
        "id", "viewpoint", "viewpoint_bucket", "fg_bucket", "fg_ratio",
        "dice", "iou", "precision", "recall", "pixel_accuracy",
    ]
    with per_sample_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({k: record.get(k, "") for k in fieldnames})

    # ========== 阶段 5：复制 Dice 最高 / 最低的三联图到 best/worst 目录 ==========
    best_cases_dir.mkdir(parents=True, exist_ok=True)
    worst_cases_dir.mkdir(parents=True, exist_ok=True)

    sorted_by_dice = sorted(records, key=lambda r: r["dice"], reverse=True)
    best_records = sorted_by_dice[:NUMBER_EXTREME_CASES]
    worst_records = sorted_by_dice[-NUMBER_EXTREME_CASES:][::-1]   # 最差的排最前
    for record in best_records:
        source = triplet_dir / f"{record['id']}_triplet.png"
        if source.exists():
            shutil.copyfile(source, best_cases_dir / source.name)
    for record in worst_records:
        source = triplet_dir / f"{record['id']}_triplet.png"
        if source.exists():
            shutil.copyfile(source, worst_cases_dir / source.name)

    # ========== 阶段 6：控制台汇总 ==========
    print("\n" + "=" * 60)
    print(f"测试完成！测试集样本数: {len(records)}")
    for key in METRIC_KEYS:
        stats = overall[key]
        print(f"  {key:<14} mean={stats['mean']:.4f} | std={stats['std']:.4f} | median={stats['median']:.4f}")
    print(f"三联图已保存到 {triplet_dir}/")
    print(f"指标报告已保存到 {report_dir}/")
    print(f"好例 / 错例已分别复制到 {best_cases_dir}/ 与 {worst_cases_dir}/")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    evaluate(cfg)


if __name__ == "__main__":
    main()
