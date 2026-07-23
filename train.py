"""训练入口：python train.py [--smoke] [--config configs/train.yaml]

只负责组装训练所需组件：
    读取配置 → 设置随机种子 → 创建数据集和 DataLoader → 创建模型 / 损失函数 / 优化器 / 调度器
    → 创建 Trainer → trainer.train()
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.optim as optim

ROOT = Path(__file__).resolve().parent
# datasets/ 目录与 HuggingFace datasets 库同名，故把该目录单独加入 sys.path，
# 本地数据集模块以 `from chahu import ...` 方式导入。
sys.path.insert(0, str(ROOT / "datasets"))

from datasets import load_dataset

from chahu import ChaHuBinaryDataset, build_dataloader
from data_utils import build_id_to_index, prepare_splits
from dice_loss import BCEDiceLoss
from unet import UNet, ResNetUNet
from trainer import Trainer
from utils.common import get_device, load_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--smoke", action="store_true", help="仅用 32 张样本跑 5 epoch 做管线自检")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--backbone", type=str, default=None, help="unet | resnet34")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ========== 1. 读取配置 ==========
    cfg = load_config(args.config)
    dataset_config = cfg["dataset"]
    train_config = cfg["train"]
    path_config = cfg["paths"]
    model_config = cfg.get("model", {})
    use_attention = model_config.get("attention", False)
    backbone = args.backbone or model_config.get("backbone", "unet")
    pretrained = model_config.get("pretrained", True)

    # 训练超参（命令行传入的值优先于 yaml）
    epochs = args.epochs or train_config["epochs"]
    batch_size = args.batch_size or train_config["batch_size"]
    learning_rate = args.lr or train_config["learning_rate"]
    weight_decay = args.weight_decay if args.weight_decay is not None else train_config.get("weight_decay", 0.0)
    seed = dataset_config["seed"]
    image_size = dataset_config["image_size"]
    mask_bin_threshold = dataset_config["mask_bin_threshold"]

    print("启动 U-Net 二元分割训练...")

    # ========== 2. 设置随机种子 ==========
    set_seed(seed)

    # ========== 3. 创建数据集和 DataLoader ==========
    dataset_name = dataset_config["name"]
    dataset_split = dataset_config["split"]
    print(f"加载数据集 {dataset_name} (split={dataset_split}) ...")
    dataset = load_dataset(dataset_name, split=dataset_split)
    print(f"数据集加载完成，共 {len(dataset)} 张图片")

    # 审计 + 固定种子划分（有缓存文件则直接读，保证与 test.py 一致）
    train_ids, val_ids, _ = prepare_splits(dataset, seed=seed)
    id_to_index = build_id_to_index(dataset)

    # smoke 模式：只取少量样本快速验证整条管线能跑通
    if args.smoke:
        train_ids = train_ids[:32]
        val_ids = val_ids[:8]
        epochs = min(2, epochs)
        print(f"[SMOKE] 仅用 train={len(train_ids)} / val={len(val_ids)} / epochs={epochs}")

    # 训练集启用增强，验证集不增强
    train_set = ChaHuBinaryDataset(
        dataset, train_ids, id_to_index, img_size=image_size,
        train=True, mask_bin_threshold=mask_bin_threshold,
    )
    val_set = ChaHuBinaryDataset(
        dataset, val_ids, id_to_index, img_size=image_size,
        train=False, mask_bin_threshold=mask_bin_threshold,
    )
    print(f"训练集 {len(train_set)} 张 | 验证集 {len(val_set)} 张")

    use_cuda = torch.cuda.is_available()
    train_loader = build_dataloader(train_set, batch_size, shuffle=True, pin_memory=use_cuda)
    val_loader = build_dataloader(val_set, batch_size, shuffle=False, pin_memory=use_cuda)

    # ========== 4. 创建模型 ==========
    device = get_device()
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        print(f"使用设备: {device} | GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"使用设备: {device}")

    if backbone == "resnet34":
        model = ResNetUNet(n_channels=3, n_classes=1, attention=use_attention, pretrained=pretrained).to(device)
        print(f"模型: ResNetUNet (encoder=resnet34, pretrained={pretrained}, attention={use_attention})")
    else:
        model = UNet(n_channels=3, n_classes=1, bilinear=True, attention=use_attention).to(device)
        print(f"模型: UNet (attention={use_attention})")

    # ========== 5. 创建损失函数 ==========
    criterion = BCEDiceLoss()

    # ========== 6. 创建优化器 ==========
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # ========== 7. 创建学习率调度器（可选） ==========
    # 监控 val_dice（mode=max）：连续 plateau_patience 个 epoch 未提升则学习率乘 plateau_factor
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=train_config["plateau_factor"],
        patience=train_config["plateau_patience"], min_lr=train_config["min_lr"],
    )

    # 本次运行的完整配置，随 checkpoint 一起保存，便于复现
    run_config = {
        "dataset_name": dataset_name,
        "dataset_split": dataset_split,
        "image_size": image_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "backbone": backbone,
        "pretrained": pretrained,
        "seed": seed,
        "early_stop_patience": train_config["early_stop_patience"],
        "early_stop_min_delta": train_config["early_stop_min_delta"],
        "plateau_patience": train_config["plateau_patience"],
        "plateau_factor": train_config["plateau_factor"],
        "min_lr": train_config["min_lr"],
        "mask_bin_threshold": mask_bin_threshold,
        "loss": "BCE + Dice",
        "model": ("ResNetUNet" if backbone == "resnet34" else "UNet") + ("+Attn" if use_attention else ""),
        "attention": use_attention,
        "smoke": args.smoke,
    }

    # ========== 8. 创建 Trainer 并启动训练 ==========
    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        early_stop_patience=train_config["early_stop_patience"],
        early_stop_min_delta=train_config["early_stop_min_delta"],
        checkpoint_dir=Path(path_config["checkpoint_dir"]),
        log_dir=Path(path_config["log_dir"]),
        output_dir=Path(path_config["output_dir"]),
        run_config=run_config,
    )
    trainer.train()


if __name__ == "__main__":
    main()
