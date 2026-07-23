"""Trainer：封装训练 / 验证循环、早停、checkpoint 保存、CSV 日志、训练曲线。

用法：
    trainer = Trainer(model=..., device=..., train_loader=..., val_loader=...,
                      criterion=..., optimizer=..., scheduler=..., ...)
    trainer.train()

train() 按阶段顺序执行：
    初始化输出 → 逐 epoch（训练 → 验证 → 记录 → 保存 → 早停判断）→ 画训练曲线
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from metrics import batch_dice_score
from utils.checkpoint import save_checkpoint
from utils.logger import append_metrics_row, init_metrics_csv
from utils.visualize import plot_training_curve


class Trainer:
    def __init__(self, model: nn.Module, device: torch.device,
                 train_loader: DataLoader, val_loader: DataLoader,
                 criterion: nn.Module, optimizer: optim.Optimizer,
                 scheduler: optim.lr_scheduler.ReduceLROnPlateau | None = None,
                 epochs: int = 50,
                 early_stop_patience: int = 10,
                 early_stop_min_delta: float = 1.0e-4,
                 checkpoint_dir: Path = Path("checkpoints"),
                 log_dir: Path = Path("logs"),
                 output_dir: Path = Path("outputs"),
                 run_config: dict | None = None) -> None:
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.early_stop_patience = early_stop_patience
        self.early_stop_min_delta = early_stop_min_delta

        # ---------- 输出路径 ----------
        self.checkpoint_dir = Path(checkpoint_dir)
        self.best_model_path = self.checkpoint_dir / "best.pth"
        self.last_model_path = self.checkpoint_dir / "last.pth"
        self.log_dir = Path(log_dir)
        self.metrics_csv = self.log_dir / "metrics.csv"
        self.training_curve_png = Path(output_dir) / "plots" / "training_curve.png"

        # AMP 混合精度：仅 CUDA 下启用，省显存加速训练
        self.use_cuda = device.type == "cuda"
        self.amp_enabled = self.use_cuda
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

        # 本次运行的完整配置，随 checkpoint 一起保存，便于复现
        self.run_config = dict(run_config or {})
        self.run_config["amp"] = self.amp_enabled

    def train(self) -> None:
        # ========== 阶段 1：初始化输出（checkpoint 目录 / CSV 日志） ==========
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        init_metrics_csv(self.metrics_csv)

        best_dice = float("-inf")
        epochs_without_improvement = 0

        # ========== 阶段 2：逐 epoch 训练 ==========
        for epoch in range(1, self.epochs + 1):
            # ---------- 2.1 训练：前向 → 损失 → 反向 → 更新 ----------
            average_train_loss = self._train_one_epoch(epoch)

            # ---------- 2.2 验证：冻结权重，计算 val_loss 与 val_dice ----------
            average_val_loss, average_val_dice = self._validate(epoch)
            current_learning_rate = self.optimizer.param_groups[0]["lr"]

            # ---------- 2.3 记录：控制台 + CSV ----------
            print(
                f"Epoch {epoch:3d} | Train Loss: {average_train_loss:.4f} | "
                f"Val Loss: {average_val_loss:.4f} | Val Dice: {average_val_dice:.4f} | "
                f"LR: {current_learning_rate:.2e}"
            )
            append_metrics_row(
                self.metrics_csv, [epoch, f"{average_train_loss:.6f}", f"{average_val_loss:.6f}",
                                   f"{average_val_dice:.6f}", f"{current_learning_rate:.6e}"]
            )

            # ---------- 2.4 保存 checkpoint ----------
            # last.pth 每个 epoch 都覆盖保存；best.pth 只在 val_dice 创新高时保存
            state = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
                "best_dice": best_dice,
                "config": self.run_config,
            }
            save_checkpoint(state, self.last_model_path)

            # ---------- 2.5 早停判断 / 最佳模型 ----------
            if average_val_dice > best_dice + self.early_stop_min_delta:
                best_dice = average_val_dice
                epochs_without_improvement = 0
                state["best_dice"] = best_dice
                save_checkpoint(state, self.best_model_path)
                print(f"   保存最佳模型！当前最佳 Dice = {best_dice:.4f}")
            else:
                epochs_without_improvement += 1
                print(
                    f"   验证 Dice 未提升，早停计数: {epochs_without_improvement}/{self.early_stop_patience}"
                )

            # 学习率调度器同样监控 val_dice
            if self.scheduler is not None:
                self.scheduler.step(average_val_dice)

            if epochs_without_improvement >= self.early_stop_patience:
                print(f"   触发早停：连续 {self.early_stop_patience} 个 epoch 验证 Dice 未提升。")
                break

        # ========== 阶段 3：收尾（打印结果 + 画训练曲线） ==========
        print("\n训练完成！")
        print(f"最佳验证 Dice 分数: {best_dice:.4f}")
        print(f"模型已保存至: {self.best_model_path}")

        plot_training_curve(self.metrics_csv, self.training_curve_png)
        if self.training_curve_png.exists():
            print(f"训练曲线已保存至: {self.training_curve_png}")

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        train_loss_sum = 0.0
        number_train_batches = 0
        for images, masks in tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} [Train]", leave=False):
            images = images.to(self.device, non_blocking=self.use_cuda)
            masks = masks.to(self.device, non_blocking=self.use_cuda)

            self.optimizer.zero_grad(set_to_none=True)
            # autocast 内自动用半精度做前向与损失计算
            with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                logits = self.model(images)
                loss = self.criterion(logits, masks)
            # GradScaler 先放大 loss 防半精度下梯度下溢，再反向、更新
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            train_loss_sum += loss.item()
            number_train_batches += 1

        return train_loss_sum / max(number_train_batches, 1)

    def _validate(self, epoch: int) -> tuple[float, float]:
        self.model.eval()
        val_loss_sum = 0.0
        val_dice_sum = 0.0
        number_val_batches = 0
        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.epochs} [Val]", leave=False):
                images = images.to(self.device, non_blocking=self.use_cuda)
                masks = masks.to(self.device, non_blocking=self.use_cuda)
                with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                    logits = self.model(images)
                    loss = self.criterion(logits, masks)
                val_loss_sum += loss.item()
                # dice 用 float32 计算，避免半精度累计误差
                val_dice_sum += batch_dice_score(logits.float(), masks)
                number_val_batches += 1

        average_val_loss = val_loss_sum / max(number_val_batches, 1)
        average_val_dice = val_dice_sum / max(number_val_batches, 1)
        return average_val_loss, average_val_dice
