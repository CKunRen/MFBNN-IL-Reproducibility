"""
训练工具
Training Utilities
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from .model import MFBNN
from .loss import MFBNNLoss, NLLLoss


class MFBNNTrainer:
    """
    MFBNN训练器
    MFBNN Trainer

    支持：
    - 多保真度数据训练
    - 增量学习
    - 早停机制
    - 学习率调度
    """
    def __init__(self,
                 model: MFBNN,
                 lambda_kl: float = 0.1,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-4,
                 device: str = 'auto'):
        """
        Args:
            model: MFBNN模型
            lambda_kl: KL散度权重 (λ)
            learning_rate: 学习率
            weight_decay: 权重衰减
            device: 设备 ('auto', 'cuda', 'cpu')
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.lambda_kl = lambda_kl
        self.learning_rate = learning_rate

        # 损失函数
        self.criterion = MFBNNLoss(lambda_kl=lambda_kl)

        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # 学习率调度器
        self.scheduler = None

        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'mse_loss': [],
            'kl_loss': []
        }

    def set_scheduler(self,
                      scheduler_type: str = 'cosine',
                      T_max: int = 100,
                      step_size: int = 50,
                      gamma: float = 0.5):
        """
        设置学习率调度器

        Args:
            scheduler_type: 调度器类型 ('cosine', 'step', 'plateau')
            T_max: 余弦退火周期
            step_size: StepLR步长
            gamma: 学习率衰减因子
        """
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=T_max, eta_min=1e-6
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
        elif scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=gamma, patience=10
            )

    def train(self,
              x_hf: np.ndarray,
              y_hf: np.ndarray,
              x_lf_list: Optional[List[np.ndarray]] = None,
              y_lf_list: Optional[List[np.ndarray]] = None,
              num_epochs: int = 500,
              batch_size: int = 32,
              validation_split: float = 0.0,
              patience: int = 50,
              verbose: bool = True) -> Dict:
        """
        训练模型

        Args:
            x_hf: HF输入数据 [n_hf, input_dim]
            y_hf: HF目标数据 [n_hf, output_dim]
            x_lf_list: LF输入数据列表（可选）
            y_lf_list: LF目标数据列表（可选）
            num_epochs: 训练轮数
            batch_size: 批量大小
            validation_split: 验证集比例
            patience: 早停耐心值
            verbose: 是否打印训练信息

        Returns:
            history: 训练历史字典
        """
        # 转换为张量
        x_hf_tensor = torch.FloatTensor(x_hf)
        y_hf_tensor = torch.FloatTensor(y_hf).view(-1, 1) if y_hf.ndim == 1 else torch.FloatTensor(y_hf)

        # 创建数据加载器
        n_samples = len(x_hf)
        if validation_split > 0:
            n_val = int(n_samples * validation_split)
            n_train = n_samples - n_val
            indices = torch.randperm(n_samples)
            train_indices = indices[:n_train]
            val_indices = indices[n_train:]

            train_dataset = TensorDataset(x_hf_tensor[train_indices], y_hf_tensor[train_indices])
            val_dataset = TensorDataset(x_hf_tensor[val_indices], y_hf_tensor[val_indices])
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            n_train = n_samples
            train_dataset = TensorDataset(x_hf_tensor, y_hf_tensor)
            val_loader = None

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # 早停
        best_loss = float('inf')
        best_state = None
        patience_counter = 0

        # 训练循环
        for epoch in range(num_epochs):
            # 训练阶段
            self.model.train()
            epoch_loss = 0.0
            epoch_mse = 0.0
            epoch_kl = 0.0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()

                # 采样前向传播
                hf_output, lf_outputs = self.model.sample_forward(batch_x)

                # 计算损失
                kl = self.model.kl_divergence()
                loss_dict = self.criterion(
                    hf_pred=hf_output,
                    hf_target=batch_y,
                    kl_divergence=kl,
                    n_samples=n_train
                )

                loss = loss_dict['total_loss']
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_mse += loss_dict['mse_loss'].item()
                epoch_kl += loss_dict['kl_loss'].item()

            # 平均损失
            avg_loss = epoch_loss / len(train_loader)
            avg_mse = epoch_mse / len(train_loader)
            avg_kl = epoch_kl / len(train_loader)

            self.history['train_loss'].append(avg_loss)
            self.history['mse_loss'].append(avg_mse)
            self.history['kl_loss'].append(avg_kl)

            # 验证阶段
            if val_loader is not None:
                val_loss = self._validate(val_loader, n_train)
                self.history['val_loss'].append(val_loss)
                current_loss = val_loss
            else:
                current_loss = avg_loss

            # 学习率调度
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(current_loss)
                else:
                    self.scheduler.step()

            # 早停检查
            if current_loss < best_loss:
                best_loss = current_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

            # 打印进度
            if verbose and (epoch + 1) % 50 == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}, "
                      f"MSE: {avg_mse:.6f}, KL: {avg_kl:.6f}, LR: {lr:.6f}")

        # 恢复最佳模型
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self.history

    def _validate(self, val_loader: DataLoader, n_train: int) -> float:
        """验证模型"""
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                hf_output, _ = self.model.sample_forward(batch_x)
                kl = self.model.kl_divergence()
                loss_dict = self.criterion(
                    hf_pred=hf_output,
                    hf_target=batch_y,
                    kl_divergence=kl,
                    n_samples=n_train
                )
                val_loss += loss_dict['total_loss'].item()

        return val_loss / len(val_loader)

    def predict(self,
                x: np.ndarray,
                return_std: bool = True,
                n_samples: int = 100) -> Tuple[np.ndarray, ...]:
        """
        预测

        Args:
            x: 输入数据 [n, input_dim]
            return_std: 是否返回标准差
            n_samples: 蒙特卡洛采样数（用于估计不确定性）

        Returns:
            mean: 预测均值
            std: 预测标准差（如果return_std=True）
        """
        self.model.eval()
        x_tensor = torch.FloatTensor(x).to(self.device)

        with torch.no_grad():
            # 解析形式的均值和方差
            hf_mean, hf_var, lf_means, lf_vars = self.model(x_tensor)

            mean = hf_mean.cpu().numpy()
            var = hf_var.cpu().numpy()

            if return_std:
                # 蒙特卡洛采样估计总不确定性
                predictions = []
                for _ in range(n_samples):
                    hf_output, _ = self.model.sample_forward(x_tensor)
                    predictions.append(hf_output.cpu().numpy())

                predictions = np.array(predictions)
                mc_mean = np.mean(predictions, axis=0)
                mc_std = np.std(predictions, axis=0)

                return mc_mean, mc_std

        return mean

    def evaluate(self,
                 x: np.ndarray,
                 y: np.ndarray) -> Dict:
        """
        评估模型

        Args:
            x: 输入数据
            y: 真实值

        Returns:
            metrics: 评估指标字典
        """
        mean, std = self.predict(x, return_std=True)
        y = y.reshape(-1, 1) if y.ndim == 1 else y

        # 计算指标
        rmse = np.sqrt(np.mean((mean - y) ** 2))
        mae = np.mean(np.abs(mean - y))
        max_ae = np.max(np.abs(mean - y))

        # em和es
        mu_true = np.mean(y)
        std_true = np.std(y)
        mu_pred = np.mean(mean)
        std_pred = np.std(mean)

        em = np.abs(mu_pred - mu_true) / np.abs(mu_true) * 100
        es = np.abs(std_pred - std_true) / std_true * 100

        return {
            'rmse': rmse,
            'mae': mae,
            'max_ae': max_ae,
            'em': em,
            'es': es,
            'mean_uncertainty': np.mean(std)
        }

    def save(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': self.model.get_config()
        }, path)

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
