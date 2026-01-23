"""
损失函数定义
Loss Functions: MSE + λ * KL Divergence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class MFBNNLoss(nn.Module):
    """
    MFBNN损失函数
    Loss = MSE + λ * KL散度

    MFBNN Loss Function
    Loss = MSE + λ * KL Divergence
    """
    def __init__(self,
                 lambda_kl: float = 0.1,
                 reduction: str = 'mean'):
        """
        Args:
            lambda_kl: KL散度权重 (λ)
            reduction: 损失归约方式 ('mean', 'sum', 'none')
        """
        super(MFBNNLoss, self).__init__()
        self.lambda_kl = lambda_kl
        self.reduction = reduction

    def forward(self,
                hf_pred: torch.Tensor,
                hf_target: torch.Tensor,
                kl_divergence: torch.Tensor,
                n_samples: int,
                lf_preds: Optional[List[torch.Tensor]] = None,
                lf_targets: Optional[List[torch.Tensor]] = None,
                lf_weights: Optional[List[float]] = None) -> dict:
        """
        计算总损失

        Args:
            hf_pred: HF预测值 [batch_size, output_dim]
            hf_target: HF真实值 [batch_size, output_dim]
            kl_divergence: KL散度（来自model.kl_divergence()）
            n_samples: 训练样本总数（用于归一化KL散度）
            lf_preds: LF预测值列表（可选，用于多任务学习）
            lf_targets: LF真实值列表（可选）
            lf_weights: LF损失权重列表（可选）

        Returns:
            loss_dict: 包含各项损失的字典
                - total_loss: 总损失
                - mse_loss: MSE损失
                - kl_loss: KL散度损失
                - lf_losses: LF损失列表（如果提供）
        """
        # HF MSE损失
        mse_loss = F.mse_loss(hf_pred, hf_target, reduction=self.reduction)

        # KL散度损失（归一化）
        kl_loss = self.lambda_kl * kl_divergence / n_samples

        # 总损失
        total_loss = mse_loss + kl_loss

        # 损失字典
        loss_dict = {
            'total_loss': total_loss,
            'mse_loss': mse_loss,
            'kl_loss': kl_loss
        }

        # 如果提供了LF目标，计算LF损失
        if lf_preds is not None and lf_targets is not None:
            lf_losses = []
            if lf_weights is None:
                lf_weights = [1.0] * len(lf_preds)

            for i, (pred, target, weight) in enumerate(zip(lf_preds, lf_targets, lf_weights)):
                lf_loss = weight * F.mse_loss(pred, target, reduction=self.reduction)
                lf_losses.append(lf_loss)
                total_loss = total_loss + lf_loss

            loss_dict['lf_losses'] = lf_losses
            loss_dict['total_loss'] = total_loss

        return loss_dict


class NLLLoss(nn.Module):
    """
    负对数似然损失（用于概率预测）
    Negative Log-Likelihood Loss for probabilistic predictions

    假设输出为高斯分布：y ~ N(mean, var)
    NLL = 0.5 * (log(var) + (y - mean)^2 / var)
    """
    def __init__(self,
                 lambda_kl: float = 0.1,
                 reduction: str = 'mean',
                 eps: float = 1e-6):
        """
        Args:
            lambda_kl: KL散度权重
            reduction: 归约方式
            eps: 数值稳定性
        """
        super(NLLLoss, self).__init__()
        self.lambda_kl = lambda_kl
        self.reduction = reduction
        self.eps = eps

    def forward(self,
                mean: torch.Tensor,
                var: torch.Tensor,
                target: torch.Tensor,
                kl_divergence: torch.Tensor,
                n_samples: int) -> dict:
        """
        计算NLL损失

        Args:
            mean: 预测均值 [batch_size, output_dim]
            var: 预测方差 [batch_size, output_dim]
            target: 真实值 [batch_size, output_dim]
            kl_divergence: KL散度
            n_samples: 训练样本总数

        Returns:
            loss_dict: 损失字典
        """
        # 确保方差为正
        var = var + self.eps

        # NLL损失
        nll = 0.5 * (torch.log(var) + (target - mean) ** 2 / var)

        if self.reduction == 'mean':
            nll_loss = nll.mean()
        elif self.reduction == 'sum':
            nll_loss = nll.sum()
        else:
            nll_loss = nll

        # KL散度损失
        kl_loss = self.lambda_kl * kl_divergence / n_samples

        # 总损失
        total_loss = nll_loss + kl_loss

        return {
            'total_loss': total_loss,
            'nll_loss': nll_loss,
            'kl_loss': kl_loss
        }


class ELBOLoss(nn.Module):
    """
    证据下界损失 (Evidence Lower Bound)
    ELBO = E[log p(y|x,w)] - KL(q(w)||p(w))

    用于变分推断的标准损失函数
    """
    def __init__(self,
                 lambda_kl: float = 1.0,
                 num_mc_samples: int = 1,
                 reduction: str = 'mean'):
        """
        Args:
            lambda_kl: KL散度权重（通常为1.0）
            num_mc_samples: 蒙特卡洛采样数
            reduction: 归约方式
        """
        super(ELBOLoss, self).__init__()
        self.lambda_kl = lambda_kl
        self.num_mc_samples = num_mc_samples
        self.reduction = reduction

    def forward(self,
                model,
                x: torch.Tensor,
                y: torch.Tensor,
                n_samples: int) -> dict:
        """
        计算ELBO损失

        Args:
            model: MFBNN模型
            x: 输入 [batch_size, input_dim]
            y: HF目标值 [batch_size, output_dim]
            n_samples: 训练样本总数

        Returns:
            loss_dict: 损失字典
        """
        # 蒙特卡洛采样估计期望
        log_likelihood = 0.0
        for _ in range(self.num_mc_samples):
            hf_output, _ = model.sample_forward(x)
            # 假设高斯似然，方差为1
            log_likelihood = log_likelihood - 0.5 * F.mse_loss(hf_output, y, reduction='sum')

        log_likelihood = log_likelihood / self.num_mc_samples

        # KL散度
        kl_divergence = model.kl_divergence()
        kl_loss = self.lambda_kl * kl_divergence / n_samples

        # ELBO = log_likelihood - KL
        # 最小化 -ELBO = -log_likelihood + KL
        elbo = log_likelihood - kl_loss
        loss = -elbo

        if self.reduction == 'mean':
            loss = loss / x.size(0)

        return {
            'total_loss': loss,
            'log_likelihood': log_likelihood,
            'kl_loss': kl_loss,
            'elbo': elbo
        }
