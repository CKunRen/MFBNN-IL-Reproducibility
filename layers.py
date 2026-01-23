"""
贝叶斯层定义
Bayesian Layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class BayesianLinearNoBias(nn.Module):
    """
    贝叶斯线性层 - 仅有权重，无偏差
    预测均值和方差的解析表达

    Bayesian Linear Layer - weights only, no bias
    Analytical expression for mean and variance prediction
    """
    def __init__(self, in_features: int, out_features: int):
        super(BayesianLinearNoBias, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 权重的均值和log方差（变分参数）
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_log_var = nn.Parameter(torch.Tensor(out_features, in_features))

        # 先验分布参数
        self.prior_mu = 0.0
        self.prior_log_var = 0.0  # log(1) = 0, 即方差为1

        self.reset_parameters()

    def reset_parameters(self):
        """初始化参数"""
        nn.init.kaiming_normal_(self.weight_mu, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.weight_log_var, -5.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播，返回均值和方差的解析表达

        Args:
            x: 输入张量 [batch_size, in_features]

        Returns:
            mean: 输出均值 [batch_size, out_features]
            var: 输出方差 [batch_size, out_features]
        """
        # 均值: E[y] = E[W] * x = weight_mu * x
        mean = F.linear(x, self.weight_mu, bias=None)

        # 方差: Var[y] = x^2 * Var[W] = x^2 * exp(weight_log_var)
        weight_var = torch.exp(self.weight_log_var)
        var = F.linear(x ** 2, weight_var, bias=None)

        return mean, var

    def sample_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        采样前向传播（用于训练时的随机采样）

        Args:
            x: 输入张量 [batch_size, in_features]

        Returns:
            output: 采样输出 [batch_size, out_features]
        """
        weight_std = torch.exp(0.5 * self.weight_log_var)
        weight_eps = torch.randn_like(self.weight_mu)
        weight = self.weight_mu + weight_std * weight_eps
        return F.linear(x, weight, bias=None)

    def kl_divergence(self) -> torch.Tensor:
        """
        计算KL散度: KL(q(w|θ) || p(w))
        """
        prior_var = torch.exp(torch.tensor(self.prior_log_var, device=self.weight_mu.device))
        weight_var = torch.exp(self.weight_log_var)

        kl = 0.5 * (self.prior_log_var - self.weight_log_var +
                    (weight_var + (self.weight_mu - self.prior_mu) ** 2) / prior_var - 1)

        return kl.sum()

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias=False'
