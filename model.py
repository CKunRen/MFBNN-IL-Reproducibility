"""
MFBNN主模型
Main MFBNN Model
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from .modules import BaseClassNetwork, AttentionModule, HFPredictionModule


class MFBNN(nn.Module):
    """
    多保真度贝叶斯神经网络 (MFBNN)
    Multi-Fidelity Bayesian Neural Network

    包含：
    - 基类网络（全局特征提取）- 仅1个
    - 多个注意力模块（局部特征提取）- 数量=LF模型数量
    - HF预测模块
    - 信息流矩阵M（控制注意力模块间的信息传递）
    """
    def __init__(self,
                 input_dim: int,
                 num_lf_models: int,
                 M: Optional[torch.Tensor] = None,
                 # 基类网络参数
                 base_conv_channels: List[int] = [32, 64],
                 base_conv_kernel_sizes: List[int] = [3, 3],
                 base_linear_dims: List[int] = [128, 64],
                 base_output_dim: int = 32,
                 # 注意力模块参数
                 attention_dims: List[int] = [64, 32],
                 attention_output_dim: int = 1,
                 # HF预测模块参数
                 hf_linear_dims: List[int] = [64, 32],
                 hf_output_dim: int = 1):
        """
        Args:
            input_dim: 输入维度
            num_lf_models: LF模型数量
            M: 信息流矩阵 [num_lf_models, num_lf_models]
               M[i,j]=1 表示第i个注意力模块的输出传递给第j个注意力模块的贝叶斯层
            base_conv_channels: 基类网络卷积通道数列表
            base_conv_kernel_sizes: 基类网络卷积核大小列表
            base_linear_dims: 基类网络线性层维度列表
            base_output_dim: 基类网络输出维度
            attention_dims: 注意力模块维度列表
            attention_output_dim: 注意力模块贝叶斯层输出维度
            hf_linear_dims: HF预测模块线性层维度列表
            hf_output_dim: HF预测模块输出维度
        """
        super(MFBNN, self).__init__()

        self.input_dim = input_dim
        self.num_lf_models = num_lf_models
        self.attention_hidden_dim = attention_dims[-1] if attention_dims else base_output_dim

        # 信息流矩阵M
        if M is None:
            # 默认：层次型配置，LF_i传递给LF_{i+1}
            M = torch.zeros(num_lf_models, num_lf_models)
            for i in range(num_lf_models - 1):
                M[i, i + 1] = 1.0
        self.register_buffer('M', M)

        # 基类网络（只有一个）
        self.base_network = BaseClassNetwork(
            input_dim=input_dim,
            conv_channels=base_conv_channels,
            conv_kernel_sizes=base_conv_kernel_sizes,
            linear_dims=base_linear_dims,
            output_dim=base_output_dim
        )

        # 注意力模块（每个LF一个）
        self.attention_modules = nn.ModuleList([
            AttentionModule(
                input_dim=base_output_dim,
                attention_dims=attention_dims,
                bayesian_output_dim=attention_output_dim
            )
            for _ in range(num_lf_models)
        ])

        # 特征转换层（用于M矩阵控制的信息传递）
        self.feature_transform = nn.ModuleList([
            nn.Linear(self.attention_hidden_dim, self.attention_hidden_dim)
            for _ in range(num_lf_models)
        ])

        # HF预测模块
        self.hf_module = HFPredictionModule(
            base_feature_dim=base_output_dim,
            num_lf_models=num_lf_models,
            lf_output_dim=attention_output_dim,
            linear_dims=hf_linear_dims,
            output_dim=hf_output_dim
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,
                                                  List[torch.Tensor], List[torch.Tensor]]:
        """
        前向传播（解析形式，返回均值和方差）

        Args:
            x: 输入张量 [batch_size, input_dim]

        Returns:
            hf_mean: HF预测均值 [batch_size, hf_output_dim]
            hf_var: HF预测方差 [batch_size, hf_output_dim]
            lf_means: 所有LF预测均值列表
            lf_vars: 所有LF预测方差列表
        """
        # 1. 基类网络提取全局特征
        base_features = self.base_network(x)

        # 2. 第一轮：计算所有注意力模块的输出（不考虑外部特征）
        lf_means = []
        lf_vars = []
        attention_outputs = []

        for i, attention_module in enumerate(self.attention_modules):
            mean, var, attn_out = attention_module(base_features, external_features=None)
            lf_means.append(mean)
            lf_vars.append(var)
            attention_outputs.append(attn_out)

        # 3. 第二轮：根据M矩阵融合信息并重新计算
        for j in range(self.num_lf_models):
            external_features = self._collect_external_features(j, attention_outputs)

            if external_features is not None:
                mean, var, _ = self.attention_modules[j](base_features, external_features)
                lf_means[j] = mean
                lf_vars[j] = var

        # 4. HF预测模块
        hf_mean, hf_var = self.hf_module(base_features, lf_means, lf_vars)

        return hf_mean, hf_var, lf_means, lf_vars

    def sample_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        采样前向传播（用于训练）

        Returns:
            hf_output: HF采样输出
            lf_outputs: LF采样输出列表
        """
        # 基类网络
        base_features = self.base_network(x)

        # 注意力模块（第一轮）
        lf_outputs = []
        attention_outputs = []

        for i, attention_module in enumerate(self.attention_modules):
            output, attn_out = attention_module.sample_forward(base_features, external_features=None)
            lf_outputs.append(output)
            attention_outputs.append(attn_out)

        # 根据M矩阵重新计算（第二轮）
        for j in range(self.num_lf_models):
            external_features = self._collect_external_features(j, attention_outputs)

            if external_features is not None:
                output, _ = self.attention_modules[j].sample_forward(base_features, external_features)
                lf_outputs[j] = output

        # 获取均值和方差用于HF模块
        lf_means = []
        lf_vars = []
        for j in range(self.num_lf_models):
            external_features = self._collect_external_features(j, attention_outputs)
            mean, var, _ = self.attention_modules[j](base_features, external_features)
            lf_means.append(mean)
            lf_vars.append(var)

        # HF模块
        hf_output = self.hf_module.sample_forward(base_features, lf_means, lf_vars)

        return hf_output, lf_outputs

    def _collect_external_features(self, j: int, attention_outputs: List[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        收集所有传递给模块j的外部特征（由M矩阵控制）

        Args:
            j: 目标模块索引
            attention_outputs: 所有注意力模块的输出列表

        Returns:
            external_features: 融合后的外部特征，或None
        """
        external_features = None
        for i in range(self.num_lf_models):
            if self.M[i, j] > 0:
                transformed = self.feature_transform[i](attention_outputs[i])
                if external_features is None:
                    external_features = self.M[i, j] * transformed
                else:
                    external_features = external_features + self.M[i, j] * transformed
        return external_features

    def kl_divergence(self) -> torch.Tensor:
        """返回所有贝叶斯层的KL散度之和"""
        kl = torch.tensor(0.0, device=next(self.parameters()).device)

        # 注意力模块的KL散度
        for attention_module in self.attention_modules:
            kl = kl + attention_module.kl_divergence()

        # HF模块的KL散度
        kl = kl + self.hf_module.kl_divergence()

        return kl

    def get_config(self) -> dict:
        """返回网络配置"""
        return {
            'input_dim': self.input_dim,
            'num_lf_models': self.num_lf_models,
            'M': self.M.cpu().numpy().tolist()
        }

    def update_M(self, M: torch.Tensor):
        """
        更新信息流矩阵M

        Args:
            M: 新的信息流矩阵 [num_lf_models, num_lf_models]
        """
        assert M.shape == self.M.shape, f"M shape mismatch: expected {self.M.shape}, got {M.shape}"
        self.M.copy_(M.to(self.M.device))
