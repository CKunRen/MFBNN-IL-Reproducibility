"""
网络模块定义
Network Modules: BaseClassNetwork, AttentionModule, HFPredictionModule
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .layers import BayesianLinearNoBias


class BaseClassNetwork(nn.Module):
    """
    基类网络：一维卷积层 + 线性层
    用于元学习全局特征提取

    Base Class Network: 1D Conv + Linear layers
    For meta-learning global feature extraction
    """
    def __init__(self,
                 input_dim: int,
                 conv_channels: List[int] = [32, 64],
                 conv_kernel_sizes: List[int] = [3, 3],
                 linear_dims: List[int] = [128, 64],
                 output_dim: int = 32):
        """
        Args:
            input_dim: 输入维度
            conv_channels: 卷积层通道数列表
            conv_kernel_sizes: 卷积核大小列表
            linear_dims: 线性层维度列表
            output_dim: 输出特征维度
        """
        super(BaseClassNetwork, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # 构建一维卷积层
        conv_layers = []
        in_channels = 1  # 输入通道数为1
        for i, (out_channels, kernel_size) in enumerate(zip(conv_channels, conv_kernel_sizes)):
            conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.BatchNorm1d(out_channels))
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_layers) if conv_layers else nn.Identity()

        # 计算卷积后的特征维度
        conv_output_dim = conv_channels[-1] * input_dim if conv_channels else input_dim

        # 构建线性层
        linear_layers = []
        in_dim = conv_output_dim
        for out_dim in linear_dims:
            linear_layers.append(nn.Linear(in_dim, out_dim))
            linear_layers.append(nn.ReLU())
            in_dim = out_dim

        linear_layers.append(nn.Linear(in_dim, output_dim))
        self.linear_layers = nn.Sequential(*linear_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, input_dim]

        Returns:
            features: 全局特征 [batch_size, output_dim]
        """
        batch_size = x.size(0)

        # 调整维度用于1D卷积: [batch_size, 1, input_dim]
        x = x.unsqueeze(1)

        # 卷积层
        x = self.conv_layers(x)

        # 展平
        x = x.view(batch_size, -1)

        # 线性层
        features = self.linear_layers(x)

        return features


class AttentionModule(nn.Module):
    """
    注意力模块：注意力机制 + 贝叶斯层
    用于提取局部特征并预测均值和方差

    Attention Module: Attention mechanism + Bayesian layer
    For local feature extraction and mean/variance prediction
    """
    def __init__(self,
                 input_dim: int,
                 attention_dims: List[int] = [64, 32],
                 bayesian_output_dim: int = 1,
                 num_heads: int = 4):
        """
        Args:
            input_dim: 输入特征维度（来自基类网络）
            attention_dims: 注意力层维度列表
            bayesian_output_dim: 贝叶斯层输出维度
            num_heads: 注意力头数
        """
        super(AttentionModule, self).__init__()

        self.input_dim = input_dim
        self.bayesian_output_dim = bayesian_output_dim

        # 注意力层
        attention_layers = []
        in_dim = input_dim
        for out_dim in attention_dims:
            attention_layers.append(nn.Linear(in_dim, out_dim))
            attention_layers.append(nn.ReLU())
            in_dim = out_dim

        self.attention_layers = nn.Sequential(*attention_layers) if attention_layers else nn.Identity()
        self.attention_output_dim = attention_dims[-1] if attention_dims else input_dim

        # 注意力权重计算
        self.attention_weights = nn.Sequential(
            nn.Linear(self.attention_output_dim, self.attention_output_dim // 2),
            nn.Tanh(),
            nn.Linear(self.attention_output_dim // 2, 1),
            nn.Softmax(dim=1)
        )

        # 贝叶斯输出层（无偏差）
        self.bayesian_layer = BayesianLinearNoBias(self.attention_output_dim, bayesian_output_dim)

    def forward(self, x: torch.Tensor,
                external_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入特征 [batch_size, input_dim]
            external_features: 来自其他注意力模块的外部特征（由M矩阵控制）

        Returns:
            mean: 预测均值 [batch_size, bayesian_output_dim]
            var: 预测方差 [batch_size, bayesian_output_dim]
            attention_output: 注意力层输出（用于传递给其他模块）
        """
        # 注意力层处理
        attention_output = self.attention_layers(x)

        # 如果有外部特征输入（来自其他注意力模块，由M控制）
        if external_features is not None:
            attention_output = attention_output + external_features

        # 贝叶斯层输出均值和方差
        mean, var = self.bayesian_layer(attention_output)

        return mean, var, attention_output

    def sample_forward(self, x: torch.Tensor,
                       external_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        采样前向传播（用于训练）
        """
        attention_output = self.attention_layers(x)

        if external_features is not None:
            attention_output = attention_output + external_features

        output = self.bayesian_layer.sample_forward(attention_output)

        return output, attention_output

    def kl_divergence(self) -> torch.Tensor:
        """返回贝叶斯层的KL散度"""
        return self.bayesian_layer.kl_divergence()


class HFPredictionModule(nn.Module):
    """
    HF预测模块：线性层 + 贝叶斯层
    输入为基类网络输出和注意力模块的预测均值+方差

    HF Prediction Module: Linear layers + Bayesian layer
    Input: base network output + attention module predictions (mean + variance)
    """
    def __init__(self,
                 base_feature_dim: int,
                 num_lf_models: int,
                 lf_output_dim: int = 1,
                 linear_dims: List[int] = [64, 32],
                 output_dim: int = 1):
        """
        Args:
            base_feature_dim: 基类网络输出维度
            num_lf_models: LF模型数量（注意力模块数量）
            lf_output_dim: 每个LF模型的输出维度
            linear_dims: 线性层维度列表
            output_dim: 最终输出维度
        """
        super(HFPredictionModule, self).__init__()

        # 输入维度 = 基类特征 + 所有LF的均值和方差
        self.input_dim = base_feature_dim + num_lf_models * lf_output_dim * 2

        # 线性层
        linear_layers = []
        in_dim = self.input_dim
        for out_dim in linear_dims:
            linear_layers.append(nn.Linear(in_dim, out_dim))
            linear_layers.append(nn.ReLU())
            in_dim = out_dim

        self.linear_layers = nn.Sequential(*linear_layers) if linear_layers else nn.Identity()
        self.linear_output_dim = linear_dims[-1] if linear_dims else self.input_dim

        # 贝叶斯输出层（无偏差）
        self.bayesian_layer = BayesianLinearNoBias(self.linear_output_dim, output_dim)

    def forward(self, base_features: torch.Tensor,
                lf_means: List[torch.Tensor],
                lf_vars: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            base_features: 基类网络输出 [batch_size, base_feature_dim]
            lf_means: LF模型预测均值列表
            lf_vars: LF模型预测方差列表

        Returns:
            mean: HF预测均值 [batch_size, output_dim]
            var: HF预测方差 [batch_size, output_dim]
        """
        # 拼接所有输入
        inputs = [base_features]
        for mean, var in zip(lf_means, lf_vars):
            inputs.append(mean)
            inputs.append(var)

        x = torch.cat(inputs, dim=-1)

        # 线性层
        x = self.linear_layers(x)

        # 贝叶斯层输出
        mean, var = self.bayesian_layer(x)

        return mean, var

    def sample_forward(self, base_features: torch.Tensor,
                       lf_means: List[torch.Tensor],
                       lf_vars: List[torch.Tensor]) -> torch.Tensor:
        """采样前向传播"""
        inputs = [base_features]
        for mean, var in zip(lf_means, lf_vars):
            inputs.append(mean)
            inputs.append(var)

        x = torch.cat(inputs, dim=-1)
        x = self.linear_layers(x)
        output = self.bayesian_layer.sample_forward(x)

        return output

    def kl_divergence(self) -> torch.Tensor:
        """返回贝叶斯层的KL散度"""
        return self.bayesian_layer.kl_divergence()
