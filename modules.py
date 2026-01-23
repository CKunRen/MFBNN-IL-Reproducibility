"""
Network Modules for MFBNN-IL
(Multi-Fidelity BNN Modeling using Incremental Learning)

Modules: BaseClassNetwork, AttentionModule, HFPredictionModule
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .layers import BayesianLinearNoBias


class BaseClassNetwork(nn.Module):
    """
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
            input_dim: Input dimension
            conv_channels: List of convolution channel sizes
            conv_kernel_sizes: List of convolution kernel sizes
            linear_dims: List of linear layer dimensions
            output_dim: Output feature dimension
        """
        super(BaseClassNetwork, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Build 1D convolution layers
        conv_layers = []
        in_channels = 1  # Input channel is 1
        for i, (out_channels, kernel_size) in enumerate(zip(conv_channels, conv_kernel_sizes)):
            conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.BatchNorm1d(out_channels))
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_layers) if conv_layers else nn.Identity()

        # Calculate feature dimension after convolution
        conv_output_dim = conv_channels[-1] * input_dim if conv_channels else input_dim

        # Build linear layers
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
        Forward pass

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            features: Global features [batch_size, output_dim]
        """
        batch_size = x.size(0)

        # Reshape for 1D convolution: [batch_size, 1, input_dim]
        x = x.unsqueeze(1)

        # Convolution layers
        x = self.conv_layers(x)

        # Flatten
        x = x.view(batch_size, -1)

        # Linear layers
        features = self.linear_layers(x)

        return features


class AttentionModule(nn.Module):
    """
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
            input_dim: Input feature dimension (from base network)
            attention_dims: List of attention layer dimensions
            bayesian_output_dim: Bayesian layer output dimension
            num_heads: Number of attention heads
        """
        super(AttentionModule, self).__init__()

        self.input_dim = input_dim
        self.bayesian_output_dim = bayesian_output_dim

        # Attention layers
        attention_layers = []
        in_dim = input_dim
        for out_dim in attention_dims:
            attention_layers.append(nn.Linear(in_dim, out_dim))
            attention_layers.append(nn.ReLU())
            in_dim = out_dim

        self.attention_layers = nn.Sequential(*attention_layers) if attention_layers else nn.Identity()
        self.attention_output_dim = attention_dims[-1] if attention_dims else input_dim

        # Attention weight computation
        self.attention_weights = nn.Sequential(
            nn.Linear(self.attention_output_dim, self.attention_output_dim // 2),
            nn.Tanh(),
            nn.Linear(self.attention_output_dim // 2, 1),
            nn.Softmax(dim=1)
        )

        # Bayesian output layer (no bias)
        self.bayesian_layer = BayesianLinearNoBias(self.attention_output_dim, bayesian_output_dim)

    def forward(self, x: torch.Tensor,
                external_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            x: Input features [batch_size, input_dim]
            external_features: External features from other attention modules (controlled by M matrix)

        Returns:
            mean: Predicted mean [batch_size, bayesian_output_dim]
            var: Predicted variance [batch_size, bayesian_output_dim]
            attention_output: Attention layer output (for passing to other modules)
        """
        # Attention layer processing
        attention_output = self.attention_layers(x)

        # If external features are provided (from other attention modules, controlled by M)
        if external_features is not None:
            attention_output = attention_output + external_features

        # Bayesian layer outputs mean and variance
        mean, var = self.bayesian_layer(attention_output)

        return mean, var, attention_output

    def sample_forward(self, x: torch.Tensor,
                       external_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sampled forward pass (for training)
        """
        attention_output = self.attention_layers(x)

        if external_features is not None:
            attention_output = attention_output + external_features

        output = self.bayesian_layer.sample_forward(attention_output)

        return output, attention_output

    def kl_divergence(self) -> torch.Tensor:
        """Return KL divergence of the Bayesian layer"""
        return self.bayesian_layer.kl_divergence()


class HFPredictionModule(nn.Module):
    """
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
            base_feature_dim: Base network output dimension
            num_lf_models: Number of LF models (attention modules)
            lf_output_dim: Output dimension of each LF model
            linear_dims: List of linear layer dimensions
            output_dim: Final output dimension
        """
        super(HFPredictionModule, self).__init__()

        # Input dimension = base features + all LF means and variances
        self.input_dim = base_feature_dim + num_lf_models * lf_output_dim * 2

        # Linear layers
        linear_layers = []
        in_dim = self.input_dim
        for out_dim in linear_dims:
            linear_layers.append(nn.Linear(in_dim, out_dim))
            linear_layers.append(nn.ReLU())
            in_dim = out_dim

        self.linear_layers = nn.Sequential(*linear_layers) if linear_layers else nn.Identity()
        self.linear_output_dim = linear_dims[-1] if linear_dims else self.input_dim

        # Bayesian output layer (no bias)
        self.bayesian_layer = BayesianLinearNoBias(self.linear_output_dim, output_dim)

    def forward(self, base_features: torch.Tensor,
                lf_means: List[torch.Tensor],
                lf_vars: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            base_features: Base network output [batch_size, base_feature_dim]
            lf_means: List of LF model predicted means
            lf_vars: List of LF model predicted variances

        Returns:
            mean: HF predicted mean [batch_size, output_dim]
            var: HF predicted variance [batch_size, output_dim]
        """
        # Concatenate all inputs
        inputs = [base_features]
        for mean, var in zip(lf_means, lf_vars):
            inputs.append(mean)
            inputs.append(var)

        x = torch.cat(inputs, dim=-1)

        # Linear layers
        x = self.linear_layers(x)

        # Bayesian layer output
        mean, var = self.bayesian_layer(x)

        return mean, var

    def sample_forward(self, base_features: torch.Tensor,
                       lf_means: List[torch.Tensor],
                       lf_vars: List[torch.Tensor]) -> torch.Tensor:
        """Sampled forward pass"""
        inputs = [base_features]
        for mean, var in zip(lf_means, lf_vars):
            inputs.append(mean)
            inputs.append(var)

        x = torch.cat(inputs, dim=-1)
        x = self.linear_layers(x)
        output = self.bayesian_layer.sample_forward(x)

        return output

    def kl_divergence(self) -> torch.Tensor:
        """Return KL divergence of the Bayesian layer"""
        return self.bayesian_layer.kl_divergence()
