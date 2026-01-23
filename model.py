"""
Main MFBNN-IL Model
(Multi-Fidelity BNN Modeling using Incremental Learning)
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from .modules import BaseClassNetwork, AttentionModule, HFPredictionModule


class MFBNN(nn.Module):
    """
    MFBNN-IL: Multi-Fidelity BNN Modeling using Incremental Learning

    Components:
    - Base class network (global feature extraction) - only 1
    - Multiple attention modules (local feature extraction) - count = number of LF models
    - HF prediction module
    - Information flow matrix M (controls information transfer between attention modules)
    """
    def __init__(self,
                 input_dim: int,
                 num_lf_models: int,
                 M: Optional[torch.Tensor] = None,
                 # Base network parameters
                 base_conv_channels: List[int] = [32, 64],
                 base_conv_kernel_sizes: List[int] = [3, 3],
                 base_linear_dims: List[int] = [128, 64],
                 base_output_dim: int = 32,
                 # Attention module parameters
                 attention_dims: List[int] = [64, 32],
                 attention_output_dim: int = 1,
                 # HF prediction module parameters
                 hf_linear_dims: List[int] = [64, 32],
                 hf_output_dim: int = 1):
        """
        Args:
            input_dim: Input dimension
            num_lf_models: Number of LF models
            M: Information flow matrix [num_lf_models, num_lf_models]
               M[i,j]=1 means the output of attention module i is passed to the Bayesian layer of module j
            base_conv_channels: List of base network convolution channel sizes
            base_conv_kernel_sizes: List of base network convolution kernel sizes
            base_linear_dims: List of base network linear layer dimensions
            base_output_dim: Base network output dimension
            attention_dims: List of attention module dimensions
            attention_output_dim: Attention module Bayesian layer output dimension
            hf_linear_dims: List of HF prediction module linear layer dimensions
            hf_output_dim: HF prediction module output dimension
        """
        super(MFBNN, self).__init__()

        self.input_dim = input_dim
        self.num_lf_models = num_lf_models
        self.attention_hidden_dim = attention_dims[-1] if attention_dims else base_output_dim

        # Information flow matrix M
        if M is None:
            # Default: hierarchical configuration, LF_i passes to LF_{i+1}
            M = torch.zeros(num_lf_models, num_lf_models)
            for i in range(num_lf_models - 1):
                M[i, i + 1] = 1.0
        self.register_buffer('M', M)

        # Base class network (only one)
        self.base_network = BaseClassNetwork(
            input_dim=input_dim,
            conv_channels=base_conv_channels,
            conv_kernel_sizes=base_conv_kernel_sizes,
            linear_dims=base_linear_dims,
            output_dim=base_output_dim
        )

        # Attention modules (one per LF)
        self.attention_modules = nn.ModuleList([
            AttentionModule(
                input_dim=base_output_dim,
                attention_dims=attention_dims,
                bayesian_output_dim=attention_output_dim
            )
            for _ in range(num_lf_models)
        ])

        # Feature transform layers (for information transfer controlled by M matrix)
        self.feature_transform = nn.ModuleList([
            nn.Linear(self.attention_hidden_dim, self.attention_hidden_dim)
            for _ in range(num_lf_models)
        ])

        # HF prediction module
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
        Forward pass (analytical form, returns mean and variance)

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            hf_mean: HF predicted mean [batch_size, hf_output_dim]
            hf_var: HF predicted variance [batch_size, hf_output_dim]
            lf_means: List of all LF predicted means
            lf_vars: List of all LF predicted variances
        """
        # 1. Base network extracts global features
        base_features = self.base_network(x)

        # 2. First pass: compute all attention module outputs (without external features)
        lf_means = []
        lf_vars = []
        attention_outputs = []

        for i, attention_module in enumerate(self.attention_modules):
            mean, var, attn_out = attention_module(base_features, external_features=None)
            lf_means.append(mean)
            lf_vars.append(var)
            attention_outputs.append(attn_out)

        # 3. Second pass: fuse information according to M matrix and recompute
        for j in range(self.num_lf_models):
            external_features = self._collect_external_features(j, attention_outputs)

            if external_features is not None:
                mean, var, _ = self.attention_modules[j](base_features, external_features)
                lf_means[j] = mean
                lf_vars[j] = var

        # 4. HF prediction module
        hf_mean, hf_var = self.hf_module(base_features, lf_means, lf_vars)

        return hf_mean, hf_var, lf_means, lf_vars

    def sample_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Sampled forward pass (for training)

        Returns:
            hf_output: HF sampled output
            lf_outputs: List of LF sampled outputs
        """
        # Base network
        base_features = self.base_network(x)

        # Attention modules (first pass)
        lf_outputs = []
        attention_outputs = []

        for i, attention_module in enumerate(self.attention_modules):
            output, attn_out = attention_module.sample_forward(base_features, external_features=None)
            lf_outputs.append(output)
            attention_outputs.append(attn_out)

        # Recompute according to M matrix (second pass)
        for j in range(self.num_lf_models):
            external_features = self._collect_external_features(j, attention_outputs)

            if external_features is not None:
                output, _ = self.attention_modules[j].sample_forward(base_features, external_features)
                lf_outputs[j] = output

        # Get mean and variance for HF module
        lf_means = []
        lf_vars = []
        for j in range(self.num_lf_models):
            external_features = self._collect_external_features(j, attention_outputs)
            mean, var, _ = self.attention_modules[j](base_features, external_features)
            lf_means.append(mean)
            lf_vars.append(var)

        # HF module
        hf_output = self.hf_module.sample_forward(base_features, lf_means, lf_vars)

        return hf_output, lf_outputs

    def _collect_external_features(self, j: int, attention_outputs: List[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Collect all external features passed to module j (controlled by M matrix)

        Args:
            j: Target module index
            attention_outputs: List of all attention module outputs

        Returns:
            external_features: Fused external features, or None
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
        """Return the sum of KL divergence from all Bayesian layers"""
        kl = torch.tensor(0.0, device=next(self.parameters()).device)

        # KL divergence from attention modules
        for attention_module in self.attention_modules:
            kl = kl + attention_module.kl_divergence()

        # KL divergence from HF module
        kl = kl + self.hf_module.kl_divergence()

        return kl

    def get_config(self) -> dict:
        """Return network configuration"""
        return {
            'input_dim': self.input_dim,
            'num_lf_models': self.num_lf_models,
            'M': self.M.cpu().numpy().tolist()
        }

    def update_M(self, M: torch.Tensor):
        """
        Update information flow matrix M

        Args:
            M: New information flow matrix [num_lf_models, num_lf_models]
        """
        assert M.shape == self.M.shape, f"M shape mismatch: expected {self.M.shape}, got {M.shape}"
        self.M.copy_(M.to(self.M.device))
