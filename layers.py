"""
Bayesian Layers for MFBNN-IL
(Multi-Fidelity BNN Modeling using Incremental Learning)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class BayesianLinearNoBias(nn.Module):
    """
    Bayesian Linear Layer - weights only, no bias
    Analytical expression for mean and variance prediction
    """
    def __init__(self, in_features: int, out_features: int):
        super(BayesianLinearNoBias, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight mean and log variance (variational parameters)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_log_var = nn.Parameter(torch.Tensor(out_features, in_features))

        # Prior distribution parameters
        self.prior_mu = 0.0
        self.prior_log_var = 0.0  # log(1) = 0, i.e., variance = 1

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.kaiming_normal_(self.weight_mu, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.weight_log_var, -5.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass, returns analytical mean and variance

        Args:
            x: Input tensor [batch_size, in_features]

        Returns:
            mean: Output mean [batch_size, out_features]
            var: Output variance [batch_size, out_features]
        """
        # Mean: E[y] = E[W] * x = weight_mu * x
        mean = F.linear(x, self.weight_mu, bias=None)

        # Variance: Var[y] = x^2 * Var[W] = x^2 * exp(weight_log_var)
        weight_var = torch.exp(self.weight_log_var)
        var = F.linear(x ** 2, weight_var, bias=None)

        return mean, var

    def sample_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sampled forward pass (for stochastic sampling during training)

        Args:
            x: Input tensor [batch_size, in_features]

        Returns:
            output: Sampled output [batch_size, out_features]
        """
        weight_std = torch.exp(0.5 * self.weight_log_var)
        weight_eps = torch.randn_like(self.weight_mu)
        weight = self.weight_mu + weight_std * weight_eps
        return F.linear(x, weight, bias=None)

    def kl_divergence(self) -> torch.Tensor:
        """
        Compute KL divergence: KL(q(w|theta) || p(w))
        """
        prior_var = torch.exp(torch.tensor(self.prior_log_var, device=self.weight_mu.device))
        weight_var = torch.exp(self.weight_log_var)

        kl = 0.5 * (self.prior_log_var - self.weight_log_var +
                    (weight_var + (self.weight_mu - self.prior_mu) ** 2) / prior_var - 1)

        return kl.sum()

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias=False'
