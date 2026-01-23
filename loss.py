"""
Loss Functions for MFBNN-IL
(Multi-Fidelity BNN Modeling using Incremental Learning)

Loss = MSE + lambda * KL Divergence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class MFBNNLoss(nn.Module):
    """
    MFBNN-IL Loss Function
    Loss = MSE + lambda * KL Divergence
    """
    def __init__(self,
                 lambda_kl: float = 0.1,
                 reduction: str = 'mean'):
        """
        Args:
            lambda_kl: KL divergence weight (lambda)
            reduction: Loss reduction method ('mean', 'sum', 'none')
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
        Compute total loss

        Args:
            hf_pred: HF predictions [batch_size, output_dim]
            hf_target: HF targets [batch_size, output_dim]
            kl_divergence: KL divergence (from model.kl_divergence())
            n_samples: Total number of training samples (for normalizing KL divergence)
            lf_preds: List of LF predictions (optional, for multi-task learning)
            lf_targets: List of LF targets (optional)
            lf_weights: List of LF loss weights (optional)

        Returns:
            loss_dict: Dictionary containing loss components
                - total_loss: Total loss
                - mse_loss: MSE loss
                - kl_loss: KL divergence loss
                - lf_losses: List of LF losses (if provided)
        """
        # HF MSE loss
        mse_loss = F.mse_loss(hf_pred, hf_target, reduction=self.reduction)

        # KL divergence loss (normalized)
        kl_loss = self.lambda_kl * kl_divergence / n_samples

        # Total loss
        total_loss = mse_loss + kl_loss

        # Loss dictionary
        loss_dict = {
            'total_loss': total_loss,
            'mse_loss': mse_loss,
            'kl_loss': kl_loss
        }

        # If LF targets are provided, compute LF losses
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
    Negative Log-Likelihood Loss for probabilistic predictions

    Assumes Gaussian output: y ~ N(mean, var)
    NLL = 0.5 * (log(var) + (y - mean)^2 / var)
    """
    def __init__(self,
                 lambda_kl: float = 0.1,
                 reduction: str = 'mean',
                 eps: float = 1e-6):
        """
        Args:
            lambda_kl: KL divergence weight
            reduction: Reduction method
            eps: Numerical stability
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
        Compute NLL loss

        Args:
            mean: Predicted mean [batch_size, output_dim]
            var: Predicted variance [batch_size, output_dim]
            target: Target values [batch_size, output_dim]
            kl_divergence: KL divergence
            n_samples: Total number of training samples

        Returns:
            loss_dict: Loss dictionary
        """
        # Ensure variance is positive
        var = var + self.eps

        # NLL loss
        nll = 0.5 * (torch.log(var) + (target - mean) ** 2 / var)

        if self.reduction == 'mean':
            nll_loss = nll.mean()
        elif self.reduction == 'sum':
            nll_loss = nll.sum()
        else:
            nll_loss = nll

        # KL divergence loss
        kl_loss = self.lambda_kl * kl_divergence / n_samples

        # Total loss
        total_loss = nll_loss + kl_loss

        return {
            'total_loss': total_loss,
            'nll_loss': nll_loss,
            'kl_loss': kl_loss
        }


class ELBOLoss(nn.Module):
    """
    Evidence Lower Bound (ELBO) Loss
    ELBO = E[log p(y|x,w)] - KL(q(w)||p(w))

    Standard loss function for variational inference
    """
    def __init__(self,
                 lambda_kl: float = 1.0,
                 num_mc_samples: int = 1,
                 reduction: str = 'mean'):
        """
        Args:
            lambda_kl: KL divergence weight (usually 1.0)
            num_mc_samples: Number of Monte Carlo samples
            reduction: Reduction method
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
        Compute ELBO loss

        Args:
            model: MFBNN-IL model
            x: Input [batch_size, input_dim]
            y: HF target values [batch_size, output_dim]
            n_samples: Total number of training samples

        Returns:
            loss_dict: Loss dictionary
        """
        # Monte Carlo sampling to estimate expectation
        log_likelihood = 0.0
        for _ in range(self.num_mc_samples):
            hf_output, _ = model.sample_forward(x)
            # Assume Gaussian likelihood with unit variance
            log_likelihood = log_likelihood - 0.5 * F.mse_loss(hf_output, y, reduction='sum')

        log_likelihood = log_likelihood / self.num_mc_samples

        # KL divergence
        kl_divergence = model.kl_divergence()
        kl_loss = self.lambda_kl * kl_divergence / n_samples

        # ELBO = log_likelihood - KL
        # Minimize -ELBO = -log_likelihood + KL
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
