"""
MFBNN - Multi-Fidelity Bayesian Neural Network
多保真度贝叶斯神经网络

Modules:
    - layers: Bayesian layers (BayesianLinearNoBias)
    - modules: Network modules (BaseClassNetwork, AttentionModule, HFPredictionModule)
    - model: Main MFBNN model
    - loss: Loss functions (MFBNNLoss, NLLLoss, ELBOLoss)
    - train: Training utilities (MFBNNTrainer)
"""

from .layers import BayesianLinearNoBias
from .modules import BaseClassNetwork, AttentionModule, HFPredictionModule
from .model import MFBNN
from .loss import MFBNNLoss, NLLLoss, ELBOLoss
from .train import MFBNNTrainer

__all__ = [
    'BayesianLinearNoBias',
    'BaseClassNetwork',
    'AttentionModule',
    'HFPredictionModule',
    'MFBNN',
    'MFBNNLoss',
    'NLLLoss',
    'ELBOLoss',
    'MFBNNTrainer'
]

__version__ = '1.0.0'
