"""
Multi-Fidelity GNN with 2 LF models + 1 HF model (Optimized Version v2)
多保真度图神经网络：2个低保真度模型 + 1个高保真度模型
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import savemat
import os
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)

# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# =============================================================================
# 数据归一化器
# =============================================================================

class Normalizer:
    """数据归一化器"""
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = np.mean(data)
        self.std = np.std(data)
        if self.std < 1e-8:
            self.std = 1.0

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


# =============================================================================
# 改进的图神经网络
# =============================================================================

class ImprovedGraphConv(nn.Module):
    """改进的图卷积层（带残差连接）"""
    def __init__(self, in_dim, out_dim):
        super(ImprovedGraphConv, self).__init__()
        self.fc_self = nn.Linear(in_dim, out_dim)
        self.fc_neighbor = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

        # 残差连接（如果维度不同则用投影）
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x, edge_index, n_nodes):
        src, dst = edge_index

        # 自身特征变换
        h_self = self.fc_self(x)

        # 邻居聚合
        neighbor_feats = x[src]
        h_neighbor_raw = self.fc_neighbor(neighbor_feats)

        h_neighbor = torch.zeros_like(h_self)
        h_neighbor.scatter_add_(0, dst.unsqueeze(-1).expand_as(h_neighbor_raw), h_neighbor_raw)

        # 度数归一化
        degree = torch.zeros(h_self.size(0), device=x.device)
        degree.scatter_add_(0, dst, torch.ones(dst.size(0), device=x.device))
        degree = degree.clamp(min=1).unsqueeze(-1)
        h_neighbor = h_neighbor / degree

        # 残差连接
        out = h_self + h_neighbor + self.residual(x)
        out = self.bn(out)

        return F.relu(out)


class ImprovedGNN(nn.Module):
    """改进的GNN回归模型"""
    def __init__(self, input_dim=1, hidden_dim=128, output_dim=1, n_layers=4, n_nodes=20, dropout=0.1):
        super(ImprovedGNN, self).__init__()

        self.n_nodes = n_nodes
        self.dropout = dropout

        # 节点编码器（更深）
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 图卷积层
        self.gnn_layers = nn.ModuleList([
            ImprovedGraphConv(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])

        # 输出层（更深）
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x, edge_index):
        batch_size = x.size(0)

        x_flat = x.view(-1, 1)
        edge_index_batch = self._batch_edge_index(edge_index, batch_size)

        h = self.node_encoder(x_flat)

        for gnn_layer in self.gnn_layers:
            h = gnn_layer(h, edge_index_batch, self.n_nodes)
            h = F.dropout(h, p=self.dropout, training=self.training)

        h = h.view(batch_size, self.n_nodes, -1)
        h_global = h.mean(dim=1)

        return self.decoder(h_global)

    def _batch_edge_index(self, edge_index, batch_size):
        src, dst = edge_index
        n_edges = src.size(0)
        offsets = torch.arange(batch_size, device=edge_index.device) * self.n_nodes
        src_batch = src.repeat(batch_size) + offsets.repeat_interleave(n_edges)
        dst_batch = dst.repeat(batch_size) + offsets.repeat_interleave(n_edges)
        return torch.stack([src_batch, dst_batch])


class ImprovedDiscrepancyGNN(nn.Module):
    """改进的偏差修正GNN模型"""
    def __init__(self, input_dim=1, hidden_dim=128, output_dim=1, n_layers=4, n_nodes=20, dropout=0.1):
        super(ImprovedDiscrepancyGNN, self).__init__()

        self.n_nodes = n_nodes
        self.dropout = dropout

        # 节点编码器（输入包括原始特征和LF预测）
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 图卷积层
        self.gnn_layers = nn.ModuleList([
            ImprovedGraphConv(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])

        # 输出层
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x, lf_pred, edge_index):
        batch_size = x.size(0)

        lf_feat = lf_pred.expand(-1, self.n_nodes).unsqueeze(-1)
        x_aug = torch.cat([x.unsqueeze(-1), lf_feat], dim=-1)
        x_flat = x_aug.view(-1, 2)

        edge_index_batch = self._batch_edge_index(edge_index, batch_size)

        h = self.node_encoder(x_flat)

        for gnn_layer in self.gnn_layers:
            h = gnn_layer(h, edge_index_batch, self.n_nodes)
            h = F.dropout(h, p=self.dropout, training=self.training)

        h = h.view(batch_size, self.n_nodes, -1)
        h_global = h.mean(dim=1)

        return self.decoder(h_global)

    def _batch_edge_index(self, edge_index, batch_size):
        src, dst = edge_index
        n_edges = src.size(0)
        offsets = torch.arange(batch_size, device=edge_index.device) * self.n_nodes
        src_batch = src.repeat(batch_size) + offsets.repeat_interleave(n_edges)
        dst_batch = dst.repeat(batch_size) + offsets.repeat_interleave(n_edges)
        return torch.stack([src_batch, dst_batch])


# =============================================================================
# 训练函数（带早停和余弦退火）
# =============================================================================

def train_model_gnn(model, train_loader, edge_index, num_epochs=500, lr=0.001,
                    model_type='base', patience=50, verbose=True):
    """训练GNN模型（带早停）"""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    model.train()
    best_loss = float('inf')
    best_state = None
    patience_counter = 0
    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            if model_type == 'base':
                x_batch, y_batch = batch
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                pred = model(x_batch, edge_index)
                loss = F.mse_loss(pred, y_batch)
            else:
                x_batch, lf_pred_batch, delta_batch = batch
                x_batch = x_batch.to(device)
                lf_pred_batch = lf_pred_batch.to(device)
                delta_batch = delta_batch.to(device)
                optimizer.zero_grad()
                pred_delta = model(x_batch, lf_pred_batch, edge_index)
                loss = F.mse_loss(pred_delta, delta_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)

        # 早停检查
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break

        if verbose and (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")

    # 恢复最佳模型
    if best_state is not None:
        model.load_state_dict(best_state)

    return losses