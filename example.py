"""
MFBNN 使用示例
MFBNN Usage Example

展示如何使用MFBNN进行多保真度回归
Demonstrates how to use MFBNN for multi-fidelity regression
"""

import numpy as np
import torch
import sys
import os

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MFBNN_Network import MFBNN, MFBNNTrainer


def generate_test_data(n_hf=50, n_lf=200, input_dim=5, noise_hf=0.01, noise_lf=0.05):
    """
    生成测试数据
    Generate test data for multi-fidelity regression
    """
    # 高保真度数据
    x_hf = np.random.uniform(-1, 1, (n_hf, input_dim))
    y_hf = np.sum(x_hf ** 2, axis=1) + noise_hf * np.random.randn(n_hf)

    # 低保真度数据 (LF1 - 最低保真度)
    x_lf1 = np.random.uniform(-1, 1, (n_lf, input_dim))
    y_lf1 = 0.8 * np.sum(x_lf1 ** 2, axis=1) + 0.1 + noise_lf * np.random.randn(n_lf)

    # 低保真度数据 (LF2 - 中等保真度)
    x_lf2 = np.random.uniform(-1, 1, (n_lf, input_dim))
    y_lf2 = 0.95 * np.sum(x_lf2 ** 2, axis=1) + 0.02 + 0.5 * noise_lf * np.random.randn(n_lf)

    return x_hf, y_hf, [x_lf1, x_lf2], [y_lf1, y_lf2]


def main():
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)

    # 参数设置
    input_dim = 5
    num_lf_models = 2

    print("=" * 60)
    print("MFBNN - Multi-Fidelity Bayesian Neural Network Example")
    print("=" * 60)

    # 生成数据
    print("\n1. 生成测试数据...")
    x_hf, y_hf, x_lf_list, y_lf_list = generate_test_data(
        n_hf=100, n_lf=300, input_dim=input_dim
    )
    print(f"   HF数据: {x_hf.shape[0]} 个样本")
    print(f"   LF1数据: {x_lf_list[0].shape[0]} 个样本")
    print(f"   LF2数据: {x_lf_list[1].shape[0]} 个样本")

    # 创建信息流矩阵M
    # M[i,j]=1 表示LF_i的注意力输出传递给LF_j
    # 层次型: LF1 -> LF2
    M = torch.zeros(num_lf_models, num_lf_models)
    M[0, 1] = 1.0  # LF1的信息传递给LF2

    print(f"\n2. 信息流矩阵M:\n{M}")

    # 创建模型
    print("\n3. 创建MFBNN模型...")
    model = MFBNN(
        input_dim=input_dim,
        num_lf_models=num_lf_models,
        M=M,
        # 基类网络参数
        base_conv_channels=[32, 64],
        base_conv_kernel_sizes=[3, 3],
        base_linear_dims=[128, 64],
        base_output_dim=32,
        # 注意力模块参数
        attention_dims=[64, 32],
        attention_output_dim=1,
        # HF预测模块参数
        hf_linear_dims=[64, 32],
        hf_output_dim=1
    )

    # 打印模型结构
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   总参数量: {total_params:,}")
    print(f"   可训练参数量: {trainable_params:,}")

    # 创建训练器
    print("\n4. 创建训练器...")
    trainer = MFBNNTrainer(
        model=model,
        lambda_kl=0.1,  # KL散度权重
        learning_rate=0.001,
        weight_decay=1e-4,
        device='auto'
    )

    # 设置学习率调度器
    trainer.set_scheduler(scheduler_type='cosine', T_max=200)

    print(f"   设备: {trainer.device}")
    print(f"   λ (KL权重): {trainer.lambda_kl}")
    print(f"   学习率: {trainer.learning_rate}")

    # 训练模型
    print("\n5. 开始训练...")
    history = trainer.train(
        x_hf=x_hf,
        y_hf=y_hf,
        num_epochs=300,
        batch_size=32,
        validation_split=0.2,
        patience=50,
        verbose=True
    )

    print(f"\n   训练完成，共 {len(history['train_loss'])} 轮")
    print(f"   最终训练损失: {history['train_loss'][-1]:.6f}")
    if history['val_loss']:
        print(f"   最终验证损失: {history['val_loss'][-1]:.6f}")

    # 生成测试数据
    print("\n6. 模型评估...")
    x_test = np.random.uniform(-1, 1, (200, input_dim))
    y_test = np.sum(x_test ** 2, axis=1)

    # 评估
    metrics = trainer.evaluate(x_test, y_test)
    print(f"   RMSE: {metrics['rmse']:.6f}")
    print(f"   MAE: {metrics['mae']:.6f}")
    print(f"   Max AE: {metrics['max_ae']:.6f}")
    print(f"   em (%): {metrics['em']:.2f}")
    print(f"   es (%): {metrics['es']:.2f}")
    print(f"   平均不确定性: {metrics['mean_uncertainty']:.6f}")

    # 预测示例
    print("\n7. 预测示例...")
    x_sample = np.array([[0.5, -0.3, 0.2, -0.1, 0.4]])
    mean, std = trainer.predict(x_sample, return_std=True, n_samples=100)
    y_true = np.sum(x_sample ** 2)

    print(f"   输入: {x_sample[0]}")
    print(f"   真实值: {y_true:.6f}")
    print(f"   预测均值: {mean[0, 0]:.6f}")
    print(f"   预测标准差: {std[0, 0]:.6f}")
    print(f"   95%置信区间: [{mean[0, 0] - 1.96*std[0, 0]:.6f}, {mean[0, 0] + 1.96*std[0, 0]:.6f}]")

    # 保存模型
    print("\n8. 保存模型...")
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mfbnn_model.pth')
    trainer.save(save_path)
    print(f"   模型已保存到: {save_path}")

    print("\n" + "=" * 60)
    print("示例完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
