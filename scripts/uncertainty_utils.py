#封装了所有数学计算和辅助逻辑。
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from monai.metrics import compute_ece

def enable_dropout(model):
    """
    在推理阶段强制开启 Dropout 层 (用于 MC Dropout)
    """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def compute_uncertainty(prob_stack, method='entropy'):
    """
    计算不确定性图
    Args:
        prob_stack: (T, B, C, H, W) 形状的概率堆叠张量
        method: 'entropy' 或 'variance'
    Returns:
        uncertainty_map: (B, H, W)
    """
    # 计算 T 次采样的平均概率 (Bayesian Model Averaging)
    mean_prob = torch.mean(prob_stack, dim=0)  # (B, C, H, W)
    
    if method == 'entropy':
        # Entropy = - sum(p * log(p))
        epsilon = 1e-8
        entropy = -torch.sum(mean_prob * torch.log(mean_prob + epsilon), dim=1)
        return entropy
        
    elif method == 'variance':
        # 计算每个像素、每个类别的方差，然后取平均
        var = torch.var(prob_stack, dim=0)  # (B, C, H, W)
        mean_var = torch.mean(var, dim=1)   # (B, H, W)
        return mean_var
        
    else:
        raise ValueError(f"Unknown uncertainty method: {method}")

def temperature_scaling(logits, temp=1.5):
    """
    简单的温度缩放 (Temperature Scaling)
    注意：理想情况下 temp 应该在验证集上优化得到
    """
    return torch.div(logits, temp)

def save_uncertainty_report(metrics_list, save_path):
    """保存指标到 CSV"""
    df = pd.DataFrame(metrics_list)
    df.to_csv(save_path, index=False)
    print(f"Metrics report saved to {save_path}")

def calculate_ece(probs, labels, n_bins=10):
    """
    计算 Expected Calibration Error (ECE)
    probs: (N, C, ...)
    labels: (N, ...)
    """
    # 将输入展平为 (N_total, C) 和 (N_total,)
    if probs.dim() > 2:
        probs = probs.permute(0, 2, 3, 1).reshape(-1, probs.shape[1])
        labels = labels.flatten()
    
    confidences, predictions = torch.max(probs, 1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=probs.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)

    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        # 属于当前 bin 的样本
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()