# scripts/uncertainty_utils.py
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monai.metrics import compute_hausdorff_distance
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# --- 保留您原有的工具函数 ---
def enable_dropout(model):
    """在推理阶段强制开启 Dropout 层 (用于 MC Dropout)"""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def compute_uncertainty(prob_stack, method='entropy'):
    """计算不确定性图"""
    mean_prob = torch.mean(prob_stack, dim=0)
    if method == 'entropy':
        epsilon = 1e-8
        return -torch.sum(mean_prob * torch.log(mean_prob + epsilon), dim=0)
    elif method == 'variance':
        return torch.mean(torch.var(prob_stack, dim=0), dim=0)
    return None

def temperature_scaling(logits, temp=1.5):
    """简单的温度缩放"""
    return torch.div(logits, temp)

def calculate_ece(probs, labels, n_bins=10):
    """计算 Expected Calibration Error (ECE)"""
    if probs.dim() == 4:
        probs = probs.permute(0, 2, 3, 1).reshape(-1, probs.shape[1])
        labels = labels.flatten()
    confidences, predictions = torch.max(probs, 1)
    accuracies = predictions.eq(labels)
    ece = torch.zeros(1, device=probs.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece.item()

# --- 新增的可视化与评估指标 ---
def calculate_hd95(pred, gt, num_classes=4):
    """计算 95% 豪斯多夫距离 (HD95)"""
    if isinstance(pred, np.ndarray): pred = torch.from_numpy(pred)
    if isinstance(gt, np.ndarray): gt = torch.from_numpy(gt)
    pred_ext = pred.unsqueeze(0).unsqueeze(0) if pred.ndim == 2 else pred.unsqueeze(1)
    gt_ext = gt.unsqueeze(0).unsqueeze(0) if gt.ndim == 2 else gt.unsqueeze(1)
    pred_oh = F.one_hot(pred_ext.long(), num_classes).squeeze(1).permute(0, 3, 1, 2).float()
    gt_oh = F.one_hot(gt_ext.long(), num_classes).squeeze(1).permute(0, 3, 1, 2).float()
    hd95_vals = compute_hausdorff_distance(pred_oh[:, 1:], gt_oh[:, 1:], percentile=95)
    valid_vals = hd95_vals[~torch.isinf(hd95_vals)]
    return valid_vals.mean().item() if valid_vals.numel() > 0 else 0.0

def plot_reliability_diagram(probs_flat, labels_flat, n_bins=10, save_path=None):
    """绘制可靠性曲线"""
    confidences = np.max(probs_flat, axis=1)
    predictions = np.argmax(probs_flat, axis=1)
    accuracies = (predictions == labels_flat)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        bin_accs.append(np.mean(accuracies[mask]) if mask.any() else 0)
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect Calibration")
    plt.bar(bin_boundaries[:-1], bin_accs, width=1/n_bins, alpha=0.3, edgecolor="black", color="blue", align='edge')
    plt.xlabel("Confidence"); plt.ylabel("Accuracy"); plt.title("Reliability Diagram")
    if save_path: plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_uncertainty_performance(errors, uncertainties, save_dir):
    """绘制 ROC 和 PR 曲线"""
    fpr, tpr, _ = roc_curve(errors, uncertainties)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(errors, uncertainties)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}'); plt.title('Uncertainty ROC'); plt.legend()
    plt.subplot(1, 2, 2); plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.3f}'); plt.title('Uncertainty PR'); plt.legend()
    plt.savefig(f"{save_dir}/uncertainty_analysis.png", bbox_inches='tight')
    plt.close()