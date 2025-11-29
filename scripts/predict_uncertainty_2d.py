import os
import sys
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import monai.transforms as mt
from torch.utils.data import DataLoader
from torchmetrics.functional import dice_score
import torch.nn.functional as F
import wandb
import pytorch_lightning as pl
# -------------------------------------------------------------------------
# 1. 路径修复与模型导入
# -------------------------------------------------------------------------
# 将项目根目录加入路径，确保能导入 models
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# 尝试导入模型（根据文件名可能需要微调）
try:
    from models.attn_unet_2d import AttU_Net2D
    from models.unet_2d import Unet_2d
    from scripts.data_2d import test_loader_ACDC
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure your project structure is correct.")
    sys.exit(1)

# -------------------------------------------------------------------------
# 2. 核心工具函数 (不确定性、ECE、辅助功能)
# -------------------------------------------------------------------------

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def enable_dropout(model):
    """推理时强制开启 Dropout"""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def compute_uncertainty(prob_stack, method='entropy'):
    """计算不确定性"""
    # prob_stack: (T, C, H, W)
    mean_prob = torch.mean(prob_stack, dim=0)  # (C, H, W)
    
    if method == 'entropy':
        epsilon = 1e-8
        entropy = -torch.sum(mean_prob * torch.log(mean_prob + epsilon), dim=0) # (H, W)
        return entropy
    elif method == 'variance':
        var = torch.var(prob_stack, dim=0) # (C, H, W)
        mean_var = torch.mean(var, dim=0)  # (H, W)
        return mean_var
    else:
        raise ValueError(f"Unknown method: {method}")

def calculate_ece(probs, labels, n_bins=10):
    """计算 Expected Calibration Error"""
    # probs: (N, C, H, W) -> 需要展平
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

def save_nifti(data, affine, path):
    """保存 .nii.gz 文件"""
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()
    if data.ndim == 3 and data.shape[0] == 1:
        data = np.squeeze(data)
    nifti_img = nib.Nifti1Image(data.astype(np.float32), affine)
    nib.save(nifti_img, path)

# -------------------------------------------------------------------------
# 3. 高级可视化与统计函数 
# -------------------------------------------------------------------------

def normalize_map(m, cap_percentile=95):
    """归一化并截断极端值"""
    if m.max() == m.min():
        return m
    cap_val = np.percentile(m, cap_percentile)
    m_capped = np.clip(m, 0, cap_val)
    return (m_capped - m_capped.min()) / (m_capped.max() - m_capped.min() + 1e-8)

def draw_contours(img_rgb, mask, color=(0, 255, 0), thickness=1):
    """在图像上绘制轮廓"""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_rgb, contours, -1, color, thickness)
    return img_rgb

def generate_advanced_visuals(img, mask_gt, mask_pred, unc_map, case_id, save_dir, method_name):
    """
    生成三视图：
    1. GT(绿) vs Pred(黄) 轮廓对比
    2. 预测结果 + 不确定性热力图叠加 + 高风险区域(红框)
    3. 纯不确定性热力图
    """
    # 准备基础 RGB 图
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img_rgb = np.stack([img_norm]*3, axis=-1)
    img_rgb = (img_rgb * 255).astype(np.uint8).copy()
    
    # --- View 1: 边界对比 ---
    view1 = img_rgb.copy()
    if mask_gt is not None:
        view1 = draw_contours(view1, (mask_gt > 0).astype(np.uint8), color=(0, 255, 0), thickness=2) # 绿 GT
    view1 = draw_contours(view1, (mask_pred > 0).astype(np.uint8), color=(255, 255, 0), thickness=2) # 黄 Pred
    
    # --- View 2: 不确定性叠加 ---
    unc_norm = normalize_map(unc_map, cap_percentile=95)
    unc_heatmap = cv2.applyColorMap((unc_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    view2 = cv2.addWeighted(img_rgb, 0.6, unc_heatmap, 0.4, 0)
    
    # 标记高风险区域 (Top 5%)
    threshold = np.percentile(unc_map, 95)
    high_risk_mask = (unc_map > threshold).astype(np.uint8)
    view2 = draw_contours(view2, high_risk_mask, color=(0, 0, 255), thickness=1) # 红 High Risk

    # --- 绘图 ---
    fig = plt.figure(figsize=(18, 6))
    
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(view1)
    ax1.set_title(f"Case {case_id}: GT(Green) vs Pred(Yellow)", fontsize=12)
    ax1.axis('off')
    
    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(view2)
    ax2.set_title("Uncertainty Overlay + High Risk(Red)", fontsize=12)
    ax2.axis('off')
    
    ax3 = plt.subplot(1, 3, 3)
    im = ax3.imshow(unc_norm, cmap='jet', vmin=0, vmax=1)
    ax3.set_title("Normalized Uncertainty Map", fontsize=12)
    ax3.axis('off')
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    # 保存
    vis_dir = os.path.join(save_dir, "vis", method_name)
    os.makedirs(vis_dir, exist_ok=True)
    plt.savefig(os.path.join(vis_dir, f"{case_id}.png"), bbox_inches='tight', dpi=150)
    plt.close()
    
    # 返回统计数据
    high_risk_area = np.sum(high_risk_mask)
    roi_area = np.sum(mask_gt > 0) + 1e-8
    return {
        "high_risk_ratio": high_risk_area / roi_area,
        "mean_unc": np.mean(unc_map),
        "p95_unc": threshold
    }

def generate_statistics_plots(metrics_df, save_dir):
    """生成统计分析图表"""
    stats_dir = os.path.join(save_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)
    
    # 1. 相关性分析 (Dice vs Mean Uncertainty)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=metrics_df, x="mean_entropy", y="dice", hue="high_risk_ratio", palette="viridis")
    sns.regplot(data=metrics_df, x="mean_entropy", y="dice", scatter=False, color="red")
    plt.title("Correlation: Segmentation Quality vs Uncertainty")
    plt.xlabel("Mean Entropy")
    plt.ylabel("Dice Score")
    plt.savefig(os.path.join(stats_dir, "dice_vs_uncertainty.png"))
    plt.close()
    
    # 2. 不确定性分布
    plt.figure(figsize=(8, 6))
    sns.histplot(data=metrics_df, x="mean_entropy", kde=True, bins=20)
    plt.title("Distribution of Case-level Uncertainty")
    plt.savefig(os.path.join(stats_dir, "uncertainty_histogram.png"))
    plt.close()

# -------------------------------------------------------------------------
# 4. 主逻辑
# -------------------------------------------------------------------------

def Pad_images(image):
    b, c, h, w = image.shape
    new_x = (16 - (h % 16)) + h
    new_y = (16 - (w % 16)) + w
    result = torch.full((b, c, new_x, new_y), image.min())
    xx = (new_x - h) // 2
    yy = (new_y - w) // 2
    result[:, :, xx:xx + h, yy:yy + w] = image
    return result, (xx, yy)

def UnPad_images(image, indices, org_shape):
    b, c, h, w = org_shape
    xx, yy = indices
    return image[:, :, xx:xx + h, yy:yy + w]

# -------------------------------------------------------------------------
#  修复 torch.load 找不到 Train2D 的问题
# 必须定义这个类，以便 pickle 能正确反序列化加载模型
# -------------------------------------------------------------------------
class Train2D(pl.LightningModule):
    def __init__(self):
        super(Train2D, self).__init__()
        # 这里可以是空的，因为 torch.load 会直接把保存的 self.net 覆盖回来
        self.net = None 

    def forward(self, x):
        return self.net(x)
    
def run_inference():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--model_path", default="unet/best_models", help="Directory containing model checkpoints")
    parser.add_argument("--output_dir", default="outputs/uncertainty_results", help="Directory to save results")
    parser.add_argument("--fold", type=int, default=1, help="Fold to use for single model/MC Dropout")
    parser.add_argument("--use_wandb", action="store_true", help="Enable WandB logging")
    args = parser.parse_args()

    if args.use_wandb:
        wandb.init(project="ACDC_Uncertainty_Vis", name=f"Fold_{args.fold}_Inference")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载配置
    if os.path.exists(args.config):
        cfg = load_config(args.config)
        unc_cfg = cfg.get("uncertainty_estimation", {})
        methods = unc_cfg.get("methods", [])
    else:
        print("Warning: Config file not found, using default MC Dropout settings.")
        methods = ["mc_dropout"]

    mc_dropout_enabled = "mc_dropout" in methods
    deep_ensemble_enabled = "deep_ensemble" in methods
    
    # 确定采样次数 T
    T = 20 if mc_dropout_enabled else 1
    if deep_ensemble_enabled:
        T = 1
        print("Mode: Deep Ensemble Enabled")
    elif mc_dropout_enabled:
        print(f"Mode: MC Dropout Enabled (T={T})")
    else:
        print("Mode: Standard Inference (No Uncertainty)")

    # 准备数据
    test_transform = mt.Compose([
        mt.ToTensorD(keys=["image", "mask"], allow_missing_keys=False)
    ])
    # 注意：这里调用的是 scripts.data_2d.test_loader_ACDC
    test_loader = DataLoader(test_loader_ACDC(test_index=None, transform=test_transform), batch_size=1, shuffle=False)

    # 加载模型
    model_files = []
    search_path = Path(args.model_path)
    if not search_path.exists():
        print(f"Error: Model path {search_path} does not exist.")
        sys.exit(1)

    if deep_ensemble_enabled:
        # 查找所有 fold 的模型
        model_files = list(search_path.glob("*_Best_*_Fold_*.pt"))
        print(f"Found {len(model_files)} models for ensemble.")
    else:
        # 查找指定 fold 的模型
        found = list(search_path.glob(f"*_Fold_{args.fold}.pt"))
        if found:
            model_files.append(found[0])
            print(f"Loaded single model: {found[0].name}")
        else:
            print(f"Error: No model found for Fold {args.fold} in {args.model_path}")
            sys.exit(1)

    models = []
    for mp in model_files:
        net = torch.load(mp, map_location=device)
        net.eval()
        models.append(net)

    # 准备输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "nifti"), exist_ok=True)
    
    metrics_report = []
    soft = torch.nn.Softmax(dim=1)

    print("Starting inference...")
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader)):
            img_org = batch["image"].to(device)
            mask = batch["mask"].long().to(device).squeeze(1) # (1, H, W)
            
            # Padding
            img_padded, pad_indices = Pad_images(img_org)
            img_padded = img_padded.to(device)
            
            probs_list = []
            
            # 模型推理
            for model in models:
                if mc_dropout_enabled:
                    enable_dropout(model)
                    loop_iters = T
                else:
                    model.eval()
                    loop_iters = 1
                
                for _ in range(loop_iters):
                    logits = model(img_padded)
                    prob = soft(logits)
                    # Unpad
                    prob = UnPad_images(prob, pad_indices, (1, 4, img_org.shape[2], img_org.shape[3]))
                    probs_list.append(prob.cpu())

            # 堆叠预测: (Total_T, C, H, W)
            probs_stack = torch.cat(probs_list, dim=0)
            
            # 计算平均预测与不确定性
            mean_prob = torch.mean(probs_stack, dim=0) # (C, H, W)
            pred_mask = torch.argmax(mean_prob, dim=0) # (H, W)
            unc_entropy = compute_uncertainty(probs_stack, method='entropy')
            unc_variance = compute_uncertainty(probs_stack, method='variance')
            
            # 计算 ECE
            ece_score = calculate_ece(mean_prob.unsqueeze(0), mask.cpu())

            # 计算 Dice (用于分析)
            dice_val = dice_score(
                pred_mask.cpu().unsqueeze(0).unsqueeze(0), 
                mask.cpu().unsqueeze(0).unsqueeze(0),
                bg=True, no_fg_score=0.0, reduction='elementwise_mean'
            ).item()

            base_name = f"case_{idx:03d}"
            
            # 1. 保存 NIfTI (可选)
            # affine = np.eye(4)
            # save_nifti(img_org.squeeze(), affine, os.path.join(args.output_dir, "nifti", f"{base_name}_img.nii.gz"))
            # save_nifti(pred_mask, affine, os.path.join(args.output_dir, "nifti", f"{base_name}_pred.nii.gz"))
            # save_nifti(unc_entropy, affine, os.path.join(args.output_dir, "nifti", f"{base_name}_unc_entropy.nii.gz"))
            # save_nifti(unc_variance, affine, os.path.join(args.output_dir, "nifti", f"{base_name}_unc_var.nii.gz"))

            # 2. 生成高级可视化 (覆盖原有的简单绘图)
            img_np = img_org.cpu().squeeze().numpy()
            mask_gt_np = mask.cpu().numpy().squeeze() # 确保是 (H, W)
            pred_np = pred_mask.cpu().numpy()
            unc_np = unc_entropy.cpu().numpy()
            
            method_name = "mc_dropout" if mc_dropout_enabled else ("ensemble" if deep_ensemble_enabled else "single")
            vis_stats = generate_advanced_visuals(
                img_np, mask_gt_np, pred_np, unc_np, 
                case_id=base_name, 
                save_dir=args.output_dir, 
                method_name=method_name
            )

            #  WandB 日志记录
            if args.use_wandb:
                # 定义类别标签 (根据 ACDC 数据集)
                class_labels = {0: "BG", 1: "RV", 2: "MYO", 3: "LV"}
                
                wandb.log({
                    # 1. 交互式分割图 (原图 + 预测 + GT)
                    f"visualization/{base_name}": wandb.Image(
                        img_np, 
                        masks={
                            "predictions": {
                                "mask_data": pred_np,
                                "class_labels": class_labels
                            },
                            "ground_truth": {
                                "mask_data": mask_gt_np,
                                "class_labels": class_labels
                            }
                        }, 
                        caption=f"{base_name}: Pred vs GT"
                    ),
                    
                    # 2. 不确定性热力图
                    f"uncertainty_map/{base_name}": wandb.Image(
                        plt.imshow(unc_np, cmap='jet'), 
                        caption="Uncertainty Heatmap"
                    ),
                    
                    # 3. 标量指标
                    "metrics/dice": dice_val,
                    "metrics/mean_entropy": vis_stats["mean_unc"],
                    "metrics/high_risk_ratio": vis_stats["high_risk_ratio"]
                })
                plt.close() #                   关闭 plt.imshow 创建的图，防止内存泄漏
            
            # 3. 收集指标
            metrics_report.append({
                "id": base_name,
                "dice": dice_val,
                "ece": ece_score,
                "mean_entropy": vis_stats["mean_unc"],
                "p95_entropy": vis_stats["p95_unc"],
                "high_risk_ratio": vis_stats["high_risk_ratio"]
            })

    # 循环结束: 生成统计图表
    if metrics_report:
        df = pd.DataFrame(metrics_report)
        generate_statistics_plots(df, args.output_dir)
        save_path = os.path.join(args.output_dir, "final_metrics.csv")
        df.to_csv(save_path, index=False)
        print(f"Inference complete. Results saved to {args.output_dir}")
    else:
        print("No metrics collected.")

if __name__ == "__main__":
    run_inference()