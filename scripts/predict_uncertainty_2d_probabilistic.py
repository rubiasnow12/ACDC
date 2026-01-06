import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from pathlib import Path
import monai.transforms as mt
from torch.utils.data import DataLoader
from torchmetrics.functional import dice_score

# -------------------------------------------------------------------------
# 1. 环境配置与依赖导入
# -------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# 关键修复：导入 Train2D 以便 torch.load 能够识别旧模型中的类定义
try:
    from scripts.train_2d import Train2D
except ImportError:
    # 如果是因为脚本运行位置问题，尝试定义一个桩类防止报错
    class Train2D(nn.Module): pass

from models.probabilistic_unet_2d import ProbabilisticUnet
from scripts.uncertainty_utils import *
from scripts.data_2d import test_loader_ACDC

# -------------------------------------------------------------------------
# 2. 辅助工具函数
# -------------------------------------------------------------------------

class ModelWithTemperature(nn.Module):
    """用于加载经过温度缩放校准的模型"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    def forward(self, x):
        return self.model(x) / self.temperature

def normalize_map(m):
    """归一化不确定性热力图"""
    if m.max() == m.min(): return m
    m_capped = np.clip(m, 0, np.percentile(m, 95)) # 抑制离群点
    return (m_capped - m_capped.min()) / (m_capped.max() - m_capped.min() + 1e-8)

def draw_contours(img_rgb, mask, color):
    """在 RGB 图像上绘制分割轮廓线"""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_rgb, contours, -1, color, 2)
    return img_rgb

def generate_advanced_visuals(img, mask_gt, mask_pred, unc_map, case_id, save_dir):
    """生成三视图：GT对比、不确定性叠加、热力图"""
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img_rgb = np.stack([img_norm]*3, axis=-1)
    img_rgb = (img_rgb * 255).astype(np.uint8).copy()
    
    # 视图1: GT(绿色) vs 预测(黄色)
    view1 = draw_contours(img_rgb.copy(), mask_gt > 0, (0, 255, 0))
    view1 = draw_contours(view1, mask_pred > 0, (255, 255, 0))
    
    # 视图2: 不确定性叠加
    unc_norm = normalize_map(unc_map)
    unc_heatmap = cv2.applyColorMap((unc_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    view2 = cv2.addWeighted(img_rgb, 0.6, unc_heatmap, 0.4, 0)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1); plt.imshow(view1); plt.title("GT(Green) vs Pred(Yellow)"); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(view2); plt.title("Uncertainty Overlay"); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(unc_norm, cmap='jet'); plt.title("Uncertainty Map"); plt.axis('off')
    
    os.makedirs(os.path.join(save_dir, "vis"), exist_ok=True)
    plt.savefig(os.path.join(save_dir, "vis", f"{case_id}.png"), bbox_inches='tight')
    plt.close()

def Pad_images(image):
    """填充图像至16的倍数，确保设备对齐"""
    b, c, h, w = image.shape
    nx, ny = (16 - (h % 16)) + h, (16 - (w % 16)) + w
    res = torch.full((b, c, nx, ny), image.min(), device=image.device, dtype=image.dtype)
    xx, yy = (nx - h) // 2, (ny - w) // 2
    res[:, :, xx:xx + h, yy:yy + w] = image
    return res, (xx, yy)

def UnPad_images(image, indices, org_shape):
    """将填充后的图像还原"""
    b, c, h, w = org_shape
    xx, yy = indices
    return image[:, :, xx:xx + h, yy:yy + w]

# -------------------------------------------------------------------------
# 3. 推理逻辑
# -------------------------------------------------------------------------

def run_inference():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="unet/best_models", help="权重目录")
    parser.add_argument("--output_dir", default="outputs/bayesian_results", help="保存目录")
    parser.add_argument("--fold", type=int, default=1, help="测试的 Fold")
    parser.add_argument("--samples", type=int, default=20, help="采样次数 T=20")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 动态加载模型
    search_path = Path(args.model_path)
    # 只加载特定 Fold 的所有 .pt 文件（包含 Ensemble 的可能）
    model_files = list(search_path.glob(f"*Fold_{args.fold}*.pt"))
    if not model_files:
        print(f"Error: No models found for Fold {args.fold}")
        return

    models = []
    for mp in model_files:
        print(f"Loading: {mp.name}")
        # 使用 weights_only=False 处理旧版本全对象保存的模型
        net = torch.load(mp, map_location=device)
        net.eval()
        models.append(net)

    # 2. 准备数据
    test_loader = DataLoader(test_loader_ACDC(None, transform=mt.Compose([mt.ToTensorD(keys=["image", "mask"])])), batch_size=1)

    # 容器
    metrics_report, all_px_errors, all_px_unc, all_probs, all_gts = [], [], [], [], []

    print(f"Starting Inference (T={args.samples})...")
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader)):
            img_org, mask_gt = batch["image"].to(device), batch["mask"].long().to(device).squeeze(1)
            img_padded, p_idx = Pad_images(img_org)
            org_shape = (1, 4, img_org.shape[2], img_org.shape[3])
            
            probs_list = []
            for model in models:
                # A. 贝叶斯推断逻辑 (Probabilistic U-Net)
                if isinstance(model, ProbabilisticUnet):
                    model(img_padded, training=False)
                    for _ in range(args.samples):
                        logits = model.sample(testing=True)
                        p = torch.softmax(UnPad_images(logits, p_idx, org_shape), dim=1)
                        probs_list.append(p.cpu())
                # B. MC Dropout 或常规推理
                else:
                    enable_dropout(model) # 即使是非贝叶斯也强制开启 dropout 采样
                    for _ in range(args.samples if "mcdropout" in args.output_dir.lower() else 1):
                        logits = model(img_padded)
                        p = torch.softmax(UnPad_images(logits, p_idx, org_shape), dim=1)
                        probs_list.append(p.cpu())

            # 3. 统计不确定性与聚合
            probs_stack = torch.cat(probs_list, dim=0) # [T, 4, H, W]
            mean_prob = torch.mean(probs_stack, dim=0) # [4, H, W]
            pred_mask = torch.argmax(mean_prob, dim=0) # [H, W]
            
            # 计算熵图
            unc_entropy = compute_uncertainty(probs_stack, 'entropy')
            
            # 4. 计算指标 (Dice, HD95, ECE)
            d_val = dice_score(pred_mask.unsqueeze(0).unsqueeze(0), mask_gt.cpu().unsqueeze(0).unsqueeze(0), bg=True).item()
            h_val = calculate_hd95(pred_mask.numpy(), mask_gt.cpu().numpy().squeeze())
            e_val = calculate_ece(mean_prob.unsqueeze(0), mask_gt.cpu())

            # 5. 可视化
            case_id = f"case_{idx:03d}"
            generate_advanced_visuals(img_org.cpu().squeeze().numpy(), mask_gt.cpu().numpy().squeeze(), 
                                      pred_mask.numpy(), unc_entropy.numpy(), case_id, args.output_dir)

            # 6. 收集全局评估数据
            err_mask = (pred_mask.numpy() != mask_gt.cpu().numpy().squeeze()).astype(np.int32)
            all_px_errors.append(err_mask.flatten())
            all_px_unc.append(unc_entropy.numpy().flatten())
            all_probs.append(mean_prob.numpy().transpose(1, 2, 0).reshape(-1, 4))
            all_gts.append(mask_gt.cpu().numpy().flatten())

            metrics_report.append({"id": case_id, "dice": d_val, "hd95": h_val, "ece": e_val})

    # 7. 保存结果并绘制曲线
    if metrics_report:
        df = pd.DataFrame(metrics_report)
        df.to_csv(os.path.join(args.output_dir, "final_metrics.csv"), index=False)
        
        # 可靠性曲线 (ECE 验证)
        plot_reliability_diagram(np.concatenate(all_probs), np.concatenate(all_gts), 
                                 save_path=os.path.join(args.output_dir, "reliability_diagram.png"))
        
        # ROC/PR 曲线 (不确定性作为错误检测器的效果)
        plot_uncertainty_performance(np.concatenate(all_px_errors), np.concatenate(all_px_unc), args.output_dir)
        
        print(f"\nSummary for Fold {args.fold}:")
        print(f"Mean Dice: {df['dice'].mean():.4f} | Mean HD95: {df['hd95'].mean():.4f} | Mean ECE: {df['ece'].mean():.4f}")

if __name__ == "__main__":
    run_inference()