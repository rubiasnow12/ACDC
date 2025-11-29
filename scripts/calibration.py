import torch
from torch import nn, optim
import torch.nn.functional as F
import os
import yaml
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
import monai.transforms as mt
import pytorch_lightning as pl
# 导入你的项目模块
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.attn_unet_2d import AttU_Net2D
from models.unet_2d import Unet_2d
from scripts.data_2d import val_loader_ACDC  # 确保这里能导入 val_loader

class ModelWithTemperature(nn.Module):
    """
    包装器：将原模型包裹起来，添加一个可学习的温度参数
    """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        # 初始化温度为 1.5 (经验值) 或 1.0
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        核心逻辑：logits / temperature
        如果是分割任务，temperature 会广播到每个像素
        """
        return logits / self.temperature

    def set_temperature(self, valid_loader, device):
        """
        在验证集上优化 T
        """
        self.to(device)
        self.model.eval()
        nll_criterion = nn.CrossEntropyLoss().to(device)
        ece_criterion = _ECELoss().to(device)

        # 1. 收集验证集所有的 Logits 和 Labels
        logits_list = []
        labels_list = []
        
        print("Collecting logits on validation set...")
        with torch.no_grad():
            for batch in tqdm(valid_loader):
                # 1. 准备数据
                if isinstance(batch, dict):
                    input = batch["image"].float() # 先保持在 CPU 方便 Pad 处理，或者直接转 tensor
                    label = batch["mask"].long().to(device).squeeze(1)
                else: 
                    input, label = batch
                    input = input.float()
                    label = label.long().to(device).squeeze(1)

                # 2.  Padding: 确保输入是 16 的倍数
                input_padded, pad_indices = Pad_images(input)
                input_padded = input_padded.to(device)
                
                # 3. 模型推理
                logits_padded = self.model(input_padded)
                
                # 4.  Unpadding: 恢复到原始尺寸，以便和 Label 对齐
                # 注意：UnPad 需要原始输入的 shape (B, C, H, W)，这里假设 logits 通道数不变，只恢复 H, W
                # 构造一个 shape 传给 UnPad_images: (Batch, Channel, Orig_H, Orig_W)
                orig_shape = (logits_padded.shape[0], logits_padded.shape[1], input.shape[2], input.shape[3])
                logits = UnPad_images(logits_padded, pad_indices, orig_shape)
                
                # 5. 收集结果
                logits_list.append(logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1]).cpu())
                labels_list.append(label.flatten().cpu())

        # 合并所有数据
        logits = torch.cat(logits_list).to(device)
        labels = torch.cat(labels_list).to(device)

        # 2. 计算校准前的指标
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print(f'Before temperature scaling - NLL: {before_temperature_nll:.3f}, ECE: {before_temperature_ece:.3f}')

        # 3. 优化温度参数 T
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            # loss = CrossEntropy(logits / T, label)
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        # 4. 计算校准后的指标
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        
        print(f'Optimal temperature: {self.temperature.item():.3f}')
        print(f'After temperature scaling - NLL: {after_temperature_nll:.3f}, ECE: {after_temperature_ece:.3f}')

        return self.temperature.item()

# 添加这两个函数来处理尺寸问题
def Pad_images(image):
    b, c, h, w = image.shape
    new_x = (16 - (h % 16)) + h
    new_y = (16 - (w % 16)) + w
    # 如果已经是 16 的倍数，直接返回
    if new_x == h and new_y == w:
        return image, (0, 0)
        
    result = torch.full((b, c, new_x, new_y), image.min(), dtype=image.dtype)
    xx = (new_x - h) // 2
    yy = (new_y - w) // 2
    result[:, :, xx:xx + h, yy:yy + w] = image
    return result, (xx, yy)

def UnPad_images(image, indices, org_shape):
    b, c, h, w = org_shape
    xx, yy = indices
    return image[:, :, xx:xx + h, yy:yy + w]
class _ECELoss(nn.Module):
    """
    计算 ECE (Expected Calibration Error) 的辅助类
    """
    def __init__(self, n_bins=15):
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece

class Train2D(pl.LightningModule):
    def __init__(self):
        super(Train2D, self).__init__()
        # 这里可以是空的，因为 torch.load 会直接把保存的 self.net 覆盖回来
        self.net = None 

    def forward(self, x):
        return self.net(x)
    
def run_calibration():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the best model .pt file")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--fold", type=int, default=1, help="Which fold validation set to use")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载原始模型
    print(f"Loading model from {args.model_path}")
    original_model = torch.load(args.model_path, map_location=device)
    
    # 2. 包装模型
    scaled_model = ModelWithTemperature(original_model)
    
    # 3. 准备验证数据
    # 注意：这里需要根据 fold 获取对应的 validation set
    # 由于 data_2d.py 需要 split 索引，这里简单模拟一下或复用 split 逻辑
    # 为简化，这里假设已经有办法获取 val_loader，或者使用下面的模拟加载
    from sklearn.model_selection import KFold
    import numpy as np
    
    # 简单的复用逻辑：重新生成一次 split 拿到 val_idx
    # 警告：这依赖于 random_state 一致，确保和训练时一样
    dataset_len = 100 # ACDC 有 100 个病人
    splits = KFold(n_splits=5, shuffle=True, random_state=42) # 必须和 train_2d.py 一致
    train_idx, val_idx = list(splits.split(np.arange(dataset_len)))[args.fold - 1]
    
    val_transform = mt.Compose([
        mt.ToTensorD(keys=["image", "mask"], allow_missing_keys=False)
    ])
    
    # 使用项目中的 loader
    val_loader = val_loader_ACDC(val_index=val_idx, transform=val_transform)
    # 必须使用 batch_size=1 避免 padding 问题干扰校准，或者自己处理 padding
    val_dataloader = DataLoader(val_loader, batch_size=1, shuffle=False)

    # 4. 执行温度缩放优化
    optimal_T = scaled_model.set_temperature(val_dataloader, device)
    
    # 5. 保存结果
    # 可以选择保存整个包装后的模型，或者只保存 T 值
    save_path = args.model_path.replace(".pt", "_calibrated.pt")
    torch.save(scaled_model, save_path)
    print(f"Calibrated model saved to {save_path}")
    
    # 也可以把 T 写入一个 txt
    txt_path = args.model_path.replace(".pt", "_temperature.txt")
    with open(txt_path, "w") as f:
        f.write(str(optimal_T))

if __name__ == "__main__":
    run_calibration()