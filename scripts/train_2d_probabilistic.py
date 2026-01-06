import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import monai.transforms as mt
from torchmetrics.functional import dice_score

# -------------------------------------------------------------------------
# 1. 路径修复与模型导入
# -------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.attn_unet_2d import AttU_Net2D
from models.unet_2d import Unet_2d
from models.probabilistic_unet_2d import ProbabilisticUnet
from scripts.data_2d import train_loader_ACDC, val_loader_ACDC

# -------------------------------------------------------------------------
# 2. PyTorch Lightning 训练模块
# -------------------------------------------------------------------------

class Train2D(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        # 模型选择
        if self.hparams.model_choice == "UNet2D":
            self.net = Unet_2d(drop=self.hparams.dropout_rate)
        elif self.hparams.model_choice == "UNet2D_Attention":
            self.net = AttU_Net2D(drop=self.hparams.dropout_rate)
        elif self.hparams.model_choice == "ProbabilisticUNet":
            # 贝叶斯 U-Net 配置
            self.net = ProbabilisticUnet(input_channels=1, num_classes=4)
        
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, mask=None, training=True):
        if isinstance(self.net, ProbabilisticUnet):
            return self.net(x, mask, training=training)
        return self.net(x)

    def training_step(self, batch, batch_idx):
        img, mask = batch["image"].float(), batch["mask"]
        
        if isinstance(self.net, ProbabilisticUnet):
            # 贝叶斯训练：最小化 ELBO (Evidence Lower Bound)
            self.net(img, mask, training=True)
            loss = self.net.elbo(mask)
            self.log('train_elbo', loss, on_step=True, prog_bar=True, batch_size=img.shape[0])
            return loss
        else:
            # 普通 U-Net 训练：交叉熵
            out = self(img)
            loss = self.loss_func(out, mask.squeeze(1).long())
            self.log('train_loss', loss, on_step=True, prog_bar=True, batch_size=img.shape[0])
            return loss

    def validation_step(self, batch, batch_idx):
        img, mask = batch["image"].float(), batch["mask"].long()
        
        if isinstance(self.net, ProbabilisticUnet):
            # 验证时从先验空间采样一次进行评估
            self.net(img, training=False)
            out = self.net.sample(testing=True)
        else:
            out = self(img)
            
        # 计算 Dice 指标 (4类平均)
        soft_out = torch.softmax(out, dim=1)
        d_val = dice_score(soft_out, mask.squeeze(1), bg=True).mean()
        self.log('val_dice', d_val, on_epoch=True, prog_bar=True, batch_size=img.shape[0])
        return d_val

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10
        )
        return {
            "optimizer": optimizer, 
            "lr_scheduler": scheduler, 
            "monitor": "val_dice"
        }

# -------------------------------------------------------------------------
# 3. 数据流水线与执行逻辑
# -------------------------------------------------------------------------

def run_training():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_choice", default="ProbabilisticUNet", type=str)
    parser.add_argument("--dropout_rate", default=0.3, type=float)
    parser.add_argument("--lr", default=0.0005, type=float)
    parser.add_argument("--maximum_epochs", default=100, type=int)
    args = parser.parse_args()

    # 核心修复：添加 SpatialPadD 确保所有图像尺寸统一为 384x384 (16的倍数)
    # 这解决了 "stack expects each tensor to be equal size" 的错误
    data_transforms = mt.Compose([
        # 核心修复：只保留填充和转张量逻辑
        mt.SpatialPadD(keys=["image", "mask"], spatial_size=[384, 384], mode="edge"),
        mt.ToTensorD(keys=["image", "mask"]),
    ])

    checkpoint_path = "./unet/best_models"
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # 5折交叉验证
    splits = KFold(n_splits=5, shuffle=True, random_state=42)
    # 获取数据集（此处假设 train_loader_ACDC 支持传入自定义 transform）
    full_dataset = train_loader_ACDC(None, transform=data_transforms)

    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(full_dataset)))):
        print(f"\n>>>> Starting Fold {fold + 1} <<<<")
        
        train_loader = DataLoader(
            train_loader_ACDC(train_idx, transform=data_transforms), 
            batch_size=8, shuffle=True
        )
        val_loader = DataLoader(
            val_loader_ACDC(val_idx, transform=data_transforms), 
            batch_size=1
        )
        
        model = Train2D(vars(args))
        
        # 设置训练器
        trainer = pl.Trainer(
            max_epochs=args.maximum_epochs, 
            gpus=1, 
            logger=TensorBoardLogger("./unet/tb_logs", name=f"{args.model_choice}_Fold_{fold+1}"),
            log_every_n_steps=10 # ACDC 数据集较小，调低日志频率以看到进度
        )
        
        trainer.fit(model, train_loader, val_loader)
        
        # 保存该 Fold 的最佳权重
        final_save_path = os.path.join(checkpoint_path, f"{args.model_choice}_Fold_{fold+1}.pt")
        torch.save(model.net, final_save_path)
        print(f"Fold {fold+1} model saved to {final_save_path}")

if __name__ == "__main__":  
    run_training()