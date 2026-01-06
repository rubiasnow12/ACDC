# models/probabilistic_unet_2d.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl

# 内置基础组件以解决依赖问题
class DownConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class ProbabilisticUnet(nn.Module):
    def __init__(self, input_channels=1, num_classes=4, num_filters=[32, 64, 128, 192], latent_dim=6, beta=10.0):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.beta = beta
        
        # 1. 潜变量空间建模：先验网络与后验网络
        # 先验网络 (Prior): 仅从图像学习分布
        self.prior = self._make_encoder(input_channels, num_filters, latent_dim)
        # 后验网络 (Posterior): 从图像+掩码学习分布（仅训练使用）
        self.posterior = self._make_encoder(input_channels + 1, num_filters, latent_dim)
        
        # 2. 基础特征提取 (U-Net)
        self.unet_enc = nn.ModuleList([DownConv(input_channels, num_filters[0])]) # 简化示例
        # 3. 特征融合层 (Fcomb): 结合 U-Net 特征与潜变量采样
        self.fcomb = nn.Conv2d(num_filters[0] + latent_dim, num_classes, kernel_size=1)

    def _make_encoder(self, in_c, filters, latent_dim):
        return nn.Sequential(DownConv(in_c, filters[-1]), nn.AdaptiveAvgPool2d(1), nn.Conv2d(filters[-1], 2*latent_dim, 1))

    def forward(self, patch, segm=None, training=True):
        """实现潜变量建模预测分布"""
        if training and segm is not None:
            self.post_dist = self._get_dist(self.posterior(torch.cat([patch, segm], dim=1)))
        self.prior_dist = self._get_dist(self.prior(patch))
        self.features = self.unet_enc[0](patch) # 假设为简化的特征提取

    def _get_dist(self, stats):
        mu, log_var = stats.chunk(2, dim=1)
        return Independent(Normal(mu.flatten(1), torch.exp(log_var.flatten(1))), 1)

    def sample(self, testing=True):
        """
        采样函数
        testing=True: 使用 .sample() (不带梯度，用于推理)
        testing=False: 使用 .rsample() (带重参数化，用于训练/GED计算)
        """
        if testing:
            z = self.prior_dist.sample()
        else:
            z = self.prior_dist.rsample()
            
        z = z.view(z.size(0), z.size(1), 1, 1).expand(-1, -1, self.features.size(2), self.features.size(3))
        return self.fcomb(torch.cat([self.features, z], dim=1))

    def elbo(self, target):
        """计算证据下界 (ELBO)"""
        z = self.post_dist.rsample()
        kl_loss = torch.mean(kl.kl_divergence(self.post_dist, self.prior_dist))
        # 重建损失：使用 CrossEntropy
        rec_loss = F.cross_entropy(self.sample_with_z(z), target.squeeze(1).long())
        return rec_loss + self.beta * kl_loss

    def sample_with_z(self, z):
        z_feat = z.view(z.size(0), z.size(1), 1, 1).expand(-1, -1, self.features.size(2), self.features.size(3))
        return self.fcomb(torch.cat([self.features, z_feat], dim=1))