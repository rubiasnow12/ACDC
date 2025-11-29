# Cardiac Segmentation with Uncertainty Estimation & Reliability Analysis

这是一个基于 ACDC 数据集的 2D 心脏分割项目，集成了**不确定性估计 (Deep Ensemble / MC Dropout)** 和 **可靠性校准 (Temperature Scaling)**。项目支持 WandB 在线可视化，可直观分析模型在高风险区域的表现。

##  功能特性

  * **基础分割**: 基于 U-Net / Attention U-Net 的准确分割。
  * **不确定性估计**:
      * **Deep Ensemble**: 集成 5 折交叉验证模型，生成高质量不确定性热力图。
      * **MC Dropout**: 单模型多次采样估计。
  * **可靠性校准**: 使用 Temperature Scaling 优化模型概率输出（ECE 指标）。
  * **高级可视化**: 集成 WandB，提供交互式分割图、不确定性热力图及统计报表。

##  环境依赖

请确保安装以下依赖库：

```bash
pip install -r requirements.txt
```

主要依赖：`torch`, `monai`, `pytorch-lightning`, `wandb`, `torchmetrics`, `matplotlib`

##  数据准备

1.  请将 ACDC 数据集放置在项目根目录下的 `data/raw/ACDC/database` 文件夹中：

    ```text
    project/
    ├── data/
    │   └── raw/
    │       └── ACDC/
    │           └── database/
    │               ├── training/   <-- 包含 patientXXX 文件夹
    │               └── testing/
    ```

2.  生成数据划分文件 (`splits.pkl`)：

    ```bash
    python scripts/split_patients.py
    ```

    *这将在 `data/` 目录下生成划分文件，确保 5 折交叉验证的一致性。*

##  快速开始

### 1\. 模型训练 (5-Fold Cross Validation)

运行训练脚本，这将自动进行 5 折交叉验证并保存最佳模型：

```bash
python scripts/train_2d.py --model_choice UNet2D_Attention --maximum_epochs 400
```

  * **输出**: 模型权重保存在 `unet/best_models/`。
  * **日志**: TensorBoard 日志保存在 `unet/tb_logs/`，WandB 日志（如配置）在线可见。

### 2\. 不确定性估计与推理 (Uncertainty Estimation)

利用训练好的模型（支持 Deep Ensemble 模式）进行推理，生成分割图和不确定性图。

```bash
python scripts/predict_uncertainty_2d.py \
    --model_path unet/best_models \
    --output_dir outputs/ensemble_results \
    --use_wandb
```

  * **--model\_path**: 指向包含 `.pt` 模型文件的目录。
  * **--use\_wandb**: 强烈推荐开启，可在 WandB 网页端查看交互式热力图。
  * **Deep Ensemble**: 如果目录下有多个 Fold 的模型，脚本会自动识别并开启集成模式。

**结果解读 (WandB / Output Dir)**:

  * **Pred vs GT**: 黄色轮廓为预测，绿色轮廓为金标准。
  * **Uncertainty Map**: 颜色越亮（黄/红）表示模型越不确定（通常在边界处）。
  * **High Risk Area**: 红色标记区域为模型极其不确定的高风险区。

### 3\. 可靠性校准 (Calibration)

如果模型存在“过度自信”问题（Dice 高但 ECE 高），使用此脚本进行温度缩放校准。

```bash
python scripts/calibration.py \
    --model_path unet/best_models/UNet2D_Attention_Best_0.3_Fold_1.pt \
    --fold 1
```

  * 脚本会自动计算最佳温度 $T$。
  * 输出校准前后的 NLL (Negative Log Likelihood) 和 ECE (Expected Calibration Error)。
  * 生成校准后的模型文件 `*_calibrated.pt`。

##  配置文件

主要参数位于 `configs/config.yaml`，你可以修改：

  * `uncertainty_estimation`: 选择 `deep_ensemble` 或 `mc_dropout`。
  * `training`: 学习率、Batch Size 等。

##  目录结构说明

```text
.
├── configs/             # 配置文件
├── data/                # 数据存放区
├── models/              # 网络结构定义 (U-Net, Attn U-Net)
├── scripts/             # 核心脚本
│   ├── train_2d.py              # 训练入口
│   ├── predict_uncertainty_2d.py# 不确定性推理
│   ├── calibration.py           # 可靠性校准
│   ├── data_2d.py               # 数据加载 (Pathlib 增强版)
│   └── ...
├── unet/                # 训练产出 (权重, 日志)
└── outputs/             # 推理产出 (可视化图表)
```

