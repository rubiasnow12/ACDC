
# Cardiac Segmentation with Uncertainty Estimation & Reliability Analysis

这是一个基于 ACDC 数据集的 2D/3D 心脏分割项目，重点集成了**贝叶斯深度学习 (Probabilistic U-Net)**、**多种不确定性估计方法 (Deep Ensemble / MC Dropout)** 以及**临床评估导向的统计分析与可视化**。

##  功能特性

* **多样化分割模型**:
* 标准模型：U-Net, Attention U-Net (支持 2D/3D)。
* **贝叶斯概率模型**: 集成 **Probabilistic U-Net**，通过潜变量空间建模预测的多样性。


* **多维度不确定性估计**:
* **Deep Ensemble**: 集成 5 折交叉验证模型提供鲁棒的不确定性估计。
* **MC Dropout**: 推理阶段通过 Dropout 采样获取模型预测的变异性。
* **Probabilistic Sampling**: 贝叶斯潜空间采样，学习复杂的条件概率分布。


* **可靠性校准**: 使用 **Temperature Scaling** 优化预测概率，降低预期校准误差 (ECE)。
* **统计分析与评估**:
* 计算 Dice, HD95 (95% 豪斯多夫距离) 及 ECE 指标。
* **错误检测分析**: 绘制 ROC 和 PR 曲线，评估不确定性作为错误检测器的能力。
* **相关性研究**: 分析分割质量 (Dice) 与不确定性 (Entropy) 的相关性，验证临床预警价值。



##  高级可视化与统计

项目提供了丰富的可视化和统计工具，帮助理解模型决策：

### 1. 三视图可视化

脚本会自动生成每个病例的对比图：

* **View 1: 轮廓对比**: 绿色 (GT) 与黄色 (预测) 轮廓叠加，直观展示分割偏差。
* **View 2: 不确定性叠加**: 将热力图叠加在原图上，并用红框标记高风险区域（Top 5% 不确定性区域）。
* **View 3: 热力图**: 纯不确定性分布图（Entropy/Variance），定位模型困惑区域。

### 2. 统计图表

* **可靠性曲线 (Reliability Diagram)**: 评估预测置信度与实际准确率的匹配程度。
* **相关性散点图**: 绘制 Dice vs. Mean Entropy，并进行回归分析。
* **不确定性分布图**: 统计所有病例的平均熵分布。

##  快速开始

### 1. 环境准备

```bash
pip install -r requirements.txt

```

### 2. 数据划分

确保 ACDC 数据集位于 `data/raw/`，然后运行：

```bash
python scripts/split_patients.py  # 基于病理类别进行分层抽样

```

### 3. 模型训练

* **标准/注意力模型**:
```bash
python scripts/train_2d.py --model_choice UNet2D_Attention

```


* **贝叶斯概率模型 (Probabilistic)**:
```bash
python scripts/train_2d_probabilistic.py --model_choice ProbabilisticUNet

```



### 4. 推理、不确定性估计与可视化

* **贝叶斯模型推理**:
```bash
python scripts/predict_uncertainty_2d_probabilistic.py --samples 20 --use_wandb

```


* **集成/MC Dropout 推理**:
```bash
python scripts/predict_uncertainty_2d.py --model_path unet/best_models --use_wandb

```



### 5. 可靠性校准

若需降低 ECE 指标，运行：

```bash
python scripts/calibration.py --model_path path/to/model.pt --fold 1

```

##  目录结构

```text
├── models/
│   ├── probabilistic_unet_2d.py  # 贝叶斯潜变量模型
│   ├── attn_unet_2d.py           # 注意力机制 U-Net
│   └── ...
├── scripts/
│   ├── predict_uncertainty_2d_probabilistic.py # 贝叶斯采样与高级可视化
│   ├── uncertainty_utils.py      # 统计绘图与指标计算工具
│   ├── calibration.py            # 温度缩放校准
│   └── ...
└── outputs/                      # 存储 final_metrics.csv 及统计图表

```