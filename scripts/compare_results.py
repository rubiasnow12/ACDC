# scripts/compare_results.py
import pandas as pd
from scipy import stats
import argparse
import numpy as np

def perform_test(file_a, file_b, metric='dice'):
    df_a = pd.read_csv(file_a).sort_values(by='id')
    df_b = pd.read_csv(file_b).sort_values(by='id')
    
    # 确保 ID 一一对应
    data_a = df_a[metric].values
    data_b = df_b[metric].values
    
    # 执行配对 t 检验 (如果数据服从正态分布)
    t_stat, p_t = stats.ttest_rel(data_a, data_b)
    
    # 执行 Wilcoxon 符号秩检验 (非参数检验，更稳健)
    w_stat, p_w = stats.wilcoxon(data_a, data_b)
    
    mask = np.isfinite(data_a) & np.isfinite(data_b)
    clean_a = data_a[mask]
    clean_b = data_b[mask]

    print(f"--- {metric.upper()} 统计对比 ---")
    print(f"方法 A 均值: {clean_a.mean():.4f}")
    print(f"方法 B 均值: {clean_b.mean():.4f}")
    print(f"Paired T-test P-value: {p_t:.6f}")
    print(f"Wilcoxon P-value: {p_w:.6f}")
    
    if p_w < 0.05:
        print("结论: 差异具有统计学显著性 (p < 0.05)")
    else:
        print("结论: 差异不具有统计学显著性 (p >= 0.05)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_a", required=True, help="第一个指标 CSV 文件")
    parser.add_argument("--file_b", required=True, help="第二个指标 CSV 文件")
    parser.add_argument("--metric", default="dice", help="要对比的指标 (dice, hd95, ece)")
    args = parser.parse_args()
    
    perform_test(args.file_a, args.file_b, args.metric)