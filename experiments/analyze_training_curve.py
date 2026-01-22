# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import platform

# ==============================================================================
# 0. 科研绘图风格配置 (绝对安全版)
# ==============================================================================
def set_pub_style():
    # === 核心修改：删除所有自定义字体设置 ===
    # 不再指定任何 font.family 列表，直接使用 Matplotlib 内部默认字体
    # 这能100%解决 "findfont" 报错，因为默认字体是内置在库里的
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False 

    # 论文级参数设置 (保持美观)
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['figure.dpi'] = 300

# 配色方案 (Nature 风格)
COLORS = {
    "Reward": "#1F77B4",  # 经典蓝
    "Critic": "#D62728",  # 经典红
    "Actor": "#2CA02C",   # 经典绿
    "Alpha": "#FF7F0E"    # 经典橙
}

# ==============================================================================
# 1. 解析日志函数
# ==============================================================================
def parse_training_log(log_path):
    """解析 sac_training.log 提取数据"""
    data = {'Episode': [], 'Reward': [], 'Critic_Loss': [], 'Actor_Loss': [], 'Alpha': []}
    
    # 正则表达式
    pattern = re.compile(
        r'E\s+(?P<ep>\d+)/\d+\s+\|\s+'
        r'(?:平均收入|10局均值|10回合平均收入|Average Revenue):\s+(?P<rew>[-\d\.]+).*'
        r'Loss\(C/A\):\s+(?P<critic>[-\d\.]+)/(?P<actor>[-\d\.]+).*'
        r'Alpha:\s+(?P<alpha>[-\d\.]+)'
    )
    
    try:
        try:
            with open(log_path, 'r', encoding='utf-8') as f: lines = f.readlines()
        except UnicodeDecodeError:
            with open(log_path, 'r', encoding='gbk') as f: lines = f.readlines()
    except FileNotFoundError:
        return pd.DataFrame(data)

    for line in lines:
        match = pattern.search(line)
        if match:
            data['Episode'].append(int(match.group('ep')))
            data['Reward'].append(float(match.group('rew')))
            data['Critic_Loss'].append(float(match.group('critic')))
            data['Actor_Loss'].append(float(match.group('actor')))
            data['Alpha'].append(float(match.group('alpha')))
            
    df = pd.DataFrame(data)
    if not df.empty:
        df = df.drop_duplicates(subset=['Episode'], keep='last').sort_values('Episode').reset_index(drop=True)
    return df

# ==============================================================================
# 2. 核心绘图函数 (防抖 + 顺滑)
# ==============================================================================
def plot_metric_subplot(ax, df, x_col, y_col, color, title, y_label, window=60):
    if len(df) < 2: return

    smoothed_mean = df[y_col].ewm(span=window, adjust=False).mean()
    smoothed_std = df[y_col].rolling(window=window, min_periods=1).std()

    # 1. 原始数据 (高透明度)
    ax.plot(df[x_col], df[y_col], color=color, alpha=0.1, linewidth=0.8, zorder=1)

    # 2. 阴影带
    ax.fill_between(df[x_col], 
                    smoothed_mean - smoothed_std, 
                    smoothed_mean + smoothed_std, 
                    color=color, alpha=0.1, linewidth=0, zorder=2)

    # 3. 顺滑趋势线
    ax.plot(df[x_col], smoothed_mean, color=color, linewidth=2.5, label='Trend', zorder=3)

    # 使用英文标签，无需特殊字体支持
    ax.set_title(title, fontweight='bold', pad=12)
    ax.set_xlabel('Episodes')
    ax.set_ylabel(y_label)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle=':', alpha=0.6)

# ==============================================================================
# 3. 主程序
# ==============================================================================
def plot_all_metrics():
    # 设置风格 (不再加载任何外部字体)
    set_pub_style()
    
    # 自动查找日志
    candidates = ['sac_training.log', os.path.join('experiments', 'sac_training.log')]
    log_path = None
    for p in candidates:
        if os.path.exists(p):
            log_path = p
            break
    
    if not log_path:
        print("[ERROR] sac_training.log not found.")
        return

    df = parse_training_log(log_path)
    if df.empty:
        print("[ERROR] Data is empty.")
        return

    print(f"[INFO] Loaded {len(df)} episodes.")

    # 动态平滑窗口
    SMOOTH_W = max(20, int(len(df) * 0.05)) 

    # ------------------------------------------------------------------
    # PART A: 组合图 (Composite Figure)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    plot_metric_subplot(axes[0, 0], df, 'Episode', 'Reward', COLORS['Reward'], 
                        'Average Revenue Trend', 'Revenue (10k)', window=SMOOTH_W)
    
    plot_metric_subplot(axes[0, 1], df, 'Episode', 'Critic_Loss', COLORS['Critic'], 
                        'Critic Loss Convergence', 'MSE Loss', window=SMOOTH_W)
    
    plot_metric_subplot(axes[1, 0], df, 'Episode', 'Actor_Loss', COLORS['Actor'], 
                        'Actor Loss Convergence', 'Loss', window=SMOOTH_W)
    
    # Alpha
    axes[1, 1].plot(df['Episode'], df['Alpha'], color=COLORS['Alpha'], linewidth=2.5)
    axes[1, 1].set_title('Entropy Coefficient (Alpha)', fontweight='bold', pad=12)
    axes[1, 1].set_xlabel('Episodes')
    axes[1, 1].set_ylabel('Alpha Value')
    axes[1, 1].spines['top'].set_visible(False)
    axes[1, 1].spines['right'].set_visible(False)
    axes[1, 1].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout(pad=3.0)
    plt.savefig('sac_training_smooth_report.png', bbox_inches='tight')
    plt.close()

    # ------------------------------------------------------------------
    # PART B: 独立高清子图
    # ------------------------------------------------------------------
    # 格式: (文件名后缀, 列名, 颜色, 标题, Y轴标签, 是否强制平滑)
    metrics_config = [
        ('reward', 'Reward', COLORS['Reward'], 'Average Revenue Trend', 'Revenue (10k)', True),
        ('critic', 'Critic_Loss', COLORS['Critic'], 'Critic Loss Convergence', 'MSE Loss', True),
        ('actor', 'Actor_Loss', COLORS['Actor'], 'Actor Loss Convergence', 'Loss', True),
        ('alpha', 'Alpha', COLORS['Alpha'], 'Entropy Coefficient (Alpha)', 'Alpha Value', False) 
    ]

    for suffix, col, color, title, ylabel, use_smooth in metrics_config:
        fig_single, ax_single = plt.subplots(figsize=(8, 6))
        
        if use_smooth:
            plot_metric_subplot(ax_single, df, 'Episode', col, color, 
                                title, ylabel, window=SMOOTH_W)
        else:
            ax_single.plot(df['Episode'], df[col], color=color, linewidth=2.5)
            ax_single.set_title(title, fontweight='bold', pad=12)
            ax_single.set_xlabel('Episodes')
            ax_single.set_ylabel(ylabel)
            ax_single.spines['top'].set_visible(False)
            ax_single.spines['right'].set_visible(False)
            ax_single.grid(True, linestyle=':', alpha=0.6)

        save_name = f'sac_curve_{suffix}.png'
        plt.tight_layout()
        plt.savefig(save_name, bbox_inches='tight', dpi=300)
        plt.close(fig_single)
        print(f"[Output] Saved: {save_name}")

    print("\n[SUCCESS] Figures generated using default system fonts.")

if __name__ == "__main__":
    plot_all_metrics()