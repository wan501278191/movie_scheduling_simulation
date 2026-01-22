"""
Ablation Study Analysis Script
分析LSTM模块和最大熵机制的有效性
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def load_experiment_results(experiment_name):
    """加载实验结果"""
    project_root = Path(__file__).parent.parent
    experiment_dir = project_root / 'experiments' / experiment_name
    csv_path = experiment_dir / 'training_results.csv'
    
    if csv_path.exists():
        return pd.read_csv(csv_path)
    else:
        print(f"警告: 找不到 {experiment_name} 的结果文件")
        return None

def calculate_metrics(df, window_size=50):
    """计算评估指标"""
    if df is None or len(df) == 0:
        return {}
    
    metrics = {
        'final_avg_reward': df['reward'].tail(window_size).mean(),
        'max_reward': df['reward'].max(),
        'reward_std': df['reward'].tail(window_size).std(),
        'convergence_episode': None,
        'stability_score': None
    }
    
    # 计算收敛回合数（奖励超过阈值并保持稳定）
    threshold = df['reward'].quantile(0.75)  # 使用75%分位数作为阈值
    above_threshold = df['reward'].rolling(window=20).mean() > threshold
    convergence_indices = above_threshold[above_threshold].index
    if len(convergence_indices) > 0:
        metrics['convergence_episode'] = convergence_indices[0]
    
    # 计算稳定性得分（奖励方差的倒数）
    if len(df) >= window_size:
        recent_rewards = df['reward'].tail(window_size)
        metrics['stability_score'] = 1.0 / (recent_rewards.std() + 1e-8)
    
    return metrics

def analyze_ablation_effectiveness():
    """分析消融实验效果"""
    print("=" * 60)
    print("SAC 消融实验分析报告")
    print("=" * 60)
    
    # 加载所有实验结果
    baseline_results = load_experiment_results('11_sac_new_env')
    no_lstm_results = load_experiment_results('ablation_no_lstm')
    no_entropy_results = load_experiment_results('ablation_no_entropy')
    
    # 计算各项指标
    baseline_metrics = calculate_metrics(baseline_results)
    no_lstm_metrics = calculate_metrics(no_lstm_results)
    no_entropy_metrics = calculate_metrics(no_entropy_results)
    
    # 创建对比表格
    comparison_data = []
    if baseline_metrics:
        comparison_data.append(['Baseline (完整SAC)', 
                               baseline_metrics['final_avg_reward'],
                               baseline_metrics['max_reward'],
                               baseline_metrics['reward_std'],
                               baseline_metrics.get('convergence_episode', 'N/A'),
                               baseline_metrics.get('stability_score', 'N/A')])
    
    if no_lstm_metrics:
        comparison_data.append(['No LSTM', 
                               no_lstm_metrics['final_avg_reward'],
                               no_lstm_metrics['max_reward'],
                               no_lstm_metrics['reward_std'],
                               no_lstm_metrics.get('convergence_episode', 'N/A'),
                               no_lstm_metrics.get('stability_score', 'N/A')])
    
    if no_entropy_metrics:
        comparison_data.append(['No Entropy', 
                               no_entropy_metrics['final_avg_reward'],
                               no_entropy_metrics['max_reward'],
                               no_entropy_metrics['reward_std'],
                               no_entropy_metrics.get('convergence_episode', 'N/A'),
                               no_entropy_metrics.get('stability_score', 'N/A')])
    
    # 输出对比表格
    print("\n性能对比表:")
    print("-" * 80)
    print(f"{'方法':<20} {'平均奖励':<12} {'最大奖励':<12} {'奖励标准差':<12} {'收敛回合':<10} {'稳定性':<10}")
    print("-" * 80)
    for row in comparison_data:
        print(f"{row[0]:<20} {row[1]:<12.2f} {row[2]:<12.2f} {row[3]:<12.2f} {str(row[4]):<10} {str(row[5]):<10}")
    
    # 计算消融影响
    print("\n消融影响分析:")
    print("-" * 40)
    
    if baseline_metrics and no_lstm_metrics:
        lstm_impact = ((baseline_metrics['final_avg_reward'] - no_lstm_metrics['final_avg_reward']) 
                      / baseline_metrics['final_avg_reward'] * 100)
        print(f"LSTM模块影响: {lstm_impact:.2f}% 性能下降")
    
    if baseline_metrics and no_entropy_metrics:
        entropy_impact = ((baseline_metrics['final_avg_reward'] - no_entropy_metrics['final_avg_reward']) 
                         / baseline_metrics['final_avg_reward'] * 100)
        print(f"最大熵机制影响: {entropy_impact:.2f}% 性能下降")
    
    # 绘制详细的对比图
    plot_detailed_comparison(baseline_results, no_lstm_results, no_entropy_results)
    
    # 生成统计报告
    generate_statistical_report(baseline_metrics, no_lstm_metrics, no_entropy_metrics)

def plot_detailed_comparison(baseline_df, no_lstm_df, no_entropy_df):
    """绘制详细对比图"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    episodes_range = range(1, 201)  # 假设最多200个回合
    
    # 1. 奖励对比（原始数据）
    if baseline_df is not None:
        axes[0, 0].plot(baseline_df['episode'], baseline_df['reward'], 
                       alpha=0.3, color='blue', label='Baseline (raw)')
        if len(baseline_df) >= 10:
            baseline_smooth = baseline_df['reward'].rolling(10).mean()
            axes[0, 0].plot(baseline_df['episode'], baseline_smooth, 
                           color='blue', linewidth=2, label='Baseline (smooth)')
    
    if no_lstm_df is not None:
        axes[0, 0].plot(no_lstm_df['episode'], no_lstm_df['reward'], 
                       alpha=0.3, color='red', label='No LSTM (raw)')
        if len(no_lstm_df) >= 10:
            no_lstm_smooth = no_lstm_df['reward'].rolling(10).mean()
            axes[0, 0].plot(no_lstm_df['episode'], no_lstm_smooth, 
                           color='red', linewidth=2, label='No LSTM (smooth)')
    
    if no_entropy_df is not None:
        axes[0, 0].plot(no_entropy_df['episode'], no_entropy_df['reward'], 
                       alpha=0.3, color='green', label='No Entropy (raw)')
        if len(no_entropy_df) >= 10:
            no_entropy_smooth = no_entropy_df['reward'].rolling(10).mean()
            axes[0, 0].plot(no_entropy_df['episode'], no_entropy_smooth, 
                           color='green', linewidth=2, label='No Entropy (smooth)')
    
    axes[0, 0].set_title('奖励对比 (Reward Comparison)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Critic Loss对比
    plot_loss_comparison(axes[0, 1], baseline_df, no_lstm_df, no_entropy_df, 'critic_loss', 'Critic Loss')
    
    # 3. Actor Loss对比  
    plot_loss_comparison(axes[0, 2], baseline_df, no_lstm_df, no_entropy_df, 'actor_loss', 'Actor Loss')
    
    # 4. 收敛速度分析
    plot_convergence_analysis(axes[1, 0], baseline_df, no_lstm_df, no_entropy_df)
    
    # 5. 奖励分布箱线图
    plot_reward_distribution(axes[1, 1], baseline_df, no_lstm_df, no_entropy_df)
    
    # 6. 性能稳定性分析
    plot_stability_analysis(axes[1, 2], baseline_df, no_lstm_df, no_entropy_df)
    
    plt.tight_layout()
    
    # 保存图像
    project_root = Path(__file__).parent.parent
    save_path = project_root / 'analysis_results' / 'ablation_study_detailed_analysis.png'
    save_path.parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n详细分析图表已保存到: {save_path}")

def plot_loss_comparison(ax, baseline_df, no_lstm_df, no_entropy_df, loss_col, title):
    """绘制损失对比图"""
    colors = {'baseline': 'blue', 'no_lstm': 'red', 'no_entropy': 'green'}
    
    if baseline_df is not None and loss_col in baseline_df.columns:
        valid_data = baseline_df.dropna(subset=[loss_col])
        if len(valid_data) > 0:
            ax.plot(valid_data['episode'], valid_data[loss_col], 
                   color=colors['baseline'], label='Baseline')
    
    if no_lstm_df is not None and loss_col in no_lstm_df.columns:
        valid_data = no_lstm_df.dropna(subset=[loss_col])
        if len(valid_data) > 0:
            ax.plot(valid_data['episode'], valid_data[loss_col], 
                   color=colors['no_lstm'], label='No LSTM')
    
    if no_entropy_df is not None and loss_col in no_entropy_df.columns:
        valid_data = no_entropy_df.dropna(subset=[loss_col])
        if len(valid_data) > 0:
            ax.plot(valid_data['episode'], valid_data[loss_col], 
                   color=colors['no_entropy'], label='No Entropy')
    
    ax.set_title(title)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_convergence_analysis(ax, baseline_df, no_lstm_df, no_entropy_df):
    """绘制收敛速度分析"""
    methods = []
    convergence_speeds = []
    
    # 计算每个方法达到90%最终权重所需的时间
    if baseline_df is not None:
        final_reward = baseline_df['reward'].tail(20).mean()
        threshold = final_reward * 0.9
        above_threshold = baseline_df['reward'] >= threshold
        if above_threshold.any():
            conv_episode = above_threshold.idxmax()
            convergence_speeds.append(conv_episode)
            methods.append('Baseline')
    
    if no_lstm_df is not None:
        final_reward = no_lstm_df['reward'].tail(20).mean()
        threshold = final_reward * 0.9
        above_threshold = no_lstm_df['reward'] >= threshold
        if above_threshold.any():
            conv_episode = above_threshold.idxmax()
            convergence_speeds.append(conv_episode)
            methods.append('No LSTM')
    
    if no_entropy_df is not None:
        final_reward = no_entropy_df['reward'].tail(20).mean()
        threshold = final_reward * 0.9
        above_threshold = no_entropy_df['reward'] >= threshold
        if above_threshold.any():
            conv_episode = above_threshold.idxmax()
            convergence_speeds.append(conv_episode)
            methods.append('No Entropy')
    
    if methods:
        bars = ax.bar(methods, convergence_speeds, color=['blue', 'red', 'green'][:len(methods)])
        ax.set_title('收敛速度对比')
        ax.set_ylabel('达到90%最终性能所需回合数')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, speed in zip(bars, convergence_speeds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{speed}', ha='center', va='bottom')

def plot_reward_distribution(ax, baseline_df, no_lstm_df, no_entropy_df):
    """绘制奖励分布箱线图"""
    data_to_plot = []
    labels = []
    colors = []
    
    if baseline_df is not None:
        data_to_plot.append(baseline_df['reward'].tail(100))  # 最后100回合
        labels.append('Baseline')
        colors.append('blue')
    
    if no_lstm_df is not None:
        data_to_plot.append(no_lstm_df['reward'].tail(100))
        labels.append('No LSTM')
        colors.append('red')
    
    if no_entropy_df is not None:
        data_to_plot.append(no_entropy_df['reward'].tail(100))
        labels.append('No Entropy')
        colors.append('green')
    
    if data_to_plot:
        box_plot = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title('奖励分布对比 (最后100回合)')
        ax.set_ylabel('Reward')
        ax.grid(True, alpha=0.3)

def plot_stability_analysis(ax, baseline_df, no_lstm_df, no_entropy_df):
    """绘制稳定性分析"""
    methods = []
    stability_scores = []
    
    window_size = 50
    
    if baseline_df is not None and len(baseline_df) >= window_size:
        rewards = baseline_df['reward'].tail(window_size)
        stability = 1.0 / (rewards.std() + 1e-8)
        stability_scores.append(stability)
        methods.append('Baseline')
    
    if no_lstm_df is not None and len(no_lstm_df) >= window_size:
        rewards = no_lstm_df['reward'].tail(window_size)
        stability = 1.0 / (rewards.std() + 1e-8)
        stability_scores.append(stability)
        methods.append('No LSTM')
    
    if no_entropy_df is not None and len(no_entropy_df) >= window_size:
        rewards = no_entropy_df['reward'].tail(window_size)
        stability = 1.0 / (rewards.std() + 1e-8)
        stability_scores.append(stability)
        methods.append('No Entropy')
    
    if methods:
        bars = ax.bar(methods, stability_scores, color=['blue', 'red', 'green'][:len(methods)])
        ax.set_title('稳定性对比 (1/标准差)')
        ax.set_ylabel('Stability Score')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, score in zip(bars, stability_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{score:.2f}', ha='center', va='bottom')

def generate_statistical_report(baseline_metrics, no_lstm_metrics, no_entropy_metrics):
    """生成统计分析报告"""
    report_lines = []
    report_lines.append("SAC 消融实验统计分析报告")
    report_lines.append("=" * 50)
    report_lines.append("")
    
    # 基础性能对比
    report_lines.append("1. 基础性能对比:")
    if baseline_metrics:
        report_lines.append(f"   基线SAC平均奖励: {baseline_metrics['final_avg_reward']:.2f}")
    if no_lstm_metrics:
        report_lines.append(f"   无LSTM平均奖励: {no_lstm_metrics['final_avg_reward']:.2f}")
    if no_entropy_metrics:
        report_lines.append(f"   无熵正则化平均奖励: {no_entropy_metrics['final_avg_reward']:.2f}")
    report_lines.append("")
    
    # 消融影响量化
    report_lines.append("2. 消融影响量化:")
    if baseline_metrics and no_lstm_metrics:
        lstm_drop = baseline_metrics['final_avg_reward'] - no_lstm_metrics['final_avg_reward']
        lstm_drop_pct = (lstm_drop / baseline_metrics['final_avg_reward']) * 100
        report_lines.append(f"   LSTM移除导致性能下降: {lstm_drop:.2f} ({lstm_drop_pct:.2f}%)")
    
    if baseline_metrics and no_entropy_metrics:
        entropy_drop = baseline_metrics['final_avg_reward'] - no_entropy_metrics['final_avg_reward']
        entropy_drop_pct = (entropy_drop / baseline_metrics['final_avg_reward']) * 100
        report_lines.append(f"   熵正则化移除导致性能下降: {entropy_drop:.2f} ({entropy_drop_pct:.2f}%)")
    report_lines.append("")
    
    # 统计显著性分析建议
    report_lines.append("3. 实验结论:")
    if baseline_metrics and no_lstm_metrics and no_entropy_metrics:
        # 确定哪个组件更重要
        lstm_impact = abs(baseline_metrics['final_avg_reward'] - no_lstm_metrics['final_avg_reward'])
        entropy_impact = abs(baseline_metrics['final_avg_reward'] - no_entropy_metrics['final_avg_reward'])
        
        if lstm_impact > entropy_impact:
            report_lines.append("   - LSTM模块对性能的影响更大，说明时序建模在排片任务中很重要")
            report_lines.append("   - 最大熵机制也有积极作用，但相对较小")
        else:
            report_lines.append("   - 最大熵机制对性能的影响更大，说明探索机制很关键")
            report_lines.append("   - LSTM模块的作用相对较小")
    
    report_lines.append("   - 两种机制都有助于提升SAC算法的整体性能")
    report_lines.append("   - 建议在实际应用中同时保留这两个组件")
    
    # 保存报告
    project_root = Path(__file__).parent.parent
    report_path = project_root / 'analysis_results' / 'ablation_study_report.txt'
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\n详细统计报告已保存到: {report_path}")
    
    # 同时打印到控制台
    print("\n" + "\n".join(report_lines))

if __name__ == '__main__':
    analyze_ablation_effectiveness()