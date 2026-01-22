"""
Ablation Study Training Script
进行LSTM有效性和最大熵机制有效性的消融实验
"""

import os
import sys
import numpy as np
import pandas as pd
import argparse
import json
import random
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# 环境配置
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from envs.simsche import CinemaGym
from tools.memory import MemoryDict


def setup_dual_loggers(experiment_dir):
    stats_logger = logging.getLogger('ablation_training')
    stats_logger.setLevel(logging.INFO)
    if not stats_logger.handlers:
        sfh = logging.FileHandler(os.path.join(experiment_dir, 'ablation_training.log'), encoding='utf-8')
        print("ablation_training path:", os.path.join(experiment_dir, 'ablation_training.log'))
        sfh.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        stats_logger.addHandler(sfh)
    stats_logger.propagate = False

    sim_logger = logging.getLogger('ablation_simulation')
    sim_logger.setLevel(logging.INFO)
    if not sim_logger.handlers:
        dfh = logging.FileHandler(os.path.join(experiment_dir, 'ablation_simulation.log'), encoding='utf-8')
        print("ablation_simulation path:", os.path.join(experiment_dir, 'ablation_simulation.log'))
        dfh.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        sim_logger.addHandler(dfh)
    sim_logger.propagate = False
    return stats_logger, sim_logger


def set_seeds(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    import tensorflow as tf
    tf.random.set_seed(int(seed))


def apply_dynamic_hard_constraint(action, state, total_show_count):
    avg_revenue = state[:, -1, 9]
    min_alloc = np.zeros(len(action))
    worthy_mask = avg_revenue > 100.0
    min_alloc[worthy_mask] = 1.0 / total_show_count
    constrained = np.maximum(action, min_alloc)
    return constrained / np.sum(constrained)


def train_ablation_study(args, ablation_type):
    """
    运行消融实验
    
    Args:
        args: 配置参数
        ablation_type: 消融类型 ('no_lstm' 或 'no_entropy')
    """
    set_seeds(args.random_seed)
    import tensorflow as tf

    # 根据消融类型选择不同的策略类
    if ablation_type == 'no_lstm':
        from policys.sac_policy_ablation_no_lstm import SACAgent
        experiment_name = 'ablation_no_lstm'
    elif ablation_type == 'no_entropy':
        from policys.sac_policy_ablation_no_entropy import SACAgent
        experiment_name = 'ablation_no_entropy'
    else:
        raise ValueError(f"Unknown ablation type: {ablation_type}")

    experiment_dir = os.path.join(project_root, 'experiments', experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    stats_logger, sim_logger = setup_dual_loggers(experiment_dir)

    env = CinemaGym(logger=sim_logger, online_movie_count=args.online_movie_count,
                    total_show_count=args.total_show_count, enable_logging=True)

    agent = SACAgent(online_movie_num=args.online_movie_count,
                     feature_columns=env.sac_feature_columns, look_back_horizon=7,
                     alpha_min=getattr(args, 'alpha_min', 1e-6), 
                     alpha_lr_decay_factor=getattr(args, 'alpha_lr_decay_factor', 0.1))

    memory = MemoryDict(capacity=args.memory_capacity)
    BEST_MODEL_PATH = os.path.join(experiment_dir, 'sac_policy_best.h5')

    # 断点加载逻辑
    if os.path.exists(BEST_MODEL_PATH.replace(".h5", "_actor.weights.h5")):
        agent.load_weights(BEST_MODEL_PATH)
        stats_logger.info(f">>> 加载已有权重 ({ablation_type})。")

    total_steps = 0
    max_income = -1.0
    episode_rewards_list = []
    ten_ep_rewards = []
    all_c_losses, all_a_losses, all_alphas = [], [], []

    stats_logger.info(f"======== {experiment_name} 消融实验启动 ========")

    with tqdm(range(args.num_episodes), desc=f"{ablation_type} Training") as pbar:
        for episode in pbar:
            state = env.reset()
            ep_reward = 0
            curr_ep_c_loss, curr_ep_a_loss = [], []

            for t in range(40):
                sac_obs = state[:, :, :7]

                # 前 start_training_steps 步随机采样
                if total_steps < args.start_training_steps:
                    env_action = np.random.rand(args.online_movie_count);
                    env_action /= np.sum(env_action)
                else:
                    env_action, _ = agent.step(sac_obs[np.newaxis, ...])

                action_final = apply_dynamic_hard_constraint(env_action, state, args.total_show_count)
                next_state, reward, done, info = env.step(action_final)
                memory.push(sac_obs, action_final, reward, next_state[:, :, :7], done, info)

                # 内存够一个 batch 时才更新
                if total_steps >= args.start_training_steps and len(memory.buffer) >= args.batch_size:
                    c_l, a_l, alpha = agent.learn(memory.sample(args.batch_size))
                    curr_ep_c_loss.append(c_l);
                    curr_ep_a_loss.append(a_l);
                    all_alphas.append(alpha)

                state = next_state
                ep_reward += reward
                total_steps += 1
                if done: break

            episode_rewards_list.append(ep_reward)
            ten_ep_rewards.append(ep_reward)
            if curr_ep_c_loss:
                all_c_losses.append(np.mean(curr_ep_c_loss))
                all_a_losses.append(np.mean(curr_ep_a_loss))

            if ep_reward > max_income:
                max_income = ep_reward
                stats_logger.info(f">>> 新高点！保存最佳模型 ({ablation_type})。")
                stats_logger.info(f"    历史单回合最高收入: {max_income:.1f} (万)")
                agent.save_weights(BEST_MODEL_PATH)

            if (episode + 1) % 10 == 0:
                avg_10 = np.mean(ten_ep_rewards)
                c_l_val = all_c_losses[-1] if all_c_losses else 0
                a_l_val = all_a_losses[-1] if all_a_losses else 0
                stats_logger.info(
                    f"E {episode + 1}/{args.num_episodes} | 10回合平均收入: {avg_10:.1f} | Loss(C/A): {c_l_val:.2f}/{a_l_val:.2f} | Alpha: {all_alphas[-1] if all_alphas else 1.0:.4f}")
                ten_ep_rewards = []

            pbar.set_postfix({'收益': f'{ep_reward:.1f}', '最高': f'{max_income:.1f}'})

    # 绘制训练曲线
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes[0, 0].plot(episode_rewards_list, color='blue', alpha=0.3)
        if len(episode_rewards_list) >= 10:
            axes[0, 0].plot(pd.Series(episode_rewards_list).rolling(10).mean(), color='blue')
        axes[0, 0].set_title("收益 (Reward)");
        axes[0, 0].grid(True)
        if all_c_losses: axes[0, 1].plot(all_c_losses, color='red'); axes[0, 1].set_title("Critic Loss"); axes[
            0, 1].grid(True)
        if all_a_losses: axes[1, 0].plot(all_a_losses, color='green'); axes[1, 0].set_title("Actor Loss"); axes[
            1, 0].grid(True)
        if all_alphas: axes[1, 1].plot(all_alphas, color='orange'); axes[1, 1].set_title("Alpha (Temperature)"); axes[
            1, 1].grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(experiment_dir, "training_curves.png"))
        plt.close()
    except:
        pass

    # 保存结果统计数据
    # 确保所有数组长度一致
    max_len = len(episode_rewards_list)
    
    # 构建DataFrame，确保所有列长度相同
    data_dict = {
        'episode': list(range(max_len)),
        'reward': episode_rewards_list,
        'critic_loss': [np.nan] * max_len,
        'actor_loss': [np.nan] * max_len,
        'alpha': [np.nan] * max_len
    }
    
    # 填充实际的数据值
    for i, val in enumerate(all_c_losses):
        if i < max_len:
            data_dict['critic_loss'][i] = val
    for i, val in enumerate(all_a_losses):
        if i < max_len:
            data_dict['actor_loss'][i] = val
    for i, val in enumerate(all_alphas):
        if i < max_len:
            data_dict['alpha'][i] = val
    
    results_df = pd.DataFrame(data_dict)
    results_df.to_csv(os.path.join(experiment_dir, 'training_results.csv'), index=False)
    
    return episode_rewards_list, all_c_losses, all_a_losses, all_alphas


def compare_ablation_results():
    """比较不同消融实验的结果"""
    base_dir = os.path.join(project_root, 'experiments')
    
    # 加载各实验结果
    results = {}
    for exp_name in ['ablation_no_lstm', 'ablation_no_entropy']:
        csv_path = os.path.join(base_dir, exp_name, 'training_results.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            results[exp_name] = df
    
    if not results:
        print("未找到消融实验结果文件")
        return
    
    # 绘制对比图
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 收益对比
    for exp_name, df in results.items():
        label = 'No LSTM' if 'no_lstm' in exp_name else 'No Entropy'
        axes[0, 0].plot(df['episode'], df['reward'], alpha=0.3, label=f'{label} (raw)')
        if len(df) >= 10:
            rolling_mean = df['reward'].rolling(10).mean()
            axes[0, 0].plot(df['episode'], rolling_mean, label=f'{label} (smooth)', linewidth=2)
    axes[0, 0].set_title("收益对比 (Reward Comparison)")
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Critic Loss对比
    for exp_name, df in results.items():
        label = 'No LSTM' if 'no_lstm' in exp_name else 'No Entropy'
        valid_data = df.dropna(subset=['critic_loss'])
        if len(valid_data) > 0:
            axes[0, 1].plot(valid_data['episode'], valid_data['critic_loss'], label=label)
    axes[0, 1].set_title("Critic Loss 对比")
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Actor Loss对比
    for exp_name, df in results.items():
        label = 'No LSTM' if 'no_lstm' in exp_name else 'No Entropy'
        valid_data = df.dropna(subset=['actor_loss'])
        if len(valid_data) > 0:
            axes[1, 0].plot(valid_data['episode'], valid_data['actor_loss'], label=label)
    axes[1, 0].set_title("Actor Loss 对比")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Alpha对比
    for exp_name, df in results.items():
        label = 'No LSTM' if 'no_lstm' in exp_name else 'No Entropy'
        valid_data = df.dropna(subset=['alpha'])
        if len(valid_data) > 0:
            axes[1, 1].plot(valid_data['episode'], valid_data['alpha'], label=label)
    axes[1, 1].set_title("Alpha 对比")
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    comparison_path = os.path.join(base_dir, 'ablation_comparison.png')
    plt.savefig(comparison_path)
    plt.close()
    print(f"消融实验对比图已保存到: {comparison_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='11_sac_new_env.json', help='配置文件路径')
    parser.add_argument('--ablation_type', type=str, choices=['no_lstm', 'no_entropy', 'both'], 
                       default='both', help='消融实验类型')
    args, _ = parser.parse_known_args()

    # 智能寻找配置文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = [
        args.config,
        os.path.join(current_dir, args.config),
        os.path.join(current_dir, '../configs', args.config),
        os.path.join(current_dir, 'configs', args.config),
        os.path.join(os.getcwd(), args.config)
    ]

    config_path = None
    for path in candidate_paths:
        if os.path.exists(path):
            config_path = path
            break

    if config_path is None:
        raise FileNotFoundError(f"找不到配置文件: {args.config}")

    print(f"正在加载配置文件: {config_path}")

    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        for k, v in config.items():
            setattr(args, k, v)

    # 运行消融实验
    all_results = {}
    
    if args.ablation_type in ['no_lstm', 'both']:
        print("\n=== 开始 LSTM 消融实验 ===")
        rewards, c_losses, a_losses, alphas = train_ablation_study(args, 'no_lstm')
        all_results['no_lstm'] = {
            'rewards': rewards,
            'critic_losses': c_losses,
            'actor_losses': a_losses,
            'alphas': alphas
        }
    
    if args.ablation_type in ['no_entropy', 'both']:
        print("\n=== 开始 最大熵机制 消融实验 ===")
        rewards, c_losses, a_losses, alphas = train_ablation_study(args, 'no_entropy')
        all_results['no_entropy'] = {
            'rewards': rewards,
            'critic_losses': c_losses,
            'actor_losses': a_losses,
            'alphas': alphas
        }
    
    # 生成对比报告
    print("\n=== 生成消融实验对比报告 ===")
    compare_ablation_results()
    
    # 打印关键统计信息
    print("\n=== 消融实验结果摘要 ===")
    for exp_type, results_dict in all_results.items():
        rewards = results_dict['rewards']
        avg_reward = np.mean(rewards[-50:]) if len(rewards) >= 50 else np.mean(rewards)
        max_reward = np.max(rewards)
        print(f"{exp_type}: 平均奖励(最后50轮)={avg_reward:.2f}, 最大奖励={max_reward:.2f}")