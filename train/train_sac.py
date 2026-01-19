# python train/train_sac.py --config configs/08_sac_base_2000_test.json
# python train/train_sac.py --config configs/01_sac_base_2000.json
# python train/train_sac.py --config configs/02_sac_base_2000_test.json
# python train/train_sac.py --config configs/03_sac_base_2000_test.json
# python train/train_sac.py --config configs/04_sac_base_2000_test.json
# python train/train_sac.py --config configs/04_sac_base_2000_test.json


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
from policys.sac_policy import SACAgent
from tools.memory import MemoryDict


def setup_dual_loggers(experiment_dir):
    stats_logger = logging.getLogger('sac_training')
    stats_logger.setLevel(logging.INFO)
    if not stats_logger.handlers:
        sfh = logging.FileHandler(os.path.join(experiment_dir, 'sac_training.log'), encoding='utf-8')
        print("sac_training path:", os.path.join(experiment_dir, 'sac_training.log'))
        sfh.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        stats_logger.addHandler(sfh)
    stats_logger.propagate = False

    sim_logger = logging.getLogger('sac_simulation')
    sim_logger.setLevel(logging.INFO)
    if not sim_logger.handlers:
        dfh = logging.FileHandler(os.path.join(experiment_dir, 'sac_simulation.log'), encoding='utf-8')
        print("sac_training path:", os.path.join(experiment_dir, 'sac_simulation.log'))
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


def train_sac_only(args):
    set_seeds(args.random_seed)
    import tensorflow as tf

    experiment_dir = os.path.join(project_root, 'experiments', Path(args.config).stem)
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

    # === 【还原以前的断点加载逻辑】 ===
    if os.path.exists(BEST_MODEL_PATH.replace(".h5", "_actor.weights.h5")):
        agent.load_weights(BEST_MODEL_PATH)
        stats_logger.info(">>> 加载已有权重。")

    total_steps = 0
    max_income = -1.0
    episode_rewards_list = []
    ten_ep_rewards = []
    all_c_losses, all_a_losses, all_alphas = [], [], []

    stats_logger.info(f"======== 训练启动: {Path(args.config).stem} ========")

    with tqdm(range(args.num_episodes), desc="Training") as pbar:
        for episode in pbar:
            state = env.reset()
            ep_reward = 0
            curr_ep_c_loss, curr_ep_a_loss = [], []

            for t in range(40):
                sac_obs = state[:, :, :7]

                # 遵循你以前的逻辑：前 start_training_steps 步雷打不动地随机采样
                if total_steps < args.start_training_steps:
                    env_action = np.random.rand(args.online_movie_count);
                    env_action /= np.sum(env_action)
                else:
                    env_action, _ = agent.step(sac_obs[np.newaxis, ...])

                action_final = apply_dynamic_hard_constraint(env_action, state, args.total_show_count)
                next_state, reward, done, info = env.step(action_final)
                memory.push(sac_obs, action_final, reward, next_state[:, :, :7], done, info)

                # === 【核心修复点】只有内存够一个 batch 时才更新，防止崩溃 ===
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
                stats_logger.info(">>> 新高点！保存最佳模型。")
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

    # === 【画图逻辑，一个字都没改你的】 ===
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
        plt.savefig(os.path.join(experiment_dir, "sac_training_curves.png"))
        plt.close()
    except:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 修改1: 去掉 required=True，加上 default，防止不传参报错
    parser.add_argument('--config', type=str, default='11_sac_new_env.json', help='配置文件路径')
    args, _ = parser.parse_known_args()

    # 修改2: 智能寻找文件路径 (支持当前目录、configs目录、以及绝对路径)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = [
        args.config,  # 命令行指定的路径
        os.path.join(current_dir, args.config),  # 当前脚本同级目录
        os.path.join(current_dir, '../configs', args.config),  # configs子目录
        os.path.join(current_dir, 'configs', args.config),  # configs子目录
        os.path.join(os.getcwd(), args.config)  # 工作目录
    ]

    config_path = None
    for path in candidate_paths:
        if os.path.exists(path):
            config_path = path
            break

    if config_path is None:
        raise FileNotFoundError(f"找不到配置文件: {args.config}，请检查文件位置！")

    print(f"正在加载配置文件: {config_path}")

    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        # 将 json 参数注入 args
        for k, v in config.items():
            setattr(args, k, v)

    # 开始训练
    train_sac_only(args)
