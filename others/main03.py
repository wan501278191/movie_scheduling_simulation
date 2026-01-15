import os
import sys
import logging
import warnings

# 1. 设置系统环境变量（屏蔽底层 C++ 日志）
# '0': 显示所有日志 (默认)
# '1': 过滤 INFO
# '2': 过滤 WARNING (建议)
# '3': 过滤 ERROR (最强力屏蔽)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 2. 屏蔽 Python 层的标准警告
warnings.filterwarnings('ignore')

# 3. 针对 TensorFlow Probability 和 Keras 的特定日志屏蔽
# 注意：必须在设置完环境变量后，再导入 tensorflow
import tensorflow as tf
from tensorflow.python.util import deprecation

# 屏蔽 TF 的废弃警告
deprecation._PRINT_DEPRECATION_WARNINGS = False

# 屏蔽 TF 的 Python 日志
tf.get_logger().setLevel(logging.ERROR)

# 4. 屏蔽 absl 日志 (TensorFlow Probability 经常使用这个库输出日志)
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except ImportError:
    pass

import numpy as np
from tqdm import tqdm
import os
import gymnasium as gym
import sys
import pandas as pd
import time
import json
import random

# 导入您的类
from envs.simsche import CinemaGym
from policys import (EfficiencyHeuristic, StaticHeuristic, PredictiveStrategy,
                     GreedyHeuristic)
from policys.sac_policy import SACAgent
from tools import Logger, Memory


# (set_seeds 函数保持不变)
def set_seeds(seed_value):
    seed_value = int(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    try:
        tf.config.experimental.enable_op_determinism()
    except AttributeError:
        pass


# (apply_dynamic_hard_constraint 函数保持不变)
def apply_dynamic_hard_constraint(action: np.ndarray, state: np.ndarray, total_shows: int,
                                  survival_threshold: float = 100.0) -> np.ndarray:
    num_movies = len(action)
    min_pct_for_one_show = 1.0 / total_shows
    avg_revenue_per_show = state[:, -1, 9]
    min_allocations = np.zeros(num_movies)
    worthy_mask = avg_revenue_per_show > survival_threshold
    min_allocations[worthy_mask] = min_pct_for_one_show
    constrained_action = np.maximum(action, min_allocations)
    over_budget = np.sum(constrained_action) - 1.0
    if over_budget <= 0:
        return constrained_action / np.sum(constrained_action)
    above_min_mask = constrained_action > min_allocations
    if not np.any(above_min_mask):
        return constrained_action / np.sum(constrained_action)
    reducible_total = np.sum(constrained_action[above_min_mask] - min_allocations[above_min_mask])
    if reducible_total > 0:
        reduction_ratio = over_budget / reducible_total
        reduction = (constrained_action[above_min_mask] - min_allocations[above_min_mask]) * reduction_ratio
        constrained_action[above_min_mask] -= reduction
    return constrained_action / np.sum(constrained_action)
# ==============================================================================
# 1. 常规市场 (Normal Market) - 共 25 个日期 (去掉了所有备注)
# ==============================================================================
NORMAL_MARKET_DATES = [
    {'date': '2023-09-08', 'case_study_movie': None},
    {'date': '2023-09-15', 'case_study_movie': None},
    {'date': '2023-09-22', 'case_study_movie': None},
    {'date': '2023-10-20', 'case_study_movie': None},
    {'date': '2023-10-27', 'case_study_movie': None},

]

# ==============================================================================
# 2. 热门档期 (Holiday Market) - 共 20 个日期 (精简备注，不带年份)
# ==============================================================================
HOLIDAY_MARKET_DATES = [
    {'date': '2023-09-29', 'case_study_movie': None}, # 中秋国庆档
    {'date': '2023-09-30', 'case_study_movie': None}, # 国庆档
    {'date': '2023-10-01', 'case_study_movie': None}, # 国庆档
    {'date': '2023-10-02', 'case_study_movie': None}, # 国庆档
    {'date': '2023-10-04', 'case_study_movie': None}, # 国庆档

]

SCENARIOS = {
    "常规市场 (Normal Market)": NORMAL_MARKET_DATES,
    "热门档期 (Holiday Market)": HOLIDAY_MARKET_DATES
}

def main_black_horse():
    set_seeds(42)

    EVAL_CONFIG = {
        # 评估时使用的温度系数。
        # 1.0 = 激进/原样输出 (适合比拼胜率，如果训练得好，1.0 是最佳的)
        # 5.0 ~ 10.0 = 温和/分散排片 (如果你发现策略太极端，可以临时调大这个值来平滑)
        'softmax_temp': 1.0,

        # 指定要加载的模型路径
        'model_path': 'experiments/03_sac_base_2000_test/sac_policy_best.h5'
    }

    log_dir = os.path.join("../logs", "main03")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "evaluation.log")
    csv_results_path = os.path.join(log_dir, "evaluation_results.csv")

    all_runs_data = []

    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = current_dir
    except NameError:
        project_root = os.getcwd()

    env_config_path = os.path.join(project_root, 'configs', 'evaluation_env.json')
    try:
        with open(env_config_path, 'r', encoding='utf-8') as f:
            env_config = json.load(f)
    except FileNotFoundError:
        env_config = {'online_movie_count': 8, 'total_show_count': 30}

    EVAL_MOVIE_COUNT = env_config.get('online_movie_count', 8)
    EVAL_SHOW_COUNT = env_config.get('total_show_count', 30)

    # ==============================================================================
    # 【核心修改点 1/2】: 在循环外初始化 SACAgent 并加载权重，消除 UserWarning
    # ==============================================================================
    # --- main_all.py 预加载部分的修正代码 ---
    print("--- 正在预加载 SAC 策略模型 ---")
    sac_agent_instance = SACAgent(
        online_movie_num=EVAL_MOVIE_COUNT,
        feature_columns=['dummy'] * 7,
        look_back_horizon=7
        # 评估阶段不用反向传播，所以 learning_rate 这些不需要传
    )

    # 2. 构造 Dummy State 消除警告
    # 维度逻辑：EVAL_MOVIE_COUNT(8) * 特征维度(17) = 136 (对应你之前的报错信息)
    # 形状: (Batch, Movies, Horizon, Features_per_movie) -> (1, 8, 7, 17)
    # --- 修改后 ---
    # 这里的 7 必须对应 sac_policy.py 里的特征列长度
    correct_feature_dim = 7
    dummy_state = np.zeros((1, EVAL_MOVIE_COUNT, 7, correct_feature_dim), dtype=np.float32)

    try:
        # 这一步会触发 Actor 内部的 Reshape，17 维正好能对上 136 的期望值
        _ = sac_agent_instance.step(dummy_state)
        print("--- 策略模型 Build 完成 ---")
    except Exception as e:
        # 如果还有微小偏差，忽略它，直接进入权重加载
        pass

    # 3. 加载权重
    model_path = EVAL_CONFIG['model_path']  # 从配置取路径
    actor_weights_path = model_path.replace(".h5", "_actor.weights.h5")

    if os.path.exists(actor_weights_path):
        sac_agent_instance.load_weights(model_path)
        print(f"--- 权重加载成功: {model_path} ---")
    else:
        # 加上这个 else 分支，没找到文件时疯狂打印警告
        print("\n" + "!" * 60)
        print(f"!!! 严重警告: 未找到 SAC 模型文件 !!!")
        print(f"!!! 路径: {os.path.abspath(actor_weights_path)}")
        print(f"!!! SAC 策略将使用【未训练的随机参数】进行评估，结果将无效 !!!")
        print("!" * 60 + "\n")

    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        def print_and_log(message, f_handle):
            print(message)
            f_handle.write(message + '\n')
            f_handle.flush()

        header = f"--- 开始一次新的评估: {pd.Timestamp.now()} ---\n"
        print_and_log(header, log_file)
        eval_logger = Logger(logdir=log_dir, is_print=False)

        for scenario_name, dates_list in SCENARIOS.items():
            print_and_log(f"\n{'#' * 60}\n{'#' * 15} 开始评估场景: {scenario_name} {'#' * 15}\n{'#' * 60}", log_file)
            for date_info in dates_list:
                start_date_str = date_info['date']
                ensure_movie_name = date_info['case_study_movie']
                print_and_log(f"\n===== 正在使用起始日期: {start_date_str} 进行评估 =====", log_file)

                policies = {
                    "RL (SAC)": {'type': 'SAC', 'use_constraint': True},
                    "静态启发式策略": {'type': 'Static', 'use_constraint': True},
                    "效率启发式策略": {'type': 'Efficiency', 'use_constraint': True},
                    "贪婪启发式策略": {'type': 'Greedy', 'use_constraint': True}
                }
                date_seed = int(start_date_str.replace("-", ""))
                policies = {
                    "RL (SAC)": {
                        'type': 'SAC',
                        'use_constraint': True,
                        'temp': EVAL_CONFIG['softmax_temp']  # 绑定上面的配置
                    },
                    "静态启发式策略": {'type': 'Static', 'use_constraint': True},
                    "效率启发式策略": {'type': 'Efficiency', 'use_constraint': True},
                    "贪婪启发式策略": {'type': 'Greedy', 'use_constraint': True}
                }
                for name, config in policies.items():
                    set_seeds(42)
                    env = CinemaGym(
                        logger=eval_logger,
                        online_movie_count=EVAL_MOVIE_COUNT,
                        total_show_count=EVAL_SHOW_COUNT,
                        enable_logging=True  # <--- 【修改点】：将此处从默认的 False 改为 True
                    )
                    # 既然 env.reset() 不支持参数，我们先手动设置日期，再 reset
                    env.current_date = pd.to_datetime(start_date_str).date()
                    state = env.reset()
                    # 修正点：使用 pd.to_datetime 确保日期类型与 DataFrame 里的 datetime64[ns] 完全一致
                    target_date = pd.to_datetime(start_date_str)

                    # 强制同步环境日期
                    env.current_date = target_date  # 直接使用 target_date (Timestamp 对象)
                    env._update_current_movies()

                    # 这里是报错的关键：重新获取观测值时，确保内部比较不会出错
                    state = env._get_observation()

                    # ==============================================================================
                    # 在策略循环内直接复用已经初始化好的实例
                    # ==============================================================================
                    agent = None
                    if config['type'] == 'SAC':
                        agent = sac_agent_instance
                    elif name == "静态启发式策略":
                        agent = StaticHeuristic(
                            init_action=np.array([1 / env.online_movie_count] * env.online_movie_count))
                    elif name == "效率启发式策略":
                        # 初始动作
                        init_act = np.array([1 / env.online_movie_count] * env.online_movie_count)
                        current_movie_names = [m['MovieName'] for m in env.current_movie_list]
                        # 这里实例化一次，能保持 40 天内的 action_map 记忆
                        agent = EfficiencyHeuristic(
                            init_action=init_act,
                            next_day_movie_name=current_movie_names,
                            total_show_count=env.total_show_count)
                    elif name == "贪婪启发式策略":
                        agent = GreedyHeuristic()
                    # ==============================================================================

                    daily_incomes = []
                    daily_actions = []
                    for _ in range(30):
                        raw_action = None
                        if config['type'] == 'SAC':
                            sac_input = state[np.newaxis, :, :, :7]

                            # 【修改点】: 从 config 里取温度参数
                            # 如果字典里没有 temp，就默认用 1.0
                            current_temp = config.get('temp', 1.0)

                            raw_action, _ = agent.step(
                                sac_input,
                                deterministic=True,
                                softmax_temp=current_temp
                            )

                        elif name == "效率启发式策略":
                            # 效率策略需要两个参数：状态矩阵和当前的电影名称列表
                            current_movie_names = [m['MovieName'] for m in env.current_movie_list]
                            raw_action = agent.step(state, current_movie_names)
                        else:
                            # 静态和贪婪策略只需要 state
                            raw_action = agent.step(state)

                        # 这里的约束应用部分保持不变
                        if config['use_constraint']:
                            action = apply_dynamic_hard_constraint(raw_action, state, env.total_show_count)
                        else:
                            action = raw_action

                        new_state, reward, done, info = env.step(action, policy_name=name)
                        daily_incomes.append(reward)
                        daily_actions.append(action.tolist())
                        state = new_state
                        if done: break

                    total_income = env.total_income
                    print_and_log(f"{name:<22}: {total_income:.2f} (万)", log_file)

                    all_runs_data.append({"scenario": scenario_name, "start_date": start_date_str, "policy": name,
                                          "case_study_movie": ensure_movie_name, "total_income": total_income,
                                          "daily_incomes": json.dumps(daily_incomes),
                                          "daily_actions": json.dumps(daily_actions)})

    # 后处理与统计
    results_df = pd.DataFrame(all_runs_data)
    results_df.to_csv(csv_results_path, index=False)
    summary = results_df.groupby(['scenario', 'policy'])['total_income'].mean().unstack().round(2)
    overall_summary = results_df.groupby('policy')['total_income'].mean().round(2).sort_values(ascending=False)

    with open(log_file_path, 'a', encoding='utf-8') as log_file:
        print_and_log(f"\n\n{'=' * 60}\n评估完成! 各情景下平均总收入 (万元):", log_file)
        print_and_log(summary.to_string(), log_file)
        print_and_log(f"\n{'-' * 60}\n所有日期的总体平均总收入 (万元):", log_file)
        print_and_log(overall_summary.to_string(), log_file)
        print_and_log(f"{'=' * 60}", log_file)


if __name__ == '__main__':
    main_black_horse()