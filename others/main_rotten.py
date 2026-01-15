import tensorflow as tf
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
    avg_revenue_per_show = state[:, -1, 8]
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
# 60部“烂片/高开低走”电影的完整列表
# ==============================================================================
ROTTEN_MARKET_DATES = [
    # --- 2019 ---
    {'date': '2019-06-01', 'case_study_movie': '哆啦A梦：大雄的月球探险记'},
    {'date': '2019-08-09', 'case_study_movie': '上海堡垒'},
    {'date': '2019-09-20', 'case_study_movie': '小Q'},

    # --- 2020 ---
    {'date': '2020-07-24', 'case_study_movie': '多力特的奇幻冒险'},
    {'date': '2020-07-24', 'case_study_movie': '喋血战士'},
    {'date': '2020-07-31', 'case_study_movie': '抵达之谜'},
    {'date': '2020-08-07', 'case_study_movie': '1917'},
    {'date': '2020-10-30', 'case_study_movie': '数码宝贝大冒险：最后的进化'},
    {'date': '2020-11-09', 'case_study_movie': '地狱男爵：血皇后崛起'},
    {'date': '2020-11-27', 'case_study_movie': '一秒钟'},

    # --- 2021 ---
    {'date': '2021-02-26', 'case_study_movie': '猫和老鼠'},
    {'date': '2021-06-11', 'case_study_movie': '你好世界'},
    {'date': '2021-06-18', 'case_study_movie': '了不起的老爸'},
    {'date': '2021-06-25', 'case_study_movie': '我没谈完的那场恋爱'},
    {'date': '2021-07-01', 'case_study_movie': '革命者'},
    {'date': '2021-11-12', 'case_study_movie': '梅艳芳'},
    {'date': '2021-11-19', 'case_study_movie': '铁道英雄'},

    # --- 2022 ---
    {'date': '2022-01-14', 'case_study_movie': '黑客帝国：矩阵重启'},
    {'date': '2022-03-14', 'case_study_movie': '神秘海域'},
    {'date': '2022-04-03', 'case_study_movie': '精灵旅社4：变身大冒险'},
    {'date': '2022-04-15', 'case_study_movie': '边缘行者'},
    {'date': '2022-04-22', 'case_study_movie': '亡命救护车'},
    {'date': '2022-05-28', 'case_study_movie': '哆啦A梦：大雄的宇宙小战争2021'},
    {'date': '2022-06-18', 'case_study_movie': '一周的朋友'},

    # --- 2023 ---
    {'date': '2023-02-14', 'case_study_movie': '不能流泪的悲伤'},
    {'date': '2023-06-01', 'case_study_movie': '哆啦A梦：大雄与天空的理想乡'},
    {'date': '2023-06-01', 'case_study_movie': '天空之城'},
    {'date': '2023-08-22', 'case_study_movie': '念念相忘'},

    # --- 2024 ---
    {'date': '2024-05-31', 'case_study_movie': '哆啦A梦：大雄的地球交响乐'},
    {'date': '2024-06-15', 'case_study_movie': '排球少年!! 垃圾场决战'},
    {'date': '2024-07-26', 'case_study_movie': '异人之下'},
    {'date': '2024-08-16', 'case_study_movie': '异形：夺命舰'},

]

SCENARIOS = {
    "高开低走/烂片案例 (Rotten Movies)": ROTTEN_MARKET_DATES
}


def main_rotten():
    set_seeds(42)

    # ==============================================================================
    # 【修改】: 更改输出路径，增加中间文件夹 "rotten_movies"
    # ==============================================================================
    log_dir = os.path.join("../logs", "rotten_movies")
    os.makedirs(log_dir, exist_ok=True)

    log_file_path = os.path.join(log_dir, "evaluation_rotten.log")
    csv_results_path = os.path.join(log_dir, "rotten_evaluation_results.csv")

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
        print(f"--- 成功加载评估环境配置文件: {env_config_path} ---")
    except FileNotFoundError:
        print(f"--- [警告] 未找到评估环境配置文件 {env_config_path}, 将使用默认值 ---")
        env_config = {'online_movie_count': 8, 'total_show_count': 45}

    EVAL_MOVIE_COUNT = env_config.get('online_movie_count', 8)
    EVAL_SHOW_COUNT = env_config.get('total_show_count', 45)
    print(f"    - 评估环境: {EVAL_MOVIE_COUNT} 部电影, {EVAL_SHOW_COUNT} 场排片")

    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        def print_and_log(message, f_handle):
            print(message)
            f_handle.write(message + '\n')
            f_handle.flush()

        header = f"--- 开始一次新的【烂片/高开低走】场景评估: {pd.Timestamp.now()} ---\n"
        print_and_log(header, log_file)
        eval_logger = Logger(logdir='logs', is_print=False)

        for scenario_name, dates_list in SCENARIOS.items():
            print_and_log(f"\n{'#' * 60}\n{'#' * 15} 开始评估场景: {scenario_name} {'#' * 15}\n{'#' * 60}", log_file)
            for date_info in dates_list:
                start_date_str = date_info['date']
                ensure_movie_name = date_info['case_study_movie']
                print_and_log(f"\n===== 正在使用起始日期: {start_date_str} 进行评估 =====", log_file)

                if ensure_movie_name:
                    print_and_log(f"      (本次为案例分析, 将确保包含电影: 《{ensure_movie_name}》)", log_file)

                index_printed_this_date = False

                policies = {
                    "RL (SAC)": {'type': 'SAC', 'use_constraint': True},
                    "静态启发式策略": {'type': 'Static', 'use_constraint': True},
                    "效率启发式策略": {'type': 'Efficiency', 'use_constraint': True},
                    "贪婪启发式策略": {'type': 'Greedy', 'use_constraint': True}
                }

                for name, config in policies.items():
                    env = CinemaGym(
                        logger=eval_logger,
                        online_movie_count=EVAL_MOVIE_COUNT,
                        total_show_count=EVAL_SHOW_COUNT
                    )
                    # 重置环境
                    state = env.reset(fix_start_date=start_date_str, ensure_movie=ensure_movie_name, policy_name=name)

                    case_study_index = None
                    if ensure_movie_name:
                        try:
                            case_study_index = env.next_day_movie_name.index(ensure_movie_name)
                            if not index_printed_this_date:
                                print_and_log(f"      (自动查找到《{ensure_movie_name}》的索引为: {case_study_index})",
                                              log_file)
                                index_printed_this_date = True
                        except ValueError:
                            if not index_printed_this_date:
                                print_and_log(f"      (警告: 未能在最终电影列表中找到《{ensure_movie_name}》)", log_file)
                                index_printed_this_date = True

                    agent = None
                    if config['type'] == 'SAC':
                        agent = SACAgent(online_movie_num=env.online_movie_count, feature_columns=env.feature_columns,
                                         look_back_horizon=env.look_back_horizon)
                        # 模型路径保持不变
                        model_path = 'experiments/00_lstm_efficiency_feature/sac_policy_best.h5'
                        if os.path.exists(model_path.replace(".h5", "_actor.weights.h5")):
                            agent.load_weights(model_path)
                    elif name == "静态启发式策略":
                        agent = StaticHeuristic(
                            init_action=np.array([1 / env.online_movie_count] * env.online_movie_count))
                    elif name == "效率启发式策略":
                        agent = EfficiencyHeuristic(
                            init_action=np.array([1 / env.online_movie_count] * env.online_movie_count),
                            next_day_movie_name=env.next_day_movie_name,
                            total_show_count=env.total_show_count)
                    elif name == "贪婪启发式策略":
                        agent = GreedyHeuristic()

                    daily_incomes = []
                    daily_actions = []
                    for _ in range(env.total_step):
                        raw_action = None
                        if config['type'] == 'SAC':
                            env_action, _ = agent.step(state[np.newaxis, :, :, :], deterministic=True)
                            raw_action = env_action
                        elif config['type'] == 'Efficiency':
                            raw_action = agent.step(state, env.next_day_movie_name)
                        else:
                            raw_action = agent.step(state)

                        # 应用硬约束
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
                                          "case_study_movie": ensure_movie_name, "case_study_index": case_study_index,
                                          "total_income": total_income,
                                          "daily_incomes": json.dumps(daily_incomes),
                                          "daily_actions": json.dumps(daily_actions)})

    results_df = pd.DataFrame(all_runs_data)
    results_df.to_csv(csv_results_path, index=False)

    numeric_cols_to_avg = ['total_income']
    summary = results_df.groupby(['scenario', 'policy'])[numeric_cols_to_avg].mean().unstack().round(2)
    overall_summary = results_df.groupby('policy')[numeric_cols_to_avg].mean().round(2)
    overall_summary = overall_summary.sort_values(by='total_income', ascending=False)

    with open(log_file_path, 'a', encoding='utf-8') as log_file:
        summary_header = f"\n\n{'=' * 60}\n评估完成! 详细数据已保存至: {csv_results_path}\n{'-' * 60}\n各情景下平均总收入 (万元):"
        print_and_log(summary_header, log_file)
        print_and_log(summary.to_string(), log_file)

        overall_summary_header = f"\n{'-' * 60}\n所有日期的总体平均总收入 (万元):"
        print_and_log(overall_summary_header, log_file)
        print_and_log(overall_summary.to_string(), log_file)
        print_and_log(f"{'=' * 60}", log_file)


if __name__ == '__main__':
    main_rotten()