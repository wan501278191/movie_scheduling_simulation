import os
import sys
import logging
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import gymnasium as gym
import pandas as pd
import time
import json
import random

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 【添加这一行】强制使用 CPU，彻底消除 GPU 并行带来的随机性
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# '0': 显示所有日志 (默认)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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


# 导入您的类
from envs.simsche import CinemaGym
from policys import (EfficiencyHeuristic, StaticHeuristic, PredictiveStrategy,
                     GreedyHeuristic)
from policys.sac_policy import SACAgent
from tools import Logger, Memory


def set_seeds(seed_value):
    seed_value = int(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    # --- [核心修改] 强制 TensorFlow 使用确定性算法 ---
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    # 物理层面限制多线程产生的随机偏差
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

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


def load_movie_release_dates():
    """
    从 data/rl_data_final.csv 中读取所有电影的上映时间
    返回字典：{电影名: 上映日期字符串 (yyyy-MM-dd格式)}
    """
    try:
        # 读取CSV文件
        csv_path = os.path.join("data", "rl_data_final.csv")
        df = pd.read_csv(csv_path)

        # 确保日期列是datetime类型
        df['real_date'] = pd.to_datetime(df['real_date'])

        # 按电影名分组，取每部电影的最早播放时间作为上映时间
        release_dates = df.groupby('MovieName')['real_date'].min()

        # 转换为 yyyy-MM-dd 字符串格式
        movie_release_dict = {}
        for movie_name, release_date in release_dates.items():
            movie_release_dict[movie_name] = release_date.strftime('%Y-%m-%d')

        print(f"成功加载 {len(movie_release_dict)} 部电影的上映时间数据")
        return movie_release_dict

    except FileNotFoundError:
        print(f"错误: 找不到文件 data/rl_data_final.csv")
        return {}
    except Exception as e:
        print(f"读取上映时间数据时出错: {e}")
        return {}


# ==============================================================================
# 1. 黑马 (低开高走)
# ==============================================================================
BLACK_HORSE_LIST = [
    {'date': '2024-02-10', 'case_study_movie': '第二十条_seg0'},  # 票房: 24.49亿 | 首日排片: 19.49%
    {'date': '2024-02-10', 'case_study_movie': '熊出没·逆转时空_seg0'},  # 票房: 19.80亿 | 首日排片: 11.27%
    {'date': '2023-09-28', 'case_study_movie': '坚如磐石_seg0'},  # 票房: 13.48亿 | 首日排片: 12.00%
    {'date': '2023-09-28', 'case_study_movie': '前任4：英年早婚_seg0'},  # 票房: 10.10亿 | 首日排片: 16.44%
    {'date': '2023-09-28', 'case_study_movie': '志愿军：雄兵出击_seg0'},  # 票房: 8.27亿 | 首日排片: 14.58%
    {'date': '2024-04-03', 'case_study_movie': '你想活出怎样的人生_seg0'},  # 票房: 7.91亿 | 首日排片: 19.93%
    {'date': '2024-08-16', 'case_study_movie': '异形：夺命舰_seg0'},  # 票房: 7.85亿 | 首日排片: 14.04%
    {'date': '2024-05-01', 'case_study_movie': '末路狂花钱_seg0'},  # 票房: 7.81亿 | 首日排片: 15.09%
    {'date': '2023-09-29', 'case_study_movie': '莫斯科行动_seg0'},  # 票房: 6.62亿 | 首日排片: 13.57%
    {'date': '2024-03-01', 'case_study_movie': '周处除三害_seg0'},  # 票房: 6.48亿 | 首日排片: 8.98%
    {'date': '2024-05-01', 'case_study_movie': '九龙城寨之围城_seg0'},  # 票房: 6.04亿 | 首日排片: 9.44%
    {'date': '2023-12-30', 'case_study_movie': '金手指_seg0'},  # 票房: 5.72亿 | 首日排片: 16.12%
]

# ==============================================================================
# 2. 烂片 (高开低走)
# ==============================================================================
FLOP_LIST = [
    # --- 知名商业片案例 ---
    {'date': '2024-06-15', 'case_study_movie': '排球少年!! 垃圾场决战_seg0'},
    {'date': '2024-06-28', 'case_study_movie': '海关战线_seg0'},
    {'date': '2024-07-26', 'case_study_movie': '异人之下_seg0'},
    {'date': '2024-05-31', 'case_study_movie': '哆啦A梦：大雄的地球交响乐_seg0'},
    {'date': '2023-11-10', 'case_study_movie': '惊奇队长2_seg0'},
    {'date': '2024-07-05', 'case_study_movie': '欢迎来到我身边_seg0'},
    # --- 合理的中小成本/黑马失败案例 ---
    {'date': '2024-04-09', 'case_study_movie': '幸福慢车_seg0'},
    {'date': '2024-09-27', 'case_study_movie': '与你相遇的日子_seg0'},
    {'date': '2024-05-28', 'case_study_movie': '我，就是风!_seg0'},
    {'date': '2024-07-27', 'case_study_movie': '八十天环游地球_seg0'},
]

SCENARIOS = {
    "黑马逆袭 (Black Horse)": BLACK_HORSE_LIST,
    "高开低走 (High Start Low End)": FLOP_LIST
}

EVAL_ALL_MOVIES = False

if EVAL_ALL_MOVIES:
    movie_release_date_dict = load_movie_release_dates()
    SCENARIOS = {
        "全部电影（Whole Movies）": [
            {'date': release_date, 'case_study_movie': movie_name}
            for movie_name, release_date in movie_release_date_dict.items()
        ][:100],
    }

def collect_scheduling_data():
    """收集各策略的排片率数据"""
    set_seeds(42)
    
    # 设置日志文件
    log_dir = os.path.join("logs", "main_case_show")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "scheduling_evaluation.log")
    
    def print_and_log(message, f_handle):
        print(message)
        f_handle.write(message + '\n')
        f_handle.flush()
    
    EVAL_CONFIG = {
        'softmax_temp': 1.0,
        'model_path': 'experiments/11_sac_new_env/sac_policy_best.h5'
    }
    
    eval_logger = Logger(logdir=log_dir, is_print=False)
    
    # 打开日志文件
    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        header = f"--- 开始排片率评估: {pd.Timestamp.now()} ---\n"
        print_and_log(header, log_file)
    
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
        
        # 预加载 SAC 模型
        print_and_log("--- 正在预加载 SAC 策略模型 ---", log_file)
        sac_agent_instance = SACAgent(
            online_movie_num=EVAL_MOVIE_COUNT,
            feature_columns=['dummy'] * 7,
            look_back_horizon=7
        )
        
        correct_feature_dim = 7
        dummy_state = np.zeros((1, EVAL_MOVIE_COUNT, 7, correct_feature_dim), dtype=np.float32)
        
        try:
            _ = sac_agent_instance.step(dummy_state)
            print_and_log("--- 策略模型 Build 完成 ---", log_file)
        except Exception:
            pass
        
        # 加载权重
        model_path = EVAL_CONFIG['model_path']
        actor_weights_path = model_path.replace(".h5", "_actor.weights.h5")
        
        if os.path.exists(actor_weights_path):
            sac_agent_instance.load_weights(model_path)
            print_and_log(f"--- 权重加载成功: {model_path} ---", log_file)
        else:
            print_and_log(f"!!! 警告: 未找到 SAC 模型文件: {actor_weights_path} !!!", log_file)
        
        # 存储所有数据
        all_scheduling_data = []
        
        # 策略配置
        policies = {
            "RL (SAC)": {
                'type': 'SAC',
                'use_constraint': True,
                'temp': EVAL_CONFIG['softmax_temp']
            },
            "静态启发式策略": {'type': 'Static', 'use_constraint': True},
            "效率启发式策略": {'type': 'Efficiency', 'use_constraint': True},
            "贪婪启发式策略": {'type': 'Greedy', 'use_constraint': True}
        }
        
        # 遍历所有场景和电影
        for scenario_name, dates_list in SCENARIOS.items():
            print_and_log(f"\n{'#' * 60}\n{'#' * 15} 开始评估场景: {scenario_name} {'#' * 15}\n{'#' * 60}", log_file)
        
            for date_info in dates_list:
                start_date_str = date_info['date']
                target_movie_name = date_info['case_study_movie']
                
                print_and_log(f"\n===== 正在使用起始日期: {start_date_str} 评估电影: {target_movie_name} =====", log_file)
            
                # 为每种策略收集数据
                for policy_name, config in policies.items():
                    set_seeds(42)
                
                    # 创建环境
                    env = CinemaGym(
                        logger=eval_logger,
                        online_movie_count=EVAL_MOVIE_COUNT,
                        total_show_count=EVAL_SHOW_COUNT,
                        enable_logging=False
                    )
                
                    # 设置起始日期
                    target_date = pd.to_datetime(start_date_str)
                    env.current_date = target_date
                    env._update_current_movies()
                    state = env._get_observation()
                
                    # 初始化策略代理
                    agent = None
                    if config['type'] == 'SAC':
                        agent = sac_agent_instance
                    elif policy_name == "静态启发式策略":
                        agent = StaticHeuristic(
                            init_action=np.array([1 / env.online_movie_count] * env.online_movie_count))
                    elif policy_name == "效率启发式策略":
                        init_act = np.array([1 / env.online_movie_count] * env.online_movie_count)
                        current_movie_names = [m['MovieName'] for m in env.current_movie_list]
                        agent = EfficiencyHeuristic(
                            init_action=init_act,
                            next_day_movie_name=current_movie_names,
                            total_show_count=env.total_show_count)
                    elif policy_name == "贪婪启发式策略":
                        agent = GreedyHeuristic()
                
                    # 收集30天的数据
                    movie_scheduling_rates = []  # 存储目标电影每天的排片率
                
                    for day in range(30):
                        # 获取动作
                        raw_action = None
                        if config['type'] == 'SAC':
                            sac_input = state[np.newaxis, :, :, :7]
                            current_temp = config.get('temp', 1.0)
                            raw_action, _ = agent.step(
                                sac_input,
                                deterministic=True,
                                softmax_temp=current_temp
                            )
                        elif policy_name == "效率启发式策略":
                            current_movie_names = [m['MovieName'] for m in env.current_movie_list]
                            raw_action = agent.step(state, current_movie_names)
                        else:
                            raw_action = agent.step(state)
                    
                        # 应用约束
                        if config['use_constraint']:
                            action = apply_dynamic_hard_constraint(raw_action, state, env.total_show_count)
                        else:
                            action = raw_action
                    
                        # 找到目标电影在当前排片中的索引和排片率
                        current_movie_names = [m['MovieName'] for m in env.current_movie_list]
                        scheduling_rate = 0.0
                        
                        if target_movie_name in current_movie_names:
                            movie_index = current_movie_names.index(target_movie_name)
                            scheduling_rate = action[movie_index]
                        
                        movie_scheduling_rates.append(scheduling_rate)
                    
                        # 记录每日排片详情
                        current_movie_names = [m['MovieName'] for m in env.current_movie_list]
                        daily_schedule_info = []
                        for i, (movie_name, sched_rate) in enumerate(zip(current_movie_names, action)):
                            daily_schedule_info.append(f"{movie_name}: {sched_rate:.4f}")
                        
                        # 执行动作
                        new_state, reward, done, info = env.step(action, policy_name=policy_name)
                        state = new_state
                        
                        if done:
                            break
                
                    # 保存数据
                    all_scheduling_data.append({
                        'scenario': scenario_name,
                        'start_date': start_date_str,
                        'target_movie': target_movie_name,
                        'policy': policy_name,
                        'scheduling_rates': movie_scheduling_rates
                    })
                    
                    # 记录最终的影片选择和排片率
                    final_movie_names = [m['MovieName'] for m in env.current_movie_list]
                    final_schedule_info = []
                    for i, (movie_name, sched_rate) in enumerate(zip(final_movie_names, action)):
                        final_schedule_info.append(f"{movie_name}: {sched_rate:.4f}")
                    
                    print_and_log(f"{policy_name:<22}: 收集完成 ({len(movie_scheduling_rates)} 天数据)", log_file)
                    print_and_log(f"    选片结果: {final_movie_names}", log_file)
                    print_and_log(f"    排片率: {final_schedule_info}", log_file)
    
    return all_scheduling_data


def plot_scheduling_curves(data, output_dir="scheduling_plots"):
    """绘制排片率变化曲线"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 按场景分组
    scenarios = {}
    for item in data:
        scenario = item['scenario']
        if scenario not in scenarios:
            scenarios[scenario] = []
        scenarios[scenario].append(item)
    
    # 为每个场景生成图表
    for scenario_name, scenario_data in scenarios.items():
        # 按目标电影分组
        movies = {}
        for item in scenario_data:
            movie = item['target_movie']
            if movie not in movies:
                movies[movie] = []
            movies[movie].append(item)
        
        # 为每部电影生成图表
        for movie_name, movie_data in movies.items():
            plt.figure(figsize=(12, 8))
            
            # 绘制每种策略的曲线
            for item in movie_data:
                policy_name = item['policy']
                rates = item['scheduling_rates']
                days = list(range(1, len(rates) + 1))
                
                plt.plot(days, rates, marker='o', linewidth=2, markersize=4, label=policy_name)
            
            plt.xlabel('上映后第几天', fontsize=12)
            plt.ylabel('排片率', fontsize=12)
            plt.title(f'{scenario_name}\n电影: {movie_name}', fontsize=14, pad=20)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.xlim(1, 30)
            plt.ylim(0, 1)
            
            # 设置x轴刻度
            plt.xticks(range(1, 31, 2))
            
            # 保存图片
            safe_movie_name = movie_name.replace('/', '_').replace(':', '_')
            filename = f"{scenario_name}_{safe_movie_name}_scheduling_curve.png"
            filepath = os.path.join(output_dir, filename)
            plt.tight_layout()
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"已保存: {filepath}")


def main():
    print("开始收集排片率数据...")
    scheduling_data = collect_scheduling_data()
    
    print(f"\n共收集到 {len(scheduling_data)} 条数据")
    
    print("\n开始生成排片率变化曲线...")
    plot_scheduling_curves(scheduling_data)
    
    print("\n完成！所有图表已保存到 scheduling_plots 目录")


if __name__ == '__main__':
    main()