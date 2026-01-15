import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import tensorflow as tf
from tensorflow.keras import layers
import os


# 静态启发式排座模式
class StaticHeuristic(object):
    def __init__(self, init_action):
        super(StaticHeuristic, self).__init__()
        self.action = init_action

    def step(self, feature: np.ndarray) -> np.ndarray:
        return self.action


# 效率启发式排座模式
class EfficiencyHeuristic(object):
    def __init__(self, init_action: np.ndarray, next_day_movie_name: List[str], total_show_count: int):
        super(EfficiencyHeuristic, self).__init__()
        self.next_day_movie_name = next_day_movie_name
        self.action_map = {k: v for k, v in zip(next_day_movie_name, init_action)}
        self.total_show_count = total_show_count

    def step(self, feature: np.ndarray, next_day_movie_name: List[str]) -> np.ndarray:
        min_proportion_for_one_show = 1.0 / self.total_show_count

        # 1. 票房占比 (Index 7)
        box_office_array = feature[:, -1, 7]
        total_box_office = np.sum(box_office_array)
        box_office_ratio = box_office_array / (total_box_office if total_box_office > 0 else 1.0)

        # 2. 场次占比 (Index 8)
        total_show_array = feature[:, -1, 8]
        total_shows = np.sum(total_show_array)
        show_ratio = total_show_array / (total_shows if total_shows > 0 else 1.0)

        next_action_map = {}
        # 默认份额 (仅用于字典里查不到名字时的兜底)
        default_share = 1.0 / len(next_day_movie_name) if len(next_day_movie_name) > 0 else 0.125

        for i, movie_name in enumerate(next_day_movie_name):
            if movie_name == 'Padding':
                next_action_map[movie_name] = 0.0
                continue

            last_day_proportion = self.action_map.get(movie_name, default_share)

            bf_r = box_office_ratio[i]  # 产出
            s_r = show_ratio[i]  # 投入

            # === 【原始效率逻辑】 ===
            # 这里去掉了所有关于 s_r <= 0 的特殊保护
            # 新片(s_r=0, bf_r=0) 既不满足 > 0，也不满足 < 0，会自动落入 else (维持现状)

            if bf_r > (s_r * 1.5):
                # 效率极高 (>1.5) -> 加场
                next_action_map[movie_name] = last_day_proportion * 1.8

            elif bf_r < (s_r * 0.8):
                # 效率较低 (<1.0) -> 砍场
                next_action_map[movie_name] = max(min_proportion_for_one_show, last_day_proportion * 0.5)

            else:
                # 中间地带 (1.0 ~ 1.5) 或 新片(0=0) -> 维持现状
                next_action_map[movie_name] = last_day_proportion

        # 更新记忆
        self.action_map = next_action_map

        # 归一化
        action = np.array([self.action_map.get(name, 0.0) for name in next_day_movie_name])
        sum_act = np.sum(action)
        if sum_act < 1e-6:
            return np.ones(len(action)) / len(action)

        return action / sum_act

# 贪婪启发式排片模式
class GreedyHeuristic(object):
    def __init__(self):
        super(GreedyHeuristic, self).__init__()

    def step(self, feature: np.ndarray) -> np.ndarray:
        # feature shape: (Movie_Num=8, Lookback=7, Feature_Dim=12)
        # 取最近一天 (-1)

        # Index 9: 场均收入 (用于判断强弱)
        kernel_feature = feature[:, -1, 9].copy()

        # Index 1: Gold_Strength (想看人数/热度) - 这是一个非常关键的特征！
        # 用于区分 "新上映的真电影" (热度>0) 和 "Padding填充位" (热度=0)
        gold_strength = feature[:, -1, 1]

        # 1. 识别 Padding：如果热度极低且场均收入为0，认为是Padding，强制为0
        # 注意：这里用 1e-5 是为了防止浮点误差
        is_padding = (gold_strength < 1e-5) & (kernel_feature < 1e-5)

        # 2. 识别有效电影（非0收入）
        valid_mask = kernel_feature > 1e-5

        # 3. 计算填充值 (用现有有效电影的平均值，或者给一个基础分)
        if np.sum(valid_mask) > 0:
            avg_val = np.mean(kernel_feature[valid_mask])
        else:
            # 如果全是0 (比如第一天全是新片)，设为1.0
            avg_val = 1.0

        # 4. 关键逻辑：
        # - 如果是有效电影：保持原值
        # - 如果是 Padding：强制 0
        # - 如果是 新电影 (非Padding 但 收入为0)：赋予平均值，给它冷启动机会

        # 先把所有 0 值填成平均值
        kernel_feature[~valid_mask] = avg_val

        # 再把 Padding 的位置 归零
        kernel_feature[is_padding] = 0.0

        # 5. 归一化输出
        dampened_feature = (kernel_feature + 1e-8) ** 0.5

        # 再次确保 padding 是 0 (防止 1e-8 产生微小排片)
        dampened_feature[is_padding] = 0.0

        sum_feat = np.sum(dampened_feature)
        if sum_feat < 1e-6:
            # 如果算出来全是0（极端情况），平均分给非Padding的电影
            valid_movies_count = len(dampened_feature) - np.sum(is_padding)
            if valid_movies_count > 0:
                action = np.zeros_like(dampened_feature)
                action[~is_padding] = 1.0 / valid_movies_count
                return action
            else:
                return np.ones(len(dampened_feature)) / len(dampened_feature)

        action = dampened_feature / sum_feat
        return action
# PredictiveNetwork 和 PredictiveStrategy 保持不变...
class PredictiveNetwork(tf.keras.Model):
    def __init__(self, weekindex_feature_num: int, hidden_num: int):
        super(PredictiveNetwork, self).__init__()
        self.MLP_Weekindex_Layer = tf.keras.Sequential([
            layers.Dense(weekindex_feature_num, activation='relu', name='week_dense_1'),
            layers.Dense(hidden_num, activation='relu', name='week_dense_2'),
            layers.Dense(hidden_num, name='week_dense_3'),
        ], name='MLP_Weekindex')
        self.MLP_Income_Layer = tf.keras.Sequential([
            layers.Dense(hidden_num, activation='relu', name='income_dense_1'),
            layers.Dense(hidden_num, name='income_dense_2'),
        ], name='MLP_Income')
        self.MLP_Predict_Layer = tf.keras.Sequential([
            layers.Dense(hidden_num, activation='relu', name='predict_dense_1'),
            layers.Dense(1, name='predict_dense_2'),
        ], name='MLP_Predict')

    def call(self, inputs: tf.Tensor, training=None, mask=None) -> tf.Tensor:
        feature_num = inputs.shape[3] - 7
        income_feature = inputs[:, :, -1, 0:feature_num]
        weekindex_feature = inputs[:, :, -1, feature_num:]
        income_hidden = self.MLP_Income_Layer(income_feature)
        weekindex_hidden = self.MLP_Weekindex_Layer(weekindex_feature)
        predict_input = weekindex_hidden * income_hidden
        predict_output = self.MLP_Predict_Layer(predict_input)[:, :, 0] / 5 + 5.0
        return predict_output


class PredictiveStrategy(object):
    def __init__(self, online_movie_num: int, feature_num: int, hidden_num: int, init_state: np.ndarray):
        super(PredictiveStrategy, self).__init__()
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.project_root = os.path.join(current_dir, '..')
        except NameError:
            self.project_root = '.'
        self.online_movie_num = online_movie_num
        self.feature_num = feature_num
        self.hidden_num = hidden_num
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.0001, decay_steps=5000, decay_rate=0.96, staircase=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-5)
        self.policy = PredictiveNetwork(hidden_num=hidden_num, weekindex_feature_num=14)
        self.policy(self.feature_engineer(init_state))

        model_path_relative = 'experiments/predict/predict_policy.weights.h5'
        model_path_absolute = os.path.join(self.project_root, model_path_relative)
        if os.path.exists(model_path_absolute):
            try:
                self.load_weights(model_path_relative)
            except Exception:
                pass

    def step(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ori_action = self.policy(self.feature_engineer(state))
        ori_action = np.clip(ori_action, 0.02, 1000)
        normed_action = ori_action ** 2 / np.sum(ori_action ** 2)
        return ori_action, normed_action

    def feature_engineer(self, states: np.ndarray) -> np.ndarray:
        total_box_office = np.sum(states[:, :, :, 0], axis=1)[:, np.newaxis, :]
        total_show = np.sum(states[:, :, :, 1], axis=1)[:, np.newaxis, :]
        total_box_office[total_box_office == 0] = 1.0
        total_show[total_show == 0] = 1.0
        box_office_ratio = states[:, :, :, 0] / total_box_office
        show_ratio = states[:, :, :, 1] / total_show
        states = np.concatenate([states, box_office_ratio[:, :, :, np.newaxis], show_ratio[:, :, :, np.newaxis]],
                                axis=-1)
        return states

    def load_weights(self, path):
        if not self.policy.built:
            dummy_input_shape = (1, self.online_movie_num, self.feature_num, 7)
            self.policy.build(input_shape=dummy_input_shape)
        absolute_path = os.path.join(self.project_root, path)
        self.policy.load_weights(absolute_path)