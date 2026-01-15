import pandas as pd
import numpy as np
import gymnasium as gym
import os
from gymnasium import spaces
from typing import Tuple, List, Dict, Any
import random
from datetime import timedelta


class CinemaGym(gym.Env):
    def __init__(self, logger,
                 online_movie_count: int = 8,
                 total_show_count: int = 30,
                 start_date: str = '2019-03-01', end_date: str = '2023-08-31',
                 csv_path: str = None,
                 enable_logging: bool = False):

        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(current_dir, '..')

        # 默认读取 data/rl_data_final.csv
        if csv_path is None:
            self.csv_path = os.path.join(project_root, 'data', 'rl_data_final.csv')
        else:
            self.csv_path = csv_path

        self.logger = logger
        self.enable_logging = enable_logging

        self.online_movie_count = online_movie_count
        self.total_show_count = total_show_count
        self.look_back_horizon = 7

        # === 核心定义：混合特征空间 (12维) ===
        # Index 0-6: SAC 用的归一化特征
        self.sac_feature_columns = [
            'Eff_Index', 'Gold_Strength', 'Market_Share', 'Occ_Rate',
            'Time_State', 'Days_Running', 'Avg_Price'
        ]
        # Index 7-11: 规则策略/Reward 用的原始绝对值
        self.raw_feature_columns = [
            '当前票房(元)',  # Index 7
            '当前场次',  # Index 8
            '场均收入',  # Index 9
            'Box_Share',  # Index 10
            'Market_Total_Show'  # Index 11
        ]
        self.feature_columns = self.sac_feature_columns + self.raw_feature_columns
        self.feature_dim = len(self.feature_columns)

        self._load_data()
        self.start_dt = pd.to_datetime(start_date)
        self.end_dt = pd.to_datetime(end_date)

        self.current_date = None
        self.current_movie_list = []
        self.last_movie_names = []
        self.total_income = 0.0
        self.step_num = 0
        self.done = False

    def _load_data(self):
        if self.enable_logging:
            self.logger.info(f"正在加载训练数据: {self.csv_path}")

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"找不到文件: {self.csv_path}，请先运行 data_factory.py")

        df = pd.read_csv(self.csv_path)
        df['real_date'] = pd.to_datetime(df['real_date'])

        # 建立索引：日期 -> 当天所有电影
        self.data_by_date = {k: v for k, v in df.groupby('real_date')}
        self.available_dates = sorted(list(self.data_by_date.keys()))
        # 建立索引：电影名 -> 历史数据
        self.data_by_movie = {k: v.sort_values('real_date') for k, v in df.groupby('MovieName')}

        if self.enable_logging:
            self.logger.info(f"数据加载完成，覆盖 {len(self.available_dates)} 天")

    @property
    def action_space(self):
        return gym.spaces.Box(low=0, high=1.0, shape=(self.online_movie_count,), dtype=np.float32)

    @property
    def observation_space(self):
        # 状态形状: (8, 7, 12)
        shape = (self.online_movie_count, self.look_back_horizon, self.feature_dim)
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)

    def reset(self, seed=None, options=None) -> np.ndarray:
        if seed is not None:
            super().reset(seed=seed)
            random.seed(seed)
            np.random.seed(seed)

        super().reset(seed=seed)
        self.total_income = 0.0
        self.step_num = 0
        self.done = False

        valid_start_indices = [
            i for i, d in enumerate(self.available_dates)
            if d >= self.start_dt and d <= self.end_dt
        ]
        if not valid_start_indices:
            valid_start_indices = range(len(self.available_dates) - 45)

        start_idx = random.choice(valid_start_indices[:-45])
        self.current_date = self.available_dates[start_idx]

        # --- [修改位置] 重置时清空上日记录，确保第一天是纯随机选片 ---
        self.last_movie_names = []
        self._update_current_movies()

        state = self._get_observation()
        if self.enable_logging:
            self.logger.info(f"======== 环境重置: {self.current_date.strftime('%Y-%m-%d')} ========")
        return state

    # 添加 policy_name 参数，赋予默认值 None 以保持兼容性
    def step(self, action: np.ndarray, policy_name: str = None) -> Tuple[np.ndarray, float, bool, Dict]:
        # 动作转场次
        shows_float = action * self.total_show_count
        shows_int = np.floor(shows_float).astype(int)
        remainder = self.total_show_count - shows_int.sum()
        if remainder > 0:
            residues = shows_float - shows_int
            top_indices = np.argsort(residues)[-remainder:]
            shows_int[top_indices] += 1

        day_income = 0.0
        movie_details = []

        for idx, movie_row in enumerate(self.current_movie_list):
            avg_income = movie_row['场均收入']
            movie_income = shows_int[idx] * avg_income
            day_income += movie_income
            movie_details.append({
                'name': movie_row['MovieName'],
                'shows': shows_int[idx],
                'avg_income': avg_income,
                'total_income': movie_income
            })

        reward = day_income / 10000.0
        self.total_income += reward

        # === 修改后的详细日志格式 ===
        if self.enable_logging:
            date_str = self.current_date.strftime('%Y-%m-%d')
            # 获取星期几 (Python 中 0 是周一, 6 是周日，这里转为 1-7)
            weekday = self.current_date.weekday() + 1
            strategy_name = policy_name if policy_name is not None else "未知策略"

            self.logger.info(f"======== [ {date_str} | 策略: {strategy_name} ] ========")
            self.logger.info(f"======== [ {date_str} 周{weekday} | 策略: {strategy_name} ] ========")

            for m in movie_details:
                # 使用中文全角空格或固定宽度进行对齐
                name_display = f"{m['name']}"
                self.logger.info(
                    f"电影名称：{name_display: ^16}，"
                    f"场次：{m['shows']:<2} ，"
                    f"单场均收入：{int(m['avg_income']):<4} ，"
                    f"总收入：{int(m['total_income']):<6}"
                )

            self.logger.info(f"截止目前的 总收入：{self.total_income:.2f} (万元)")
        # =========================
        self.step_num += 1
        curr_idx = self.available_dates.index(self.current_date)
        if curr_idx + 1 < len(self.available_dates):
            self.current_date = self.available_dates[curr_idx + 1]
        else:
            self.done = True

        self._update_current_movies()
        next_state = self._get_observation()

        if self.step_num >= 30:
            self.done = True

        info = {
            'total_income': self.total_income,
            'MovieNames': [m['name'] for m in movie_details],
            'MovieDayIncome': [m['total_income'] for m in movie_details]
        }
        return next_state, reward, self.done, info

    def _update_current_movies(self):
        """
        [修改后的选片逻辑]
        1. 继承：保留昨日片单中今日未下映的电影。
        2. 补位：若不足8部，从今日剩余池中随机抽样补足。
        """
        if self.current_date not in self.data_by_date:
            self.done = True
            return

        today_df = self.data_by_date[self.current_date]
        today_movie_names_all = set(today_df['MovieName'].values)

        selected_movies_rows = []
        current_names_set = set()

        # --- 步骤 1: 连贯性检查 (继承昨日) ---
        for name in self.last_movie_names:
            if name in today_movie_names_all:
                row = today_df[today_df['MovieName'] == name].iloc[0]
                selected_movies_rows.append(row)
                current_names_set.add(name)

        # --- 步骤 2: 随机补位 (补足 8 部) ---
        if len(selected_movies_rows) < self.online_movie_count:
            # 找出还没被选中的今日可选电影
            remaining_pool = sorted(list(today_movie_names_all - current_names_set))

            needed = self.online_movie_count - len(selected_movies_rows)
            if len(remaining_pool) > 0:
                # 纯随机抽样补位，不看票房，不看新旧
                num_to_sample = min(len(remaining_pool), needed)
                chosen_extras = random.sample(remaining_pool, num_to_sample)

                for name in chosen_extras:
                    row = today_df[today_df['MovieName'] == name].iloc[0]
                    selected_movies_rows.append(row)
                    current_names_set.add(name)

        # --- 步骤 3: 赋值与记录 ---
        self.current_movie_list = selected_movies_rows.copy()
        # 更新记录，供下一天使用
        self.last_movie_names = [m['MovieName'] for m in self.current_movie_list if m['MovieName'] != 'Padding']

        # --- 步骤 4: Padding (如果整个市场都不够8部) ---
        while len(self.current_movie_list) < self.online_movie_count:
            dummy = self.current_movie_list[0].copy() if self.current_movie_list else pd.Series(0,
                                                                                                index=today_df.columns)
            dummy['MovieName'] = 'Padding'
            dummy['场均收入'] = 0
            for col in self.feature_columns:
                if col in dummy: dummy[col] = 0
            self.current_movie_list.append(dummy)

    def _get_observation(self) -> np.ndarray:
        obs_batch = []
        target_date = self.current_date

        # 1. 计算时间状态 (Time_State) - 捕捉周末效应
        weekday = target_date.weekday()  # 0=周一, 6=周日
        if weekday >= 5:
            time_state = 1.0  # 周末
        elif weekday == 4:
            time_state = 0.5  # 周五
        else:
            time_state = 0.0  # 平日

        for movie_row in self.current_movie_list:
            # === 处理 Padding 填充位 ===
            if movie_row['MovieName'] == 'Padding':
                obs_batch.append(np.zeros((self.look_back_horizon, self.feature_dim)))
                continue

            # === 获取历史数据 ===
            full_history = self.data_by_movie[movie_row['MovieName']]
            # 取 target_date 之前的数据
            past_data = full_history[full_history['real_date'] < target_date].tail(self.look_back_horizon)

            # 初始化全0矩阵 (Time_Steps, Feature_Dim)
            matrix = np.zeros((self.look_back_horizon, self.feature_dim))

            if len(past_data) > 0:
                # === Case A: 老片，有历史数据 ===
                # 填充到矩阵的后半部分（保留时间顺序）
                raw_matrix = past_data[self.feature_columns].values.astype(np.float32)
                matrix[-len(raw_matrix):] = raw_matrix

                # 覆盖 Time_State 为当前的星期状态 (比历史记录更重要)
                # 注意：这里假设 Time_State 是第 5 列 (Index 4)
                matrix[-len(raw_matrix):, 4] = time_state

            else:
                # === Case B: 新片，无历史数据，伪造初始状态 ===
                # 特征顺序: ['Eff_Index', 'Gold_Strength', 'Market_Share', 'Occ_Rate', 'Time_State', 'Days_Running', 'Avg_Price']
                default_vec = np.zeros(self.feature_dim)
                default_vec[0] = 1.0  # Eff_Index: 1.0 代表供需平衡
                default_vec[1] = 1.0  # Gold_Strength: 默认给一个基础热度
                default_vec[2] = 0.0  # Market_Share: 还没排片，自然是0
                default_vec[3] = 15.0  # Occ_Rate: 给予 15% 的初始上座率预期 (冷启动保护)
                default_vec[4] = time_state  # Time_State: 今天的星期状态
                default_vec[5] = 0.0  # Days_Running: 第0天
                default_vec[6] = 40.0  # Avg_Price: 默认票价 40元

                # 将这个初始状态放入时间序列的最后一步
                matrix[-1] = default_vec

            # === 2. 统一归一化 (Normalization) ===
            # Index 0: Eff_Index (供需指数) & Index 1: Gold_Strength (热度)
            # 维持 3.0 的高阈值以体现超级爆款的供不应求。
            ceiling_index = np.log1p(3.0)
            matrix[:, 0] = np.clip(np.log1p(matrix[:, 0]) / ceiling_index, 0, 1)
            matrix[:, 1] = np.clip(np.log1p(matrix[:, 1]) / ceiling_index, 0, 1)
            # --- B类：线性分布，设定爆款天花板，线性缩放后截断 ---
            # Index 6: Avg_Price (平均票价) - 【核心修改】
            # 核实：爆款全国均价峰值在 55-58元区间。设定 60元为天花板。
            # 效果：58元 -> 0.97. 极大提升了票价区间的区分度。
            matrix[:, 6] = np.clip(matrix[:, 6] / 60.0, 0, 1)
            # --- C类：时间衰减型 (使用 Tanh 非线性挤压) ---

            # Index 5: Days_Running (上映天数)
            # 爆款均为长线电影，Tanh 处理适用。
            matrix[:, 5] = np.tanh(matrix[:, 5] / 20.0)
            # --- D类：原生 0-1 特征 (保持不变) ---
            # Index 2: Market_Share, Index 3: Occ_Rate, Index 4: Time_State

            # --- 最终安全检查 ---
            sac_feature_len = len(self.sac_feature_columns)  # 也就是 7

            # 只对前7列做 0-1 截断
            matrix[:, :sac_feature_len] = np.clip(matrix[:, :sac_feature_len], 0, 1)

            # 后面的列 (Index 7-11) 保持原样，不要截断！

            obs_batch.append(matrix)

        return np.array(obs_batch, dtype=np.float32)