# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import sys
from scipy.spatial import distance

# === 配置 ===
# 默认读取的文件名，你可以修改这里，或者在命令行传参
DEFAULT_FILE = "rl_data_final_v3_2.csv"


def generate_report(file_path):
    if not os.path.exists(file_path):
        print(f"[错误] 找不到文件: {file_path}")
        return

    print(f"正在读取数据 {file_path} 并开始深度质量审计...")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"[错误] 读取失败: {e}")
        return

    # 基础信息
    total_samples = len(df)
    unique_movies = df['MovieName'].nunique() if 'MovieName' in df.columns else 0

    # 报告头部
    print("=" * 90)
    print("            电影排片强化学习数据质量审计证明书 (评估版)")
    print("=" * 90)
    print(f"审计对象: {os.path.basename(file_path)}")
    print(f"样本总量: {total_samples}")
    print(f"独立电影总数: {unique_movies} 个")
    print("-" * 90)

    # -------------------------------------------------------------------------
    # 0. 零值专项审计
    # -------------------------------------------------------------------------
    print("\n>>> 0. 零值专项审计 (当前数据状态)")
    print("-" * 75)
    print(f"{'列名':>15}  {'零值数量':>10}  {'零值占比(%)':>12}")

    # 需要检查零值的列
    audit_cols = [
        'Time_State', 'Box_Share', 'Eff_Index', '当前票房(万)', '当前人次(万)',
        'Gold_Strength', '场均收入', 'Occ_Rate', 'Market_Share', '当前场次',
        'Avg_Price', 'Days_Running', 'Market_Total_Show'
    ]

    for col in audit_cols:
        if col in df.columns:
            # 统计 NaN 或 <= 0.0001 的值
            zero_count = len(df[(df[col].isna()) | (df[col] <= 0.0001)])
            ratio = (zero_count / total_samples) * 100
            print(f"{col:>15}  {zero_count:10d}  {ratio:12.2f}")
        else:
            # 如果列不存在，不做输出或提示缺失
            pass

    print("\n[注]: 若 Time_State 零值高属正常现象(平日为0)。")

    # -------------------------------------------------------------------------
    # 1. 数据分布自检
    # -------------------------------------------------------------------------
    print("\n\n>>> 1. 数据分布自检 (基础统计)")
    print("-" * 75)

    stats_cols = [
        'Eff_Index', 'Gold_Strength', 'Market_Share', 'Occ_Rate', 'Time_State',
        'Days_Running', 'Avg_Price', '当前票房(万)', '当前场次', '场均收入',
        'Box_Share', '当前人次(万)', 'Market_Total_Show'
    ]

    # 筛选存在的列
    valid_cols = [c for c in stats_cols if c in df.columns]

    if valid_cols:
        # 获取描述性统计并转置
        desc = df[valid_cols].describe().T
        # 格式化输出 (Pandas 直接转字符串打印)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.float_format', '{:.3f}'.format)
        print(desc[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']])

    print("\n" + "-" * 75)

    # 关键指标硬核检查
    if '场均收入' in df.columns:
        min_rev = df['场均收入'].min()
        status = "通过(PASS)" if min_rev > 0.001 else "警告(WARNING)"
        print(f"【关键指标核查】场均收入最小值: {min_rev:.2f} -> {status}")

    if 'Occ_Rate' in df.columns:
        max_occ = df['Occ_Rate'].max()
        status = "通过(PASS)" if max_occ <= 1.0 else "警告(WARNING)"
        print(f"【关键指标核查】上座率最大值: {max_occ:.4f} -> {status}")

    # -------------------------------------------------------------------------
    # 2. 爆款 vs 炮灰 (Tier Analysis)
    # -------------------------------------------------------------------------
    print("\n\n>>> 2. 爆款 vs 炮灰：各指标群体均值对比")
    print("-" * 75)

    if 'MovieName' in df.columns and '当前票房(万)' in df.columns:
        # 按电影汇总总票房，确定层级
        movie_groups = df.groupby('MovieName')['当前票房(万)'].sum().sort_values(ascending=False)
        n_movies = len(movie_groups)

        top_n = int(n_movies * 0.05)  # Top 5%
        bottom_n = int(n_movies * 0.2)  # Bottom 20%

        if top_n > 0 and bottom_n > 0:
            top_movies = movie_groups.head(top_n).index
            bottom_movies = movie_groups.tail(bottom_n).index
            # 中间剩下的
            mid_movies = movie_groups.iloc[top_n: -bottom_n].index

            tiers_map = {
                '[头部爆款] (Top 5%)': top_movies,
                '[中腰部] (20-80%)': mid_movies,
                '[长尾/炮灰] (Bottom 20%)': bottom_movies
            }

            # 要对比的特征列
            compare_cols = ['Eff_Index', 'Gold_Strength', 'Market_Share', 'Occ_Rate', 'Avg_Price']
            existing_compare = [c for c in compare_cols if c in df.columns]

            # 打印表头
            header = f"{'Tier':>25} " + " ".join([f"{c:>12}" for c in existing_compare]) + "   Count"
            print(header)

            tier_stats = {}  # 存储均值向量用于计算距离

            for tier_name, movies in tiers_map.items():
                # 筛选出属于该层级的行
                sub_df = df[df['MovieName'].isin(movies)]
                if sub_df.empty:
                    continue

                means = sub_df[existing_compare].mean()
                count = len(movies)

                # 打印行
                row_str = f"{tier_name:>25} " + " ".join(
                    [f"{means[c]:12.6f}" for c in existing_compare]) + f" {count:7d}"
                print(row_str)

                tier_stats[tier_name] = means.fillna(0).values

            # -----------------------------------------------------------------
            # 3. 特征空间分离度 (Distance)
            # -----------------------------------------------------------------
            print("\n\n>>> 3. 特征空间分离度证明 (SAC 识别能力验证)")
            print("-" * 75)

            k1 = '[头部爆款] (Top 5%)'
            k2 = '[长尾/炮灰] (Bottom 20%)'

            if k1 in tier_stats and k2 in tier_stats:
                vec_top = tier_stats[k1]
                vec_bottom = tier_stats[k2]

                # 计算欧氏距离
                dist = distance.euclidean(vec_top, vec_bottom)
                print(
                    f"爆款 (Top 5%) 与 炮灰 (Bottom 20%) 在 {len(existing_compare)}维特征空间中的欧氏距离: {dist:.4f}")

                if dist > 1.0:
                    judge = "【极佳】"
                elif dist > 0.5:
                    judge = "【良好】"
                else:
                    judge = "【一般】"

                print(f"综合质量判定: {judge} 特征组合区分度评价")
                print("[数学解读]: 距离越大，Agent 越容易区分状态好坏。")
        else:
            print("[提示] 电影数量过少，无法进行分层对比。")
    else:
        print("[提示] 缺少 MovieName 或 当前票房(万) 列，跳过分层分析。")

    print("\n" + "=" * 90)


if __name__ == "__main__":
    # 如果命令行带了参数，就用参数的文件路径
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    else:
        target_file = DEFAULT_FILE

    generate_report(target_file)