# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import sys
from scipy.spatial import distance
import itertools

# === 配置 ===
# 默认读取的文件名
DEFAULT_FILE = "rl_data_final.csv"
# 输出报告的文件名
REPORT_OUTPUT_FILE = "quality_audit_report.txt"


class DualLogger:
    """
    自定义Logger，同时将信息输出到控制台和文件
    """

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # 立即写入，防止丢失

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def generate_report(file_path):
    if not os.path.exists(file_path):
        print(f"[错误] 找不到文件: {file_path}")
        return

    # --- 1. 重定向输出到文件 ---
    # 保存原始 stdout 以便恢复（可选）
    original_stdout = sys.stdout
    sys.stdout = DualLogger(REPORT_OUTPUT_FILE)

    print(f"正在读取数据 {file_path} 并开始深度质量审计...")
    print(f"完整报告将同步保存至: {os.path.abspath(REPORT_OUTPUT_FILE)}")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"[错误] 读取失败: {e}")
        return

    # 基础信息
    total_samples = len(df)
    unique_movies = df['MovieName'].nunique() if 'MovieName' in df.columns else 0

    # 报告头部
    print("=" * 120)
    print("            电影排片强化学习数据质量审计证明书 (全面评估版)")
    print("=" * 120)
    print(f"审计对象: {os.path.basename(file_path)}")
    print(f"样本总量: {total_samples}")
    print(f"独立电影总数: {unique_movies} 个")
    print("-" * 120)

    # -------------------------------------------------------------------------
    # 0. 零值专项审计
    # -------------------------------------------------------------------------
    print("\n>>> 0. 零值专项审计 (当前数据状态)")
    print("-" * 75)
    print(f"{'列名':>15}  {'零值数量':>10}  {'零值占比(%)':>12}")

    audit_cols = [
        'Time_State', 'Box_Share', 'Eff_Index', '当前票房(元)', '当前人次',
        'Gold_Strength', '场均收入', 'Occ_Rate', 'Market_Share', '当前场次',
        'Avg_Price', 'Days_Running', 'Market_Total_Show'
    ]

    for col in audit_cols:
        if col in df.columns:
            zero_count = len(df[(df[col].isna()) | (df[col] <= 0.0001)])
            ratio = (zero_count / total_samples) * 100
            print(f"{col:>15}  {zero_count:10d}  {ratio:12.2f}")

    print("\n[注]: 若 Time_State 零值高属正常现象(平日为0)。")

    # -------------------------------------------------------------------------
    # 1. 数据分布自检 (修改：不再转置，Feature为列，Stat为行)
    # -------------------------------------------------------------------------
    print("\n\n>>> 1. 数据分布自检 (基础统计)")
    print("-" * 120)

    stats_cols = [
        'Eff_Index', 'Gold_Strength', 'Market_Share', 'Occ_Rate', 'Time_State',
        'Days_Running', 'Avg_Price', '当前票房(元)', '当前场次', '场均收入',
        'Box_Share', '当前人次', 'Market_Total_Show'
    ]

    valid_cols = [c for c in stats_cols if c in df.columns]

    if valid_cols:
        # 获取描述性统计 (不加 .T，保持默认格式: Index是count/mean/std...)
        desc = df[valid_cols].describe()

        # 格式化输出设置
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 2000)  # 增加宽度防止换行
        pd.set_option('display.float_format', '{:.3f}'.format)

        print(desc)

    print("\n" + "-" * 120)

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
    print("-" * 120)

    if 'MovieName' in df.columns and '当前票房(元)' in df.columns:
        movie_groups = df.groupby('MovieName')['当前票房(元)'].sum().sort_values(ascending=False)
        n_movies = len(movie_groups)

        top_n = int(n_movies * 0.05)  # Top 5%
        bottom_n = int(n_movies * 0.2)  # Bottom 20%

        if top_n > 0 and bottom_n > 0:
            top_movies = movie_groups.head(top_n).index
            bottom_movies = movie_groups.tail(bottom_n).index
            mid_movies = movie_groups.iloc[top_n: -bottom_n].index

            tiers_map = {
                '[头部爆款] (Top 5%)': top_movies,
                '[中腰部] (20-80%)': mid_movies,
                '[长尾/炮灰] (Bottom 20%)': bottom_movies
            }

            compare_cols = ['Eff_Index', 'Gold_Strength', 'Market_Share', 'Occ_Rate', 'Avg_Price']
            existing_compare = [c for c in compare_cols if c in df.columns]

            header = f"{'Tier':>25} " + " ".join([f"{c:>12}" for c in existing_compare]) + "   Count"
            print(header)

            tier_stats = {}

            for tier_name, movies in tiers_map.items():
                sub_df = df[df['MovieName'].isin(movies)]
                if sub_df.empty: continue

                means = sub_df[existing_compare].mean()
                count = len(movies)

                row_str = f"{tier_name:>25} " + " ".join(
                    [f"{means[c]:12.6f}" for c in existing_compare]) + f" {count:7d}"
                print(row_str)

                tier_stats[tier_name] = means.fillna(0).values

            # -----------------------------------------------------------------
            # 3. 特征空间分离度 (两两计算)
            # -----------------------------------------------------------------
            print("\n\n>>> 3. 特征空间分离度证明 (SAC 识别能力验证 - 两两对比)")
            print("-" * 120)

            # 获取所有层级的两两组合
            pairs = list(itertools.combinations(tier_stats.keys(), 2))

            for name1, name2 in pairs:
                vec1 = tier_stats[name1]
                vec2 = tier_stats[name2]
                dist = distance.euclidean(vec1, vec2)

                # 简单的评价逻辑
                if dist > 1.0:
                    judge = "【极佳】"
                elif dist > 0.5:
                    judge = "【良好】"
                else:
                    judge = "【一般】"

                print(f"{name1:<18} vs {name2:<18} -> 欧氏距离: {dist:.4f}  {judge}")

            print("\n[数学解读]: 距离越大，Agent 越容易区分状态好坏。")
            print("           只要爆款与其他两类的距离足够大，模型就能学到“往爆款方向走”的策略。")

        else:
            print("[提示] 电影数量过少，无法进行分层对比。")
    else:
        print("[提示] 缺少 MovieName 或 当前票房(万) 列，跳过分层分析。")

    print("\n" + "=" * 120)

    # 恢复标准输出 (虽然程序结束也会自动关闭，但好习惯)
    sys.stdout = original_stdout
    print(f"报告已生成完毕，请查看文件: {REPORT_OUTPUT_FILE}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    else:
        target_file = DEFAULT_FILE

    generate_report(target_file)