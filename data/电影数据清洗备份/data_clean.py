# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import warnings

# 忽略 pandas 的一些不必要的警告
warnings.filterwarnings('ignore')

# 设置显示选项，确保控制台输出对齐
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)


def print_data_report(df, stage_name):
    """
    打印详细的数据统计报告 (Sanity Check + 群体特征)
    """
    print("\n" + "#" * 80)
    print(f"   >>> {stage_name} 数据统计报告")
    print("#" * 80)

    print(f"[INFO] 数据总行数: {len(df)}")
    print(f"[INFO] 电影片段数: {df['MovieName'].nunique()}")

    # ------------------------------------------------------
    # 1. 基础统计 (Sanity Check)
    # ------------------------------------------------------
    print("\n" + "=" * 60)
    print(">>> 1. 数据质量自检 (Sanity Check)")
    print("=" * 60)

    # 筛选数值列进行展示
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # 排除辅助列
    display_cols = [c for c in numeric_cols if c not in ['seg_id', 'Week_Index', 'log_income']]

    # 打印 describe
    if not df.empty:
        desc = df[display_cols].describe().round(3)
        print(desc.to_string())
    else:
        print("[WARN] 数据为空，无法统计")

    # 阈值检查
    print("-" * 30)
    if 'Occ_Rate' in df.columns and not df.empty:
        max_occ = df['Occ_Rate'].max()
        if max_occ > 1.0:
            print(f"[WARN] 上座率 > 1.0 (最大值: {max_occ})")
        else:
            print(f"[PASS] 上座率范围正常 (0.0 ~ 1.0)")

    if 'Market_Share' in df.columns and not df.empty:
        max_share = df['Market_Share'].max()
        if max_share > 1.0:
            print(f"[WARN] 排片占比 > 1.0 (最大值: {max_share})")
        else:
            print(f"[PASS] 排片占比范围正常 (0.0 ~ 1.0)")

    # ------------------------------------------------------
    # 2. 爆款 vs 烂片特征差异 (Group Stats)
    # ------------------------------------------------------
    print("\n" + "=" * 60)
    print(">>> 2. 爆款 vs 烂片：群体特征差异")
    print("=" * 60)

    if df.empty:
        return

    # 按电影分组统计
    movie_stats = df.groupby('MovieName').agg({
        '当前票房(万)': 'sum',
        'Eff_Index': 'mean',
        'Gold_Strength': 'mean',
        'Market_Share': 'mean',
        'Occ_Rate': 'mean',
        'Avg_Price': 'mean'
    }).reset_index()

    # 动态计算分位点
    q95 = movie_stats['当前票房(万)'].quantile(0.95)
    q50 = movie_stats['当前票房(万)'].quantile(0.50)
    q20 = movie_stats['当前票房(万)'].quantile(0.20)

    def get_tier(box):
        if box >= q95:
            return '[头部爆款] (Top 5%)'
        elif box >= q20 and box <= q50:
            return '[中腰部] (20-50%)'
        elif box < q20:
            return '[长尾/炮灰] (Bottom 20%)'
        return '其他'

    movie_stats['Tier'] = movie_stats['当前票房(万)'].apply(get_tier)

    # 聚合展示
    tier_summary = movie_stats[movie_stats['Tier'] != '其他'].groupby('Tier').agg({
        'Eff_Index': 'mean',
        'Gold_Strength': 'mean',
        'Market_Share': 'mean',
        'Occ_Rate': 'mean',
        'Avg_Price': 'mean',
        'MovieName': 'count'
    }).reset_index()

    cols = ['Tier', 'Eff_Index', 'Gold_Strength', 'Market_Share', 'Occ_Rate', 'Avg_Price', 'MovieName']
    print(tier_summary[cols].to_string(index=False))


def main():
    # 1. 设置文件路径 (请确保文件名正确)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'rl_data_final.csv')
    output_path = os.path.join(current_dir, 'rl_data_cleaned.csv')

    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        # 尝试在上级目录找
        file_path = 'rl_data_final.csv'
        if not os.path.exists(file_path):
            print("请确认 CSV 文件路径是否正确。")
            return

    # === 2. 读取原始数据 ===
    print("正在读取数据...")
    df = pd.read_csv(file_path)

    # 提取 seg_id 辅助列 (用于判断是否为后期长尾)
    df['seg_id'] = df['MovieName'].apply(lambda x: int(x.split('_seg')[-1]) if '_seg' in x else 0)

    # === 3. 打印清洗前报告 ===
    print_data_report(df, "清洗前 (Raw Data)")

    # === 4. 执行清洗逻辑 ===
    print("\n" + ">" * 20 + " 正在执行清洗与修复... " + "<" * 20)

    cleaned_segments = []
    stats = {'dropped_zero': 0, 'dropped_short_late': 0, 'repaired': 0}

    # 按片段分组处理
    grouped = df.groupby('MovieName')
    total_groups = len(grouped)

    for i, (name, group) in enumerate(grouped):
        # 简单的进度显示 (每1000个显示一次)
        if i % 1000 == 0:
            print(f"处理进度: {i}/{total_groups} ...")

        group = group.sort_values('real_date').copy()

        # --- A. 智能插值修复 ---
        # 只有当片段里曾经有过收入(max > 0.1)时，才值得修
        max_income = group['场均收入'].max()

        if max_income > 0.1:
            # 标记 <= 0.1 的值为 NaN
            mask_zeros = group['场均收入'] <= 0.1
            if mask_zeros.any():
                stats['repaired'] += 1
                group.loc[mask_zeros, '场均收入'] = np.nan

                # [关键修改] 使用新的 bfill/ffill 方法，避免 FutureWarning
                group['场均收入'] = group['场均收入'].interpolate(method='linear', limit_direction='both')
                group['场均收入'] = group['场均收入'].bfill().ffill()

        # --- B. 过滤逻辑 ---
        # 重新计算（因为插值可能改变了数值）
        current_max_income = group['场均收入'].max()
        seg_id = group['seg_id'].iloc[0]
        duration = len(group)

        # 规则1: 剔除全0死片段 (救不回来的)
        if current_max_income <= 0.1 or pd.isna(current_max_income):
            stats['dropped_zero'] += 1
            continue

        # 规则2: 剔除后期(seg>0)且过短(<=2天)的片段
        if seg_id > 0 and duration <= 2:
            stats['dropped_short_late'] += 1
            continue

        cleaned_segments.append(group)

    # === 5. 合并并保存 ===
    if cleaned_segments:
        final_df = pd.concat(cleaned_segments, ignore_index=True)
        final_df.drop(columns=['seg_id'], inplace=True, errors='ignore')

        print(f"\n[清洗统计]")
        print(f"  - 剔除全0收入片段 (无效): {stats['dropped_zero']} 个")
        print(f"  - 剔除后期短命片段 (噪声): {stats['dropped_short_late']} 个")
        print(f"  - 执行插值修复片段 (补全): {stats['repaired']} 个")

        # === 6. 打印清洗后报告 ===
        print_data_report(final_df, "清洗后 (Cleaned Data)")

        # 保存
        final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n[SUCCESS] 清洗后的数据已保存至: {os.path.abspath(output_path)}")

    else:
        print("[ERROR] 清洗后数据为空！请检查筛选逻辑。")


if __name__ == "__main__":
    main()