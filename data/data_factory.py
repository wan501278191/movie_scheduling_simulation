# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import glob
import warnings
import re
from tqdm import tqdm

warnings.filterwarnings('ignore')

# === 配置 ===
MIN_SEGMENT_LENGTH = 2
DEAD_SEGMENT_THRESHOLD = 0.8  # 垃圾时间阈值


def standardize_columns(df):
    """
    列名标准化：去除空格，统一常见列名格式
    """
    df.columns = df.columns.str.strip()
    col_map = {}
    for col in df.columns:
        clean_name = re.sub(r'\(.*?\)|（.*?）', '', col).strip()
        if clean_name in ['当前票房', '票房']:
            col_map[col] = '当前票房（元）'
        elif clean_name in ['当前人次', '人次']:
            col_map[col] = '当前人次'
        elif clean_name in ['当前场次', '场次']:
            col_map[col] = '当前场次'
        elif clean_name in ['平均票价', '票价', '当前平均票价']:
            col_map[col] = '当前平均票价'
        elif clean_name in ['场均收入', '场均红利']:
            col_map[col] = '场均收入'
        elif clean_name in ['场次占比']:
            col_map[col] = '场次占比'
        elif clean_name in ['票房占比']:
            col_map[col] = '票房占比'
        elif clean_name in ['上座率']:
            col_map[col] = '上座率'
        elif '日期' in clean_name and '天数' in clean_name:
            col_map[col] = '日期/上映天数'

    if col_map:
        df = df.rename(columns=col_map)
    return df


def smart_fill_series(original_series, fallback_calculation=None):
    """
    v3.2 核心逻辑：
    1. 读取原列
    2. 0值转NaN -> 线性插值 (Interpolate)
    3. 仍为空 -> 使用 fallback_calculation 兜底
    """
    # 1. 确保是数字，强制把 0 当作缺失值处理 (以便插值)
    s = pd.to_numeric(original_series, errors='coerce').replace(0, np.nan)

    # 2. 线性插值 (利用前后天的数据补全中间的0)
    s_interp = s.interpolate(method='linear', limit=3, limit_direction='both')

    # 3. 如果插值后还是 NaN (说明整段都是0)，才使用计算公式覆盖
    if fallback_calculation is not None:
        s_interp = s_interp.fillna(fallback_calculation)

    return s_interp


def build_global_market_view(all_files):
    """
    构建大盘字典 (仅用于第三步兜底计算)
    """
    print(f"\n[Phase 1] 正在构建全局大盘字典 (仅做计算兜底用)...")
    global_market_dict = {}

    for file_path in tqdm(all_files, desc="扫描大盘"):
        try:
            df = pd.read_excel(file_path, engine='openpyxl')
            df = standardize_columns(df)

            if not set(['日期/上映天数', '当前场次', '场次占比']).issubset(df.columns): continue

            df['real_date'] = pd.to_datetime(df['日期/上映天数'].str.split('|').str[0], errors='coerce')
            df = df.dropna(subset=['real_date'])

            curr_show = pd.to_numeric(df['当前场次'], errors='coerce').fillna(0)
            share = pd.to_numeric(df['场次占比'], errors='coerce').fillna(0)
            share = np.where(share > 1.5, share / 100.0, share)  # 修正百分比

            # 只有当 share > 0 时才能反推
            valid_mask = share > 0.0001
            implied_total = curr_show[valid_mask] / share[valid_mask]

            for date, total in zip(df.loc[valid_mask, 'real_date'], implied_total):
                if total > global_market_dict.get(date, 0):
                    global_market_dict[date] = int(total)
        except:
            continue
    return global_market_dict


def make_all_data_final(input_folder, output_file):
    print(f"[数据工厂 v3.2] 启动: 严格执行 读取优先 -> 插值 -> 计算兜底")

    all_files = glob.glob(os.path.join(input_folder, "**", "*.xlsx"), recursive=True)
    all_files = [f for f in all_files if not os.path.basename(f).startswith('~$')]

    # Phase 1: 准备大盘数据 (备用)
    global_market_dict = build_global_market_view(all_files)

    processed_dfs = []
    stats = {"files_processed": 0, "segments_saved": 0}

    # Phase 2: 清洗
    print(f"\n[Phase 2] 开始处理...")
    for file_path in tqdm(all_files, desc="Processing"):
        try:
            file_name = os.path.basename(file_path)
            movie_base_name = file_name.rsplit('.', 1)[0]

            try:
                df = pd.read_excel(file_path, engine='openpyxl')
            except:
                continue

            df = standardize_columns(df)
            if '日期/上映天数' not in df.columns: continue

            # --- 基础列 ---
            df['real_date'] = pd.to_datetime(df['日期/上映天数'].str.split('|').str[0], errors='coerce')
            df['Days_Running'] = pd.to_numeric(df['日期/上映天数'].str.split('|').str[1], errors='coerce')
            # 修复上映天数
            if df['Days_Running'].isna().all():
                df = df.sort_values('real_date')
                if not df.empty:
                    s_date = df['real_date'].iloc[0]
                    df['Days_Running'] = (df['real_date'] - s_date).dt.days + 1

            # 基础数值 (补0)
            df['当前票房(万)'] = pd.to_numeric(df.get('当前票房(万)', 0), errors='coerce').fillna(0)
            df['当前人次(万)'] = pd.to_numeric(df.get('当前人次(万)', 0), errors='coerce').fillna(0)
            df['当前场次'] = pd.to_numeric(df.get('当前场次', 0), errors='coerce').fillna(0)

            # 准备大盘总数 (作为备用计算参数)
            total_market_show = df['real_date'].map(global_market_dict).fillna(0)
            df['Market_Total_Show'] = total_market_show

            # =========================================================
            # 核心修改: 严格执行 读取 -> 插值 -> 计算
            # =========================================================

            # --- 1. 票价 (Avg_Price) ---
            # 备用公式: 票房 / 人次
            calc_price = (df['当前票房(万)'] * 10000) / (df['当前人次(万)'] * 10000).replace(0, np.nan)

            # 逻辑: 读原列 -> 插值 -> 用公式补
            orig_price = df.get('当前平均票价', pd.Series(np.nan))
            final_price = smart_fill_series(orig_price, fallback_calculation=calc_price)

            # 最终范围清洗
            final_price = final_price.fillna(35.0)
            df['Avg_Price'] = np.where((final_price > 300) | (final_price < 5), 35.0, final_price)

            # --- 2. 场均收入 (Avg_Show_Rev) ---
            # 备用公式: 票房 / 场次
            calc_rev = (df['当前票房(万)'] * 10000) / df['当前场次'].replace(0, np.nan)

            # 逻辑: 读原列 -> 插值 -> 用公式补
            orig_rev = df.get('场均收入', pd.Series(np.nan))
            df['场均收入'] = smart_fill_series(orig_rev, fallback_calculation=calc_rev).fillna(0)

            # --- 3. 场次占比 (Market_Share) ---
            # 备用公式: 当前场次 / 大盘总场次
            calc_share = df['当前场次'] / df['Market_Total_Show'].replace(0, np.nan)

            # 读取原列并预处理 (先把 45.0 变成 0.45，否则插值会出错)
            orig_share = pd.to_numeric(df.get('场次占比', pd.Series(np.nan)), errors='coerce')
            orig_share = np.where(orig_share > 1.5, orig_share / 100.0, orig_share)
            orig_share = pd.Series(orig_share)  # 转回 Series

            # 逻辑: 读原列 -> 插值 -> 用公式补
            df['Market_Share'] = smart_fill_series(orig_share, fallback_calculation=calc_share).fillna(0).clip(0, 1.0)

            # --- 4. 其他百分比 (无公式兜底，仅插值) ---
            # 上座率
            orig_occ = pd.to_numeric(df.get('上座率', pd.Series(np.nan)), errors='coerce')
            orig_occ = np.where(orig_occ > 1.5, orig_occ / 100.0, orig_occ)
            df['Occ_Rate'] = smart_fill_series(pd.Series(orig_occ)).fillna(0).clip(0, 1.0)

            # 票房占比
            orig_box_share = pd.to_numeric(df.get('票房占比', pd.Series(np.nan)), errors='coerce')
            orig_box_share = np.where(orig_box_share > 1.5, orig_box_share / 100.0, orig_box_share)
            df['Box_Share'] = smart_fill_series(pd.Series(orig_box_share)).fillna(0).clip(0, 1.0)

            # --- 5. 指数类 ---
            if '黄金场场次占比' in df.columns and '黄金场人次占比' in df.columns:
                g_show = pd.to_numeric(df['黄金场场次占比'], errors='coerce').fillna(0)
                g_person = pd.to_numeric(df['黄金场人次占比'], errors='coerce').fillna(0)
                if g_show.max() > 1.5: g_show /= 100.0
                if g_person.max() > 1.5: g_person /= 100.0
                ratio = g_person / g_show.replace(0, np.nan)
                df['Gold_Strength'] = ratio.fillna(1.0).clip(0, 5.0)
            else:
                df['Gold_Strength'] = 1.0

            eff = pd.to_numeric(df.get('供需指数', 1.0), errors='coerce').fillna(1.0)
            df['Eff_Index'] = eff.clip(0, 5.0)

            # --- 6. 切片保存 ---
            df = df.dropna(subset=['real_date', 'Days_Running']).sort_values('real_date')
            df = df.drop_duplicates(subset=['real_date'], keep='first')

            df['date_diff'] = df['real_date'].diff().dt.days
            df['segment_id'] = (df['date_diff'].fillna(1) > 1).astype(int).cumsum()

            for seg_id, seg_df in df.groupby('segment_id'):
                if len(seg_df) < MIN_SEGMENT_LENGTH: continue

                # 死刑判定: 场均收入几乎全是0则丢弃
                if (seg_df['场均收入'] <= 0.1).mean() > DEAD_SEGMENT_THRESHOLD:
                    continue

                final_seg = seg_df.copy()
                final_seg['MovieName'] = f"{movie_base_name}_seg{seg_id}"
                final_seg['Week_Index'] = final_seg['real_date'].dt.dayofweek
                final_seg['Time_State'] = final_seg['Week_Index'].apply(
                    lambda x: 1.0 if x >= 5 else (0.5 if x == 4 else 0.0)
                )

                cols = [
                    'MovieName', 'real_date', 'Eff_Index', 'Gold_Strength', 'Market_Share',
                    'Occ_Rate', 'Time_State', 'Days_Running', 'Avg_Price',
                    '当前票房(万)', '当前场次', '场均收入', 'Box_Share', '当前人次(万)', 'Market_Total_Show'
                ]

                for c in cols:
                    if c not in final_seg.columns: final_seg[c] = 0

                processed_dfs.append(final_seg[cols])
                stats["segments_saved"] += 1

            stats["files_processed"] += 1

        except Exception:
            pass

    if processed_dfs:
        final_df = pd.concat(processed_dfs, ignore_index=True)
        final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n[SUCCESS] 清洗完成 v3.2！保留片段: {stats['segments_saved']}")
    else:
        print("[ERROR] 无数据生成。")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = current_dir
    for root, dirs, files in os.walk(current_dir):
        for d in dirs:
            if "20" in d and len(d) == 4:
                data_dir = os.path.join(root, d)
                break
        if data_dir != current_dir: break

    output_path = os.path.join(current_dir, "rl_data_final.csv")
    make_all_data_final(data_dir, output_path)