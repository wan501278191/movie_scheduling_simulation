# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import glob
import warnings

warnings.filterwarnings('ignore')

# === 配置 ===
MIN_SEGMENT_LENGTH = 2


def make_all_data_final(input_folder, output_file):
    print(f"[数据工厂] 启动强力清洗模式: {os.path.abspath(input_folder)}")

    all_files = glob.glob(os.path.join(input_folder, "**", "*.xlsx"), recursive=True)
    all_files = [f for f in all_files if not os.path.basename(f).startswith('~$')]

    print(f"[INFO] 扫描到 {len(all_files)} 个文件...")

    processed_dfs = []
    stats = {"files_processed": 0, "segments_saved": 0}

    for i, file_path in enumerate(all_files):
        if i % 100 == 0: print(f"   [进度] {i}/{len(all_files)} ...")

        try:
            file_name = os.path.basename(file_path)
            movie_base_name = file_name.rsplit('.', 1)[0]
            try:
                df = pd.read_excel(file_path, engine='openpyxl')
            except:
                continue

            if '日期/上映天数' not in df.columns: continue

            # --- 1. 日期与生命周期 ---
            df['real_date'] = pd.to_datetime(df['日期/上映天数'].str.split('|').str[0], errors='coerce')
            df['Days_Running'] = pd.to_numeric(df['日期/上映天数'].str.split('|').str[1], errors='coerce')
            if df['Days_Running'].isna().all():
                df = df.sort_values('real_date')
                s_date = df['real_date'].iloc[0]
                df['Days_Running'] = (df['real_date'] - s_date).dt.days + 1

            # --- 2. 价格 (Avg_Price) [强力清洗] ---
            if '当前平均票价' in df.columns:
                price = pd.to_numeric(df['当前平均票价'], errors='coerce').fillna(35.0)
                # 异常处理：票价 > 300 或 < 5 的，强制设为 35
                price = np.where((price > 300) | (price < 5), 35.0, price)
                df['Avg_Price'] = price
            else:
                df['Avg_Price'] = 35.0

            # --- 3. 百分比数据 (0~1) [强力清洗] ---
            # 规则：只要该列最大值 > 1.5，就说明是百分数(如 45.0)，全体除以 100
            # 即使除完，也要 clip 到 0~1 之间，防止 102% 这种脏数据

            # A. 场次占比
            if '场次占比' in df.columns:
                val = pd.to_numeric(df['场次占比'], errors='coerce').fillna(0)
                if val.max() > 1.5: val /= 100.0
                df['Market_Share'] = val.clip(0, 1.0)
            else:
                df['Market_Share'] = 0.0

            # B. 上座率
            if '上座率' in df.columns:
                val = pd.to_numeric(df['上座率'], errors='coerce').fillna(0)
                if val.max() > 1.5: val /= 100.0
                df['Occ_Rate'] = val.clip(0, 1.0)  # 锁死在 0-1
            else:
                df['Occ_Rate'] = 0.0

            # C. 票房占比
            if '票房占比' in df.columns:
                val = pd.to_numeric(df['票房占比'], errors='coerce').fillna(0)
                if val.max() > 1.5: val /= 100.0
                df['Box_Share'] = val.clip(0, 1.0)
            else:
                df['Box_Share'] = 0.0

            # --- 4. 黄金场硬度 [强力清洗] ---
            if '黄金场场次占比' in df.columns and '黄金场人次占比' in df.columns:
                g_show = pd.to_numeric(df['黄金场场次占比'], errors='coerce').fillna(0)
                g_person = pd.to_numeric(df['黄金场人次占比'], errors='coerce').fillna(0)

                # 同样的百分比处理逻辑
                if g_show.max() > 1.5: g_show /= 100.0
                if g_person.max() > 1.5: g_person /= 100.0

                g_show = g_show.clip(0, 1.0)
                g_person = g_person.clip(0, 1.0)

                # 计算比率，并截断
                ratio = g_person / g_show.replace(0, np.nan)
                ratio = ratio.fillna(1.0)
                # 截断：最高不超过 5.0 (5倍效率已经是神了，900肯定是bug)
                df['Gold_Strength'] = ratio.clip(0, 5.0).round(3)
            else:
                df['Gold_Strength'] = 1.0

            # --- 5. 供需指数 [强力清洗] ---
            if '供需指数' in df.columns:
                eff = pd.to_numeric(df['供需指数'], errors='coerce').fillna(1.0)
                # 截断：最高不超过 5.0
                df['Eff_Index'] = eff.clip(0, 5.0)
            else:
                df['Eff_Index'] = 1.0

            # --- 6. 辅助数据 ---
            # 大盘反推
            curr_show = pd.to_numeric(df['当前场次'], errors='coerce').fillna(0)
            # 避免除以0
            denom = df['Market_Share'].replace(0, np.nan)
            df['Market_Total_Show'] = (curr_show / denom).fillna(0).astype(int)

            df['当前票房(万)'] = pd.to_numeric(df['当前票房(万)'], errors='coerce').fillna(0)
            df['场均收入'] = pd.to_numeric(df['场均收入'], errors='coerce').fillna(0)
            df['当前场次'] = curr_show
            df['当前人次(万)'] = pd.to_numeric(df['当前人次(万)'], errors='coerce').fillna(0)

            # --- 7. 切片 ---
            df = df.dropna(subset=['real_date', 'Days_Running']).sort_values('real_date')
            df = df.drop_duplicates(subset=['real_date'], keep='first')

            df['date_diff'] = df['real_date'].diff().dt.days
            df['segment_id'] = (df['date_diff'].fillna(1) > 1).astype(int).cumsum()

            for seg_id, seg_df in df.groupby('segment_id'):
                if len(seg_df) < MIN_SEGMENT_LENGTH: continue

                final_seg = seg_df.copy()
                final_seg['MovieName'] = f"{movie_base_name}_seg{seg_id}"

                # Time State
                final_seg['Week_Index'] = final_seg['real_date'].dt.dayofweek
                final_seg['Time_State'] = final_seg['Week_Index'].apply(
                    lambda x: 1.0 if x >= 5 else (0.5 if x == 4 else 0.0)
                )

                # 选列
                target_cols = [
                    'MovieName', 'real_date',
                    'Eff_Index', 'Gold_Strength', 'Market_Share', 'Occ_Rate',
                    'Time_State', 'Days_Running', 'Avg_Price',
                    '当前票房(万)', '当前场次', '场均收入', 'Box_Share', '当前人次(万)', 'Market_Total_Show'
                ]

                # 补全
                for c in target_cols:
                    if c not in final_seg.columns: final_seg[c] = 0

                processed_dfs.append(final_seg[target_cols])
                stats["segments_saved"] += 1

            stats["files_processed"] += 1

        except Exception:
            pass

    if processed_dfs:
        final_df = pd.concat(processed_dfs, ignore_index=True)
        final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n[SUCCESS] 清洗完成！")
        print(f"数据已强制标准化：百分比列<=1.0, 票价在[5,300]之间, 指数<=5.0")
    else:
        print("[ERROR] 无数据。")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = current_dir
    for root, dirs, files in os.walk(current_dir):
        for d in dirs:
            if "2019" in d:
                data_dir = os.path.join(root, d)
                break
        if data_dir != current_dir: break
    output_path = os.path.join(current_dir, "rl_data_final.csv")
    make_all_data_final(data_dir, output_path)