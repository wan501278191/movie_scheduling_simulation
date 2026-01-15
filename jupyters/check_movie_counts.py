import pandas as pd
from tqdm import tqdm
import os

# 确保可以正确导入您项目中的工具函数
from tools import query_valid_movie_name_in_date


def find_insufficient_movie_periods(start_date='2019-01-01', end_date='2024-09-30', threshold=6):
    """
    扫描一个日期范围，找出每日上映电影数量不足的时期。

    :param start_date: str, 扫描开始日期
    :param end_date: str, 扫描结束日期
    :param threshold: int, 电影数量的下限阈值
    """

    calendar_path = '../data/movie_calendar.csv'

    print(f"正在从路径: '{calendar_path}' 加载电影日历...")

    if not os.path.exists(calendar_path):
        print(f"错误：文件未找到！请检查路径是否正确: {os.path.abspath(calendar_path)}")
        return

    # --- 修改点在这里 ---
    # 在 parse_dates 列表中加入 '下线日期'，将两列都转换为日期格式
    movie_calendar = pd.read_csv(calendar_path, parse_dates=['上映日期', '下线日期'])
    # --- 修改结束 ---

    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    insufficient_dates = []

    print(f"正在扫描从 {start_date} 到 {end_date} 的每一天...")
    for current_date in tqdm(date_range):
        valid_movie_table = query_valid_movie_name_in_date(current_date, movie_calendar)
        if len(valid_movie_table) < threshold:
            insufficient_dates.append(current_date.strftime('%Y-%m-%d'))

    if not insufficient_dates:
        print("恭喜！在指定范围内，每天的电影数量都足够。")
        return

    print("\n--- 诊断完成 ---")
    print(f"在以下日期，上映的电影数量少于 {threshold} 部：")

    start_period = insufficient_dates[0]
    for i in range(1, len(insufficient_dates)):
        prev_date = pd.to_datetime(insufficient_dates[i - 1])
        curr_date = pd.to_datetime(insufficient_dates[i])
        if (curr_date - prev_date).days > 1:
            end_period = insufficient_dates[i - 1]
            if start_period == end_period:
                print(start_period)
            else:
                print(f"{start_period} 至 {end_period}")
            start_period = insufficient_dates[i]

    end_period = insufficient_dates[-1]
    if start_period == end_period:
        print(start_period)
    else:
        print(f"{start_period} 至 {end_period}")


if __name__ == '__main__':
    find_insufficient_movie_periods()