import os
import pandas as pd
import glob
import numpy as np


def gene_movie_calendar() -> pd.DataFrame:
    """
    生成全部电影的上映和下线日期表
    :return: pd.DataFrame,
        全部电影的上映和下线日期表，columns['电影名称', '上映日期', '下线日期']
    """
    # --- 新增：动态计算项目根目录，这是最小改动的核心 ---
    # 获取当前文件(tools.py)所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 从当前目录返回上一级 (..)，得到项目根目录
    project_root = os.path.join(current_dir, '..')

    # 根据项目根目录，构建所有需要用到的文件的绝对路径
    calendar_csv_path = os.path.join(project_root, 'data', 'movie_calendar.csv')
    raw_data_dir = os.path.join(project_root, 'data', '2019-2024电影每日票房信息')
    # --- 新增结束 ---

    # --- 修改：使用计算出的绝对路径 ---
    if os.path.exists(calendar_csv_path):
        movie_online_date_df = pd.read_csv(calendar_csv_path)
        movie_online_date_df['上映日期'] = pd.to_datetime(movie_online_date_df['上映日期'])
        movie_online_date_df['下线日期'] = pd.to_datetime(movie_online_date_df['下线日期'])
        return movie_online_date_df

    else:
        # --- 修改：使用计算出的绝对路径 ---
        movie_filename_list = glob.glob(os.path.join(raw_data_dir, '*.xlsx'))
        # 增加过滤临时文件的逻辑，使代码更健壮
        movie_filename_list = [f for f in movie_filename_list if not os.path.basename(f).startswith('~$')]

        movie_online_date_list = []
        for item in movie_filename_list:
            movie_name = os.path.basename(item).strip('.xlsx')  # 使用 os.path.basename 更健壮
            item_data = pd.read_excel(item, index_col=0)
            online_date = item_data['日期/上映天数'].iloc[0].split('|')[0]
            offline_date = item_data['日期/上映天数'].iloc[-1].split('|')[0]
            movie_online_date_list.append([movie_name, online_date, offline_date])
        movie_online_date_df = pd.DataFrame(movie_online_date_list, columns=['电影名称', '上映日期', '下线日期'])

        movie_online_date_df['上映日期'] = pd.to_datetime(movie_online_date_df['上映日期'])
        movie_online_date_df['下线日期'] = pd.to_datetime(movie_online_date_df['下线日期'])
        movie_online_date_df.sort_values(by='上映日期', inplace=True)

        # --- 修改：使用计算出的绝对路径 ---
        # 同时确保目录存在
        os.makedirs(os.path.dirname(calendar_csv_path), exist_ok=True)
        movie_online_date_df.to_csv(calendar_csv_path, index=False, encoding='utf-8-sig')
        return movie_online_date_df


# in tools.py

def query_valid_movie_name_in_date(current_date: pd.Timestamp, movie_calendar: pd.DataFrame) -> pd.DataFrame:
    """
    根据上映和下线日期表，筛选出在当前日期还在上映的电影

    :param current_date: pd.Timestamp,
        系统当前日期
    :param movie_calendar: pd.DataFrame,
        上映和下线日期表，columns['电影名称', '上映日期', '下线日期']
    :return: pd.DataFrame,
        在当前日期还在上映的电影列表，columns['电影名称', '上映日期', '下线日期']
    """
    # ========================= [ 关键修改 ] =========================
    # 将 < 修改为 <=，将 > 修改为 >=，以包含上映和下线的当天
    valid_index = ((movie_calendar['上映日期'] <= current_date) &
                   (movie_calendar['下线日期'] >= current_date))
    # ================================================================

    valid_data = movie_calendar[valid_index]
    return valid_data


def trans_weekindex_to_array(week_index: np.ndarray) -> np.ndarray:
    """
    将周索引转换为对应的数组
    :param week_index: int,
        周索引
    :return: np.ndarray,
        对应的数组
    """
    week_onehot_list = []
    for wi in week_index:
        week_onehot = np.zeros(7)
        week_onehot[int(wi)] = 1
        week_onehot_list.append(week_onehot)
    week_onehot_array = np.array(week_onehot_list)
    return week_onehot_array

#将收入转换为效用值
def trans_income_to_utility_type_1(income: float, base_income) -> float:
    if income <= base_income:  #25000
        utility = income
    else:
        a = 21000
        d = -190000
        utility = a * np.log(income) + d # 防止高收入导致过大数值差距
    return utility
