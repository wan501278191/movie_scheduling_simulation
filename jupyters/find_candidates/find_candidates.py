# -*- coding: utf-8 -*-
import pandas as pd
import os

# ================= 配置区域 =================

# 1. 自动定位路径
# 逻辑：当前脚本在 jupyters/xxx/ -> 上两级找到 data/ -> 读取 rl_data_final.csv
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# 备用：如果自动定位失败，请取消下面这行的注释并手动修改
# DATA_DIR = r"D:\App\Pycharm\1223MovieSchedulingSimulation\data"

INPUT_FILE = os.path.join(DATA_DIR, 'rl_data_final.csv')
OUTPUT_CSV = os.path.join(CURRENT_DIR, 'candidates_result.csv')
OUTPUT_TXT = os.path.join(CURRENT_DIR, 'candidates_summary.txt')

# 2. 筛选时间范围 (只看 2023 年的数据)
ENABLE_DATE_FILTER = True
TEST_START_DATE = '2023-09-01'
TEST_END_DATE = '2024-9-30'

# 3. 判定阈值
# 黑马：总票房 > 5亿 (50000万) 且 首日排片 < 20%
DARK_HORSE_GROSS = 50000
DARK_HORSE_ALLOC = 20.0

# 烂片：总票房 < 1.5亿 (15000万) 且 首日排片 > 20%
FLOP_GROSS = 2000
FLOP_ALLOC = 10.0


# ===========================================

def main():
    print(f"--- 启动极速筛选 (基于训练数据 rl_data_final.csv) ---")
    print(f"读取数据: {INPUT_FILE}")

    if not os.path.exists(INPUT_FILE):
        print(f"错误: 找不到文件 {INPUT_FILE}，请检查路径。")
        return

    # 1. 读取数据 (秒读)
    try:
        df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        print(f"读取失败: {e}")
        return

    print(f"数据加载完毕，正在分析 {len(df)} 条记录...")

    # 2. 数据预处理
    # 这一步是为了计算总票房。
    # 您的数据中“当前票房(万)”实际上是元，所以除以 10000 换算成万元
    # 如果您的数据已经是万元，请把 /10000.0 去掉
    df['BoxOffice_Wan'] = df['当前票房(万)'] / 10000.0

    # 清洗片名 (去掉 _seg0 后缀)
    df['CleanName'] = df['MovieName'].apply(lambda x: x.split('_seg')[0] if '_seg' in str(x) else x)

    # 3. 聚合计算：一部电影一行
    # group_df 保存每部电影的聚合信息
    # - sum: 算总票房
    # - first: 拿第一天的数据 (假设数据是按时间排序的，或者稍后我们过滤 Days_Running=1)

    # 更严谨的做法：先提取首日数据，再提取总票房，然后合并

    # 3.1 算总票房
    total_box = df.groupby('CleanName')['BoxOffice_Wan'].sum().reset_index()
    total_box.rename(columns={'BoxOffice_Wan': 'TotalBox'}, inplace=True)

    # 3.2 找首日数据 (Days_Running == 1)
    first_days = df[df['Days_Running'] == 1][['CleanName', 'real_date', 'Market_Share']].copy()
    # Market_Share 0.2 变成 20.0%
    first_days['FirstDayAlloc'] = first_days['Market_Share'] * 100.0
    first_days.rename(columns={'real_date': 'ReleaseDate'}, inplace=True)
    # 去重（防止同一天有多个 _seg 记录）
    first_days.drop_duplicates(subset=['CleanName'], inplace=True)

    # 3.3 合并
    merged = pd.merge(first_days, total_box, on='CleanName', how='inner')

    # 4. 筛选逻辑
    results = []

    # 转换日期对象
    merged['ReleaseDate'] = pd.to_datetime(merged['ReleaseDate'])
    start_dt = pd.to_datetime(TEST_START_DATE)
    end_dt = pd.to_datetime(TEST_END_DATE)

    for _, row in merged.iterrows():
        name = row['CleanName']
        gross = row['TotalBox']
        alloc = row['FirstDayAlloc']
        date = row['ReleaseDate']

        # 日期筛选
        if ENABLE_DATE_FILTER:
            if not (start_dt <= date <= end_dt):
                continue

        category = None
        # 判定黑马
        if gross > DARK_HORSE_GROSS and alloc < DARK_HORSE_ALLOC:
            category = "黑马 (低开高走)"
        # 判定烂片
        elif gross < FLOP_GROSS and alloc > FLOP_ALLOC:
            category = "烂片 (高开低走)"

        if category:
            results.append({
                '电影名称': name,
                '类型': category,
                '上映日期': date.strftime('%Y-%m-%d'),
                '总票房(万)': round(gross, 2),
                '首日排片(%)': round(alloc, 2)
            })

    # 5. 输出
    if not results:
        print("未找到符合条件的电影。")
        return

    res_df = pd.DataFrame(results)
    # 排序
    res_df.sort_values(by=['类型', '总票房(万)'], ascending=[True, False], inplace=True)

    # 保存 CSV
    res_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    # 保存 TXT
    with open(OUTPUT_TXT, 'w', encoding='utf-8') as f:
        f.write(f"筛选结果 (区间: {TEST_START_DATE} ~ {TEST_END_DATE})\n{'=' * 60}\n")

        for cat, group in res_df.groupby('类型'):
            header = f"\n【{cat}】 共 {len(group)} 部:"
            print(header)
            f.write(header + "\n")
            for _, r in group.iterrows():
                line = f"  - 《{r['电影名称']}》 ({r['上映日期']}): 总票房 {r['总票房(万)'] / 10000:.2f}亿, 首日排片 {r['首日排片(%)']}%"
                print(line)
                f.write(line + "\n")

    print(f"\n结果已保存至: {OUTPUT_CSV}")


if __name__ == '__main__':
    main()