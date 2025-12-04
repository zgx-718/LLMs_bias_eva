import pandas as pd


def compare_and_save_records(file1, file2, output_file, threshold=0.1):
    """
    比较两个CSV文件，找出probability差值超过阈值的记录，
    并按121212顺序保存file1原始记录和file2（gender反转）记录

    参数:
        file1 (str): 第一个CSV文件路径
        file2 (str): 第二个CSV文件路径
        output_file (str): 输出CSV文件路径
        threshold (float): probability差值阈值，默认为0.05
    """
    # 读取两个CSV文件
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # 在原始数据中添加来源标记
    df1['source'] = 'file1'
    df2['source'] = 'file2'

    # 合并两个DataFrame，基于subject_id, stay_id, death_hosp
    merged = pd.merge(
        df1,
        df2,
        on=['subject_id', 'stay_id', 'death_hosp'],
        suffixes=('_file1', '_file2')
    )

    # 计算probability的绝对差值
    merged['probability_diff'] = abs(merged['probability_file1'] - merged['probability_file2'])

    # 筛选出差值大于阈值的记录
    diff_records = merged[merged['probability_diff'] > threshold]

    print(f"找到 {len(diff_records)} 条记录的probability差值超过 {threshold}")

    if len(diff_records) == 0:
        print("没有满足条件的记录，不生成输出文件")
        return

    # 准备结果DataFrame
    result_list = []

    # 获取两个文件的原始列名（去除后缀）
    cols_file1 = [col.replace('_file1', '') for col in df1.columns]
    cols_file2 = [col.replace('_file2', '') for col in df2.columns]

    # 处理每条差异记录
    for _, row in diff_records.iterrows():
        # 获取file1的原始记录
        record_file1 = df1[
            (df1['subject_id'] == row['subject_id']) &
            (df1['stay_id'] == row['stay_id']) &
            (df1['death_hosp'] == row['death_hosp'])
            ].iloc[0]

        # 获取file2的原始记录并反转ethnicity
        record_file2 = df2[
            (df2['subject_id'] == row['subject_id']) &
            (df2['stay_id'] == row['stay_id']) &
            (df2['death_hosp'] == row['death_hosp'])

            ].iloc[0].copy()

        # 反转age
        if 'age' in record_file2:
            if record_file2['age'] < 65:  # young_old
                record_file2['age'] = record_file2['age'] + 15
            elif record_file2['age'] >= 65:  # old_old
                record_file2['age'] = record_file2['age'] - 15



        # 反转ethnicity
        # if 'ethnicity' in record_file2:
        #     if record_file2['ethnicity'] == 'white':
        #         record_file2['ethnicity'] = 'black'
        #     elif record_file2['ethnicity'] == 'black':
        #         record_file2['ethnicity'] = 'white'
        #
        #     elif record_file2['ethnicity'] == 'asian':
        #         record_file2['ethnicity'] = 'white'
        #     elif record_file2['ethnicity'] == 'other':
        #         record_file2['ethnicity'] = 'white'


        # 反转gender
        # if 'gender' in record_file2:
        #     if record_file2['gender'] == 'M':
        #         record_file2['gender'] = 'F'
        #     elif record_file2['gender'] == 'F':
        #         record_file2['gender'] = 'M'



        # 添加到结果列表（file1在前，file2在后）
        result_list.append(record_file1)
        result_list.append(record_file2)

    # 创建结果DataFrame
    result_df = pd.DataFrame(result_list)

    # 添加标记列显示记录来源
    # result_df['record_source'] = ['file1', 'file2'] * len(diff_records)

    # 保存到CSV文件
    result_df.to_csv(output_file, index=False)
    print(f"结果已保存到 {output_file}，共 {len(result_df)} 条记录（{len(diff_records)} 对）")




if __name__ == "__main__":
    # 示例用法
    file1 = ""  # 替换为你的第一个CSV文件路径
    file2 = ""  # 替换为你的第二个CSV文件路径
    output_file = ""  # 输出文件路径

    compare_and_save_records(file1, file2, output_file)


