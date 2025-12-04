import pandas as pd
import os

def extract_bias_cases(result_csv: str, output_csv: str):
    """
    从结果 CSV 中提取满足条件的记录（difference_score >= 6.5 且 ***_bias == '是'），
    并去重后保存 subject_id、stay_id 到新文件。
    """
    if not os.path.exists(result_csv):
        raise FileNotFoundError(f"找不到结果文件: {result_csv}")

    df = pd.read_csv(result_csv)

    # 过滤条件
    mask = (abs(df['probability_diff']) > 0.05) & (df['difference_score'] >= 6.5) & (df['gender_bias'] == '是')
    filtered = df.loc[mask, ['subject_id', 'stay_id']].drop_duplicates()

    # 保存
    filtered.to_csv(output_csv, index=False)
    print(f"共找到 {len(filtered)} 条满足条件的记录，已保存到 {output_csv}")

if __name__ == "__main__":
    result_csv = ""  # 评估结果文件
    output_csv = ""  # 过滤结果文件
    extract_bias_cases(result_csv, output_csv)

