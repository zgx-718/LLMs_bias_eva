import pandas as pd
import numpy as np
import json


# 1. 读取CSV文件
def read_csv(file_path):
    return pd.read_csv(file_path)


# 2. 数据预处理
def preprocess_data(df):
    categorical_columns = [
        'gender', 'ethnicity', 'admission_type', 'electivesurgery',
        'vent', 'code_status', 'death_hosp'
    ]
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


# 3. 缺失情况统计
def report_missing(df, columns):
    print("\n=== 缺失比例报告 ===")
    for col in columns:
        missing_rate = df[col].isna().mean()
        print(f"{col:<20} 缺失比例: {missing_rate:.2%}")
    print("===================\n")


# 4. 生成 JSON 文件（保留 NaN -> null）
def generate_json(df, output_path):
    json_data = []
    for _, row in df.iterrows():
        def safe_value(x):
            # 保留 NaN 为 None（JSON中显示为 null）
            if pd.isna(x):
                return None
            try:
                # 尝试转换为 float 或 int
                val = float(x)
                if val.is_integer():
                    return int(val)
                return round(val, 1)
            except (ValueError, TypeError):
                return str(x)

        patient_data = {
            "subject_id": safe_value(row.get('subject_id')),
            "stay_id": safe_value(row.get('stay_id')),
            "first_careunit": row.get('first_careunit'),
            "gender": row.get('gender'),
            "age": safe_value(row.get('age')),
            "ethnicity": row.get('ethnicity'),
            "admission_type": row.get('admission_type'),
            "electivesurgery": row.get('electivesurgery'),
            "vent": row.get('vent'),
            "creatinine_max": safe_value(row.get('creatinine_max')),
            "bilirubin_max": safe_value(row.get('bilirubin_max')),
            "platelet_min": safe_value(row.get('platelet_min')),
            "wbc_max": safe_value(row.get('wbc_max')),
            "glucose_max": safe_value(row.get('glucose_max')),
            "lactate_max": safe_value(row.get('lactate_max')),
            "troponin_max": safe_value(row.get('troponin_max')),
            "gcs_min": safe_value(row.get('gcs_min')),
            "urineoutput": safe_value(row.get('urineoutput')),
            "spo2_min": safe_value(row.get('spo2_min')),
            "heart_rate_mean": safe_value(row.get('heart_rate_mean')),
            "mbp_mean": safe_value(row.get('mbp_mean')),
            "resp_rate_mean": safe_value(row.get('resp_rate_mean')),
            "temperature_mean": safe_value(row.get('temperature_mean')),
            "norepinephrine": row.get('norepinephrine'),
            "epinephrine": row.get('epinephrine'),
            "apsiii": safe_value(row.get('apsiii')),
            "sofa": safe_value(row.get('sofa')),
            "cci_score": safe_value(row.get('cci_score')),
            "code_status": row.get('code_status'),
            "death_hosp": row.get('death_hosp')
        }
        json_data.append(patient_data)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)
    print(f"✅ JSON 文件已生成：{output_path}")


# 5. 主函数
def main():
    input_csv = '你的数据路径'
    output_json = '数据插补后保存路径'

    df = read_csv(input_csv)
    df_processed = preprocess_data(df)

    # 可选：仅用于查看缺失比例
    missing_columns = [
        'creatinine_max', 'bilirubin_max', 'platelet_min', 'wbc_max',
        'glucose_max', "lactate_max", 'troponin_max', 'spo2_min',
        'heart_rate_mean', 'mbp_mean', 'resp_rate_mean', 'temperature_mean'
    ]
    report_missing(df_processed, missing_columns)

    # 不进行任何插补，直接输出
    generate_json(df_processed, output_json)


if __name__ == "__main__":
    main()
