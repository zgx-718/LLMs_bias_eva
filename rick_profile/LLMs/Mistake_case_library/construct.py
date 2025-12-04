# -*- coding: utf-8 -*-
import math

import pandas as pd
import json


def extract_case_info(csv_path: str, json_path: str, output_path: str):
    # 1. 读取 CSV
    csv_df = pd.read_csv(csv_path)

    # 2. 读取 JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)  # 假设是一个列表，每个元素是一个病例字典

    # 3. 将 JSON 数据按 subject_id 建立索引
    json_index = {str(item.get("subject_id", "")): item for item in json_data}

    # 4. 定义关键症状字段
    key_symptoms = ["gcs_min", "lactate_max", "creatinine_max", "mbp_mean",
                    "sofa", "apsiii", "cci_score"]

    results = []

    # 5. 遍历 CSV 记录
    for _, row in csv_df.iterrows():
        # 确保subject_id为字符串类型以匹配索引
        sid = str(row.get("subject_id", ""))
        if sid in json_index and sid:  # 确保sid非空
            record = json_index[sid]

            # 预测死亡率
            prob = row.get("probability", None)
            if prob is not None:
                try:
                    prob = round(float(prob), 3)  # 保留三位小数
                except (ValueError, TypeError):
                    prob = None

            # 实际结局
            # 实际结局
            death_flag = row.get("death_hosp", None)
            if death_flag == 1:
                outcome_text = "实际死亡"
            elif death_flag == 0:
                outcome_text = "实际存活"
            else:
                outcome_text = "未知"

            # 倾向判断
            if prob is not None:
                if prob < 0.2:
                    tendency_text = "预测为低死亡率（倾向存活）"
                elif 0.2 <= prob < 0.4:
                    tendency_text = "预测为轻度中等死亡率"
                else:  # prob >= 0.4
                    tendency_text = "预测为中/高死亡率（倾向死亡）"

                prediction_error_text = f"预测死亡率为{prob:.2f}，{tendency_text}，但{outcome_text}"
            else:
                prediction_error_text = f"无预测数据，{outcome_text}"

            def safe_value(value):
                """将None、NaN、空字符串统一转换为'未记录'"""
                if value is None or (isinstance(value, float) and math.isnan(value)) or value == '' or str(
                        value).lower() == 'null':
                    return '未记录'
                return value
            # 构建临床文本
            clinical_text = (
                f"患者，年龄{record.get('age', '无记录')}岁，性别{'男' if record.get('gender') == 'M' else '女' if record.get('gender') == 'F' else '未知'}，"
                f"种族{record.get('ethnicity', '无记录')}。"
                f"入ICU时GCS评分为{safe_value(record.get('gcs_min'))}，反映意识状态。"
                f"生命体征显示最高乳酸水平：{safe_value(record.get('lactate_max'))} mmol/L，"
                f"肌钙蛋白最大值: {safe_value(record.get('troponin_max'))} ng/mL，"
                f"肌酐最大值: {safe_value(record.get('creatinine_max'))} mg/dL，"
                f"最低血氧饱和度：{safe_value(record.get('spo2_min'))}%。"
                f"入院后首个24小时尿量: {safe_value(record.get('urineoutput'))} mL。"
                f"平均动脉压：{safe_value(record.get('mbp_mean'))} mmHg，"
                f"呼吸频率平均：{safe_value(record.get('resp_rate_mean'))}次/分，心率平均：{safe_value(record.get('heart_rate_mean'))} bpm。"
                f"SOFA评分为{safe_value(record.get('sofa'))}，APACHE III评分为{safe_value(record.get('apsiii'))}。"
                f"CCI评分: {safe_value(record.get('cci_score'))}。"
                f"是否使用呼吸机:{'是' if record.get('vent') == 1 else '否'}，"
                f"复苏指令状态: {'允许复苏' if record.get('code_status', 0) != 0 else '不允许复苏'}"
            )

            # 构造输出字典
            case_info = {
                "subject_id": record.get("subject_id"),
                "age": str(record.get("age", "无记录")),
                "gender": record.get("gender"),
                "ethnicity": record.get("ethnicity"),
                "death_hosp": record.get("death_hosp"),
                "key_symptom": {symptom: record.get(symptom, "无记录") for symptom in key_symptoms},
                "clinical_text": clinical_text,
                "prediction_error": prediction_error_text,
                "bias_type": "xx偏倚"
            }
            results.append(case_info)

    # 6. 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"匹配完成，共找到 {len(results)} 条记录，已保存到 {output_path}")


if __name__ == "__main__":
    extract_case_info(
        csv_path="",  # 输入对应偏倚csv文件
        json_path="",  # 原始数据集
        output_path=""  # 构建对应偏倚案例文件
    )



