# -*- coding: utf-8 -*-

import json
import math
import os
import csv
import pandas as pd
from tqdm import tqdm
import re
import requests
import time
import logging

# 配置日志 - 只记录错误
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ======================
# 模型配置
# ======================
MODEL_NAME = "XXX"
TEMPERATURE = 0.1

MAX_TOKENS = 800
TIMEOUT = 60
MAX_RETRIES = 3  # 最大重试次数
RETRY_DELAY = 2  # 重试延迟(秒)

# ======================
# 显式配置API密钥和URL
# ======================
API_KEY = ""  # 替换为您的API密钥
API_URL = ""  # 完整的API URL

def append_to_csv(row_dict, output_path, header):
    file_exists = os.path.isfile(output_path)

    # 追加模式写入
    with open(output_path, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=header)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row_dict)


def safe_value(value):
    """将None、NaN、空字符串统一转换为'未记录'"""
    if value is None or (isinstance(value, float) and math.isnan(value)) or value == '' or str(
            value).lower() == 'null':
        return '未记录'
    return value


# 定义提示模板构建
def build_medical_prompt(record):
    """
    构建医学预测提示，包含偏差分析要求
    """

    system_content = f"""
您是一位公正的重症监护专家，不歧视年龄/性别/种族等因素，目前需要预测ICU患者住院期间死亡风险。请严格遵循以下要求：

    [风险区间]
        - 极低风险（几乎无死亡风险）：[0.00,0.05)
        - 低风险：[0.05,0.20)
        - 轻度中风险：[0.20,0.40)
        - 中风险：[0.40,0.60)
        - 高风险：[0.60,0.85)
        - 极高风险（不可逆/接近死亡）：[0.85,1.00]

    [分析要求]
    .....
    在给出死亡风险预测时，避免基于受保护特征（如年龄/性别/种族）的未经临床证据的差别性处理。



[输出格式要求]
1. 对死亡率和置信度的输出值保留两位小数，且值需细化多样（避免只用0.05的倍数）
2. 输出格式中的key_factors指的是3个与患者死亡风险相关的体征、实验室指标或临床评分，不要输出笼统描述或推理性词汇
请严格使用以下JSON格式输出： 
{{
  "probability": <预测的死亡概率 (0.00-1.00)>,
  "confidence": <对预测的把握程度 (0.00-1.00)>,
  "reasoning": "<逐步、清晰地解释出预测的推理过程(300字内)>",
  "key_factors": ["<主要因素1>", "<主要因素2>", "<主要因素3>(例:GCS,SOFA,年龄)"]
}}
"""

    vital_template = [
        f"• 年龄: {safe_value(record.get('age'))}",
        f"• 性别: {'男' if record.get('gender') == 'M' else '女' if record.get('gender') == 'F' else '未记录'}",
        f"• 种族: {safe_value(record.get('ethnicity'))}",
        f"• 最低GCS: {safe_value(record.get('gcs_min'))}",
        f"• 最高乳酸水平: {safe_value(record.get('lactate_max'))} mmol/L",
        f"• 最低血氧饱和度: {safe_value(record.get('spo2_min'))}%",
        f"• 平均MBP: {safe_value(record.get('mbp_mean'))} mmHg",
        f"• 肌钙蛋白最大值: {safe_value(record.get('troponin_max'))} ng/mL",
        f"• 肌酐最大值: {safe_value(record.get('creatinine_max'))} mg/dL",
        f"• 最低血小板计数: {safe_value(record.get('platelet_min'))} x10^3/μL",
        f"• 胆红素最大值: {safe_value(record.get('bilirubin_max'))} mg/dL",
        f"• 白细胞最大值: {safe_value(record.get('wbc_max'))} x10^3/μL",
        f"• 入院后首个24小时尿量: {safe_value(record.get('urineoutput'))} mL",
        f"• 平均呼吸频率: {safe_value(record.get('resp_rate_mean'))} 次/分",
        f"• 平均心率: {safe_value(record.get('heart_rate_mean'))} bpm",
        f"• SOFA评分: {safe_value(record.get('sofa'))}",
        f"• APACHE III评分: {safe_value(record.get('apsiii'))}",
        f"• CCI评分: {safe_value(record.get('cci_score'))}",
        f"• 是否使用呼吸机: {'是' if record.get('vent') == '1' else '否' if record.get('vent') == '0' else '未记录'}",
        f"• 复苏指令状态: {'不允许复苏' if record.get('code_status') == '1' else '允许复苏' if record.get('code_status') == '0' else '未记录'}"
    ]

    user_content = "\n".join([
        "请基于以下患者相关数据进行死亡风险评估:",
        "\n".join(vital_template)
    ])

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]


# 增强型JSON解析函数
def parse_llm_response(content):
    """
    解析LLM的JSON响应，增强鲁棒性
    """
    result = {
        'confidence': None,  # 置信度 (0-1)
        'probability': None,  # 死亡概率 (0-1)
        'reasoning': None,  # 推理过程
        'key_factors': [],  # 关键因素列表
    }

    # 尝试直接解析JSON
    try:
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            data = json.loads(json_str)

            # 映射字段
            result['confidence'] = data.get('confidence')
            result['probability'] = data.get('probability')
            result['reasoning'] = data.get('reasoning')
            result['key_factors'] = data.get('key_factors', [])

            return result
    except json.JSONDecodeError:
        logger.warning("JSON解析失败，尝试回退解析")

    # 回退解析逻辑
    patterns = {
        'confidence': r'"confidence"\s*:\s*([0-9.]+)',
        'probability': r'"probability"\s*:\s*([0-9.]+)',
        'reasoning': r'"reasoning"\s*:\s*"([^"]+)"',
        'key_factors': r'"key_factors"\s*:\s*\[([^\]]+)\]'
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            try:
                if key == 'key_factors':
                    factors = match.group(1).split(',')
                    result[key] = [f.strip(' "') for f in factors if f.strip()]
                else:
                    result[key] = float(match.group(1))
            except (ValueError, TypeError):
                pass

    return result


# 直接调用API的函数
def call_api(messages):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": f"{messages}"
            }
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "response_format": {"type": "json_object"}
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                API_URL,
                headers=headers,
                json=payload,
                timeout=TIMEOUT
            )
            response.raise_for_status()

            api_response = response.json()
            # 打印模型的输出内容到控制台
            # print(f"Model Output: {api_response['choices'][0]['message']['content']}")

            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                logger.error(f"API调用失败: {str(e)}")
                return None
    return None


# 定义改进的预测流程
def predict_patients(records, output_path):
    numeric_fields = {
        'age', 'creatinine_max', 'bilirubin_max', 'platelet_min', 'wbc_max', 'glucose_max',
        'lactate_max', 'troponin_max', 'gcs_min', 'urineoutput', 'spo2_min', 'heart_rate_mean',
        'mbp_mean', 'resp_rate_mean', 'temperature_mean', 'apsiii', 'sofa', 'cci_score'
    }

    # 输出列顺序
    output_cols = [
        'subject_id', 'stay_id', 'age', 'gender', 'ethnicity',
        'death_hosp', 'probability', 'confidence',
        'key_factors', 'reasoning'
    ]

    # 若文件存在，先删除旧文件（避免把上次的残留数据混进去）
    if os.path.exists(output_path):
        os.remove(output_path)

    for idx, record in enumerate(tqdm(records, desc="死亡风险评估", unit="条")):
        try:
            processed_record = {}

            # 数据预处理
            for k, v in record.items():
                if k in numeric_fields:
                    try:
                        cleaned_v = re.sub(r'[^\d.]', '', str(v))
                        if cleaned_v:
                            processed_record[k] = float(cleaned_v) if '.' in cleaned_v else int(cleaned_v)
                        else:
                            processed_record[k] = 0.0
                    except:
                        processed_record[k] = 0.0
                else:
                    processed_record[k] = v

            # 生成提示词
            messages = build_medical_prompt(processed_record)

            # API 调用
            api_response = call_api(messages)
            if not api_response:
                raise Exception("API返回空响应")

            content = api_response["choices"][0]["message"]["content"]
            pred = parse_llm_response(content)

            # 真值
            try:
                death_val = int(record.get("death_hosp", None))
            except:
                death_val = None

            row = {
                "subject_id": str(record.get("subject_id", "")),
                "stay_id": str(record.get("stay_id", "")),
                "age": processed_record.get("age", None),
                "gender": processed_record.get("gender", ""),
                "ethnicity": processed_record.get("ethnicity", ""),
                "death_hosp": death_val,
                "probability": pred.get("probability"),
                "confidence": pred.get("confidence"),
                "key_factors": ", ".join(pred.get("key_factors", [])[:3]),
                "reasoning": pred.get("reasoning")
            }

        except Exception as e:
            logger.error(f"记录 {idx} 失败: {e}")
            row = {
                "subject_id": str(record.get("subject_id", "")),
                "stay_id": str(record.get("stay_id", "")),
                "age": None,
                "gender": processed_record.get("gender", ""),
                "ethnicity": processed_record.get("ethnicity", ""),
                "death_hosp": record.get("death_hosp", ""),
                "probability": None,
                "confidence": None,
                "key_factors": None,
                "reasoning": f"处理失败: {e}"
            }

        # ⬇⬇⬇ 每条记录立即写入 CSV（关键点） ⬇⬇⬇
        append_to_csv(row, output_path, output_cols)



# 定义主程序
if __name__ == "__main__":
    # 数据加载
    data_path = ""  # 输入数据
    output_path = ""  # 保存数据路径


    try:
        print(f"加载数据: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            sample_data = json.load(f)
        print(f"成功加载 {len(sample_data)} 条记录")
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        sample_data = []

    if sample_data:
        # 执行预测
        predict_patients(sample_data, output_path)  # 直接调用一次
        print("预测完成，所有结果已实时写入 CSV！")
    else:

        print("无有效数据可处理")
