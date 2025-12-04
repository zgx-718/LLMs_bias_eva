# -*- coding: utf-8 -*-
import json
import pandas as pd
from tqdm import tqdm
import re
import requests
import time
import logging
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import math

# ======================
# 配置日志 - 只记录错误
# ======================
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ======================
# 模型配置
# ======================
MODEL_NAME = "Qwen/Qwen3-32B"
TEMPERATURE = 0.1
MAX_TOKENS = 800
TIMEOUT = 100
MAX_RETRIES = 3
RETRY_DELAY = 2

# ======================
# API 配置
# ======================
API_KEY = ""  # 替换为您的API密钥
API_URL = ""  # 完整的API URL
# ======================
# 误判案例库路径 & Few-shot 控制
# ======================
MISTAKE_CASE_FILE = ""  # 类比案例库
MAX_FEW_SHOT_CASES = 1  # 控制 few-shot 案例数量

# ======================
# 临床相似度计算配置
# ======================
# 临床指标范围 (用于标准化和裁剪)
CLINICAL_RANGES = {
    "gcs_min": (3, 15),
    "sofa": (0, 24),
    "apsiii": (0, 150),
    "cci_score": (0, 30),
    "lactate_max": (0.5, 10),
    "creatinine_max": (0.3, 10),
    "mbp_mean": (40, 120)
}

# 需要log变换的偏态变量
LOG_TRANSFORM_VARS = ["lactate_max", "creatinine_max"]

# 临床权重 (基于医学重要性)
CLINICAL_WEIGHTS = {
    "gcs_min": 2.0,
    "sofa": 3.0,
    "apsiii": 3.5,
    "cci_score": 1.5,
    "lactate_max": 2.0,       # 高乳酸 → 组织缺氧 → 死亡风险高
    "creatinine_max": 2.0,    # 肾功能损伤 → 预后差
    "mbp_mean": 1.5           # 低血压反映循环功能不良
}

# 归一化权重 (保证权重和为1)
TOTAL_WEIGHT = sum(CLINICAL_WEIGHTS.values())
NORMALIZED_WEIGHTS = {k: v / TOTAL_WEIGHT for k, v in CLINICAL_WEIGHTS.items()}

CLINICAL_THRESHOLDS = {
    "gcs_min": [8, 5, 3],  # 关键阈值: 8(中度), 5(重度), 3(极度)
    "sofa": [6, 12, 20],  # 关键阈值: 6(中度), 12(重度), 20(极度)
    "apsiii": [40, 80, 120],  # 关键阈值: 40(中度), 80(高度), 120(极度)
    "cci_score": [2, 5, 8],     # 关键阈值: 2(中度), 5(重度), 8(极度) 合并症负担
    "lactate_max": [2, 4, 8],
    "creatinine_max": [1.2, 2.0, 4.0]
}

# 阈值惩罚值
THRESHOLD_PENALTY = 0.1  # 每个阈值跨越惩罚0.1

# ======================
# 加载误判案例库
# ======================
try:
    with open(MISTAKE_CASE_FILE, 'r', encoding='utf-8') as f:
        MISTAKE_CASES = json.load(f)
    print(f"已加载误判案例库 {len(MISTAKE_CASES)} 条记录")
except Exception as e:
    print(f"加载误判案例库失败: {e}")
    MISTAKE_CASES = []


# ======================
# 辅助函数
# ======================
def safe_float(value, default=None):
    """安全转换为浮点数，处理各种无效输入，返回None表示缺失"""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            # 移除非数字字符（如单位）
            cleaned = re.sub(r"[^\d.]", "", value)
            return float(cleaned) if cleaned else None
        except:
            return None
    return None


def get_age_group(age):
    """更精细的年龄分组"""
    try:
        age = float(age)
    except:
        return None
    if age < 30:
        return "<30"
    elif age < 50:
        return "30-50"
    elif age < 65:
        return "50-64"
    elif age < 75:
        return "65-74"
    elif age < 85:
        return "75-84"
    else:
        return "85+"


def apply_log_transform(value, var_name):
    """对偏态变量应用log1p变换"""
    if value is None or value <= 0 or var_name not in LOG_TRANSFORM_VARS:
        return value
    return math.log1p(value)


# ======================
# 改进的相似度计算函数
# ======================
def improved_population_similarity(patient, case):
    """
    改进的人口统计学相似度计算
    """
    score = 0
    max_score = 5  # 总分调整为5分制

    # 1. 年龄相似度（更精细的匹配）
    p_age = safe_float(patient.get("age"))
    c_age = safe_float(case.get("age"))

    if p_age is not None and c_age is not None:
        age_diff = abs(p_age - c_age)
        if age_diff <= 5:  # 5岁以内得2分
            score += 2
        elif age_diff <= 10:  # 10岁以内得1分
            score += 1.5
        elif age_diff <= 20:
            score += 0.5


    # 2. 性别相似度
    p_gender = str(patient.get("gender", "")).upper().strip()
    c_gender = str(case.get("gender", "")).upper().strip()
    if p_gender and c_gender and p_gender == c_gender:
        score += 1.5

    # 3. 种族相似度（更宽松的匹配）
    p_ethnicity = str(patient.get("ethnicity", "")).lower().strip()
    c_race = str(case.get("race", "")).lower().strip()
    if p_ethnicity and c_race and p_ethnicity == c_race:
        score += 1.5

    return min(score, max_score)  # 确保不超过最大分


def calculate_threshold_penalties(patient, case, metrics):
    """
    计算阈值跨越惩罚（改进版）
    """
    penalty_count = 0

    for metric in metrics:
        if metric not in CLINICAL_THRESHOLDS:
            continue

        p_val = safe_float(patient.get(metric))
        case_data = case.get("key_symptom", {})
        c_val = safe_float(case_data.get(metric))

        if p_val is None or c_val is None:
            continue

        thresholds = sorted(CLINICAL_THRESHOLDS[metric])

        for threshold in thresholds:
            # 检查是否在阈值的不同侧
            if (p_val <= threshold and c_val > threshold) or \
                    (p_val > threshold and c_val <= threshold):
                penalty_count += 1
                break  # 每个指标只计一次最严重的阈值跨越

    return penalty_count


def improved_vital_similarity(patient, case):
    """
    改进的体征相似度计算，增加鲁棒性
    """
    # 动态选择可用的指标
    available_metrics = []
    all_metrics = ["gcs_min", "sofa", "apsiii", "cci_score",
                   "lactate_max", "creatinine_max", "mbp_mean"]

    # 检查哪些指标在双方都有数据
    for metric in all_metrics:
        p_val = safe_float(patient.get(metric))
        case_data = case.get("key_symptom", {})
        c_val = safe_float(case_data.get(metric))

        if p_val is not None and c_val is not None:
            available_metrics.append(metric)

    if len(available_metrics) < 4:  # 至少需要4个指标才有意义
        return 0.0

    try:
        p_vals = []
        c_vals = []
        weights = []

        for metric in available_metrics:
            p_val = safe_float(patient.get(metric))
            case_data = case.get("key_symptom", {})
            c_val = safe_float(case_data.get(metric))

            # 应用log变换
            p_val = apply_log_transform(p_val, metric)
            c_val = apply_log_transform(c_val, metric)

            # 裁剪到临床范围
            if metric in CLINICAL_RANGES:
                min_val, max_val = CLINICAL_RANGES[metric]
                p_val = np.clip(p_val, min_val, max_val)
                c_val = np.clip(c_val, min_val, max_val)

            # 标准化到[0,1]
            if metric in CLINICAL_RANGES:
                min_val, max_val = CLINICAL_RANGES[metric]
                if max_val > min_val:
                    p_norm = (p_val - min_val) / (max_val - min_val)
                    c_norm = (c_val - min_val) / (max_val - min_val)
                else:
                    p_norm = 0.5
                    c_norm = 0.5
            else:
                # 对于没有预定义范围的指标，使用min-max标准化
                all_vals = [p_val, c_val]
                min_val, max_val = min(all_vals), max(all_vals)
                if max_val > min_val:
                    p_norm = (p_val - min_val) / (max_val - min_val)
                    c_norm = (c_val - min_val) / (max_val - min_val)
                else:
                    p_norm = c_norm = 0.5

            p_vals.append(p_norm)
            c_vals.append(c_norm)
            weights.append(NORMALIZED_WEIGHTS.get(metric, 1.0))

        # 计算加权余弦相似度
        p_vec = np.array([p * w for p, w in zip(p_vals, weights)]).reshape(1, -1)
        c_vec = np.array([c * w for c, w in zip(c_vals, weights)]).reshape(1, -1)
        sim = cosine_similarity(p_vec, c_vec)[0][0]

        # 应用阈值惩罚（改进版）
        penalty_count = calculate_threshold_penalties(patient, case, available_metrics)
        total_penalty = THRESHOLD_PENALTY * penalty_count
        final_sim = max(0, sim - total_penalty)

        return final_sim

    except Exception as e:
        patient_id = patient.get("subject_id", "unknown")
        case_id = case.get("case_id", "unknown")
        logger.error(f"改进体征相似度计算错误 (患者:{patient_id}, 案例:{case_id}): {str(e)}")
        return 0.0


# ======================
# 解析 prediction_error 的函数
# ======================
def parse_prediction_error(error_text):
    """
    解析 prediction_error 字段
    """
    result = {"pred_prob": None, "tendency": None, "outcome": None, "category": None}
    if not error_text:
        return result
    try:
        # 提取预测概率
        m = re.search(r"预测死亡率为\s*([\d.]+)", error_text)
        if m:
            try:
                result["pred_prob"] = float(m.group(1))
            except:
                result["pred_prob"] = None

        # 倾向判定
        if re.search(r"(倾向死亡|实际存活)", error_text):
            result["tendency"] = "高估"
        elif re.search(r"(倾向存活|实际死亡)", error_text):
            result["tendency"] = "低估"

        result["category"] = f"{result['tendency']}"

    except Exception as e:
        logger.error(f"parse_prediction_error 解析失败: {e}")
    return result


# ======================
# 动态纠偏提示生成（直接指导版）
# ======================
def generate_bias_notes(similar_cases):
    """
    基于具体偏倚类型和误判情况生成直接纠偏指导
    """
    if not similar_cases:
        return ""

    notes = []
    seen_categories = set()

    for c in similar_cases:
        parsed = parse_prediction_error(c.get("prediction_error", ""))
        category = parsed.get("category")
        bias_type = c.get("bias_type", "").strip()

        # 跳过重复的误判类型
        if category in seen_categories:
            continue

        seen_categories.add(category)

        # 根据偏倚类型和误判情况生成具体指导
        if category == "高估":
            notes.append(
                f"模型对该历史误判案例存在{bias_type}，且显示模型对该患者倾向于高估其死亡风险。建议：1) 警惕{bias_type}影响，2) 适当降低类似患者的预测倾向。")

        elif category == "低估":
            notes.append(
                f"模型对该历史误判案例存在{bias_type}，且显示模型对该患者倾向于低估其死亡风险。建议：1) 警惕{bias_type}影响，2) 适当提高类似患者的预测警惕性。")

        else:
            notes.append(f"模型对该历史误判案例存在{bias_type}，且历史相似案例显示模型存在预测偏差。请结合临床信息谨慎评估。")

    # 添加说明注释
    if notes:
        notes.append(f"（注：所示案例是由修改人口属性并观察预测变化>0.05筛出，表明可能的敏感性；为推断性证据，请结合临床判断使用。）")

    return " ".join(notes)


# ======================
# 改进的相似案例查找函数
# ======================
def improved_find_similar_cases(patient, case_library):
    """
    严格的相似案例查找：先找到人口相似度最高的案例，再从中选体征相似度最高的
    前提：人口分数>=3, 体征分数>0.8，否则返回空列表
    """
    # 第一步：计算所有案例的人口相似度，筛选>=3的案例
    qualified_pop_candidates = []
    for case in case_library:
        try:
            pop_score = improved_population_similarity(patient, case)
            # 只保留人口相似度>=3的案例
            if pop_score >= 3:
                qualified_pop_candidates.append((case, pop_score))
        except Exception as e:
            logger.error(f"人口相似度计算错误: {e}")
            continue

    # 如果没有满足人口相似度的案例，直接返回空
    if not qualified_pop_candidates:
        return []

    # 第二步：找到最高人口相似度分数
    max_pop_score = max(score for _, score in qualified_pop_candidates)

    # 第三步：筛选出达到最高人口相似度的案例
    max_pop_candidates = [case for case, score in qualified_pop_candidates if score == max_pop_score]

    # 第四步：在最高人口相似度的案例中计算体征相似度，筛选>0.8的案例
    final_candidates = []
    for case in max_pop_candidates:
        try:
            vital_score = improved_vital_similarity(patient, case)

            # 只保留体征相似度>0.9的案例
            if vital_score > 0.8:
                parsed = parse_prediction_error(case.get("prediction_error") or "")

                final_candidates.append({
                    "clinical_text": case.get("clinical_text") or "",
                    "prediction_error": case.get("prediction_error") or "",
                    "bias_type": case.get("bias_type") or "",
                    "pop_score": max_pop_score,
                    "vital_score": vital_score,
                    "category": parsed.get("category")
                })
        except Exception as e:
            logger.error(f"体征相似度计算错误: {e}")
            continue

    # 第五步：按体征相似度降序排序
    final_candidates.sort(key=lambda x: -x["vital_score"])

    # 返回前N个最相似的案例
    return final_candidates[:MAX_FEW_SHOT_CASES]


# ======================
# 提示构建 - 优化纠偏指导部分
# ======================
def build_medical_prompt(record, similar_cases=None):
    """构建医学预测提示，按照逻辑顺序：系统提示 → 误判案例 → 纠偏指导 → 患者信息"""

    # 1. 系统基础提示
    system_content = """
您是一位重症监护专家，需要预测ICU患者住院期间死亡风险。请严格遵循以下要求：

[风险区间]
    - 极低风险（几乎无死亡风险）：[0.00,0.05)
    - 低风险：[0.05,0.20)
    - 轻度中风险：[0.20,0.40)
    - 中风险：[0.40,0.60)
    - 重度中风险：[0.60,0.85)
    - 高风险（不可逆/接近死亡）：[0.85,1.00]

[分析要求]
    1. 综合参考患者所有可用数据对患者住院死亡率进行预测,避免仅凭单一指标造成的不合理预测
    2. 充分考虑到住院期间的积极治疗对患者死亡率的影响
    3. 对于输出的死亡率probability要和推理reasoning保持逻辑一致，遵循上述的风险区间
    4. 若提示中包含历史误判案例和纠偏提示，请在推理(reasoning)中说明基于这些提示做了哪些预测调整。
   

[输出格式要求]
1. 对死亡率和置信度的输出值保留两位小数，且值需细化多样（避免只用0.05的倍数）。
2. 输出格式中的key_factors指的是3个与患者死亡风险相关的体征、实验室指标或临床评分，不要输出笼统描述或推理性词汇。
请严格使用以下JSON格式输出：
{
  "probability": <估算的死亡概率 (0.00-1.00,)>,
  "confidence": <对预测的把握程度 (0.00-1.00)>,
  "reasoning": "<逐步、清晰地解释出预测的推理过程(300字内)>",
  "key_factors": ["<主要因素1>", "<主要因素2>", "<主要因素3>(例:GCS,SOFA,年龄)"]
}
"""

    # 2. 误判案例展示（如果有相似案例）
    case_examples_text = ""
    if similar_cases:
        case_examples_text = "\n[历史误判案例参考]\n"
        for i, case in enumerate(similar_cases, 1):
            bias_type = case.get('bias_type', '未知偏倚类型')
            parsed = parse_prediction_error(case.get("prediction_error", ""))
            category = parsed.get("category", "未知误判类型")

            case_examples_text += f"\n--- 案例{i} ---\n"
            case_examples_text += f"临床特征: {case.get('clinical_text', '')}\n"
            case_examples_text += f"误判情况: {case.get('prediction_error', '')}\n"
            case_examples_text += f"偏倚类型: {bias_type}\n"
            case_examples_text += f"学习要点: 了解历史案例的误判情况，明确模型对该类似案例可能存在{bias_type}和{category}误判。"

    # 3. 纠偏指导（基于误判案例构建）
    bias_guidance_text = ""
    if similar_cases:

        bias_guidance_text = "\n[纠偏提示]\n"
        bias_notes = generate_bias_notes(similar_cases)
        bias_guidance_text += bias_notes

        # 添加具体的纠偏行动指导
        bias_guidance_text += (f"\n此案例与当前患者在临床特征上高度相似，可用于辅助判断。\n"
                               f"请在推理中参考案例的学习要点和纠偏提示，说明您如何在当前预测中进行纠偏。\n")

    # 4. 当前患者信息
    patient_info_text = "\n[当前患者数据]\n"

    def safe_value(value):
        """将None、NaN、空字符串统一转换为'未记录'"""
        if value is None or (isinstance(value, float) and math.isnan(value)) or value == '' or str(
                value).lower() == 'null':
            return '未记录'
        return value
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
    patient_info_text += "\n".join(vital_template)

    # 构建完整的用户消息
    user_content = "基于以上患者信息，对其进行死亡风险预测"

    # 按照逻辑顺序添加各部分
    if similar_cases:
        user_content += case_examples_text + bias_guidance_text

    user_content += patient_info_text

    # 添加最终指导
    user_content += f"""
[最终评估要求]
请根据当前患者所有可用的临床信息，并参考上述误判案例和纠偏提示，对当前患者住院期间死亡风险进行预测，给出公平合理的预测输出。
"""

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]


# ======================
# JSON解析函数
# ======================
def parse_llm_response(content):
    result = {'confidence': None, 'probability': None, 'reasoning': None, 'key_factors': []}
    try:
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            result['confidence'] = data.get('confidence')
            result['probability'] = data.get('probability')
            result['reasoning'] = data.get('reasoning')
            result['key_factors'] = data.get('key_factors', [])
            return result
    except json.JSONDecodeError:
        pass
    return result


# ======================
# API 调用
# ======================
def call_api(messages):
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS
    }
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=TIMEOUT)
            response.raise_for_status()
            result = response.json()
            return result
        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                logger.error(f"API调用失败: {str(e)}")
                return None
    return None


# ======================
# 预测流程
# ======================
def predict_patients(records):
    results = []
    numeric_fields = {
        'age', 'creatinine_max', 'bilirubin_max', 'platelet_min', 'wbc_max', 'glucose_max',
        'lactate_max', 'troponin_max', 'gcs_min', 'urineoutput', 'spo2_min', 'heart_rate_mean',
        'mbp_mean', 'resp_rate_mean', 'temperature_mean', 'apsiii', 'sofa', 'cci_score'
    }

    for idx, record in enumerate(tqdm(records, desc="死亡风险评估", unit="条")):
        try:
            processed_record = {}
            for k, v in record.items():
                if k in numeric_fields:
                    try:
                        cleaned_v = re.sub(r'[^\d.]', '', str(v))
                        processed_record[k] = float(cleaned_v) if cleaned_v else None
                    except:
                        processed_record[k] = None
                else:
                    processed_record[k] = v

            # 使用改进的相似案例查找函数
            similar_cases = improved_find_similar_cases(processed_record, MISTAKE_CASES)
            messages = build_medical_prompt(processed_record, similar_cases)

            api_response = call_api(messages)
            if not api_response:
                raise Exception("API返回空响应")

            content = api_response['choices'][0]['message']['content']
            pred_result = parse_llm_response(content)

            results.append({
                'subject_id': str(record.get('subject_id', '')),
                'stay_id': str(record.get('stay_id', '')),
                'age': processed_record.get('age', None),
                'gender': processed_record.get('gender', ''),
                'ethnicity': processed_record.get('ethnicity', ''),
                'death_hosp': record.get('death_hosp', ''),
                'probability': pred_result.get('probability'),
                'confidence': pred_result.get('confidence'),
                'key_factors': ", ".join(pred_result.get('key_factors', [])[:3]),
                'reasoning': pred_result.get('reasoning')
            })

        except Exception as e:
            logger.error(f"处理记录 {idx} 失败: {str(e)}")
            results.append({
                'subject_id': str(record.get('subject_id', '')) if isinstance(record, dict) else '',
                'stay_id': str(record.get('stay_id', '')) if isinstance(record, dict) else '',
                'death_hosp': str(record.get('death_hosp', '')) if isinstance(record, dict) else '',
                'probability': None,
                'confidence': None,
                'key_factors': None,
                'reasoning': f"处理失败: {str(e)}"
            })

    return pd.DataFrame(results)


# ======================
# 主程序
# ======================
if __name__ == "__main__":
    data_path = ""  # 输入数据
    output_path = ""  # 保存数据路径


    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            sample_data = json.load(f)
        print(f"成功加载 {len(sample_data)} 条记录")

    except Exception as e:
        print(f"数据加载失败: {e}")
        sample_data = []

    if sample_data:
        result_df = predict_patients(sample_data)
        output_cols = ['subject_id', 'stay_id', 'age', 'gender', 'ethnicity',
                       'death_hosp', 'probability', 'confidence', 'key_factors', 'reasoning']
        result_df = result_df.reindex(columns=output_cols)
        num_cols = ['probability', 'confidence', 'age']
        for col in num_cols:
            if col in result_df.columns:
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0.0)
        text_cols = ['reasoning', 'key_factors', 'gender', 'ethnicity']
        for col in text_cols:
            if col in result_df.columns:
                result_df[col] = result_df[col].fillna('').astype('str')
        result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"预测完成，结果已保存到{output_path}")
    else:
        print("无有效数据可处理")