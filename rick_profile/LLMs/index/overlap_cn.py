# -*- coding: utf-8 -*-
"""
Subgroup Bias Analysis
(1) Top-3 weighted Jaccard
(2) All-feature weighted Jaccard + JS Divergence + Spearman
"""
import pandas as pd
import numpy as np
import ast
import re
import os
from itertools import combinations
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr
from numpy.linalg import norm

# ------------------------- 中文特征 → 标准变量名映射 -------------------------
MAPPING_DICT = {
    "年龄": "age", "高龄": "age", "血氧饱和度": "spo2_min", "APACHE III评分": "apsiii",
    "GCS": "gcs_min", "意识障碍": "gcs_min", "SOFA": "sofa", "CCI": "cci_score", "乳酸": "lactate_max",
    "尿量": "urineoutput", "肌酐": "creatinine_max", "肌钙蛋白": "troponin_max",
    "血小板": "platelet_min", "胆红素": "bilirubin_max", "白细胞": "wbc_max", "平均动脉压": "mbp_mean",
    "MBP": "mbp_mean", "机械通气": "vent", "复苏指令": "code_status"
}

VALID_FEATURES = set(MAPPING_DICT.values())

# ------------------------- 工具函数 -------------------------
def normalize_key_factors(factor_str, mapping_dict=MAPPING_DICT):
    if isinstance(factor_str, list):
        factors = factor_str
    elif isinstance(factor_str, str) and factor_str.startswith("["):
        try:
            factors = ast.literal_eval(factor_str)
        except Exception:
            factors = [factor_str]
    else:
        factors = re.split(r"[，,]", str(factor_str))

    normalized = []
    for f in factors:
        f_clean = f.strip()
        mapped = None
        for k, v in mapping_dict.items():
            if k in f_clean:
                mapped = v
                break
        if f_clean:
            normalized.append(mapped if mapped else f_clean)
    return normalized


def get_feature_counts(factor_lists):
    counts = {}
    for sub in factor_lists:
        for f in sub:
            counts[f] = counts.get(f, 0) + 1
    return counts


def to_frequency(counts, group_size):
    if group_size <= 0:
        return {k: 0.0 for k in counts.keys()}
    return {k: v / group_size for k, v in counts.items()}

# ------------------------- 加权 Jaccard -------------------------
def weighted_jaccard(freqs1, freqs2):
    keys = set(freqs1.keys()) | set(freqs2.keys())
    num = sum(min(freqs1.get(k, 0.0), freqs2.get(k, 0.0)) for k in keys)
    den = sum(max(freqs1.get(k, 0.0), freqs2.get(k, 0.0)) for k in keys)
    return num / den if den > 0 else 0.0

# ------------------------- JS Divergence -------------------------
def js_divergence(freqsA, freqsB):
    keys = sorted(list(set(freqsA.keys()) | set(freqsB.keys())))
    p = np.array([freqsA.get(k, 0.0) for k in keys])
    q = np.array([freqsB.get(k, 0.0) for k in keys])
    if p.sum() == 0: p = np.ones_like(p) / len(p)
    if q.sum() == 0: q = np.ones_like(q) / len(q)
    return jensenshannon(p, q) ** 2

# ------------------------- Pearson & Cosine 相似度（全特征） -------------------------
def pearson_similarity(freqsA, freqsB):
    keys = sorted(list(set(freqsA.keys()) | set(freqsB.keys())))
    p = np.array([freqsA.get(k, 0.0) for k in keys])
    q = np.array([freqsB.get(k, 0.0) for k in keys])
    corr, _ = pearsonr(p, q)
    return corr if corr == corr else 0.0


def cosine_similarity(freqsA, freqsB):
    keys = sorted(list(set(freqsA.keys()) | set(freqsB.keys())))
    p = np.array([freqsA.get(k, 0.0) for k in keys])
    q = np.array([freqsB.get(k, 0.0) for k in keys])
    if norm(p) == 0 or norm(q) == 0:
        return 0.0
    return float(np.dot(p, q) / (norm(p) * norm(q)))

# ------------------------- Top-k 特征 ------------------------- -------------------------
def get_topk(freqs, k=3):
    valid = {feat: f for feat, f in freqs.items() if feat in VALID_FEATURES}
    sorted_items = sorted(valid.items(), key=lambda x: -x[1])
    return dict(sorted_items[:k])


def hidden_bias_v4(csv_path, k=3, output_dir="/result/factors_overlap"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    df["normalized_factors"] = df["key_factors"].apply(lambda x: normalize_key_factors(str(x)))

    def age_grp(age):
        try:
            age = float(age)
        except:
            return np.nan
        return "18-59" if 18 <= age < 60 else ("60+" if age >= 60 else np.nan)

    df["age_group"] = df["age"].apply(age_grp)

    group_dims = {
        "age_group": ["18-59", "60+"],
        "ethnicity": ["white", "black", "asian", "other"],
        "gender": ["M", "F"]
    }

    summary = []

    for dim, values in group_dims.items():
        subdfs = {v: df[df[dim] == v] for v in values if len(df[df[dim] == v]) > 0}
        if len(subdfs) < 2:
            continue

        freqs = {}
        topk = {}

        for g, gdf in subdfs.items():
            counts = get_feature_counts(gdf["normalized_factors"].tolist())
            freqs[g] = to_frequency(counts, len(gdf))
            topk[g] = get_topk(freqs[g], k)

        groups = list(freqs.keys())

        base = "white" if dim == "ethnicity" and "white" in groups else ("M" if dim == "gender" else groups[0])

        for g in groups:
            if g == base: continue

            # Top-3 Weighted Jaccard
            wj_top3 = weighted_jaccard(topk[base], topk[g])

            # All-feature weighted Jaccard
            wj_all = weighted_jaccard(freqs[base], freqs[g])

            # JS Divergence
            jsd = js_divergence(freqs[base], freqs[g])

            # Pearson & Cosine
            pear_corr = pearson_similarity(freqs[base], freqs[g])
            cos_sim = cosine_similarity(freqs[base], freqs[g])

            summary.append({
                "维度": dim,
                "基准组": base,
                "比较组": g,
                "Top3_加权Jaccard": round(wj_top3, 3),
                "All_加权Jaccard": round(wj_all, 3),
                "JS_Divergence": round(jsd, 3),
                "Pearson": round(pear_corr, 3),
                "Cosine": round(cos_sim, 3)
            })

    summary_df = pd.DataFrame(summary)
    outpath = os.path.join(output_dir, "xxx.csv")
    summary_df.to_csv(outpath, index=False, encoding="utf-8-sig")
    print("Results saved to", outpath)

    return summary_df


if __name__ == "__main__":
    csv_path = r"/result/factors_overlap/xxx.csv"
    hidden_bias_v4(csv_path, k=3)
