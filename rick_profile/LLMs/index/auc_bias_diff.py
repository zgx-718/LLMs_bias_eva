# -*- coding: utf-8 -*-
"""
左图是各亚组AUROC对比，右图是亚组间最大AUC差值
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from pathlib import Path

# ↓↓↓↓↓ 1. 把文件路径写死在这里 ↓↓↓↓↓
CONFIG = [
    ("/result/predictions/qwen3-32B_prediction1_base.csv", 'base', '#1f77b4'),
    ("/result/predictions/qwen3-32B_prediction1_base.csv", 'fair', '#ff7f0e'),
    ("/result/predictions/qwen3-32B_prediction1_system2.csv", 'system2', '#2ca02c'),
    ("/result/predictions/qwen3-32B_prediction1_perfect.csv", 'CAP', '#d62728'),
]
# ↑↑↑↑↑ 支持任意多个 ↑↑↑↑↑

# 2. 常量
PRED_COL = "probability"
LABEL_COL = "death_hosp"


def age_group(age):
    if pd.isna(age):
        return None
    age = float(age)
    return "age_18-59" if 18 <= age < 60 else "age_60+"


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return np.nan if len(np.unique(y_true)) < 2 else roc_auc_score(y_true, y_score)


def one_file_aucs(fp: Path) -> dict:
    """计算一个文件中各亚组的AUROC"""
    df = pd.read_csv(fp)
    df["age_group"] = df["age"].apply(age_group)

    masks = {
        "male": df["gender"].str.upper() == "M",
        "female": df["gender"].str.upper() == "F",
        "age_18-59": df["age_group"] == "age_18-59",
        "age_60+": df["age_group"] == "age_60+",
        "asian": df["ethnicity"].str.lower() == "asian",
        "black": df["ethnicity"].str.lower() == "black",
        "other": df["ethnicity"].str.lower() == "other",
        "white": df["ethnicity"].str.lower() == "white",
    }
    return {g: auroc(df[mask][LABEL_COL], df[mask][PRED_COL]) for g, mask in masks.items()}


def one_file_diff(fp: Path) -> dict:
    """计算一个文件中各亚组间的最大AUC差值"""
    aucs = one_file_aucs(fp)

    return {
        "Gender": abs(aucs["male"] - aucs["female"]),
        "Age": abs(aucs["age_18-59"] - aucs["age_60+"]),
        "Race": abs(np.nanmax([aucs[g] for g in ["asian", "black", "other", "white"]]) -
                    np.nanmin([aucs[g] for g in ["asian", "black", "other", "white"]])),
    }


# 4. 主流程
def main():
    # 解析配置
    file_paths, labels, colors = zip(*CONFIG)

    # 获取各亚组AUROC和差值数据
    subgroup_aucs = {lbl: one_file_aucs(Path(fp)) for fp, lbl, _ in CONFIG}
    diffs = {lbl: one_file_diff(Path(fp)) for fp, lbl, _ in CONFIG}

    # 准备左图数据 - 各亚组AUROC对比
    subgroups = ["male", "female", "age_18-59", "age_60+", "asian", "black", "other", "white"]
    subgroup_names = ["Male", "Female", "Age 18-59", "Age 60+", "Asian", "Black", "Other", "White"]

    # 准备右图数据 - 亚组间最大AUC差值
    df_diff = pd.DataFrame(diffs).T  # 行=label，列=属性

    # 创建画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ===== 左图：各亚组AUROC对比 =====
    x_left = np.arange(len(subgroups))
    width_left = 0.8 / len(labels)

    for i, (lbl, color) in enumerate(zip(labels, colors)):
        auc_values = [subgroup_aucs[lbl][subgroup] for subgroup in subgroups]
        ax1.bar(x_left + i * width_left, auc_values, width_left,
                label=lbl, color=color, alpha=0.8)
        # 柱顶写数值
        for j, val in enumerate(auc_values):
            ax1.text(x_left[j] + i * width_left, val + 0.005,
                     f"{val:.2f}", ha='center', va='bottom',
                     color='black', fontsize=7)
    ax1.legend(loc='upper right', fontsize=7, title='Method', title_fontsize=7,
               frameon=False, ncol=1, bbox_to_anchor=(1, 1))
    ax1.set_xticks(x_left + width_left * (len(labels) - 1) / 2)
    ax1.set_xticklabels(subgroup_names, rotation=45, ha='right')
    ax1.set_ylabel("AUROC")
    ax1.set_title("AUROC across subgroups")
    ax1.legend()
    ax1.grid(axis="y", ls="--", alpha=0.4)
    ax1.set_ylim(0.5, 1.0)  # AUROC通常范围

    # ===== 右图：亚组间最大AUC差值 =====
    x_right = np.arange(3)  # Gender/Age/Race
    width_right = 0.15

    for i, (lbl, row) in enumerate(df_diff.iterrows()):
        ax2.bar(x_right + i * width_right, row.values, width_right,
                label=lbl, color=colors[i], alpha=0.8)
        # 在柱顶写数值
        for j, val in enumerate(row.values):
            ax2.text(x_right[j] + i * width_right, val + 0.002,
                     f"{val:.3f}", ha='center', va='bottom',
                     color='black', fontsize=8)

    ax2.axhline(0.05, color='black', linestyle='--', linewidth=1.2, label='0.05 threshold')
    ax2.set_xticks(x_right + width_right * (len(df_diff) - 1) / 2)
    ax2.set_xticklabels(df_diff.columns)
    ax2.set_ylabel("Absolute AUC difference")
    ax2.set_title("Absolute AUC bias differences across models")
    ax2.legend()
    ax2.grid(axis="y", ls="--", alpha=0.4)
    ax2.set_ylim(0, 0.1)  # 纵轴 0~0.1

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()