import pandas as pd
import numpy as np
import os

# --------------------------------------------------
# 1. 路径与阈值
file_path = '/result/predictions/xxx.csv'
threshold = 0.3  # 可在此处统一调整
# --------------------------------------------------

df = pd.read_csv(file_path)


def age_group_simple(age):
    """简化年龄分组：18-59 vs 60+"""
    if pd.isna(age):
        return None
    age = float(age)
    if age < 60:
        return '18-59'
    else:
        return '60+'


# 应用年龄分组
df['age'] = df['age'].apply(age_group_simple)
df['pred_label'] = (df['probability'] >= threshold).astype(int)


# 2. 计算 EOD 相关指标的工具函数
def calculate_eod_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan

    return tpr, fpr


# 3. 需要比较的组对 - 只保留指定的三组年龄比较
comparison_map = {
    'gender': [('M', 'F')],
    'age': [('18-59', '60+')],  # 主要年龄分组比较
    'ethnicity': [
        ('white', 'black'),
        ('white', 'other'),
        ('white', 'asian')
    ]
}

records = []

# 4. 主循环 - 计算完整的EOD指标
for dim, pairs in comparison_map.items():
    for g1, g2 in pairs:
        sub1 = df[df[dim] == g1]
        tpr1, fpr1 = calculate_eod_metrics(sub1['death_hosp'].values, sub1['pred_label'].values)

        sub2 = df[df[dim] == g2]
        tpr2, fpr2 = calculate_eod_metrics(sub2['death_hosp'].values, sub2['pred_label'].values)

        # 计算EOD的两个核心差异
        tpr_diff = round(tpr1 - tpr2, 3) if not (np.isnan(tpr1) or np.isnan(tpr2)) else np.nan
        fpr_diff = round(fpr1 - fpr2, 3) if not (np.isnan(fpr1) or np.isnan(fpr2)) else np.nan

        # EOD值取两个差异的绝对值中的最大值
        if not (np.isnan(tpr_diff) or np.isnan(fpr_diff)):
            eod_value = round(max(abs(tpr_diff), abs(fpr_diff)), 3)
        else:
            eod_value = np.nan

        records.append([
            dim, g1, g2,
            round(tpr1, 3) if not np.isnan(tpr1) else 'NaN',
            round(tpr2, 3) if not np.isnan(tpr2) else 'NaN',
            tpr_diff,
            round(fpr1, 3) if not np.isnan(fpr1) else 'NaN',
            round(fpr2, 3) if not np.isnan(fpr2) else 'NaN',
            fpr_diff,
            eod_value
        ])

# 5. 输出完整结果
out_df = pd.DataFrame(records,
                      columns=['Dimension', 'Group1', 'Group2',
                               'TPR1', 'TPR2', 'ΔTPR',
                               'FPR1', 'FPR2', 'ΔFPR',
                               'EOD'])
print("完整EOD分析结果:")
print(out_df.to_string(index=False))

# 6. 保存
output_dir = '/result/EOD/'
os.makedirs(output_dir, exist_ok=True)
out_path = os.path.join(output_dir, 'complete_eod_analysis_ACP.csv')
out_df.to_csv(out_path, index=False, encoding='utf-8-sig')
print(f"\n已保存完整结果：{out_path}")

# 7. EOD总结报告
print("\n" + "=" * 50)
print("EOD 公平性总结报告")
print("=" * 50)

# 按维度分组显示
dimension_order = ['gender', 'age', 'age_group', 'ethnicity']
dimension_names = {
    'gender': '性别',
    'age': '年龄分组(18-59 vs 60+)',
    # 'age_group': '年龄分组(内部比较)',
    'ethnicity': '种族'
}

for dim in dimension_order:
    if dim in dimension_names:
        dim_records = [r for r in records if r[0] == dim]
        print(f"\n【{dimension_names[dim]}】:")

        for record in dim_records:
            g1, g2, eod = record[1], record[2], record[-1]
            tpr_diff, fpr_diff = record[5], record[8]

            if not np.isnan(eod):
                fairness_level = "✓ 公平" if eod <= 0.05 else "⚠ 轻微偏倚" if eod <= 0.1 else "✗ 显著偏倚"
                print(f"  {g1} vs {g2}: EOD = {eod} ({fairness_level})")
                print(f"    TPR差异: {tpr_diff}, FPR差异: {fpr_diff}")
            else:
                print(f"  {g1} vs {g2}: 数据不足无法计算")
