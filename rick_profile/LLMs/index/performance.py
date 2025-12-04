
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, brier_score_loss
from scipy.stats import norm

# 输入文件路径
input_csv = "/result/predictions/xxx.csv"
output_csv = '/result/evaluation_results.csv'

# 提取文件名前缀作为name字段的值
name = input_csv.split('\\')[-1].split('_')[0]  # 提取文件名前缀

# 读取CSV文件
try:
    df = pd.read_csv(input_csv)
except FileNotFoundError:
    print(f"文件 {input_csv} 未找到，请检查文件路径！")
    exit()
except Exception as e:
    print(f"读取文件时发生错误：{e}")
    exit()

# 提取相关字段
y_true = df['death_hosp']  # 实际标签
y_prob = df['probability']  # 预测概率
y_pred = (pd.to_numeric(y_prob, errors='coerce') > 0.3).astype(int)

# 计算指标
auroc = roc_auc_score(y_true, y_prob)
precision, recall, _ = precision_recall_curve(y_true, y_prob)
auprc = auc(recall, precision)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
brier_score = brier_score_loss(y_true, y_prob)
observed_deaths = y_true.sum()
expected_deaths = y_prob.sum()
smr = observed_deaths / expected_deaths

# 计算95%置信区间
n = len(y_true)
se_auroc = np.sqrt((auroc * (1 - auroc)) / n)
ci_auroc = norm.interval(0.95, loc=auroc, scale=se_auroc)

se_auprc = np.sqrt((auprc * (1 - auprc)) / n)
ci_auprc = norm.interval(0.95, loc=auprc, scale=se_auprc)

se_accuracy = np.sqrt((accuracy * (1 - accuracy)) / n)
ci_accuracy = norm.interval(0.95, loc=accuracy, scale=se_accuracy)

se_f1 = np.sqrt((f1 * (1 - f1)) / n)
ci_f1 = norm.interval(0.95, loc=f1, scale=se_f1)

se_precision = np.sqrt((precision * (1 - precision)) / n)
ci_precision = norm.interval(0.95, loc=precision, scale=se_precision)

se_sensitivity = np.sqrt((sensitivity * (1 - sensitivity)) / (tp + fn))
ci_sensitivity = norm.interval(0.95, loc=sensitivity, scale=se_sensitivity)

se_specificity = np.sqrt((specificity * (1 - specificity)) / (tn + fp))
ci_specificity = norm.interval(0.95, loc=specificity, scale=se_specificity)

se_npv = np.sqrt((npv * (1 - npv)) / (tn + fn))
ci_npv = norm.interval(0.95, loc=npv, scale=se_npv)

se_ppv = np.sqrt((ppv * (1 - ppv)) / (tp + fp))
ci_ppv = norm.interval(0.95, loc=ppv, scale=se_ppv)

se_brier = np.sqrt((brier_score * (1 - brier_score)) / n)
ci_brier = norm.interval(0.95, loc=brier_score, scale=se_brier)

se_smr = np.sqrt((observed_deaths / expected_deaths**2) + (observed_deaths / expected_deaths**3))
ci_smr = norm.interval(0.95, loc=smr, scale=se_smr)

# 保留3位小数
auroc = f"{round(auroc, 3)}({round(ci_auroc[0], 3)}-{round(ci_auroc[1], 3)})"
auprc = f"{round(auprc, 3)}({round(ci_auprc[0], 3)}-{round(ci_auprc[1], 3)})"
accuracy = f"{round(accuracy, 3)}( {round(ci_accuracy[0], 3)}-{round(ci_accuracy[1], 3)})"
f1 = f"{round(f1, 3)}({round(ci_f1[0], 3)}-{round(ci_f1[1], 3)})"
precision = f"{round(precision, 3)}({round(ci_precision[0], 3)}-{round(ci_precision[1], 3)})"
sensitivity = f"{round(sensitivity, 3)}({round(ci_sensitivity[0], 3)}-{round(ci_sensitivity[1], 3)})"
specificity = f"{round(specificity, 3)}({round(ci_specificity[0], 3)}-{round(ci_specificity[1], 3)})"
npv = f"{round(npv, 3)}({round(ci_npv[0], 3)}-{round(ci_npv[1], 3)})"
ppv = f"{round(ppv, 3)}({round(ci_ppv[0], 3)}-{round(ci_ppv[1], 3)})"
brier_score = f"{round(brier_score, 3)}({round(ci_brier[0], 3)}-{round(ci_brier[1], 3)})"
smr = f"{round(smr, 3)}({round(ci_smr[0], 3)}-{round(ci_smr[1], 3)})"

# 创建结果字典
results = {
    'name': [name],
    'AUROC': [auroc],
    'AUPRC': [auprc],
    'Accuracy': [accuracy],
    'F1 Score': [f1],
    'Precision': [precision],
    'Sensitivity (Recall)': [sensitivity],
    'Specificity': [specificity],
    'NPV': [npv],
    'PPV': [ppv],
    'Brier Score(95% CI)': [brier_score],
    'SMR(95% CI)': [smr]
}

# 将结果保存到新的CSV文件
try:
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv, index=False)
    print(f"评估结果已成功保存到 {output_csv}")
except Exception as e:
    print(f"保存文件时发生错误：{e}")

# 打印结果到控制台
print("\n评估结果：")
print(f"Name: {name}")
print(f"AUROC: {auroc}")
print(f"AUPRC: {auprc}")
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
print(f"Precision: {precision}")
print(f"Sensitivity (Recall): {sensitivity}")
print(f"Specificity: {specificity}")
print(f"NPV: {npv}")
print(f"PPV: {ppv}")
print(f"Brier Score: {brier_score}")
print(f"SMR: {smr}")

