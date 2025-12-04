import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, brier_score_loss
from xgboost import XGBClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
import os

# 路径设置（如需修改请修改这里）
data_path = 'mimic.csv'   # mimic-iv 数据路径
result_path = 'result_metrics.csv'

# 读取数据，仅用 mimic.csv
raw_data = pd.read_csv(data_path)

# 数据预处理（直接插补，不删缺失行）
X = raw_data.drop(['death_hosp'], axis=1)
y = raw_data['death_hosp']

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# MICE插补
mice = IterativeImputer(random_state=42, max_iter=10, sample_posterior=True)
X_train_imp = mice.fit_transform(X_train)
X_test_imp = mice.transform(X_test)

# XGBoost参数网格（可简化或丰富）
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1]
}
# 网格搜索+5折CV
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
gs = GridSearchCV(xgb_clf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
gs.fit(X_train_imp, y_train)

# 最优模型训练与预测
y_pred_prob = gs.predict_proba(X_test_imp)[:, 1]
y_pred = gs.predict(X_test_imp)

# 性能指标收集
roc_auc = roc_auc_score(y_test, y_pred_prob)
auprc = average_precision_score(y_test, y_pred_prob)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
specificity = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0]+cm[0,1]) else 0
npv = cm[0,0] / (cm[0,0] + cm[1,0]) if (cm[0,0]+cm[1,0]) else 0
brier = brier_score_loss(y_test, y_pred_prob)

# 保存结果到csv
metrics = {
    'AUROC': roc_auc,
    'AUPRC': auprc,
    'Accuracy': acc,
    'F1 Score': f1,
    'Precision': prec,
    'Sensitivity (Recall)': recall,
    'Specificity': specificity,
    'NPV': npv,
    'Brier Score': brier
}
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv(result_path, index=False)
print('模型性能保存至', result_path)