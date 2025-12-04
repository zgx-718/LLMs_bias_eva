Project Title:  
**Co-Optimizing Fairness and Performance: A Clinically-Adaptive Prompting Framework to Boost LLM Performance in ICU Mortality Prediction**

---

### Project Overview  
Built on MIMIC-IV, this project integrates multiple prompting strategies (base, fair, system2, CAP, etc.) to predict in-hospital mortality and deliver multi-dimensional fairness / bias analyses. It is designed for medical-AI model development, evaluation, and fairness monitoring.

---

### Key Features  
1. **Mortality-risk prediction** (multi-method comparison)  
2. **Fairness & bias analysis** (sex, age, race, etc.)  
3. **Mis-classified case extraction & case-library construction**  
4. **Key-feature attribution and comprehensive statistical metrics**  
5. **Support for major ICU scores, vital signs, and lab indicators**

---

### Folder Structure  
- `data/` – data-preprocessing scripts  
- `index/` – fairness statistics & visualization scripts  
- `Mistake_case_library/` – tools for bias-case analysis, comparison & merging  
- `result/` – output analyses (EOD, factors_overlap, predictions, …)  
- `death_predictions_xxx.py` – prediction scripts for different models  

---

### Environment  
- Python ≥ 3.8  
- See `requirements.txt` for third-party packages  

---

### Quick Start  
1. Install dependencies:  
   `pip install -r requirements.txt`  
2. Prepare raw clinical CSVs; convert to JSON with `data/dataPreprocessing.py` if needed.  
3. Run the desired script (prediction, EOD, factor attribution, AUROC, …).  
4. Results appear in `result/` and its sub-folders.

---

### Typical Use-Cases  
- Compare and report ICU mortality model performance  
- Conduct multi-dimensional fairness / bias analysis (sex, age, race)  
- Track, merge, and attribute bias-prone cases


# MIMIC-IV Geriatric ICU Mortality Prediction with XGBoost

This project leverages the MIMIC-IV database to build an **in-hospital mortality prediction model** using **XGBoost**.  
It automatically evaluates and exports key performance metrics to `result_metrics.csv`, including:

- AUROC  
- AUPRC  
- Accuracy  
- F1  
- Precision  
- Recall / Sensitivity  
- Specificity  
- NPV  
- Brier Score  

---

## Quick Start

1. **Extract data**  
   Run `extract_mimiciv.sql` to generate `mimic.csv` (MIMIC-IV only, geriatric ICU patients).

2. **(Optional) Adjust paths**  
   Edit file paths inside `run_xgboost_mimiciv.py` if necessary.

3. **Train & evaluate**  
   ```bash
   python run_xgboost_mimiciv.py
   ```
   A single command produces `result_metrics.csv` with all metrics.

---

## File Map

| File | Purpose |
|------|---------|
| `extract_mimiciv.sql` | SQL query to create `mimic.csv` |
| `run_xgboost_mimiciv.py` | Pre-processing, XGBoost modelling, tuning, and metric export |
| `mimic.csv` | Raw extracted data (not shipped; generate locally) |
| `result_metrics.csv` | Output: all performance metrics |
| `requirements.txt` | Python dependencies |
| `README.md` | This document |
