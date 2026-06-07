# Lung Cancer Risk Prediction — XAI Clinical Dashboard

> **Data Science** | Python · scikit-learn · LightGBM · SHAP · Streamlit

## Overview

A clinical decision support prototype that pairs real-time lung cancer risk scores with SHAP explanations and a counterfactual What-If engine — so a clinician sees not just a probability, but exactly why, and what a patient can do about it.

## The Problem This Solves

Standard ML risk models give clinicians a number: 73% probability. That number

is useless without context. A doctor can't act on a probability — they need to know which factors are driving it, for this specific patient, right now.

This project builds that missing layer. The dashboard predicts individual lung cancer

risk from 29 patient variables and immediately renders a SHAP waterfall explanation —

so a clinician sees not just High Risk but exactly why: elevated CRP, reduced

FEV1, 30 pack-years. The explanation and the prediction arrive together.

A secondary "What-If" engine lets a clinician manipulate modifiable factors (smoking

cessation, radon mitigation, exercise) and see the projected risk change in real time —

with ranked single-intervention impact so the highest-leverage behavior change is

always surfaced first.


## Limitations
**The data is synthetic — the metrics are not meaningful benchmarks.** 
This model was trained and evaluated on a procedurally generated dataset. A 0.9991 AUC means the model learned to reverse-engineer the data-generating function, not to detect lung cancer. These numbers exist to demonstrate methodology, not to make performance claims. Any recruiter or clinician reading this should treat them as implementation evidence, not predictive evidence.

**The What-If engine shows model behavior, not biology.**

When the dashboard projects that quitting smoking reduces risk by 18%, that is the model responding to a changed input — not a clinically validated causal estimate. Confounding, reverse causation, and model extrapolation are all uncontrolled. This is a demonstration of XAI interaction design, not an epidemiological tool.

**SHAP explains the model — not the disease.**

SHAP values tell you why this model ranked pack-years as the strongest predictor. They do not confirm that pack-years causes lung cancer. The clinical alignment section shows the model's logic is coherent with medical literature — it does not validate the model's predictions against real patient outcomes.

**No subgroup validation on real populations.**

The fairness audit audits the model's behavior across synthetic demographic splits. Whether that fairness holds on real clinical populations — where data collection bias, access disparities, and comorbidity patterns differ — is entirely unknown.

**This has not been tested with clinicians.**
No usability testing, workflow integration study, or clinical review has been conducted. The risk stratification thresholds (0.48, 0.62) are methodologically justified but not clinically validated. A real deployment would require IRB review, clinician co-design, and prospective outcome tracking.


## Risk Stratification Logic

```python
if risk_score > 0.62:
    # CRITICAL RISK → Immediate referral for Low-Dose CT (LDCT)
elif risk_score > 0.48:
    # MODERATE RISK → Short-term follow-up (3–6 months)
else:
    # LOW RISK → Routine annual screening
```

## Quick Start

```bash
git clone https://github.com/alexbiuckians/lung-cancer-risk.git
cd lung-cancer-risk
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```
## How to Run

# Install dependencies
pip install -r requirements.txt

# Run the core pipeline
python pipeline.py --data data/lung_cancer.xlsx

# Clinical Validation & Fairness Audit
python calibration_fairness.py

# Registry & Metadata Integration
python metadata_generator.py

# Application Deployment
streamlit run app/streamlit_app.py


## About This Project
This project builds a clinically-reasoned, bias-audited ML decision support tool with full XAI explainability — including SHAP explanations, a counterfactual What-If engine, fairness auditing across demographic subgroups, and probability calibration. Built independently to demonstrate what  production-grade clinical ML actually looks like: not just a model, but a  complete pipeline from raw data to explainable, audited, deployable application.
## Implementation Evidence (5-fold Stratified CV)
> ⚠️ These metrics reflect a synthetic dataset with a known data-generating 
> process. They demonstrate methodology, not real-world predictive performance.
| Metric | Score |
|--------|-------|
| ROC-AUC | **0.9999** |
| F2-Score | **0.9928** |
| MCC | **0.9795** |
| Recall | **0.9984** |
| Precision | **0.9800** |

> **Why F2 and MCC?** In a 75/25 imbalanced dataset, accuracy is misleading. F2 weights recall 2x over precision — a missed cancer is far more costly than a false alarm. MCC is the gold-standard single metric for imbalanced binary classification.
.
## Clinical Validation of Feature Importance

The model's top SHAP features align closely with established clinical evidence
for lung cancer risk, lending credibility to its internal logic.

**Pack-years** is the model's single strongest predictor. This is clinically 
coherent — cumulative tobacco exposure is the most well-established dose-response 
risk factor for lung cancer, recognized by both the IARC and ACS.

**Age** ranks second, consistent with USPSTF 2021 screening guidelines which 
set age 50 as the lower eligibility threshold for annual low-dose CT screening, 
reflecting the sharp rise in incidence after midlife.

**Chest X-ray abnormality** ranks third — a direct radiological finding and 
standard indicator of pulmonary compromise flagged in oncology workup protocols.

**CRP level (systemic inflammation)** ranks fourth. Elevated CRP is associated 
with chronic airway inflammation, a known precursor environment for malignant 
cell development, and is increasingly included in multi-biomarker lung cancer 
risk models in published literature.

**Air pollution index** rounds out the top five — consistent with EPA and WHO 
evidence linking PM2.5 particulate exposure to elevated lung cancer incidence.

Notably, the model's clinical audit shows lifestyle factors carry the 
strongest predictive signal (46.9%), followed by clinical markers (35.7%) 
and demographics (17.4%). This distribution reflects the synthetic 
data-generating function rather than validated clinical weighting priorities.

> ⚠️ Metrics reflect a synthetic dataset with a known data-generating process.
> Real-world performance requires validation on clinical data such as the
> National Lung Screening Trial (NLST).
## Project Structure

```
lung-cancer-risk/
├── data/
│   └── lung_cancer.xlsx                        ← synthetic dataset (not tracked in git)
├── app/
│   ├── streamlit_app.py                        ← Phase 4: XAI clinical dashboard
│   └── model/
│       ├── final_model.pkl                     ← trained model
│       ├── shap_explainer.pkl                  ← SHAP explainer
│       ├── feature_meta.json                   ← features, threshold, average patient
│       └── slider_config.json                  ← min/max/default for all 29 inputs
├── outputs/                                    ← all generated by pipeline.py
│   ├── 01_descriptive_stats.csv
│   ├── 02a_full_correlation_matrix.png
│   ├── 02b_smoking_clinical_correlation.png
│   ├── 02c_high_correlation_pairs.csv
│   ├── 03a_vif_analysis.csv
│   ├── 03b_vif_barchart.png
│   ├── 04_clean_dataset.csv
│   ├── 05_imbalance_strategy_comparison.csv
│   ├── 05_imbalance_strategy_barchart.png
│   ├── 06_precision_recall_curve.png
│   ├── 07_phase1_metadata.json
│   ├── 08_model_leaderboard.csv
│   ├── 08_model_leaderboard_chart.png
│   ├── 08b_optuna_lgbm_tuning.csv
│   ├── 09_pr_curves_all_models.png
│   ├── 10_threshold_sensitivity.png
│   ├── 11_confusion_matrix_best_model.png
│   ├── 12_lgbm_feature_importance.csv
│   ├── 12_lgbm_feature_importance.png
│   ├── 12b_lgbm_feature_selection.csv
│   ├── phase2_metadata.json
│   ├── phase3_metadata.json
│   ├── clinical_audit_log.json
│   ├── shap/
│   │   ├── 13_shap_summary_dot.png
│   │   ├── 14_shap_summary_bar.png
│   │   ├── 15_lifestyle_vs_clinical_drivers.png
│   │   ├── 16_waterfall_Non-smoker_Flagged_High_Risk.png
│   │   ├── 16_waterfall_High_Environmental_Exposure.png
│   │   ├── 16_waterfall_True_High_Risk_Low_Confidence.png
│   │   ├── 17_shap_dependence_plots.png
│   │   └── 18_shap_medical_alignment.csv
│   ├── calibration/                            ← generated by phase6_calibration_fairness.py
│   │   ├── P1_calibration_curves.png
│   │   └── P1_calibration_summary.json
│   └── fairness/                               ← generated by phase6_calibration_fairness.py
│       ├── P3_subgroup_metrics.csv
│       ├── P3_subgroup_performance.png
│       └── P3_equity_gaps.json
├── pipeline.py                                 ← Phases 1–3: data → model → SHAP
├── phase6_calibration_fairness.py              ← calibration + fairness audit
├── patch_feature_meta.py                       ← adds average_patient to feature_meta.json
├── requirements.txt
└── README.md
```
## Key Design Decisions

**Logistic Regression as final model** — head-to-head cross-validation showed Logistic Regression outperformed both Random Forest and LightGBM on F2 (0.9928 vs 0.9582) and AUC (0.9999 vs 0.9976). This is expected and honest: the synthetic data-generating function is fundamentally linear, so a linear model  wins. LightGBM remains in the pipeline as a benchmark comparison


**F2 threshold tuning (0.48, not 0.5)** — the default 0.5 threshold 
optimizes accuracy, which is the wrong goal in cancer screening. A missed 
cancer is catastrophically more costly than a false alarm. The Precision-Recall 
curve identified 0.48 as the threshold that maximizes F2, achieving 98.4% 
precision and 100% recall on the held-out test set.

## Tech Stack

`Python 3.10` · `scikit-learn` · `LightGBM` ·  `imbalanced-learn` · `SHAP` · `Streamlit` · `Optuna` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `joblib` · `statsmodels`
---
*Clinical decision support prototype demonstrating XAI methodology on research data. 
Dataset: [Kaggle source / synthetic]. Not for clinical use.*
*Clinical decision support prototype demonstrating XAI methodology on research data. 
Dataset: [Kaggle source / synthetic]. Not for clinical use.*
