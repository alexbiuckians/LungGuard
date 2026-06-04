import sys
sys.stdout.reconfigure(encoding="utf-8")
import warnings
warnings.filterwarnings("ignore")
import json 
import joblib
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score,
    matthews_corrcoef, make_scorer, fbeta_score,
    precision_score, recall_score
)
try:
    import shap
    SHAP_AVAILABLE = True
    shap.initjs()
except ImportError:
    SHAP_AVAILABLE = False
    print("[WARN] shap not installed: pip install shap")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("[WARN] lightgbm not installed: pip install lightgbm")

RANDOM_STATE = 42


import argparse

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--data", default=None)
_args, _ = _parser.parse_known_args()

_DEMO_MODE = os.environ.get("DEMO_MODE", "0").lower() in ("1", "true", "yes")

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

lung_cancer_data = (
    _args.data
    or os.environ.get("LUNG_CANCER_DATA")
    or os.path.join(_PROJECT_ROOT, "data", "Lung_Cancer.xlsx")
)
_APP_MODEL_DIR = os.path.join(_PROJECT_ROOT, "app", "model")
os.makedirs(_APP_MODEL_DIR, exist_ok=True)

def _generate_synthetic_dataset() -> str:
    """Generate a reproducible 1,000-row synthetic demo dataset and return its path."""
    print("[INFO] DEMO MODE — generating synthetic dataset (not for clinical use)")
    _n   = 1000
    _rng = np.random.default_rng(42)

    _age                = _rng.integers(30, 85, _n)
    _gender             = _rng.integers(0, 2, _n)
    _smoker             = (_rng.random(_n) < 0.35).astype(int)
    _smoking_years      = (_smoker * _rng.integers(0, 45, _n))
    _cigarettes_per_day = (_smoker * _rng.integers(0, 40, _n))
    _pack_years         = (_cigarettes_per_day * _smoking_years / 20).astype(int)
    _passive_smoking    = _rng.integers(0, 2, _n)
    _air_pollution      = _rng.integers(0, 10, _n)
    _radon              = _rng.integers(0, 10, _n)
    _occupational       = _rng.integers(0, 2, _n)
    _family_history     = _rng.integers(0, 2, _n)
    _copd               = ((_pack_years > 20) & (_rng.random(_n) < 0.4)).astype(int)
    _asthma             = (_rng.random(_n) < 0.12).astype(int)
    _prev_tb            = (_rng.random(_n) < 0.05).astype(int)
    _chr_cough          = _rng.integers(0, 2, _n)
    _chest_pain         = _rng.integers(0, 2, _n)
    _sob                = _rng.integers(0, 2, _n)
    _fatigue            = _rng.integers(0, 2, _n)
    _xray               = (_rng.random(_n) < 0.15).astype(int)
    _bmi                = _rng.integers(17, 40, _n)
    _o2_sat             = _rng.integers(90, 100, _n)
    _fev1               = _rng.integers(20, 50, _n)
    _crp                = _rng.integers(0, 20, _n)
    _exercise           = _rng.integers(0, 15, _n)
    _diet               = _rng.integers(1, 10, _n)
    _alcohol            = _rng.integers(0, 30, _n)
    _healthcare         = _rng.integers(0, 5, _n)
    _education          = _rng.integers(8, 20, _n)
    _income             = _rng.integers(1, 10, _n)
    _risk_logit = (
        -2.0                          # less extreme baseline
        + 0.04 * (_age - 50)          # age matters gradually
        + 0.8  * _smoker              # smoking is a factor, not a death sentence
        + 0.05 * _pack_years          # dose-response relationship
        + 1.5  * _copd                # COPD is strong
        + 1.2  * _xray                # X-ray finding is strong
        + 0.08 * (_crp - 5)          # inflammation matters
        - 0.06 * (_fev1 - 30)        # low lung function increases risk
        - 0.08 * (_o2_sat - 95)      # low O2 increases risk
        + 0.3  * _radon / 10
        + _rng.normal(0, 0.8, _n)    # more noise = more gradation
    )
    _risk_prob  = 1 / (1 + np.exp(-_risk_logit))
    _risk_label = (_risk_prob > 0.5).astype(int)

    _synth = pd.DataFrame({
        "age": _age, "gender": _gender,
        "smoker": _smoker, "smoking_years": _smoking_years,
        "cigarettes_per_day": _cigarettes_per_day, "pack_years": _pack_years,
        "passive_smoking": _passive_smoking,
        "air_pollution_index": _air_pollution,
        "radon_exposure": _radon,
        "occupational_exposure": _occupational,
        "family_history_cancer": _family_history,
        "copd": _copd, "asthma": _asthma, "previous_tb": _prev_tb,
        "chronic_cough": _chr_cough, "chest_pain": _chest_pain,
        "shortness_of_breath": _sob, "fatigue": _fatigue,
        "xray_abnormal": _xray, "bmi": _bmi,
        "oxygen_saturation": _o2_sat, "fev1_x10": _fev1, "crp_level": _crp,
        "exercise_hours_per_week": _exercise, "diet_quality": _diet,
        "alcohol_units_per_week": _alcohol, "healthcare_access": _healthcare,
        "education_years": _education, "income_level": _income,
        "lung_cancer_risk": _risk_label,
    })

    os.makedirs("data", exist_ok=True)
    _synth_path = "data/synthetic_demo_dataset.xlsx"
    _synth.to_excel(_synth_path, index=False)
    print(f"[INFO] Synthetic dataset saved to: {_synth_path}")
    return _synth_path

# ── Resolve dataset path ──────────────────────────────────────────────────────
if lung_cancer_data and os.path.exists(lung_cancer_data):
    pass  # Real dataset provided via CLI or env var — use it
elif _DEMO_MODE:
    lung_cancer_data = _generate_synthetic_dataset()
elif lung_cancer_data and not os.path.exists(lung_cancer_data):
    raise FileNotFoundError(
        f"Dataset not found at: {lung_cancer_data}\n"
        "Options:\n"
        "  1. Pass your real data:  python pipeline.py --data /path/to/data.xlsx\n"
        "  2. Set env var:          LUNG_CANCER_DATA=/path/to/data.xlsx\n"
        "  3. Run demo mode:        DEMO_MODE=1 python pipeline.py"
    )
else:
    raise FileNotFoundError(
        "No dataset specified.\n"
        "Options:\n"
        "  1. Pass your real data:  python pipeline.py --data /path/to/data.xlsx\n"
        "  2. Set env var:          LUNG_CANCER_DATA=/path/to/data.xlsx\n"
        "  3. Run demo mode:        DEMO_MODE=1 python pipeline.py"
    )

print(f"[INFO] Loading dataset from: {lung_cancer_data}")
Lung_cancer = pd.read_excel(lung_cancer_data)

#The target focuses what I will try to predict
target = "lung_cancer_risk"
os.makedirs("outputs", exist_ok=True)


#This contains every column name in the dataset except lung_cancer_risk
features_columns = [c for c in Lung_cancer.columns if c != target]

#This shows the basic information of the rows and columns
print(f"Shape : {Lung_cancer.shape[0]:,} rows x {Lung_cancer.shape[1]} columns")
#This shows the missing values
print(f"Missing values : {Lung_cancer.isnull().sum().sum()}")
# Count duplicate rows
print(f"Duplicate rows : {Lung_cancer.duplicated().sum()}")
#Finds the distribution of the target variable
value_count_of_cancer = Lung_cancer[target].value_counts()
#Calculates the imbalance ratio
imbalance_ratio = value_count_of_cancer[0] / value_count_of_cancer[1]
#This prints the class distribution
print(f"Class 0 (low risk):  {value_count_of_cancer[0]:,}  ({value_count_of_cancer[0]/len(Lung_cancer)*100:.1f}%)")
print(f"Class 1 (high risk): {value_count_of_cancer[1]:,}  ({value_count_of_cancer[1]/len(Lung_cancer)*100:.1f}%)")
print(f"Imbalance ratio:{imbalance_ratio:.2f}:1")

Lung_cancer.describe().T.to_csv("outputs/01_descriptive_stats.csv")
Lung_cancer.describe().T

#This section will have 2 correlation matrices.
## One will have all variables.
# The other will have a focused view on smoking variables and clinical markers.

#This is the full correlation heatmap for every single pair of variables in the dataset
corr_full = Lung_cancer[features_columns + [target]].corr()

fig, ax = plt.subplots(figsize=(18, 15))
mask = np.zeros_like(corr_full, dtype=bool)
mask[np.triu_indices_from(mask, k=1)] = True

sns.heatmap(corr_full, mask=mask, annot=True, fmt=".2f",
    cmap="RdYlGn", center=0, linewidths=0.4, annot_kws={"size": 7}, ax=ax)
ax.set_title("Full Correlation Matrix")
plt.tight_layout()
fig.savefig("outputs/02a_full_correlation_matrix.png", dpi=150)
plt.show()
smoking_variables  = ["smoker", "smoking_years", "cigarettes_per_day",
                 "pack_years", "passive_smoking"]
clinical_variables = ["oxygen_saturation", "fev1_x10", "crp_level",
                 "xray_abnormal", "copd", "asthma", "previous_tb",
                 "chronic_cough", "chest_pain", "shortness_of_breath", "fatigue"]

#This is the focused heatmap to see the most important variables
correlation_of_smoke = Lung_cancer[smoking_variables + clinical_variables + [target]].corr()

fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(correlation_of_smoke, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, annot_kws={"size": 9}, ax=ax)

ax.set_title("Smoking Variables x Clinical Markers Correlation")
plt.tight_layout()
fig.savefig("outputs/02b_smoking_clinical_correlation.png", dpi=150)
plt.show()
#This is the automated high correlation dectection to see if there are any pairs higher than .7

high_correlation_threshold = 0.70
high_pairs = []
for index1, feature1 in enumerate(features_columns):
    for index2, feature2 in enumerate(features_columns):
        if index2 <= index1:
            continue
        correlation_value = corr_full.loc[feature1,feature2]
        if abs(correlation_value) >= high_correlation_threshold:
            high_pairs.append({"var_1": feature1, "var_2": feature2, "correlation": round(correlation_value, 4)})

if high_pairs:
    high_pairs_Lung_Cancer = pd.DataFrame(high_pairs).sort_values("correlation", ascending=False)
else:
    high_pairs_Lung_Cancer = pd.DataFrame(columns=["var_1", "var_2", "correlation"])
high_pairs_Lung_Cancer.to_csv("outputs/02c_high_correlation_pairs.csv", index=False)
high_pairs_Lung_Cancer

#This section is the VIF Analysis. This will quantify 
# how much each feature's variance is inflated by correlation with other features. 
#This prepares the data
X_raw_data_lung= Lung_cancer[features_columns].copy()

# SOLUTION: Make statsmodels a hard requirement. If it's missing, fail loudly
#   with an install instruction rather than silently run bad math.
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant
except ImportError:
    raise ImportError(
        "[ERROR] statsmodels is required for VIF analysis.\n"
        "  Install it with:  pip install statsmodels\n"
        "  Then add to requirements.txt: statsmodels>=0.14.0"
    )

#This calculates the VIF Score (multivariate, mathematically correct)
X_constant = add_constant(X_raw_data_lung)
vif_data = pd.DataFrame({
    "feature": X_constant.columns,
    "VIF": [
        variance_inflation_factor(X_constant.values, index)
        for index in range(X_constant.shape[1])]
}).query("feature != 'const'").sort_values("VIF", ascending=False)

#This categorizes the results
vif_data["flag"] = vif_data["VIF"].apply(
    lambda v: "HIGH (>10)" if v > 10 else ("MODERATE (5-10)" if v > 5 else "OK"))
vif_data.to_csv("outputs/03a_vif_analysis.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 8))
colors = ["#e74c3c" if v > 10 else "#f39c12" if v > 5 else "#27ae60"
          for v in vif_data["VIF"]]
#This creates the visual warning 
ax.barh(vif_data["feature"], vif_data["VIF"].clip(upper=50), color=colors)
ax.axvline(5,  color="#f39c12", ls="--", lw=1.5, label="Moderate (5)")
ax.axvline(10, color="#e74c3c", ls="--", lw=1.5, label="High (10)")
ax.set_xlabel("VIF Score")
ax.set_title("Variance Inflation Factor by Feature")
ax.legend()
plt.tight_layout() 
fig.savefig("outputs/03b_vif_barchart.png", dpi=150)
plt.show()
vif_data

#This chunk of code removes the variables that are not useful.
#This drops smoking_years and cigarettes_per_day if they are high on the VIF list and keeps pack_years.
#This is due to the fact that they are calculated in formula packs_year=(cigarettes_per_dayxsmoking_years)/20
#This identifies the troublemakers
high_vif = vif_data[vif_data["VIF"] > 10]["feature"].tolist()


# smoking_years and cigarettes_per_day are mathematically redundant —
# pack_years = (cigarettes_per_day × smoking_years) / 20 — so drop them.
# Explicitly protect 'smoker' and 'pack_years': they carry independent
# clinical signal (binary flag + lifetime dose) and must stay in the model.
SMOKING_REDUNDANT = {"smoking_years", "cigarettes_per_day"}
SMOKING_KEEP      = {"smoker", "pack_years"}

features_to_drop = list(SMOKING_REDUNDANT & set(high_vif))

for f in high_vif:
    if f not in SMOKING_KEEP and f not in features_to_drop:
        features_to_drop.append(f)
#This drops the variables
lung_cancer_clean = Lung_cancer.drop(columns=features_to_drop).copy()
features_clean = [c for c in lung_cancer_clean.columns if c != target]
#This is the summary report
print(f"Original : {len(features_columns)} features")
print(f"Dropped  : {len(features_to_drop)} -> {features_to_drop}")
print(f"Retained : {len(features_clean)} features")

lung_cancer_clean.to_csv("outputs/04_clean_dataset.csv", index=False)
X_lung = lung_cancer_clean[features_clean]
y_lung = lung_cancer_clean[target]


#This chunk looks at three conditions with a 5-fold stratified cross validation.
#Condition A: baseline with no balancing.
#Condition B:  class_weight='balanced'. 
#Condition C:  SMOTE oversampling.
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
SCORING = ["roc_auc", "f1", "average_precision", "recall", "precision", "accuracy"]
def cv_metrics(pipe, X_lung, y_lung, label):
    r = cross_validate(pipe, X_lung, y_lung, cv=CV, scoring=SCORING, return_train_score=False)
    return {"Model": label,
            "ROC-AUC":   round(r["test_roc_auc"].mean(), 4),
            "F1":        round(r["test_f1"].mean(), 4),
            "Recall":    round(r["test_recall"].mean(), 4),
            "Precision": round(r["test_precision"].mean(), 4),
            "Avg-Prec":  round(r["test_average_precision"].mean(), 4)}

#This is the baseline
pipe_base = Pipeline([("sc", StandardScaler()),
                      ("lr", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))])
m_baseline = cv_metrics(pipe_base, X_lung, y_lung, "LR — no balancing")

# The class_weight is'balanced'
pipeline_balanced = Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(class_weight="balanced",
                                               max_iter=1000, random_state=RANDOM_STATE))])
m_balanced = cv_metrics(pipeline_balanced, X_lung, y_lung, "LR — class_weight='balanced'")

#These lines build a pipeline that scale the data
m_smote = None
pipe_smote = ImbPipeline([("sc", StandardScaler()),
                          ("sm", SMOTE(random_state=RANDOM_STATE)),
                          ("lr", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))])
m_smote = cv_metrics(pipe_smote, X_lung, y_lung, "LR + SMOTE")

#This organizes the results
rows = [m_baseline, m_balanced] + ([m_smote] if m_smote else [])
comparison_df = pd.DataFrame(rows).set_index("Model")
#This is the bar chart for visualizations
comparison_df.to_csv("outputs/05_imbalance_strategy_comparison.csv")
#This saves the evidence
fig, ax = plt.subplots(figsize=(10, 5))
comparison_df[["ROC-AUC", "F1", "Recall", "Precision", "Avg-Prec"]].plot(
    kind="bar", ax=ax, rot=15, colormap="Set2", edgecolor="white")
ax.set_ylim(0, 1.05); ax.set_title("Imbalance Strategy Comparison (5-fold CV)")
plt.tight_layout(); fig.savefig("outputs/05_imbalance_strategy_barchart.png", dpi=150); plt.show()
comparison_df

#This chunk shifts the model's sensitivity to prioritize safety over perfect accuracy
#This chunk scales the data
scaler = StandardScaler()
X_scaled_lung = scaler.fit_transform(X_lung)

#This chunk trains the final model
logistic_regression_final = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
logistic_regression_final.fit(X_scaled_lung, y_lung)
y_probability = logistic_regression_final.predict_proba(X_scaled_lung)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_lung, y_probability)
ap = average_precision_score(y_lung, y_probability)

#This chunk finds the safety-first threshold
f2 = (5 * precision * recall) / np.where((4 * precision + recall) > 0,
                                           4 * precision + recall, 1)
best_index    = np.argmax(f2[:-1])
best_threshold = thresholds[best_index]
#This reports and plots the results
print(f"Average Precision  : {ap:.4f}")
print(f"Best F2 threshold  : {best_threshold:.3f}")
print(f"  -> Precision: {precision[best_index]:.3f}  |  Recall: {recall[best_index]:.3f}")

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(recall, precision, lw=2, color="#2980b9", label=f"PR Curve (AP={ap:.3f})")
ax.scatter(recall[best_index], precision[best_index], s=120, zorder=5, color="#e74c3c",
           label=f"Best F2 threshold = {best_threshold:.3f}")
ax.axhline(value_count_of_cancer[1] / len(Lung_cancer), color="grey", ls="--", lw=1, label="Random baseline")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve — Balanced Logistic Regression"); ax.legend()
plt.tight_layout(); fig.savefig("outputs/06_precision_recall_curve.png", dpi=150); plt.show()


#This phase will show the clean report
meta = {
    "features_original": features_columns,
    "features_dropped":  features_to_drop,
    "features_clean":   features_clean,
    "target":           target,
    "n_rows":           len(Lung_cancer),
    "class_counts":     value_count_of_cancer.to_dict(),
    "imbalance_ratio":  round(imbalance_ratio, 3),
    "best_threshold_f2":        round(float(best_threshold), 4),
    "best_threshold_precision": round(float(precision[best_index]), 4),
    "best_threshold_recall":    round(float(recall[best_index]), 4),
    "average_precision":        round(float(ap), 4),
}

with open("outputs/07_phase1_metadata.json", "w") as fh:
    json.dump(meta, fh, indent=2)

print("Phase 1 complete. Metadata saved to outputs/07_phase1_metadata.json")
print(f"\nFeatures retained : {len(features_clean)}")
print(f"Recommended threshold (F2) : {meta['best_threshold_f2']}")
print(f"Carry forward to Phase 2   : class_weight='balanced'")



#PHASE 2
# SOLUTION: 20-iteration Optuna search on F2-score (your clinical priority metric).
#   Adds ~2min to runtime but gives you defensible hyperparameters + a logged audit trail.

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("[WARN] optuna not installed. Using defaults. Run: pip install optuna")

best_lgbm_params = {
    "n_estimators": 500, "learning_rate": 0.05, "num_leaves": 63
}  # safe fallback if Optuna unavailable

if LGB_AVAILABLE and OPTUNA_AVAILABLE:
    f2_scorer_optuna = make_scorer(lambda yt, yp: fbeta_score(yt, yp, beta=2))
    CV_opt = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    def lgbm_objective(trial):
        params = {
            "n_estimators":  trial.suggest_int("n_estimators", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves":    trial.suggest_int("num_leaves", 20, 100),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "subsample":     trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        scale_pos = (y_lung == 0).sum() / (y_lung == 1).sum()
        clf = lgb.LGBMClassifier(
            **params, scale_pos_weight=scale_pos,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        )
        scores = cross_validate(clf, X_lung, y_lung, cv=CV_opt,
                                scoring={"f2": f2_scorer_optuna})
        return scores["test_f2"].mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(lgbm_objective, n_trials=20, show_progress_bar=False)
    best_lgbm_params = study.best_params
    print(f"[Optuna] Best LightGBM params (F2={study.best_value:.4f}):")
    for k, v in best_lgbm_params.items():
        print(f"  {k}: {v}")

    optuna_results = pd.DataFrame([
        {**t.params, "f2_score": t.value} for t in study.trials
    ]).sort_values("f2_score", ascending=False)
    optuna_results.to_csv("outputs/08b_optuna_lgbm_tuning.csv", index=False)
    print("[Optuna] Full tuning log saved to outputs/08b_optuna_lgbm_tuning.csv")

#These are the model definitions. There are three models compared. The first is the Logistic Regression, which is balanced. It is the Phase 1 baseline. The second is the random forest, which is ensemble tree method. The third is the LightGBM, which is gradient boosting. It typically is the best on tabular clinical data
lr = Pipeline([
    ("sc", StandardScaler()),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE))
])

rf = Pipeline([
    ("sc", StandardScaler()),
    ("clf", RandomForestClassifier(
        n_estimators=300, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1
    ))
])

models = {"Logistic Regression": lr, "Random Forest": rf}

if LGB_AVAILABLE:
    scale_pos = (y_lung == 0).sum() / (y_lung == 1).sum()
    # Uses Optuna-tuned params if available, otherwise falls back to safe defaults
    lgbm = lgb.LGBMClassifier(
        **best_lgbm_params,
        scale_pos_weight=scale_pos, random_state=RANDOM_STATE,
        n_jobs=-1, verbose=-1
    )
    models["LightGBM"] = lgbm

print("Models defined:", list(models.keys()))


# This chunk is the Cross-Validated Evaluation, which is 5-fold. These prioritize metrics for imbalanced clinical data.  The *F2-score weighs the recall twice as much
#  over precision. False negatives are costly The Matthews Correlation Coefficient, or  MCC,
# is best single metric for imbalanced binary classification. The Recall is used sensitivity to minimize missed cancers
f2_scorer  = make_scorer(lambda yt, yp: fbeta_score(yt, yp, beta=2))
mcc_scorer = make_scorer(matthews_corrcoef)

SCORING = {
    "roc_auc": "roc_auc", "f2": f2_scorer, "mcc": mcc_scorer,
    "recall": "recall", "precision": "precision", "f1": "f1",
}

leaderboard_rows = []
cv_results_all   = {}

for name, model in models.items():
    print(f"  Training {name} ...")
    res = cross_validate(model, X_lung, y_lung, cv=CV, scoring=SCORING, return_train_score=False)
    cv_results_all[name] = res
    leaderboard_rows.append({
        "Model":     name,
        "ROC-AUC":   round(res["test_roc_auc"].mean(), 4),
        "F2":        round(res["test_f2"].mean(), 4),
        "MCC":       round(res["test_mcc"].mean(), 4),
        "Recall":    round(res["test_recall"].mean(), 4),
        "Precision": round(res["test_precision"].mean(), 4),
        "F1":        round(res["test_f1"].mean(), 4),
    })
    print(f"    ROC-AUC={res['test_roc_auc'].mean():.4f}  F2={res['test_f2'].mean():.4f}  "
          f"MCC={res['test_mcc'].mean():.4f}  Recall={res['test_recall'].mean():.4f}")

leaderboard = (pd.DataFrame(leaderboard_rows)
               .set_index("Model")
               .sort_values("F2", ascending=False))
leaderboard.to_csv("outputs/08_model_leaderboard.csv")
print("\nModel Leaderboard (sorted by F2):")
leaderboard

## This chunk does the model leaderboard visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

leaderboard[["ROC-AUC", "F2", "MCC", "Recall", "Precision"]].plot(
    kind="bar", ax=axes[0], rot=20, colormap="Set2", edgecolor="white")
axes[0].set_ylim(0, 1.05)
axes[0].set_title("Model Leaderboard — Key Metrics (5-fold CV)", fontsize=12)
axes[0].legend(loc="lower right", fontsize=9)
axes[0].axhline(0.8, color="grey", ls=":", lw=1)

leaderboard[["F2", "MCC"]].plot(kind="barh", ax=axes[1], colormap="coolwarm", edgecolor="white")
axes[1].set_title("F2-Score vs MCC (clinical priority metrics)", fontsize=12)
axes[1].set_xlim(0, 1.05)

plt.tight_layout()
fig.savefig("outputs/08_model_leaderboard_chart.png", dpi=150)
plt.show()

#  The chunk does the precision-recall curves for all Models.
# The Average Precision, which is the area under PR curve,
# is the most informative metric for imbalanced classification.

X_train, X_test, y_train, y_test = train_test_split(
    X_lung, y_lung, test_size=0.2, stratify=y_lung, random_state=RANDOM_STATE
)

fig, ax = plt.subplots(figsize=(9, 7))
colors = ["#2980b9", "#27ae60", "#e74c3c"]

for (name, model), color in zip(models.items(), colors):
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    ax.plot(rec, prec, lw=2, color=color, label=f"{name} (AP={ap:.3f})")

ax.axhline(y_test.mean(), color="grey", ls="--", lw=1, label="Random baseline")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves — All Models", fontsize=13)
ax.legend(fontsize=10)
plt.tight_layout()
fig.savefig("outputs/09_pr_curves_all_models.png", dpi=150)
plt.show()

# This chunk does the threshold sensitivity for the best model

#This Sweeps the classification threshold from 0.05 to 0.95.
# This plots how Precision, Recall, F2, and MCC change.
#  The clinical team can select the operating point.
best_model_name = leaderboard["F2"].idxmax()
best_model      = models[best_model_name]
print(f"Best model by F2: {best_model_name}")
best_model.fit(X_train, y_train)
y_prob_best = best_model.predict_proba(X_test)[:, 1]

thresholds = np.arange(0.05, 0.96, 0.01)
precisions, recalls, f2s, mccs = [], [], [], []

for t in thresholds:
    y_pred_t = (y_prob_best >= t).astype(int)
    precisions.append(precision_score(y_test, y_pred_t, zero_division=0))
    recalls.append(recall_score(y_test, y_pred_t, zero_division=0))
    f2s.append(fbeta_score(y_test, y_pred_t, beta=2, zero_division=0))
    mccs.append(matthews_corrcoef(y_test, y_pred_t))

best_f2_idx  = int(np.argmax(f2s))
best_mcc_idx = int(np.argmax(mccs))

fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(thresholds, precisions, label="Precision", color="#2980b9", lw=2)
ax.plot(thresholds, recalls,    label="Recall",    color="#e74c3c",  lw=2)
ax.plot(thresholds, f2s,        label="F2-Score",  color="#27ae60",  lw=2)
ax.plot(thresholds, mccs,       label="MCC",       color="#8e44ad",  lw=2, ls="--")
ax.axvline(thresholds[best_f2_idx],  color="#27ae60", ls=":", lw=1.5,
           label=f"Best F2 @ {thresholds[best_f2_idx]:.2f}")
ax.axvline(thresholds[best_mcc_idx], color="#8e44ad", ls=":", lw=1.5,
           label=f"Best MCC @ {thresholds[best_mcc_idx]:.2f}")
ax.set_xlabel("Classification Threshold"); ax.set_ylabel("Score")
ax.set_title(f"Threshold Sensitivity — {best_model_name}", fontsize=13)
ax.legend(fontsize=9); ax.set_ylim(0, 1.05)
plt.tight_layout()
fig.savefig("outputs/10_threshold_sensitivity.png", dpi=150)
plt.show()

print(f"Best threshold (F2) : {thresholds[best_f2_idx]:.2f}")
print(f"Best threshold (MCC): {thresholds[best_mcc_idx]:.2f}")


#This chunk does the confusion matrix at the optimal threshold
y_pred_optimal = (y_prob_best >= thresholds[best_f2_idx]).astype(int)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm = confusion_matrix(y_test, y_pred_optimal)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["Low Risk", "High Risk"],
            yticklabels=["Low Risk", "High Risk"])
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
axes[0].set_title(f"Confusion Matrix — {best_model_name}\nthreshold={thresholds[best_f2_idx]:.2f}")

report_dict = classification_report(y_test, y_pred_optimal,
                                     target_names=["Low Risk", "High Risk"],
                                     output_dict=True)
report_df = pd.DataFrame(report_dict).T.iloc[:2, :3]
sns.heatmap(report_df, annot=True, fmt=".3f", cmap="YlGn", ax=axes[1], vmin=0, vmax=1)
axes[1].set_title("Classification Report (per class)")

plt.tight_layout()
fig.savefig("outputs/11_confusion_matrix_best_model.png", dpi=150)
plt.show()
print("\nHeld-out test set report:")
print(f"  Test size : {len(y_test)} patients")
print(f"  Threshold : {thresholds[best_f2_idx]:.2f}")
print(classification_report(y_test, y_pred_optimal, target_names=["Low Risk", "High Risk"]))



# This chunk does the LightGBM Feature Importance

#The Built-in feature importance shows which features drive the most information gain across all trees. 
# Low importance and  high VIF makes strong candidate for further pruning.
if LGB_AVAILABLE:
    lgbm_fitted = models["LightGBM"]
    importance_df = pd.DataFrame({
        "feature":    features_clean,
        "importance": lgbm_fitted.feature_importances_
    }).sort_values("importance", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors_imp = ["#2980b9" if i < 10 else "#bdc3c7" for i in range(len(importance_df))]
    ax.barh(importance_df["feature"], importance_df["importance"], color=colors_imp)
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title("LightGBM Feature Importance", fontsize=13)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig("outputs/12_lgbm_feature_importance.png", dpi=150)
    plt.show()

    importance_df.to_csv("outputs/12_lgbm_feature_importance.csv", index=False)
    print(importance_df.to_string(index=False))
else:
    print("LightGBM not available. Install: pip install lightgbm")

# LightGBM Feature Selection — prune zero-importance features before Phase 3
# Features with zero gain across all trees carry no predictive signal and add noise.
# Only runs if LightGBM was available; otherwise features_clean is unchanged.
features_pruned = features_clean  # fallback: no pruning if LightGBM unavailable

if LGB_AVAILABLE:
    zero_importance = importance_df[importance_df["importance"] == 0]["feature"].tolist()
    features_pruned = [f for f in features_clean if f not in zero_importance]

    print(f"\nLightGBM Feature Selection:")
    print(f"  Before pruning : {len(features_clean)} features")
    print(f"  Zero-importance dropped : {len(zero_importance)} -> {zero_importance}")
    print(f"  After pruning  : {len(features_pruned)} features")

    # Rebuild X_lung and y_lung with the pruned feature set
    X_lung = lung_cancer_clean[features_pruned]

    pruning_report = pd.DataFrame({
        "feature":    features_clean,
        "importance": importance_df.set_index("feature").reindex(features_clean)["importance"].values,
        "kept":       ["Yes" if f in features_pruned else "No (zero importance)" for f in features_clean]
    }).sort_values("importance", ascending=False)
    pruning_report.to_csv("outputs/12b_lgbm_feature_selection.csv", index=False)
    print("  Selection report saved to outputs/12b_lgbm_feature_selection.csv")

    features_clean = features_pruned


#This saves phases 2 for phase 3
meta2 = {
    "best_model_name":    best_model_name,
    "best_threshold_f2":  round(float(thresholds[best_f2_idx]), 4),
    "best_threshold_mcc": round(float(thresholds[best_mcc_idx]), 4),
    "leaderboard":        leaderboard.reset_index().to_dict(orient="records"),
    "best_lgbm_params":   best_lgbm_params
}
with open("outputs/phase2_metadata.json", "w") as f:
    json.dump(meta2, f, indent=2)

print("Phase 2 complete.")
print(f"Best model : {best_model_name}")
print(f"Threshold  : {meta2['best_threshold_f2']}")



#PHASE 3
## 2. SHAP Explainer Setup
# SOLUTION: Use the same 80/20 train/test split from Phase 2. Compute SHAP on
#   X_test only. The CV metrics from Phase 2 remain your honest generalization estimate.
output_dir = "outputs/shap"
os.makedirs(output_dir, exist_ok=True)

# Re-use the Phase 2 train/test split for honest SHAP computation
X_train_shap, X_test_shap, y_train_shap, y_test_shap = train_test_split(
    X_lung, y_lung, test_size=0.2, stratify=y_lung, random_state=RANDOM_STATE
)

if best_model_name == "LightGBM" and LGB_AVAILABLE:
    scale_pos = (y_lung == 0).sum() / (y_lung == 1).sum()
    with open("outputs/phase2_metadata.json", "r") as f:
        _meta2 = json.load(f)
    _params = _meta2.get("best_lgbm_params", {
        "n_estimators": 500, "learning_rate": 0.05, "num_leaves": 63
    })
    model = lgb.LGBMClassifier(
        **_params,
        scale_pos_weight=scale_pos, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
    )
    model.fit(X_train_shap, y_train_shap)
    X_shap = X_test_shap  

elif best_model_name == "Random Forest":
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train_shap, y_train_shap)
    X_shap = X_test_shap

else:
    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X_train_shap)
    X_scaled_test  = scaler.transform(X_test_shap)
    X_shap = pd.DataFrame(X_scaled_test, columns=features_clean)
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
    model.fit(pd.DataFrame(X_scaled_train, columns=features_clean), y_train_shap)

# Keep y aligned with X_shap
y_shap = y_test_shap.reset_index(drop=True)
X_shap = X_shap.reset_index(drop=True)

print(f"[SHAP] Model fitted on {len(X_train_shap)} rows, explaining {len(X_shap)} held-out test rows.")

print(f"Model fitted: {best_model_name}  |  X_shap shape: {X_shap.shape}")
if SHAP_AVAILABLE:
    if best_model_name in ["LightGBM", "Random Forest"]:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    else:
        explainer   = shap.LinearExplainer(model, X_shap)
        shap_values = explainer.shap_values(X_shap)
        sv = shap_values

    base_val = explainer.expected_value
    if isinstance(base_val, list): base_val = base_val[1]
    print(f"SHAP values shape : {sv.shape}")
    print(f"Base rate (logit) : {base_val:.4f}")
else:
    print("SHAP not available.")
if SHAP_AVAILABLE:
    os.makedirs("app/model", exist_ok=True)
    joblib.dump(explainer, "app/model/shap_explainer.pkl")
    print("✅ SHAP explainer saved.")
def clinical_audit(patient_data, patient_shap, feature_names):
    feature_map = dict(zip(feature_names, patient_shap))
    flags = []

    if patient_data.get('smoker', 0) == 0 and feature_map.get('smoker', 0) > 0.1:
        flags.append("Anomaly: Non-smoker penalized for smoking status.")

    clinical_markers = ['chest_pain', 'shortness_of_breath', 'chronic_cough']
    if feature_map.get('fatigue', 0) > 0.2 and all(patient_data.get(m, 0) == 0 for m in clinical_markers):
        flags.append("Warning: Risk driven by fatigue without primary clinical symptoms.")

    if patient_data.get('age', 99) < 35 and patient_data.get('pack_years', 0) == 0:
        if feature_map.get('pack_years', 0) > 0.15:
            flags.append("Anomaly: Pack-years driving risk in young non-smoker.")

    if patient_data.get('copd', 0) == 0 and feature_map.get('fev1_x10', 0) > 0.2:
        flags.append("Flag: FEV1 is a major driver but COPD not diagnosed — verify lung function data.")

    env_vars = ['radon_exposure', 'air_pollution_index', 'occupational_exposure']
    if all(patient_data.get(v, 0) == 0 for v in env_vars) and \
       any(feature_map.get(v, 0) > 0.1 for v in env_vars):
        flags.append("Anomaly: Environmental features driving risk despite zero exposure values.")

    symptom_vars = ['chronic_cough', 'chest_pain', 'shortness_of_breath',
                    'fatigue', 'xray_abnormal']
    if all(patient_data.get(s, 0) == 0 for s in symptom_vars):
        flags.append("Note: High risk predicted with zero reported symptoms — verify input completeness.")

    return flags

# Execute the audit using your existing 'sv' and 'X_shap'
if SHAP_AVAILABLE:
    print("\n--- RUNNING CLINICAL LOGIC AUDIT ---")
    audit_reports = []

    for i in range(len(X_shap)):
        current_shap_values = sv[i]
        notes = clinical_audit(X_shap.iloc[i], current_shap_values, features_clean)
        if notes:
            audit_reports.append({"patient_row": i, "issues": notes})
            print(f"Patient {i} Flagged: {notes}")

    # Save this so it shows up in your Phase 3 Milestone Report
    with open("outputs/clinical_audit_log.json", "w") as f:
        json.dump(audit_reports, f, indent=4)
## 3. Global Explanation — SHAP Summary Dot Plot

#Shows which features matter most AND their direction. Red = high feature value pushes risk up. Blue = low feature value pushes risk down.
if SHAP_AVAILABLE:
    fig, ax = plt.subplots(figsize=(10, 9))
    shap.summary_plot(sv, X_shap, plot_type="dot", max_display=20, show=False)
    plt.title("SHAP Summary Plot — Global Feature Importance\n"
              "(color = feature value  |  x-axis = impact on predicted risk)",
              fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig("outputs/shap/13_shap_summary_dot.png", dpi=150, bbox_inches="tight")
    plt.show()

## 4. SHAP Bar Plot — Mean Absolute Impact
if SHAP_AVAILABLE:
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(sv, X_shap, plot_type="bar", max_display=20, show=False)
    plt.title("SHAP Feature Importance — Mean |SHAP value|", fontsize=12)
    plt.tight_layout()
    plt.savefig("outputs/shap/14_shap_summary_bar.png", dpi=150, bbox_inches="tight")
    plt.show()
## 5. Clinical vs Lifestyle Driver Analysis

#**Key question:** Are high-risk predictions driven more by clinical markers or lifestyle/environmental factors?
if SHAP_AVAILABLE:
    LIFESTYLE  = ["smoker","smoking_years","cigarettes_per_day","pack_years","passive_smoking",
                  "air_pollution_index","radon_exposure","occupational_exposure",
                  "exercise_hours_per_week","diet_quality","alcohol_units_per_week","bmi"]
    CLINICAL   = ["oxygen_saturation","fev1_x10","crp_level","xray_abnormal","copd",
                  "asthma","previous_tb","chronic_cough","chest_pain","shortness_of_breath","fatigue"]
    DEMOGRAPHIC= ["age","gender","education_years","income_level","healthcare_access",
                  "family_history_cancer"]

    mean_shap = pd.DataFrame({"feature": features_clean, "mean_shap": np.abs(sv).mean(axis=0)})

    def grp_total(feats):
        return mean_shap[mean_shap["feature"].isin(feats)]["mean_shap"].sum()

    ls_imp  = grp_total(LIFESTYLE)
    cl_imp  = grp_total(CLINICAL)
    dm_imp  = grp_total(DEMOGRAPHIC)
    total   = ls_imp + cl_imp + dm_imp

    print(f"Lifestyle   : {ls_imp/total*100:.1f}%")
    print(f"Clinical    : {cl_imp/total*100:.1f}%")
    print(f"Demographic : {dm_imp/total*100:.1f}%")

    fig, ax = plt.subplots(figsize=(7, 5))
    cats   = ["Lifestyle\nFactors", "Clinical\nMarkers", "Demographic"]
    vals   = [ls_imp, cl_imp, dm_imp]
    colors = ["#e74c3c", "#2980b9", "#27ae60"]
    bars   = ax.bar(cats, vals, color=colors, edgecolor="white", width=0.5)
    ax.set_ylabel("Total Mean |SHAP value|")
    ax.set_title("What Drives High-Risk Predictions?\nLifestyle vs Clinical vs Demographic", fontsize=12)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{val/total*100:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig("outputs/shap/15_lifestyle_vs_clinical_drivers.png", dpi=150)
    plt.show()
## 6. Local Explanation — SHAP Waterfall Plots for Edge Cases

#Three clinically interesting edge cases:
#1. Non-smoker flagged high risk (driven by environmental factors)
#2. Highest environmental exposure + high risk
#3. True high-risk patient with lowest model confidence
if SHAP_AVAILABLE:
    y_pred_prob = model.predict_proba(X_shap)[:, 1]

    # Edge case 1: non-smoker flagged high risk
    smoker_col = "smoker" if "smoker" in X_shap.columns else None
    if smoker_col:
        mask_ns = (X_shap[smoker_col] == 0).values
        cand_probs = y_pred_prob * mask_ns
        ec1_idx   = int(cand_probs.argmax())
    else:
        ec1_idx   = int(y_pred_prob.argmax())

    # Edge case 2: high environmental exposure + high risk
    env_cols  = [c for c in ["radon_exposure", "air_pollution_index"] if c in X_shap.columns]
    env_score = X_shap[env_cols].sum(axis=1).values if env_cols else y_pred_prob
    ec2_idx   = int((env_score * (y_pred_prob >= thresholds[best_f2_idx])).argmax())

    # Edge case 3: true high-risk with lowest predicted probability
    hr_idx = np.where(y_shap.values == 1)[0]
    ec3_idx   = int(hr_idx[y_pred_prob[hr_idx].argmin()])

    edge_cases = {
        "Non-smoker Flagged High Risk":           ec1_idx,
        "High Environmental Exposure + High Risk": ec2_idx,
        "True High Risk — Low Model Confidence":   ec3_idx,
    }

    for title, idx in edge_cases.items():
        print(f"\n{'='*55}")
        print(f"  {title}")
        print(f"  True label : {'HIGH RISK' if y_shap.iloc[idx]==1 else 'LOW RISK'}")
        print(f"  Pred prob  : {y_pred_prob[idx]:.3f}")

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=sv[idx], base_values=base_val,
                data=X_shap.iloc[idx].values, feature_names=features_clean
            ),
            max_display=15, show=False
        )
        plt.title(f"SHAP Waterfall — {title}\n"
                  f"True={'HIGH' if y_shap.iloc[idx]==1 else 'LOW'}  |  Prob={y_pred_prob[idx]:.3f}",
                  fontsize=11)
        plt.tight_layout()
        safe = title.replace(" ","_").replace("/","_")
        plt.savefig(f"outputs/shap/16_waterfall_{safe}.png", dpi=150, bbox_inches="tight")
        plt.show()
## 7. SHAP Dependence Plots — Top Feature Interactions

#For the top 3 features by mean |SHAP|, shows how the SHAP value changes across the feature range, colored by the most interacting second feature.
if SHAP_AVAILABLE:
    top3 = mean_shap.sort_values("mean_shap", ascending=False)["feature"].head(3).tolist()
    print(f"Top 3 features: {top3}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, feat in zip(axes, top3):
        shap.dependence_plot(feat, sv, X_shap, ax=ax, show=False, alpha=0.5)
        ax.set_title(f"SHAP Dependence: {feat}", fontsize=11)
    plt.tight_layout()
    fig.savefig("outputs/shap/17_shap_dependence_plots.png", dpi=150, bbox_inches="tight")
    plt.show()

## 8. Logic Check — SHAP vs Medical Literature

#Cross-reference the top SHAP features with known lung cancer risk factors from medical research to confirm the model is not learning noise.
if SHAP_AVAILABLE:
    MEDICAL_KNOWN = {
        "pack_years":          "Strong — dose-response well established (IARC, ACS)",
        "smoker":              "Strong — primary risk factor",
        "smoking_years":       "Strong — duration of exposure",
        "cigarettes_per_day":  "Strong — intensity of exposure",
        "crp_level":           "Moderate — systemic inflammation marker",
        "fev1_x10":            "Moderate — lung function decline (COPD link)",
        "oxygen_saturation":   "Moderate — functional impairment indicator",
        "xray_abnormal":       "Strong — direct radiological finding",
        "copd":                "Strong — COPD is an independent risk factor",
        "radon_exposure":      "Moderate-Strong — 2nd leading cause of lung cancer (EPA)",
        "air_pollution_index": "Moderate — PM2.5 particulate association",
        "age":                 "Strong — incidence rises sharply after age 55",
        "family_history_cancer": "Moderate — genetic predisposition",
        "asthma":              "Weak-Moderate — some evidence of elevated risk",
        "occupational_exposure": "Moderate — asbestos, silica, diesel fumes",
    }

    check = mean_shap.sort_values("mean_shap", ascending=False).reset_index(drop=True)
    check["rank"] = range(1, len(check) + 1)
    check["medical_alignment"] = check["feature"].map(MEDICAL_KNOWN).fillna("Minimal / demographic variable")
    print(check[["rank","feature","mean_shap","medical_alignment"]].to_string(index=False))
    check.to_csv("outputs/shap/18_shap_medical_alignment.csv", index=False)
## 9. Save Phase 3 Metadata for Phase 4
if SHAP_AVAILABLE:
    meta3 = {
        "shap_available": True,
        "top_10_features_by_shap": mean_shap.sort_values("mean_shap", ascending=False)["feature"].head(10).tolist(),
        "lifestyle_pct":   round(ls_imp/total*100, 2),
        "clinical_pct":    round(cl_imp/total*100, 2),
        "demographic_pct": round(dm_imp/total*100, 2),
        "edge_cases": list(edge_cases.keys()),
    }
else:
    meta3 = {"shap_available": False}

with open("outputs/phase3_metadata.json", "w") as f:
    json.dump(meta3, f, indent=2)

print("Phase 3 complete.")
print(f"SHAP outputs saved to outputs/shap/")
if SHAP_AVAILABLE:
    print(f"Top features: {meta3['top_10_features_by_shap'][:5]}")
os.makedirs("app/model", exist_ok=True)
joblib.dump(model, "app/model/final_model.pkl")
if SHAP_AVAILABLE:
    joblib.dump(explainer, "app/model/shap_explainer.pkl")
print("[INFO] Model and explainer saved to app/model/")
meta_path = "app/model/feature_meta.json"
existing = {}
if os.path.exists(meta_path):
    with open(meta_path) as f:
        existing = json.load(f)
existing["features"]        = list(X_lung.columns)
existing["threshold"]       = round(float(thresholds[best_f2_idx]), 4)
existing["best_model"]      = best_model_name
existing["average_patient"] = X_lung.mean().round(2).to_dict()
with open(meta_path, "w") as f:
    json.dump(existing, f, indent=2)

slider_cfg = {}
for feat in features_clean:
    col = X_lung[feat]
    is_binary = sorted(col.dropna().unique().tolist()) == [0, 1]
    slider_cfg[feat] = {
        "min":     int(col.min()),
        "max":     int(col.max()),
        "default": int(col.median()),
        "binary":  is_binary,
    }
with open("app/model/slider_config.json", "w") as f:
    json.dump(slider_cfg, f, indent=2)
print("[INFO] slider_config.json saved to app/model/")
