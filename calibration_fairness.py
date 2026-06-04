"""
phase6_calibration_fairness.py
───────────────────────────────
Priority 1: Calibration Analysis   — Brier score, reliability diagram, isotonic vs sigmoid
Priority 2: Uncertainty language   — constants you paste into app/streamlit_app.py
Priority 3: Fairness/subgroup      — performance by smoker status, gender, age group

Run AFTER your main pipeline (pipeline.py) has already produced:
  app/model/final_model.pkl
  outputs/04_clean_dataset.csv
  app/model/feature_meta.json

Usage:
    python phase6_calibration_fairness.py
    python phase6_calibration_fairness.py --data outputs/04_clean_dataset.csv
"""

import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8')

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    brier_score_loss, roc_auc_score, f1_score, recall_score,
    precision_score, matthews_corrcoef, fbeta_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH   = "app/model/final_model.pkl"
META_PATH    = "app/model/feature_meta.json"
CLEAN_DATA   = "outputs/04_clean_dataset.csv"
TARGET       = "lung_cancer_risk"
RANDOM_STATE = 42

os.makedirs("outputs/calibration", exist_ok=True)
os.makedirs("outputs/fairness",    exist_ok=True)

# ── CLI override ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--data", default=CLEAN_DATA)
args, _ = parser.parse_known_args()
data_path = args.data

# ── Load artifacts ─────────────────────────────────────────────────────────────
print(f"[INFO] Loading model from  : {MODEL_PATH}")
print(f"[INFO] Loading dataset from: {data_path}")

model = joblib.load(MODEL_PATH)

df = pd.read_csv(data_path)

with open(META_PATH) as f:
    meta = json.load(f)

features  = meta["features"]
threshold = meta.get("threshold", 0.5)

X = df[features]
y = df[TARGET]

print(f"[INFO] Dataset shape : {X.shape}")
print(f"[INFO] Threshold     : {threshold:.4f}")
print(f"[INFO] Class balance : {y.value_counts().to_dict()}")

# NOTE: CalibratedClassifierCV with cv=5 nested inside cross_val_predict
# with cv=5 produces slightly optimistic calibration estimates due to
# partial overlap between calibration and evaluation folds.
# For a rigorous deployment audit, use a dedicated held-out calibration set.
# ═══════════════════════════════════════════════════════════════════════════════
# PRIORITY 1 — CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*60)
print("  PRIORITY 1: CALIBRATION ANALYSIS")
print("═"*60)

# ── 1a. Raw (uncalibrated) probabilities via 5-fold CV ────────────────────────
#   Using cross_val_predict so every row gets an out-of-fold prediction.
#   This avoids the optimism bias of predicting on training data.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

y_prob_raw = cross_val_predict(
    model, X, y, cv=cv, method="predict_proba"
)[:, 1]

brier_raw = brier_score_loss(y, y_prob_raw)
print(f"\n[Raw Model] Brier Score (lower=better, 0=perfect): {brier_raw:.4f}")
print(f"  Interpretation: {'Well calibrated (<0.10)' if brier_raw < 0.10 else 'Needs calibration (≥0.10)'}")

# ── 1b. Isotonic calibration ──────────────────────────────────────────────────
#   Use cross_val_predict (same cv as raw) to avoid data-leakage vs. raw OOF scores.
cal_isotonic = CalibratedClassifierCV(model, method="isotonic", cv=5)
y_prob_isotonic = cross_val_predict(
    cal_isotonic, X, y, cv=cv, method="predict_proba"
)[:, 1]
brier_isotonic  = brier_score_loss(y, y_prob_isotonic)
print(f"\n[Isotonic ] Brier Score : {brier_isotonic:.4f}")

# ── 1c. Sigmoid (Platt scaling) calibration ───────────────────────────────────
cal_sigmoid = CalibratedClassifierCV(model, method="sigmoid", cv=5)
y_prob_sigmoid = cross_val_predict(
    cal_sigmoid, X, y, cv=cv, method="predict_proba"
)[:, 1]
brier_sigmoid  = brier_score_loss(y, y_prob_sigmoid)
print(f"[Sigmoid  ] Brier Score : {brier_sigmoid:.4f}")

# ── 1d. Reliability diagram (calibration curve) ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#0d1117")

for ax in axes:
    ax.set_facecolor("#161b22")
    for spine in ax.spines.values():
        spine.set_color("#21262d")
    ax.tick_params(colors="#8b949e")

ax_cal, ax_brier = axes

# Perfect calibration reference line
ax_cal.plot([0, 1], [0, 1], "--", lw=1, color="#8b949e", label="Perfect calibration", alpha=0.7)

n_bins = 10
for label, probs, color in [
    ("Raw (uncalibrated)", y_prob_raw,      "#58a6ff"),
    ("Isotonic",           y_prob_isotonic, "#3fb950"),
    ("Sigmoid (Platt)",    y_prob_sigmoid,  "#d29922"),
]:
    frac_pos, mean_pred = calibration_curve(y, probs, n_bins=n_bins, strategy="uniform")
    ax_cal.plot(mean_pred, frac_pos, "s-", color=color, lw=2, markersize=6, label=label)

ax_cal.set_xlabel("Mean Predicted Probability", color="#8b949e", fontsize=10)
ax_cal.set_ylabel("Fraction of Positives",      color="#8b949e", fontsize=10)
ax_cal.set_title("Reliability Diagram\n(Calibration Curve)",
                 color="#e6edf3", fontsize=11, pad=10)
ax_cal.legend(facecolor="#161b22", edgecolor="#21262d",
              labelcolor="#e6edf3", fontsize=9)

# ── 1e. Brier score bar chart ─────────────────────────────────────────────────
methods = ["Raw", "Isotonic", "Sigmoid (Platt)"]
scores  = [brier_raw, brier_isotonic, brier_sigmoid]
colors  = ["#58a6ff", "#3fb950", "#d29922"]
bars    = ax_brier.bar(methods, scores, color=colors, edgecolor="#21262d", width=0.5)

# Annotation: perfect = 0, uninformative = 0.25
ax_brier.axhline(0.00, color="#3fb950", ls="--", lw=1, alpha=0.6, label="Perfect (0.00)")
ax_brier.axhline(0.25, color="#f85149", ls="--", lw=1, alpha=0.6, label="Uninformative (0.25)")
ax_brier.set_ylabel("Brier Score (lower = better)", color="#8b949e", fontsize=10)
ax_brier.set_title("Brier Score Comparison\nRaw vs. Calibrated Models",
                   color="#e6edf3", fontsize=11, pad=10)
ax_brier.set_ylim(0, 0.30)

for bar, score in zip(bars, scores):
    ax_brier.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f"{score:.4f}",
        ha="center", va="bottom", fontsize=10,
        fontweight="bold", color="#e6edf3"
    )

ax_brier.legend(facecolor="#161b22", edgecolor="#21262d",
                labelcolor="#e6edf3", fontsize=9)

plt.tight_layout()
fig.savefig("outputs/calibration/P1_calibration_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("[SAVED] outputs/calibration/P1_calibration_curves.png")

# ── 1f. Which calibration wins? ───────────────────────────────────────────────
best_method = min(
    [("Raw", brier_raw), ("Isotonic", brier_isotonic), ("Sigmoid (Platt)", brier_sigmoid)],
    key=lambda x: x[1]
)
print(f"\n✅ Best calibration: {best_method[0]} (Brier = {best_method[1]:.4f})")

# ── 1g. Save calibration summary ─────────────────────────────────────────────
cal_summary = {
    "brier_raw":       round(brier_raw,      4),
    "brier_isotonic":  round(brier_isotonic, 4),
    "brier_sigmoid":   round(brier_sigmoid,  4),
    "best_method":     best_method[0],
    "note": (
        "Brier score: 0.00=perfect, 0.25=uninformative (random). "
        "Clinical threshold: <0.10 considered well-calibrated."
    )
}
with open("outputs/calibration/P1_calibration_summary.json", "w") as f:
    json.dump(cal_summary, f, indent=2)
print("[SAVED] outputs/calibration/P1_calibration_summary.json")


# ═══════════════════════════════════════════════════════════════════════════════
# PRIORITY 2 — UNCERTAINTY LANGUAGE CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*60)
print("  PRIORITY 2: UNCERTAINTY LANGUAGE")
print("═"*60)

# These are the exact strings to paste into your streamlit_app.py.
# Search for "disclaimer" in that file and use these instead.

UNCERTAINTY_STRINGS = {
    "top_banner": (
        "⚠️  **Educational Demonstration Only** — "
        "This tool is a GWU Data Science portfolio project exploring XAI methodology. "
        "It is **not a diagnostic tool**, has not undergone clinical validation, "
        "and must not be used for patient care or treatment decisions."
    ),
    "risk_score_caption": (
        "Risk estimate based on statistical patterns in research data. "
        "Requires clinical validation before real-world use."
    ),
    "shap_caption": (
        "SHAP values show which features influenced this model's estimate — "
        "not causal clinical factors. Correlation ≠ causation."
    ),
    "score_tooltip": (
        "This probability is a model output, not a clinical diagnosis. "
        "Confidence intervals are not shown. "
        "Individual predictions carry inherent uncertainty."
    ),
    "footer_disclaimer": (
        "Research & Educational Use Only. "
        "Not a medical device. Trained on synthetic/research data. "
        "Not validated in clinical settings. "
        "Always consult a licensed medical professional."
    ),
}

print("\n[Uncertainty strings generated — copy these into streamlit_app.py]")
for key, val in UNCERTAINTY_STRINGS.items():
    print(f"\n  KEY: {key}")
    print(f"  → {val[:80]}...")

with open("outputs/calibration/P2_uncertainty_strings.json", "w") as f:
    json.dump(UNCERTAINTY_STRINGS, f, indent=2)
print("\n[SAVED] outputs/calibration/P2_uncertainty_strings.json")


# ═══════════════════════════════════════════════════════════════════════════════
# PRIORITY 3 — FAIRNESS / SUBGROUP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*60)
print("  PRIORITY 3: FAIRNESS / SUBGROUP ANALYSIS")
print("═"*60)

# Use out-of-fold probabilities for honest evaluation
y_pred_binary = (y_prob_raw >= threshold).astype(int)

def subgroup_metrics(mask, group_name, min_samples=50):
    """
    Returns a dict of metrics for the subset defined by boolean mask.
    Warns if the subgroup is too small to be meaningful.
    """
    n = mask.sum()
    _nan_row = {
        "Group": group_name, "N": int(n), "Prevalence": np.nan,
        "ROC-AUC": np.nan, "Recall": np.nan, "Precision": np.nan,
        "F2-Score": np.nan, "MCC": np.nan, "Brier": np.nan,
    }
    if n == 0:
        print(f"  ⚠️  {group_name}: 0 samples — skipping")
        return _nan_row
    if n < min_samples:
        print(f"  ⚠️  {group_name}: only {n} samples — results unreliable (need >={min_samples})")

    y_true_g = y.values[mask]
    y_prob_g  = y_prob_raw[mask]
    y_pred_g  = y_pred_binary[mask]

    # Guard: need both classes to compute AUC
    if len(np.unique(y_true_g)) < 2:
        auc = np.nan
    else:
        auc = roc_auc_score(y_true_g, y_prob_g)

    # Guard: MCC is undefined when true labels or predictions are all one class
    if len(np.unique(y_true_g)) < 2 or len(np.unique(y_pred_g)) < 2:
        mcc = np.nan
    else:
        mcc = matthews_corrcoef(y_true_g, y_pred_g)

    prevalence = y_true_g.mean()

    return {
        "Group":      group_name,
        "N":          int(n),
        "Prevalence": round(prevalence, 3),
        "ROC-AUC":    round(auc, 4) if not np.isnan(auc) else np.nan,
        "Recall":     round(recall_score(y_true_g, y_pred_g, zero_division=0), 4),
        "Precision":  round(precision_score(y_true_g, y_pred_g, zero_division=0), 4),
        "F2-Score":   round(fbeta_score(y_true_g, y_pred_g, beta=2, zero_division=0), 4),
        "MCC":        round(mcc, 4) if not np.isnan(mcc) else np.nan,
        "Brier":      round(brier_score_loss(y_true_g, y_prob_g), 4),
    }


rows = []

# ── 3a. Smoker vs Non-smoker ──────────────────────────────────────────────────
if "smoker" in df.columns:
    rows.append(subgroup_metrics(df["smoker"].values == 1, "Smoker"))
    rows.append(subgroup_metrics(df["smoker"].values == 0, "Non-Smoker"))
else:
    print("  [SKIP] 'smoker' column not found")

# ── 3b. Gender ────────────────────────────────────────────────────────────────
if "gender" in df.columns:
    # Assume 0=female, 1=male (common encoding; adjust if yours differs)
    rows.append(subgroup_metrics(df["gender"].values == 1, "Male   (gender=1)"))
    rows.append(subgroup_metrics(df["gender"].values == 0, "Female (gender=0)"))
else:
    print("  [SKIP] 'gender' column not found")

# ── 3c. Age groups ────────────────────────────────────────────────────────────
if "age" in df.columns:
    age = df["age"].values
    rows.append(subgroup_metrics(age < 50,              "Age < 50"))
    rows.append(subgroup_metrics((age >= 50) & (age < 65), "Age 50–64"))
    rows.append(subgroup_metrics(age >= 65,             "Age ≥ 65"))
else:
    print("  [SKIP] 'age' column not found")

# ── 3d. COPD (high-risk clinical subgroup) ────────────────────────────────────
if "copd" in df.columns:
    rows.append(subgroup_metrics(df["copd"].values == 1, "COPD = Yes"))
    rows.append(subgroup_metrics(df["copd"].values == 0, "COPD = No"))

fairness_df = pd.DataFrame(rows)
print("\n[Subgroup Performance Table]")
print(fairness_df.to_string(index=False))
fairness_df.to_csv("outputs/fairness/P3_subgroup_metrics.csv", index=False)
print("\n[SAVED] outputs/fairness/P3_subgroup_metrics.csv")

# ── 3e. Visualise subgroup Recall and ROC-AUC ─────────────────────────────────
# Recall is the metric that matters most clinically (missed cancers cost lives).
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#0d1117")
for ax in axes:
    ax.set_facecolor("#161b22")
    for spine in ax.spines.values():
        spine.set_color("#21262d")
    ax.tick_params(colors="#8b949e", labelsize=8)

# Filter to rows that have numeric AUC
plot_df = fairness_df[fairness_df["ROC-AUC"].notna()].copy()
plot_df["ROC-AUC"] = plot_df["ROC-AUC"].astype(float)

# Palette: colour by group type
group_colors = []
for g in plot_df["Group"]:
    if "Smok" in g:      group_colors.append("#58a6ff")
    elif "Male" in g or "Female" in g: group_colors.append("#d29922")
    elif "Age" in g:     group_colors.append("#3fb950")
    else:                group_colors.append("#f85149")

ax_recall, ax_auc = axes

# Recall chart
b1 = ax_recall.barh(plot_df["Group"], plot_df["Recall"],
                    color=group_colors, edgecolor="#21262d")
ax_recall.axvline(0.95, color="#f85149", ls="--", lw=1.2,
                  label="Target Recall (0.95)")
ax_recall.set_xlabel("Recall (Sensitivity)", color="#8b949e", fontsize=9)
ax_recall.set_title("Recall by Subgroup\n(Higher = fewer missed cancers)",
                    color="#e6edf3", fontsize=10, pad=8)
ax_recall.legend(facecolor="#161b22", edgecolor="#21262d",
                 labelcolor="#e6edf3", fontsize=8)
ax_recall.set_xlim(0, 1.05)
for bar, val in zip(b1, plot_df["Recall"]):
    ax_recall.text(min(val + 0.01, 1.0), bar.get_y() + bar.get_height() / 2,
                   f"{val:.3f}", va="center", fontsize=8, color="#e6edf3")

# ROC-AUC chart
b2 = ax_auc.barh(plot_df["Group"], plot_df["ROC-AUC"],
                 color=group_colors, edgecolor="#21262d")
ax_auc.axvline(0.85, color="#d29922", ls="--", lw=1.2,
               label="Acceptable AUC (0.85)")
ax_auc.set_xlabel("ROC-AUC", color="#8b949e", fontsize=9)
ax_auc.set_title("ROC-AUC by Subgroup\n(Discrimination ability)",
                 color="#e6edf3", fontsize=10, pad=8)
ax_auc.legend(facecolor="#161b22", edgecolor="#21262d",
              labelcolor="#e6edf3", fontsize=8)
ax_auc.set_xlim(0.5, 1.05)
for bar, val in zip(b2, plot_df["ROC-AUC"]):
    ax_auc.text(min(val + 0.005, 1.02), bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8, color="#e6edf3")

# Legend for group types
legend_patches = [
    mpatches.Patch(color="#58a6ff", label="Smoking status"),
    mpatches.Patch(color="#d29922", label="Gender"),
    mpatches.Patch(color="#3fb950", label="Age group"),
    mpatches.Patch(color="#f85149", label="COPD status"),
]
fig.legend(handles=legend_patches, loc="lower center", ncol=4,
           facecolor="#161b22", edgecolor="#21262d", labelcolor="#e6edf3",
           fontsize=8, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig("outputs/fairness/P3_subgroup_performance.png", dpi=150, bbox_inches="tight")
plt.show()
print("[SAVED] outputs/fairness/P3_subgroup_performance.png")

# ── 3f. Equity gap analysis ───────────────────────────────────────────────────
# Flag any subgroup where Recall drops more than 0.05 below the overall recall.
overall_recall = recall_score(y, y_pred_binary)
print(f"\n[Overall Recall] {overall_recall:.4f}")
print("\n[Equity Gap Check] (subgroups with Recall > 5pp below overall)")

equity_gaps = []
for _, row in fairness_df.iterrows():
    if isinstance(row["Recall"], (int, float)) and not np.isnan(row["Recall"]):
        gap = overall_recall - row["Recall"]
        if gap > 0.05:
            equity_gaps.append({
                "Group": row["Group"],
                "Recall": row["Recall"],
                "Gap vs Overall": round(gap, 4)
            })

if equity_gaps:
    gaps_df = pd.DataFrame(equity_gaps)
    print(gaps_df.to_string(index=False))
    print("  ⚠️  These subgroups have meaningfully lower recall — document this in your presentation.")
else:
    print("  ✅ No major equity gaps detected (all subgroups within 5pp of overall recall).")

gaps_out = equity_gaps if equity_gaps else [{"note": "No significant equity gaps detected."}]
with open("outputs/fairness/P3_equity_gaps.json", "w") as f:
    json.dump(gaps_out, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY REPORT
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*60)
print("  COMPLETE — OUTPUTS GENERATED")
print("═"*60)

summary = {
    "calibration": {
        "brier_raw":      cal_summary["brier_raw"],
        "brier_isotonic": cal_summary["brier_isotonic"],
        "brier_sigmoid":  cal_summary["brier_sigmoid"],
        "best_method":    cal_summary["best_method"],
        "plot":           "outputs/calibration/P1_calibration_curves.png",
    },
    "uncertainty_strings": "outputs/calibration/P2_uncertainty_strings.json",
    "fairness": {
        "subgroup_table": "outputs/fairness/P3_subgroup_metrics.csv",
        "plot":           "outputs/fairness/P3_subgroup_performance.png",
        "equity_gaps":    "outputs/fairness/P3_equity_gaps.json",
        "overall_recall": round(overall_recall, 4),
    },
    "note_on_priority_4": (
        "Feature stability analysis (bootstrap importance, fold-wise SHAP variance) "
        "is deferred. ROI is low for a June 2nd portfolio presentation. "
        "Add post-presentation if targeting research roles."
    )
}

with open("outputs/phase6_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("""
Files produced:
  outputs/calibration/P1_calibration_curves.png   ← reliability diagram + Brier bar chart
  outputs/calibration/P1_calibration_summary.json ← Brier scores for all three methods
  outputs/calibration/P2_uncertainty_strings.json ← exact copy-paste strings for Streamlit app
  outputs/fairness/P3_subgroup_metrics.csv         ← full metrics table by subgroup
  outputs/fairness/P3_subgroup_performance.png     ← Recall + AUC bar charts by subgroup
  outputs/fairness/P3_equity_gaps.json             ← groups flagged for fairness review
  outputs/phase6_summary.json                      ← master summary
""")
