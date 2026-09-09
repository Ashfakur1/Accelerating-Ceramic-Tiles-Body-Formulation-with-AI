#!/usr/bin/env python3
"""
train_forward_model.py
Trains a multi-output ML forward model mapping ceramic tile composition
variables to mechanical and dimensional properties.

KEY DECISIONS & JUSTIFICATIONS
  [A] Lab batches always in test set — prevents data leakage into CV folds.
      The 8 experimental batches are reserved for held-out validation since
      they were fabricated independently and represent real production-scale
      compositions. Including them in training folds would constitute data
      leakage given that the synthetic dataset was generated from their
      measured properties.
      Ref: Pawlowsky-Glahn & Buccianti (2011) DOI:10.1002/9781119976462

  [B] Six candidate models evaluated via 5-fold cross-validation on the
      synthetic training set. The best-performing model (highest mean R²)
      is selected and retrained on the full synthetic training set.
      Native multi-output RandomForest preferred over MultiOutputRegressor
      where physically motivated — MOR, WA, and fired shrinkage are
      correlated via sintering density; assuming output independence
      underestimates target covariance. However, final model selection is
      CV-driven. If a wrapped model achieves superior cross-validated R²,
      it is retained on empirical grounds and the output-independence
      assumption is noted as a limitation.

  [C] PDP computed manually on ORIGINAL-SCALE X using the full pipeline.
      Passing scaled data to PartialDependenceDisplay yields standardised
      x-axes (z-score units) that are uninterpretable by practitioners.
      Manual computation calls pipeline.predict(X_modified) directly so
      x-values are naturally in wt%.

  [D] Single rectangular heatmap (composition features × targets) only.
      Process variables confirmed to have near-zero Pearson correlation
      with all three targets (|r| < 0.07) and are excluded from the
      heatmap.

  [E] Feature importance bar chart retains composition variables only,
      as process variables are held constant across all batches and carry
      no predictive signal.

  [F] Composition variables only are used as model features. Process
      variables (press_bar, kiln_temp_C, etc.) are held constant at the
      values recorded during fabrication of the 8 calibration batches and
      confirmed to show near-zero correlation with all three targets
      (|r| < 0.07). Including them would add noise without predictive
      benefit and would prevent deployment in settings where process
      parameters differ slightly from the calibration conditions.

NOTE — Experimental-batch performance metrics
  R² is undefined or misleading for n=8 held-out points drawn from a
  distribution shift relative to the synthetic training set. RMSE is
  reported instead for the experimental subset, as it carries the same
  physical units as the target and is robust to small sample sizes.
"""

import json, math, warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sns.set(style="whitegrid", context="talk", font_scale=1.1)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOTDIR  = Path(__file__).parent
DATADIR  = ROOTDIR / "data"
MODELDIR = ROOTDIR / "models"
PLOTDIR  = ROOTDIR / "plots"
for p in [MODELDIR, PLOTDIR]:
    p.mkdir(exist_ok=True, parents=True)

# ── Unified font scale ────────────────────────────────────────────────────────
_FS_TITLE = 18
_FS_AX    = 16
_FS_TICK  = 14
_FS_LABEL = 13
_FS_ANNOT = 12
_DPI      = 300

# ── Load data ─────────────────────────────────────────────────────────────────
df   = pd.read_csv(DATADIR / "dataset.csv")
with open(DATADIR / "metadata.json") as f:
    meta = json.load(f)

# Data-provenance stamp — carried over from generate_dataset.py so every
# figure in this script is tied to the SAME raw-data version as the figures
# produced upstream. If dataset.csv is regenerated from an updated CSV but
# this script is run against a stale copy (or vice versa), the stamps will
# visibly disagree instead of the mismatch hiding in the numbers alone.
DATA_HASH    = meta.get("data_hash", "unknown")
GENERATED_AT = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def _stamp(fig) -> None:
    fig.text(0.995, 0.002, f"data:{DATA_HASH}  generated:{GENERATED_AT}",
              ha="right", va="bottom", fontsize=6, color="0.6",
              family="monospace")

def savefig(fig, stem: str) -> None:
    _stamp(fig)
    for ext in ("pdf", "png"):
        fig.savefig(PLOTDIR / f"{stem}.{ext}", dpi=_DPI, bbox_inches="tight")
    plt.close(fig)

materials   = meta["materials"]
TARGET_COLS = ["MOR_MPa", "WA_pct", "Shrinkage_pct"]

TGT_LABELS = {
    "MOR_MPa":       "Firing MOR (MPa)",
    "WA_pct":        "Water Absorption (%)",
    "Shrinkage_pct": "Fired Shrinkage (%)",
}
MAT_SHORT = {
    "AG98_wtpct":     "AG98",
    "AG22_wtpct":     "AG22",
    "AG23_wtpct":     "AG23",
    "SodaF_wtpct":    "Soda Feldspar",
    "PotashF_wtpct":  "Potash Feldspar",
    "Crushing_wtpct": "Crushing",
    "ETP_wtpct":      "ETP Clay",
    "NaSil_wtpct":    "Na-Silicate",
}

# ── Feature selection: composition variables only (see Note [F]) ──────────────
feature_cols = [f"{m}_wtpct" for m in materials
                if f"{m}_wtpct" in df.columns]
comp_cols    = feature_cols

print(f"Features ({len(feature_cols)}): {feature_cols}")

X = df[feature_cols]
y = df[TARGET_COLS]

# ── Stratified split: lab batches always in test set (see [A]) ────────────────
lab_mask  = df["source"] == "lab_batch"
synth_idx = df[~lab_mask].index
lab_idx   = df[lab_mask].index

synth_train_idx, synth_test_idx = train_test_split(
    synth_idx, test_size=0.20, random_state=7
)
train_idx = synth_train_idx
test_idx  = synth_test_idx.tolist() + lab_idx.tolist()

X_train, X_test = X.loc[train_idx], X.loc[test_idx]
y_train, y_test = y.loc[train_idx].values, y.loc[test_idx].values

print(f"Train: {len(X_train)} synthetic | "
      f"Test:  {len(synth_test_idx)} synthetic + {len(lab_idx)} experimental")

preproc = ColumnTransformer(
    [("num", StandardScaler(), feature_cols)], remainder="drop"
)

# ── Input–output correlation heatmap ─────────────────────────────────────────
# Shows the Pearson correlation between each composition variable and each
# target property. Provides the scientific basis for understanding which
# raw materials drive which properties — essential context for interpreting
# the forward model and the inverse design outputs.
comp_only_X = X[comp_cols]
corr_io = pd.concat([comp_only_X, y], axis=1).corr().loc[comp_cols, TARGET_COLS]

corr_io_display = corr_io.copy()
corr_io_display.index   = [MAT_SHORT.get(c, c) for c in corr_io_display.index]
corr_io_display.columns = [TGT_LABELS.get(c, c) for c in corr_io_display.columns]
corr_io.to_csv(DATADIR / "input_output_correlation.csv")

fig, ax = plt.subplots(figsize=(11, 10))
sns.heatmap(corr_io_display,
            annot=True, fmt=".2f", cmap="coolwarm",
            center=0, vmin=-1, vmax=1,
            linewidths=0.5, annot_kws={"size": 16}, ax=ax)
ax.set_title(
    "Pearson Correlation Coefficients Between\n"
    "Composition Variables and Target Properties",
    pad=20, fontsize=20, fontweight="bold"
)
plt.xticks(rotation=45, ha="right", fontsize=16)
plt.yticks(rotation=0,  fontsize=16)
ax.collections[0].colorbar.ax.tick_params(labelsize=16)
plt.tight_layout()
savefig(fig, "input_output_correlation_heatmap")
print("Saved: input_output_correlation_heatmap.pdf / .png")

# ── Candidate models (see [B]) ────────────────────────────────────────────────
candidate_models: dict = {
    "RandomForest_native": RandomForestRegressor(
        n_estimators=500, random_state=7, n_jobs=-1
    ),
    "RandomForest_wrapped": MultiOutputRegressor(
        RandomForestRegressor(n_estimators=300, random_state=7, n_jobs=-1)
    ),
    "MLP": MultiOutputRegressor(
        MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=3000,
                     random_state=42, early_stopping=True,
                     n_iter_no_change=30)
    ),
}
for _lib, _cls, _name, _kw in [
    ("xgboost",  "XGBRegressor",      "XGB",
     {"n_estimators":500,"random_state":7,"n_jobs":-1,"verbosity":0}),
    ("lightgbm", "LGBMRegressor",     "LGBM",
     {"n_estimators":500,"random_state":7,"n_jobs":-1,
      "verbose":-1,"min_gain_to_split":0.0}),
    ("catboost", "CatBoostRegressor", "CatBoost",
     {"iterations":500,"verbose":0,"random_seed":7}),
]:
    try:
        candidate_models[_name] = MultiOutputRegressor(
            getattr(__import__(_lib), _cls)(**_kw)
        )
    except Exception:
        pass

# ── 5-fold CV on synthetic rows only ─────────────────────────────────────────
cv = KFold(n_splits=5, shuffle=True, random_state=7)
model_scores: dict[str, float] = {}
print("\nCross-validation (5-fold, synthetic rows only, R²):")
for name, model in candidate_models.items():
    try:
        pipe   = Pipeline([("preproc", preproc), ("reg", model)])
        scores = cross_val_score(pipe, X_train, y_train,
                                 cv=cv, scoring="r2", n_jobs=-1)
        model_scores[name] = float(np.mean(scores))
        print(f"  {name:<28s}  mean={model_scores[name]:.4f}  "
              f"std={np.std(scores):.4f}")
    except Exception as e:
        print(f"  {name:<28s}  skipped: {e}")

best_name = max(model_scores, key=model_scores.get)
print(f"\nBest model: {best_name}  (R² = {model_scores[best_name]:.4f})")

# ── Final training ────────────────────────────────────────────────────────────
final_model = Pipeline([
    ("preproc", preproc),
    ("reg",     candidate_models[best_name]),
])
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

mae = np.mean(np.abs(y_test - y_pred), axis=0)
r2  = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(3)]

print("\nTest-set performance (synthetic + experimental batches):")
for t, m, r in zip(TARGET_COLS, mae, r2):
    print(f"  {t:<25s}  MAE={m:.4f}  R²={r:.4f}")

# ── lab_mask_test: identify experimental rows in the test set ─────────────────
lab_idx_set   = set(lab_idx)
lab_mask_test = np.array([idx in lab_idx_set for idx in test_idx])

if lab_mask_test.sum() > 0:
    y_tl = y_test[lab_mask_test]
    y_pl = y_pred[lab_mask_test]
    print(f"\nExperimental-batch-only metrics (n={int(lab_mask_test.sum())}, held-out, "
          "R² not reported — see Note in module docstring):")
    units = {"MOR_MPa": "MPa", "WA_pct": "%", "Shrinkage_pct": "%"}
    for i, t in enumerate(TARGET_COLS):
        mae_e  = float(np.mean(np.abs(y_tl[:, i] - y_pl[:, i])))
        rmse_e = float(np.sqrt(np.mean((y_tl[:, i] - y_pl[:, i]) ** 2)))
        print(f"  {t:<25s}  MAE={mae_e:.4f} {units[t]}"
              f"  RMSE={rmse_e:.4f} {units[t]}")

# ── Save CSV artefacts ────────────────────────────────────────────────────────
pd.DataFrame({
    **{f"y_true_{t}": y_test[:, i] for i, t in enumerate(TARGET_COLS)},
    **{f"y_pred_{t}": y_pred[:, i] for i, t in enumerate(TARGET_COLS)},
}).to_csv(DATADIR / "parity_data.csv", index=False)

# ── Feature importance ────────────────────────────────────────────────────────
# Mean feature importance averaged across all three target properties.
# Composition variables only — process variables excluded (see Note [E]).
# Quantifies which raw materials most strongly determine the predicted
# properties, providing physical interpretability for the inverse design
# recommendations.
try:
    reg = final_model.named_steps["reg"]
    imp = (reg.feature_importances_
           if hasattr(reg, "feature_importances_")
           else np.mean([e.feature_importances_ for e in reg.estimators_], axis=0))

    # Normalise to sum = 1 so x-axis is always in [0, 1]
    imp = imp / imp.sum()
    fi = (pd.DataFrame({"feature": feature_cols, "importance": imp})
          .sort_values("importance", ascending=False))
    fi["label"] = fi["feature"].map(lambda c: MAT_SHORT.get(c, c))
    fi.to_csv(DATADIR / "feature_importances.csv", index=False)

    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = ["#E53935" if "_wtpct" in f else "#90A4AE" for f in fi["feature"]]
    sns.barplot(x="importance", y="label", data=fi,
                palette=palette, ax=ax, edgecolor="white")
    ax.legend(handles=[
        Patch(color="#E53935", label="Composition variable"),
    ], fontsize=_FS_LABEL, loc="lower right")
    ax.set_title("Mean Feature Importance Across All Target Properties",
                 pad=14, fontsize=_FS_TITLE, fontweight="bold")
    ax.set_xlabel("Importance", fontsize=_FS_AX)
    ax.set_ylabel("", fontsize=_FS_AX)
    ax.tick_params(labelsize=_FS_TICK)
    plt.tight_layout()
    savefig(fig, "feature_importances")
    print("Saved: feature_importances.pdf / .png")
except Exception as e:
    print(f"Feature importance: {e}")

# ── Parity plots ──────────────────────────────────────────────────────────────
# R² reported for the full test set (synthetic + experimental combined).
# For the experimental subset (n=8), RMSE in original units is reported
# instead of R², which is unreliable at small sample sizes with a
# distribution shift relative to the synthetic training set.
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
units = {"MOR_MPa": "MPa", "WA_pct": "%", "Shrinkage_pct": "%"}
for i, t in enumerate(TARGET_COLS):
    ax   = axes[i]
    smsk = ~lab_mask_test

    r2_combined = r2_score(y_test[:, i], y_pred[:, i])

    if lab_mask_test.sum() > 0:
        rmse_exp = float(np.sqrt(np.mean(
            (y_test[lab_mask_test, i] - y_pred[lab_mask_test, i]) ** 2
        )))
        exp_label = (f"RMSE (exp, n=8) = {rmse_exp:.3f} {units[t]}")
    else:
        exp_label = ""

    ax.scatter(y_test[smsk, i], y_pred[smsk, i],
               alpha=0.55, s=40, color="teal", label="Synthetic")
    if lab_mask_test.sum() > 0:
        ax.scatter(y_test[lab_mask_test, i], y_pred[lab_mask_test, i],
                   alpha=0.9, s=130, color="crimson", marker="*",
                   label="Experimental", zorder=5)
    lo = min(y_test[:, i].min(), y_pred[:, i].min())
    hi = max(y_test[:, i].max(), y_pred[:, i].max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5)
    ax.set_xlabel("Measured", fontsize=_FS_AX)
    ax.set_ylabel("Predicted", fontsize=_FS_AX)
    ax.tick_params(labelsize=_FS_TICK)
    ax.set_title(
        f"{TGT_LABELS[t]}\n"
        f"R² (all) = {r2_combined:.4f}  |  {exp_label}",
        fontsize=_FS_AX
    )
    ax.legend(fontsize=_FS_LABEL)

plt.suptitle("Predicted vs. Measured Values: Forward Model Parity Plots",
             fontsize=_FS_TITLE, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.93])
savefig(fig, "parity_plots")
print("Saved: parity_plots.pdf / .png")

# ── PDP Section ───────────────────────────────────────────────────────────────
# Partial dependence plots show how each target property responds to
# variation in a single composition variable, marginalised over the
# synthetic training set. X-values are in original wt% units (not
# standardised), making the plots directly interpretable by practitioners.
# See Note [C] for the rationale for manual PDP computation.
print(f"\nComputing PDPs on original-scale composition variables "
      f"(marginalised over synthetic training set, n = {len(X_train)}) …")

def compute_pdp(pipeline, X_orig: pd.DataFrame,
                feature: str, n_grid: int = 60):
    grid = np.linspace(X_orig[feature].min(), X_orig[feature].max(), n_grid)
    means = []
    for val in grid:
        Xt = X_orig.copy()
        Xt[feature] = val
        means.append(pipeline.predict(Xt).mean(axis=0))
    return grid, np.array(means)

_FS_PDP_SUPTITLE = 26
_FS_PDP_TITLE    = 24
_FS_PDP_LABEL    = 20
_FS_PDP_TICK     = 18

colors = ["#1565C0", "#D32F2F", "#388E3C"]

for t_idx, tname in enumerate(TARGET_COLS):
    n_comp = len(comp_cols)
    ncols  = 4
    nrows  = math.ceil(n_comp / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 7.2, nrows * 6.4),
                             dpi=250)
    axes = axes.flatten()

    for ax_idx, feat in enumerate(comp_cols):
        ax = axes[ax_idx]
        grid, means = compute_pdp(final_model, X_train, feat)

        ax.plot(grid, means[:, t_idx], color=colors[t_idx], linewidth=3.5)

        short = MAT_SHORT.get(feat, feat.replace("_wtpct", ""))
        ax.set_xlabel(f"{short} (wt%)", fontsize=_FS_PDP_LABEL, labelpad=10)
        ax.set_ylabel(TGT_LABELS[tname],  fontsize=_FS_PDP_LABEL, labelpad=10)
        ax.set_title(short, fontsize=_FS_PDP_TITLE, fontweight="bold", pad=15)
        ax.tick_params(labelsize=_FS_PDP_TICK, width=1.5, length=8)
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

    for ax in axes[n_comp:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Partial Dependence of {TGT_LABELS[tname]}\n"
        "on Composition Variables (wt%)\n"
        f"(marginalised over synthetic training set, n = {len(X_train)})",
        fontsize=_FS_PDP_SUPTITLE, fontweight="bold", y=0.96
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    savefig(fig, f"PDP_{tname}")
    print(f"  Saved: PDP_{tname}.pdf / .png")

# ── Save model ────────────────────────────────────────────────────────────────
joblib.dump(final_model, MODELDIR / "forward_model.joblib")
with open(MODELDIR / "feature_cols.json", "w") as f:
    json.dump(feature_cols, f, indent=2)

print("\nAll figures → plots/")
print("Model      → models/forward_model.joblib")
print("Done.")