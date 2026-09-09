#!/usr/bin/env python3
"""
streamlit_app.py
Ceramic tile composition inverse design tool.
Input: target mechanical/dimensional properties.
Output: recommended raw-material compositions via three methods.

SESSION STATE DESIGN
  Results from each method are stored in st.session_state so they persist
  across Streamlit reruns triggered by widget interactions (slider, number
  input, tab switch). Without session_state, any widget change wipes
  previously computed results because Streamlit re-executes the entire
  script on every interaction.

  Keys used:
    st.session_state["res_nn"]     - result dict from Method 1
    st.session_state["res_opt"]    - result dict from Method 2
    st.session_state["res_bay"]    - result dict from Method 3
    st.session_state["trial_vals"] - Bayesian convergence values
    st.session_state["identical"]  - bool flag from Method 2
    st.session_state["tgt_nn"]     - target dict at time of Method 1 run
    st.session_state["tgt_opt"]    - target dict at time of Method 2 run
    st.session_state["tgt_bay"]    - target dict at time of Method 3 run
    st.session_state["tgt_all"]    - target dict at time of Compare run
    st.session_state["res_all"]    - all three results for Compare tab

MODEL LOADING
  The forward model and dataset are loaded once per session via
  @st.cache_resource and @st.cache_data respectively. This avoids
  reloading the joblib model on every Streamlit rerun, which would
  otherwise add ~1-2 s latency to every widget interaction.

PRICE / CO2 ARCHITECTURE (new in this version)
  Raw-material prices and CO2 factors are NOT baked into the model or the
  training dataset. They live in dated CSV files under data/cost/ and
  data/co2/ (see price_loader.py) and are reloaded on demand via
  inverse_design.refresh_prices(). This app calls refresh_prices() right
  before every optimisation run, so the "Cost + CO2 Optimised" and
  "Bayesian Optimisation" tabs always use the latest price/CO2 snapshot
  without any retraining. The currently-loaded price/CO2 file names and
  dates are shown in the sidebar for transparency.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from inverse_design import (
    TARGET_COLS,
    clamp_targets,
    inverse_non_optimized,
    inverse_optimized,
    inverse_bayesian_optimization,
    refresh_prices,
    get_price_info,
    TGT_LABELS,
    MAT_SHORT,
)

# ── Unified font scale ──────────────────────────────────────────────────────
_FS_TITLE = 13
_FS_AX = 11
_FS_TICK = 10
_FS_LABEL = 10
_FS_ANNOT = 8

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATADIR = BASE_DIR / "data"

# ── Page config — must be first Streamlit command ───────────────────────────
st.set_page_config(layout="wide",
                    page_title="Ceramic Tile Composition Designer")

# ── File checks ──────────────────────────────────────────────────────────────
for req in ["dataset.csv", "metadata.json"]:
    if not (DATADIR / req).exists():
        st.error(f"Required file not found: {DATADIR / req}")
        st.stop()

@st.cache_data
def _load_dataset(_mtime: float) -> pd.DataFrame:
    return pd.read_csv(DATADIR / "dataset.csv")

@st.cache_data
def _load_meta(_mtime: float) -> dict:
    with open(DATADIR / "metadata.json") as f:
        return json.load(f)

# Passing each file's modification time as a cache-key argument means
# Streamlit automatically reloads the moment generate_dataset.py rewrites
# dataset.csv / metadata.json -- no manual "Clear cache" or app restart
# needed. If the file hasn't changed, the mtime argument is identical and
# the cached result is reused (no wasted re-parsing on every rerun).
dataset = _load_dataset((DATADIR / "dataset.csv").stat().st_mtime)
meta = _load_meta((DATADIR / "metadata.json").stat().st_mtime)
materials = meta["materials"]

# Dataset range from synthetic rows only — consistent with inverse_design.py
_ds_synth = dataset[dataset["source"] == "synthetic"]
MOR_min, MOR_max = float(_ds_synth["MOR_MPa"].min()), float(_ds_synth["MOR_MPa"].max())
WA_min, WA_max = float(_ds_synth["WA_pct"].min()), float(_ds_synth["WA_pct"].max())
Shrk_min, Shrk_max = float(_ds_synth["Shrinkage_pct"].min()), float(_ds_synth["Shrinkage_pct"].max())

# ── ISO 13006 water-absorption classification (computed from the ACTUAL
#    synthetic dataset range, not hard-coded — so it never goes stale when
#    lab_batches_raw.csv is edited and the pipeline is re-run) ────────────────
def _iso13006_wa_class(wa_min: float, wa_max: float) -> str:
    """
    Map a water-absorption range (%) to the ISO 13006 annex group(s) it
    spans: BIa (<=0.5%), BIb (0.5-3%), BIIa (3-6%), BIIb (6-10%), BIII (>10%).
    Returns e.g. "BIb" or "BIa/BIb" if the range straddles a boundary.
    """
    groups = [
        (0.0, 0.5, "BIa"),
        (0.5, 3.0, "BIb"),
        (3.0, 6.0, "BIIa"),
        (6.0, 10.0, "BIIb"),
        (10.0, float("inf"), "BIII"),
    ]
    spanned = [name for lo, hi, name in groups if wa_max > lo and wa_min < hi]
    return "/".join(spanned) if spanned else "BIII"

_WA_CLASS = _iso13006_wa_class(WA_min, WA_max)

# ── Initialise session state ────────────────────────────────────────────────
for key in ["res_nn", "res_opt", "res_bay", "trial_vals",
            "identical", "tgt_nn", "tgt_opt", "tgt_bay",
            "tgt_all", "res_all"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ── Title ────────────────────────────────────────────────────────────────────
st.title("Ceramic Tile Inverse Composition Design")
st.caption(
    "Physics-informed surrogate model trained on 8 experimental batches "
    "+ 1,000 synthetic samples. AKIJ Ceramics Ltd., Bangladesh."
)

# ── Model Scope & Limitations ────────────────────────────────────────────────
with st.expander("Model Scope & Limitations (read before use)",
                  expanded=False):
    st.markdown("""
**Training scope:** This model was developed for a single homogeneous floor
tile body at laboratory scale (109.2 x 54.6 mm green dimensions, 100 bar pressing
pressure, 1210 C kiln temperature, 90 min kiln cycle) at AKIJ Ceramics Ltd.,
Bangladesh.

**Raw material specificity:** Predictions are calibrated to the chemical and
physical characteristics of raw materials sourced by AKIJ Ceramics Ltd.
Substituting materials from different suppliers may shift property responses
in ways the model cannot anticipate.

**Extrapolation risk:** Compositions outside the training ranges listed below
are extrapolated. Model accuracy degrades with increasing distance from the
calibration space.

**Price / CO2 architecture:** Raw-material prices and CO2 emission factors
are stored OUTSIDE the training dataset, in dated CSV snapshots
(data/cost/, data/co2/). The prediction model (Composition -> Properties)
never needs retraining when prices change — only the optimiser re-reads the
latest snapshot. See the sidebar for the currently loaded price/CO2 dates.

**Industrial application:** For an industrial facility seeking to apply this
framework, the approach requires constructing a dataset from real laboratory
or production data corresponding to the specific tile type and surface finish
of interest, while keeping the relevant process parameters fixed. The framework
should then be re-trained on that facility-specific dataset. Prediction accuracy
improves proportionally with the volume and representativeness of the real data
provided.

| Material | Min (wt%) | Max (wt%) |
|----------|-----------|-----------|
| AG98 (High Plastic Clay) | 15.0 | 20.0 |
| AG22 (Low Plastic Clay)  |  2.5 |  4.0 |
| AG23 (Semi-Plastic Clay) | 10.0 | 15.0 |
| Soda Feldspar            | 37.0 | 43.0 |
| Potash Feldspar          | 15.0 | 22.0 |
| Crushing                 |  2.0 |  3.5 |
| ETP Clay                 |  2.0 |  3.1 |
| Sodium Silicate          |  0.5 |  1.5 |
    """)

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR — fixed process parameters + live price/CO2 database status
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("Process Parameters")
    st.caption(
        "Fixed at the values used during fabrication of the 8 calibration "
        "batches. Displayed for reproducibility - do not vary."
    )
    for label, val in [
        ("Pressing pressure (bar)", 100),
        ("Dryer residence time (min)", 45),
        ("Kiln residence time (min)", 90),
        ("Kiln temperature (C)", 1210),
        ("Gas calorific value (Kcal/Nm3)", 8300),
    ]:
        st.number_input(label, value=float(val), disabled=True)

    st.divider()
    st.subheader("Typical operating midpoints")
    for label, val in [
        ("Dryer temperature (C)", 180.0),
        ("Green tile length (mm)", 109.20),
        ("Green tile width (mm)", 54.60),
        ("Green tile thickness (mm)", 9.80),
        ("Green tile weight (g)", 98.50),
        ("Fired tile length (mm)", 98.00),
        ("Fired tile weight (g)", 95.05),
        ("Gas consumption (Nm3/m2)", 1.41),
    ]:
        st.number_input(label, value=val, disabled=True)

    st.divider()
    st.subheader("Raw-Material Price / CO2 Database")
    st.caption(
        "Cost and CO2 factors are read from the LATEST dated file in "
        "data/cost/ and data/co2/. Drop in a new dated CSV and click "
        "refresh - no retraining needed."
    )
    if st.button("Refresh prices now", key="refresh_prices_btn"):
        refresh_prices()
        st.success("Price / CO2 tables reloaded.")

    _pinfo = get_price_info()
    for _label, _key in [("Cost table", "cost"), ("CO2 table", "co2")]:
        _entry = _pinfo.get(_key, {})
        _file = _entry.get("file")
        _as_of = _entry.get("as_of")
        if _file is not None:
            st.write(f"**{_label}:** `{Path(_file).name}`")
            st.write(f"as of: {_as_of.date() if _as_of else 'unknown'}")
        else:
            st.write(f"**{_label}:** using fallback (metadata.json)")

# Always start from the latest price/CO2 snapshot when the app (re)loads.
refresh_prices()

# ═══════════════════════════════════════════════════════════════════════════
# TARGET PROPERTIES INPUT
# ═══════════════════════════════════════════════════════════════════════════
st.subheader("Target Properties")
c1, c2, c3 = st.columns(3)
with c1:
    MOR_MPa = st.number_input(
        f"Firing MOR (MPa)  [{MOR_min:.1f} - {MOR_max:.1f}]",
        min_value=MOR_min, max_value=MOR_max,
        value=round((MOR_min + MOR_max) / 2, 1),
        step=0.5, format="%.1f",
    )
with c2:
    WA_pct = st.number_input(
        f"Water Absorption (%)  [{WA_min:.2f} - {WA_max:.2f}]",
        min_value=WA_min, max_value=WA_max,
        value=round((WA_min + WA_max) / 2, 2),
        step=0.01, format="%.2f",
        help=(
            "Water absorption as a percentage (not fraction). "
            f"Dataset range ({WA_min:.2f}-{WA_max:.2f}%) falls in ISO 13006 "
            f"Class {_WA_CLASS}. Recomputed automatically from the current "
            "lab_batches_raw.csv / dataset.csv — will update if the "
            "calibration batches change."
        ),
    )
with c3:
    Shrinkage_pct = st.number_input(
        f"Fired Shrinkage (%)  [{Shrk_min:.1f} - {Shrk_max:.1f}]",
        min_value=Shrk_min, max_value=Shrk_max,
        value=round((Shrk_min + Shrk_max) / 2, 1),
        step=0.1, format="%.1f",
    )

# ── Method tabs ──────────────────────────────────────────────────────────────
st.subheader("Inverse Design Method")
tabs = st.tabs([
    "Non-Optimised (NN)",
    "Cost + CO2 Optimised (NN)",
    "Bayesian Optimisation",
    "Compare All Methods",
])

# ── Shared helpers ───────────────────────────────────────────────────────────
def _table(result: dict) -> pd.DataFrame:
    comp_renamed = {MAT_SHORT.get(m, m): v
                    for m, v in result["composition_wtpct"].items()}
    return pd.DataFrame([{
        **comp_renamed,
        **{TGT_LABELS.get(k, k): v for k, v in result["predicted"].items()},
        "Cost (Tk/kg)": result["cost_Tk_per_kg"],
        "CO2 (kg/kg)": result["CO2_kg_per_kg"],
    }])

def _comp_bar(result: dict, title: str, color: str):
    comp = result["composition_wtpct"]
    labels = [MAT_SHORT.get(m, m) for m in comp.keys()]
    vals = list(comp.values())

    fig, ax = plt.subplots(figsize=(7, 3.8))
    bars = ax.bar(range(len(labels)), vals, color=color,
                  alpha=0.85, edgecolor="white")
    y_max = max(vals) if vals else 1.0
    for b in bars:
        h = b.get_height()
        if h > y_max * 0.12:
            ax.text(b.get_x() + b.get_width() / 2, h * 0.5,
                    f"{h:.1f}",
                    ha="center", va="center",
                    fontsize=_FS_ANNOT, fontweight="bold",
                    color="white")
        else:
            ax.text(b.get_x() + b.get_width() / 2, h + y_max * 0.01,
                    f"{h:.1f}",
                    ha="center", va="bottom",
                    fontsize=_FS_ANNOT, fontweight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=_FS_TICK)
    ax.set_ylabel("Weight (%)", fontsize=_FS_AX)
    ax.tick_params(axis="y", labelsize=_FS_TICK)
    ax.set_title(title, fontsize=_FS_AX, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax.set_ylim(0, y_max * 1.15)
    plt.tight_layout()
    return fig

def _error_bars(result: dict, tgt: dict, title: str, color: str):
    """Absolute prediction error vs. target for each property."""
    errors = {p: abs(result["predicted"][p] - tgt[p]) for p in TARGET_COLS}
    labels = [TGT_LABELS.get(p, p) for p in TARGET_COLS]
    fmt = {"MOR_MPa": ".2f", "WA_pct": ".4f", "Shrinkage_pct": ".3f"}
    units = {"MOR_MPa": "MPa", "WA_pct": "%", "Shrinkage_pct": "%"}

    fig, ax = plt.subplots(figsize=(5, 3.8))
    bars = ax.bar(labels, errors.values(), color=color,
                  alpha=0.85, edgecolor="white")
    y_max = max(errors.values()) if max(errors.values()) > 0 else 1.0
    offset = y_max * 0.04
    for b, (p, err) in zip(bars, errors.items()):
        ax.text(b.get_x() + b.get_width() / 2,
                b.get_height() + offset,
                f"{format(err, fmt[p])}\n{units[p]}",
                ha="center", va="bottom",
                fontsize=_FS_ANNOT, fontweight="bold")
    ax.set_ylabel("Absolute Error", fontsize=_FS_AX)
    ax.tick_params(labelsize=_FS_TICK)
    ax.set_title(title, fontsize=_FS_AX, fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax.set_ylim(0, y_max * 1.30)
    plt.tight_layout()
    return fig

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Non-Optimised
# ═══════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown(
        "Returns the single dataset sample **closest to the target** in "
        "scaled property space (equal weight for MOR, WA, and Shrinkage). "
        "No cost or CO2 consideration. "
        "This method is a **baseline** - it makes no attempt to minimise "
        "cost or environmental impact."
    )
    if st.button("Run Non-Optimised", key="run_nn"):
        MOR_c, WA_c, SH_c = clamp_targets(MOR_MPa, WA_pct, Shrinkage_pct)
        tgt_now = {"MOR_MPa": MOR_c, "WA_pct": WA_c, "Shrinkage_pct": SH_c}
        st.session_state["tgt_nn"] = tgt_now
        with st.spinner("Searching dataset..."):
            st.session_state["res_nn"] = inverse_non_optimized(MOR_c, WA_c, SH_c)

    if st.session_state["res_nn"] is not None:
        res = st.session_state["res_nn"]
        tgt = st.session_state["tgt_nn"]
        st.markdown("#### Recommended Composition")
        st.dataframe(_table(res), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            fig = _comp_bar(res, "Raw Material Composition (wt%)", "#E24A33")
            st.pyplot(fig)
            plt.close(fig)
        with c2:
            fig = _error_bars(res, tgt, "Absolute Error vs. Target", "#E24A33")
            st.pyplot(fig)
            plt.close(fig)
        col_a, col_b = st.columns(2)
        col_a.metric("Batch cost", f"{res['cost_Tk_per_kg']:.4f} Tk/kg")
        col_b.metric("CO2 emission", f"{res['CO2_kg_per_kg']:.5f} kg/kg")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Cost + CO2 Optimised
# ═══════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown(
        "Searches the **10 nearest neighbours** in scaled property space and "
        "selects the composition with the lowest combined cost + CO2 rank, "
        "using the **latest** price/CO2 database. "
        "If this returns the same result as the Non-Optimised method, the "
        "10-nearest neighbourhood lacks sufficient compositional diversity "
        "for this target - a known limitation of dataset-lookup inverse design."
    )
    if st.button("Run Cost + CO2 Optimised", key="run_opt"):
        refresh_prices()  # always use the latest cost/CO2 snapshot
        MOR_c, WA_c, SH_c = clamp_targets(MOR_MPa, WA_pct, Shrinkage_pct)
        tgt_now = {"MOR_MPa": MOR_c, "WA_pct": WA_c, "Shrinkage_pct": SH_c}
        st.session_state["tgt_opt"] = tgt_now
        with st.spinner("Optimising among 10 neighbours..."):
            res_opt, identical = inverse_optimized(MOR_c, WA_c, SH_c)
        st.session_state["res_opt"] = res_opt
        st.session_state["identical"] = identical

    if st.session_state["res_opt"] is not None:
        res = st.session_state["res_opt"]
        tgt = st.session_state["tgt_opt"]
        identical = st.session_state["identical"]

        if identical:
            st.warning(
                "Methods 1 and 2 returned the **same composition**. "
                "The 10 nearest neighbours in property space share similar "
                "cost and CO2 profiles for this target. "
                "Consider using Bayesian Optimisation (Tab 3) for a "
                "composition that actively minimises cost + CO2 across the "
                "full feasible space."
            )

        st.markdown("#### Recommended Composition")
        st.dataframe(_table(res), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            fig = _comp_bar(res, "Raw Material Composition (wt%)", "#348ABD")
            st.pyplot(fig)
            plt.close(fig)
        with c2:
            fig = _error_bars(res, tgt, "Absolute Error vs. Target", "#348ABD")
            st.pyplot(fig)
            plt.close(fig)
        col_a, col_b = st.columns(2)
        col_a.metric("Batch cost", f"{res['cost_Tk_per_kg']:.4f} Tk/kg")
        col_b.metric("CO2 emission", f"{res['CO2_kg_per_kg']:.5f} kg/kg")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — Bayesian Optimisation
# ═══════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown(
        "Searches the **continuous composition space** using Optuna Bayesian "
        "optimisation to minimise normalised cost + normalised CO2 subject to "
        "meeting the target properties, using the **latest** price/CO2 "
        "database. Unlike the nearest-neighbour methods, Bayesian "
        "optimisation can identify compositions not present in the "
        "dataset, enabling genuinely improved cost and CO2 performance. "
        "More accurate than nearest-neighbour but slower (~10-30 s). "
        "The objective function is dimensionless: cost and CO2 are each "
        "normalised by their dataset range before summation with the "
        "property-matching penalty."
    )
    n_trials = st.slider("Optimisation trials", 50, 300, 200, 50)

    if st.button("Run Bayesian Optimisation", key="run_bayes"):
        refresh_prices()  # always use the latest cost/CO2 snapshot
        MOR_c, WA_c, SH_c = clamp_targets(MOR_MPa, WA_pct, Shrinkage_pct)
        tgt_now = {"MOR_MPa": MOR_c, "WA_pct": WA_c, "Shrinkage_pct": SH_c}
        st.session_state["tgt_bay"] = tgt_now
        with st.spinner(f"Running {n_trials} trials..."):
            res_bay, trial_vals, _ = inverse_bayesian_optimization(
                MOR_c, WA_c, SH_c, n_trials=n_trials
            )
        st.session_state["res_bay"] = res_bay
        st.session_state["trial_vals"] = trial_vals

    if st.session_state["res_bay"] is not None:
        res = st.session_state["res_bay"]
        tgt = st.session_state["tgt_bay"]
        trial_vals = st.session_state["trial_vals"]

        st.markdown("#### Recommended Composition")
        st.dataframe(_table(res), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            fig = _comp_bar(res, "Raw Material Composition (wt%)", "#8EBA42")
            st.pyplot(fig)
            plt.close(fig)
        with c2:
            fig = _error_bars(res, tgt, "Absolute Error vs. Target", "#8EBA42")
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("#### Convergence")
        fig, ax = plt.subplots(figsize=(8, 3.8))
        t = np.arange(1, len(trial_vals) + 1)
        ax.plot(t, trial_vals,
                color="#BDBDBD", lw=1.0, alpha=0.7, label="Trial objective")
        ax.plot(t, np.minimum.accumulate(trial_vals),
                color="#E53935", lw=2.2, label="Best so far")
        ax.set_xlabel("Trial", fontsize=_FS_AX)
        ax.set_ylabel(
            "Objective\n(norm. cost + norm. CO2 + penalty)",
            fontsize=_FS_AX,
        )
        ax.tick_params(labelsize=_FS_TICK)
        ax.set_title(
            "Convergence of the Bayesian Optimisation Objective Function",
            fontsize=_FS_TITLE, fontweight="bold",
        )
        ax.legend(fontsize=_FS_LABEL)
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        col_a, col_b = st.columns(2)
        col_a.metric("Batch cost", f"{res['cost_Tk_per_kg']:.4f} Tk/kg")
        col_b.metric("CO2 emission", f"{res['CO2_kg_per_kg']:.5f} kg/kg")

        wa_pred = res["predicted"]["WA_pct"]
        wa_tgt = tgt["WA_pct"]
        if wa_pred < wa_tgt - 0.05:
            st.info(
                f"**Water Absorption note:** Predicted WA ({wa_pred:.3f}%) "
                f"is below the target ({wa_tgt:.3f}%). This is **intentional** - "
                "the Bayesian objective penalises WA *over-achievement* only "
                "(lower WA = lower porosity = better frost resistance, "
                f"ISO 13006 Class {_WA_CLASS}). "
                "The composition satisfies the water absorption specification."
            )

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — Compare All Methods
# ═══════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown(
        "Run all three methods and compare compositions, predicted properties, "
        "cost, and CO2 side by side, using the **latest** price/CO2 database. "
        "Cost and CO2 reductions are reported relative to the "
        "Non-Optimised (NN) baseline."
    )
    st.info(
        "This tab runs all three methods simultaneously on the same target "
        "values for a consistent side-by-side comparison."
    )
    n_trials_cmp = st.slider("Bayesian trials (comparison)", 50, 300, 200, 50)

    if st.button("Compare All Methods", key="run_all", type="primary"):
        refresh_prices()  # always use the latest cost/CO2 snapshot
        MOR_c, WA_c, SH_c = clamp_targets(MOR_MPa, WA_pct, Shrinkage_pct)
        tgt_now = {"MOR_MPa": MOR_c, "WA_pct": WA_c, "Shrinkage_pct": SH_c}
        st.session_state["tgt_all"] = tgt_now
        with st.spinner("Running all methods..."):
            res_nn = inverse_non_optimized(MOR_c, WA_c, SH_c)
            res_opt, identical = inverse_optimized(MOR_c, WA_c, SH_c)
            res_bay, _, _ = inverse_bayesian_optimization(
                MOR_c, WA_c, SH_c, n_trials=n_trials_cmp
            )
        st.session_state["res_all"] = {
            "Non-Optimised (NN)": res_nn,
            "Cost + CO2 Optimised (NN)": res_opt,
            "Bayesian Optimisation": res_bay,
        }
        st.session_state["identical"] = identical

    if st.session_state["res_all"] is not None:
        results = st.session_state["res_all"]
        tgt = st.session_state["tgt_all"]
        identical = st.session_state["identical"]
        colors = ["#E24A33", "#348ABD", "#8EBA42"]

        if identical:
            st.warning(
                "Non-Optimised (NN) and Cost + CO2 Optimised (NN) returned "
                "the **same composition**. See Tab 2 for explanation."
            )

        # ── Comparison table ─────────────────────────────────────────────
        st.subheader("Recommended Compositions")
        cmp_df = pd.DataFrame({
            name: {
                **{MAT_SHORT.get(m, m): v
                   for m, v in res["composition_wtpct"].items()},
                **{TGT_LABELS.get(k, k): v
                   for k, v in res["predicted"].items()},
                "Cost (Tk/kg)": res["cost_Tk_per_kg"],
                "CO2 (kg/kg)": res["CO2_kg_per_kg"],
            }
            for name, res in results.items()
        }).T
        st.dataframe(cmp_df, use_container_width=True)

        # ── Cost & CO2 % reduction vs Non-Optimised baseline ────────────
        st.subheader("Cost & CO2 Reduction vs. Non-Optimised Baseline")
        baseline_cost = results["Non-Optimised (NN)"]["cost_Tk_per_kg"]
        baseline_co2 = results["Non-Optimised (NN)"]["CO2_kg_per_kg"]

        reduction_rows = []
        for name, res in results.items():
            cost_chg = (res["cost_Tk_per_kg"] - baseline_cost) / baseline_cost * 100
            co2_chg = (res["CO2_kg_per_kg"] - baseline_co2) / baseline_co2 * 100
            reduction_rows.append({
                "Method": name,
                "Cost (Tk/kg)": res["cost_Tk_per_kg"],
                "Cost Change (%)": round(cost_chg, 2),
                "CO2 (kg/kg)": res["CO2_kg_per_kg"],
                "CO2 Change (%)": round(co2_chg, 2),
            })
        red_df = pd.DataFrame(reduction_rows).set_index("Method")
        st.dataframe(red_df, use_container_width=True)

        bay_cost_chg = red_df.loc["Bayesian Optimisation", "Cost Change (%)"]
        bay_co2_chg = red_df.loc["Bayesian Optimisation", "CO2 Change (%)"]
        mc1, mc2 = st.columns(2)
        mc1.metric(
            label="Bayesian - Cost change vs. baseline",
            value=f"{bay_cost_chg:+.2f} %",
            delta=f"{bay_cost_chg:.2f}%",
            delta_color="inverse",
        )
        mc2.metric(
            label="Bayesian - CO2 change vs. baseline",
            value=f"{bay_co2_chg:+.2f} %",
            delta=f"{bay_co2_chg:.2f}%",
            delta_color="inverse",
        )

        # ── Fig 1: Composition bar chart ─────────────────────────────────
        st.subheader("Fig. 1 - Raw Material Composition (wt%)")
        comp_df = pd.DataFrame({n: r["composition_wtpct"]
                                 for n, r in results.items()})
        mat_labels = [MAT_SHORT.get(m, m) for m in comp_df.index]
        n_mats = len(comp_df)
        n_methods = len(results)
        bw = 0.22
        x = np.arange(n_mats)

        fig, ax = plt.subplots(figsize=(9, 4.5))
        offsets = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2,
                               n_methods) * bw
        for offset, (name, color) in zip(offsets, zip(results.keys(), colors)):
            bars = ax.bar(x + offset, comp_df[name], bw,
                          label=name, color=color, alpha=0.85)
            y_max_local = comp_df[name].max()
            for b in bars:
                h = b.get_height()
                if h == 0:
                    continue
                if h > y_max_local * 0.25:
                    ax.text(b.get_x() + b.get_width() / 2, h * 0.5,
                            f"{h:.1f}",
                            ha="center", va="center",
                            fontsize=6, fontweight="bold",
                            color="white", rotation=90)
                else:
                    ax.text(b.get_x() + b.get_width() / 2, h + 0.15,
                            f"{h:.1f}",
                            ha="center", va="bottom",
                            fontsize=6, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(mat_labels, rotation=35, ha="right",
                           fontsize=_FS_TICK)
        ax.set_ylabel("Weight (%)", fontsize=_FS_AX)
        ax.tick_params(axis="y", labelsize=_FS_TICK)
        ax.set_title("Comparison of Recommended Compositions",
                     fontsize=_FS_TITLE, fontweight="bold")
        ax.legend(fontsize=_FS_LABEL - 1, loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")
        y_ceil = comp_df.values.max()
        ax.set_ylim(0, y_ceil * 1.20)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # ── Fig 2: Cost vs CO2 ────────────────────────────────────────────
        st.subheader("Fig. 2 - Batch Cost vs. CO2 Emission")
        fig, ax = plt.subplots(figsize=(6, 4))
        _markers = ["o", "s", "*"]
        _edge_colors = ["#E24A33", "#348ABD", "#8EBA42"]
        _face_colors = ["none", "none", "#8EBA42"]
        _sizes = [160, 160, 220]
        _lws = [2.0, 2.0, 1.5]

        for (name, res), ec, fc, marker, sz, lw in zip(
                results.items(), _edge_colors, _face_colors,
                _markers, _sizes, _lws
        ):
            ax.scatter(res["cost_Tk_per_kg"], res["CO2_kg_per_kg"],
                       facecolors=fc, edgecolors=ec,
                       s=sz, marker=marker, linewidths=lw,
                       label=name, zorder=3)
        ax.set_xlabel("Batch Cost (Tk/kg)", fontsize=_FS_AX)
        ax.set_ylabel("CO2 Emission (kg/kg, cradle-to-gate)", fontsize=_FS_AX)
        ax.tick_params(labelsize=_FS_TICK)
        ax.set_title("Batch Cost vs. CO2 Emission",
                     fontsize=_FS_TITLE, fontweight="bold")
        ax.legend(fontsize=_FS_LABEL - 1, loc="best")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # ── Fig 3: Absolute error vs target ───────────────────────────────
        st.subheader("Fig. 3 - Absolute Prediction Error vs. Target")
        n_props = len(TARGET_COLS)
        bw = 0.22
        x_pos = np.arange(n_props)
        units = {"MOR_MPa": "MPa", "WA_pct": "%", "Shrinkage_pct": "%"}
        fmt_map = {"MOR_MPa": ".2f", "WA_pct": ".4f", "Shrinkage_pct": ".3f"}
        offsets = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2,
                              n_methods) * bw

        all_errs = [
            [abs(res["predicted"][p] - tgt[p]) for p in TARGET_COLS]
            for res in results.values()
        ]
        global_y_max = max(e for row in all_errs for e in row) if all_errs else 1.0

        fig, ax = plt.subplots(figsize=(7, 4))
        for offset, (name, res, color, errs) in zip(
            offsets,
            [(n, r, c, e) for (n, r), c, e
             in zip(results.items(), colors, all_errs)]
        ):
            bars = ax.bar(x_pos + offset, errs, bw,
                          label=name, color=color,
                          alpha=0.85, edgecolor="white")
            for b, err, p in zip(bars, errs, TARGET_COLS):
                if err == 0:
                    continue
                label_txt = f"{format(err, fmt_map[p])}\n{units[p]}"
                ax.text(b.get_x() + b.get_width() / 2,
                        b.get_height() + global_y_max * 0.02,
                        label_txt,
                        ha="center", va="bottom",
                        fontsize=6)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [TGT_LABELS.get(p, p) for p in TARGET_COLS],
            fontsize=_FS_TICK, rotation=15, ha="right",
        )
        ax.set_ylabel("Absolute Error (predicted vs. target)", fontsize=_FS_AX)
        ax.tick_params(axis="y", labelsize=_FS_TICK)
        ax.set_title("Absolute Prediction Error vs. Target (lower = better)",
                     fontsize=_FS_TITLE, fontweight="bold")
        ax.legend(fontsize=_FS_LABEL - 1)
        ax.grid(True, linestyle="--", alpha=0.45, axis="y")
        ax.set_ylim(0, global_y_max * 1.40)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)