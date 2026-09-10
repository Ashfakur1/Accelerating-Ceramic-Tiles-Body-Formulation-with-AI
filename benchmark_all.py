#!/usr/bin/env python3
"""
==============================================================================
benchmark_all.py
Ceramic-Tile AI — Conference Paper Deployment/Performance Benchmark Suite
(single-file version)

Consolidates everything that used to live in:
    _common.py, 01_model_footprint.py, 02_cold_vs_warm_start.py,
    03_latency_breakdown.py, 04_response_time_100_requests.py,
    05_scalability_test.py, 06_throughput.py, 07_stress_test_concurrency.py,
    08_resource_usage.py, 09_robustness_test.py, 10_optimization_convergence.py,
    11_ablation_study.py, 12_deployment_summary_table.py, run_all.py

into ONE script. Every benchmark also now saves a PNG figure (paper-ready)
in addition to the CSV/JSON it used to produce.

--------------------------------------------------------------------------
FOLDER LAYOUT ASSUMED (put this file inside the "Conference" folder):

    E:\\Ashfakur\\MSC\\Thesis\\ML Project 01\\
        data\\dataset.csv
        data\\metadata.json
        models\\forward_model.joblib
        models\\feature_cols.json
        Conference\\
            benchmark_all.py      <- this file
            results\\             <- csv / json, created automatically
            results\\figures\\    <- png, created automatically

If your data\\ / models\\ folders live somewhere else, edit PARENT_DIR below.
--------------------------------------------------------------------------

USAGE (from PyCharm terminal or a normal cmd/PowerShell, inside Conference\\):

    python benchmark_all.py                 # run everything, in order
    python benchmark_all.py --quick         # fast smoke-test (small trial counts)
    python benchmark_all.py --list          # show benchmark names
    python benchmark_all.py --only footprint,coldwarm,latency
    python benchmark_all.py --skip response100,scalability   # skip the slow ones

Requirements (same as before):
    pip install psutil optuna joblib scikit-learn pandas numpy matplotlib catboost
==============================================================================
"""
import argparse
import copy
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# =============================================================================
# 0. PATHS & CONFIG  (edit PARENT_DIR here if your data/models live elsewhere)
# =============================================================================
THIS_DIR    = Path(__file__).resolve().parent
PARENT_DIR  = THIS_DIR.parent                     # ".../ML Project 01"
DATADIR     = PARENT_DIR / "data"
MODELDIR    = PARENT_DIR / "models"
RESULTS_DIR = THIS_DIR / "results"
FIG_DIR     = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH    = MODELDIR / "forward_model.joblib"
FEATCOLS_PATH = MODELDIR / "feature_cols.json"
DATASET_PATH  = DATADIR / "dataset.csv"
METADATA_PATH = DATADIR / "metadata.json"

for _p in (MODEL_PATH, FEATCOLS_PATH, DATASET_PATH, METADATA_PATH):
    if not _p.exists():
        raise FileNotFoundError(
            f"Required artifact not found: {_p}\n"
            "Check that PARENT_DIR in benchmark_all.py points at the folder "
            "containing data\\ and models\\ (default assumes this script "
            "sits inside a 'Conference' subfolder of that project)."
        )

with open(METADATA_PATH) as f:
    META = json.load(f)
with open(FEATCOLS_PATH) as f:
    FEATURE_COLS: list = json.load(f)

MATERIALS = META["materials"]
BOUNDS    = META["bounds"]
COST      = META.get("cost_tk_per_kg", {m: 1.0 for m in MATERIALS})
_co2_raw  = META.get("co2_kg_per_kg", {m: 0.0 for m in MATERIALS})
CO2 = {m: float(np.mean(v)) if isinstance(v, (list, tuple)) else float(v)
       for m, v in _co2_raw.items()}

TARGET_COLS = ["MOR_MPa", "WA_pct", "Shrinkage_pct"]

DATASET = pd.read_csv(DATASET_PATH)
DATASET_SYNTH = DATASET[DATASET["source"] == "synthetic"].reset_index(drop=True)

_PROP_MIN = {t: float(DATASET_SYNTH[t].min()) for t in TARGET_COLS}
_PROP_MAX = {t: float(DATASET_SYNTH[t].max()) for t in TARGET_COLS}
_PROP_RANGE = {t: max(_PROP_MAX[t] - _PROP_MIN[t], 1e-9) for t in TARGET_COLS}

_COST_MIN = float(DATASET_SYNTH["cost_Tk_per_kg"].min())
_COST_RANGE = max(float(DATASET_SYNTH["cost_Tk_per_kg"].max()) - _COST_MIN, 1e-9)
_CO2_MIN = float(DATASET_SYNTH["CO2_kg_per_kg"].min())
_CO2_RANGE = max(float(DATASET_SYNTH["CO2_kg_per_kg"].max()) - _CO2_MIN, 1e-9)

_rng = np.random.default_rng(123)

# Trial / request counts -- overridden to small values by --quick
N_OPTUNA_TRIALS      = 200
N_REQUESTS_100       = 100
SCALABILITY_COUNTS   = [100, 500, 1000, 5000]
STRESS_CONCURRENCY   = [1, 5, 10, 20, 50, 100]
STRESS_REQS_PER_LVL  = 200
ABLATION_N_TARGETS   = 15
ABLATION_TRIAL_BUDG  = [200, 50, 20, 5]
N_REPEATS_SMALL      = 5


# =============================================================================
# 1. SHARED HELPERS  (was _common.py)
# =============================================================================
def load_model():
    return joblib.load(MODEL_PATH)


def sample_random_composition() -> dict:
    free = [m for m in MATERIALS if m != "SodaF"]
    lo = np.array([BOUNDS[m][0] for m in free])
    hi = np.array([BOUNDS[m][1] for m in free])
    soda_lo, soda_hi = BOUNDS["SodaF"]
    while True:
        v = _rng.uniform(lo, hi)
        s = 100.0 - v.sum()
        if soda_lo <= s <= soda_hi:
            c = dict(zip(free, v))
            c["SodaF"] = s
            return c


def sample_random_targets() -> dict:
    return {t: float(_rng.uniform(_PROP_MIN[t], _PROP_MAX[t])) for t in TARGET_COLS}


def build_input_row(comp: dict) -> pd.DataFrame:
    row = {f"{m}_wtpct": comp[m] for m in MATERIALS}
    return pd.DataFrame([row])[FEATURE_COLS]


def validate_composition(comp: dict):
    for m in MATERIALS:
        if m not in comp or comp[m] is None:
            return False, f"Please enter {m}."
        try:
            val = float(comp[m])
        except (TypeError, ValueError):
            return False, f"{m} must be numeric."
        if val < 0:
            return False, f"Negative composition not allowed ({m}={val})."
        lo, hi = BOUNDS[m]
        if val < lo or val > hi:
            return False, f"Input exceeds permissible range for {m} ({lo}-{hi})."
    return True, "OK"


def enforce_bounds(comp: dict) -> dict:
    clipped = {m: float(np.clip(comp[m], BOUNDS[m][0], BOUNDS[m][1])) for m in MATERIALS}
    total = sum(clipped.values())
    return {m: v / total * 100.0 for m, v in clipped.items()}


def bayesian_objective_factory(model, targets: dict, trial_log=None):
    import optuna

    def _objective(trial: "optuna.Trial") -> float:
        free_mats = [m for m in MATERIALS if m != "SodaF"]
        comp = {m: trial.suggest_float(f"c_{m}", BOUNDS[m][0], BOUNDS[m][1]) for m in free_mats}

        soda_f = 100.0 - sum(comp.values())
        soda_lo, soda_hi = BOUNDS["SodaF"]
        soda_pen = 0.0
        if soda_f < soda_lo:
            soda_pen = (soda_lo - soda_f) * 50.0
        elif soda_f > soda_hi:
            soda_pen = (soda_f - soda_hi) * 50.0
        comp["SodaF"] = float(np.clip(soda_f, soda_lo, soda_hi))
        comp = enforce_bounds(comp)

        pred = model.predict(build_input_row(comp))[0]
        mor_p, wa_p, sh_p = pred[0], pred[1], pred[2]

        cost = sum(comp[m] / 100 * COST[m] for m in MATERIALS)
        co2 = sum(comp[m] / 100 * CO2[m] for m in MATERIALS)
        norm_cost = (cost - _COST_MIN) / _COST_RANGE
        norm_co2 = (co2 - _CO2_MIN) / _CO2_RANGE

        penalty = (
            max(0.0, targets["MOR_MPa"] - mor_p) / _PROP_RANGE["MOR_MPa"] * 5.0
            + max(0.0, wa_p - targets["WA_pct"]) / _PROP_RANGE["WA_pct"] * 5.0
            + abs(sh_p - targets["Shrinkage_pct"]) / _PROP_RANGE["Shrinkage_pct"] * 5.0
            + soda_pen
        )
        obj = norm_cost + norm_co2 + penalty
        if trial_log is not None:
            trial_log.append(obj)
        return obj

    return _objective


def run_bayesian_optimization(model, targets: dict, n_trials: int = 200, trial_log=None):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    objective = bayesian_objective_factory(model, targets, trial_log)
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial.params
    best_c = {m: best[f"c_{m}"] for m in MATERIALS if m != "SodaF"}
    best_c["SodaF"] = float(np.clip(100.0 - sum(best_c.values()), *BOUNDS["SodaF"]))
    best_c = enforce_bounds(best_c)
    pred = model.predict(build_input_row(best_c))[0]
    return best_c, pred, study


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def save_json(obj, filename: str) -> Path:
    path = RESULTS_DIR / filename
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    print(f"  -> saved {path}")
    return path


def save_df(df: pd.DataFrame, filename: str) -> Path:
    path = RESULTS_DIR / filename
    df.to_csv(path, index=False)
    print(f"  -> saved {path}")
    return path


def save_fig(fig, filename: str) -> Path:
    path = FIG_DIR / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved {path}")
    return path


# Registry that collects a one-line summary from every benchmark, used to
# build the final deployment-summary table without re-reading files.
SUMMARY_ROWS: list = []


def _remember(metric: str, value: str):
    SUMMARY_ROWS.append((metric, value))


# =============================================================================
# 2. MODEL FOOTPRINT   (was 01_model_footprint.py)
# =============================================================================
def bench_model_footprint():
    print("\n[1] Model footprint ...")

    def file_size_mb(p):
        return p.stat().st_size / (1024 * 1024)

    model_size_mb = file_size_mb(MODEL_PATH)
    dataset_size_mb = file_size_mb(DATASET_PATH)

    model = load_model()
    tmp_path = RESULTS_DIR / "_tmp_resave.joblib"
    t0 = time.perf_counter()
    joblib.dump(model, tmp_path)
    serialize_s = time.perf_counter() - t0
    tmp_path.unlink(missing_ok=True)

    load_times = []
    for _ in range(N_REPEATS_SMALL):
        t0 = time.perf_counter()
        joblib.load(MODEL_PATH)
        load_times.append(time.perf_counter() - t0)

    result = {
        "model_size_MB": round(model_size_mb, 3),
        "dataset_size_MB": round(dataset_size_mb, 3),
        "serialization_time_s": round(serialize_s, 4),
        "deserialization_time_s_mean": round(sum(load_times) / len(load_times), 4),
        "deserialization_time_s_all": [round(t, 4) for t in load_times],
        "recommended_min_ram_GB": 2,
        "recommended_min_disk_MB": 120,
    }
    save_json(result, "model_footprint.json")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(["Model", "Dataset"], [model_size_mb, dataset_size_mb], color=["#4C72B0", "#DD8452"])
    axes[0].set_ylabel("Size (MB)")
    axes[0].set_title("File Size on Disk")
    axes[1].bar(["Serialize\n(save)", "Deserialize\n(load, mean)"],
                [serialize_s * 1000, result["deserialization_time_s_mean"] * 1000],
                color=["#55A868", "#C44E52"])
    axes[1].set_ylabel("Time (ms)")
    axes[1].set_title("Serialization / Load Time")
    fig.suptitle("Model Footprint")
    fig.tight_layout()
    save_fig(fig, "01_model_footprint.png")

    _remember("Model Size", f"{result['model_size_MB']} MB")
    print(f"  model={result['model_size_MB']} MB  load={result['deserialization_time_s_mean']*1000:.2f} ms")
    return result


# =============================================================================
# 3. COLD vs WARM START   (was 02_cold_vs_warm_start.py)
# =============================================================================
_COLD_START_SCRIPT = '''
import sys, time, json
sys.path.insert(0, r"{this_dir}")
t0 = time.perf_counter()
import joblib
from benchmark_all import MODEL_PATH, build_input_row, sample_random_composition
model = joblib.load(MODEL_PATH)
comp = sample_random_composition()
row = build_input_row(comp)
pred = model.predict(row)
t1 = time.perf_counter()
print(json.dumps({{"cold_start_s": t1 - t0}}))
'''


def bench_cold_vs_warm_start():
    print("\n[2] Cold start vs warm start ...")

    # ---- cold start: fresh subprocess per repeat ----
    cold_times = []
    script_text = _COLD_START_SCRIPT.format(this_dir=str(THIS_DIR))
    script_path = RESULTS_DIR / "_cold_start_probe.py"
    script_path.write_text(script_text)
    for i in range(N_REPEATS_SMALL):
        out = subprocess.run([sys.executable, str(script_path)],
                              capture_output=True, text=True, check=True)
        line = [l for l in out.stdout.strip().splitlines() if l.startswith("{")][-1]
        cold_times.append(json.loads(line)["cold_start_s"])
        print(f"  cold start repeat {i + 1}/{N_REPEATS_SMALL}: {cold_times[-1]:.3f} s")
    script_path.unlink(missing_ok=True)

    # ---- warm start: model already in memory ----
    model = load_model()
    for _ in range(10):
        model.predict(build_input_row(sample_random_composition()))
    warm_times = []
    for i in range(N_REPEATS_SMALL):
        row = build_input_row(sample_random_composition())
        t0 = time.perf_counter()
        model.predict(row)
        warm_times.append(time.perf_counter() - t0)
        print(f"  warm start repeat {i + 1}/{N_REPEATS_SMALL}: {warm_times[-1]*1000:.2f} ms")

    result = {
        "cold_start_s_mean": round(sum(cold_times) / len(cold_times), 4),
        "cold_start_s_all": [round(t, 4) for t in cold_times],
        "warm_start_s_mean": round(sum(warm_times) / len(warm_times), 6),
        "warm_start_s_all": [round(t, 6) for t in warm_times],
    }
    save_json(result, "cold_warm_start.json")
    save_df(pd.DataFrame({
        "State": ["Cold Start", "Warm Start"],
        "Response Time (s)": [result["cold_start_s_mean"], result["warm_start_s_mean"]],
    }), "cold_warm_start.csv")

    fig, ax = plt.subplots(figsize=(6, 4))
    means = [result["cold_start_s_mean"], result["warm_start_s_mean"]]
    stds = [np.std(cold_times), np.std(warm_times)]
    ax.bar(["Cold Start", "Warm Start"], means, yerr=stds, capsize=5,
           color=["#C44E52", "#55A868"])
    ax.set_ylabel("Response Time (s)")
    ax.set_yscale("log")
    ax.set_title("Cold Start vs Warm Start")
    fig.tight_layout()
    save_fig(fig, "02_cold_vs_warm_start.png")

    _remember("Cold Start", f"{result['cold_start_s_mean']} s")
    _remember("Warm Start", f"{result['warm_start_s_mean']*1000:.2f} ms")
    print(f"  cold={result['cold_start_s_mean']:.3f}s  warm={result['warm_start_s_mean']*1000:.2f}ms")
    return result


# =============================================================================
# 4. LATENCY BREAKDOWN   (was 03_latency_breakdown.py)
# =============================================================================
def _one_full_request(model):
    comp = sample_random_composition()
    targets = sample_random_targets()

    t0 = time.perf_counter()
    validate_composition(comp)
    t1 = time.perf_counter()

    row = build_input_row(comp)
    pred = model.predict(row)[0]
    t2 = time.perf_counter()

    best_c, best_pred, _ = run_bayesian_optimization(model, targets, n_trials=N_OPTUNA_TRIALS)
    t3 = time.perf_counter()

    _ = pd.DataFrame([{"material": m, "wt_pct": v} for m, v in best_c.items()])
    t4 = time.perf_counter()

    return {
        "input_validation_ms": (t1 - t0) * 1000,
        "prediction_ms": (t2 - t1) * 1000,
        "optimization_ms": (t3 - t2) * 1000,
        "visualization_ms": (t4 - t3) * 1000,
        "total_ms": (t4 - t0) * 1000,
    }


def bench_latency_breakdown():
    print(f"\n[3] Latency breakdown ({N_REPEATS_SMALL} trials x {N_OPTUNA_TRIALS} Optuna trials) ...")
    model = load_model()
    model.predict(build_input_row(sample_random_composition()))  # warm-up

    rows = [_one_full_request(model) for _ in range(N_REPEATS_SMALL)]
    for i, r in enumerate(rows):
        print(f"  trial {i + 1}/{N_REPEATS_SMALL}: total = {r['total_ms']:.1f} ms")

    df = pd.DataFrame(rows)
    means = df.mean().round(2)
    table = pd.DataFrame({
        "Component": ["Input validation", "Prediction", "Optimization", "Visualization", "Total"],
        "Time (ms)": [means["input_validation_ms"], means["prediction_ms"],
                       means["optimization_ms"], means["visualization_ms"], means["total_ms"]],
    })
    save_df(table, "latency_breakdown.csv")
    save_json({"mean": means.to_dict(), "all_trials": df.to_dict(orient="records")},
               "latency_breakdown.json")

    fig, ax = plt.subplots(figsize=(7, 4))
    stages = table["Component"][:-1]
    times = table["Time (ms)"][:-1]
    ax.bar(stages, times, color="#4C72B0")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"End-to-End Latency Breakdown (mean of {N_REPEATS_SMALL} trials)")
    plt.xticks(rotation=15)
    fig.tight_layout()
    save_fig(fig, "03_latency_breakdown.png")

    _remember("Prediction Time", f"{means['prediction_ms']:.2f} ms")
    _remember("Optimization Time", f"{means['optimization_ms']/1000:.2f} s")
    _remember("Total Response Time", f"{means['total_ms']/1000:.2f} s")
    print(table.to_string(index=False))
    return {"mean": means.to_dict()}


# =============================================================================
# 5. RESPONSE TIME - 100 REQUESTS   (was 04_response_time_100_requests.py)
# =============================================================================
def bench_response_time_100():
    print(f"\n[4] Response time over {N_REQUESTS_100} random requests "
          f"({N_OPTUNA_TRIALS} Optuna trials each -- this can take a while) ...")
    model = load_model()
    model.predict(build_input_row(sample_random_composition()))

    response_times_s = []
    for i in range(N_REQUESTS_100):
        targets = sample_random_targets()
        t0 = time.perf_counter()
        run_bayesian_optimization(model, targets, n_trials=N_OPTUNA_TRIALS)
        response_times_s.append(time.perf_counter() - t0)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{N_REQUESTS_100} done (last = {response_times_s[-1]:.3f} s)")

    df = pd.DataFrame({"request_id": range(1, N_REQUESTS_100 + 1),
                        "response_time_s": response_times_s})
    save_df(df, "response_time_100.csv")

    summary = {
        "n_requests": N_REQUESTS_100,
        "average_s": round(sum(response_times_s) / len(response_times_s), 3),
        "maximum_s": round(max(response_times_s), 3),
        "minimum_s": round(min(response_times_s), 3),
    }
    save_json(summary, "response_time_100_summary.json")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df["request_id"], df["response_time_s"], marker=".", linewidth=1, alpha=0.7)
    ax.axhline(summary["average_s"], color="crimson", linestyle="--", label=f"mean={summary['average_s']}s")
    ax.set_xlabel("Request #")
    ax.set_ylabel("Response Time (s)")
    ax.set_title(f"Response Time over {N_REQUESTS_100} Random Requests")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "04_response_time_100.png")

    _remember("Avg Response Time (100 req)", f"{summary['average_s']} s")
    print(summary)
    return summary


# =============================================================================
# 6. SCALABILITY   (was 05_scalability_test.py)
# =============================================================================
def bench_scalability():
    print(f"\n[5] Scalability sweep over {SCALABILITY_COUNTS} prediction requests ...")
    model = load_model()
    model.predict(build_input_row(sample_random_composition()))

    rows = []
    for n in SCALABILITY_COUNTS:
        print(f"  running {n} sequential prediction requests ...")
        comps = [sample_random_composition() for _ in range(n)]
        latencies = []
        for c in comps:
            row = build_input_row(c)
            t0 = time.perf_counter()
            model.predict(row)
            latencies.append(time.perf_counter() - t0)
        avg_latency = sum(latencies) / len(latencies)
        rows.append({
            "requests": n,
            "avg_latency_ms": round(avg_latency * 1000, 4),
            "total_time_s": round(sum(latencies), 3),
            "throughput_req_per_s": round(n / sum(latencies), 2),
        })
        print(f"    -> avg latency {avg_latency*1000:.3f} ms | throughput {rows[-1]['throughput_req_per_s']} req/s")

    df = pd.DataFrame(rows)
    save_df(df, "scalability.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(df["requests"], df["avg_latency_ms"], marker="o", color="#4C72B0")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Number of Requests")
    axes[0].set_ylabel("Avg Latency (ms)")
    axes[0].set_title("Latency vs Load")
    axes[0].grid(alpha=0.3)
    axes[1].plot(df["requests"], df["throughput_req_per_s"], marker="s", color="#55A868")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Number of Requests")
    axes[1].set_ylabel("Throughput (req/s)")
    axes[1].set_title("Throughput vs Load")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "05_scalability.png")

    print(df.to_string(index=False))
    return df.to_dict(orient="records")


# =============================================================================
# 7. THROUGHPUT   (was 06_throughput.py)
# =============================================================================
def bench_throughput():
    print("\n[6] Throughput (pure inference + full pipeline) ...")
    model = load_model()
    model.predict(build_input_row(sample_random_composition()))

    n_predictions = 1000
    n_full_requests = 10

    comps = [sample_random_composition() for _ in range(n_predictions)]
    t0 = time.perf_counter()
    for c in comps:
        model.predict(build_input_row(c))
    inference_elapsed = time.perf_counter() - t0
    predictions_per_sec = n_predictions / inference_elapsed

    print(f"  full-pipeline throughput over {n_full_requests} requests "
          f"({N_OPTUNA_TRIALS} Optuna trials each) ...")
    t0 = time.perf_counter()
    for _ in range(n_full_requests):
        run_bayesian_optimization(model, sample_random_targets(), n_trials=N_OPTUNA_TRIALS)
    full_elapsed = time.perf_counter() - t0
    requests_per_sec = n_full_requests / full_elapsed
    requests_per_min = requests_per_sec * 60

    result = {
        "pure_inference": {
            "n_predictions": n_predictions,
            "elapsed_s": round(inference_elapsed, 3),
            "predictions_per_sec": round(predictions_per_sec, 2),
        },
        "full_pipeline": {
            "n_requests": n_full_requests,
            "n_optuna_trials_each": N_OPTUNA_TRIALS,
            "elapsed_s": round(full_elapsed, 3),
            "requests_per_sec": round(requests_per_sec, 4),
            "requests_per_min": round(requests_per_min, 2),
        },
    }
    save_json(result, "throughput.json")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(["Pure Inference"], [predictions_per_sec], color="#4C72B0")
    axes[0].set_ylabel("Predictions / sec")
    axes[0].set_title("Inference Throughput")
    axes[1].bar(["Full Pipeline"], [requests_per_min], color="#DD8452")
    axes[1].set_ylabel("Requests / min")
    axes[1].set_title("Full-Pipeline Throughput")
    fig.tight_layout()
    save_fig(fig, "06_throughput.png")

    _remember("Inference Throughput", f"{result['pure_inference']['predictions_per_sec']} predictions/sec")
    _remember("Pipeline Throughput", f"{result['full_pipeline']['requests_per_min']} requests/min")
    print(result)
    return result


# =============================================================================
# 8. STRESS TEST / CONCURRENCY   (was 07_stress_test_concurrency.py)
# =============================================================================
LOCUST_FILE_TEMPLATE = '''
from locust import HttpUser, task, between

class StreamlitAppUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def load_home_page(self):
        self.client.get("/")
'''


def _single_prediction(model):
    row = build_input_row(sample_random_composition())
    t0 = time.perf_counter()
    model.predict(row)
    return time.perf_counter() - t0


def bench_stress_test():
    from concurrent.futures import ThreadPoolExecutor, as_completed
    print(f"\n[7] Stress test across concurrency levels {STRESS_CONCURRENCY} ...")
    model = load_model()
    model.predict(build_input_row(sample_random_composition()))
    process = psutil.Process()

    rows = []
    for n_workers in STRESS_CONCURRENCY:
        print(f"  concurrency={n_workers} ({STRESS_REQS_PER_LVL} total requests) ...")
        process.cpu_percent(interval=None)
        mem_before = process.memory_info().rss / (1024 * 1024)

        latencies = []
        t_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_single_prediction, model) for _ in range(STRESS_REQS_PER_LVL)]
            for f in as_completed(futures):
                latencies.append(f.result())
        t_end = time.perf_counter()

        cpu_after = process.cpu_percent(interval=0.5)
        mem_after = process.memory_info().rss / (1024 * 1024)

        rows.append({
            "concurrent_users": n_workers,
            "total_requests": STRESS_REQS_PER_LVL,
            "avg_latency_ms": round(sum(latencies) / len(latencies) * 1000, 3),
            "max_latency_ms": round(max(latencies) * 1000, 3),
            "wall_clock_s": round(t_end - t_start, 3),
            "throughput_req_per_s": round(STRESS_REQS_PER_LVL / (t_end - t_start), 2),
            "cpu_percent": cpu_after,
            "memory_MB": round(mem_after, 2),
            "memory_delta_MB": round(mem_after - mem_before, 2),
        })
        print(f"    -> avg latency {rows[-1]['avg_latency_ms']} ms | "
              f"throughput {rows[-1]['throughput_req_per_s']} req/s | "
              f"CPU {rows[-1]['cpu_percent']}% | RAM {rows[-1]['memory_MB']} MB")

    df = pd.DataFrame(rows)
    save_df(df, "stress_test.csv")
    (RESULTS_DIR / "locustfile_template.py").write_text(LOCUST_FILE_TEMPLATE)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(df["concurrent_users"], df["avg_latency_ms"], marker="o", color="#C44E52")
    axes[0].set_xlabel("Concurrent Users")
    axes[0].set_ylabel("Avg Latency (ms)")
    axes[0].set_title("Latency vs Concurrency")
    axes[0].grid(alpha=0.3)
    axes[1].plot(df["concurrent_users"], df["throughput_req_per_s"], marker="s", color="#55A868")
    axes[1].set_xlabel("Concurrent Users")
    axes[1].set_ylabel("Throughput (req/s)")
    axes[1].set_title("Throughput vs Concurrency")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "07_stress_test.png")

    print(df.to_string(index=False))
    return df.to_dict(orient="records")


# =============================================================================
# 9. RESOURCE USAGE   (was 08_resource_usage.py)
# =============================================================================
def bench_resource_usage():
    print("\n[8] Resource usage (CPU / memory) ...")

    def mb(v):
        return v / (1024 * 1024)

    process = psutil.Process()
    mem_before_load = mb(process.memory_info().rss)
    t0 = time.perf_counter()
    model = load_model()
    load_time_s = time.perf_counter() - t0
    mem_after_load = mb(process.memory_info().rss)

    model.predict(build_input_row(sample_random_composition()))

    cpu_samples = []
    process.cpu_percent(interval=None)
    peak_mem = mem_after_load

    t0 = time.perf_counter()
    for _ in range(N_REPEATS_SMALL):
        run_bayesian_optimization(model, sample_random_targets(), n_trials=N_OPTUNA_TRIALS)
        cpu_samples.append(process.cpu_percent(interval=None))
        peak_mem = max(peak_mem, mb(process.memory_info().rss))
    elapsed = time.perf_counter() - t0

    result = {
        "memory_before_load_MB": round(mem_before_load, 2),
        "memory_after_load_MB": round(mem_after_load, 2),
        "startup_memory_increase_MB": round(mem_after_load - mem_before_load, 2),
        "model_load_time_s": round(load_time_s, 3),
        "peak_memory_MB": round(peak_mem, 2),
        "avg_cpu_percent": round(sum(cpu_samples) / len(cpu_samples), 2) if cpu_samples else None,
        "workload_elapsed_s": round(elapsed, 3),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "system_total_ram_GB": round(psutil.virtual_memory().total / (1024 ** 3), 2),
    }
    save_json(result, "resource_usage.json")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(["Before Load", "After Load", "Peak (workload)"],
                [mem_before_load, mem_after_load, peak_mem],
                color=["#4C72B0", "#DD8452", "#C44E52"])
    axes[0].set_ylabel("Memory (MB)")
    axes[0].set_title("Memory Usage")
    axes[1].bar(["Avg CPU %"], [result["avg_cpu_percent"] or 0], color="#55A868")
    axes[1].set_ylabel("CPU (%)")
    axes[1].set_title("CPU Usage During Workload")
    fig.tight_layout()
    save_fig(fig, "08_resource_usage.png")

    _remember("Peak Memory", f"{result['peak_memory_MB']} MB")
    _remember("CPU Usage", f"{result['avg_cpu_percent']} %")
    print(result)
    return result


# =============================================================================
# 10. ROBUSTNESS TEST   (was 09_robustness_test.py)
# =============================================================================
def _build_robustness_cases():
    base = sample_random_composition()
    cases = [{"name": "valid_baseline", "input": dict(base), "expect_valid": True}]

    neg = copy.deepcopy(base)
    m = MATERIALS[0]
    neg[m] = -10.0
    cases.append({"name": f"negative_{m}", "input": neg, "expect_valid": False})

    over = copy.deepcopy(base)
    m2 = MATERIALS[0]
    over[m2] = BOUNDS[m2][1] + 50.0
    cases.append({"name": f"out_of_range_high_{m2}", "input": over, "expect_valid": False})

    under = copy.deepcopy(base)
    m3 = MATERIALS[1]
    under[m3] = max(BOUNDS[m3][0] - 5.0, 0.01)
    cases.append({"name": f"out_of_range_low_{m3}", "input": under, "expect_valid": False})

    missing = copy.deepcopy(base)
    m4 = MATERIALS[2]
    del missing[m4]
    cases.append({"name": f"missing_{m4}", "input": missing, "expect_valid": False})

    wrong_type = copy.deepcopy(base)
    m5 = MATERIALS[3]
    wrong_type[m5] = "not_a_number"
    cases.append({"name": f"wrong_datatype_{m5}", "input": wrong_type, "expect_valid": False})

    none_val = copy.deepcopy(base)
    m6 = MATERIALS[4]
    none_val[m6] = None
    cases.append({"name": f"none_value_{m6}", "input": none_val, "expect_valid": False})

    return cases


def bench_robustness():
    print("\n[9] Robustness test (bad input handling) ...")
    cases = _build_robustness_cases()
    rows = []
    for case in cases:
        crashed = False
        try:
            is_valid, message = validate_composition(case["input"])
        except Exception as exc:
            crashed = True
            is_valid, message = False, f"UNCAUGHT EXCEPTION: {exc}"
        correct = (is_valid == case["expect_valid"])
        rows.append({
            "test_case": case["name"],
            "expected_valid": case["expect_valid"],
            "actual_valid": is_valid,
            "message": message,
            "handled_gracefully": not crashed,
            "result": "PASS" if (correct and not crashed) else "FAIL",
        })
        print(f"  [{rows[-1]['result']}] {case['name']}: {message}")

    df = pd.DataFrame(rows)
    save_df(df, "robustness_test.csv")
    n_pass = (df["result"] == "PASS").sum()

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#55A868" if r == "PASS" else "#C44E52" for r in df["result"]]
    ax.barh(df["test_case"], [1] * len(df), color=colors)
    ax.set_xticks([])
    ax.set_title(f"Robustness Test: {n_pass}/{len(df)} cases passed")
    for i, r in enumerate(df["result"]):
        ax.text(0.5, i, r, ha="center", va="center", color="white", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "09_robustness_test.png")

    _remember("Robustness", f"{n_pass}/{len(df)} passed")
    print(f"  {n_pass}/{len(df)} cases passed")
    return {"n_pass": int(n_pass), "n_total": len(df)}


# =============================================================================
# 11. OPTIMIZATION CONVERGENCE   (was 10_optimization_convergence.py)
# =============================================================================
def bench_optimization_convergence():
    print(f"\n[10] Optimization convergence over {N_OPTUNA_TRIALS} trials ...")
    model = load_model()
    model.predict(build_input_row(sample_random_composition()))

    fixed_target = {"MOR_MPa": 55.0, "WA_pct": 3.6, "Shrinkage_pct": 11.0}
    trial_log: list = []
    best_c, best_pred, study = run_bayesian_optimization(
        model, fixed_target, n_trials=N_OPTUNA_TRIALS, trial_log=trial_log)

    running_best = []
    best_so_far = float("inf")
    for v in trial_log:
        best_so_far = min(best_so_far, v)
        running_best.append(best_so_far)

    df = pd.DataFrame({
        "trial": range(1, len(trial_log) + 1),
        "objective_value": trial_log,
        "best_so_far": running_best,
    })
    save_df(df, "optimization_convergence.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["trial"], df["objective_value"], alpha=0.35, label="Trial objective")
    ax.plot(df["trial"], df["best_so_far"], color="crimson", linewidth=2, label="Best so far")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Objective value (norm. cost + CO2 + penalty)")
    ax.set_title("Bayesian Optimization (TPE) Convergence")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "10_optimization_convergence.png")

    print(f"  best objective value: {running_best[-1]:.4f}")
    return {"best_objective": running_best[-1]}


# =============================================================================
# 12. ABLATION STUDY   (was 11_ablation_study.py)
# =============================================================================
def _nn_baseline(targets: dict) -> dict:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X = scaler.fit_transform(DATASET_SYNTH[TARGET_COLS].values)
    nbrs = NearestNeighbors(n_neighbors=1).fit(X)
    q = scaler.transform([[targets[t] for t in TARGET_COLS]])
    idx = nbrs.kneighbors(q, return_distance=False)[0][0]
    row = DATASET_SYNTH.iloc[idx]
    return {
        "MOR_MPa": float(row["MOR_MPa"]), "WA_pct": float(row["WA_pct"]),
        "Shrinkage_pct": float(row["Shrinkage_pct"]),
        "cost_Tk_per_kg": float(row["cost_Tk_per_kg"]),
        "CO2_kg_per_kg": float(row["CO2_kg_per_kg"]),
    }


def _property_gap(pred: dict, targets: dict) -> float:
    return float(np.mean([
        abs(pred[t] - targets[t]) / max(abs(targets[t]), 1e-9) * 100 for t in TARGET_COLS
    ]))


def bench_ablation_study():
    print(f"\n[11] Ablation study ({ABLATION_N_TARGETS} targets, trial budgets {ABLATION_TRIAL_BUDG}) ...")
    model = load_model()
    model.predict(build_input_row(sample_random_composition()))

    # A. optimizer vs nearest-neighbour baseline
    rows_a = []
    for i in range(ABLATION_N_TARGETS):
        targets = sample_random_targets()
        nn_result = _nn_baseline(targets)

        t0 = time.perf_counter()
        best_c, best_pred, _ = run_bayesian_optimization(model, targets, n_trials=200)
        opt_time = time.perf_counter() - t0
        opt_cost = sum(best_c[m] / 100 * COST[m] for m in MATERIALS)
        opt_co2 = sum(best_c[m] / 100 * CO2[m] for m in MATERIALS)
        opt_pred = {"MOR_MPa": float(best_pred[0]), "WA_pct": float(best_pred[1]),
                    "Shrinkage_pct": float(best_pred[2])}

        rows_a.append({
            "target_id": i + 1,
            "nn_cost": round(nn_result["cost_Tk_per_kg"], 4),
            "nn_co2": round(nn_result["CO2_kg_per_kg"], 5),
            "nn_property_gap_pct": round(_property_gap(nn_result, targets), 3),
            "optimized_cost": round(opt_cost, 4),
            "optimized_co2": round(opt_co2, 5),
            "optimized_property_gap_pct": round(_property_gap(opt_pred, targets), 3),
            "optimized_time_s": round(opt_time, 3),
            "cost_reduction_pct": round((nn_result["cost_Tk_per_kg"] - opt_cost) / nn_result["cost_Tk_per_kg"] * 100, 2),
            "co2_reduction_pct": round((nn_result["CO2_kg_per_kg"] - opt_co2) / nn_result["CO2_kg_per_kg"] * 100, 2),
        })
        print(f"  target {i+1}/{ABLATION_N_TARGETS}: cost reduction={rows_a[-1]['cost_reduction_pct']}%, "
              f"CO2 reduction={rows_a[-1]['co2_reduction_pct']}%")

    df_a = pd.DataFrame(rows_a)
    save_df(df_a, "ablation_optimizer_vs_nn.csv")

    # B. trial-budget ablation
    fixed_target = {"MOR_MPa": 55.0, "WA_pct": 3.6, "Shrinkage_pct": 11.0}
    rows_b = []
    for n_trials in ABLATION_TRIAL_BUDG:
        t0 = time.perf_counter()
        _, _, study = run_bayesian_optimization(model, fixed_target, n_trials=n_trials)
        elapsed = time.perf_counter() - t0
        rows_b.append({"n_trials": n_trials, "best_objective_value": round(study.best_value, 4),
                        "optimization_time_s": round(elapsed, 3)})
        print(f"  n_trials={n_trials}: best objective={rows_b[-1]['best_objective_value']}, "
              f"time={rows_b[-1]['optimization_time_s']}s")
    df_b = pd.DataFrame(rows_b)
    save_df(df_b, "ablation_trial_budget.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(["Cost reduction %", "CO2 reduction %"],
                [df_a["cost_reduction_pct"].mean(), df_a["co2_reduction_pct"].mean()],
                color=["#4C72B0", "#55A868"])
    axes[0].set_title("Optimizer vs Nearest-Neighbour\n(mean over targets)")
    axes[0].set_ylabel("% reduction")
    axes[1].plot(df_b["n_trials"], df_b["best_objective_value"], marker="o", color="#C44E52")
    axes[1].set_xlabel("Optuna Trial Budget")
    axes[1].set_ylabel("Best Objective Value")
    axes[1].set_title("Solution Quality vs Trial Budget")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "11_ablation_study.png")

    print(f"  mean cost reduction: {df_a['cost_reduction_pct'].mean():.2f}%")
    print(f"  mean CO2 reduction : {df_a['co2_reduction_pct'].mean():.2f}%")
    return {"cost_reduction_pct_mean": df_a["cost_reduction_pct"].mean(),
            "co2_reduction_pct_mean": df_a["co2_reduction_pct"].mean()}


# =============================================================================
# 13. DEPLOYMENT SUMMARY TABLE   (was 12_deployment_summary_table.py)
# =============================================================================
def bench_deployment_summary():
    print("\n[12] Deployment summary table ...")
    if not SUMMARY_ROWS:
        print("  No metrics collected yet -- run other benchmarks first (or use --only all).")
        return {}

    df = pd.DataFrame(SUMMARY_ROWS, columns=["Metric", "Value"])
    save_df(df, "deployment_summary_table.csv")

    fig, ax = plt.subplots(figsize=(7, 0.5 + 0.4 * len(df)))
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.5)
    ax.set_title("Deployment Summary Table", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "12_deployment_summary_table.png")

    print(df.to_string(index=False))
    return df.to_dict(orient="records")


# =============================================================================
# 14. RUNNER
# =============================================================================
BENCHMARKS = {
    "footprint":    ("Model footprint",            bench_model_footprint),
    "coldwarm":     ("Cold vs warm start",         bench_cold_vs_warm_start),
    "latency":      ("Latency breakdown",          bench_latency_breakdown),
    "response100":  ("Response time (100 reqs)",   bench_response_time_100),
    "scalability":  ("Scalability",                bench_scalability),
    "throughput":   ("Throughput",                 bench_throughput),
    "stress":       ("Stress test / concurrency",  bench_stress_test),
    "resource":     ("Resource usage",             bench_resource_usage),
    "robustness":   ("Robustness test",            bench_robustness),
    "convergence":  ("Optimization convergence",   bench_optimization_convergence),
    "ablation":     ("Ablation study",              bench_ablation_study),
    "summary":      ("Deployment summary table",   bench_deployment_summary),
}
ORDER = list(BENCHMARKS.keys())  # summary must stay last


def apply_quick_mode():
    global N_OPTUNA_TRIALS, N_REQUESTS_100, SCALABILITY_COUNTS
    global STRESS_CONCURRENCY, STRESS_REQS_PER_LVL, ABLATION_N_TARGETS, ABLATION_TRIAL_BUDG
    N_OPTUNA_TRIALS = 20
    N_REQUESTS_100 = 10
    SCALABILITY_COUNTS = [10, 50, 100]
    STRESS_CONCURRENCY = [1, 5, 10]
    STRESS_REQS_PER_LVL = 20
    ABLATION_N_TARGETS = 3
    ABLATION_TRIAL_BUDG = [50, 20, 5]
    print("*** QUICK MODE: using small trial/request counts for a fast smoke test ***")


def main():
    parser = argparse.ArgumentParser(description="Run all (or selected) deployment benchmarks.")
    parser.add_argument("--only", type=str, default=None,
                         help="Comma-separated benchmark keys to run, e.g. footprint,coldwarm")
    parser.add_argument("--skip", type=str, default=None,
                         help="Comma-separated benchmark keys to skip")
    parser.add_argument("--quick", action="store_true",
                         help="Use small trial/request counts for a fast smoke test")
    parser.add_argument("--list", action="store_true", help="List benchmark keys and exit")
    args = parser.parse_args()

    if args.list:
        print("Available benchmarks:")
        for key, (desc, _) in BENCHMARKS.items():
            print(f"  {key:12s} - {desc}")
        return

    if args.quick:
        apply_quick_mode()

    to_run = ORDER
    if args.only:
        keys = [k.strip() for k in args.only.split(",")]
        to_run = [k for k in ORDER if k in keys]
        if "summary" not in to_run:
            to_run.append("summary")  # always rebuild the summary at the end
    if args.skip:
        skip_keys = {k.strip() for k in args.skip.split(",")}
        to_run = [k for k in to_run if k not in skip_keys]

    print("=" * 70)
    print("Ceramic-Tile AI — Deployment/Performance Benchmark Suite")
    print(f"Results -> {RESULTS_DIR}")
    print(f"Figures -> {FIG_DIR}")
    print(f"Running: {', '.join(to_run)}")
    print("=" * 70)

    for key in to_run:
        desc, fn = BENCHMARKS[key]
        t0 = time.perf_counter()
        try:
            fn()
        except Exception as exc:
            print(f"  !! {desc} FAILED: {exc}")
        print(f"[{desc} finished in {time.perf_counter() - t0:.1f} s]")

    print("\nAll requested benchmarks finished.")
    print(f"CSV / JSON  -> {RESULTS_DIR}")
    print(f"Figures(PNG)-> {FIG_DIR}")


if __name__ == "__main__":
    main()
