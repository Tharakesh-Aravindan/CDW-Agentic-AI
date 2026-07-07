"""
sparsity_robustness.py
======================
Multi-seed robustness check for the data-availability degradation experiment.

Suman's notebook ran the sparsity experiment once with a fixed random seed.
At low data-availability levels (e.g. 25%), the training set shrinks to only
2-3 samples after feature engineering, which makes any single result highly
sensitive to the random seed. This script re-runs the experiment across 10
seeds (0-9) and reports the MEDIAN MAPE and inter-quartile range (IQR) per
model per availability level, so we can distinguish a robust finding from a
lucky single run.

Input : out/annual_cdw_series.csv   (produced by the forecasting notebook)
Output: out/sparsity_robustness.csv (median + IQR per model per level)
        out/sparsity_robustness_plot.png

Run:  python sparsity_robustness.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUT_DIR = Path("out")
SERIES_CSV = OUT_DIR / "annual_cdw_series.csv"

# The three most recent MEASURED years are the validation set. These are held
# out of every training subset, at every availability level, so validation
# MAPE is always computed against real measured data.
VAL_YEARS = [2021, 2022, 2023]

# Data-availability levels: fraction of the training history to keep. At each
# level we keep the MOST RECENT fraction of years before the validation point.
AVAILABILITY_LEVELS = [1.00, 0.75, 0.50, 0.25]

# Random seeds for the robustness check.
SEEDS = list(range(10))  # 0, 1, 2, ..., 9

# Minimum training rows to keep after truncation, so feature engineering
# (which needs lag1 + lag2) still has at least a couple of usable samples.
MIN_TRAIN_ROWS = 4


def load_series() -> pd.DataFrame:
    """Load the reconstructed annual C&D waste series (2004-2025)."""
    if not SERIES_CSV.exists():
        raise FileNotFoundError(
            f"{SERIES_CSV} not found. Run the forecasting notebook first "
            f"(the cell that saves annual_cdw_series.csv)."
        )
    df = pd.read_csv(SERIES_CSV)
    df = df.sort_values("year").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Feature engineering (matches the forecasting notebook)
# ---------------------------------------------------------------------------
FEAT_COLS = ["lag1", "lag2", "rolling2", "diff1", "years_since_start"]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw annual series into lag-based ML features.

    For each year t, we build:
      lag1              = value at year t-1   (last year)
      lag2              = value at year t-2   (two years ago)
      rolling2          = mean of the two previous years
      diff1             = year-on-year change (velocity)
      years_since_start = t - first_year      (long-run trend index)

    Rows where lag1/lag2 are undefined (the first two years) are dropped,
    because a model cannot use a feature that has no value.
    """
    d = df.sort_values("year").reset_index(drop=True).copy()
    d["lag1"] = d["VALUE"].shift(1)
    d["lag2"] = d["VALUE"].shift(2)
    d["rolling2"] = d["VALUE"].shift(1).rolling(window=2).mean()
    d["diff1"] = d["VALUE"].shift(1) - d["VALUE"].shift(2)
    d["years_since_start"] = d["year"] - d["year"].min()
    d = d.dropna(subset=FEAT_COLS).reset_index(drop=True)
    return d


    # ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------
# Each wrapper takes:
#   train_df : the training rows (already feature-engineered)
#   target_features : the feature row for the year we want to predict
#   seed : random seed for reproducibility / robustness testing
# and returns a single predicted VALUE (float).
#
# Injecting `seed` into every model is what makes the robustness check
# possible: the same train/predict call with a different seed gives a
# different result, and we measure how much that variation matters.

from sklearn.metrics import mean_absolute_percentage_error


def predict_xgboost(train_df, target_features, seed):
    """XGBoost regressor on lag features."""
    from xgboost import XGBRegressor
    model = XGBRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        reg_alpha=1.0, reg_lambda=2.0, subsample=0.8, colsample_bytree=0.8,
        random_state=seed, verbosity=0,
    )
    model.fit(train_df[FEAT_COLS], train_df["VALUE"])
    return float(model.predict(target_features[FEAT_COLS])[0])


def predict_lightgbm(train_df, target_features, seed):
    from lightgbm import LGBMRegressor
    model = LGBMRegressor(
        n_estimators=300, num_leaves=4, max_depth=3, learning_rate=0.05,
        reg_alpha=1.0, reg_lambda=2.0,
        bagging_fraction=0.8, bagging_freq=1, bagging_seed=seed,
        feature_fraction=0.8, feature_fraction_seed=seed,
        random_state=seed, verbose=-1,
    )
    model.fit(train_df[FEAT_COLS], train_df["VALUE"])
    return float(model.predict(target_features[FEAT_COLS])[0])


def predict_svr(train_df, target_features, seed):
    """Support Vector Regression on standardised lag features.

    SVR itself is deterministic (no seed), but we accept `seed` for a
    uniform interface. Features are standardised because SVR is scale-sensitive.
    """
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[FEAT_COLS])
    X_target = scaler.transform(target_features[FEAT_COLS])
    model = SVR(kernel="rbf", C=1e6, gamma="scale")
    model.fit(X_train, train_df["VALUE"])
    return float(model.predict(X_target)[0])


def predict_arima(train_df, target_features, seed):
    """ARIMA on the raw VALUE series (ignores lag features).

    ARIMA is deterministic given the data, so `seed` has no effect; we accept
    it for a uniform interface. We fit ARIMA(1,1,1) as a robust default for a
    short series and forecast one step ahead.
    """
    from statsmodels.tsa.arima.model import ARIMA
    series = train_df["VALUE"].values
    try:
        model = ARIMA(series, order=(1, 1, 1)).fit()
        return float(model.forecast(steps=1)[0])
    except Exception:
        # Fallback for very short series: predict last value (persistence)
        return float(series[-1])

def predict_stl_lightgbm(train_df, target_features, seed):
    """STL trend decomposition + LightGBM on residuals.

    Splits the series into a smooth trend and residuals, extrapolates the
    trend linearly, and predicts the residual with LightGBM. Falls back to
    plain LightGBM if the series is too short for STL (needs >= 2 periods).
    """
    from lightgbm import LGBMRegressor
    series = train_df["VALUE"].values
    if len(series) < 6:
        # STL needs a reasonable length; fall back to plain LightGBM
        return predict_lightgbm(train_df, target_features, seed)
    try:
        from statsmodels.tsa.seasonal import STL
        stl = STL(series, period=2, robust=True).fit()
        trend = stl.trend
        # Linear extrapolation of the trend one step ahead
        slope = trend[-1] - trend[-2]
        trend_next = trend[-1] + slope
        # LightGBM on the residual using lag features
        resid = series - trend
        train_resid = train_df.copy()
        train_resid["VALUE"] = resid
        model = LGBMRegressor(
            n_estimators=200, num_leaves=4, max_depth=3, learning_rate=0.05,
            random_state=seed, verbose=-1,
        )
        model.fit(train_resid[FEAT_COLS], train_resid["VALUE"])
        resid_next = float(model.predict(target_features[FEAT_COLS])[0])
        return float(trend_next + resid_next)
    except Exception:
        return predict_lightgbm(train_df, target_features, seed)


def predict_prophet(train_df, target_features, seed):
    """Facebook Prophet on the (year, VALUE) series.

    Prophet is deterministic; `seed` accepted for uniform interface. Prophet
    is designed for sub-daily seasonal data, so on an annual series it
    essentially fits a trend. Included as a comparison lower bound.
    """
    try:
        from prophet import Prophet
        import logging
        logging.getLogger("prophet").setLevel(logging.ERROR)
        logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
        target_year = int(target_features["year"].iloc[0])
        pdf = pd.DataFrame({
            "ds": pd.to_datetime(train_df["year"].astype(str) + "-01-01"),
            "y": train_df["VALUE"].values,
        })
        m = Prophet(yearly_seasonality=False, weekly_seasonality=False,
                    daily_seasonality=False)
        m.fit(pdf)
        future = pd.DataFrame({"ds": [pd.to_datetime(f"{target_year}-01-01")]})
        forecast = m.predict(future)
        return float(forecast["yhat"].iloc[0])
    except Exception:
        # Fallback: persistence
        return float(train_df["VALUE"].iloc[-1])


def predict_tft(train_df, target_features, seed):
    """A small MLP neural model as a lightweight TFT-inspired representative.

    NOTE: This is a simplified neural stand-in, not the full Temporal Fusion
    Transformer. On a series this short, a full TFT cannot be reliably
    trained; this MLP serves to characterise how a small neural model behaves
    under the same data-availability conditions. `seed` controls weight init.
    """
    try:
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[FEAT_COLS])
        X_target = scaler.transform(target_features[FEAT_COLS])
        y_scaler = StandardScaler()
        y_train = y_scaler.fit_transform(train_df[["VALUE"]]).ravel()
        model = MLPRegressor(
            hidden_layer_sizes=(16, 8), max_iter=2000,
            random_state=seed, early_stopping=False,
        )
        model.fit(X_train, y_train)
        pred_scaled = model.predict(X_target)
        return float(y_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0])
    except Exception:
        return float(train_df["VALUE"].iloc[-1])

# Registry of all seven models, used by the experiment and the smoke test.
MODELS = [
    ("XGBoost", predict_xgboost),
    ("LightGBM", predict_lightgbm),
    ("SVR", predict_svr),
    ("ARIMA", predict_arima),
    ("STL+LGBM", predict_stl_lightgbm),
    ("Prophet", predict_prophet),
    ("TFT/MLP", predict_tft),
]

# ---------------------------------------------------------------------------
# Experiment core
# ---------------------------------------------------------------------------
def get_sparse_train(full_feat, val_year, availability):
    """Return the training rows for a given validation year and availability.

    Steps:
      1. Keep only feature rows STRICTLY BEFORE the validation year (no leakage).
      2. Keep the most-recent `availability` fraction of those rows, with a
         floor of MIN_TRAIN_ROWS so feature-based models still have data.
    """
    before = full_feat[full_feat["year"] < val_year]
    n_total = len(before)
    n_keep = max(int(round(n_total * availability)), MIN_TRAIN_ROWS)
    n_keep = min(n_keep, n_total)          # can't keep more than we have
    return before.tail(n_keep).reset_index(drop=True)


def run_experiment(full_feat):
    """Run the full seed x availability x model experiment.

    For each (seed, availability, model) combination, walk-forward validate
    over VAL_YEARS: for each validation year, train on the sparse training
    subset and predict that year, then compute MAPE across the validation
    years. Returns a tidy DataFrame with one row per combination.
    """
    from sklearn.metrics import mean_absolute_percentage_error
    records = []
    total = len(SEEDS) * len(AVAILABILITY_LEVELS) * len(MODELS)
    done = 0

    for seed in SEEDS:
        for availability in AVAILABILITY_LEVELS:
            for model_name, model_fn in MODELS:
                actuals, preds = [], []
                for val_year in VAL_YEARS:
                    train = get_sparse_train(full_feat, val_year, availability)
                    target = full_feat[full_feat["year"] == val_year]
                    if target.empty or train.empty:
                        continue
                    actual = float(target["VALUE"].iloc[0])
                    try:
                        pred = model_fn(train, target, seed)
                    except Exception:
                        pred = float(train["VALUE"].iloc[-1])  # persistence
                    actuals.append(actual)
                    preds.append(pred)

                if actuals:
                    mape = mean_absolute_percentage_error(actuals, preds) * 100
                    n_train = len(get_sparse_train(full_feat, VAL_YEARS[0],
                                                   availability))
                    records.append({
                        "model": model_name,
                        "availability_pct": int(availability * 100),
                        "n_train_rows": n_train,
                        "seed": seed,
                        "mape": mape,
                    })
                done += 1
                print(f"\r  progress: {done}/{total} combinations", end="")

    print()  # newline after progress
    return pd.DataFrame(records)

def summarise(results):
    """Aggregate raw per-seed results into median + IQR per model x level.

    For each (model, availability) we report across the SEEDS:
      median MAPE, 25th percentile (q1), 75th percentile (q3),
      and IQR = q3 - q1 (the spread — the whole point of the seed sweep).
    """
    grouped = results.groupby(["model", "availability_pct"])["mape"]
    summary = grouped.agg(
        median_mape="median",
        q1=lambda s: s.quantile(0.25),
        q3=lambda s: s.quantile(0.75),
        min_mape="min",
        max_mape="max",
    ).reset_index()
    summary["iqr"] = summary["q3"] - summary["q1"]
    summary = summary.sort_values(["model", "availability_pct"])
    return summary

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_results(summary, path=OUT_DIR / "sparsity_robustness_plot.png"):
    """Plot median MAPE vs data availability, one line per model, with IQR bands."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Order models by their 100%-data median so the legend reads sensibly
    order = (summary[summary["availability_pct"] == 100]
             .sort_values("median_mape")["model"].tolist())

    for model_name in order:
        sub = summary[summary["model"] == model_name].sort_values("availability_pct")
        x = sub["availability_pct"].values
        med = sub["median_mape"].values
        q1 = sub["q1"].values
        q3 = sub["q3"].values
        line, = ax.plot(x, med, marker="o", linewidth=1.8, label=model_name)
        # Shade IQR only where there is spread (deterministic models have none)
        if (q3 - q1).max() > 0.05:
            ax.fill_between(x, q1, q3, alpha=0.15, color=line.get_color())

    ax.set_xlabel("Training-data availability (%)")
    ax.set_ylabel("MAPE (%)  —  median across 10 seeds")
    ax.set_title("Forecasting robustness under data-availability degradation\n"
                 "(median MAPE, shaded bands = inter-quartile range)")
    ax.set_xticks([25, 50, 75, 100])
    ax.invert_xaxis()  # show 100% on the left, degrading to 25% on the right
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_series()
    feat = make_features(df)
    print(f"Loaded {len(df)} rows, {len(feat)} after features")
    n_runs = len(SEEDS) * len(AVAILABILITY_LEVELS) * len(MODELS)
    print(f"Running experiment: {len(SEEDS)} seeds x "
          f"{len(AVAILABILITY_LEVELS)} levels x {len(MODELS)} models "
          f"= {n_runs} runs\n")

    results = run_experiment(feat)
    summary = summarise(results)

    OUT_DIR.mkdir(exist_ok=True)
    results.to_csv(OUT_DIR / "sparsity_robustness_raw.csv", index=False)
    summary.to_csv(OUT_DIR / "sparsity_robustness.csv", index=False)

    print("\n=== MEDIAN MAPE (with IQR) by model x availability ===\n")
    for model_name, _ in MODELS:
        sub = summary[summary["model"] == model_name]
        print(f"{model_name}:")
        for _, r in sub.iterrows():
            print(f"    {int(r['availability_pct']):>3}% data: "
                  f"median {r['median_mape']:5.1f}%  "
                  f"IQR [{r['q1']:4.1f}, {r['q3']:4.1f}]  "
                  f"(spread {r['iqr']:4.1f})")
        print()

    plot_path = plot_results(summary)

    print(f"Saved: {OUT_DIR/'sparsity_robustness.csv'}")
    print(f"Saved: {OUT_DIR/'sparsity_robustness_raw.csv'}")
    print(f"Saved: {plot_path}")
