"""
modern_models.py
================
Extends the C&D waste forecasting comparison with modern deep-learning
sequence models (LSTM, RNN, GRU), evaluated on the same 22-point annual
series, with the same walk-forward validation and MAPE metric as the
classical models in sparsity_robustness.py.

PURPOSE (read before running):
  This is an HONEST comparative extension. On a 22-point annual series,
  deep-learning sequence models are expected to UNDERPERFORM simple
  baselines -- they are data-hungry and overfit on tiny samples. That is
  the finding: it demonstrates empirically that on sparse annual data,
  model complexity does not help, reinforcing the paper's core result
  that forecast error is governed by training-window composition, not
  model sophistication. We are NOT claiming these models win; we are
  characterising why they do not, which directly answers the question
  "why not use newer models?".

Requirements: tensorflow (or keras), numpy, pandas, scikit-learn
  pip install tensorflow scikit-learn pandas numpy
  (If on Renku, request a small GPU/CPU session -- these are tiny models.)

Input : out/annual_cdw_series.csv   (produced by the forecasting notebook)
Output: out/modern_models_results.csv
        out/modern_models_comparison.png

Run:  python modern_models.py
"""
from __future__ import annotations
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # quiet TensorFlow logs
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("out")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Validation years -- must match sparsity_robustness.py for a fair comparison
VAL_YEARS = [2021, 2022, 2023]
N_LAGS = 2          # use the last 2 years to predict the next (matches lag features)
SEEDS = list(range(10))   # same 10-seed protocol as the classical experiment


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def load_series() -> pd.DataFrame:
    """Load the annual CDW series produced by the forecasting notebook."""
    df = pd.read_csv(OUT / "annual_cdw_series.csv")
    df = df.sort_values("year").reset_index(drop=True)
    return df


def make_supervised(values: np.ndarray, n_lags: int = N_LAGS):
    """Turn a 1-D series into (X, y) sequence pairs for supervised learning.

    Each sample is `n_lags` consecutive years used to predict the next year.
    Returns X shaped (samples, n_lags, 1) as required by Keras RNN layers.
    """
    X, y = [], []
    for i in range(len(values) - n_lags):
        X.append(values[i:i + n_lags])
        y.append(values[i + n_lags])
    X = np.array(X).reshape(-1, n_lags, 1)
    y = np.array(y)
    return X, y


def mape(actual: float, predicted: float) -> float:
    """Absolute percentage error for a single point (series is one value/yr)."""
    return abs(actual - predicted) / abs(actual) * 100.0


# ---------------------------------------------------------------------------
# Model builders (kept deliberately small -- 22 points cannot support more)
# ---------------------------------------------------------------------------
def build_model(kind: str, n_lags: int, seed: int):
    """Build a small single-layer recurrent model of the requested kind.

    We keep these intentionally minimal (8 units, 1 layer) because the series
    is tiny; larger networks overfit instantly. The seed fixes weight
    initialisation so we can measure seed-sensitivity, exactly as the paper
    does for the classical models.
    """
    import tensorflow as tf
    tf.random.set_seed(seed)
    np.random.seed(seed)

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, SimpleRNN, GRU, Dense, Input

    layer = {"LSTM": LSTM, "RNN": SimpleRNN, "GRU": GRU}[kind]
    model = Sequential([
        Input(shape=(n_lags, 1)),
        layer(8, activation="tanh"),   # 8 units -- small on purpose
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


# ---------------------------------------------------------------------------
# Walk-forward evaluation (mirrors the classical experiment)
# ---------------------------------------------------------------------------
def evaluate_model(kind: str, df: pd.DataFrame, seed: int) -> list[dict]:
    """Walk-forward: for each validation year, train only on prior years."""
    import tensorflow as tf
    results = []
    series = df["VALUE"].to_numpy(dtype=float)
    years = df["year"].to_numpy()

    # Scale to stabilise training (fit on full series; small-data pragmatic choice)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()

    for val_year in VAL_YEARS:
        idx = int(np.where(years == val_year)[0][0])
        # Train only on data strictly before the validation year (no leakage)
        train_scaled = scaled[:idx]
        if len(train_scaled) <= N_LAGS:
            continue
        X_tr, y_tr = make_supervised(train_scaled, N_LAGS)

        # The input to predict val_year is the N_LAGS years immediately before it
        x_input = scaled[idx - N_LAGS:idx].reshape(1, N_LAGS, 1)

        model = build_model(kind, N_LAGS, seed)
        model.fit(X_tr, y_tr, epochs=200, verbose=0, batch_size=4)

        pred_scaled = float(model.predict(x_input, verbose=0)[0, 0])
        pred = float(scaler.inverse_transform([[pred_scaled]])[0, 0])
        actual = float(series[idx])

        results.append({
            "model": kind, "seed": seed, "val_year": val_year,
            "actual": actual, "predicted": pred, "mape": mape(actual, pred),
        })
        tf.keras.backend.clear_session()
    return results


def run():
    df = load_series()
    print(f"Loaded {len(df)} annual points ({df['year'].min()}-{df['year'].max()})")
    print(f"Running LSTM / RNN / GRU x {len(SEEDS)} seeds, walk-forward on {VAL_YEARS}\n")

    all_rows = []
    for kind in ["LSTM", "RNN", "GRU"]:
        for seed in SEEDS:
            all_rows.extend(evaluate_model(kind, df, seed))
        # per-model summary
        sub = pd.DataFrame([r for r in all_rows if r["model"] == kind])
        med = sub.groupby("seed")["mape"].mean().median()
        iqr = sub.groupby("seed")["mape"].mean()
        spread = iqr.quantile(0.75) - iqr.quantile(0.25)
        print(f"{kind:5s}: median MAPE {med:5.1f}%  |  seed IQR spread {spread:4.1f} pp")

    res = pd.DataFrame(all_rows)
    res.to_csv(OUT / "modern_models_results.csv", index=False)

    # Summary table: median MAPE + IQR per model (comparable to Table IV / V)
    summary = (res.groupby(["model", "seed"])["mape"].mean()
                 .groupby("model")
                 .agg(median_mape="median",
                      q1=lambda s: s.quantile(0.25),
                      q3=lambda s: s.quantile(0.75))
                 .round(2))
    summary["iqr_spread"] = (summary["q3"] - summary["q1"]).round(2)
    summary.to_csv(OUT / "modern_models_summary.csv")
    print("\n=== SUMMARY (median MAPE across 10 seeds) ===")
    print(summary.to_string())

    # Comparison plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(6, 4))
    order = summary.sort_values("median_mape").index.tolist()
    ax.bar(order, summary.loc[order, "median_mape"],
           yerr=summary.loc[order, "iqr_spread"], capsize=4, color="#4477aa")
    ax.axhline(8.43, color="#d62728", linestyle="--",
               label="Naive 3-yr MA baseline (8.43%)")
    ax.set_ylabel("Median MAPE (%) across 10 seeds")
    ax.set_title("Modern sequence models on the 22-point series")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(FIG / "modern_models_comparison.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: {OUT/'modern_models_results.csv'}")
    print(f"Saved: {OUT/'modern_models_summary.csv'}")
    print(f"Saved: {FIG/'modern_models_comparison.png'}")


if __name__ == "__main__":
    run()
