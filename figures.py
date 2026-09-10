"""
figures.py
==========
Generates the data-driven figures for the CDW paper, so that every figure is
reproducible from the pipeline outputs rather than drawn by hand. Reads the
CSVs written to out/ and saves the figures to out/figures/.

Produces:
  fig2_clusters.png   - facility-year archetypes (PCA projection)
  fig3_anomalies.png  - Isolation Forest scores, with the IMS trajectory inset
  fig_recovery.png    - recovery rate vs the EU 70% target

Run Command:  python figures.py
"""

import matplotlib as mpl
mpl.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("out")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "font.family": "serif", "figure.dpi": 150})

# Relabel the two recycler clusters so the backfilling one is distinct
def archetype_label(row):
    if row["cluster"] == 2:
        return "Backfilling Recyclers"
    if row["cluster"] == 0:
        return "General Recyclers"
    return row["archetype"]

# ---------------------------------------------------------------------------
# FIGURE 2 - Facility archetypes (PCA projection of 504 facility-years)
# ---------------------------------------------------------------------------
def fig_clusters():
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    fc = pd.read_csv(OUT / "facility_clusters.csv")
    fc["label"] = fc.apply(archetype_label, axis=1)

    feats = ["recovery_share","disposal_share","other_share","backfill_share",
             "hazardous_share","n_low_codes","n_records"]
    X = fc[feats].copy()
    X["log_t"] = np.log1p(fc["total_accepted_t"])
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(Xs)
    fc["pc1"], fc["pc2"] = xy[:,0], xy[:,1]

    fig, ax = plt.subplots(figsize=(7, 5))
    order = ["General Recyclers","Backfilling Recyclers","Transfer/storage",
             "Disposal/Landfill","Small transfer/storage"]
    colors = {"General Recyclers":"#2ca02c","Backfilling Recyclers":"#9467bd",
              "Transfer/storage":"#1f77b4","Disposal/Landfill":"#d62728",
              "Small transfer/storage":"#ff7f0e"}
    for lab in order:
        sub = fc[fc["label"]==lab]
        if len(sub)==0: continue
        ax.scatter(sub["pc1"], sub["pc2"], s=28, alpha=0.7,
                   c=colors.get(lab,"#777"), label=f"{lab} (n={len(sub)})",
                   edgecolors="white", linewidths=0.4)
    # mark anomalies
    an = fc[fc["is_anomaly"]]
    ax.scatter(an["pc1"], an["pc2"], s=90, facecolors="none",
               edgecolors="black", linewidths=1.2, label=f"Anomaly (n={len(an)})")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}% var)")
    ax.set_title("Facility-year behavioural archetypes (504 obs, KMeans k=5)")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG/"fig2_clusters.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig2_clusters.png")

# ---------------------------------------------------------------------------
# FIGURE 3 - Anomaly scores + IMS trajectory inset
# ---------------------------------------------------------------------------
def fig_anomalies():
    fc = pd.read_csv(OUT / "facility_clusters.csv").sort_values("anomaly_score")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = ["#d62728" if a else "#4477aa" for a in fc["is_anomaly"]]
    ax.bar(range(len(fc)), fc["anomaly_score"], color=colors, width=1.0)
    ax.axhline(0, color="black", linewidth=0.6)
    n_anom = int(fc["is_anomaly"].sum())
    ax.set_xlabel("Facility-year (sorted by anomaly score)")
    ax.set_ylabel("Isolation Forest score")
    ax.set_title(f"Anomaly detection: {n_anom} of {len(fc)} facility-years flagged (5% contamination)")
    # inset: IMS trajectory
    ims = pd.read_csv(OUT/"facility_clusters.csv")
    ims = ims[ims["licensee"].str.contains("Integrated Materials", na=False)].sort_values("year")
    if len(ims):
        iax = fig.add_axes([0.58, 0.55, 0.33, 0.30])
        iax.plot(ims["year"], ims["recovery_share"]*100, marker="o", color="#d62728")
        iax.set_title("IMS recovery share", fontsize=7)
        iax.set_ylabel("%", fontsize=6); iax.tick_params(labelsize=6)
        iax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG/"fig3_anomalies.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig3_anomalies.png")

# ---------------------------------------------------------------------------
# FIGURE (recovery rate trend) - supporting
# ---------------------------------------------------------------------------
def fig_recovery():
    kpi = pd.read_csv(OUT/"kpi_recovery_rate.csv")
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(kpi["year"], kpi["recovery_rate"]*100, marker="o", linewidth=2, color="#2ca02c")
    ax.axhline(70, color="#d62728", linestyle="--", label="EU 70% target")
    for _,r in kpi.iterrows():
        ax.annotate(f"{r['recovery_rate']*100:.1f}%", (r["year"], r["recovery_rate"]*100),
                    textcoords="offset points", xytext=(0,8), ha="center", fontsize=8)
    ax.set_xlabel("Year"); ax.set_ylabel("Recovery rate (%)")
    ax.set_title("Licensee-population recovery rate vs EU target")
    ax.set_xticks(kpi["year"]); ax.set_ylim(60, 105)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG/"fig_recovery.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_recovery.png")

# ---------------------------------------------------------------------------
# FIGURE — Unified model comparison (classical + modern, full data)
# ---------------------------------------------------------------------------
def fig_model_comparison():
    import numpy as np
    # Classical models: full-data (100%) median MAPE from the sparsity experiment
    sp = pd.read_csv(OUT / "sparsity_robustness.csv")
    full = sp[sp["availability_pct"] == 100].copy()
    classical = dict(zip(full["model"], full["median_mape"]))
    classical_iqr = dict(zip(full["model"], full["q3"] - full["q1"]))

    # Modern deep models: from modern_models.py summary
    mm = pd.read_csv(OUT / "modern_models_summary.csv")   # columns: model, median_mape, q1, q3, iqr_spread
    for _, r in mm.iterrows():
        classical[r["model"]] = r["median_mape"]
        classical_iqr[r["model"]] = r["iqr_spread"]

    # Naive baseline (constant reference)
    NAIVE = 8.43

    # Assemble, sort ascending by MAPE
    rows = [(m, v, classical_iqr.get(m, 0.0)) for m, v in classical.items()]
    rows.append(("Naive 3-yr MA", NAIVE, 0.0))
    rows.sort(key=lambda t: t[1])
    names  = [r[0] for r in rows]
    mape   = [r[1] for r in rows]
    iqr    = [r[2] for r in rows]

    # Family colours (deep models = purple so they stand out)
    deep = {"LSTM", "RNN", "GRU", "TFT/MLP"}
    def colour(n):
        if n == "Naive 3-yr MA": return "#888888"
        if n in deep:            return "#9467bd"
        return "#4477aa"
    colors = [colour(n) for n in names]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(names))
    ax.bar(x, mape, yerr=iqr, capsize=3, color=colors, edgecolor="white", linewidth=0.6)
    ax.axhline(NAIVE, color="#d62728", linestyle="--", linewidth=1.3, zorder=0)
    ax.text(len(names) - 0.4, NAIVE + 0.8, f"Naive baseline ({NAIVE}%)",
            color="#d62728", fontsize=8, ha="right", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("Median MAPE (%) across 10 seeds")
    ax.set_title("Forecasting model comparison on the 22-point annual series")
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="#4477aa", label="Classical"),
               Patch(facecolor="#9467bd", label="Deep (recurrent)"),
               Patch(facecolor="#888888", label="Naive baseline")]
    ax.legend(handles=handles, fontsize=8, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG / "fig_model_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_model_comparison.png")

def fig_combined_panel():
    """Two-panel figure: (a) cluster PCA, (b) anomaly scores. Saves page space."""
    import numpy as np, pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    fc = pd.read_csv(OUT / "facility_clusters.csv")
    def lab(r):
        if r["cluster"] == 2: return "Backfilling Recyclers"
        if r["cluster"] == 0: return "General Recyclers"
        return r["archetype"]
    fc["label"] = fc.apply(lab, axis=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4.72, 2.1))

    # Panel (a): PCA clusters
    feats = ["recovery_share","disposal_share","other_share","backfill_share",
             "hazardous_share","n_low_codes","n_records"]
    X = fc[feats].copy(); X["log_t"] = np.log1p(fc["total_accepted_t"])
    xy = PCA(n_components=2, random_state=42).fit_transform(StandardScaler().fit_transform(X))
    fc["pc1"], fc["pc2"] = xy[:,0], xy[:,1]
    order = ["General Recyclers","Backfilling Recyclers","Transfer/storage",
             "Disposal/Landfill","Small transfer/storage"]
    colors = {"General Recyclers":"#2ca02c","Backfilling Recyclers":"#9467bd",
              "Transfer/storage":"#1f77b4","Disposal/Landfill":"#d62728",
              "Small transfer/storage":"#ff7f0e"}
    for l in order:
        s = fc[fc["label"]==l]
        if len(s): ax1.scatter(s["pc1"], s["pc2"], s=22, alpha=0.7, c=colors.get(l,"#777"),
                               label=f"{l} (n={len(s)})", edgecolors="white", linewidths=0.3)
    an = fc[fc["is_anomaly"]]
    ax1.scatter(an["pc1"], an["pc2"], s=70, facecolors="none", edgecolors="black",
                linewidths=1.0, label=f"Anomaly (n={len(an)})")
    ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2")
    ax1.set_title("(a) Behavioural archetypes (KMeans k=5)", fontsize=8, fontweight="bold")
    ax1.legend(fontsize=5.5, loc="best", framealpha=0.85, handletextpad=0.3, borderpad=0.3)
    ax1.grid(True, alpha=0.3)

    # Panel (b): anomaly scores
    fcs = fc.sort_values("anomaly_score")
    bar_colors = ["#d62728" if a else "#4477aa" for a in fcs["is_anomaly"]]
    ax2.bar(range(len(fcs)), fcs["anomaly_score"], color=bar_colors, width=1.0)
    ax2.axhline(0, color="black", linewidth=0.6)
    ax2.set_xlabel("Facility-year (sorted by score)"); ax2.set_ylabel("Isolation Forest score")
    ax2.set_title(f"(b) Anomaly detection: {int(fc['is_anomaly'].sum())} of {len(fc)} flagged",
                  fontsize=10, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=6)
        ax.xaxis.label.set_size(7)
        ax.yaxis.label.set_size(7)

    plt.tight_layout()
    fig.savefig(FIG / "fig_combined_panel.png", dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("Saved fig_combined_panel.png")

if __name__ == "__main__":
    fig_clusters()
    fig_anomalies()
    fig_recovery()
    fig_model_comparison()
    fig_combined_panel()
    print(f"\nAll figures written to {FIG.resolve()}/")