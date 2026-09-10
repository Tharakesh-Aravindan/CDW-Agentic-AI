"""Poster figures for DCU FEC Research Day 2026 (A1, 594 x 841 mm portrait).

Every figsize is the PHYSICAL printed size in inches, so a fontsize of 22 here
renders as 22pt on the printed poster. Do not scale these in PowerPoint --
drop them in at 100% and they will be correct.

Run:  pip install qrcode[pil]
      python poster_figures.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path("out")            # where facility_analysis.py writes its CSVs
OUT  = Path("poster"); OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

BLUE, ORANGE, INK, SLATE = "#4878a8", "#d98c2b", "#222222", "#5a6b7d"

def find_csv(name):
    for p in (Path(name), Path("out")/name, Path("../out")/name):
        if p.exists():
            print(f"reading {p.resolve()}")
            return p
    raise FileNotFoundError(f"{name} not in ., ./out, or ../out")

def hero_chart():
    """Centrepiece: reported vs statutory recovery rate, 2021-2025."""
    years     = ["2021", "2022", "2023", "2024", "2025"]
    published = [74.1, 79.9, 71.4, 90.9, 98.5]
    statutory = [57.8, 60.9, 63.6, 73.6, 60.2]

    x = np.arange(len(years)); w = 0.38
    fig, ax = plt.subplots(figsize=(21.0, 6.5))

    b1 = ax.bar(x - w/2, published, w, color=BLUE,
                label="As reported by licensed facilities")
    b2 = ax.bar(x + w/2, statutory, w, color=ORANGE,
                label="Counted as EU law requires")

    ax.axhline(70, color=INK, ls="--", lw=3.5, zorder=3)
    ax.text(-0.45, 71.5, "EU target: 70%", ha="left",
            fontsize=26, fontweight="bold", color=INK)

    for bars in (b1, b2):
        for bar in bars:
            ax.annotate(f"{bar.get_height():.1f}%",
                        (bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=26, fontweight="bold", color=INK)

    ax.set_xticks(x); ax.set_xticklabels(years, fontsize=30)
    ax.set_ylabel("Recovery rate (%)", fontsize=28)
    ax.set_ylim(0, 118)
    ax.tick_params(axis="y", labelsize=24)
    ax.legend(fontsize=26, loc="upper center", frameon=False, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(1.8)
    ax.grid(axis="y", alpha=0.15); ax.set_axisbelow(True)

    fig.savefig(OUT / "hero_recovery.png"); plt.close(fig)
    print("saved hero_recovery.png")


def cluster_scatter():
    """Facility archetypes in PCA space, poster scale.

    Colours by `cluster` (5 groups), not `archetype` -- the archetype
    labeller in facility_analysis.py gives clusters 0 and 2 the same
    name, which merges General Recyclers (118) and Backfilling-Dominated
    (46) into one legend entry of n=164.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    fc = pd.read_csv(find_csv("facility_clusters.csv"))

    # --- name each cluster from its own treatment profile -------------
    prof = fc.groupby("cluster")[
        ["recovery_share", "backfill_share", "disposal_share", "other_share"]
    ].mean()

    def name(r):
        if r.backfill_share > 0.50:  return "Backfilling-Dominated"
        if r.recovery_share > 0.50:  return "General Recyclers"
        if r.disposal_share > 0.50:  return "Disposal / Landfill"
        if r.other_share    > 0.90:  return "Transfer / Storage"
        return "Small Storage"

    labels = {cid: name(row) for cid, row in prof.iterrows()}
    counts = fc["cluster"].value_counts().to_dict()

    print("\n--- clusters (must match Table 2: 118/46/153/40/147) ---")
    for cid in sorted(labels):
        print(f"  cluster {cid}: {labels[cid]:<24} n = {counts[cid]}")
    print()

    # --- PCA ----------------------------------------------------------
    feats = ["recovery_share", "disposal_share", "other_share",
             "backfill_share", "hazardous_share", "n_low_codes", "n_records"]
    X = fc[feats].copy()
    X["log_t"] = np.log1p(fc["total_accepted_t"])
    P = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X))

    # --- plot: fixed colour per archetype, hero groups drawn last -----
    palette = {"General Recyclers":     "#1f77b4",   # blue
               "Backfilling-Dominated": "#ff7f0e",   # orange
               "Transfer / Storage":    "#2ca02c",   # green
               "Disposal / Landfill":   "#d62728",   # red
               "Small Storage":         "#9467bd"}   # purple
    draw_order = ["Transfer / Storage", "Small Storage", "Disposal / Landfill",
                  "General Recyclers", "Backfilling-Dominated"]
    legend_order = ["General Recyclers", "Backfilling-Dominated",
                    "Transfer / Storage", "Disposal / Landfill", "Small Storage"]

    fig, ax = plt.subplots(figsize=(7.0, 3.9))   # = 17.8 x 11.7 cm printed
    handles = {}
    for arch in draw_order:
        cids = [c for c, l in labels.items() if l == arch]
        if not cids:
            continue
        m = fc["cluster"].isin(cids).values
        h = ax.scatter(P[m, 0], P[m, 1], s=70, c=palette[arch], alpha=0.85,
                       edgecolors="white", linewidths=0.5,
                       label=f"{arch} (n={m.sum()})")
        handles[arch] = h

    ax.set_xlabel("PC1", fontsize=13)
    ax.set_ylabel("PC2", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.legend([handles[a] for a in legend_order if a in handles],
              [handles[a].get_label() for a in legend_order if a in handles],
              fontsize=11.5, loc="upper left", frameon=True, framealpha=0.92,
              borderpad=0.5, labelspacing=0.35, handletextpad=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.15)
    ax.set_axisbelow(True)
    plt.tight_layout()

    fig.savefig(OUT / "clusters_v2.png", dpi=400,
                bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("saved clusters_v2.png")


def pipeline_diagram():
    """Poster-scale redraw of the system pipeline."""
    rows = [
        ("DATA INGESTION", "#eef3f8", [
            ("EPA Licensee", "4,955 C&D records"),
            ("CSO + Archive", "national series"),
            ("Policy timeline", "EoW / by-product"),
            ("Eurostat", "EU context")]),
        ("PREPROCESSING", "#eef3f8", [
            ("Treatment classification", "R/D codes to recovery / disposal / other"),
            ("Facility-year aggregation", "504 observations, 118 facilities")]),
        ("MACHINE LEARNING", "#fdf1e0", [
            ("KMeans", "k = 5 archetypes"),
            ("Isolation Forest", "26 flagged"),
            ("Forecasting", "11 models"),
            ("Triage", "RF / LogReg")]),
        ("GOVERNED AGENTS", "#e8f0e8", [
            ("ComplianceMonitor", "ok / needs_review / error"),
            ("Forecaster", "OLS + interval, deterministic")]),
        ("GOVERNANCE & AUDIT", "#f3e8ee", [
            ("Orchestrator: human-in-the-loop gate", "no autonomous approval")]),
        ("", "#f3e8ee", [
            ("Provenance audit log (append-only)",
             "code version, model spec, legal basis, SHA-256")]),
    ]

    fig, ax = plt.subplots(figsize=(10.5, 8.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    top, bottom = 92, 4
    n = len(rows)
    band = (top - bottom) / n
    box_h = band * 0.60

    for i, (label, colour, boxes) in enumerate(rows):
        y = top - band * i - box_h
        m = len(boxes); margin, gap = 2, 2
        bw = (100 - 2*margin - gap*(m-1)) / m
        for j, (t1, t2) in enumerate(boxes):
            x = margin + j*(bw + gap)
            ax.add_patch(FancyBboxPatch(
                (x, y), bw, box_h,
                boxstyle="round,pad=0.3,rounding_size=1.2",
                fc=colour, ec=SLATE, lw=1.5))
            ax.text(x + bw/2, y + box_h*0.63, t1, ha="center", va="center",
                    fontsize=13, fontweight="bold", color=INK)
            ax.text(x + bw/2, y + box_h*0.27, t2, ha="center", va="center",
                    fontsize=10, color=SLATE)
        if label:
            ax.text(margin, y + box_h + 0.9, label, ha="left", va="bottom",
                    fontsize=11, fontweight="bold", color=SLATE)
        if i < n - 1:
            ax.annotate("", xy=(50, top - band*(i+1)), xytext=(50, y),
                        arrowprops=dict(arrowstyle="-|>", lw=2.2, color=SLATE))

    fig.savefig(OUT / "pipeline.png"); plt.close(fig)
    print("saved pipeline.png")


def qr():
    import qrcode
    qrcode.make("https://anonymous.4open.science/r/cdw-governed-agents"
                ).save(OUT / "qr.png")
    print("saved qr.png")

def mape_chart():
    """Forecasting comparison. Naive baseline wins."""
    models = ["Prophet", "LightGBM", "XGBoost", "RNN", "LSTM",
              "ARIMA", "GRU", "Naive 3-yr average"]
    vals   = [58.14, 19.04, 17.41, 12.79, 11.98, 11.85, 9.70, 8.43]
    colors = [BLUE]*7 + [ORANGE]

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    bars = ax.barh(models, vals, color=colors)
    for bar, v in zip(bars, vals):
        ax.annotate(f"{v:.2f}", (v, bar.get_y() + bar.get_height()/2),
                    xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=17, fontweight="bold")
    ax.set_xlabel("Median MAPE (%) — lower is better", fontsize=18)
    ax.tick_params(labelsize=17)
    ax.set_xlim(0, 72)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(OUT / "mape.png")
    plt.close(fig)
    print("saved mape.png")

if __name__ == "__main__":
    hero_chart()
    cluster_scatter()
    pipeline_diagram()
    mape_chart()
    qr()
    