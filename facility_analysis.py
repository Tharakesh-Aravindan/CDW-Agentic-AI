"""
Facility-Level ML Analysis for CDW Licensees
============================================
Unsupervised machine-learning module that restructures the cleaned CDW
record-level data into FACILITY-YEAR observations and applies:

  1. K-Means clustering  -> discovers behavioural archetypes of waste
     facilities (true recyclers, landfills, transfer stations, ...),
     with k selected by silhouette score (not chosen arbitrarily).

  2. Isolation Forest    -> flags anomalous facility-years (facilities
     behaving unusually for their type / changing sharply year-to-year),
     which feeds the ComplianceMonitor agent's review queue.

Why facility-year units?
  The record-level table has ~4,955 rows but only 118 facilities. Collapsed
  to annual national totals it is just 5 points. The honest unit of analysis
  for behavioural ML is the FACILITY-YEAR: each facility's behaviour in each
  reporting year. This yields ~500 observations with engineered features --
  a defensible sample size for unsupervised learning.

Methodological notes (for the dissertation):
  - Features are engineered from the pipeline's `effective_tier` so the ML
    is consistent with the pipeline's treatment-classification choices
    (including backfilling counted as recovery, R12/R13 as 'other').
  - Tonnage is log-transformed (log1p) because it spans 0 to ~1.5M tonnes.
  - Clustering uses standardised features; k chosen by silhouette score.
  - There is repeated-measures structure (a facility appears up to 5 times).
    This is acceptable for clustering (each facility-year is a legitimate
    behavioural snapshot) but any future SUPERVISED model must split
    train/test BY FACILITY to avoid leakage.

Inputs
  out/cdw_clean.parquet   (produced by cdw_pipeline.py)

Outputs (written to out/)
  facility_year_features.csv   -- the engineered facility-year table
  facility_clusters.csv        -- facility-year + cluster label + anomaly flag
  cluster_profiles.csv         -- mean feature values per cluster (for writeup)
  silhouette_scores.csv        -- k-selection evidence (for writeup)

Run:
  python facility_analysis.py
  python facility_analysis.py --k 4 --contamination 0.05
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest

OUT_DIR = Path("./out")
CLEAN_PARQUET = OUT_DIR / "cdw_clean.parquet"

# Features used for clustering & anomaly detection.
SHARE_FEATURES = [
    "recovery_share",
    "disposal_share",
    "other_share",
    "backfill_share",
    "hazardous_share",
]
COUNT_FEATURES = ["n_low_codes", "n_records"]
# log_tonnage is added separately (log-transformed)


# Map the pipeline's detailed effective_tier labels to broad families.
# Robust to variations: matches on the leading keyword, and treats
# prep/storage (R12/R13) as 'other' since the final fate is unknown.
def tier_family(label: str) -> str:
    s = str(label).lower()
    if "prep_storage" in s or "prep-storage" in s:
        return "other"
    if s.startswith("recovery"):
        return "recovery"
    if s.startswith("disposal"):
        return "disposal"
    return "unknown"


# ---------------------------------------------------------------------------
# 1. Feature engineering: record-level CDW -> facility-year observations
# ---------------------------------------------------------------------------
def build_facility_year(cdw: pd.DataFrame) -> pd.DataFrame:
    """Collapse the record-level CDW table into one row per facility-year.

    Expects columns: licensee, year, accepted_t, effective_tier, low_code,
    low_desc, and (optionally) onsite_treatment / final_treatment for the
    backfilling flag.
    """
    df = cdw.copy()

    # Backfilling flag: R05 backfilling specifically, from either treatment col.
    treat_cols = [c for c in ("final_treatment", "onsite_treatment") if c in df.columns]
    if treat_cols:
        backfill_text = (
            df[treat_cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
        )
        df["_is_backfill"] = backfill_text.str.contains("backfilling", na=False)
    else:
        df["_is_backfill"] = False

    # Hazardous flag: asterisk in LoW code or 'hazardous' in description.
    df["_is_hazardous"] = (
        df["low_code"].astype(str).str.contains(r"\*", na=False)
        | df["low_desc"].astype(str).str.contains("hazardous", case=False, na=False)
    )

    # Collapse the pipeline's detailed tier labels into broad families.
    df["_tier"] = df["effective_tier"].apply(tier_family)

    def _agg(g: pd.DataFrame) -> pd.Series:
        total = g["accepted_t"].sum()
        rec = g.loc[g["_tier"] == "recovery", "accepted_t"].sum()
        dis = g.loc[g["_tier"] == "disposal", "accepted_t"].sum()
        oth = g.loc[g["_tier"] == "other", "accepted_t"].sum()
        bf = g.loc[g["_is_backfill"], "accepted_t"].sum()
        haz = g.loc[g["_is_hazardous"], "accepted_t"].sum()
        denom = total if total > 0 else np.nan
        rec_dis = rec + dis
        return pd.Series({
            "total_accepted_t": total,
            "n_records": len(g),
            "n_low_codes": g["low_code"].nunique(),
            "recovery_share": (rec / denom) if total > 0 else 0.0,
            "disposal_share": (dis / denom) if total > 0 else 0.0,
            "other_share": (oth / denom) if total > 0 else 0.0,
            "backfill_share": (bf / denom) if total > 0 else 0.0,
            "hazardous_share": (haz / denom) if total > 0 else 0.0,
            # recovery_rate excludes 'other'; NaN when fate is entirely unknown
            "recovery_rate": (rec / rec_dis) if rec_dis > 0 else np.nan,
        })

    fy = df.groupby(["licensee", "year"], group_keys=False).apply(
        _agg, include_groups=False
    ).reset_index()
    return fy


# ---------------------------------------------------------------------------
# 2. Build the standardised feature matrix
# ---------------------------------------------------------------------------
def make_matrix(fy: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    X = fy[SHARE_FEATURES + COUNT_FEATURES].copy()
    X["log_tonnage"] = np.log1p(fy["total_accepted_t"])
    cols = SHARE_FEATURES + COUNT_FEATURES + ["log_tonnage"]
    Xs = StandardScaler().fit_transform(X[cols].values)
    return Xs, cols


# ---------------------------------------------------------------------------
# 3. Choose k by silhouette score
# ---------------------------------------------------------------------------
def select_k(Xs: np.ndarray, k_range=range(2, 7)) -> pd.DataFrame:
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
        rows.append({"k": k, "silhouette": round(silhouette_score(Xs, km.labels_), 4)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Main analysis
# ---------------------------------------------------------------------------
def run(k: int | None = None, contamination: float = 0.05) -> dict:
    if not CLEAN_PARQUET.exists():
        raise FileNotFoundError(
            f"{CLEAN_PARQUET} not found. Run cdw_pipeline.py first."
        )
    cdw = pd.read_parquet(CLEAN_PARQUET)

    fy = build_facility_year(cdw)
    Xs, cols = make_matrix(fy)

    # --- k selection ---
    sil = select_k(Xs)
    OUT_DIR.mkdir(exist_ok=True)
    sil.to_csv(OUT_DIR / "silhouette_scores.csv", index=False)
    if k is None:
        # default to the smallest k within 0.02 of the best score (parsimony)
        best = sil.loc[sil["silhouette"].idxmax()]
        good = sil[sil["silhouette"] >= best["silhouette"] - 0.02]
        k = int(good["k"].min())

    # --- clustering ---
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
    fy["cluster"] = km.labels_

    # --- anomaly detection ---
    iso = IsolationForest(contamination=contamination, random_state=42).fit(Xs)
    fy["anomaly_score"] = iso.decision_function(Xs)
    fy["is_anomaly"] = iso.predict(Xs) == -1

    # --- cluster profiles (for dissertation table) ---
    prof = fy.groupby("cluster")[
        SHARE_FEATURES + ["total_accepted_t", "n_low_codes", "n_records"]
    ].mean()
    prof["n_facility_years"] = fy.groupby("cluster").size()
    prof = prof.round(3)

    # --- label clusters with human-readable archetypes ---
    def archetype(r) -> str:
        if r["recovery_share"] >= 0.6:
            return "Recycler/Recovery"
        if r["disposal_share"] >= 0.6:
            return "Disposal/Landfill"
        if r["other_share"] >= 0.6:
            if r["total_accepted_t"] < 10_000:
                return "Small transfer/storage"
            return "Transfer/storage"
        return "Mixed"
    prof["archetype"] = prof.apply(archetype, axis=1)
    label_map = prof["archetype"].to_dict()
    fy["archetype"] = fy["cluster"].map(label_map)

    # --- save outputs ---
    fy.to_csv(OUT_DIR / "facility_clusters.csv", index=False)
    fy.drop(columns=["cluster", "anomaly_score", "is_anomaly", "archetype"]).to_csv(
        OUT_DIR / "facility_year_features.csv", index=False
    )
    prof.reset_index().to_csv(OUT_DIR / "cluster_profiles.csv", index=False)

    return {
        "n_facility_years": len(fy),
        "n_facilities": fy["licensee"].nunique(),
        "k_selected": k,
        "silhouette": sil,
        "profiles": prof,
        "n_anomalies": int(fy["is_anomaly"].sum()),
        "anomalies": fy[fy["is_anomaly"]].sort_values("anomaly_score"),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Facility-level ML analysis of CDW licensees.")
    p.add_argument("--k", type=int, default=None,
                   help="Number of clusters (default: auto-select by silhouette)")
    p.add_argument("--contamination", type=float, default=0.05,
                   help="Expected anomaly fraction for Isolation Forest (default 0.05)")
    args = p.parse_args()

    res = run(k=args.k, contamination=args.contamination)

    print(f"Facility-year observations : {res['n_facility_years']}")
    print(f"Distinct facilities        : {res['n_facilities']}")
    print(f"Clusters (k)               : {res['k_selected']}")
    print("\n--- Silhouette scores (k selection) ---")
    print(res["silhouette"].to_string(index=False))
    print("\n--- Cluster profiles ---")
    print(res["profiles"].to_string())
    print(f"\n--- Anomalies flagged: {res['n_anomalies']} ---")
    cols = ["licensee", "year", "total_accepted_t",
            "recovery_share", "disposal_share", "other_share", "archetype"]
    print(res["anomalies"][cols].head(15).to_string(index=False))
    print(f"\nOutputs written to {OUT_DIR.resolve()}/")
    print("  facility_year_features.csv, facility_clusters.csv,")
    print("  cluster_profiles.csv, silhouette_scores.csv")
