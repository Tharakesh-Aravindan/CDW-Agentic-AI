from __future__ import annotations

"""
CDW Data Pipeline
-----------------
Unifies the two data sources you have into a single tidy dataset focused on
construction & demolition waste (CDW), with derived indicators the agents
will reason over.

Sources
  1. EPA Licensee Waste Data (2021-2025) -- facility-level, granular
  2. CSO Waste Generation by NACE (2004-2020) -- national, sector-wide

CDW is identified by LoW (List of Waste) Chapter 17 in the EPA data
and by NACE sector F (Construction) in the CSO data.

Treatment codes follow EU Waste Framework Directive:
  R-codes = recovery (R1 energy, R2-R11 recycling/reuse, R12-R13 prep/storage)
  D-codes = disposal (D1, D5 landfill; D10 incineration without recovery; etc.)
"""

import re
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(".")
EPA_YEARS = [2021, 2022, 2023, 2024, 2025]


# ---------------------------------------------------------------------------
# Treatment-code classification
# ---------------------------------------------------------------------------
def classify_treatment(code_str: str) -> str:
    """
    Map a treatment description (EPA format) to a CE-relevant tier.
    Aligned with EU WFD waste hierarchy.
    """
    if not isinstance(code_str, str):
        return "unknown"
    s = code_str.strip().upper()
    # Match the leading R/D code
    m = re.match(r"([RD])\s*0?(\d+)", s)
    if not m:
        return "unknown"
    letter, num = m.group(1), int(m.group(2))
    if letter == "R":
        if num == 1:
            return "recovery_energy"          # R1
        if 2 <= num <= 11:
            return "recovery_recycling_reuse" # R2-R11 (incl. backfilling R5)
        if num in (12, 13):
            return "recovery_prep_storage"    # R12 exchange, R13 storage
    elif letter == "D":
        if num in (1, 5, 12):
            return "disposal_landfill"
        if num == 10:
            return "disposal_incineration"
        return "disposal_other"
    return "unknown"


# ---------------------------------------------------------------------------
# EPA loader
# ---------------------------------------------------------------------------
def load_epa() -> pd.DataFrame:
    frames = []
    for y in EPA_YEARS:
        path = PROJECT_DIR / f"EPA-Licensee-Waste-Data-{y}.xlsx"
        df = pd.read_excel(path, sheet_name=f"Reporting Year {y}")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    # Tidy column names
    df = df.rename(columns={
        "Reporting Year": "year",
        "Licence Profile": "licensee",
        "Waste Source": "waste_source",
        "Waste Action": "waste_action",
        "List of Waste Code": "low_code",
        "Low Code Description": "low_desc",
        "Respondent Waste Description": "respondent_desc",
        "Accepted Waste Quantity": "accepted_t",
        "Accepted waste quantity from abroad": "accepted_abroad_t",
        "Accepted Waste Quantity from Ireland": "accepted_ireland_t",
        "On Site Waste Treatment Type": "onsite_treatment",
        "Waste Quantity Tonnes per Year": "transferred_t",
        "Next Destination Waste Treatment Type": "next_treatment",
        "Final Destination Abroad": "final_abroad",
        "Final Destination Waste Treatment Type": "final_treatment",
    })

    # Derive LoW chapter (first 2 digits of code)
    df["low_code"] = df["low_code"].astype(str)
    df["low_chapter"] = df["low_code"].str.extract(r"^\s*(\d{2})")[0]

    # Classify treatments into CE tiers
    df["onsite_tier"] = df["onsite_treatment"].apply(classify_treatment)
    df["final_tier"]  = df["final_treatment"].apply(classify_treatment)

    # Effective destination tier: prefer final, fall back to onsite
    df["effective_tier"] = df["final_tier"].where(
        df["final_tier"] != "unknown", df["onsite_tier"]
    )

    return df


def epa_cdw(df_epa: pd.DataFrame) -> pd.DataFrame:
    """Restrict EPA data to construction & demolition (LoW chapter 17)."""
    return df_epa[df_epa["low_chapter"] == "17"].copy()


# ---------------------------------------------------------------------------
# CSO loader
# ---------------------------------------------------------------------------
def load_cso_generation() -> pd.DataFrame:
    df = pd.read_csv(PROJECT_DIR / "waste.csv")
    df = df.rename(columns={
        "Year": "year",
        "NACE Rev. 2 Activity": "nace",
        "Hazardousness": "hazardousness",
        "Waste Category": "waste_category",
        "VALUE": "tonnes",
    })
    return df


def load_cso_treatment() -> pd.DataFrame:
    df = pd.read_csv(PROJECT_DIR / "waste_treated.csv")
    df = df.rename(columns={
        "Year": "year",
        "Hazardousness": "hazardousness",
        "Waste Category": "waste_category",
        "Waste Management Operation": "operation",
        "VALUE": "tonnes",
    })
    return df


def cso_construction_series(df_gen: pd.DataFrame) -> pd.DataFrame:
    """National construction (NACE F) waste totals, biennial 2004-2020."""
    mask = (
        df_gen["nace"].str.contains("Construction", case=False, na=False)
        & (df_gen["waste_category"] == "Total waste")
        & (df_gen["hazardousness"]
           == "Hazardous and non-hazardous - Total[HAZ_NHAZ]")
    )
    out = (
        df_gen.loc[mask, ["year", "tonnes"]]
        .dropna()
        .sort_values("year")
        .reset_index(drop=True)
    )
    return out


# ---------------------------------------------------------------------------
# KPI / indicator computation
# ---------------------------------------------------------------------------
def cdw_recovery_rate(df_cdw: pd.DataFrame) -> pd.DataFrame:
    """
    Recovery rate per year, EU WFD-style:
        recovery_t / (recovery_t + disposal_t)

    The EU 70% target for non-hazardous CDW is the policy benchmark.
    Here we use accepted_t as the volume basis (what licensees took in).
    Excludes 'recovery_prep_storage' (R12/R13) since it is a holding step,
    not a final outcome -- this is a sensitivity choice agents can revisit.
    """
    df = df_cdw.copy()
    df["accepted_t"] = pd.to_numeric(df["accepted_t"], errors="coerce").fillna(0)

    def bucket(t):
        if t in ("recovery_energy", "recovery_recycling_reuse"):
            return "recovery"
        if t.startswith("disposal"):
            return "disposal"
        return "other"

    df["bucket"] = df["effective_tier"].apply(bucket)
    pivot = (
        df.groupby(["year", "bucket"])["accepted_t"].sum().unstack(fill_value=0)
    )
    pivot["recovery_rate"] = pivot.get("recovery", 0) / (
        pivot.get("recovery", 0) + pivot.get("disposal", 0)
    ).replace(0, pd.NA)
    pivot["meets_eu_70_target"] = pivot["recovery_rate"] >= 0.70
    return pivot.reset_index()


def cdw_by_material(df_cdw: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Top CDW material streams by tonnage, summed across all years."""
    df = df_cdw.copy()
    df["accepted_t"] = pd.to_numeric(df["accepted_t"], errors="coerce").fillna(0)
    return (
        df.groupby("low_desc")["accepted_t"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )


def licensee_compliance_scorecard(df_cdw: pd.DataFrame) -> pd.DataFrame:
    """
    Per-licensee, per-year scorecard the ComplianceMonitor agent will read.
    Each row is an audit-ready unit.
    """
    df = df_cdw.copy()
    df["accepted_t"] = pd.to_numeric(df["accepted_t"], errors="coerce").fillna(0)
    df["bucket"] = df["effective_tier"].apply(
        lambda t: "recovery" if t.startswith("recovery_") and t != "recovery_prep_storage"
        else "disposal" if t.startswith("disposal_")
        else "other"
    )
    g = df.groupby(["licensee", "year"])
    out = pd.DataFrame({
        "total_accepted_t": g["accepted_t"].sum(),
        "n_records":        g.size(),
        "n_low_codes":      g["low_code"].nunique(),
    }).reset_index()

    pivot = (
        df.groupby(["licensee", "year", "bucket"])["accepted_t"]
          .sum().unstack(fill_value=0).reset_index()
    )
    out = out.merge(pivot, on=["licensee", "year"], how="left")
    out["recovery_rate"] = out.get("recovery", 0) / (
        out.get("recovery", 0) + out.get("disposal", 0)
    ).replace(0, pd.NA)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading EPA data...")
    epa = load_epa()
    cdw = epa_cdw(epa)
    print(f"  EPA total rows: {len(epa):,} | CDW rows: {len(cdw):,}")

    print("\nLoading CSO data...")
    gen = load_cso_generation()
    trt = load_cso_treatment()
    constr_series = cso_construction_series(gen)
    print(f"  CSO Construction (F) series:")
    print(constr_series.to_string(index=False))

    print("\n--- KPI: Annual CDW recovery rate (EPA licensees) ---")
    rr = cdw_recovery_rate(cdw)
    print(rr.round(3).to_string(index=False))

    print("\n--- Top CDW materials by tonnage (2021-2025) ---")
    print(cdw_by_material(cdw).to_string(index=False))

    print("\n--- Licensee scorecard sample (5 rows) ---")
    sc = licensee_compliance_scorecard(cdw)
    print(sc.head().round(3).to_string(index=False))

    # Save outputs for the agents to consume
    out_dir = Path("./out"); out_dir.mkdir(exist_ok=True)
    cdw.to_parquet(out_dir / "cdw_clean.parquet", index=False)
    rr.to_csv(out_dir / "kpi_recovery_rate.csv", index=False)
    sc.to_csv(out_dir / "licensee_scorecard.csv", index=False)
    constr_series.to_csv(out_dir / "cso_construction_series.csv", index=False)
    print(f"\nSaved outputs to {out_dir.resolve()}")
