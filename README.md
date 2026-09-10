# CDW Agentic AI - Governed Multi-Agent System for Construction & Demolition Waste Compliance

An MSc practicum project (Dublin City University) implementing a governed multi-agent
AI system for monitoring construction and demolition (C&D) waste compliance in Ireland,
combining rule-based compliance checking, unsupervised and supervised machine learning,
time-series forecasting, and a full provenance audit trail.

**Authors:** Tharakesh Aravindan Suresh Thanuja, Suman Neupane
**Supervisor:** Dr Muhammad Salman Pathan

---

## Overview

The system ingests EPA-licensed facility waste returns, computes recovery-rate compliance
against the EU Waste Framework Directive 70% target, discovers behavioural archetypes via
clustering, flags anomalies, forecasts future recovery, and routes every decision through
a human-in-the-loop governance layer. Each agent action is logged with full provenance:
code version, model specification, cited legal basis, and SHA-1 data fingerprints.

---

## Repository Structure

```
.
├── data/                       # All input datasets (EPA, CSO, Eurostat, policy CSVs)
│   ├── EPA-Licensee-Waste-Data-2021..2025.xlsx
│   ├── waste.csv                       # CSO waste generation (NACE)
│   ├── waste_treated.csv               # CSO waste treatment
│   ├── C&D-Data-Archive-Summary-CSV.csv
│   ├── cso_construction_indicators.csv
│   ├── policy_levy_timeline.csv
│   ├── policy_eow_byproduct_decisions.csv
│   └── eurostat_cd_waste_*.csv
├── notebooks/
│   └── ML_Forecasting.ipynb    # 7-model forecasting study (Suman)
├── out/                        # Generated outputs (created on run; git-ignored)
├── cdw_pipeline.py             # Data cleaning + recovery-rate KPIs
├── facility_analysis.py        # KMeans clustering + Isolation Forest (unsupervised)
├── facility_classifier.py      # Supervised needs_review classifier (GroupKFold)
├── sparsity_robustness.py      # Multi-seed forecasting robustness experiment
├── agents.py                   # Governed multi-agent system + audit trail
├── figures.py                  # Regenerates paper figures from out/ CSVs
├── requirements.txt
└── README.md
```

---

## Setup

Requires **Python 3.10+**.

```bash
pip install -r requirements.txt
```

---

## How to Run (order matters - later scripts read earlier outputs)

```bash
# 1. Clean data and compute compliance KPIs
python cdw_pipeline.py

# 2. Clustering + anomaly detection (needs step 1)
python facility_analysis.py

# 3. Supervised classifier (needs step 2)
python facility_classifier.py

# 4. Forecasting study - run the notebook top to bottom
#    (produces out/annual_cdw_series.csv, needed by step 5)
jupyter notebook notebooks/ML_Forecasting.ipynb

# 5. Multi-seed sparsity robustness (needs step 4)
python sparsity_robustness.py

# 6. Regenerate all figures (needs steps 1-5)
python figures.py

# 7. Run the agents (needs steps 1-2)
python agents.py compliance --licensee "Integrated Materials Solutions Limited Partnership - W0129" --year 2023
python agents.py forecast --horizon 2
```

All outputs are written to `out/`. The agent audit trail is appended to `out/audit_log.jsonl`.

---

## Cross-Platform Note (Windows / macOS)

All scripts use Python's `pathlib`, so forward/back slashes are handled automatically -
the code runs identically on Windows and macOS.

**If the notebook throws a `FileNotFoundError` on `waste.csv` (or any data file):**
this is a *working-directory* issue, not an OS issue. The notebook uses a path relative
to the `notebooks/` folder (`../data/`). It must be launched with `notebooks/` as the
working directory. Either:

- Start Jupyter from inside the `notebooks/` folder, **or**
- Replace the path cell with the repo-root-anchored version below, which works regardless
  of where the notebook is launched from:

```python
from pathlib import Path
# Anchor to the repo root so the path works on any machine, any launch directory
DATA = (Path.cwd() / "data") if (Path.cwd() / "data").exists() else (Path.cwd().parent / "data")
```

Also ensure `data/waste.csv` is actually present after cloning (`git pull` / verify the
file synced) - a missing file produces the same error.

---

## Data Sources

- EPA Licensee Waste Data 2021–2025 - https://www.epa.ie/
- EPA C&D Data Archive 2004–2023 - https://www.epa.ie/
- CSO Waste & Construction Statistics - https://data.cso.ie/
- Eurostat waste tables - https://ec.europa.eu/eurostat/

---

## Generative AI Disclosure

In accordance with DCU School of Computing policy, the authors disclose the use of
Anthropic's Claude to assist with drafting/refining code, suggesting methodological
design choices for the clustering and anomaly detection, and drafting initial prose which
the authors reviewed, edited, and own. All data, experiments, and findings were verified
by the authors.
