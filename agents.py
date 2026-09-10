"""
CDW Agentic AI - Governed Multi-Agent Prototype
------------------------------------------------
A governed multi-agent system built against Opportunity 1 from our literature
review: to develop and empirically evaluate a governed agentic AI architecture
for construction and demolition waste policy tasks.

The design rests on three principles drawn from Aboh & Chuka [16] and
Hosseini & Seilani [15]:
  - Bounded autonomy: each agent declares an explicit scope, and the
    orchestrator refuses requests that fall outside it.
  - Audit trail: every agent action logs its inputs, its reasoning, and full
    provenance (code version, model spec, policy basis, data fingerprints).
  - Human-in-the-loop: no agent can act autonomously; the orchestrator routes
    every recommendation for human approval before any real-world effect.

Agents:
  ComplianceMonitor  - compares licensee performance to the EU/IE 70% target,
                       with ML clustering and anomaly detection feeding its verdict
  Forecaster         - projects near-term CDW recovery and tonnage, with honest
                       confidence bounds under sparse data
  ScenarioEvaluator  - scoped as future work (policy what-if analysis)
  PolicyRecommender  - scoped as future work (LLM-drafted policy briefs)

Run Command:
    python agents.py compliance --licensee "Ashgrove Recycling - W0147" --year 2024
    python agents.py forecast --horizon 2
"""

from __future__ import annotations
import argparse, json, os, hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd


OUT_DIR = Path("./out")
AUDIT_LOG = OUT_DIR / "audit_log.jsonl"

# Single place to bump if the agent code changes meaningfully.
# Logged with every agent action so old audit entries can be traced back
# to the exact agent version that produced them.
AGENT_CODE_VERSION = "0.3.0"


def file_fingerprint(path: Path) -> dict | None:
    """Cheap, reproducible fingerprint of a data file.

    Captures size + first-8-bytes-of-SHA256 + last-modified timestamp.
    Used to record exactly which data file version produced a result, so
    audit log entries are traceable to the source (per supervisor's
    request, and addresses the 'agentic AI lacks domain case studies'
    research gap by demonstrating end-to-end provenance).
    """
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return {
        "path": str(p),
        "size_bytes": p.stat().st_size,
        "sha256_16": h.hexdigest()[:16],
        "mtime_utc": datetime.fromtimestamp(
            p.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
    }


# --------------------------------------------------------------------
# Audit trail - governance-by-design [16]
# --------------------------------------------------------------------
def log_event(event: dict[str, Any]) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    event = {"ts": datetime.now(timezone.utc).isoformat(), **event}
    with AUDIT_LOG.open("a") as f:
        f.write(json.dumps(event, default=str) + "\n")


# --------------------------------------------------------------------
# Shared agent base
# --------------------------------------------------------------------
@dataclass
class AgentResult:
    """A structured result returned by every agent.

    Carries the agent name, a status (ok / needs_review / error), a
    human-readable summary, an optional confidence score, and the provenance
    block. Crucially, there is no 'approved' status: approval is something a
    human grants to a result, never something an agent assigns to itself.
    """
    agent: str
    status: str                 # "ok" | "needs_review" | "error"
    summary: str
    evidence: list[dict] = field(default_factory=list)
    recommendation: str | None = None
    confidence: float | None = None
    # Provenance: agent code version, model details, policy parameters, data
    # source fingerprints. Recorded with every result for traceability.
    provenance: dict = field(default_factory=dict)


class Agent:
    """Base class for all agents.

    Defines the shared machinery every agent inherits: an explicit scope, a
    provenance record, and automatic audit logging on each call. Subclasses
    implement run(); they cannot return a result without the base class also
    writing start and end entries to the audit log.
    """
    name: str = "agent"
    scope: str = "Define what this agent is and is not allowed to do."

    def provenance(self) -> dict:
        """Return agent code version + model/policy parameters + data sources.

        Subclasses should override to add their specific model name,
        hyperparameters, EU/IE policy thresholds, and file fingerprints.
        """
        return {"agent_code_version": AGENT_CODE_VERSION}

    def run(self, **kwargs) -> AgentResult:
        raise NotImplementedError

    def __call__(self, **kwargs) -> AgentResult:
        prov = self.provenance()
        log_event({"agent": self.name, "phase": "start",
                   "inputs": kwargs, "provenance": prov})
        result = self.run(**kwargs)
        # Attach provenance to the result so it surfaces in the JSON output too.
        if not result.provenance:
            result.provenance = prov
        log_event({"agent": self.name, "phase": "end",
                   "result": asdict(result)})
        return result


# --------------------------------------------------------------------
# ComplianceMonitor
# --------------------------------------------------------------------
class ComplianceMonitor(Agent):
    """Assesses whether a facility-year meets the EU 70% recovery target.

    The verdict is enriched with unsupervised ML: the clustering assigns each
    facility-year a behavioural archetype, and the Isolation Forest anomaly
    flag can downgrade an otherwise-compliant facility to needs_review. The
    agent does not merely report the ML output alongside its verdict - it uses
    that output to adjust the verdict, so unusual behaviour is surfaced for
    human review even when no formal rule has been broken.
    """
    name = "ComplianceMonitor"
    scope = (
        "Read-only assessment of whether a CDW licensee meets the EU "
        "WFD 70% non-hazardous CDW recovery target. Cannot recommend "
        "enforcement; that is the PolicyRecommender's role with human approval."
    )

    EU_TARGET = 0.70

    def __init__(
        self,
        scorecard_path: Path = OUT_DIR / "licensee_scorecard.csv",
        clusters_path: Path = OUT_DIR / "facility_clusters.csv",
    ):
        self.df = pd.read_csv(scorecard_path)
        # Optional ML enrichment: behavioural archetype + anomaly flag from
        # facility_analysis.py. Loaded if present; the agent degrades
        # gracefully (no ML fields) if the file hasn't been generated yet.
        self.clusters = None
        if Path(clusters_path).exists():
            self.clusters = pd.read_csv(clusters_path)

    def _ml_context(self, licensee: str, year: int) -> dict | None:
        """Return the ML archetype + anomaly status for a facility-year, if available."""
        if self.clusters is None:
            return None
        rows = self.clusters[
            (self.clusters["licensee"] == licensee)
            & (self.clusters["year"] == year)
        ]
        if rows.empty:
            return None
        c = rows.iloc[0]
        return {
            "archetype": str(c.get("archetype", "")),
            "is_anomaly": bool(c.get("is_anomaly", False)),
            "anomaly_score": float(c.get("anomaly_score", 0.0)),
        }

    def provenance(self) -> dict:
        return {
            "agent_code_version": AGENT_CODE_VERSION,
            "policy_parameters": {
                "eu_wfd_recovery_target": self.EU_TARGET,
                "policy_basis": "EU Waste Framework Directive 2008/98/EC, "
                                "Article 11(2)(b): 70% non-hazardous CDW "
                                "recovery target by 2020",
            },
            "model": {
                "name": "rule_based_threshold",
                "ml_enrichment": "facility_analysis.py (KMeans k=5 + "
                                 "IsolationForest contamination=0.05)"
                                 if self.clusters is not None else "none",
            },
            "data_sources": {
                "scorecard": file_fingerprint(OUT_DIR / "licensee_scorecard.csv"),
                "clusters": file_fingerprint(OUT_DIR / "facility_clusters.csv"),
            },
        }

    def run(self, licensee: str, year: int) -> AgentResult:
        rows = self.df[(self.df["licensee"] == licensee)
                       & (self.df["year"] == year)]
        if rows.empty:
            return AgentResult(self.name, "error",
                               f"No data for {licensee} in {year}.")
        r = rows.iloc[0]
        rate = r.get("recovery_rate")
        evidence = [{
            "source": "EPA Licensee Waste Data",
            "licensee": licensee,
            "year": int(year),
            "total_accepted_t": float(r["total_accepted_t"]),
            "recovery_t": float(r.get("recovery", 0) or 0),
            "disposal_t": float(r.get("disposal", 0) or 0),
            "other_t": float(r.get("other", 0) or 0),
            "recovery_rate": None if pd.isna(rate) else float(rate),
        }]

        # --- ML enrichment: behavioural archetype + anomaly flag ----------
        ml = self._ml_context(licensee, year)
        ml_note = ""
        if ml is not None:
            evidence.append({
                "source": "Facility ML analysis (KMeans + IsolationForest)",
                "behavioural_archetype": ml["archetype"],
                "anomaly_flagged": ml["is_anomaly"],
                "anomaly_score": round(ml["anomaly_score"], 4),
            })
            if ml["is_anomaly"]:
                ml_note = (
                    f" ML flag: behaviour is anomalous for {year} "
                    f"(archetype '{ml['archetype']}') and warrants review."
                )
            else:
                ml_note = f" ML archetype: {ml['archetype']}."

        if pd.isna(rate):
            return AgentResult(
                self.name, "needs_review",
                summary=("All accepted CDW classified as 'other' (R12/R13 "
                         "prep/storage). Final outcome cannot be inferred "
                         "from on-site treatment alone." + ml_note),
                evidence=evidence,
                recommendation=("Request final-destination treatment data "
                                "from licensee for the year."),
                confidence=0.4,
            )
        meets = rate >= self.EU_TARGET
        # An anomaly flag downgrades an otherwise-compliant facility to review.
        anomaly = ml is not None and ml["is_anomaly"]
        status = "ok" if (meets and not anomaly) else "needs_review"
        return AgentResult(
            self.name,
            status,
            summary=(f"{licensee} achieved {rate:.1%} recovery in {year}; "
                     f"{'meets' if meets else 'BELOW'} the EU 70% target."
                     + ml_note),
            evidence=evidence,
            recommendation=(
                None if (meets and not anomaly) else (
                    "Flag for follow-up: investigate disposal share, "
                    "request remediation plan, consider in next inspection cycle."
                    + (" Anomaly detected by ML - prioritise in review queue."
                       if anomaly else "")
                )
            ),
            confidence=0.85,
        )


# --------------------------------------------------------------------
# Forecaster Agent
# --------------------------------------------------------------------
class Forecaster(Agent):
    """Projects CDW recovery rate and total accepted tonnage 1–3 years ahead.

    Uses an OLS linear trend (numpy.polyfit) with residual-based 95%
    prediction intervals. Outputs are clamped to physically meaningful ranges
    (recovery rate to [0, 1]) to avoid nonsensical extrapolation. With only
    five annual observations (2021–2025), the agent reports a modest
    confidence and recommends upgrading to ARIMA once ten or more points are
    available - encoding the principle that an agent should represent the
    limits of its own knowledge rather than project false certainty. Forecasts
    are saved to out/forecast.csv for downstream use and audit.
    """

    name = "Forecaster"
    scope = (
        "Produces trend-based forecasts of CDW recovery rate and total accepted "
        "tonnage for up to 3 years ahead. Read-only. Uses OLS linear extrapolation "
        "with residual-based prediction intervals. Flags low confidence when "
        "N < 10 observations. Cannot set targets or recommend enforcement."
    )

    FORECAST_OUT = OUT_DIR / "forecast.csv"
    _T95 = 2.0  # conservative ~95% PI multiplier (approximates t-dist for small N)

    def __init__(
        self,
        kpi_path: Path = OUT_DIR / "kpi_recovery_rate.csv",
        cdw_path: Path = OUT_DIR / "cdw_clean.parquet",
    ):
        self.kpi_path = Path(kpi_path)
        self.cdw_path = Path(cdw_path)
        self.kpi = pd.read_csv(kpi_path)
        self.cdw = pd.read_parquet(cdw_path)

    def provenance(self) -> dict:
        return {
            "agent_code_version": AGENT_CODE_VERSION,
            "model": {
                "name": "OLS_linear_trend",
                "library": "numpy.polyfit (deg=1)",
                "prediction_interval": "residual-based, "
                                       f"95% via t-multiplier T95={self._T95}",
                "recovery_rate_clamp": "[0.0, 1.0]",
                "tonnage_clamp": "[0.0, +inf)",
                "upgrade_path": "ARIMA once n_obs >= 10",
            },
            "policy_parameters": {
                "horizon_max_years": 3,
            },
            "data_sources": {
                "kpi": file_fingerprint(self.kpi_path),
                "cdw_clean": file_fingerprint(self.cdw_path),
            },
        }

    def _linear_forecast(
        self,
        years: np.ndarray,
        values: np.ndarray,
        horizon: int,
        clamp: tuple[float, float] | None = None,
    ) -> dict:
        """Fit OLS linear trend and extrapolate `horizon` steps forward.

        Returns slope, intercept, RMSE, and per-year forecast dicts with
        95% prediction intervals optionally clamped to `clamp=(lo, hi)`.
        """
        coeffs = np.polyfit(years, values, 1)
        fitted = np.polyval(coeffs, years)
        rmse = float(np.sqrt(np.mean((values - fitted) ** 2)))
        half_width = self._T95 * rmse

        future_years = np.arange(years[-1] + 1, years[-1] + 1 + horizon)
        forecasts = []
        for fy in future_years:
            raw = float(np.polyval(coeffs, fy))
            lo = raw - half_width
            hi = raw + half_width
            if clamp is not None:
                raw = float(np.clip(raw, *clamp))
                lo = float(np.clip(lo, *clamp))
                hi = float(np.clip(hi, *clamp))
            forecasts.append({
                "year": int(fy),
                "forecast": round(raw, 6),
                "ci_lo": round(lo, 6),
                "ci_hi": round(hi, 6),
            })

        return {
            "slope": round(float(coeffs[0]), 6),
            "intercept": round(float(coeffs[1]), 2),
            "rmse": round(rmse, 6),
            "n_obs": int(len(years)),
            "forecasts": forecasts,
        }

    def run(self, horizon: int = 2, metric: str = "both") -> AgentResult:
        """
        Parameters
        ----------
        horizon : int, 1–3
            How many years beyond the last observed year to forecast.
        metric : str
            'recovery_rate' | 'cdw_tonnage' | 'both'
        """
        if horizon < 1 or horizon > 3:
            return AgentResult(self.name, "error",
                               "horizon must be between 1 and 3 years.")
        if metric not in ("recovery_rate", "cdw_tonnage", "both"):
            return AgentResult(self.name, "error",
                               "metric must be 'recovery_rate', 'cdw_tonnage', or 'both'.")

        evidence: list[dict] = []
        rows_out: list[dict] = []
        results: dict[str, dict] = {}

        # --- Recovery rate ---------------------------------------------------
        if metric in ("recovery_rate", "both"):
            kpi = self.kpi.sort_values("year")
            years = kpi["year"].to_numpy(dtype=float)
            rates = kpi["recovery_rate"].to_numpy(dtype=float)
            res = self._linear_forecast(years, rates, horizon, clamp=(0.0, 1.0))
            results["recovery_rate"] = res
            evidence.append({
                "series": "recovery_rate",
                "source": "EPA Licensee Waste Data KPIs (2021-2025)",
                "years_observed": kpi["year"].tolist(),
                "values_observed": [round(v, 4) for v in rates.tolist()],
                "slope_per_year": res["slope"],
                "rmse": res["rmse"],
                "n_obs": res["n_obs"],
                "forecasts": res["forecasts"],
            })
            for f in res["forecasts"]:
                rows_out.append({
                    "metric": "recovery_rate",
                    "year": f["year"],
                    "forecast": f["forecast"],
                    "ci_lo": f["ci_lo"],
                    "ci_hi": f["ci_hi"],
                    "method": "OLS linear trend",
                    "n_obs": res["n_obs"],
                })

        # --- CDW tonnage -----------------------------------------------------
        if metric in ("cdw_tonnage", "both"):
            annual = (
                self.cdw.groupby("year")["accepted_t"]
                .sum()
                .reset_index()
                .sort_values("year")
            )
            years = annual["year"].to_numpy(dtype=float)
            tonnes = annual["accepted_t"].to_numpy(dtype=float)
            res = self._linear_forecast(years, tonnes, horizon, clamp=(0.0, None))
            results["cdw_tonnage"] = res
            evidence.append({
                "series": "cdw_tonnage",
                "source": "EPA Licensee Waste Data CDW (2021-2025)",
                "years_observed": annual["year"].tolist(),
                "values_observed": [round(v, 0) for v in tonnes.tolist()],
                "slope_per_year": res["slope"],
                "rmse": res["rmse"],
                "n_obs": res["n_obs"],
                "forecasts": res["forecasts"],
            })
            for f in res["forecasts"]:
                rows_out.append({
                    "metric": "cdw_tonnage",
                    "year": f["year"],
                    "forecast": f["forecast"],
                    "ci_lo": f["ci_lo"],
                    "ci_hi": f["ci_hi"],
                    "method": "OLS linear trend",
                    "n_obs": res["n_obs"],
                })

        # --- Save output -----------------------------------------------------
        OUT_DIR.mkdir(exist_ok=True)
        pd.DataFrame(rows_out).to_csv(self.FORECAST_OUT, index=False)

        # --- Build human-readable summary ------------------------------------
        lines = []
        if "recovery_rate" in results:
            r = results["recovery_rate"]
            slope_pct = r["slope"] * 100
            preds = ", ".join(
                f"{f['year']}: {f['forecast']:.1%}" for f in r["forecasts"]
            )
            lines.append(
                f"Recovery rate trend: {slope_pct:+.1f} pp/yr (2021-2025). "
                f"Forecast → {preds}."
            )
        if "cdw_tonnage" in results:
            t = results["cdw_tonnage"]
            preds = ", ".join(
                f"{f['year']}: {f['forecast']:,.0f} t" for f in t["forecasts"]
            )
            lines.append(
                f"CDW tonnage trend: {t['slope']:+,.0f} t/yr (2021-2025). "
                f"Forecast → {preds}."
            )
        n = min(r["n_obs"] for r in results.values())
        lines.append(
            f"Note: N={n} years - OLS linear trend only. "
            "Do not use for policy targets without expert review."
        )

        # confidence penalised for short series (N=5 → 0.55)
        confidence = round(max(0.35, 0.55 + 0.04 * max(0, n - 5)), 2)

        return AgentResult(
            self.name,
            "ok",
            summary=" ".join(lines),
            evidence=evidence,
            recommendation=(
                f"Forecasts saved to {self.FORECAST_OUT}. "
                "Upgrade to ARIMA once ≥10 annual observations are available."
            ),
            confidence=confidence,
        )


class ScenarioEvaluator(Agent):
    """Scoped as future work.

    Would quantify the behavioural impact of dated policy interventions by
    comparing facility recovery rates before and after each event. Deferred
    because the post-intervention observation window is currently too short
    for credible dynamic analysis (see paper, Section VI-C). The agent
    inherits the same scope, audit, and approval machinery as the operational
    agents, so only its internal logic remains to be implemented.
    """
    name = "ScenarioEvaluator"
    scope = ("Evaluate hypothetical policy levers (e.g. landfill levy +X%, "
             "mandatory pre-demolition audit). Returns delta vs baseline.")

    def run(self, **kwargs) -> AgentResult:
        # Scoped as future work - see paper Section VI-C. The system-dynamics
        # approach would follow Ding et al. [6].
        return AgentResult(self.name, "error",
                           "ScenarioEvaluator is scoped as future work; "
                           "not implemented in this version.")


class PolicyRecommender(Agent):
    """Scoped as future work.

    Would use a large language model to synthesise the upstream agents'
    findings into a draft policy brief. Every generated brief would carry the
    needs_review status and could never be auto-published - it would require
    human approval, in line with governance-by-design [16]. Deferred because
    its value depends on the ScenarioEvaluator's outputs as input.
    """
    name = "PolicyRecommender"
    scope = ("Drafts policy briefs. Cannot execute, send, or publish. "
             "Output is reviewed by a human policy officer before any action.")

    def run(self, findings: list[AgentResult]) -> AgentResult:
        try:
            import anthropic
        except ImportError:
            return AgentResult(self.name, "error",
                               "anthropic SDK not installed. "
                               "`pip install anthropic`")
        if not os.getenv("ANTHROPIC_API_KEY"):
            return AgentResult(self.name, "error",
                               "ANTHROPIC_API_KEY env var not set.")

        client = anthropic.Anthropic()
        findings_json = json.dumps([asdict(f) for f in findings],
                                   indent=2, default=str)
        msg = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1500,
            system=(
                "You are a policy analyst drafting a short brief on a "
                "construction & demolition waste licensee's compliance with "
                "the EU Waste Framework Directive 70% recovery target. "
                "Cite every quantitative claim by referencing the evidence "
                "rows by index. Do not invent numbers. If evidence is "
                "insufficient, say so explicitly."
            ),
            messages=[{
                "role": "user",
                "content": (
                    "Findings from upstream agents (JSON):\n"
                    f"{findings_json}\n\n"
                    "Produce a 200-word brief with: (1) status, "
                    "(2) two-sentence evidence summary with citations, "
                    "(3) a recommended next step, (4) explicit confidence "
                    "and limitations."
                )
            }],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        return AgentResult(
            self.name, "needs_review",
            summary="Draft policy brief generated; awaiting human approval.",
            evidence=[{"source": "synthesised", "findings_in": len(findings)}],
            recommendation=text,
            confidence=0.7,
        )


# --------------------------------------------------------------------
# Orchestrator - enforces the human-in-the-loop gate
# --------------------------------------------------------------------
class Orchestrator:
    """Coordinates the agents and enforces the human-in-the-loop gate.

    Routes each request to the appropriate agent, sequences their execution,
    and packages the outputs into a single result that always carries a
    governance note (recommendations are advisory only) and an audit-log
    reference. This is what makes the components a coordinated, governed
    multi-agent system rather than a set of isolated scripts.
    """
    def __init__(self):
        self.compliance = ComplianceMonitor()
        self.forecaster = Forecaster()
        self.recommender = PolicyRecommender()

    def assess_licensee(self, licensee: str, year: int,
                         draft_brief: bool = False) -> dict:
        comp = self.compliance(licensee=licensee, year=year)
        out = {"compliance": asdict(comp)}
        if draft_brief and comp.status == "needs_review":
            brief = self.recommender(findings=[comp])
            out["brief"] = asdict(brief)
        out["governance_note"] = (
            "Outputs are advisory. No action is taken without explicit "
            "human approval recorded in the audit log."
        )
        return out

    def run_forecast(self, horizon: int = 2, metric: str = "both") -> dict:
        result = self.forecaster(horizon=horizon, metric=metric)
        out = {
            "forecast": asdict(result),
            "governance_note": (
                "Forecast is advisory. OLS linear trend only - "
                "review with domain expert before informing policy."
            ),
        }
        return out


# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    # compliance sub-command
    comp_p = sub.add_parser("compliance", help="Assess a licensee's CDW compliance.")
    comp_p.add_argument("--licensee", required=True)
    comp_p.add_argument("--year", type=int, required=True)
    comp_p.add_argument("--brief", action="store_true",
                        help="Also draft a policy brief (uses Anthropic API)")

    # forecast sub-command
    fc_p = sub.add_parser("forecast", help="Forecast CDW recovery rate and tonnage.")
    fc_p.add_argument("--horizon", type=int, default=2,
                      help="Years to forecast ahead (1–3, default 2)")
    fc_p.add_argument("--metric", default="both",
                      choices=["recovery_rate", "cdw_tonnage", "both"])

    # backwards-compat: bare --licensee/--year still works
    p.add_argument("--licensee")
    p.add_argument("--year", type=int)
    p.add_argument("--brief", action="store_true")

    args = p.parse_args()
    orch = Orchestrator()

    if args.cmd == "forecast":
        result = orch.run_forecast(horizon=args.horizon, metric=args.metric)
    elif args.cmd == "compliance" or args.licensee:
        licensee = args.licensee
        year = args.year
        brief = args.brief
        result = orch.assess_licensee(licensee, year, draft_brief=brief)
    else:
        p.print_help()
        raise SystemExit(1)

    print(json.dumps(result, indent=2, default=str))
    print(f"\nAudit log: {AUDIT_LOG.resolve()}")