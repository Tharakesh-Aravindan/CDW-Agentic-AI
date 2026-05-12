"""
CDW Agentic AI -- Starter Prototype
-----------------------------------
A small multi-agent skeleton aligned with the lit review's Opportunity 1:
"Develop and empirically evaluate a governed agentic AI architecture
tailored to construction waste policy tasks."

Design principles (from Aboh & Chuka [16] and Hosseini & Seilani [15]):
  - Bounded autonomy: each agent has an explicit scope and tools.
  - Audit trail: every recommendation logs its inputs and reasoning.
  - Human-in-the-loop: the Orchestrator requires approval before acting on
    any recommendation that would change a real policy or report.

Agents
  ComplianceMonitor  -- compares licensee performance to EU/IE targets
  Forecaster         -- (stub) predicts next-period CDW flows
  ScenarioEvaluator  -- (stub) runs what-if policy levers
  PolicyRecommender  -- (stub) synthesises briefs with citations to data

Run:
    export ANTHROPIC_API_KEY=...
    python agents.py --licensee "Ashgrove Recycling - W0147" --year 2024
"""

from __future__ import annotations
import argparse, json, os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd


OUT_DIR = Path("./out")
AUDIT_LOG = OUT_DIR / "audit_log.jsonl"


# ---------------------------------------------------------------------------
# Audit trail -- governance-by-design [16]
# ---------------------------------------------------------------------------
def log_event(event: dict[str, Any]) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    event = {"ts": datetime.now(timezone.utc).isoformat(), **event}
    with AUDIT_LOG.open("a") as f:
        f.write(json.dumps(event, default=str) + "\n")


# ---------------------------------------------------------------------------
# Shared agent base
# ---------------------------------------------------------------------------
@dataclass
class AgentResult:
    agent: str
    status: str                 # "ok" | "needs_review" | "error"
    summary: str
    evidence: list[dict] = field(default_factory=list)
    recommendation: str | None = None
    confidence: float | None = None


class Agent:
    name: str = "agent"
    scope: str = "Define what this agent is and is not allowed to do."

    def run(self, **kwargs) -> AgentResult:
        raise NotImplementedError

    def __call__(self, **kwargs) -> AgentResult:
        log_event({"agent": self.name, "phase": "start", "inputs": kwargs})
        result = self.run(**kwargs)
        log_event({"agent": self.name, "phase": "end",
                   "result": asdict(result)})
        return result


# ---------------------------------------------------------------------------
# ComplianceMonitor -- working agent
# ---------------------------------------------------------------------------
class ComplianceMonitor(Agent):
    name = "ComplianceMonitor"
    scope = (
        "Read-only assessment of whether a CDW licensee meets the EU "
        "WFD 70% non-hazardous CDW recovery target. Cannot recommend "
        "enforcement; that is the PolicyRecommender's role with human approval."
    )

    EU_TARGET = 0.70

    def __init__(self, scorecard_path: Path = OUT_DIR / "licensee_scorecard.csv"):
        self.df = pd.read_csv(scorecard_path)

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
        if pd.isna(rate):
            return AgentResult(
                self.name, "needs_review",
                summary=("All accepted CDW classified as 'other' (R12/R13 "
                         "prep/storage). Final outcome cannot be inferred "
                         "from on-site treatment alone."),
                evidence=evidence,
                recommendation=("Request final-destination treatment data "
                                "from licensee for the year."),
                confidence=0.4,
            )
        meets = rate >= self.EU_TARGET
        return AgentResult(
            self.name,
            "ok" if meets else "needs_review",
            summary=(f"{licensee} achieved {rate:.1%} recovery in {year}; "
                     f"{'meets' if meets else 'BELOW'} the EU 70% target."),
            evidence=evidence,
            recommendation=None if meets else (
                "Flag for follow-up: investigate disposal share, "
                "request remediation plan, consider in next inspection cycle."
            ),
            confidence=0.85,
        )


# ---------------------------------------------------------------------------
# Stubs -- to be implemented in subsequent phases
# ---------------------------------------------------------------------------
class Forecaster(Agent):
    name = "Forecaster"
    scope = "Forecast next-period CDW tonnage by material/region. Read-only."

    def run(self, **kwargs) -> AgentResult:
        # TODO Phase 2: ARIMA / Prophet / gradient-boosting on time-keyed slices
        return AgentResult(self.name, "ok",
                           "Stub. Implement with cdw_clean.parquet groupby year.")


class ScenarioEvaluator(Agent):
    name = "ScenarioEvaluator"
    scope = ("Evaluate hypothetical policy levers (e.g. landfill levy +X%, "
             "mandatory pre-demolition audit). Returns delta vs baseline.")

    def run(self, **kwargs) -> AgentResult:
        # TODO Phase 3: system-dynamics model inspired by Ding et al. [6]
        return AgentResult(self.name, "ok", "Stub.")


class PolicyRecommender(Agent):
    """
    Uses Claude to synthesise the upstream agents' findings into a draft
    policy brief. Outputs are NEVER auto-published -- they require human
    approval, in line with governance-by-design [16].
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


# ---------------------------------------------------------------------------
# Orchestrator -- enforces the human-in-the-loop gate
# ---------------------------------------------------------------------------
class Orchestrator:
    def __init__(self):
        self.compliance = ComplianceMonitor()
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--licensee", required=True)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--brief", action="store_true",
                   help="Also draft a policy brief (uses Anthropic API)")
    args = p.parse_args()

    orch = Orchestrator()
    result = orch.assess_licensee(args.licensee, args.year,
                                   draft_brief=args.brief)
    print(json.dumps(result, indent=2, default=str))
    print(f"\nAudit log: {AUDIT_LOG.resolve()}")
