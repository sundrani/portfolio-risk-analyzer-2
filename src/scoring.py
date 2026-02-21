from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

from .metrics import RiskMetrics


@dataclass(frozen=True)
class RiskAssessment:
    risk_score: int  # 0-100
    bucket: str      # Conservative / Moderate / Aggressive
    explanation: Dict[str, Any]


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def score_from_metrics(m: RiskMetrics) -> RiskAssessment:
    # Heuristic scoring: higher vol/drawdown/var/beta => higher risk. Higher Sharpe => lower risk.
    # Scales are calibrated to be reasonable for equity-heavy vs bond-heavy portfolios.
    vol_s = _clip01(m.annual_volatility / 0.35)            # 35% vol ~= very high
    dd_s  = _clip01(abs(m.max_drawdown) / 0.50)            # 50% drawdown ~= very high
    var_s = _clip01(abs(m.var_95) / 0.04)                  # 4% daily VaR ~= high
    cvar_s= _clip01(abs(m.cvar_95) / 0.06)                 # 6% daily CVaR ~= high
    beta_s= _clip01(abs(m.beta) / 2.0)                     # beta 2 ~= very high
    sharpe_s = _clip01((1.5 - m.sharpe_ratio) / 2.0)       # Sharpe high => less risk

    # Weighted sum (must total 1.0)
    risk = (
        0.30 * vol_s +
        0.25 * dd_s +
        0.15 * var_s +
        0.10 * cvar_s +
        0.10 * beta_s +
        0.10 * sharpe_s
    )

    risk_score = int(round(_clip01(risk) * 100))

    if risk_score < 35:
        bucket = "Conservative"
    elif risk_score < 70:
        bucket = "Moderate"
    else:
        bucket = "Aggressive"

    explanation = {
        "normalized_components": {
            "volatility": vol_s,
            "max_drawdown": dd_s,
            "var_95": var_s,
            "cvar_95": cvar_s,
            "beta": beta_s,
            "sharpe_inverse": sharpe_s,
        },
        "weights": {
            "volatility": 0.30,
            "max_drawdown": 0.25,
            "var_95": 0.15,
            "cvar_95": 0.10,
            "beta": 0.10,
            "sharpe_inverse": 0.10,
        },
        "notes": [
            "Risk score is heuristic and meant for demo/education.",
            "Use a longer history window and appropriate benchmark for production.",
        ],
    }

    return RiskAssessment(risk_score=risk_score, bucket=bucket, explanation=explanation)
