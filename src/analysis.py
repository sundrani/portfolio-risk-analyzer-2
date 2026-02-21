from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Any
import pandas as pd

from .io_utils import load_price_map
from .metrics import compute_portfolio_returns, daily_returns_from_prices, compute_metrics
from .scoring import score_from_metrics


def analyze_portfolio(portfolio: Dict[str, Any], prices_dir: str) -> Dict[str, Any]:
    holdings = portfolio.get("holdings", [])
    if not holdings:
        raise ValueError("Portfolio holdings are empty.")
    tickers = [h["ticker"] for h in holdings]
    weights = {h["ticker"]: float(h["weight"]) for h in holdings}

    benchmark = portfolio.get("benchmark")
    if benchmark and benchmark not in tickers:
        tickers_with_bench = tickers + [benchmark]
    else:
        tickers_with_bench = tickers

    price_map = load_price_map(prices_dir, tickers_with_bench)

    # compute portfolio returns (exclude benchmark weight)
    portfolio_returns = compute_portfolio_returns(
        {t: price_map[t] for t in tickers},
        weights
    )

    benchmark_returns = None
    if benchmark:
        bdf = price_map[benchmark].copy()
        bdf["date"] = pd.to_datetime(bdf["date"])
        bdf = bdf.sort_values("date").set_index("date")["close"]
        benchmark_returns = daily_returns_from_prices(bdf)

    m = compute_metrics(portfolio_returns, benchmark_returns=benchmark_returns)
    assessment = score_from_metrics(m)

    result = {
        "as_of": portfolio.get("as_of"),
        "benchmark": benchmark,
        "holdings": holdings,
        "metrics": asdict(m),
        "risk_assessment": {
            "risk_score": assessment.risk_score,
            "bucket": assessment.bucket,
            "explanation": assessment.explanation,
        },
    }
    return result
