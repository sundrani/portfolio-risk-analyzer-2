from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskMetrics:
    annual_volatility: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    beta: float


def _validate_prices(prices: pd.DataFrame) -> None:
    required = {"date", "close"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"Price data missing columns: {sorted(missing)}")


def daily_returns_from_prices(prices: pd.Series) -> pd.Series:
    return prices.pct_change().dropna()


def max_drawdown_from_returns(daily_returns: pd.Series) -> float:
    # equity curve
    curve = (1.0 + daily_returns).cumprod()
    peak = curve.cummax()
    dd = (curve / peak) - 1.0
    return float(dd.min())  # negative number


def var_cvar(daily_returns: pd.Series, alpha: float = 0.95) -> Tuple[float, float]:
    # Historical VaR/CVaR (losses => negative returns)
    q = np.quantile(daily_returns, 1 - alpha)
    tail = daily_returns[daily_returns <= q]
    cvar = tail.mean() if len(tail) else q
    return float(q), float(cvar)


def annualize_return(daily_returns: pd.Series, trading_days: int = 252) -> float:
    # geometric approximation
    compounded = (1.0 + daily_returns).prod()
    years = len(daily_returns) / trading_days
    if years <= 0:
        return 0.0
    return float(compounded ** (1.0 / years) - 1.0)


def annualize_vol(daily_returns: pd.Series, trading_days: int = 252) -> float:
    return float(daily_returns.std(ddof=1) * np.sqrt(trading_days))


def sharpe_ratio(daily_returns: pd.Series, rf_annual: float = 0.0, trading_days: int = 252) -> float:
    rf_daily = (1.0 + rf_annual) ** (1.0 / trading_days) - 1.0
    excess = daily_returns - rf_daily
    vol = excess.std(ddof=1)
    if vol == 0 or np.isnan(vol):
        return 0.0
    return float(excess.mean() / vol * np.sqrt(trading_days))


def beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1, join="inner").dropna()
    if aligned.shape[0] < 5:
        return 0.0
    pr = aligned.iloc[:, 0].values
    br = aligned.iloc[:, 1].values
    var_b = np.var(br, ddof=1)
    if var_b == 0:
        return 0.0
    cov = np.cov(pr, br, ddof=1)[0, 1]
    return float(cov / var_b)


def compute_portfolio_returns(price_map: Dict[str, pd.DataFrame], weights: Dict[str, float]) -> pd.Series:
    # Build aligned dataframe of close prices
    closes = []
    for t, df in price_map.items():
        _validate_prices(df)
        series = df.copy()
        series["date"] = pd.to_datetime(series["date"])
        series = series.sort_values("date").set_index("date")["close"].rename(t)
        closes.append(series)
    px = pd.concat(closes, axis=1, join="inner").dropna()
    if px.empty:
        raise ValueError("No overlapping dates across tickers.")
    rets = px.pct_change().dropna()
    w = pd.Series(weights).reindex(rets.columns).fillna(0.0)
    # normalize weights defensively
    s = w.sum()
    if s == 0:
        raise ValueError("Sum of weights is 0.")
    w = w / s
    portfolio = (rets * w).sum(axis=1)
    return portfolio


def compute_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    rf_annual: float = 0.0
) -> RiskMetrics:
    ar = annualize_return(portfolio_returns)
    av = annualize_vol(portfolio_returns)
    sr = sharpe_ratio(portfolio_returns, rf_annual=rf_annual)
    mdd = max_drawdown_from_returns(portfolio_returns)
    v, cv = var_cvar(portfolio_returns, alpha=0.95)
    b = beta(portfolio_returns, benchmark_returns) if benchmark_returns is not None else 0.0
    return RiskMetrics(
        annual_volatility=av,
        annual_return=ar,
        sharpe_ratio=sr,
        max_drawdown=mdd,
        var_95=v,
        cvar_95=cv,
        beta=b,
    )
