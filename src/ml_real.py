from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import joblib
except Exception:  # pragma: no cover
    # scikit-learn isn't required for analysis; only for ML training.
    RandomForestClassifier = None
    LogisticRegression = None
    train_test_split = None
    classification_report = None
    confusion_matrix = None
    Pipeline = None
    StandardScaler = None
    joblib = None


@dataclass
class TrainResult:
    model_path: str
    report: str
    confusion: List[List[int]]
    classes: List[str]
    feature_names: List[str]


def _load_prices(prices_dir: str, tickers: Iterable[str]) -> Dict[str, pd.Series]:
    """Load close prices from CSVs (<TICKER>.csv) and return aligned close series."""
    out: Dict[str, pd.Series] = {}
    base = Path(prices_dir)
    for t in tickers:
        p = base / f"{t}.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing price file: {p}")
        df = pd.read_csv(p)
        # Accept common column names
        date_col = "date" if "date" in df.columns else ("Date" if "Date" in df.columns else None)
        close_col = "close" if "close" in df.columns else ("Close" if "Close" in df.columns else None)
        if date_col is None or close_col is None:
            raise ValueError(f"{p} must contain date/close columns (found: {list(df.columns)})")
        s = df[[date_col, close_col]].copy()
        s[date_col] = pd.to_datetime(s[date_col])
        s = s.sort_values(date_col).set_index(date_col)[close_col].astype(float)
        out[t] = s
    return out


def _align_returns(price_map: Dict[str, pd.Series]) -> pd.DataFrame:
    """Return aligned daily returns DataFrame (index=date, columns=tickers)."""
    prices = pd.concat(price_map, axis=1).dropna()
    rets = prices.pct_change().dropna()
    rets.columns = prices.columns
    return rets


def _max_drawdown_from_returns(returns: pd.Series) -> float:
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def _var_cvar(returns: pd.Series, alpha: float = 0.05) -> Tuple[float, float]:
    q = float(np.quantile(returns, alpha))
    tail = returns[returns <= q]
    cvar = float(tail.mean()) if len(tail) else q
    return q, cvar


def _beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    # Align
    df = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    pr = df.iloc[:, 0]
    br = df.iloc[:, 1]
    var = float(np.var(br, ddof=1))
    if var == 0:
        return 0.0
    cov = float(np.cov(pr, br, ddof=1)[0, 1])
    return cov / var


def _feature_vector(window_port: pd.Series, window_bench: pd.Series, rf_rate_annual: float = 0.0) -> Tuple[List[float], List[str]]:
    """Compute a feature vector from a historical window."""
    # Annualization assumes ~252 trading days
    mu_daily = float(window_port.mean())
    vol_daily = float(window_port.std(ddof=1))
    ann_ret = (1.0 + mu_daily) ** 252 - 1.0
    ann_vol = vol_daily * np.sqrt(252)

    rf_daily = (1.0 + rf_rate_annual) ** (1.0 / 252.0) - 1.0
    excess = window_port - rf_daily
    sharpe = float(excess.mean() / (excess.std(ddof=1) + 1e-12) * np.sqrt(252))

    mdd = _max_drawdown_from_returns(window_port)
    var95, cvar95 = _var_cvar(window_port, alpha=0.05)
    beta = _beta(window_port, window_bench)

    feats = [ann_ret, ann_vol, sharpe, mdd, var95, cvar95, beta]
    names = ["ann_return", "ann_vol", "sharpe", "max_drawdown", "var_95", "cvar_95", "beta"]
    return feats, names


def build_training_data(
    prices_dir: str,
    assets: List[str],
    benchmark: str,
    n_portfolios: int = 300,
    lookback: int = 60,
    horizon: int = 21,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Build a supervised dataset using ONLY price history.

    We generate many random portfolios using the available assets, then:
    - Features: computed from the past `lookback` daily returns window.
    - Label: future max drawdown over the next `horizon` days.
      We bucket labels into 3 classes using tertiles (low/med/high risk).

    This avoids hand-labeling and uses objective, reproducible targets.
    """
    rng = np.random.default_rng(seed)

    price_map = _load_prices(prices_dir, assets + [benchmark])
    returns = _align_returns(price_map)  # includes benchmark col

    bench = returns[benchmark]
    asset_returns = returns[assets]

    # Build samples
    X_rows: List[List[float]] = []
    y_drawdowns: List[float] = []
    feature_names: List[str] | None = None

    # Choose random weight vectors
    weights = rng.dirichlet(alpha=np.ones(len(assets)), size=n_portfolios)

    # Choose dates that have both lookback and horizon
    # We'll use positions in the returns index
    idx = returns.index
    min_i = lookback
    max_i = len(idx) - horizon - 1
    if max_i <= min_i:
        raise ValueError("Not enough history in CSVs to build training data. Add more rows to price files.")

    for w in weights:
        # portfolio returns series
        port = (asset_returns * w).sum(axis=1)

        for i in range(min_i, max_i):
            window_port = port.iloc[i - lookback : i]
            window_bench = bench.iloc[i - lookback : i]
            feats, names = _feature_vector(window_port, window_bench)
            if feature_names is None:
                feature_names = names

            future = port.iloc[i : i + horizon]
            fut_mdd = _max_drawdown_from_returns(future)

            X_rows.append(feats)
            y_drawdowns.append(fut_mdd)

    X = pd.DataFrame(X_rows, columns=feature_names)
    y_dd = pd.Series(y_drawdowns, name="future_max_drawdown")

    # Convert continuous drawdown target into 3 buckets by tertiles
    # More negative drawdown => higher risk
    q1 = float(y_dd.quantile(1 / 3))
    q2 = float(y_dd.quantile(2 / 3))

    def bucket(dd: float) -> str:
        if dd <= q1:  # worst drawdowns
            return "Aggressive"
        if dd <= q2:
            return "Moderate"
        return "Conservative"

    y = y_dd.apply(bucket).astype(str)
    return X, y, list(feature_names)


def train_risk_classifier(
    prices_dir: str,
    assets: List[str],
    benchmark: str = "SPY",
    n_portfolios: int = 300,
    lookback: int = 60,
    horizon: int = 21,
    seed: int = 42,
    out_dir: str = "models",
    model_name: str = "risk_classifier.joblib",
) -> TrainResult:
    """Train a real classifier on price-derived labels and persist the best model."""
    if RandomForestClassifier is None:
        raise RuntimeError("scikit-learn is required for ML training. Install requirements.txt")

    X, y, feature_names = build_training_data(
        prices_dir=prices_dir,
        assets=assets,
        benchmark=benchmark,
        n_portfolios=n_portfolios,
        lookback=lookback,
        horizon=horizon,
        seed=seed,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    # Two models: RandomForest (non-linear) and LogisticRegression (baseline)
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )
    lr = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, multi_class="auto")),
        ]
    )

    rf.fit(X_train, y_train)
    lr.fit(X_train, y_train)

    # Pick the better model by accuracy on the test set
    rf_acc = float((rf.predict(X_test) == y_test).mean())
    lr_acc = float((lr.predict(X_test) == y_test).mean())
    best = rf if rf_acc >= lr_acc else lr

    y_pred = best.predict(X_test)
    rep = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=["Conservative", "Moderate", "Aggressive"])

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_path = str(out / model_name)
    joblib.dump(
        {
            "model": best,
            "feature_names": feature_names,
            "assets": assets,
            "benchmark": benchmark,
            "lookback": lookback,
            "horizon": horizon,
        },
        model_path,
    )

    return TrainResult(
        model_path=model_path,
        report=rep,
        confusion=cm.tolist(),
        classes=["Conservative", "Moderate", "Aggressive"],
        feature_names=feature_names,
    )


def predict_latest_bucket(
    model_path: str,
    prices_dir: str,
) -> Dict[str, object]:
    """Predict risk bucket for the *latest* window implied by the saved model."""
    if joblib is None:
        raise RuntimeError("joblib is required for ML inference. Install requirements.txt")

    blob = joblib.load(model_path)
    model = blob["model"]
    feature_names: List[str] = blob["feature_names"]
    assets: List[str] = blob["assets"]
    benchmark: str = blob["benchmark"]
    lookback: int = int(blob["lookback"])

    price_map = _load_prices(prices_dir, assets + [benchmark])
    returns = _align_returns(price_map)
    bench = returns[benchmark]
    asset_returns = returns[assets]

    # Use equal weights for a generic "market" portfolio if user just wants a demo prediction.
    w = np.array([1.0 / len(assets)] * len(assets))
    port = (asset_returns * w).sum(axis=1)

    window_port = port.iloc[-lookback:]
    window_bench = bench.iloc[-lookback:]
    feats, _ = _feature_vector(window_port, window_bench)
    X = pd.DataFrame([feats], columns=feature_names)

    pred = str(model.predict(X)[0])
    proba = getattr(model, "predict_proba", None)
    probs = None
    if callable(proba):
        p = model.predict_proba(X)[0]
        classes = list(model.classes_)
        probs = {classes[i]: float(p[i]) for i in range(len(classes))}

    return {"prediction": pred, "probabilities": probs, "assets": assets, "benchmark": benchmark, "lookback": lookback}
