from __future__ import annotations

from typing import Dict, List, Any
import json
import os
import pandas as pd


def load_portfolio_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_price_csv(prices_dir: str, ticker: str) -> pd.DataFrame:
    path = os.path.join(prices_dir, f"{ticker}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing price file for {ticker}: {path}")
    df = pd.read_csv(path)
    return df


def load_price_map(prices_dir: str, tickers: List[str]) -> Dict[str, pd.DataFrame]:
    return {t: load_price_csv(prices_dir, t) for t in tickers}
