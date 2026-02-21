import json
from src.io_utils import load_portfolio_json
from src.analysis import analyze_portfolio


def test_analyze_sample():
    portfolio = load_portfolio_json("data/sample_portfolio.json")
    out = analyze_portfolio(portfolio, prices_dir="data/prices")
    assert "metrics" in out
    assert "risk_assessment" in out
    assert 0 <= out["risk_assessment"]["risk_score"] <= 100
