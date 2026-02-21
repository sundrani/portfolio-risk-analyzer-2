from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io_utils import load_portfolio_json
from .analysis import analyze_portfolio
from .ml_model import train_decision_tree
from .ml_real import train_risk_classifier, predict_latest_bucket


def cmd_analyze(args: argparse.Namespace) -> int:
    portfolio = load_portfolio_json(args.portfolio)
    out = analyze_portfolio(portfolio, prices_dir=args.prices_dir)
    print(json.dumps(out, indent=2))
    return 0


def cmd_train_ml(args: argparse.Namespace) -> int:
    res = train_decision_tree(seed=args.seed)
    print("Decision Tree (synthetic risk classifier) report:")
    print(res.report)
    return 0



def cmd_train_ml_real(args: argparse.Namespace) -> int:
    res = train_risk_classifier(
        prices_dir=args.prices_dir,
        assets=args.assets,
        benchmark=args.benchmark,
        n_portfolios=args.n_portfolios,
        lookback=args.lookback,
        horizon=args.horizon,
        seed=args.seed,
        out_dir=args.out_dir,
        model_name=args.model_name,
    )
    print(f"Saved model to: {res.model_path}")
    print(res.report)
    print("Confusion matrix (rows=true, cols=pred) in order [Conservative, Moderate, Aggressive]:")
    print(json.dumps(res.confusion))
    return 0


def cmd_predict_ml(args: argparse.Namespace) -> int:
    out = predict_latest_bucket(model_path=args.model_path, prices_dir=args.prices_dir)
    print(json.dumps(out, indent=2))
    return 0

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="portfolio-risk-analyzer")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="Analyze a portfolio using local price CSVs")
    a.add_argument("--portfolio", required=True, help="Path to portfolio JSON")
    a.add_argument("--prices_dir", required=True, help="Directory with <TICKER>.csv files")
    a.set_defaults(func=cmd_analyze)

    t = sub.add_parser("train-ml", help="Train a small decision tree model on synthetic data (demo)")
    t.add_argument("--seed", type=int, default=42)
    t.set_defaults(func=cmd_train_ml)

    r = sub.add_parser(
        "train-ml-real",
        help="Train a real risk bucket classifier using price-derived labels (RandomForest + LogisticRegression baseline)",
    )
    r.add_argument("--prices_dir", required=True, help="Directory with <TICKER>.csv files")
    r.add_argument("--assets", nargs="+", required=True, help="Asset tickers used to generate random portfolios")
    r.add_argument("--benchmark", default="SPY", help="Benchmark ticker used for beta feature (default: SPY)")
    r.add_argument("--n_portfolios", type=int, default=300, help="Number of random portfolios to sample")
    r.add_argument("--lookback", type=int, default=60, help="Lookback window (trading days) for features")
    r.add_argument("--horizon", type=int, default=21, help="Future horizon (days) used to label risk via drawdown")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--out_dir", default="models", help="Directory to save trained model")
    r.add_argument("--model_name", default="risk_classifier.joblib")
    r.set_defaults(func=cmd_train_ml_real)

    pml = sub.add_parser("predict-ml", help="Predict latest risk bucket using a saved ML model")
    pml.add_argument("--model_path", required=True, help="Path to saved joblib model")
    pml.add_argument("--prices_dir", required=True, help="Directory with <TICKER>.csv files")
    pml.set_defaults(func=cmd_predict_ml)


    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
