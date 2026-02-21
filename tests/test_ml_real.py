from src.ml_real import build_training_data

def test_build_training_data_smoke():
    X, y, feature_names = build_training_data(
        prices_dir="data/prices",
        assets=["AAPL", "MSFT", "TLT"],
        benchmark="SPY",
        n_portfolios=5,
        lookback=20,
        horizon=5,
        seed=1,
    )
    assert len(X) == len(y)
    assert len(feature_names) == X.shape[1]
    assert set(y.unique()).issubset({"Conservative", "Moderate", "Aggressive"})
