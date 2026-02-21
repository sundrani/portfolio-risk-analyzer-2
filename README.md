# Portfolio Risk Analyzer (Risk Score + Allocation Bucket) 📈🛡️

A small, production-style Python project that:
- Computes portfolio risk metrics (volatility, max drawdown, VaR/CVaR, Sharpe, beta vs benchmark)
- Produces a **risk score (0–100)** and **risk bucket** (Conservative / Moderate / Aggressive)
- Optionally trains a lightweight ML classifier on synthetic data (so you can demo *Decision Trees / Model Selection* ideas)
- Exposes a **FastAPI** service with Swagger UI

> **No external API keys required.** The project runs offline using the included sample price data.  
> You can replace the sample CSVs with your own historical prices.

---

## Quick Start (VS Code)

### 1) Create & activate a virtual environment
```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Run the API
```bash
uvicorn app.main:app --reload
```

Open:
- Swagger UI: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

---

## Try it

### Analyze the included sample portfolio
```bash
python -m src.cli analyze --portfolio data/sample_portfolio.json --prices_dir data/prices
```

### Call the API
```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -H "Content-Type: application/json" \
  -d @data/sample_request.json
```


---

## ML Upgrade: Train a real risk-bucket classifier (price-derived labels)

The core analyzer is **formula + rules** (explainable quant metrics).  
This project also includes an **optional ML pipeline** that learns to predict risk buckets from price history:

- Builds a supervised dataset from historical prices
  - **Features**: computed from the past `lookback` window (ann. return, ann. vol, Sharpe, max drawdown, VaR/CVaR, beta)
  - **Label**: the **future max drawdown** over the next `horizon` days
  - Buckets into **Conservative / Moderate / Aggressive** using tertiles (no manual labels required)
- Trains **RandomForestClassifier** (plus a Logistic Regression baseline)
- Saves the best model to `models/risk_classifier.joblib`

### Train (uses the included sample prices)
```bash
python -m src.cli train-ml-real \
  --prices_dir data/prices \
  --assets AAPL MSFT TLT \
  --benchmark SPY \
  --n_portfolios 300 \
  --lookback 60 \
  --horizon 21
```

### Predict the latest bucket using the saved model
```bash
python -m src.cli predict-ml \
  --model_path models/risk_classifier.joblib \
  --prices_dir data/prices
```

> Tip: For better ML performance, replace the sample CSVs with **more history** and **more tickers**.

---

## Input Formats

### Portfolio file (`data/sample_portfolio.json`)
```json
{
  "as_of": "2026-02-21",
  "base_currency": "USD",
  "benchmark": "SPY",
  "holdings": [
    {"ticker": "AAPL", "weight": 0.35},
    {"ticker": "MSFT", "weight": 0.35},
    {"ticker": "TLT",  "weight": 0.30}
  ]
}
```

### Price CSVs (`data/prices/<TICKER>.csv`)
Each CSV must have columns:
- `date` (YYYY-MM-DD)
- `close` (float)

Example:
```csv
date,close
2025-01-02,192.53
2025-01-03,191.15
...
```

---

## Docker (optional)
```bash
docker build -t portfolio-risk-analyzer .
docker run -p 8000:8000 portfolio-risk-analyzer
```



## Project Structure
```
portfolio-risk-analyzer/
  app/                # FastAPI service
  src/                # Core logic (metrics, scoring, optional ML)
  data/               # Sample portfolio + sample prices
  tests/              # Basic unit tests
```

---
