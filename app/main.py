from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from src.analysis import analyze_portfolio

app = FastAPI(title="Portfolio Risk Analyzer", version="1.0.0")


class Holding(BaseModel):
    ticker: str = Field(..., examples=["AAPL"])
    weight: float = Field(..., ge=0.0, le=1.0, examples=[0.25])


class AnalyzeRequest(BaseModel):
    as_of: Optional[str] = Field(None, examples=["2026-02-21"])
    base_currency: Optional[str] = Field("USD", examples=["USD"])
    benchmark: Optional[str] = Field("SPY", examples=["SPY"])
    holdings: List[Holding]
    prices_dir: str = Field(..., description="Path to directory containing <TICKER>.csv price files", examples=["data/prices"])


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest) -> Dict[str, Any]:
    try:
        portfolio = {
            "as_of": req.as_of,
            "base_currency": req.base_currency,
            "benchmark": req.benchmark,
            "holdings": [h.model_dump() for h in req.holdings],
        }
        return analyze_portfolio(portfolio, prices_dir=req.prices_dir)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
