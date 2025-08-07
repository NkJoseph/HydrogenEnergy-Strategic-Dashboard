from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Literal
import pandas as pd

from analysis_wrappers import predict, optimize_to_target, available_regions

app = FastAPI(
    title="Hydrogen Forecast API",
    description="DL + RL hydrogen forecasts backed by pretrained models.",
    version="0.2.0",
)

# CORS so Dash (same host) can fetch
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class OptimizePayload(BaseModel):
    region: str = Field(..., description='"Global" or a region returned by /meta/regions')
    year: int = Field(..., ge=1900, description="Forecast start year")
    target: float = Field(..., gt=0, description="Target capacity (Mt H₂)")
    scenario: Literal["BAU", "Optimistic", "Net-Zero"] = "BAU"

def _df_to_records(df: pd.DataFrame, series: str):
    df = df.reset_index().rename(columns={df.index.name or "index": "year", df.columns[0]: "value"})
    df["series"] = series
    return df.to_dict(orient="records")

# ------------------------------------------------------------------ Endpoints
@app.get("/meta/regions", response_model=List[str])
def get_regions():
    return ["Global"] + available_regions()

@app.get("/forecast")
def get_forecast(region: str, year: int, scenario: str = "BAU"):
    try:
        df = predict(region, year, scenario)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return _df_to_records(df, "dl")

@app.post("/optimize")
def post_optimize(payload: OptimizePayload):
    try:
        df = optimize_to_target(payload.region, payload.year, payload.target, payload.scenario)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return _df_to_records(df, "rl")
