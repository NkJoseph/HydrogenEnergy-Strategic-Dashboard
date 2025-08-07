# Hydrogen Forecast Dashboard (v2)

FastAPI + Dash wrapper around pretrained deep‑learning (DL) forecasts and
reinforcement‑learning (RL) optimisation for global & regional hydrogen capacity
pathways.

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch REST API
uvicorn api:app --reload --port 8000 &

# 3. Open the Dash UI
python dashboard.py
```

* API docs: <http://localhost:8000/docs>  
* Dashboard: <http://localhost:8050>

The wrappers in **`analysis_wrappers.py`** import your research scripts
(`global analysis.py`, `regional analysis.py`) unchanged, add lightweight
helper functions, and route requests from the web layer to your pretrained
models stored in **`/pretrained-models/`**.
