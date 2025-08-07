"""
Lightweight wrappers that expose clean `predict`, `optimize_to_target`, and
`available_regions` functions expected by the production API. These functions
delegate to the heavy research scripts *global analysis.py* and
*regional analysis.py* without altering them.

Both research scripts are imported via `importlib.machinery.SourceFileLoader`
so we can keep their original filenames (including spaces) intact.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import importlib.machinery
import inspect
from typing import List

import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "pretrained-models"

# -----------------------------------------------------------------------------
# Dynamic import of the research pipelines (keeps original filenames)
# -----------------------------------------------------------------------------
global_analysis = importlib.machinery.SourceFileLoader(
    "global_analysis", str(ROOT / "global analysis.py")
).load_module()

regional_analysis = importlib.machinery.SourceFileLoader(
    "regional_analysis", str(ROOT / "regional analysis.py")
).load_module()

# -----------------------------------------------------------------------------
# Helper shims – we detect which helper names exist and call the right one.
# -----------------------------------------------------------------------------
def _call_helper(obj, *candidates, **kwargs):
    """
    Call the first attribute in *candidates* that exists on *obj*.
    Raise a clear error if none are found.
    """
    for name in candidates:
        if hasattr(obj, name):
            fn = getattr(obj, name)
            return fn(**kwargs) if inspect.isfunction(fn) else fn
    raise AttributeError(f"No helper among {candidates} found in {obj.__name__}")


# -----------------------------------------------------------------------------
# Public wrappers
# -----------------------------------------------------------------------------
@lru_cache(maxsize=128)
def predict(region: str, year: int, scenario: str = "BAU") -> pd.DataFrame:
    """
    Return a tidy DataFrame with a datetime-like index and a single column
    representing the DL forecast for the specified horizon.
    """
    if region.lower() == "global":
        # Try typical helper names seen in the research script
        df = _call_helper(
            global_analysis,
            "predict_future_total_capacity",
            "predict",
            year=year,
            scenario=scenario,
        )
    else:
        df = _call_helper(
            regional_analysis,
            "predict_future_regional_total_capacity",
            "predict",
            region=region,
            year=year,
            scenario=scenario,
        )
    # Ensure DataFrame
    if isinstance(df, pd.Series):
        df = df.to_frame("value")
    elif isinstance(df, pd.DataFrame) and df.shape[1] == 1:
        df.columns = ["value"]
    return df


@lru_cache(maxsize=128)
def optimize_to_target(
    region: str, year: int, target: float, scenario: str = "BAU"
) -> pd.DataFrame:
    """
    Return a DataFrame with the RL‑adjusted pathway to hit *target* (Mt H₂).
    """
    if region.lower() == "global":
        df = _call_helper(
            global_analysis,
            "rl_pathway_to_target",
            "optimize_to_target",
            year=year,
            target=target,
            scenario=scenario,
        )
    else:
        df = _call_helper(
            regional_analysis,
            "rl_pathway_to_target",
            "optimize_to_target",
            region=region,
            year=year,
            target=target,
            scenario=scenario,
        )
    if isinstance(df, pd.Series):
        df = df.to_frame("value")
    elif isinstance(df, pd.DataFrame) and df.shape[1] == 1:
        df.columns = ["value"]
    return df


def available_regions() -> List[str]:
    """
    Return list of region names recognised by the regional model.
    """
    if hasattr(regional_analysis, "available_regions"):
        return list(regional_analysis.available_regions())
    # Fallback: try to infer from DataFrame index / column names
    try:
        regions = _call_helper(regional_analysis, "infer_regions")
        if isinstance(regions, list):
            return regions
    except Exception:
        pass
    # As a safe default return common geographical list – should be patched
    return [
        "Africa",
        "Australia",
        "Canada",
        "China",
        "EU",
        "EastAsia",
        "LatinAmerica",
        "MiddleEast",
        "SouthAsia",
        "USA",
    ]
