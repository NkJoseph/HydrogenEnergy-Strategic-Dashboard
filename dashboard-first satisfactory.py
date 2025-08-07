"""
dashboard.py — hydrogen-forecast research UI
June 2025 • Sticky header • Target year • Tabbed panes • Colour-coded targets
"""
from __future__ import annotations

import datetime, functools, json, urllib.parse
import pandas as pd, requests, flask
import dash
from dash import dcc, html, Input, Output, State, ctx, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px

# ───── Constants ────────────────────────────────────────────────────
API_URL = "http://localhost:8000"

REGIONS = [
    "Africa", "Canada", "USA", "India", "China", "Europe",
    "East & Southeast Asia", "South Asia", "Central Asia",
    "Middle East", "Latin America", "Australia", "Oceania", "Others",
]
YEARS       = list(range(datetime.date.today().year + 1, 2201))
SCENARIOS   = ["Optimistic", "BAU", "Pessimistic"]
THEMES      = {"Light": dbc.themes.FLATLY, "Dark": dbc.themes.DARKLY}
DEFAULT_THEME = "Light"

INST_TARGETS = {"IEA 430": 430, "IPCC 155": 155, "IRENA 523": 523}
PALETTE = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]  # dashed lines

# ───── Helpers ─────────────────────────────────────────────────────
@functools.lru_cache(maxsize=256)
def _fetch(endpoint: str, *, m: str, p: str):
    payload = json.loads(p)
    r = (
        requests.get(f"{API_URL}{endpoint}", params=payload, timeout=30)
        if m == "GET"
        else requests.post(f"{API_URL}{endpoint}", json=payload, timeout=30)
    )
    r.raise_for_status()
    return r.json()

def fetch(ep, method="GET", payload=None):
    return _fetch(ep, m=method, p=json.dumps(payload or {}, sort_keys=True))

def recs_to_series(data, use_cumulative=False):
    """Convert API response to pandas Series"""
    if isinstance(data, dict) and 'years' in data:
        # New API format with years and yearly/cumulative arrays
        years = data['years']
        values = data['cumulative'] if use_cumulative else data['yearly']
        return pd.Series(values, index=years) if years and values else pd.Series(dtype=float)
    elif isinstance(data, list):
        # Old API format with array of records
        df = pd.DataFrame(data)
        return pd.Series(df["value"].values, index=df["year"]) if not df.empty else pd.Series(dtype=float)
    else:
        return pd.Series(dtype=float)

def default_series(start_year: int):
    return pd.Series([0] * 6, index=[start_year + i for i in range(6)])

# ───── Dash factory ───────────────────────────────────────────────
def make_app(theme: str = DEFAULT_THEME) -> dash.Dash:
    app = dash.Dash(__name__, external_stylesheets=[THEMES[theme]], suppress_callback_exceptions=True)
    server = app.server

    # ═════════ Sticky Global Settings bar ══════════════════════════
    global_bar = dbc.Card(
        dbc.CardBody(
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Scope"),
                            dbc.RadioItems(
                                id="scope",
                                options=[{"label": "Global", "value": "global"},
                                         {"label": "Regional", "value": "regional"}],
                                value="global",
                                inline=True,
                            ),
                        ],
                        md=2,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Regions", id="reg-lab"),
                            dcc.Dropdown(
                                id="regions",
                                options=[{"label": r, "value": r} for r in REGIONS],
                                multi=True,
                                placeholder="All regions",
                                disabled=True,
                            ),
                            dbc.Tooltip("Select one or many macro-regions when Scope = Regional",
                                        target="reg-lab"),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Target Year"),
                            dcc.Dropdown(
                                id="year",
                                options=[{"label": y, "value": y} for y in YEARS],
                                value=YEARS[0],
                                clearable=False,
                            ),
                        ],
                        md=2,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Institutional targets"),
                            dbc.Checklist(
                                id="inst",
                                options=[{"label": k, "value": v} for k, v in INST_TARGETS.items()],
                                value=[430],
                                inline=True,
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Theme"),
                            dcc.Dropdown(
                                id="theme",
                                options=[{"label": k, "value": k} for k in THEMES],
                                value=theme,
                                clearable=False,
                            ),
                        ],
                        md=2,
                    ),
                ],
                className="g-2",
            )
        ),
        style={"position": "sticky", "top": "0", "zIndex": "1020"},
        class_name="shadow-sm",
    )

    # ═════════ Forecast controls (right pane) ══════════════════════
    fc_ctrl = dbc.Card(
        dbc.CardBody(
            [
                html.Div(id="fc-reg-titles"),
                dbc.Label("Forecast view"),
                dbc.RadioItems(
                    id="fc-view",
                    options=[{"label": l, "value": v} for l, v in
                             [("Annual", "annual"), ("Cumulative", "cum"), ("Map", "map"), ("All", "all")]],
                    value="annual",
                ),
                html.Hr(className="my-2"),
                dbc.Label("Scenario (for cumulative)"),
                dcc.Dropdown(
                    id="fc-scenario",
                    options=[{"label": s, "value": s} for s in SCENARIOS],
                    value="BAU",
                    clearable=False,
                ),
                html.Hr(className="my-2"),
                dbc.ButtonGroup(
                    [
                        dbc.Button("CSV", id="fc-csv", size="sm", color="secondary"),
                        dbc.Button("PNG", id="fc-png", size="sm", color="secondary"),
                    ]
                ),
            ]
        ),
        class_name="h-100",
    )

    # ═════════ Strategic controls (right pane) ═════════════════════
    strat_ctrl = dbc.Card(
        dbc.CardBody(
            [
                html.Div(id="strat-reg-titles"),
                dbc.Label("Curve dynamics"),
                dbc.Checklist(
                    id="strat-dyn",
                    options=[{"label": "Early investment", "value": "early"},
                             {"label": "Late investment", "value": "late"}],
                    value=["early"],
                ),
                dbc.Checkbox(id="ci", label="Confidence interval", value=False),
                dbc.Checkbox(id="strat-map", label="Show map", value=False),
                html.Hr(className="my-2"),
                dbc.ButtonGroup(
                    [
                        dbc.Button("CSV", id="strat-csv", size="sm", color="secondary"),
                        dbc.Button("PNG", id="strat-png", size="sm", color="secondary"),
                    ]
                ),
            ]
        ),
        class_name="h-100",
    )

    # ═════════ Tab panes (each fills page height) ══════════════════
    forecast_tab = dbc.Row(
        [
            dbc.Col(
                [
                    html.H4("Forecasting Hydrogen Production", className="mb-1"),
                    html.H6(id="fc-sub", className="text-muted mb-2"),
                    dcc.Graph(id="fc-graph", style={"height": "70vh"}),
                ],
                md=10,
            ),
            dbc.Col(fc_ctrl, md=2),
        ],
        class_name="g-2",
    )

    strat_tab = dbc.Row(
        [
            dbc.Col(
                [
                    html.H4("Strategic Plan: Correcting the Errors", className="mb-1"),
                    html.H6(id="strat-sub", className="text-muted mb-2"),
                    dcc.Graph(id="strat-graph", style={"height": "70vh"}),
                ],
                md=10,
            ),
            dbc.Col(strat_ctrl, md=2),
        ],
        class_name="g-2",
    )

    # ═════════ Page layout ═════════════════════════════════════════
    app.layout = dbc.Container(
        [
            dbc.NavbarSimple(brand="Hydrogen Forecast Dashboard", color="primary", dark=True),
            global_bar,
            dbc.Tabs(
                [
                    dbc.Tab(forecast_tab, label="Forecasting"),
                    dbc.Tab(strat_tab, label="Strategic Plan"),
                ]
            ),
            dcc.Download(id="dl"),
            dcc.Location(id="loc", refresh=True),
        ],
        fluid=True,
        class_name="px-4",
    )

    # ───── Interactivity ──────────────────────────────────────────
    @app.callback(Output("regions", "disabled"), Input("scope", "value"))
    def _toggle_regions(scope): return scope == "global"

    # Header helpers
    def _headers(scope, regs):
        if scope == "global":
            return html.H6("Global", className="text-muted fw-semibold")
        regs = regs or REGIONS
        return [html.H6(r, className="text-muted fw-semibold") for r in regs]

    app.callback(Output("fc-reg-titles", "children"), Input("scope", "value"), Input("regions", "value"))(_headers)
    app.callback(Output("strat-reg-titles", "children"), Input("scope", "value"), Input("regions", "value"))(_headers)

    def _subtitle(scope, regs, year):
        base = f"Year: {year}"
        if scope == "global":
            return f"{base} • Scope: Global"
        regs = regs or REGIONS
        return f"{base} • Scope: {', '.join(regs)}"

    app.callback(Output("fc-sub", "children"), Input("scope", "value"),
                 Input("regions", "value"), Input("year", "value"))(_subtitle)
    app.callback(Output("strat-sub", "children"), Input("scope", "value"),
                 Input("regions", "value"), Input("year", "value"))(_subtitle)

    # ---------------- Forecast figure ----------------------------
    @app.callback(
        Output("fc-graph", "figure"),
        Input("scope", "value"), Input("regions", "value"),
        Input("fc-view", "value"), Input("fc-scenario", "value"),
        Input("inst", "value"), Input("year", "value"),
    )
    def _forecast(scope, regs, view, scenario, targets, yr):
        regs = regs or REGIONS if scope == "regional" else ["Global"]

        # Map view
        if view == "map":
            rows = []
            for r in REGIONS:
                try:
                    data = fetch("/forecast", payload={"region": r, "start": yr, "end": yr, "scenario": scenario})
                    if isinstance(data, dict) and 'yearly' in data and data['yearly']:
                        value = data['yearly'][0]  # First year value
                    else:
                        value = 0
                    rows.append({"region": r, "value": value})
                except Exception:
                    rows.append({"region": r, "value": 0})
            
            fig = px.choropleth(pd.DataFrame(rows), locations="region", locationmode="geojson-id",
                                color="value", color_continuous_scale="Viridis",
                                labels=dict(value="Production (Mt)"))
            fig.update_geos(fitbounds="locations", visible=False)
            fig.update_layout(template="simple_white", margin=dict(r=0, l=0, t=20, b=0))
            return fig

        # Line views (Annual or Cumulative)
        fig = go.Figure()
        use_cumulative = (view == "cum")
        
        for r in regs:
            try:
                # Fetch data for the full year range
                data = fetch("/forecast", payload={"region": r, "start": yr, "end": yr + 5, "scenario": scenario})
                ser = recs_to_series(data, use_cumulative=use_cumulative)
            except Exception:
                ser = default_series(yr)
            
            if ser.empty:
                ser = default_series(yr)
            
            # For cumulative view, use the data from API directly (no need to call cumsum again)
            fig.add_trace(go.Scatter(x=ser.index, y=ser, name=r, mode="lines"))

        # Target lines (colour-coded dashed) - only meaningful for cumulative view
        if use_cumulative:
            for i, v in enumerate(targets):
                col = PALETTE[i % len(PALETTE)]
                fig.add_hline(y=v, line=dict(dash="dash", color=col, width=2),
                              annotation_text=f"{v} Mt", annotation_font_color=col,
                              annotation_position="top right")

        # Update labels based on view
        y_label = "Cumulative Hydrogen Capacity (Mt)" if use_cumulative else "Annual Hydrogen Capacity (Mt)"
        title_suffix = " - Cumulative" if use_cumulative else " - Annual"
        
        fig.update_layout(template="simple_white", height=650,
                          xaxis_title="Year", yaxis_title=y_label,
                          title_text=f"Hydrogen Production Forecast{title_suffix}")
        return fig

    # ---------------- Strategic figure ---------------------------
    @app.callback(
        Output("strat-graph", "figure"),
        Input("scope", "value"), Input("regions", "value"),
        Input("strat-dyn", "value"), Input("ci", "value"),
        Input("strat-map", "value"), Input("year", "value"),
    )
    def _strategic(scope, regs, dyn, ci, show_map, yr):
        regs = regs or REGIONS if scope == "regional" else ["Global"]

        # Map view
        if show_map:
            df = pd.DataFrame({"region": REGIONS, "value": [None] * len(REGIONS)})
            fig = px.choropleth(df, locations="region", locationmode="geojson-id",
                                color="value", color_continuous_scale="Viridis")
            fig.update_geos(fitbounds="locations", visible=False)
            fig.update_layout(template="simple_white", margin=dict(r=0, l=0, t=20, b=0))
            return fig

        fig = go.Figure()
        for r in regs:
            try:
                recs = fetch("/optimize", method="POST",
                             payload={"region": r, "year": yr, "target": 430, "scenario": "BAU"})
                ser = recs_to_series(recs)
            except Exception:
                ser = default_series(yr)
            if ser.empty:
                ser = default_series(yr)
            fig.add_trace(go.Scatter(x=ser.index, y=ser, name=f"RL {r}", mode="lines"))
            if ci:
                fig.add_trace(go.Scatter(x=ser.index, y=ser * 1.1, line=dict(width=0),
                                         hoverinfo="skip", showlegend=False))
                fig.add_trace(go.Scatter(x=ser.index, y=ser * 0.9, line=dict(width=0),
                                         hoverinfo="skip", showlegend=False,
                                         fill="tonexty", fillcolor="rgba(50,150,80,.15)"))

        if "early" in dyn and "late" not in dyn:
            fig.update_layout(title_text="Early-investment focus")
        elif "late" in dyn and "early" not in dyn:
            fig.update_layout(title_text="Late-investment focus")

        fig.update_layout(template="simple_white", height=650,
                          xaxis_title="Year", yaxis_title="Hydrogen Capacity (Mt)")
        return fig

    # ---------------- Theme reload -------------------------------
    @app.callback(Output("loc", "href"), Input("theme", "value"), prevent_initial_call=True)
    def _reload_theme(selected):
        args = flask.request.args.to_dict()
        args["theme"] = selected
        return flask.request.path + "?" + urllib.parse.urlencode(args)

    return app

# ───── Run ───────────────────────────────────────────────────────
initial = flask.request.args.get("theme", DEFAULT_THEME) if flask.has_request_context() else DEFAULT_THEME
app = make_app(initial)

if __name__ == "__main__":
    app.run_server(port=8050, debug=True)
