"""
dashboard.py — Global Hydrogen Production Simulation Tool
Sticky header · Start/End year in side panes · Tight subtitle spacing
"""

from __future__ import annotations
import datetime, functools, json, urllib.parse
import pandas as pd, requests, flask
import dash
from dash import dcc, html, Input, Output, State, ctx, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px

# ───────── Config ───────────────────────────────────────────────
API_URL = "http://127.0.0.1:8000"
REGIONS = [
    "Africa", "Canada", "USA", "India", "China", "Europe",
    "East & Southeast Asia", "South Asia", "Central Asia",
    "Middle East", "Latin America", "Australia", "Oceania", "Others",
]
CUR_Y   = datetime.date.today().year
YEARS   = list(range(CUR_Y + 1, 2201))
SCEN    = ["Optimistic", "BAU", "Pessimistic"]
THEMES  = {"Light": dbc.themes.FLATLY, "Dark": dbc.themes.DARKLY}
DEFAULT_THEME = "Light"

INST_TARGETS = {"IEA 430": 430, "IPCC 155": 155, "IRENA 523": 523}
PALETTE      = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]

# ───────── Helpers ──────────────────────────────────────────────
@functools.lru_cache(maxsize=512)
def _cached(ep: str, *, m: str, p: str):
    payload = json.loads(p)
    req = requests.get if m == "GET" else requests.post
    try:
        r = req(f"{API_URL}{ep}",
                params=payload if m == "GET" else None,
                json=payload if m == "POST" else None,
                timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Error fetching {API_URL}{ep}: {e}")
        raise

def fetch(ep, method="GET", payload=None):
    return _cached(ep, m=method, p=json.dumps(payload or {}, sort_keys=True))

def to_series(data, y0, y1, use_cumulative=False):
    """Convert API response to pandas Series - now handles multiple models"""
    if isinstance(data, dict) and 'models' in data:
        # New API format with multiple models
        models_data = {}
        for model_name, model_data in data['models'].items():
            years = model_data['years']
            if use_cumulative and 'cumulative' in model_data:
                values = model_data['cumulative']
            else:
                values = model_data['yearly']
            if years and values:
                ser = pd.Series(values, index=years).loc[y0:y1]
                models_data[model_name] = ser
            else:
                models_data[model_name] = pd.Series(dtype=float)
        return models_data
    elif isinstance(data, dict) and 'years' in data:
        # Old API format with single series
        years = data['years']
        values = data['cumulative'] if use_cumulative else data['yearly']
        if years and values:
            return pd.Series(values, index=years).loc[y0:y1]
        else:
            return pd.Series(dtype=float)
    elif isinstance(data, list):
        # Legacy format with array of records
        df = pd.DataFrame(data)
        if df.empty:
            return pd.Series(dtype=float)
        return pd.Series(df["value"].values, index=df["year"]).loc[y0:y1]
    else:
        return pd.Series(dtype=float)

def default_ser(y0, y1):
    yrs = list(range(y0, y1 + 1))
    return pd.Series([0] * len(yrs), index=yrs)

# ───────── Dash factory ─────────────────────────────────────────
def make_app(theme: str = DEFAULT_THEME) -> dash.Dash:
    app = dash.Dash(__name__, external_stylesheets=[THEMES[theme]], suppress_callback_exceptions=True)
    server = app.server

    # ══ Navbar ═══════════════════════════════════════════════════
    navbar = dbc.Navbar(
        dbc.Container(
            [
                dbc.NavbarBrand("Global Hydrogen Production Simulation Tool", className="me-auto"),
                dbc.Nav(
                    [
                        dbc.NavItem(dbc.NavLink("GitHub", href="https://github.com/your/repo", external_link=True)),
                        dbc.NavItem(dbc.NavLink("Contact", href="mailto:contact@example.com")),
                        dbc.NavItem(dbc.NavLink("Resources", href="https://doi.org/...", external_link=True)),
                    ],
                    className="ms-auto", navbar=True,
                ),
            ], fluid=True
        ),
        color="primary", dark=True, class_name="mb-0",
    )

    # ══ Sticky header (targets + scope/regions) ══════════════════
    header = dbc.Card(
        dbc.CardBody(
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Scope"),
                            dbc.RadioItems(
                                id="scope", value="global", inline=True,
                                options=[{"label": "Global", "value": "global"},
                                         {"label": "Regional", "value": "regional"}],
                            ),
                        ], md=2,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Regions (when Regional)"),
                            dcc.Dropdown(
                                id="regions", multi=True, placeholder="All regions",
                                options=[{"label": r, "value": r} for r in REGIONS],
                                disabled=True,
                            ),
                        ], md=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Targets"),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Checklist(
                                            id="inst", inline=True, value=[430],
                                            options=[{"label": k, "value": v} for k, v in INST_TARGETS.items()],
                                        ), width="auto",
                                    ),
                                    dbc.Col(
                                        dbc.Input(id="custom-target", type="number", min=1,
                                                  placeholder="Custom Taget", style={"width": "16rem"}),
                                        width="auto",
                                    ),
                                ], class_name="g-1 flex-nowrap",
                            ),
                        ], md=5,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Theme"),
                            dcc.Dropdown(
                                id="theme", value=theme,
                                options=[{"label": k, "value": k} for k in THEMES],
                                clearable=False,
                            ),
                        ], md=2,
                    ),
                ], className="g-2",
            )
        ),
        style={"position": "sticky", "top": "0", "zIndex": "1020"},
        class_name="shadow-sm mb-1",
    )

    # ══ Forecast right-hand controls (inc. year range) ═══════════
    fc_ctrl = dbc.Card(
        dbc.CardBody(
            [
                html.Div(id="fc-reg-head"),
                dbc.Label("Year range"),
                dbc.Row(
                    [
                        dbc.Col(dcc.Dropdown(id="fc-start", value=YEARS[0],
                                             options=[{"label": y, "value": y} for y in YEARS],
                                             clearable=False), width=6),
                        dbc.Col(dcc.Dropdown(id="fc-end", value=min(YEARS[5], YEARS[-1]),
                                             options=[{"label": y, "value": y} for y in YEARS],
                                             clearable=False), width=6),
                    ], class_name="g-1 flex-nowrap",
                ),
                html.Hr(className="my-2"),
                dbc.Label("Forecast view"),
                dbc.RadioItems(
                    id="fc-view", value="annual",
                    options=[{"label": l, "value": v} for l, v in
                             [("Annual", "annual"), ("Cumulative", "cum"),
                              ("Map", "map")]],
                ),
                html.Hr(className="my-2"),
                dbc.Label("Scenario (cumulative)"),
                dcc.Dropdown(
                    id="fc-scn", value="BAU",
                    options=[{"label": s, "value": s} for s in SCEN], clearable=False,
                ),
                html.Hr(className="my-2"),
                dbc.Label("Models to display"),
                dbc.Checklist(
                    id="fc-models",
                    options=[
                        {"label": "Transformer", "value": "transformer"},
                        {"label": "Autoformer", "value": "autoformer"},
                        {"label": "Informer", "value": "informer"},
                        {"label": "N-BEATS", "value": "nbeats"},
                        {"label": "TFT", "value": "tft"},
                        {"label": "LogSparse", "value": "logsparsetransformer"},
                        {"label": "NeuralODE", "value": "neuralode"},
                    ],
                    value=["transformer", "autoformer", "informer"],  # Default selection
                    style={"fontSize": "0.85rem"},
                ),
                dbc.Row([
                    dbc.Col(dbc.Button("All", id="select-all-models", size="sm", color="outline-primary"), width=6),
                    dbc.Col(dbc.Button("None", id="select-no-models", size="sm", color="outline-secondary"), width=6),
                ], className="g-1 mt-1"),
                html.Hr(className="my-2"),
                dbc.ButtonGroup(
                    [dbc.Button("CSV", id="fc-csv", size="sm"),
                     dbc.Button("PNG", id="fc-png", size="sm")],
                ),
            ]
        ), class_name="h-100",
    )

    # ══ Strategic right-hand controls ════════════════════════════
    st_ctrl = dbc.Card(
        dbc.CardBody(
            [
                html.Div(id="st-reg-head"),
                dbc.Label("Year range"),
                dbc.Row(
                    [
                        dbc.Col(dcc.Dropdown(id="st-start", value=YEARS[0],
                                             options=[{"label": y, "value": y} for y in YEARS],
                                             clearable=False), width=6),
                        dbc.Col(dcc.Dropdown(id="st-end", value=min(YEARS[5], YEARS[-1]),
                                             options=[{"label": y, "value": y} for y in YEARS],
                                             clearable=False), width=6),
                    ], class_name="g-1 flex-nowrap",
                ),
                html.Hr(className="my-2"),
                dbc.Label("Curve dynamics"),
                dbc.Checklist(
                    id="st-dyn", value=["early"],
                    options=[{"label": "Early investment", "value": "early"},
                             {"label": "Late investment",  "value": "late"}],
                ),
                dbc.Checkbox(id="ci", label="Confidence interval", value=False),
                dbc.Checkbox(id="st-map", label="Show map", value=False),
                html.Hr(className="my-2"),
                dbc.ButtonGroup(
                    [dbc.Button("CSV", id="st-csv", size="sm"),
                     dbc.Button("PNG", id="st-png", size="sm")],
                ),
            ]
        ), class_name="h-100",
    )

    # ══ Tab panes (p-0 to kill column padding) ═══════════════════
    forecast_tab = dbc.Row(
        [
            dbc.Col(
                [
                    html.H4("Forecasting Hydrogen Production", className="mb-1"),
                    html.H6(id="fc-sub", className="text-muted mb-0"),
                    dcc.Graph(id="fc-graph", style={"height": "74vh", "marginTop": "0.25rem"}),
                ], md=10, class_name="p-0",
            ),
            dbc.Col(fc_ctrl, md=2),
        ], class_name="g-2",
    )

    strat_tab = dbc.Row(
        [
            dbc.Col(
                [
                    html.H4("Strategic Plan: Correcting the Errors", className="mb-1"),
                    html.H6(id="st-sub", className="text-muted mb-0"),
                    dcc.Graph(id="st-graph", style={"height": "74vh", "marginTop": "0.25rem"}),
                ], md=10, class_name="p-0",
            ),
            dbc.Col(st_ctrl, md=2),
        ], class_name="g-2",
    )

    # ══ Layout ═══════════════════════════════════════════════════
    app.layout = dbc.Container(
        [
            navbar,
            header,
            dbc.Tabs([dbc.Tab(forecast_tab, label="Forecasting"),
                      dbc.Tab(strat_tab,  label="Strategic Plan")]),
            dcc.Download(id="dl"),
            dcc.Location(id="loc", refresh=True),
        ], fluid=True, class_name="px-4",
    )

    # ─── UI logic ────────────────────────────────────────────────
    @app.callback(Output("regions", "disabled"), Input("scope", "value"))
    def _disable_regions(scope): return scope == "global"
    
    # Disable targets for annual view
    @app.callback(Output("inst", "options"), Input("fc-view", "value"))
    def _toggle_targets(view):
        if view == "annual":
            # Disable all target options for annual view
            return [{"label": k, "value": v, "disabled": True} for k, v in INST_TARGETS.items()]
        else:
            # Enable targets for other views
            return [{"label": k, "value": v} for k, v in INST_TARGETS.items()]

    # Model selection buttons
    @app.callback(
        Output("fc-models", "value"),
        Input("select-all-models", "n_clicks"),
        Input("select-no-models", "n_clicks"),
        State("fc-models", "options"),
        prevent_initial_call=True
    )
    def _toggle_model_selection(all_clicks, none_clicks, options):
        ctx_triggered = ctx.triggered[0]["prop_id"]
        if "select-all-models" in ctx_triggered:
            return [opt["value"] for opt in options]
        elif "select-no-models" in ctx_triggered:
            return []
        return no_update

    def _hdr(scope, regs):
        if scope == "global": return html.H6("Global", className="text-muted fw-semibold")
        return [html.H6(r, className="text-muted fw-semibold") for r in (regs or REGIONS)]

    app.callback(Output("fc-reg-head", "children"),
                 Input("scope", "value"), Input("regions", "value"))(_hdr)
    app.callback(Output("st-reg-head", "children"),
                 Input("scope", "value"), Input("regions", "value"))(_hdr)

    def _subtitle(scope, regs, y0, y1):
        yrs = f"Year: {y0}" if y0 == y1 else f"Years: {y0}–{y1}"
        return yrs + (" • Scope: Global" if scope == "global"
                      else " • Scope: " + ", ".join(regs or REGIONS))

    app.callback(Output("fc-sub", "children"),
                 Input("scope", "value"), Input("regions", "value"),
                 Input("fc-start", "value"), Input("fc-end", "value"))(_subtitle)
    app.callback(Output("st-sub", "children"),
                 Input("scope", "value"), Input("regions", "value"),
                 Input("st-start", "value"), Input("st-end", "value"))(_subtitle)

    # keep end ≥ start
    @app.callback(Output("fc-end", "value"),
                  Input("fc-start", "value"), State("fc-end", "value"))
    def _sync_fc(s, e): return max(e, s)
    @app.callback(Output("st-end", "value"),
                  Input("st-start", "value"), State("st-end", "value"))
    def _sync_st(s, e): return max(e, s)

    # Forecast figure -------------------------------------------------
    @app.callback(
        Output("fc-graph", "figure"),
        Input("scope", "value"), Input("regions", "value"),
        Input("fc-start", "value"), Input("fc-end", "value"),
        Input("fc-view", "value"), Input("fc-scn", "value"),
        Input("inst", "value"), Input("custom-target", "value"),
        Input("fc-models", "value"))
    def forecast(scope, regs, y0, y1, mode, scn, insts, custom, selected_models):
        regs = regs or REGIONS if scope == "regional" else ["Global"]

        # Map view - Global Choropleth
        if mode == "map":
            try:
                # Fetch global choropleth data
                map_data = fetch("/global_choropleth", payload={"year": y0, "scenario": scn})
                
                if map_data and "data" in map_data:
                    df = pd.DataFrame(map_data["data"])
                    
                    # Create global choropleth map
                    fig = px.choropleth(
                        df,
                        locations="iso_alpha",
                        color="production",
                        hover_name="country",
                        hover_data={"region": True, "production": ":.2f", "iso_alpha": False},
                        color_continuous_scale="Viridis",
                        labels={"production": "H₂ Production (Mt)"},
                        title=f"Global Hydrogen Production - {y0} ({scn} Scenario)"
                    )
                    
                    # Configure map layout for global view
                    fig.update_geos(
                        projection_type="equirectangular",
                        showland=True,
                        landcolor="lightgray",
                        showocean=True,
                        oceancolor="lightblue",
                        showlakes=True,
                        lakecolor="lightblue",
                        showcountries=True,
                        countrycolor="white"
                    )
                    
                    fig.update_layout(
                        template="simple_white",
                        margin=dict(t=60, r=20, l=20, b=20),
                        height=650,
                        coloraxis_colorbar=dict(
                            title="H₂ Production<br>(Mt/year)",
                            titleside="right"
                        )
                    )
                    
                    return fig
                else:
                    # Fallback if API fails
                    raise Exception("No map data received")
                    
            except Exception as e:
                print(f"Error fetching choropleth data: {e}")
                # Fallback to simple map
                fig = go.Figure()
                fig.add_annotation(
                    text="Map data unavailable",
                    x=0.5, y=0.5,
                    xref="paper", yref="paper",
                    showarrow=False,
                    font=dict(size=20, color="red")
                )
                fig.update_layout(
                    template="simple_white",
                    height=650,
                    margin=dict(t=20, r=0, l=0, b=0)
                )
                return fig

        # Line views (Annual or Cumulative)
        fig = go.Figure()
        use_cumulative = (mode == "cum")
        
        # Determine which endpoint to call
        endpoint = "/cumulative_forecast" if use_cumulative else "/annual_forecast"

        for r in regs:
            try:
                # Fetch data from appropriate endpoint
                data = fetch(endpoint, payload={"region": r, "start": y0, "end": y1, "scenario": scn})
                models_series = to_series(data, y0, y1, use_cumulative=use_cumulative)
                
                if isinstance(models_series, dict):
                    # Multiple models - plot only selected ones
                    for model_name, ser in models_series.items():
                        if model_name in (selected_models or []) and not ser.empty:
                            line_name = f"{r} ({model_name})" if r != "Global" else model_name
                            fig.add_trace(go.Scatter(x=ser.index, y=ser, name=line_name, mode="lines"))
                else:
                    # Single series (fallback to legacy format)
                    ser = models_series if not models_series.empty else default_ser(y0, y1)
                    fig.add_trace(go.Scatter(x=ser.index, y=ser, name=r, mode="lines"))
                    
            except Exception as e:
                print(f"Error fetching data for {r}: {e}")
                # Fallback to default series
                ser = default_ser(y0, y1)
                fig.add_trace(go.Scatter(x=ser.index, y=ser, name=f"{r} (fallback)", mode="lines"))

        # Target lines (colour-coded dashed) - only for cumulative view
        if use_cumulative and insts:
            for i, v in enumerate(insts):
                col = PALETTE[i % len(PALETTE)]
                fig.add_hline(y=v, line=dict(dash="dash", color=col, width=2),
                              annotation_text=f"{v} Mt", annotation_font_color=col,
                              annotation_position="top right")

        if use_cumulative and custom:
            fig.add_hline(y=float(custom), line=dict(dash="dot", color="red", width=2),
                          annotation_text=f"Custom {custom} Mt",
                          annotation_font_color="red", annotation_position="top right")

        # Update labels based on view
        if use_cumulative:
            y_label = "Cumulative Hydrogen Capacity (Mt)"
            model_text = f"({len(selected_models or [])} models)" if selected_models else "(No models)"
            title_suffix = f" - Cumulative {model_text}"
        else:
            y_label = "Annual Hydrogen Capacity (Mt)"
            model_text = f"({len(selected_models or [])} models)" if selected_models else "(No models)"
            title_suffix = f" - Annual {model_text}"

        fig.update_layout(
            template="simple_white", height=650,
            margin=dict(t=20, r=40, l=60, b=40),
            xaxis=dict(tickmode="linear", dtick=1, tick0=y0, range=[y0, y1]),
            xaxis_title="Year", yaxis_title=y_label,
            title_text=f"Hydrogen Production Forecast{title_suffix}",
        )
        return fig

    # Strategic figure -----------------------------------------------
    @app.callback(
        Output("st-graph", "figure"),
        Input("scope", "value"), Input("regions", "value"),
        Input("st-start", "value"), Input("st-end", "value"),
        Input("st-dyn", "value"), Input("ci", "value"),
        Input("st-map", "value"))
    def strategic(scope, regs, y0, y1, dyn, ci, mapmode):
        regs = regs or REGIONS if scope == "regional" else ["Global"]

        if mapmode:
            df = pd.DataFrame({"region": REGIONS, "value": [None] * len(REGIONS)})
            fig = px.choropleth(df, locations="region", locationmode="geojson-id",
                                color="value", color_continuous_scale="Viridis")
            fig.update_geos(fitbounds="locations", visible=False)
            fig.update_layout(template="simple_white", margin=dict(t=20, r=0, l=0, b=0))
            return fig

        fig = go.Figure()
        for r in regs:
            try:
                recs = fetch("/optimize", method="POST",
                             payload={"region": r, "start": y0, "end": y1, "target": 430})
                models_series = to_series(recs, y0, y1, use_cumulative=True)  # Strategic view uses cumulative
                
                if isinstance(models_series, dict):
                    # Multiple models - use the first available one for strategic view
                    for model_name, ser in models_series.items():
                        if not ser.empty:
                            line_name = f"RL {r} ({model_name})"
                            fig.add_trace(go.Scatter(x=ser.index, y=ser, name=line_name, mode="lines"))
                            
                            if ci:
                                fig.add_trace(go.Scatter(x=ser.index, y=ser * 1.1, line=dict(width=0),
                                                         showlegend=False, hoverinfo="skip"))
                                fig.add_trace(go.Scatter(x=ser.index, y=ser * 0.9, line=dict(width=0),
                                                         showlegend=False, hoverinfo="skip",
                                                         fill="tonexty", fillcolor="rgba(50,150,80,.15)"))
                            break  # Only use first model for strategic view
                else:
                    # Single series (legacy format)
                    ser = models_series if not models_series.empty else default_ser(y0, y1)
                    fig.add_trace(go.Scatter(x=ser.index, y=ser, name=f"RL {r}", mode="lines"))
                    
                    if ci:
                        fig.add_trace(go.Scatter(x=ser.index, y=ser * 1.1, line=dict(width=0),
                                                 showlegend=False, hoverinfo="skip"))
                        fig.add_trace(go.Scatter(x=ser.index, y=ser * 0.9, line=dict(width=0),
                                                 showlegend=False, hoverinfo="skip",
                                                 fill="tonexty", fillcolor="rgba(50,150,80,.15)"))
                        
            except Exception as e:
                print(f"Error in strategic planning for {r}: {e}")
                ser = default_ser(y0, y1)
                fig.add_trace(go.Scatter(x=ser.index, y=ser, name=f"RL {r} (fallback)", mode="lines"))

        if "early" in dyn and "late" not in dyn:
            fig.update_layout(title_text="Early investment focus")
        elif "late" in dyn and "early" not in dyn:
            fig.update_layout(title_text="Late investment focus")

        fig.update_layout(
            template="simple_white", height=650,
            margin=dict(t=20, r=40, l=60, b=40),
            xaxis=dict(tickmode="linear", dtick=1, tick0=y0, range=[y0, y1]),
            xaxis_title="Year", yaxis_title="Hydrogen Capacity (Mt)",
        )
        return fig

    # Theme reload ----------------------------------------------------
    @app.callback(Output("loc", "href"), Input("theme", "value"), prevent_initial_call=True)
    def theme_reload(sel):
        args = flask.request.args.to_dict()
        args["theme"] = sel
        return flask.request.path + "?" + urllib.parse.urlencode(args)

    return app

# ───────── Run ───────────────────────────────────────────────────
init_theme = flask.request.args.get("theme", DEFAULT_THEME) if flask.has_request_context() else DEFAULT_THEME
app = make_app(init_theme)

if __name__ == "__main__":
    app.run(port=8050, debug=True)
