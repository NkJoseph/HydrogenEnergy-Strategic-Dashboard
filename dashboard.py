"""
dashboard.py — Global Hydrogen Production Simulation Tool
Sticky header · Start/End year in side panes · Tight subtitle spacing
"""

from __future__ import annotations
import datetime, functools, json, urllib.parse, base64, os
import pandas as pd, requests, flask
import dash
from dash import dcc, html, Input, Output, State, ctx, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px

# ───────── Config ───────────────────────────────────────────────
# Allow override for Docker / cloud deployment
API_URL = os.getenv("HYDROGEN_API_URL", "http://127.0.0.1:8000")
REGIONS = [
    "Africa", "Canada", "USA", "India", "China", "Europe",
    "East & Southeast Asia", "South Asia", "Central Asia",
    "Middle East", "Latin America", "Australia", "Oceania", "Others",
]
CUR_Y   = datetime.date.today().year
YEARS   = list(range(CUR_Y + 1, 2201))
SCEN    = ["Optimistic", "BAU", "Pessimistic"]
THEMES = {
    "Light": dbc.themes.FLATLY,
    "Dark": dbc.themes.DARKLY,
    "Cosmo": dbc.themes.COSMO,
    "Solar": dbc.themes.SOLAR
}
DEFAULT_THEME = "Light"
DEFAULT_LANGUAGE = "en"

# Translation dictionary
TRANSLATIONS = {
    "en": {
        "title": "Global Hydrogen Simulation Tool",
        "forecast": "Forecasting",
        "strategic": "Strategic Plan",
        "scope": "Scope",
        "global": "Global",
        "regional": "Regional",
        "regions": "Regions",
        "start_year": "Start Year",
        "end_year": "End Year",
        "view": "View",
        "annual": "Annual",
        "cumulative": "Cumulative",
        "map": "Map",
        "scenario": "Scenario",
        "models": "Models",
        "select_all": "Select All",
        "select_none": "Select None",
        "targets": "Targets",
        "custom_target": "Custom Target",
        "confidence": "Confidence Intervals",
        "theme": "Theme",
        "language": "Language",
        "github": "GitHub",
        "early_investment": "Early investment focus",
        "late_investment": "Late investment focus",
        "hydrogen_capacity": "Hydrogen Capacity (Mt)",
        "cumulative_hydrogen": "Cumulative Hydrogen Capacity (Mt)",
        "annual_hydrogen": "Annual Hydrogen Capacity (Mt)",
        "year": "Year",
        "production_forecast": "Hydrogen Production Forecast",
        "strategic_planning": "Strategic Planning",
        "map_data_unavailable": "Map data unavailable",
        "using_models": "Using {count} model{s}",
        "global_cumulative": "Global Cumulative Hydrogen Production",
        "regional_cumulative": "Regional Cumulative Hydrogen Production",
        "scenario_label": "Scenario",
        "period_label": "Period",
        "cumulative_production": "Cumulative H₂ Production (Mt)",
        "production_mt_year": "H₂ Production (Mt/year)",
        "cumulative_production_mt": "Cumulative H₂ Production (Mt)",
        "fallback": "fallback",
        "error_fetching": "Error fetching data for",
        "error_choropleth": "Error fetching choropleth data",
        "no_data": "No data received"
    },
    "fr": {
        "title": "Outil de simulation de production mondiale d'hydrogène",
        "forecast": "Prévision",
        "strategic": "Plan stratégique",
        "scope": "Portée",
        "global": "Mondial",
        "regional": "Régional",
        "regions": "Régions",
        "start_year": "Année de début",
        "end_year": "Année de fin",
        "view": "Vue",
        "annual": "Annuel",
        "cumulative": "Cumulatif",
        "map": "Carte",
        "scenario": "Scénario",
        "models": "Modèles",
        "select_all": "Tout sélectionner",
        "select_none": "Aucune sélection",
        "targets": "Objectifs",
        "custom_target": "Objectif personnalisé",
        "confidence": "Intervalles de confiance",
        "theme": "Thème",
        "language": "Langue",
        "github": "GitHub",
        "early_investment": "Focus sur l'investissement précoce",
        "late_investment": "Focus sur l'investissement tardif",
        "hydrogen_capacity": "Capacité d'hydrogène (Mt)",
        "cumulative_hydrogen": "Capacité d'hydrogène cumulative (Mt)",
        "annual_hydrogen": "Capacité d'hydrogène annuelle (Mt)",
        "year": "Année",
        "production_forecast": "Prévision de production d'hydrogène",
        "strategic_planning": "Planification stratégique",
        "map_data_unavailable": "Données de carte indisponibles",
        "using_models": "Utilisation de {count} modèle{s}",
        "global_cumulative": "Production mondiale cumulative d'hydrogène",
        "regional_cumulative": "Production régionale cumulative d'hydrogène",
        "scenario_label": "Scénario",
        "period_label": "Période",
        "cumulative_production": "Production cumulative H₂ (Mt)",
        "production_mt_year": "Production H₂ (Mt/an)",
        "cumulative_production_mt": "Production cumulative H₂ (Mt)",
        "fallback": "secours",
        "error_fetching": "Erreur lors de la récupération des données pour",
        "error_choropleth": "Erreur lors de la récupération des données choroplèthe",
        "no_data": "Aucune donnée reçue"
    },
    "pt": {
        "title": "Ferramenta de Simulação de Produção Global de Hidrogênio",
        "forecast": "Previsão",
        "strategic": "Plano Estratégico",
        "scope": "Escopo",
        "global": "Global",
        "regional": "Regional",
        "regions": "Regiões",
        "start_year": "Ano Inicial",
        "end_year": "Ano Final",
        "view": "Visualização",
        "annual": "Anual",
        "cumulative": "Cumulativo",
        "map": "Mapa",
        "scenario": "Cenário",
        "models": "Modelos",
        "select_all": "Selecionar Todos",
        "select_none": "Selecionar Nenhum",
        "targets": "Metas",
        "custom_target": "Meta Personalizada",
        "confidence": "Intervalos de Confiança",
        "theme": "Tema",
        "language": "Idioma",
        "github": "GitHub",
        "early_investment": "Foco em investimento precoce",
        "late_investment": "Foco em investimento tardio",
        "hydrogen_capacity": "Capacidade de Hidrogênio (Mt)",
        "cumulative_hydrogen": "Capacidade de Hidrogênio Cumulativa (Mt)",
        "annual_hydrogen": "Capacidade de Hidrogênio Anual (Mt)",
        "year": "Ano",
        "production_forecast": "Previsão de Produção de Hidrogênio",
        "strategic_planning": "Planejamento Estratégico",
        "map_data_unavailable": "Dados do mapa indisponíveis",
        "using_models": "Usando {count} modelo{s}",
        "global_cumulative": "Produção Global Cumulativa de Hidrogênio",
        "regional_cumulative": "Produção Regional Cumulativa de Hidrogênio",
        "scenario_label": "Cenário",
        "period_label": "Período",
        "cumulative_production": "Produção Cumulativa H₂ (Mt)",
        "production_mt_year": "Produção H₂ (Mt/ano)",
        "cumulative_production_mt": "Produção Cumulativa H₂ (Mt)",
        "fallback": "reserva",
        "error_fetching": "Erro ao buscar dados para",
        "error_choropleth": "Erro ao buscar dados do choroplético",
        "no_data": "Nenhum dado recebido"
    },
    "zh": {
        "title": "全球氢能生产模拟工具",
        "forecast": "预测",
        "strategic": "战略规划",
        "scope": "范围",
        "global": "全球",
        "regional": "区域",
        "regions": "地区",
        "start_year": "开始年份",
        "end_year": "结束年份",
        "view": "视图",
        "annual": "年度",
        "cumulative": "累计",
        "map": "地图",
        "scenario": "情景",
        "models": "模型",
        "select_all": "全选",
        "select_none": "全不选",
        "targets": "目标",
        "custom_target": "自定义目标",
        "confidence": "置信区间",
        "theme": "主题",
        "language": "语言",
        "github": "GitHub",
        "early_investment": "早期投资重点",
        "late_investment": "后期投资重点",
        "hydrogen_capacity": "氢能容量 (Mt)",
        "cumulative_hydrogen": "累计氢能容量 (Mt)",
        "annual_hydrogen": "年度氢能容量 (Mt)",
        "year": "年份",
        "production_forecast": "氢能生产预测",
        "strategic_planning": "战略规划",
        "map_data_unavailable": "地图数据不可用",
        "using_models": "使用 {count} 个模型",
        "global_cumulative": "全球累计氢能生产",
        "regional_cumulative": "区域累计氢能生产",
        "scenario_label": "情景",
        "period_label": "时期",
        "cumulative_production": "累计 H₂ 生产 (Mt)",
        "production_mt_year": "H₂ 生产 (Mt/年)",
        "cumulative_production_mt": "累计 H₂ 生产 (Mt)",
        "fallback": "备用",
        "error_fetching": "获取数据时出错",
        "error_choropleth": "获取地图数据时出错",
        "no_data": "未收到数据"
    }
}

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
                timeout=30,
                proxies={'http': None, 'https': None})  # Explicitly disable proxies
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        # One clean warning instead of a traceback storm
        print(f"[WARNING] Back-end offline – {API_URL}{ep} ({e})")
        return {}        # caller will trigger its built-in fallback

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

# ───────── Translation helper ──────────────────────────────────
def t(key: str, lang: str = "en") -> str:
    """Get translation for a key in the specified language"""
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)

def format_translation(key: str, lang: str = "en", **kwargs) -> str:
    """Get translation and format with parameters"""
    translation = t(key, lang)
    return translation.format(**kwargs)

# ───────── Dash factory ─────────────────────────────────────────
def make_app(theme: str = DEFAULT_THEME, language: str = DEFAULT_LANGUAGE) -> dash.Dash:
    app = dash.Dash(
        __name__, 
        external_stylesheets=[
            "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
        ], 
        suppress_callback_exceptions=True,
        title="Global Hydrogen Production Simulation Tool",
        server_url='/',
        routes_pathname_prefix='/'
    )
    server = app.server

    # ══ Navbar ═══════════════════════════════════════════════════
    navbar = dbc.Navbar(
        dbc.Container(
            [
                dbc.NavbarBrand(t("title", language), id="navbar-brand", className="me-auto"),
                dbc.Nav(
                    [
                        dbc.NavItem(dbc.NavLink("GitHub", href="https://github.com/your/repo", external_link=True)),
                        dbc.NavItem(dbc.NavLink("Contact", href="mailto:contact@example.com")),
                        dbc.NavItem(dbc.NavLink("Resources", href="https://doi.org/...", external_link=True)),
                    ],
                    className="ms-auto", navbar=True,
                ),

            ], 
            fluid=True,
            className="position-relative"
        ),
        color="primary", 
        dark=True, 
        class_name="mb-0 sticky-top",
        style={"zIndex": 9998}
    )

    # ══ Sticky header (targets + scope/regions) ══════════════════
    header = dbc.Card(
        dbc.CardBody(
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label(t("scope", language)),
                            dbc.RadioItems(
                                id="scope", value="global", inline=True,
                                options=[{"label": t("global", language), "value": "global"},
                                         {"label": t("regional", language), "value": "regional"}],
                            ),
                        ], md=2,
                    ),
                    dbc.Col(
                        [
                            dbc.Label(t("regions", language)),
                            dcc.Dropdown(
                                id="regions", multi=True, placeholder=t("regions", language),
                                options=[{"label": r, "value": r} for r in REGIONS],
                                disabled=True,
                            ),
                        ], md=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Label(t("targets", language)),
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
                                                  placeholder=t("custom_target", language), style={"width": "16rem"}),
                                        width="auto",
                                    ),
                                ], class_name="g-1 flex-nowrap",
                            ),
                        ], md=5,
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
                dbc.Label(f"{t('start_year', language)} - {t('end_year', language)}"),
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
                dbc.Label(t("view", language)),
                dbc.RadioItems(
                    id="fc-view", value="annual",
                    options=[{"label": l, "value": v} for l, v in
                             [(t("annual", language), "annual"), (t("cumulative", language), "cum"),
                              (t("map", language), "map")]],
                ),
                html.Hr(className="my-2"),
                dbc.Label(f"{t('scenario', language)} (cumulative)"),
                dcc.Dropdown(
                    id="fc-scn", value="BAU",
                    options=[{"label": s, "value": s} for s in SCEN], clearable=False,
                ),
                html.Hr(className="my-2"),
                dbc.Label(t("models", language)),
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
                    dbc.Col(dbc.Button(t("select_all", language), id="select-all-models", size="sm", color="outline-primary"), width=6),
                    dbc.Col(dbc.Button(t("select_none", language), id="select-no-models", size="sm", color="outline-secondary"), width=6),
                ], className="g-1 mt-1"),
                html.Hr(className="my-2"),
                dbc.ButtonGroup(
                    [dbc.Button("CSV", id="fc-csv", size="sm"),
                     dbc.Button("PNG", id="fc-png", size="sm")],
                ),
                html.Hr(className="my-2"),
                # Settings card
                dbc.Card(
                    [
                        dbc.CardHeader("Settings"),
                        dbc.CardBody(
                            dbc.Stack(
                                [
                                    html.Div("Theme:", className="tiny-label"),
                                    dcc.Dropdown(
                                        id="theme-dropdown",
                                        value=theme,
                                        options=[{"label": k, "value": k} for k in THEMES],
                                        clearable=False,
                                        style={"minWidth": "100px"}
                                    ),
                                    html.Hr(className="my-1"),
                                    html.Div("Language:", className="tiny-label"),
                                    dcc.Dropdown(
                                        id="language-dropdown",
                                        value=language,
                                        options=[
                                            {"label": "🇺🇸 English", "value": "en"},
                                            {"label": "🇫🇷 Français", "value": "fr"},
                                            {"label": "🇵🇹 Português", "value": "pt"},
                                            {"label": "🇨🇳 中文", "value": "zh"}
                                        ],
                                        clearable=False,
                                        style={"minWidth": "100px"}
                                    ),
                                ],
                                gap=2,
                            ),
                        ),
                    ],
                    className="mt-2",
                ),
            ]
        ), class_name="h-100",
    )

    # ══ Strategic right-hand controls ════════════════════════════
    st_ctrl = dbc.Card(
        dbc.CardBody(
            [
                html.Div(id="st-reg-head"),
                dbc.Label(f"{t('start_year', language)} - {t('end_year', language)}"),
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
                    options=[{"label": t("early_investment", language), "value": "early"},
                             {"label": t("late_investment", language),  "value": "late"}],
                ),
                dbc.Checkbox(id="ci", label=t("confidence", language), value=False),
                dbc.Checkbox(id="st-map", label=f"Show {t('map', language)}", value=False),
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
                    dcc.Graph(id="fc-graph", style={"height": "600px", "marginTop": "0.25rem"}),
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
                    dcc.Graph(id="st-graph", style={"height": "600px", "marginTop": "0.25rem"}),
                ], md=10, class_name="p-0",
            ),
            dbc.Col(st_ctrl, md=2),
        ], class_name="g-2",
    )

    # ══ Layout ═══════════════════════════════════════════════════
    app.layout = dbc.Container(
        [
            # Theme and language stores
            dcc.Store(id='theme-store', data=theme),
            dcc.Store(id='language-store', data=language),
            html.Link(id='theme-link', rel='stylesheet', href=THEMES[theme]),
            
            navbar,
            header,
            dbc.Tabs([dbc.Tab(forecast_tab, id="forecast-tab", label=t("forecast", language)),
                      dbc.Tab(strat_tab, id="strategic-tab", label=t("strategic", language), disabled=True)]),
            dcc.Download(id="dl"),
            dcc.Location(id="loc", refresh=True),
            
                  ], fluid=True, class_name="px-4"
    )
    
    # Add custom CSS for footer styling using external stylesheet
    app.layout = html.Div([
        html.Link(
            rel='stylesheet',
                          href='data:text/css;base64,' + base64.b64encode("""
                /* tiny labels + dropdowns inside the settings card */
                .tiny-label                { font-size: 0.75rem; font-weight: 500; }
                .card-body .Select-control,
                .card-body .dropdown-toggle,
                .card-body .form-select    { font-size: 0.75rem !important; min-height: 26px; }
              """.encode()).decode()
        ),
        app.layout
    ])

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
                # Prepare payload with selected models, regions, and YEAR RANGE
                payload = {
                    "start": y0,  # Start year from range
                    "end": y1,    # End year from range
                    "scenario": scn,
                    "models": selected_models
                }
                
                if scope == "regional" and regs:
                    payload["regions"] = regs
                    
                map_data = fetch("/global_choropleth", payload=payload)
                
                if map_data and "data" in map_data:
                    df = pd.DataFrame(map_data["data"])
                    
                    # Create model info string
                    model_count = len(map_data.get("models", []))
                    model_info = format_translation("using_models", language, count=model_count, s="s" if model_count != 1 else "")
                    
                    # Create title based on scope and YEAR RANGE
                    period = f"{y0}-{y1}" if y0 != y1 else str(y0)
                    if scope == "regional" and regs:
                        title = f"{', '.join(regs)} | Cumulative Hydrogen Production - {period} ({scn} Scenario)"
                    else:
                        title = f"Global Cumulative Hydrogen Production - {period} ({scn} Scenario)"
                    
                    # Create choropleth map
                    fig = px.choropleth(
                        df,
                        locations="iso_alpha",
                        color="production",
                        hover_name="country",
                        hover_data={
                            "region": True, 
                            "production": ":.2f", 
                            "iso_alpha": False,
                            "start_year": False,
                            "end_year": False
                        },
                        color_continuous_scale="Viridis",
                        labels={"production": "Cumulative H₂ Production (Mt)"},
                        title=f"{title}<br><sup>{model_info}</sup>"
                    )
                    
                    # Configure map layout
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
                        margin=dict(t=100, r=20, l=20, b=20),
                        height=650,
                        coloraxis_colorbar=dict(
                            title="Cumulative H₂<br>Production (Mt)"
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
                    text=t("map_data_unavailable", language),
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

        # ---- 5 % shrink for Annual / Cumulative ----
        if mode in ("annual", "cum"):
            default_h = 650  # fallback if height unset
            fig.update_layout(
                template="simple_white", height=int(default_h * 0.95),
                margin=dict(t=80, r=40, l=60, b=40),  # 80 px top margin leaves room
                xaxis=dict(tickmode="linear", dtick=1, tick0=y0, range=[y0, y1]),
                xaxis_title="Year", yaxis_title=y_label,
                title_text=f"Hydrogen Production Forecast{title_suffix}",
                title=dict(y=0.93, yanchor="top")  # push title down a hair
            )
        else:
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
    # Theme and language callbacks
    @app.callback(
        Output('theme-link', 'href'),
        Input('theme-store', 'data')
    )
    def update_theme_href(theme):
        return THEMES[theme]

    @app.callback(
        Output('theme-store', 'data'),
        Input('theme-dropdown', 'value'),
        prevent_initial_call=True
    )
    def update_theme_store(new_theme):
        return new_theme

    @app.callback(
        Output('language-store', 'data'),
        Input('language-dropdown', 'value'),
        prevent_initial_call=True
    )
    def update_language_store(new_lang):
        return new_lang

    @app.callback(
        Output("loc", "href"), 
        Input("theme-dropdown", "value"), 
        Input("language-dropdown", "value"),
        prevent_initial_call=True
    )
    def update_url_params(theme, language):
        args = flask.request.args.to_dict()
        if theme:
            args["theme"] = theme
        if language:
            args["language"] = language
        return flask.request.path + "?" + urllib.parse.urlencode(args)

    # Dynamic UI updates based on language
    @app.callback(
        Output('navbar-brand', 'children'),
        Input('language-store', 'data')
    )
    def update_navbar_brand(lang):
        return t("title", lang)

    @app.callback(
        Output('forecast-tab', 'label'),
        Input('language-store', 'data')
    )
    def update_forecast_tab_label(lang):
        return t("forecast", lang)

    @app.callback(
        Output('strategic-tab', 'label'),
        Input('language-store', 'data')
    )
    def update_strategic_tab_label(lang):
        return t("strategic", lang)

    return app

# ───────── Run ───────────────────────────────────────────────────
init_theme = flask.request.args.get("theme", DEFAULT_THEME) if flask.has_request_context() else DEFAULT_THEME
init_language = flask.request.args.get("language", DEFAULT_LANGUAGE) if flask.has_request_context() else DEFAULT_LANGUAGE
app = make_app(init_theme, init_language)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8050))
    debug = os.environ.get("DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
