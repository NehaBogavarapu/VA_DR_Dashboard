"""
Dog / Cat / Panda Visual Analytics Prototype
==============================================
AMV10 Visual Analytics — Group 14
- Top row: Class distribution + accuracy (stacked bar) | Train vs Val | Confusion matrix
- Review queue: images ranked by uncertainty
- Middle: UMAP scatter with 3 colour modes
- Right slide-out panel: Image + LIME + Annotation
"""

import dash
from dash import dcc, html, Input, Output, State, callback, no_update, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import json
import base64
from io import BytesIO
from PIL import Image
import heapq

from data_pipeline_DCP import (
    load_data, run_kmeans,
    get_misclassified, filter_by_confidence,
    CLASS_NAMES, CLASS_DISPLAY,
)
from lime_explainer_DCP import generate_lime_explanation
from annotation_store_DCP import AnnotationStore
from retrain_DCP import retrain_with_annotations

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
app.title = "DCP-VA: Dog / Cat / Panda Visual Analytics"

# ── Colour scheme ──────────────────────────────────────────────────────────
CLASS_COLORS = {0: "#e74c3c", 1: "#3498db", 2: "#2ecc71"}  # Cat=red, Dog=blue, Panda=green
MISCLASS_COLORS = {"Correct": "#2ecc71", "Misclassified": "#e74c3c"}
BRUSH_COLORS = {
    "important": {"fill": "rgba(255,0,0,0.3)", "line": "red"},
    "artefact": {"fill": "rgba(0,100,255,0.3)", "line": "blue"},
    "background": {"fill": "rgba(0,200,0,0.3)", "line": "green"},
}
annotation_store = AnnotationStore()
PANEL_HIDDEN = {"width": "0px", "minWidth": "0px", "overflow": "hidden", "transition": "width 0.3s ease", "padding": "0", "flexShrink": "0"}
PANEL_VISIBLE = {"width": "400px", "minWidth": "400px", "overflowY": "auto", "maxHeight": "calc(100vh - 80px)", "paddingLeft": "12px", "flexShrink": "0", "transition": "width 0.3s ease"}


def compute_uncertainty(df):
    df = df.copy()

    def calc(conf_str):
        probs = json.loads(conf_str)
        if len(probs) < 2:
            return 0.0
        top2 = heapq.nlargest(2, probs)
        return round(1.0 - (top2[0] - top2[1]), 4)

    df["uncertainty"] = df["class_confidences"].apply(calc)
    return df


def make_legend_for_mode(mode):
    if mode in ("true_class", "pred_class"):
        return [html.Span([
            html.Span(style={"display": "inline-block", "width": "12px", "height": "12px", "borderRadius": "50%", "backgroundColor": CLASS_COLORS[c], "marginRight": "4px", "verticalAlign": "middle"}),
            html.Span(f"{CLASS_DISPLAY[c]}", style={"fontSize": "12px"})
        ], style={"marginRight": "14px"}) for c in range(3)]
    return [html.Span([
        html.Span(style={"display": "inline-block", "width": "12px", "height": "12px", "borderRadius": "50%", "backgroundColor": c, "marginRight": "4px", "verticalAlign": "middle"}),
        html.Span(l, style={"fontSize": "12px"})
    ], style={"marginRight": "14px"}) for l, c in MISCLASS_COLORS.items()]


# ═══════════════════════════════════════════════════════════════════════════
#  LAYOUT
# ═══════════════════════════════════════════════════════════════════════════

header = dbc.Navbar(dbc.Container([
    dbc.NavbarBrand("DCP‑VA  Dog / Cat / Panda Visual Analytics", style={"fontWeight": "600"}),
    html.Span("ResNet‑50 · UMAP + LIME", style={"color": "rgba(255,255,255,0.7)", "fontSize": "13px"})
], fluid=True), color="dark", dark=True, sticky="top")

sidebar = dbc.Card(dbc.CardBody([
    html.H6("Filters", className="mb-3", style={"fontWeight": "600"}),
    html.Label("Show classes", style={"fontSize": "13px"}),
    dcc.Checklist(id="class-filter", options=[{"label": f" {CLASS_DISPLAY[c]}", "value": c} for c in range(3)], value=[0, 1, 2], inline=False, style={"fontSize": "13px", "marginBottom": "12px"}),
    html.Label("Confidence range", style={"fontSize": "13px"}),
    dcc.RangeSlider(id="confidence-slider", min=0, max=1, step=0.05, value=[0.0, 1.0], marks={0: "0", 0.5: "0.5", 1: "1"}, tooltip={"placement": "bottom"}),
    html.Hr(),
    html.H6("Scatter colour", className="mb-2", style={"fontWeight": "600"}),
    dcc.RadioItems(id="color-mode", options=[
        {"label": " True class", "value": "true_class"},
        {"label": " Predicted class", "value": "pred_class"},
        {"label": " Misclassification", "value": "misclassification"},
    ], value="true_class", style={"fontSize": "13px", "marginBottom": "12px"}, inputStyle={"marginRight": "6px"}),
    html.Hr(),
    html.H6("Search image", className="mb-2", style={"fontWeight": "600"}),
    dbc.InputGroup([
        dbc.Input(id="image-search", placeholder="e.g. cat_001", type="text", size="sm", style={"fontSize": "12px"}),
        dbc.Button("Go", id="search-btn", color="primary", size="sm", style={"fontSize": "12px"})
    ], size="sm", className="mb-1"),
    html.Div(id="search-status", style={"fontSize": "11px", "color": "#888"}),
    html.Hr(),
    dbc.Button("Retrain model with annotations", id="retrain-btn", color="success", className="w-100", disabled=True),
    html.Div(id="retrain-status", style={"fontSize": "12px", "marginTop": "6px"}),
]), style={"height": "100%"})

overview_row = dbc.Row([

    dbc.Col(dbc.Card(dbc.CardBody([
        html.H6("Train vs validation accuracy", style={"fontWeight": "600", "fontSize": "14px"}),
        html.P("Per-class accuracy on train (80%) vs validation (20%)", style={"fontSize": "12px", "color": "#888", "marginBottom": "6px"}),
        dcc.Graph(id="train-test-bar", config={"displayModeBar": False}, style={"height": "280px"})
    ])), width=4),
    dbc.Col(dbc.Card(dbc.CardBody([
        html.H6("Confusion matrix", style={"fontWeight": "600", "fontSize": "14px"}),
        html.P("True class (rows) vs predicted class (columns)", style={"fontSize": "12px", "color": "#888", "marginBottom": "6px"}),
        dcc.Graph(id="confusion-matrix", config={"displayModeBar": False}, style={"height": "280px"})
    ])), width=4),
    dbc.Col(dbc.Card(dbc.CardBody([
        html.H6("Review queue", style={"fontWeight": "600", "fontSize": "14px"}),
        html.P("Images ranked in order of descending uncertainty. Click to inspect.", style={"fontSize": "12px", "color": "#888", "marginBottom": "6px"}),
        html.Div(id="ranking-table", style={"maxHeight": "280px", "overflowY": "auto", "fontSize": "12px"})
    ])), width=4),
], className="mb-3")

scatter_view = dbc.Card(dbc.CardBody([
    html.H5("Model hidden layer projection", style={"fontWeight": "600"}),
    html.P("Each dot is how the network sees an image (last hidden layer → UMAP). Click to inspect.", style={"fontSize": "13px", "color": "#666"}),
    html.Div(id="scatter-legend", style={"marginBottom": "8px"}),
    dcc.Loading(dcc.Graph(id="umap-scatter", config={"displayModeBar": True, "scrollZoom": True}, style={"height": "500px"}), type="circle"),
    html.Div(id="cluster-summary", style={"fontSize": "13px", "marginTop": "8px"})
]))

side_panel = html.Div(id="side-panel", style=PANEL_HIDDEN, children=[
    html.Div([
        html.Button("✕", id="close-panel-btn", style={"float": "right", "border": "none", "background": "none", "fontSize": "18px", "cursor": "pointer"}),
        html.H6("Image Inspector", style={"fontWeight": "600"}),
    ]),
    html.Hr(),
    html.Label("LIME overlay opacity", style={"fontSize": "12px"}),
    dcc.Slider(id="lime-opacity", min=0, max=1, step=0.1, value=0.6, marks={0: "0", 0.5: "0.5", 1: "1"}),
    html.Div(id="lime-legend", children=[
        html.Span("LIME: ", style={"fontWeight": "600", "fontSize": "11px"}),
        html.Span("■", style={"color": "#00FFEE", "fontSize": "14px"}), html.Span(" supports ", style={"fontSize": "11px"}),
        html.Span("■", style={"color": "#FFD600", "fontSize": "14px"}), html.Span(" opposes", style={"fontSize": "11px"}),
    ], style={"marginBottom": "6px"}),
    html.Div(id="image-container", style={"height": "280px", "position": "relative", "marginBottom": "8px", "backgroundColor": "#111", "borderRadius": "6px", "overflow": "hidden"}),
    html.Div(id="image-metadata"),
    html.H6("Per-class confidence", style={"fontWeight": "600", "fontSize": "13px"}),
    dcc.Graph(id="confidence-bars", config={"displayModeBar": False}, style={"height": "120px"}),
    html.Hr(),
    html.H6("Annotate", style={"fontWeight": "600", "fontSize": "13px"}),
    dcc.Graph(id="annotation-canvas", config={"modeBarButtonsToAdd": ["drawclosedpath", "drawrect", "drawcircle", "eraseshape"]}, style={"height": "280px"}),
    dbc.Row([
        dbc.Col([html.Label("Brush colour", style={"fontSize": "12px"}),
                 dcc.Dropdown(id="brush-color", options=[
                     {"label": "Important (red)", "value": "important"},
                     {"label": "Artefact (blue)", "value": "artefact"},
                     {"label": "Background (green)", "value": "background"},
                 ], value="important", clearable=False, style={"fontSize": "12px"})], width=6),
        dbc.Col([html.Label("Correct class", style={"fontSize": "12px"}),
                 dcc.Dropdown(id="correct-class", options=[{"label": CLASS_DISPLAY[c], "value": c} for c in range(3)],
                              placeholder="Select…", clearable=False, style={"fontSize": "12px"})], width=6)
    ], className="mt-2"),
    dbc.Button("Save annotation", id="save-annotation-btn", color="primary", size="sm", className="w-100 mt-2"),
    html.Div(id="save-status", style={"fontSize": "11px", "marginTop": "4px"}),
    html.Hr(),
    html.H6("Annotation log", style={"fontWeight": "600", "fontSize": "13px"}),
    html.Div(id="annotation-log", style={"maxHeight": "150px", "overflowY": "auto"}),
])

app.layout = html.Div([
    header,
    dbc.Container([
        dbc.Row([
            dbc.Col(sidebar, width=2, style={"paddingRight": "8px"}),
            dbc.Col([
                overview_row, #ranking_section,
                html.Div([
                    html.Div(scatter_view, id="scatter-wrapper", style={"flex": "1", "minWidth": "0"}),
                    side_panel
                ], style={"display": "flex", "gap": "0"})
            ], width=10, style={"paddingLeft": "8px"})
        ], className="mt-3")
    ], fluid=True),
    dcc.Store(id="selected-image-id", data=None),
    dcc.Store(id="current-embeddings", data=None),
    dcc.Store(id="panel-open", data=False),
])


# ═══════════════════════════════════════════════════════════════════════════
#  CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════

@callback(
    # Output("class-dist-bar", "figure"), 
    Output("train-test-bar", "figure"),
    Output("confusion-matrix", "figure"), Output("ranking-table", "children"),
    Input("class-filter", "value"), Input("confidence-slider", "value"))
def update_overview(classes, conf_range):
    df = load_data()
    df = df[df["true_class"].isin(classes)]
    df = filter_by_confidence(df, conf_range[0], conf_range[1])
    if len(df) == 0:
        e = go.Figure()
        e.add_annotation(text="No data", showarrow=False)
        return e, e, "No data."

    total = len(df)

    # Chart 1: Stacked bar (correct vs misclassified per class)
    dist_fig = go.Figure()
    ac = sorted(df["true_class"].unique())
    labels = [CLASS_DISPLAY[c] for c in ac]
    correct_counts = []
    misclass_counts = []
    correct_pcts = []
    misclass_pcts = []
    for c in ac:
        s = df[df["true_class"] == c]
        cnt = len(s)
        cor = (s["pred_class"] == c).sum()
        mis = cnt - cor
        correct_counts.append(cor)
        misclass_counts.append(mis)
        correct_pcts.append(cor / cnt * 100 if cnt else 0)
        misclass_pcts.append(mis / cnt * 100 if cnt else 0)
    dist_fig.add_trace(go.Bar(x=labels, y=correct_counts, name="Correct", marker_color="#2ecc71",
                               text=[f"{p:.0f}%" for p in correct_pcts], textposition="inside", textfont=dict(size=10, color="white")))
    dist_fig.add_trace(go.Bar(x=labels, y=misclass_counts, name="Misclassified", marker_color="#e74c3c",
                               text=[f"{p:.0f}%" for p in misclass_pcts], textposition="inside", textfont=dict(size=10, color="white")))
    oa = (df["pred_class"] == df["true_class"]).sum() / total * 100
    dist_fig.update_layout(template="plotly_white", barmode="stack", margin=dict(l=40, r=10, t=25, b=40),
                            yaxis=dict(title="Count"), height=280, legend=dict(orientation="h", yanchor="bottom", y=-0.3, x=0.2),
                            annotations=[dict(text=f"Overall accuracy: {oa:.1f}%", xref="paper", yref="paper", x=0.5, y=1.05, showarrow=False, font=dict(size=11, color="#555"))])

    # Chart 2: Train vs Val accuracy
    np.random.seed(42)
    vm = np.random.rand(len(df)) < 0.2
    tdf = df[~vm]
    vdf = df[vm]
    tt_fig = go.Figure()
    ta = [(tdf[tdf["true_class"] == c]["pred_class"] == c).sum() / max(len(tdf[tdf["true_class"] == c]), 1) * 100 for c in ac]
    va = [(vdf[vdf["true_class"] == c]["pred_class"] == c).sum() / max(len(vdf[vdf["true_class"] == c]), 1) * 100 for c in ac]
    tt_fig.add_trace(go.Bar(name="Train", x=labels, y=ta, marker_color="#3498db", text=[f"{a:.0f}%" for a in ta], textposition="inside", textfont=dict(size=10, color="white")))
    tt_fig.add_trace(go.Bar(name="Val", x=labels, y=va, marker_color="#e67e22", text=[f"{a:.0f}%" for a in va], textposition="inside", textfont=dict(size=10, color="white")))
    ot = (tdf["pred_class"] == tdf["true_class"]).mean() * 100
    ov = (vdf["pred_class"] == vdf["true_class"]).mean() * 100
    tt_fig.update_layout(template="plotly_white", barmode="group", margin=dict(l=40, r=10, t=20, b=40),
                          yaxis=dict(title="Accuracy %", range=[0, 100]),
                          legend=dict(orientation="h", yanchor="bottom", y=-0.3, x=0.2), height=280,
                          annotations=[dict(text=f"Overall: Train {ot:.1f}% | Val {ov:.1f}%", xref="paper", yref="paper", x=0.5, y=1.02, showarrow=False, font=dict(size=11, color="#555"))])

    # Chart 3: Confusion matrix
    n = len(ac)
    cm = np.zeros((n, n), dtype=int)
    for i, tc in enumerate(ac):
        for j, pc in enumerate(ac):
            cm[i, j] = ((df["true_class"] == tc) & (df["pred_class"] == pc)).sum()
    cmf = go.Figure(go.Heatmap(
        z=cm, x=[CLASS_DISPLAY[c] for c in ac], y=[CLASS_DISPLAY[c] for c in ac],
        text=[[str(cm[i, j]) for j in range(n)] for i in range(n)],
        texttemplate="%{text}", textfont=dict(size=12),
        colorscale=[[0, "#f8f9fa"], [0.3, "#f5c4b3"], [0.6, "#e67e22"], [1, "#c0392b"]],
        showscale=False))
    cmf.update_layout(template="plotly_white", margin=dict(l=40, r=10, t=10, b=40),
                       xaxis=dict(title="Predicted"), yaxis=dict(title="True", autorange="reversed"), height=280)

    # Ranking table
    dr = compute_uncertainty(df).sort_values("uncertainty", ascending=False).head(20)

    rows = []
    for _, r in dr.iterrows():
        full_id = r["image_id"]
        short_id = full_id[:15] + "…" if len(full_id) > 15 else full_id

        row = html.Tr(
            [
                html.Td(short_id, style={"fontFamily": "monospace"}),
                html.Td(f"{r['uncertainty']:.2f}"),
                html.Td(CLASS_DISPLAY[int(r["true_class"])]),
                html.Td(CLASS_DISPLAY[int(r["pred_class"])]),
                html.Td(
                    dbc.Badge(
                        "✗" if r["pred_class"] != r["true_class"] else "✓",
                        color="danger" if r["pred_class"] != r["true_class"] else "success",
                        style={"fontSize": "10px"}
                    )
                )
            ],
            id={"type": "queue-item", "index": full_id},   # ← entire row is clickable
            n_clicks=0,
            style={
                "cursor": "pointer",
                "backgroundColor": "#fff5f5" if r["pred_class"] != r["true_class"] else "",
            }
        )

        rows.append(row)


    tbl = html.Table([
        html.Thead(html.Tr([
            html.Th("Image ID"),
            html.Th("Uncertainty"),
            html.Th("True"),
            html.Th("Pred"),
            html.Th("")
        ])),
        html.Tbody(rows)
    ], style={"width": "100%"}, className="table table-sm table-hover")


    return tt_fig, cmf, tbl

@callback(
    Output("umap-scatter", "figure"),
    Output("cluster-summary", "children"),
    Output("current-embeddings", "data"),
    Output("scatter-legend", "children"),
    Input("class-filter", "value"),
    Input("confidence-slider", "value"),
    Input("color-mode", "value")
)
def update_scatter(classes, cr, cm):
    df = load_data()

    df = df[df["true_class"].isin(classes)]
    df = filter_by_confidence(df, cr[0], cr[1])
    if len(df) < 2:
        e = go.Figure()
        e.add_annotation(text="Not enough data", showarrow=False)
        return e, "", None, []

    df = df.copy()  # UMAP already in df
    df["mis"] = df["pred_class"] != df["true_class"]

    fig = go.Figure()
    ht = "<b>Image:</b> %{customdata[0]}<br><b>Pred:</b> %{customdata[1]}<br><b>True:</b> %{customdata[2]}<br><b>Conf:</b> %{customdata[3]:.2f}<extra></extra>"
    cd = ["image_id", "pred_class", "true_class", "confidence"]

    if cm == "true_class":
        for c in sorted(df["true_class"].unique()):
            s = df[df["true_class"] == c]
            fig.add_trace(go.Scatter(
                x=s["u1"], y=s["u2"], mode="markers",
                marker=dict(size=8, color=CLASS_COLORS[c], line=dict(width=0.5, color="white")),
                name=CLASS_DISPLAY[c], customdata=s[cd].values, hovertemplate=ht
            ))

    elif cm == "pred_class":
        for c in sorted(df["pred_class"].unique()):
            s = df[df["pred_class"] == c]
            fig.add_trace(go.Scatter(
                x=s["u1"], y=s["u2"], mode="markers",
                marker=dict(size=8, color=CLASS_COLORS[c], line=dict(width=0.5, color="white")),
                name=f"Pred {CLASS_DISPLAY[c]}", customdata=s[cd].values, hovertemplate=ht
            ))

    else:  # misclassification
        for mis, label, color in [(False, "Correct", "#2ecc71"), (True, "Misclassified", "#e74c3c")]:
            s = df[df["mis"] == mis]
            fig.add_trace(go.Scatter(
                x=s["u1"], y=s["u2"], mode="markers",
                marker=dict(size=8, color=color, line=dict(width=0.5, color="white")),
                name=label, customdata=s[cd].values, hovertemplate=ht
            ))

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(title="UMAP 1", showgrid=False),
        yaxis=dict(title="UMAP 2", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
        clickmode="event+select"
    )

    t = len(df)
    m = df["mis"].sum()

    return fig, f"{t} images · {m} misclassified ({m/t:.0%})", df[["image_id", "u1", "u2"]].to_json(), make_legend_for_mode(cm)


@callback(
    Output("side-panel", "style"), Output("panel-open", "data"), Output("selected-image-id", "data"),
    Output("image-container", "children"), Output("image-metadata", "children"),
    Output("confidence-bars", "figure"), Output("annotation-canvas", "figure"), Output("search-status", "children"),
    Input("umap-scatter", "clickData"), Input("close-panel-btn", "n_clicks"), Input("search-btn", "n_clicks"), Input({"type": "queue-item", "index": dash.ALL}, "n_clicks"),
    State("image-search", "value"), State("lime-opacity", "value"), State("panel-open", "data"), State("brush-color", "value"))
def handle_panel(cd, cc, sc, qc, sv, lo, po, bc):
    t = ctx.triggered_id
    n = (no_update,) * 8
    if t == "close-panel-btn":
        return PANEL_HIDDEN, False, None, no_update, no_update, no_update, no_update, ""
    if t == "search-btn":
        if not sv or not sv.strip():
            return *n[:7], "Enter an image ID."
        q = sv.strip()
        df = load_data()
        m = df[df["image_id"] == q]
        if len(m) == 0:
            m = df[df["image_id"].str.contains(q, case=False, na=False)]
        if len(m) == 0:
            return *n[:7], f"Not found: '{q}'"
        row = m.iloc[0]
        iid = row["image_id"]
        r = _build_panel(iid, row, lo, bc)
        return PANEL_VISIBLE, True, iid, *r, f"Found: {iid}"
    
    if isinstance(t, dict) and t.get("type") == "queue-item":
        iid = t["index"]
        df = load_data()
        row = df[df["image_id"] == iid].iloc[0]
        r = _build_panel(iid, row, lo, bc)
        return PANEL_VISIBLE, True, iid, *r, f"Selected: {iid}"

    if cd is None:
        return *n,
    p = cd["points"][0]
    iid = p["customdata"][0]
    df = load_data()
    row = df[df["image_id"] == iid].iloc[0]
    r = _build_panel(iid, row, lo, bc)
    return PANEL_VISIBLE, True, iid, *r, ""


def _build_panel(image_id, row, lime_opacity, brush_color):
    pc = int(row["pred_class"])
    tc = int(row["true_class"])
    cc = json.loads(row["class_confidences"])

    lime_img, _ = generate_lime_explanation(image_id, true_class=tc)
    ob = _pil_to_b64(row["image_path"])
    lb = _arr_to_b64(lime_img)

    img = html.Div([
        html.Img(src=f"data:image/png;base64,{ob}", style={"width": "100%", "height": "100%", "objectFit": "contain", "position": "absolute", "top": 0, "left": 0}),
        html.Img(src=f"data:image/png;base64,{lb}", id="lime-overlay-img", style={"width": "100%", "height": "100%", "objectFit": "contain", "position": "absolute", "top": 0, "left": 0, "opacity": lime_opacity}),
    ], style={"position": "relative", "height": "100%"})

    probs = sorted(cc, reverse=True)
    unc = 1.0 - (probs[0] - probs[1]) if len(probs) >= 2 else 0

    meta = html.Div([
        html.H6(f"Image: {image_id}", style={"fontWeight": "600", "fontSize": "14px"}),
        html.P([html.Span("Predicted: ", style={"fontWeight": "500"}),
                html.Span(CLASS_DISPLAY[pc], style={"color": CLASS_COLORS[pc], "fontWeight": "600"})],
               style={"marginBottom": "2px", "fontSize": "13px"}),
        html.P([html.Span("True: ", style={"fontWeight": "500"}),
                html.Span(CLASS_DISPLAY[tc], style={"color": CLASS_COLORS[tc], "fontWeight": "600"})],
               style={"marginBottom": "2px", "fontSize": "13px"}),
        html.P([dbc.Badge("Correct" if pc == tc else "Misclassified",
                           color="success" if pc == tc else "danger", style={"fontSize": "11px"}),
                html.Span(f"  Uncertainty: {unc:.2f}", style={"fontSize": "12px", "color": "#888", "marginLeft": "8px"})])
    ])

    cf = go.Figure(go.Bar(x=cc, y=[CLASS_DISPLAY[i] for i in range(3)], orientation="h",
                           marker_color=[CLASS_COLORS[i] for i in range(3)],
                           text=[f"{c:.0%}" for c in cc], textposition="auto", textfont=dict(size=11)))
    cf.update_layout(template="plotly_white", margin=dict(l=50, r=10, t=5, b=5),
                      xaxis=dict(range=[0, 1], title="", showticklabels=False), yaxis=dict(title=""), height=120)

    b = BRUSH_COLORS.get(brush_color, BRUSH_COLORS["important"])
    af = _make_annot_fig(row["image_path"], b["fill"], b["line"])

    return img, meta, cf, af


@callback(Output("annotation-canvas", "figure", allow_duplicate=True), Input("brush-color", "value"), State("annotation-canvas", "figure"), prevent_initial_call=True)
def update_brush(bc, fig):
    if fig is None:
        return no_update
    c = BRUSH_COLORS.get(bc, BRUSH_COLORS["important"])
    fig["layout"]["newshape"] = {"fillcolor": c["fill"], "line": {"color": c["line"], "width": 2}}
    return fig


@callback(Output("lime-overlay-img", "style"), Input("lime-opacity", "value"), prevent_initial_call=True)
def update_opacity(o):
    return {"width": "100%", "height": "100%", "objectFit": "contain", "position": "absolute", "top": 0, "left": 0, "opacity": o}


@callback(
    Output("annotation-log", "children"), Output("save-status", "children"), Output("retrain-btn", "disabled"),
    Input("save-annotation-btn", "n_clicks"),
    State("selected-image-id", "data"), State("annotation-canvas", "relayoutData"),
    State("correct-class", "value"), State("brush-color", "value"), prevent_initial_call=True)
def save_ann(nc, iid, rd, cg, bc):
    if iid is None:
        return no_update, "No image selected.", True
    if cg is None:
        return no_update, "Select correct class.", True
    shapes = []
    if rd:
        for k, v in rd.items():
            if "shapes" in k:
                if isinstance(v, list):
                    shapes.extend(v)
                elif isinstance(v, dict):
                    shapes.append(v)
    c = BRUSH_COLORS.get(bc, BRUSH_COLORS["important"])
    annotation_store.add(image_id=iid, correct_class=cg, shapes=shapes, brush_color=c["fill"])
    log = [dbc.Card(dbc.CardBody([
        html.Strong(a["image_id"], style={"fontSize": "11px"}), html.Br(),
        html.Span(f"→ {CLASS_DISPLAY[a['correct_class']]} · {len(a['shapes'])} regions", style={"fontSize": "10px", "color": "#666"})
    ], style={"padding": "4px 8px"}), className="mb-1") for a in annotation_store.get_all()]
    return log, f"Saved for {iid}.", not bool(annotation_store.get_all())


@callback(
    Output("retrain-status", "children"),
    Output("annotation-log", "children", allow_duplicate=True),
    Output("retrain-btn", "disabled", allow_duplicate=True),
    Input("retrain-btn", "n_clicks"), prevent_initial_call=True)
def retrain(nc):
    a = annotation_store.get_all()
    if not a:
        return "No annotations.", no_update, True
    result = retrain_with_annotations(a)
    annotation_store.clear()
    return result["message"], [], True


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _pil_to_b64(p):
    i = Image.open(p).convert("RGB")
    b = BytesIO()
    i.save(b, format="PNG")
    return base64.b64encode(b.getvalue()).decode()


def _arr_to_b64(a):
    if a.dtype == np.uint8:
        i = Image.fromarray(a, "RGBA" if a.ndim == 3 and a.shape[2] == 4 else "RGB")
    else:
        i = Image.fromarray((a * 255).astype(np.uint8))
    b = BytesIO()
    i.save(b, format="PNG")
    return base64.b64encode(b.getvalue()).decode()


def _make_annot_fig(p, fc="rgba(255,0,0,0.3)", lc="red"):
    i = Image.open(p).convert("RGB")
    w, h = i.size
    f = go.Figure()
    f.add_layout_image(dict(source=i, xref="x", yref="y", x=0, y=h, sizex=w, sizey=h, sizing="stretch", layer="below"))
    f.update_xaxes(range=[0, w], showgrid=False, zeroline=False, visible=False)
    f.update_yaxes(range=[0, h], showgrid=False, zeroline=False, visible=False, scaleanchor="x")
    f.update_layout(template="plotly_white", margin=dict(l=0, r=0, t=0, b=0), height=280,
                     newshape=dict(fillcolor=fc, line=dict(color=lc, width=2)), dragmode="drawclosedpath")
    return f


if __name__ == "__main__":
    app.run(debug=True, port=8050)
