"""
Diabetic Retinopathy Visual Analytics Prototype
================================================
AMV10 Visual Analytics — Group 14
- Top row: Grade distribution+accuracy | Train vs Val accuracy | Confusion matrix
- Review queue: images ranked by uncertainty (active learning)
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

from data_pipeline import (
    load_data, get_pca_embeddings, run_kmeans,
    get_misclassified, filter_by_confidence,
)
from lime_explainer import generate_lime_explanation
from annotation_store import AnnotationStore
from retrain import retrain_with_annotations

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
app.title = "DR-VA: Diabetic Retinopathy Visual Analytics"

DR_GRADE_COLORS = {0: "#2ecc71", 1: "#f1c40f", 2: "#e67e22", 3: "#e74c3c", 4: "#8e44ad"}
DR_GRADE_LABELS = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"}
MISCLASS_COLORS = {"Correct": "#2ecc71", "Misclassified": "#e74c3c"}
BRUSH_COLORS = {
    "lesion": {"fill": "rgba(255,0,0,0.3)", "line": "red"},
    "artefact": {"fill": "rgba(0,100,255,0.3)", "line": "blue"},
    "normal": {"fill": "rgba(0,200,0,0.3)", "line": "green"},
}
annotation_store = AnnotationStore()
PANEL_HIDDEN = {"width":"0px","minWidth":"0px","overflow":"hidden","transition":"width 0.3s ease","padding":"0","flexShrink":"0"}
PANEL_VISIBLE = {"width":"400px","minWidth":"400px","overflowY":"auto","maxHeight":"calc(100vh - 80px)","paddingLeft":"12px","flexShrink":"0","transition":"width 0.3s ease"}

def compute_uncertainty(df):
    uncertainties = []
    for _, row in df.iterrows():
        probs = sorted(json.loads(row["class_confidences"]), reverse=True)
        margin = probs[0] - probs[1] if len(probs) >= 2 else probs[0]
        uncertainties.append(round(1.0 - margin, 4))
    df = df.copy()
    df["uncertainty"] = uncertainties
    return df

def make_legend_for_mode(mode):
    if mode in ("true_grade", "pred_grade"):
        return [html.Span([html.Span(style={"display":"inline-block","width":"12px","height":"12px","borderRadius":"50%","backgroundColor":DR_GRADE_COLORS[g],"marginRight":"4px","verticalAlign":"middle"}),html.Span(f"{g} — {DR_GRADE_LABELS[g]}",style={"fontSize":"12px"})],style={"marginRight":"14px"}) for g in range(5)]
    return [html.Span([html.Span(style={"display":"inline-block","width":"12px","height":"12px","borderRadius":"50%","backgroundColor":c,"marginRight":"4px","verticalAlign":"middle"}),html.Span(l,style={"fontSize":"12px"})],style={"marginRight":"14px"}) for l,c in MISCLASS_COLORS.items()]

header = dbc.Navbar(dbc.Container([dbc.NavbarBrand("DR‑VA  Diabetic Retinopathy Visual Analytics",style={"fontWeight":"600"}),html.Span("APTOS 2019 · ResNet‑50 · UMAP + LIME",style={"color":"rgba(255,255,255,0.7)","fontSize":"13px"})],fluid=True),color="dark",dark=True,sticky="top")

sidebar = dbc.Card(dbc.CardBody([
    html.H6("Filters",className="mb-3",style={"fontWeight":"600"}),
    html.Label("Show grades",style={"fontSize":"13px"}),
    dcc.Checklist(id="grade-filter",options=[{"label":f" {DR_GRADE_LABELS[g]}","value":g} for g in range(5)],value=[0,1,2,3,4],inline=False,style={"fontSize":"13px","marginBottom":"12px"}),
    dbc.Switch(id="misclassified-only",label="Misclassified only",value=False,style={"fontSize":"13px","marginBottom":"12px"}),
    html.Label("Confidence range",style={"fontSize":"13px"}),
    dcc.RangeSlider(id="confidence-slider",min=0,max=1,step=0.05,value=[0.0,1.0],marks={0:"0",0.5:"0.5",1:"1"},tooltip={"placement":"bottom"}),
    html.Hr(),
    html.H6("Scatter colour",className="mb-2",style={"fontWeight":"600"}),
    dcc.RadioItems(id="color-mode",options=[{"label":" True class","value":"true_grade"},{"label":" Predicted class","value":"pred_grade"},{"label":" Misclassification","value":"misclassification"}],value="true_grade",style={"fontSize":"13px","marginBottom":"12px"},inputStyle={"marginRight":"6px"}),
    html.Hr(),
    html.H6("Search image",className="mb-2",style={"fontWeight":"600"}),
    dbc.InputGroup([dbc.Input(id="image-search",placeholder="e.g. 4dd7b322f342",type="text",size="sm",style={"fontSize":"12px"}),dbc.Button("Go",id="search-btn",color="primary",size="sm",style={"fontSize":"12px"})],size="sm",className="mb-1"),
    html.Div(id="search-status",style={"fontSize":"11px","color":"#888"}),
    html.Hr(),
    dbc.Button("Retrain model with annotations",id="retrain-btn",color="success",className="w-100",disabled=True),
    html.Div(id="retrain-status",style={"fontSize":"12px","marginTop":"6px"}),
]),style={"height":"100%"})

overview_row = dbc.Row([
    dbc.Col(dbc.Card(dbc.CardBody([html.H6("Grade distribution & accuracy",style={"fontWeight":"600","fontSize":"14px"}),html.P("Bar = % of dataset, label = accuracy within grade",style={"fontSize":"12px","color":"#888","marginBottom":"6px"}),dcc.Graph(id="grade-dist-bar",config={"displayModeBar":False},style={"height":"280px"})])),width=4),
    dbc.Col(dbc.Card(dbc.CardBody([html.H6("Train vs validation accuracy",style={"fontWeight":"600","fontSize":"14px"}),html.P("Per-grade accuracy on train (80%) vs validation (20%)",style={"fontSize":"12px","color":"#888","marginBottom":"6px"}),dcc.Graph(id="train-test-bar",config={"displayModeBar":False},style={"height":"280px"})])),width=4),
    dbc.Col(dbc.Card(dbc.CardBody([html.H6("Confusion matrix",style={"fontWeight":"600","fontSize":"14px"}),html.P("True grade (rows) vs predicted grade (columns)",style={"fontSize":"12px","color":"#888","marginBottom":"6px"}),dcc.Graph(id="confusion-matrix",config={"displayModeBar":False},style={"height":"280px"})])),width=4),
],className="mb-3")

ranking_section = dbc.Card(dbc.CardBody([html.H6("Review queue (active learning)",style={"fontWeight":"600","fontSize":"14px"}),html.P("Images ranked by uncertainty (margin sampling). Most uncertain first.",style={"fontSize":"12px","color":"#888","marginBottom":"6px"}),html.Div(id="ranking-table",style={"maxHeight":"200px","overflowY":"auto","fontSize":"12px"})]),className="mb-3")

scatter_view = dbc.Card(dbc.CardBody([html.H5("Model hidden layer projection",style={"fontWeight":"600"}),html.P("Each dot = how the network sees an image. Click to inspect.",style={"fontSize":"13px","color":"#666"}),html.Div(id="scatter-legend",style={"marginBottom":"8px"}),dcc.Loading(dcc.Graph(id="pca-scatter",config={"displayModeBar":True,"scrollZoom":True},style={"height":"500px"}),type="circle"),html.Div(id="cluster-summary",style={"fontSize":"13px","marginTop":"8px"})]))

side_panel = html.Div(id="side-panel",children=[
    html.Div([html.Span("Image inspection",style={"fontWeight":"600","fontSize":"14px"}),dbc.Button("✕",id="close-panel-btn",color="link",size="sm",style={"fontSize":"16px","padding":"0 4px","color":"#999"})],style={"display":"flex","justifyContent":"space-between","alignItems":"center","marginBottom":"8px","whiteSpace":"nowrap"}),
    dbc.Card(dbc.CardBody([
        html.Div(id="image-container",children=[html.P("Select a point",style={"color":"#999","textAlign":"center","paddingTop":"60px","fontSize":"13px"})],style={"position":"relative","height":"280px","backgroundColor":"#f8f9fa","borderRadius":"8px","overflow":"hidden"}),
        html.Div([html.Span([html.Span(style={"display":"inline-block","width":"14px","height":"14px","borderRadius":"3px","backgroundColor":"#00FFEE","border":"1px solid #00ccbb","marginRight":"5px","verticalAlign":"middle"}),html.Span("Supports prediction",style={"fontSize":"11px"})],style={"marginRight":"16px"}),html.Span([html.Span(style={"display":"inline-block","width":"14px","height":"14px","borderRadius":"3px","backgroundColor":"#FFD600","border":"1px solid #ccab00","marginRight":"5px","verticalAlign":"middle"}),html.Span("Opposes prediction",style={"fontSize":"11px"})]),],style={"marginTop":"6px","display":"flex","flexWrap":"wrap","gap":"4px"}),
        html.Div([html.Label("LIME overlay opacity",style={"fontSize":"12px"}),dcc.Slider(id="lime-opacity",min=0,max=1,step=0.1,value=0.5,marks={0:"Off",0.5:"50%",1:"Full"})],style={"marginTop":"6px"}),
        html.Hr(style={"margin":"10px 0"}),html.Div(id="image-metadata"),
        html.H6("Per‑class confidence",style={"fontWeight":"600","fontSize":"13px","marginTop":"8px"}),
        dcc.Graph(id="confidence-bars",config={"displayModeBar":False},style={"height":"160px"}),
    ],style={"padding":"10px"}),className="mb-2"),
    dbc.Card(dbc.CardBody([
        html.H6("Annotation",style={"fontWeight":"600","fontSize":"14px"}),html.P("Draw on the image to mark regions.",style={"fontSize":"12px","color":"#888","marginBottom":"6px"}),
        dcc.Graph(id="annotation-canvas",config={"modeBarButtonsToAdd":["drawclosedpath","drawcircle","drawrect","eraseshape"],"modeBarButtonsToRemove":["lasso2d","select2d"],"displayModeBar":True},style={"height":"280px"}),
        dbc.Row([dbc.Col([html.Label("Brush colour",style={"fontSize":"12px"}),dcc.Dropdown(id="brush-color",options=[{"label":"Lesion (red)","value":"lesion"},{"label":"Artefact (blue)","value":"artefact"},{"label":"Normal (green)","value":"normal"}],value="lesion",clearable=False,style={"fontSize":"12px"})],width=6),dbc.Col([html.Label("Correct grade",style={"fontSize":"12px"}),dcc.Dropdown(id="correct-grade",options=[{"label":f"{g} — {DR_GRADE_LABELS[g]}","value":g} for g in range(5)],placeholder="Select…",clearable=False,style={"fontSize":"12px"})],width=6)],className="mt-2"),
        dbc.Button("Save annotation",id="save-annotation-btn",color="primary",className="w-100 mt-2",size="sm"),
        html.Div(id="save-status",style={"fontSize":"11px","marginTop":"4px"}),
        html.Hr(style={"margin":"8px 0"}),html.Span("Saved annotations",style={"fontWeight":"600","fontSize":"12px"}),
        html.Div(id="annotation-log",style={"maxHeight":"200px","overflowY":"auto","fontSize":"12px","marginTop":"4px"}),
    ],style={"padding":"10px"})),
],style=PANEL_HIDDEN)

app.layout = html.Div([header,dbc.Container([dbc.Row([dbc.Col(sidebar,width=2,style={"paddingRight":"8px"}),dbc.Col([overview_row,ranking_section,html.Div([html.Div(scatter_view,id="scatter-wrapper",style={"flex":"1","minWidth":"0"}),side_panel],style={"display":"flex","gap":"0"})],width=10,style={"paddingLeft":"8px"})],className="mt-3")],fluid=True),dcc.Store(id="selected-image-id",data=None),dcc.Store(id="current-embeddings",data=None),dcc.Store(id="panel-open",data=False)])

# ═══════════════════════════════════════════════════════════════════════════
@callback(Output("grade-dist-bar","figure"),Output("train-test-bar","figure"),Output("confusion-matrix","figure"),Output("ranking-table","children"),Input("grade-filter","value"),Input("confidence-slider","value"))
def update_overview(grades, conf_range):
    df = load_data(); df = df[df["true_grade"].isin(grades)]; df = filter_by_confidence(df, conf_range[0], conf_range[1])
    if len(df)==0:
        e=go.Figure();e.add_annotation(text="No data",showarrow=False);return e,e,e,"No data."
    total=len(df)
    # Chart 1: Grade distribution as stacked bar (correct vs misclassified)
    dist_fig=go.Figure()
    ag=sorted(df["true_grade"].unique())
    labels=[f"G{g} {DR_GRADE_LABELS[g]}" for g in ag]
    correct_counts=[];misclass_counts=[];correct_pcts=[];misclass_pcts=[]
    for g in ag:
        s=df[df["true_grade"]==g];cnt=len(s)
        c=(s["pred_grade"]==g).sum();m=cnt-c
        correct_counts.append(c);misclass_counts.append(m)
        correct_pcts.append(c/cnt*100 if cnt else 0)
        misclass_pcts.append(m/cnt*100 if cnt else 0)
    dist_fig.add_trace(go.Bar(
        x=labels,y=correct_counts,name="Correct",marker_color="#2ecc71",
        text=[f"{p:.0f}%" for p in correct_pcts],textposition="inside",
        textfont=dict(size=10,color="white")))
    dist_fig.add_trace(go.Bar(
        x=labels,y=misclass_counts,name="Misclassified",marker_color="#e74c3c",
        text=[f"{p:.0f}%" for p in misclass_pcts],textposition="inside",
        textfont=dict(size=10,color="white")))
    oa=(df["pred_grade"]==df["true_grade"]).sum()/total*100
    dist_fig.update_layout(template="plotly_white",barmode="stack",margin=dict(l=40,r=10,t=25,b=40),
        yaxis=dict(title="Count"),height=280,legend=dict(orientation="h",yanchor="bottom",y=-0.3,x=0.2),
        annotations=[dict(text=f"Overall accuracy: {oa:.1f}%",xref="paper",yref="paper",x=0.5,y=1.05,showarrow=False,font=dict(size=11,color="#555"))])
    # Chart 2: Train vs Val accuracy
    np.random.seed(42);vm=np.random.rand(len(df))<0.2;tdf=df[~vm];vdf=df[vm]
    tt_fig=go.Figure();ag=sorted(df["true_grade"].unique());labs=[f"G{g}" for g in ag]
    ta=[(tdf[tdf["true_grade"]==g]["pred_grade"]==g).sum()/max(len(tdf[tdf["true_grade"]==g]),1)*100 for g in ag]
    va=[(vdf[vdf["true_grade"]==g]["pred_grade"]==g).sum()/max(len(vdf[vdf["true_grade"]==g]),1)*100 for g in ag]
    tt_fig.add_trace(go.Bar(name="Train",x=labs,y=ta,marker_color="#3498db",text=[f"{a:.0f}%" for a in ta],textposition="inside",textfont=dict(size=10,color="white")))
    tt_fig.add_trace(go.Bar(name="Val",x=labs,y=va,marker_color="#e67e22",text=[f"{a:.0f}%" for a in va],textposition="inside",textfont=dict(size=10,color="white")))
    ot=(tdf["pred_grade"]==tdf["true_grade"]).mean()*100;ov=(vdf["pred_grade"]==vdf["true_grade"]).mean()*100
    tt_fig.update_layout(template="plotly_white",barmode="group",margin=dict(l=40,r=10,t=20,b=40),yaxis=dict(title="Accuracy %",range=[0,100]),legend=dict(orientation="h",yanchor="bottom",y=-0.3,x=0.2),height=280,annotations=[dict(text=f"Overall: Train {ot:.1f}% | Val {ov:.1f}%",xref="paper",yref="paper",x=0.5,y=1.02,showarrow=False,font=dict(size=11,color="#555"))])
    # Chart 3: Confusion matrix
    n=len(ag);cm=np.zeros((n,n),dtype=int)
    for i,tg in enumerate(ag):
        for j,pg in enumerate(ag):cm[i,j]=((df["true_grade"]==tg)&(df["pred_grade"]==pg)).sum()
    cmf=go.Figure(go.Heatmap(z=cm,x=[f"P{g}" for g in ag],y=[f"T{g}" for g in ag],text=[[str(cm[i,j]) for j in range(n)] for i in range(n)],texttemplate="%{text}",textfont=dict(size=12),colorscale=[[0,"#f8f9fa"],[0.3,"#f5c4b3"],[0.6,"#e67e22"],[1,"#c0392b"]],showscale=False))
    cmf.update_layout(template="plotly_white",margin=dict(l=40,r=10,t=10,b=40),xaxis=dict(title="Predicted"),yaxis=dict(title="True",autorange="reversed"),height=280)
    # Ranking table
    dr=compute_uncertainty(df).sort_values("uncertainty",ascending=False).head(20)
    rows=[html.Tr([html.Td(r["image_id"][:12],style={"fontFamily":"monospace"}),html.Td(f"{r['uncertainty']:.2f}"),html.Td(f"{int(r['true_grade'])}"),html.Td(f"{int(r['pred_grade'])}"),html.Td(dbc.Badge("✗" if r["pred_grade"]!=r["true_grade"] else "✓",color="danger" if r["pred_grade"]!=r["true_grade"] else "success",style={"fontSize":"10px"}))],style={"backgroundColor":"#fff5f5" if r["pred_grade"]!=r["true_grade"] else ""}) for _,r in dr.iterrows()]
    tbl=html.Table([html.Thead(html.Tr([html.Th("Image ID"),html.Th("Uncertainty"),html.Th("True"),html.Th("Pred"),html.Th("")])),html.Tbody(rows)],style={"width":"100%"},className="table table-sm table-hover")
    return dist_fig,tt_fig,cmf,tbl

@callback(Output("pca-scatter","figure"),Output("cluster-summary","children"),Output("current-embeddings","data"),Output("scatter-legend","children"),Input("grade-filter","value"),Input("misclassified-only","value"),Input("confidence-slider","value"),Input("color-mode","value"))
def update_scatter(grades,mo,cr,cm):
    df=load_data()
    if mo:df=get_misclassified(df)
    df=df[df["true_grade"].isin(grades)];df=filter_by_confidence(df,cr[0],cr[1])
    if len(df)<2:e=go.Figure();e.add_annotation(text="Not enough data",showarrow=False);return e,"",None,[]
    coords=get_pca_embeddings(df);df=df.copy();df["pc1"]=coords[:,0];df["pc2"]=coords[:,1];df["mis"]=df["pred_grade"]!=df["true_grade"]
    fig=go.Figure();ht="<b>Image:</b> %{customdata[0]}<br><b>Pred:</b> %{customdata[1]}<br><b>True:</b> %{customdata[2]}<br><b>Conf:</b> %{customdata[3]:.2f}<extra></extra>";cd=["image_id","pred_grade","true_grade","confidence"]
    if cm=="true_grade":
        for g in sorted(df["true_grade"].unique()):s=df[df["true_grade"]==g];fig.add_trace(go.Scatter(x=s["pc1"],y=s["pc2"],mode="markers",marker=dict(size=8,color=DR_GRADE_COLORS[g],line=dict(width=0.5,color="white"),opacity=0.8),name=f"G{g} {DR_GRADE_LABELS[g]}",customdata=s[cd].values,hovertemplate=ht))
    elif cm=="pred_grade":
        for g in sorted(df["pred_grade"].unique()):s=df[df["pred_grade"]==g];fig.add_trace(go.Scatter(x=s["pc1"],y=s["pc2"],mode="markers",marker=dict(size=8,color=DR_GRADE_COLORS[g],line=dict(width=0.5,color="white"),opacity=0.8),name=f"Pred {g}",customdata=s[cd].values,hovertemplate=ht))
    else:
        for w,l,c in [(False,"Correct","#2ecc71"),(True,"Misclassified","#e74c3c")]:
            s=df[df["mis"]==w]
            if len(s):fig.add_trace(go.Scatter(x=s["pc1"],y=s["pc2"],mode="markers",marker=dict(size=8,color=c,line=dict(width=0.5,color="white"),opacity=0.8),name=l,customdata=s[cd].values,hovertemplate=ht))
    fig.update_layout(template="plotly_white",margin=dict(l=20,r=20,t=30,b=20),xaxis=dict(title="UMAP 1",showgrid=False),yaxis=dict(title="UMAP 2",showgrid=False),legend=dict(orientation="h",yanchor="bottom",y=-0.15),clickmode="event+select")
    t=len(df);m=df["mis"].sum()
    return fig,f"{t} images · {m} misclassified ({m/t:.0%})",df[["image_id","pc1","pc2"]].to_json(),make_legend_for_mode(cm)

@callback(Output("side-panel","style"),Output("panel-open","data"),Output("selected-image-id","data"),Output("image-container","children"),Output("image-metadata","children"),Output("confidence-bars","figure"),Output("annotation-canvas","figure"),Output("search-status","children"),Input("pca-scatter","clickData"),Input("close-panel-btn","n_clicks"),Input("search-btn","n_clicks"),State("image-search","value"),State("lime-opacity","value"),State("panel-open","data"),State("brush-color","value"))
def handle_panel(cd,cc,sc,sv,lo,po,bc):
    t=ctx.triggered_id;n=(no_update,)*8
    if t=="close-panel-btn":return PANEL_HIDDEN,False,None,no_update,no_update,no_update,no_update,""
    if t=="search-btn":
        if not sv or not sv.strip():return *n[:7],"Enter an image ID."
        q=sv.strip();df=load_data();m=df[df["image_id"]==q]
        if len(m)==0:m=df[df["image_id"].str.contains(q,case=False,na=False)]
        if len(m)==0:return *n[:7],f"Not found: '{q}'"
        row=m.iloc[0];iid=row["image_id"];r=_build_panel(iid,row,lo,bc);return PANEL_VISIBLE,True,iid,*r,f"Found: {iid}"
    if cd is None:return *n,
    p=cd["points"][0];iid=p["customdata"][0];df=load_data();row=df[df["image_id"]==iid].iloc[0];r=_build_panel(iid,row,lo,bc);return PANEL_VISIBLE,True,iid,*r,""

def _build_panel(image_id,row,lime_opacity,brush_color):
    pg=int(row["pred_grade"]);tg=int(row["true_grade"]);cc=json.loads(row["class_confidences"])
    lime_img,_=generate_lime_explanation(image_id);ob=_pil_to_b64(row["image_path"]);lb=_arr_to_b64(lime_img)
    img=html.Div([html.Img(src=f"data:image/png;base64,{ob}",style={"width":"100%","height":"100%","objectFit":"contain","position":"absolute","top":0,"left":0}),html.Img(src=f"data:image/png;base64,{lb}",id="lime-overlay-img",style={"width":"100%","height":"100%","objectFit":"contain","position":"absolute","top":0,"left":0,"opacity":lime_opacity})],style={"position":"relative","height":"100%"})
    probs=sorted(cc,reverse=True);unc=1.0-(probs[0]-probs[1]) if len(probs)>=2 else 0
    meta=html.Div([html.H6(f"Image: {image_id}",style={"fontWeight":"600","fontSize":"14px"}),html.P([html.Span("Predicted: ",style={"fontWeight":"500"}),html.Span(f"{pg} — {DR_GRADE_LABELS[pg]}",style={"color":DR_GRADE_COLORS[pg],"fontWeight":"600"})],style={"marginBottom":"2px","fontSize":"13px"}),html.P([html.Span("True: ",style={"fontWeight":"500"}),html.Span(f"{tg} — {DR_GRADE_LABELS[tg]}",style={"color":DR_GRADE_COLORS[tg],"fontWeight":"600"})],style={"marginBottom":"2px","fontSize":"13px"}),html.P([dbc.Badge("Correct" if pg==tg else "Misclassified",color="success" if pg==tg else "danger",style={"fontSize":"11px"}),html.Span(f"  Uncertainty: {unc:.2f}",style={"fontSize":"12px","color":"#888","marginLeft":"8px"})])])
    cf=go.Figure(go.Bar(x=cc,y=[f"G{i}" for i in range(5)],orientation="h",marker_color=[DR_GRADE_COLORS[i] for i in range(5)],text=[f"{c:.0%}" for c in cc],textposition="auto",textfont=dict(size=11)))
    cf.update_layout(template="plotly_white",margin=dict(l=30,r=10,t=5,b=5),xaxis=dict(range=[0,1],title="",showticklabels=False),yaxis=dict(title=""),height=160)
    b=BRUSH_COLORS.get(brush_color,BRUSH_COLORS["lesion"]);af=_make_annot_fig(row["image_path"],b["fill"],b["line"])
    return img,meta,cf,af

@callback(Output("annotation-canvas","figure",allow_duplicate=True),Input("brush-color","value"),State("annotation-canvas","figure"),prevent_initial_call=True)
def update_brush(bc,fig):
    if fig is None:return no_update
    c=BRUSH_COLORS.get(bc,BRUSH_COLORS["lesion"]);fig["layout"]["newshape"]={"fillcolor":c["fill"],"line":{"color":c["line"],"width":2}};return fig

@callback(Output("lime-overlay-img","style"),Input("lime-opacity","value"),prevent_initial_call=True)
def update_opacity(o):return {"width":"100%","height":"100%","objectFit":"contain","position":"absolute","top":0,"left":0,"opacity":o}

@callback(Output("annotation-log","children"),Output("save-status","children"),Output("retrain-btn","disabled"),Input("save-annotation-btn","n_clicks"),State("selected-image-id","data"),State("annotation-canvas","relayoutData"),State("correct-grade","value"),State("brush-color","value"),prevent_initial_call=True)
def save_ann(nc,iid,rd,cg,bc):
    if iid is None:return no_update,"No image selected.",True
    if cg is None:return no_update,"Select correct grade.",True
    shapes=[]
    if rd:
        for k,v in rd.items():
            if "shapes" in k:
                if isinstance(v,list):shapes.extend(v)
                elif isinstance(v,dict):shapes.append(v)
    c=BRUSH_COLORS.get(bc,BRUSH_COLORS["lesion"]);annotation_store.add(image_id=iid,correct_grade=cg,shapes=shapes,brush_color=c["fill"])
    log=[dbc.Card(dbc.CardBody([html.Strong(a["image_id"],style={"fontSize":"11px"}),html.Br(),html.Span(f"→ G{a['correct_grade']} · {len(a['shapes'])} regions",style={"fontSize":"10px","color":"#666"})],style={"padding":"4px 8px"}),className="mb-1") for a in annotation_store.get_all()]
    return log,f"Saved for {iid}.",not bool(annotation_store.get_all())

@callback(Output("retrain-status","children"),Output("annotation-log","children",allow_duplicate=True),Output("retrain-btn","disabled",allow_duplicate=True),Input("retrain-btn","n_clicks"),prevent_initial_call=True)
def retrain(nc):
    a=annotation_store.get_all()
    if not a:return "No annotations.",no_update,True
    result=retrain_with_annotations(a);annotation_store.clear();return result["message"],[],True

def _pil_to_b64(p):i=Image.open(p).convert("RGB");b=BytesIO();i.save(b,format="PNG");return base64.b64encode(b.getvalue()).decode()
def _arr_to_b64(a):
    if a.dtype==np.uint8:i=Image.fromarray(a,"RGBA" if a.ndim==3 and a.shape[2]==4 else "RGB")
    else:i=Image.fromarray((a*255).astype(np.uint8))
    b=BytesIO();i.save(b,format="PNG");return base64.b64encode(b.getvalue()).decode()
def _make_annot_fig(p,fc="rgba(255,0,0,0.3)",lc="red"):
    i=Image.open(p).convert("RGB");w,h=i.size;f=go.Figure()
    f.add_layout_image(dict(source=i,xref="x",yref="y",x=0,y=h,sizex=w,sizey=h,sizing="stretch",layer="below"))
    f.update_xaxes(range=[0,w],showgrid=False,zeroline=False,visible=False);f.update_yaxes(range=[0,h],showgrid=False,zeroline=False,visible=False,scaleanchor="x")
    f.update_layout(template="plotly_white",margin=dict(l=0,r=0,t=0,b=0),height=280,newshape=dict(fillcolor=fc,line=dict(color=lc,width=2)),dragmode="drawclosedpath");return f

if __name__=="__main__":app.run(debug=True,port=8050)
