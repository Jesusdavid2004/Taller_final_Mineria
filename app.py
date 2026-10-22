"""Segmentación de Clientes con K-Means — Panel analítico (español).

Usa exclusivamente los artefactos guardados (model/kmeans_model.pkl,
model/scaler.pkl) y data/Mall_Customers.csv. No reentrena en predicción:
el scaler guardado solo aplica transform().
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import load_dataset, validate_schema, get_feature_matrix
from src.model_service import (
    load_artifacts, check_compatibility, predict_cluster,
    validate_customer_input, centroids_original_units, cluster_summary,
    FEATURE_COLUMNS, N_CLUSTERS,
)
from src.analysis import inertia_table, silhouette_table
from src.business_insights import (
    CLUSTER_PROFILES, GROWTH_SEGMENT_ID, GROWTH_JUSTIFICATION, get_profile,
)

# ---------- Tokens de diseño ----------
NAVY = "#1C2B4A"
TEAL = "#0E7C7B"
SLATE = "#5B6570"
BORDER = "#E3E1D8"
CLUSTER_COLORS = {
    "0": "#3A6EA5",
    "1": "#0E7C7B",
    "2": "#B97E26",
    "3": "#B0526B",
    "4": "#6A5FA0",
}
FONT = "'Inter', 'Segoe UI', Roboto, Arial, sans-serif"

st.set_page_config(
    page_title="Segmentación de Clientes · K-Means",
    layout="wide",
)

st.markdown(f"""
<style>
.stApp {{ background-color: #F7F6F2; font-family: {FONT}; }}
h1 {{ color: {NAVY} !important; font-size: 30px !important; font-weight: 700 !important;
     margin-bottom: 2px !important; }}
h2 {{ color: {NAVY} !important; font-size: 22px !important; font-weight: 650 !important; }}
h3 {{ color: {NAVY} !important; font-size: 17px !important; font-weight: 600 !important; }}
html, body, p, li, label {{ font-size: 15px; }}
.stMarkdown, p, li {{ color: #2A3342 !important; }}
div[data-testid="stCaptionContainer"] {{ color: {SLATE} !important; font-size: 13.5px; }}
div[data-testid="stMetric"] {{ background-color: #FFFFFF; border: 1px solid {BORDER};
  border-radius: 8px; padding: 10px 14px; }}
div[data-testid="stMetricValue"] {{ color: {NAVY} !important; font-size: 24px !important; }}
div[data-testid="stMetricLabel"] {{ color: {SLATE} !important; font-size: 13px !important; }}
.stButton>button {{ background-color: {TEAL}; color: #FFFFFF; font-weight: 600;
  border-radius: 6px; border: none; padding: 8px 22px; }}
.stButton>button:hover {{ background-color: #0A6362; color: #FFFFFF; }}
.stTabs [data-baseweb="tab"] {{ color: {SLATE}; font-size: 14.5px; }}
.stTabs [aria-selected="true"] {{ color: {NAVY} !important; }}
section[data-testid="stSidebar"] {{ display: none; }}
</style>
""", unsafe_allow_html=True)


def style_fig(fig: go.Figure, title: str) -> go.Figure:
    """Aplica el estilo común: fondo claro, tipografía legible, títulos claros."""
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, font=dict(size=16, color=NAVY)),
        font=dict(family="Inter, Segoe UI, Arial, sans-serif", size=12.5, color="#2A3342"),
        xaxis=dict(title_font=dict(size=13), tickfont=dict(size=12)),
        yaxis=dict(title_font=dict(size=13), tickfont=dict(size=12)),
        legend=dict(font=dict(size=12), title_font=dict(size=12.5)),
        margin=dict(l=50, r=20, t=60, b=50),
    )
    return fig


@st.cache_data
def _load_all():
    df = load_dataset()
    model, scaler = load_artifacts()
    X = get_feature_matrix(df)
    Xs = scaler.transform(X)
    labels = model.predict(Xs)
    df_l = df.copy()
    df_l["Cluster"] = labels
    return df, df_l, model, scaler, Xs, labels


try:
    df, df_l, model, scaler, X_scaled, labels = _load_all()
except Exception as e:  # noqa: BLE001
    st.error(f"No se pudieron cargar los artefactos o el dataset: {e}")
    st.stop()

issues = validate_schema(df) + check_compatibility(model, scaler)
for msg in issues:
    st.warning(msg)

centroids = centroids_original_units(model, scaler)
summary = cluster_summary(df, labels)

# ---------- Encabezado compacto + KPI ----------
st.title("Segmentación de Clientes con K-Means")
st.caption("Mall Customers Dataset · K-Means con K = 5 sobre ingreso anual y puntaje de gasto estandarizados")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Clientes totales", f"{len(df)}", help="Registros del dataset")
k2.metric("Clusters (K)", f"{N_CLUSTERS}", help="Elegido por codo + silueta")
k3.metric("Ingreso medio", f"{df['Annual Income (k$)'].mean():.1f} k$",
          help="Miles de USD; no equivale a ganancia")
k4.metric("Gasto medio", f"{df['Spending Score (1-100)'].mean():.1f} / 100")

st.divider()

tab_res, tab_eda, tab_eval, tab_seg, tab_biz, tab_pred = st.tabs([
    "Resumen general",
    "Exploración de datos",
    "Evaluación del modelo",
    "Segmentos de clientes",
    "Recomendaciones comerciales",
    "Clasificar un nuevo cliente",
])

# ---------- Resumen general ----------
with tab_res:
    st.header("Resumen general")
    st.write(
        "Modelo de segmentación no supervisada: **K-Means (K = 5)** sobre ingreso anual "
        "y puntaje de gasto, estandarizados con el `StandardScaler` guardado. La evaluación "
        "combina método del codo y coeficiente de silueta."
    )
    r1, r2, r3 = st.columns(3)
    r1.metric("Silueta (K = 5)", "0.5547", help="Máximo del barrido K=2…10")
    r2.metric("Inercia (K = 5)", "65.57", help="Suma de distancias cuadradas al centroide")
    r3.metric("Segmento con mayor oportunidad", "Cluster 3",
              help="Cautelosos premium: ingreso alto, gasto bajo")
    st.write("**Tamaño y promedios por segmento**")
    st.dataframe(summary, width="stretch")
    st.success(f"**Mayor oportunidad: cluster {GROWTH_SEGMENT_ID} "
               f"({get_profile(GROWTH_SEGMENT_ID)['nombre']}).** {GROWTH_JUSTIFICATION}")

# ---------- Exploración ----------
with tab_eda:
    st.header("Exploración de datos")
    st.write(f"**{df.shape[0]} filas × {df.shape[1]} columnas** · "
             f"Valores nulos: **{int(df.isnull().sum().sum())}** · "
             f"Duplicados: **{int(df.duplicated().sum())}**")
    st.write("**Primeros registros**")
    st.dataframe(df.head(10), width="stretch")
    st.write("**Estadística descriptiva**")
    st.dataframe(df.describe().round(2), width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="Gender", color="Gender",
                           color_discrete_map={"Female": "#3A6EA5", "Male": TEAL})
        fig = style_fig(fig, "Distribución por género (112 F / 88 M)")
        fig.update_layout(xaxis_title="Género", yaxis_title="Clientes", showlegend=False)
        st.plotly_chart(fig, width="stretch")
    with c2:
        melted = df.melt(value_vars=["Age", "Annual Income (k$)", "Spending Score (1-100)"],
                         var_name="Variable", value_name="Valor")
        fig = px.box(melted, x="Variable", y="Valor", color="Variable",
                     color_discrete_map={"Age": "#3A6EA5", "Annual Income (k$)": TEAL,
                                         "Spending Score (1-100)": "#B97E26"})
        fig = style_fig(fig, "Boxplots: inspección de valores atípicos")
        fig.update_layout(xaxis_title="", yaxis_title="Valor", showlegend=False)
        st.plotly_chart(fig, width="stretch")
    corr = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].corr().round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="Teal",
                    zmin=-1, zmax=1)
    fig = style_fig(fig, "Matriz de correlación (asociación lineal, no causalidad)")
    st.plotly_chart(fig, width="stretch")
    st.info("Ingreso y gasto casi no correlacionan (r ≈ 0.01) y forman grupos visibles: "
            "son las variables idóneas. Edad–gasto r ≈ −0.33. `CustomerID` es identificador "
            "y se excluye del modelo.")

# ---------- Evaluación ----------
with tab_eval:
    st.header("Evaluación del modelo")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Método del codo (K = 1…10)")
        inert = inertia_table(X_scaled)
        st.dataframe(inert, width="stretch")
        fig = go.Figure(go.Scatter(x=inert["K"], y=inert["inercia"],
                                   mode="lines+markers",
                                   marker=dict(color=TEAL, size=8)))
        fig = style_fig(fig, "Inercia vs K — el codo está en K = 5")
        fig.update_layout(xaxis_title="K", yaxis_title="Inercia",
                          xaxis=dict(dtick=1))
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.subheader("Coeficiente de silueta (K = 2…10)")
        sil = silhouette_table(X_scaled)
        st.dataframe(sil, width="stretch")
        fig = go.Figure(go.Scatter(x=sil["K"], y=sil["silueta"],
                                   mode="lines+markers",
                                   marker=dict(color="#3A6EA5", size=8)))
        fig = style_fig(fig, "Silueta vs K — máximo en K = 5 (0.5547)")
        fig.update_layout(xaxis_title="K", yaxis_title="Silueta",
                          xaxis=dict(dtick=1))
        st.plotly_chart(fig, width="stretch")
    st.info("La inercia siempre decrece con K; el codo (mejora marginal) aparece en K = 5 "
            "(108.9 → 65.6, luego solo → 55.1). La silueta confirma: máximo 0.5547 en K = 5. "
            "Silueta ≈ 1 bien asignado, ≈ 0 en frontera, < 0 probablemente mal asignado. "
            "K = 5 final por convergencia de ambos criterios, simplicidad y utilidad de negocio.")

# ---------- Segmentos ----------
with tab_seg:
    st.header("Segmentos de clientes")
    fig = px.scatter(df_l, x="Annual Income (k$)", y="Spending Score (1-100)",
                     color=df_l["Cluster"].astype(str),
                     hover_data=["Age", "Gender"],
                     color_discrete_map=CLUSTER_COLORS)
    fig.add_trace(go.Scatter(x=centroids[FEATURE_COLUMNS[0]],
                             y=centroids[FEATURE_COLUMNS[1]],
                             mode="markers",
                             marker=dict(symbol="x", size=13, color=NAVY,
                                         line=dict(width=2)),
                             name="Centroides"))
    fig = style_fig(fig, "Segmentación: ingreso anual (miles de USD) vs puntaje de gasto")
    fig.update_layout(xaxis_title="Annual Income (k$) — miles de USD",
                      yaxis_title="Spending Score (1–100)",
                      legend_title_text="Cluster")
    st.plotly_chart(fig, width="stretch")
    st.write("**Centroides en unidades originales** (punto medio de cada grupo)")
    st.dataframe(centroids, width="stretch")
    st.write("**Tamaños y promedios por cluster**")
    st.dataframe(summary, width="stretch")
    for k in range(N_CLUSTERS):
        p = get_profile(k)
        row = summary.loc[k]
        with st.expander(f"Cluster {k} — {p['nombre']} "
                         f"(n = {int(row['n'])}, {row['pct']} %)"):
            st.write(f"Edad media/mediana: **{row['edad_media']} / {row['edad_mediana']} años** · "
                     f"Ingreso medio: **{row['ingreso_medio']} k$** · "
                     f"Gasto medio: **{row['gasto_medio']}**")
            st.write(f"**Interpretación:** {p['descripcion']}")
            st.write(f"**Estrategia:** {p['estrategia']}")

# ---------- Recomendaciones ----------
with tab_biz:
    st.header("Recomendaciones comerciales")
    st.success(f"**Mayor oportunidad: cluster {GROWTH_SEGMENT_ID} "
               f"({CLUSTER_PROFILES[GROWTH_SEGMENT_ID]['nombre']}).** "
               f"{GROWTH_JUSTIFICATION}")
    st.warning("El dataset no mide rentabilidad, frecuencia de compra ni valor de vida del "
               "cliente: son hipótesis de marketing basadas en ingreso y gasto observados.")
    for k in range(N_CLUSTERS):
        p = get_profile(k)
        st.write(f"**Cluster {k} · {p['nombre']}** — {p['estrategia']}")

# ---------- Predicción ----------
with tab_pred:
    st.header("Clasificar un nuevo cliente")
    st.write("El modelo usa **solo** ingreso anual y puntaje de gasto, transformados con el "
             "scaler guardado. La predicción **asigna el cliente a un segmento existente**; "
             "no pronostica compras futuras.")
    with st.form("pred"):
        ci, cs = st.columns(2)
        with ci:
            income = st.number_input("Ingreso anual (miles de USD, k$)",
                                     min_value=0.0, max_value=200.0, value=60.0, step=1.0)
        with cs:
            spending = st.number_input("Puntaje de gasto (1–100)",
                                       min_value=1.0, max_value=100.0, value=50.0, step=1.0)
        go_btn = st.form_submit_button("Clasificar cliente")
    if go_btn:
        try:
            validate_customer_input(income, spending)
            label = predict_cluster(float(income), float(spending), model, scaler)
            p = get_profile(label)
            row = summary.loc[label]
            st.success(f"**Cluster asignado: {label} — {p['nombre']}**")
            m1, m2, m3 = st.columns(3)
            m1.metric("Tamaño del segmento", f"{int(row['n'])} ({row['pct']} %)")
            m2.metric("Ingreso medio del segmento", f"{row['ingreso_medio']} k$")
            m3.metric("Gasto medio del segmento", f"{row['gasto_medio']}")
            st.write(f"**Recomendación:** {p['estrategia']}")
            fig = px.scatter(df_l, x="Annual Income (k$)", y="Spending Score (1-100)",
                             color=df_l["Cluster"].astype(str), opacity=0.4,
                             color_discrete_map=CLUSTER_COLORS)
            fig.add_trace(go.Scatter(x=[float(income)], y=[float(spending)],
                                     mode="markers",
                                     marker=dict(symbol="star", size=16, color=TEAL,
                                                 line=dict(width=2, color=NAVY)),
                                     name="Nuevo cliente"))
            fig = style_fig(fig, "Nuevo cliente en el mapa de segmentos")
            fig.update_layout(xaxis_title="Annual Income (k$) — miles de USD",
                              yaxis_title="Spending Score (1–100)",
                              legend_title_text="Cluster")
            st.plotly_chart(fig, width="stretch")
        except ValueError as e:
            st.error(str(e))

st.divider()
st.caption("Annual Income en miles de USD (k$), no pesos. Ingreso ≠ ganancia. "
           "Clustering no supervisado: no hay 'accuracy' sin verdad externa.")
