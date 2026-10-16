"""
================================================================================
Dashboard AgroCebada FIRA 2026  -  Streamlit
================================================================================
Muestra de forma sencilla las predicciones de rendimiento de cebada (t/ha)
generadas por el modelo jerarquico municipio -> estado (modelo_agrocebada.py).

Archivos necesarios en la MISMA carpeta:
    app_agrocebada.py        (este archivo)
    modelo_agrocebada.py     (el modelo)
    dataset_maestro.csv      (datos; tambien se puede subir desde la barra lateral)

Ejecutar:
    pip install streamlit pandas numpy scikit-learn plotly
    streamlit run app_agrocebada.py
================================================================================
"""
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modelo_agrocebada import ModeloAgroCebada

# ------------------------------------------------------------------------------
# Configuracion general
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="AgroCebada FIRA 2026",
    page_icon="🌾",
    layout="wide",
)

Y = "RENDIMIENTO_T_HA"
ID = "ID_POLIGON"
MUNI = "Municipio"
EDO = "Estado"
COLUMNAS_REQUERIDAS = [ID, MUNI, EDO, Y]

VERDE = "#2E7D32"
DORADO = "#C8A415"
GRIS = "#9E9E9E"
ROJO = "#C62828"

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem;}
    div[data-testid="stMetric"] {
        background-color: rgba(46,125,50,0.08);
        border-left: 5px solid #2E7D32;
        padding: 12px 16px;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------------------
# Carga de datos
# ------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def leer_csv(origen) -> pd.DataFrame:
    return pd.read_csv(origen)


def cargar_datos(archivo_subido):
    """Devuelve (df, mensaje_error)."""
    try:
        if archivo_subido is not None:
            df = leer_csv(archivo_subido)
        elif os.path.exists("dataset_maestro.csv"):
            df = leer_csv("dataset_maestro.csv")
        else:
            return None, (
                "No encontré **dataset_maestro.csv** en la carpeta de la app. "
                "Súbelo desde la barra lateral."
            )
    except Exception as e:  # noqa: BLE001
        return None, f"No pude leer el archivo: {e}"

    faltan = [c for c in COLUMNAS_REQUERIDAS if c not in df.columns]
    if faltan:
        return None, (
            f"Al archivo le faltan estas columnas: **{', '.join(faltan)}**. "
            f"Se necesitan: {', '.join(COLUMNAS_REQUERIDAS)}."
        )
    df = df.copy()
    df[MUNI] = df[MUNI].astype(str)
    df[EDO] = df[EDO].astype(str)
    return df, None


# ------------------------------------------------------------------------------
# Modelo (con cache para que no se recalcule en cada clic)
# ------------------------------------------------------------------------------
@st.cache_data(show_spinner="Entrenando el modelo y validándolo… (solo la primera vez)")
def correr_modelo(df: pd.DataFrame, k: float, semillas: int):
    train = df[df[Y].notna()].reset_index(drop=True)
    test = df[df[Y].isna()].reset_index(drop=True)

    modelo = ModeloAgroCebada(k=k).fit(train).calibrar_intervalo(train, n_semillas=semillas)

    y = train[Y].values
    pred_cv = y - modelo.residuos_
    pred = modelo.predict(test) if len(test) else np.array([])

    # Tabla por municipio: promedio real, n, y estimacion con contraccion
    mm = modelo.medias_municipio_.rename(columns={"mean": "prom_obs", "count": "n"}).reset_index()
    mm.columns = [MUNI, "prom_obs", "n"]
    edo_de_muni = (
        df.groupby(MUNI)[EDO].agg(lambda s: s.mode().iloc[0]).rename(EDO).reset_index()
    )
    mm = mm.merge(edo_de_muni, on=MUNI, how="left")
    mm["prom_edo"] = mm[EDO].map(modelo.medias_estado_).fillna(modelo.media_global_)
    mm["estimacion"] = (mm["n"] * mm["prom_obs"] + k * mm["prom_edo"]) / (mm["n"] + k)

    return {
        "train": train,
        "test": test,
        "pred": pred,
        "pred_cv": pred_cv,
        "residuos": modelo.residuos_,
        "tabla_muni": mm,
        "medias_estado": modelo.medias_estado_,
        "media_global": float(modelo.media_global_),
        "muni_conocidos": set(modelo.medias_municipio_.index),
        "n_muni": modelo.medias_municipio_["count"].to_dict(),
    }


def intervalo(pred, residuos, nivel):
    a = (100 - nivel) / 2
    lo, hi = np.percentile(residuos, [a, 100 - a])
    return np.clip(pred + lo, 0, None), pred + hi


def nivel_confianza(n):
    if n >= 10:
        return "Alta"
    if n >= 3:
        return "Media"
    return "Baja"


# ------------------------------------------------------------------------------
# Barra lateral
# ------------------------------------------------------------------------------
st.sidebar.title("🌾 AgroCebada")
st.sidebar.caption("Reto AgroCebada FIRA 2026")

archivo = st.sidebar.file_uploader(
    "Datos (CSV)",
    type="csv",
    help="Opcional. Si no subes nada, se usa dataset_maestro.csv de la carpeta.",
)

with st.sidebar.expander("⚙️ Ajustes avanzados"):
    k = st.slider(
        "Fuerza de contracción (k)",
        0.0, 10.0, 1.0, 0.5,
        help="Qué tanto se 'jala' la estimación de un municipio hacia el promedio de su "
             "estado. El modelo original usa k = 1.",
    )
    semillas = st.slider(
        "Repeticiones de la validación", 1, 10, 5,
        help="Más repeticiones = intervalos más estables, pero tarda más.",
    )

nivel = st.sidebar.select_slider(
    "Nivel del rango esperado",
    options=[50, 60, 70, 80, 90, 95],
    value=80,
    format_func=lambda v: f"{v}%",
    help="Con 80%, en 8 de cada 10 parcelas el rendimiento real debería caer dentro del rango.",
)

df, error = cargar_datos(archivo)
if error:
    st.title("🌾 Predicción de rendimiento de cebada")
    st.error(error)
    st.stop()

R = correr_modelo(df, k, semillas)
train, test = R["train"], R["test"]
residuos = R["residuos"]
y = train[Y].values

mae = float(np.mean(np.abs(residuos)))
rmse = float(np.sqrt(np.mean(residuos**2)))
r2 = float(1 - np.sum(residuos**2) / np.sum((y - y.mean()) ** 2))
error_rel = mae / float(y.mean()) * 100

# Tabla de predicciones con rangos y confianza
if len(test):
    lo, hi = intervalo(R["pred"], residuos, nivel)
    pred_df = pd.DataFrame(
        {
            "ID parcela": test[ID].values,
            "Estado": test[EDO].values,
            "Municipio": test[MUNI].values,
            "Rendimiento estimado (t/ha)": np.round(R["pred"], 3),
            "Mínimo esperado (t/ha)": np.round(lo, 3),
            "Máximo esperado (t/ha)": np.round(hi, 3),
            "Parcelas conocidas en el municipio": test[MUNI].map(R["n_muni"]).fillna(0).astype(int).values,
        }
    )
    pred_df["Confianza"] = pred_df["Parcelas conocidas en el municipio"].map(nivel_confianza)
else:
    pred_df = pd.DataFrame()

# ------------------------------------------------------------------------------
# Encabezado
# ------------------------------------------------------------------------------
st.title("🌾 Predicción de rendimiento de cebada")
st.markdown(
    "Estimación del **rendimiento (toneladas por hectárea)** de cada parcela, "
    "con base en lo que se ha cosechado en su municipio y su estado."
)

tab_resumen, tab_pred, tab_muni, tab_parcela, tab_calidad, tab_como = st.tabs(
    [
        "📊 Resumen",
        "📋 Predicciones",
        "🗺️ Por municipio",
        "🔎 Consultar una parcela",
        "✅ ¿Qué tan confiable es?",
        "💡 Cómo funciona",
    ]
)

# ------------------------------------------------------------------------------
# 1. RESUMEN
# ------------------------------------------------------------------------------
with tab_resumen:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Parcelas con dato real", f"{len(train):,}")
    c2.metric("Parcelas por predecir", f"{len(test):,}")
    c3.metric("Rendimiento promedio", f"{y.mean():.2f} t/ha")
    c4.metric("Error típico del modelo", f"± {mae:.2f} t/ha", f"≈ {error_rel:.0f}% del promedio", delta_color="off")

    st.markdown("### Rendimiento promedio por estado")
    est = (
        train.groupby(EDO)[Y]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "Promedio (t/ha)", "count": "Parcelas"})
        .sort_values("Promedio (t/ha)")
    )
    fig = px.bar(
        est, x="Promedio (t/ha)", y=EDO, orientation="h", text="Promedio (t/ha)",
        hover_data={"Parcelas": True}, color_discrete_sequence=[VERDE],
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(yaxis_title=None, height=max(250, 60 * len(est)), margin=dict(l=0, r=20, t=10, b=0))
    st.plotly_chart(fig, width="stretch")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### ¿Cómo se distribuyen los rendimientos?")
        fig = px.histogram(train, x=Y, nbins=30, color_discrete_sequence=[VERDE])
        fig.add_vline(x=y.mean(), line_dash="dash", line_color=ROJO, annotation_text="Promedio")
        fig.update_layout(xaxis_title="Rendimiento (t/ha)", yaxis_title="Parcelas", margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")
    with col_b:
        if len(pred_df):
            st.markdown("### Rendimiento estimado de las parcelas nuevas")
            fig = px.histogram(
                pred_df, x="Rendimiento estimado (t/ha)", nbins=30, color_discrete_sequence=[DORADO]
            )
            fig.update_layout(yaxis_title="Parcelas", margin=dict(t=10))
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No hay parcelas sin rendimiento en los datos, así que no hay predicciones nuevas.")

    st.info(
        f"**En una frase:** el modelo explica alrededor del **{r2 * 100:.0f}%** de las diferencias de "
        f"rendimiento entre parcelas y, en promedio, se equivoca por **{mae:.2f} t/ha**."
    )

# ------------------------------------------------------------------------------
# 2. PREDICCIONES
# ------------------------------------------------------------------------------
with tab_pred:
    if pred_df.empty:
        st.info("No hay parcelas por predecir en los datos cargados.")
    else:
        st.markdown(
            "Filtra por estado o municipio y descarga la tabla. "
            f"El **rango esperado** es el {nivel}% de las veces donde caerá el rendimiento real."
        )
        f1, f2, f3 = st.columns(3)
        estados = sorted(pred_df["Estado"].unique())
        sel_edo = f1.multiselect("Estado", estados, placeholder="Todos")
        base = pred_df[pred_df["Estado"].isin(sel_edo)] if sel_edo else pred_df
        municipios = sorted(base["Municipio"].unique())
        sel_muni = f2.multiselect("Municipio", municipios, placeholder="Todos")
        sel_conf = f3.multiselect("Confianza", ["Alta", "Media", "Baja"], placeholder="Todas")

        vista = base
        if sel_muni:
            vista = vista[vista["Municipio"].isin(sel_muni)]
        if sel_conf:
            vista = vista[vista["Confianza"].isin(sel_conf)]

        m1, m2, m3 = st.columns(3)
        m1.metric("Parcelas mostradas", f"{len(vista):,}")
        m2.metric("Promedio estimado", f"{vista['Rendimiento estimado (t/ha)'].mean():.2f} t/ha" if len(vista) else "—")
        m3.metric(
            "Rango del promedio",
            f"{vista['Rendimiento estimado (t/ha)'].min():.2f} – {vista['Rendimiento estimado (t/ha)'].max():.2f}" if len(vista) else "—",
        )

        st.dataframe(
            vista,
            width="stretch",
            hide_index=True,
            column_config={
                "Rendimiento estimado (t/ha)": st.column_config.NumberColumn(format="%.2f"),
                "Mínimo esperado (t/ha)": st.column_config.NumberColumn(format="%.2f"),
                "Máximo esperado (t/ha)": st.column_config.NumberColumn(format="%.2f"),
            },
        )

        d1, d2 = st.columns(2)
        d1.download_button(
            "⬇️ Descargar tabla filtrada (CSV)",
            vista.to_csv(index=False).encode("utf-8-sig"),
            "predicciones_filtradas.csv",
            "text/csv",
        )
        anexo = pred_df[["ID parcela", "Rendimiento estimado (t/ha)"]].rename(
            columns={"ID parcela": "ID_parcela", "Rendimiento estimado (t/ha)": "Rendimiento_predicho_ton_ha"}
        ).sort_values("ID_parcela")
        d2.download_button(
            "⬇️ Descargar tabla anexo para entrega (CSV)",
            anexo.to_csv(index=False).encode("utf-8"),
            "TABLA_ANEXO_predicciones.csv",
            "text/csv",
            help="Formato: ID_parcela, Rendimiento_predicho_ton_ha (todas las parcelas).",
        )

# ------------------------------------------------------------------------------
# 3. POR MUNICIPIO
# ------------------------------------------------------------------------------
with tab_muni:
    st.markdown(
        "Cada barra es un municipio. El color indica cuánta información tenemos de él: "
        "con **pocas parcelas conocidas** la estimación se apoya más en el promedio del estado."
    )
    tm = R["tabla_muni"].copy()
    tm["Confianza"] = tm["n"].map(nivel_confianza)

    cf1, cf2 = st.columns([1, 1])
    sel_e = cf1.multiselect("Estado", sorted(tm[EDO].unique()), placeholder="Todos", key="m_edo")
    top = cf2.slider("Municipios a mostrar", 5, max(5, min(60, len(tm))), min(20, len(tm)))
    if sel_e:
        tm = tm[tm[EDO].isin(sel_e)]

    orden = st.radio(
        "Ordenar por", ["Mayor rendimiento", "Menor rendimiento", "Más parcelas conocidas"], horizontal=True
    )
    if orden == "Mayor rendimiento":
        tm = tm.sort_values("estimacion", ascending=False)
    elif orden == "Menor rendimiento":
        tm = tm.sort_values("estimacion", ascending=True)
    else:
        tm = tm.sort_values("n", ascending=False)
    tm = tm.head(top).iloc[::-1]

    fig = px.bar(
        tm, x="estimacion", y=MUNI, orientation="h", color="Confianza",
        color_discrete_map={"Alta": VERDE, "Media": DORADO, "Baja": GRIS},
        category_orders={"Confianza": ["Alta", "Media", "Baja"]},
        hover_data={"n": True, "prom_obs": ":.2f", "prom_edo": ":.2f", EDO: True, "estimacion": ":.2f"},
        labels={"estimacion": "Rendimiento estimado (t/ha)", "n": "Parcelas conocidas",
                "prom_obs": "Promedio observado", "prom_edo": "Promedio del estado"},
    )
    fig.update_layout(yaxis_title=None, height=max(350, 28 * len(tm)), margin=dict(l=0, r=10, t=10, b=0))
    st.plotly_chart(fig, width="stretch")

    st.markdown("#### Variación dentro de cada municipio")
    st.caption("La caja muestra dónde está el 50% central de las parcelas; los puntos son parcelas con datos reales.")
    munis_graf = tm[MUNI].tolist()[-12:]
    sub = train[train[MUNI].isin(munis_graf)]
    if len(sub):
        fig = px.box(sub, x=MUNI, y=Y, points="all", color_discrete_sequence=[VERDE],
                     labels={Y: "Rendimiento (t/ha)", MUNI: ""})
        fig.update_layout(margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")

    with st.expander("Ver tabla completa"):
        tabla = tm.iloc[::-1][[MUNI, EDO, "n", "prom_obs", "estimacion", "Confianza"]].rename(
            columns={"n": "Parcelas conocidas", "prom_obs": "Promedio observado (t/ha)",
                     "estimacion": "Estimación del modelo (t/ha)"}
        )
        st.dataframe(tabla, hide_index=True, width="stretch",
                     column_config={"Promedio observado (t/ha)": st.column_config.NumberColumn(format="%.2f"),
                                    "Estimación del modelo (t/ha)": st.column_config.NumberColumn(format="%.2f")})

# ------------------------------------------------------------------------------
# 4. CONSULTAR UNA PARCELA
# ------------------------------------------------------------------------------
with tab_parcela:
    if pred_df.empty:
        st.info("No hay parcelas por predecir en los datos cargados.")
    else:
        st.markdown("Elige una parcela y mira su rendimiento estimado, su rango esperado y cómo se compara.")
        p1, p2 = st.columns([1, 2])
        edo_sel = p1.selectbox("Estado", sorted(pred_df["Estado"].unique()), key="p_edo")
        muni_sel = p1.selectbox(
            "Municipio", sorted(pred_df.loc[pred_df["Estado"] == edo_sel, "Municipio"].unique()), key="p_muni"
        )
        ids = pred_df[(pred_df["Estado"] == edo_sel) & (pred_df["Municipio"] == muni_sel)]["ID parcela"].tolist()
        id_sel = p1.selectbox("ID de parcela", ids)
        fila = pred_df[pred_df["ID parcela"] == id_sel].iloc[0]

        with p2:
            m1, m2, m3 = st.columns(3)
            m1.metric("Rendimiento estimado", f"{fila['Rendimiento estimado (t/ha)']:.2f} t/ha")
            m2.metric(f"Rango esperado ({nivel}%)",
                      f"{fila['Mínimo esperado (t/ha)']:.2f} – {fila['Máximo esperado (t/ha)']:.2f}")
            m3.metric("Confianza", fila["Confianza"],
                      f"{fila['Parcelas conocidas en el municipio']} parcelas conocidas", delta_color="off")

            prom_muni = R["tabla_muni"].set_index(MUNI)["prom_obs"].get(muni_sel, np.nan)
            prom_edo = float(R["medias_estado"].get(edo_sel, R["media_global"]))
            comp = pd.DataFrame(
                {
                    "Referencia": ["Esta parcela (estimado)", f"Promedio de {muni_sel}", f"Promedio de {edo_sel}", "Promedio general"],
                    "t/ha": [fila["Rendimiento estimado (t/ha)"], prom_muni, prom_edo, R["media_global"]],
                }
            ).dropna()
            fig = go.Figure(
                go.Bar(
                    x=comp["t/ha"], y=comp["Referencia"], orientation="h",
                    marker_color=[VERDE, DORADO, GRIS, GRIS][: len(comp)],
                    text=comp["t/ha"].round(2), textposition="outside",
                )
            )
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(height=260, margin=dict(l=0, r=30, t=10, b=0), xaxis_title="Rendimiento (t/ha)")
            st.plotly_chart(fig, width="stretch")

        if fila["Parcelas conocidas en el municipio"] == 0:
            st.warning(
                "De este municipio no hay parcelas con rendimiento conocido, por eso la estimación "
                "usa solo el **promedio de su estado**. Tómala con cautela."
            )
        elif fila["Confianza"] == "Baja":
            st.warning("Hay muy pocas parcelas conocidas en este municipio; la estimación es menos firme.")
        else:
            st.success("Hay suficiente información de este municipio para una estimación sólida.")

# ------------------------------------------------------------------------------
# 5. CALIDAD DEL MODELO
# ------------------------------------------------------------------------------
with tab_calidad:
    st.markdown(
        "Para medir qué tan bien funciona el modelo, se **esconde** el rendimiento de algunas parcelas "
        "conocidas, se estima, y se compara con lo que realmente ocurrió. Se repite varias veces."
    )
    q1, q2, q3 = st.columns(3)
    q1.metric("Error promedio (MAE)", f"{mae:.2f} t/ha",
              help="En promedio, la estimación se aleja del valor real por esta cantidad.")
    q2.metric("Error relativo", f"{error_rel:.1f}%",
              help="El error promedio como porcentaje del rendimiento promedio.")
    q3.metric("R² (variación explicada)", f"{r2 * 100:.0f}%",
              help="100% sería una predicción perfecta; 0% equivale a usar siempre el promedio general.")
    st.caption(f"RMSE = {rmse:.3f} t/ha (penaliza más los errores grandes).")

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("#### Real vs. estimado")
        st.caption("Mientras más cerca de la línea, mejor la estimación.")
        dfv = pd.DataFrame({"Real": y, "Estimado": R["pred_cv"], MUNI: train[MUNI].values})
        fig = px.scatter(dfv, x="Real", y="Estimado", opacity=0.55, hover_name=MUNI,
                         color_discrete_sequence=[VERDE])
        mn, mx = float(min(dfv.Real.min(), dfv.Estimado.min())), float(max(dfv.Real.max(), dfv.Estimado.max()))
        fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines", name="Acierto perfecto",
                                 line=dict(color=ROJO, dash="dash")))
        fig.update_layout(xaxis_title="Rendimiento real (t/ha)", yaxis_title="Rendimiento estimado (t/ha)",
                          margin=dict(t=10), legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")
    with g2:
        st.markdown("#### ¿Cuánto se equivoca?")
        st.caption("Diferencia entre lo real y lo estimado. Lo ideal: concentrado cerca de 0.")
        fig = px.histogram(x=residuos, nbins=40, color_discrete_sequence=[DORADO])
        lo_r, hi_r = np.percentile(residuos, [(100 - nivel) / 2, 100 - (100 - nivel) / 2])
        fig.add_vrect(x0=lo_r, x1=hi_r, fillcolor=VERDE, opacity=0.12, line_width=0,
                      annotation_text=f"{nivel}% de los casos", annotation_position="top left")
        fig.add_vline(x=0, line_color=ROJO, line_dash="dash")
        fig.update_layout(xaxis_title="Error (real − estimado, t/ha)", yaxis_title="Parcelas", margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")

    st.info(
        f"**Cómo leerlo:** en {nivel} de cada 100 parcelas, el rendimiento real queda entre "
        f"**{lo_r:+.2f}** y **{hi_r:+.2f} t/ha** respecto a la estimación."
    )

# ------------------------------------------------------------------------------
# 6. COMO FUNCIONA
# ------------------------------------------------------------------------------
with tab_como:
    st.markdown(
        """
### La idea, sin fórmulas

El análisis mostró que **el lugar (municipio) es lo que más define el rendimiento** de la cebada.
Probar más de 79 variables de satélite, clima y topografía no mejoró las predicciones una vez que
se conoce el municipio. Por eso el modelo es deliberadamente simple:

1. **Si el municipio tiene parcelas conocidas**, se parte de su promedio.
2. **Se mezcla con el promedio del estado**, para no confiar ciegamente en un municipio con pocas parcelas.
3. **Mientras más parcelas conocidas, más pesa el municipio**; con pocas, pesa más el estado.
4. **Si el municipio es nuevo**, se usa directamente el promedio de su estado.
        """
    )

    st.markdown("### 🧪 Pruébalo tú")
    s1, s2 = st.columns(2)
    n_demo = s1.slider("Parcelas conocidas en el municipio", 0, 50, 3)
    m_demo = s1.slider("Promedio del municipio (t/ha)", 1.0, 8.0, 4.5, 0.1)
    e_demo = s2.slider("Promedio del estado (t/ha)", 1.0, 8.0, 4.0, 0.1)
    k_demo = s2.slider("Fuerza de contracción k", 0.0, 10.0, float(k), 0.5)

    if n_demo == 0:
        resultado = e_demo
        peso_muni = 0.0
    else:
        resultado = (n_demo * m_demo + k_demo * e_demo) / (n_demo + k_demo)
        peso_muni = n_demo / (n_demo + k_demo) if (n_demo + k_demo) else 1.0

    r1, r2c, r3 = st.columns(3)
    r1.metric("Estimación final", f"{resultado:.2f} t/ha")
    r2c.metric("Peso del municipio", f"{peso_muni * 100:.0f}%")
    r3.metric("Peso del estado", f"{(1 - peso_muni) * 100:.0f}%")
    st.progress(float(peso_muni), text=f"{peso_muni * 100:.0f}% municipio · {(1 - peso_muni) * 100:.0f}% estado")

    st.latex(r"\text{estimación} = \frac{n \cdot \text{prom. municipio} + k \cdot \text{prom. estado}}{n + k}")
    st.caption("n = parcelas conocidas del municipio · k = fuerza de contracción (el modelo usa k = 1).")

    with st.expander("Limitaciones a tener en cuenta"):
        st.markdown(
            """
- El modelo **no distingue parcelas del mismo municipio**: todas reciben casi la misma estimación.
- No incorpora clima del ciclo actual, manejo ni variedad; no anticipa un mal año.
- En municipios con pocas parcelas conocidas la estimación es menos firme (se marca como confianza *Baja*).
- Los rangos esperados son empíricos y suponen que el comportamiento futuro se parece al de los datos usados.
            """
        )
