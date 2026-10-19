"""
AgroCebada FIRA 2026 - Dashboard con frontend renovado y cálculos oficiales.

La interfaz visual proviene del dashboard nuevo, mientras que predicciones,
validación, confianza y rangos se calculan exclusivamente con calculos.py,
que corresponde al primer programa entregado.
"""
import json
import os
import unicodedata
from html import escape

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

import calculos as C
from calculos import ID, MUNI, EDO, Y

st.set_page_config(page_title="AgroCebada FIRA 2026", page_icon="🌾", layout="wide")

CICLO = "2025"
VMIN, VMAX = 2.0, 5.6
ORDEN_CONF = ["Alta", "Media", "Baja", "Muy baja"]
VERDE = "#2E7D32"
DORADO = "#C8A415"
GRIS = "#9E9E9E"
ROJO = "#C62828"

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem;}
    div[data-testid="stMetric"] {
        background-color: rgba(46,125,50,0.07);
        border-left: 5px solid #2E7D32;
        padding: 10px 14px;
        border-radius: 10px;
    }
    .hero {
        background: linear-gradient(135deg, #E8F5E9 0%, #FFF9E6 100%);
        border: 1px solid #C8E6C9;
        border-radius: 14px;
        padding: 18px 24px;
        margin-bottom: 12px;
    }
    .hero h1 {margin: 0; font-size: 1.65rem; color: #1B5E20;}
    .hero p  {margin: 6px 0 0; color: #37474F; font-size: 0.98rem; line-height: 1.45;}

    .chip {display:inline-block; padding: 2px 12px; border-radius: 999px;
           font-size: 0.82rem; font-weight: 600; margin-bottom: 4px;}
    .chip-alta   {background:#E8F5E9; color:#1B5E20; border:1px solid #A5D6A7;}
    .chip-media  {background:#FFF3CD; color:#7A5C00; border:1px solid #FFE082;}
    .chip-baja   {background:#FFE0B2; color:#8C4A00; border:1px solid #FFCC80;}
    .chip-muybaja{background:#FFCDD2; color:#9A0F0F; border:1px solid #EF9A9A;}

    .leyenda {display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:10px; margin:6px 0 14px 0;}
    .col-exp {background:#fff; border:1px solid #E0E7E0; border-left:4px solid #C8A415;
              border-radius:10px; padding:10px 12px;}
    .col-exp b {color:#1B5E20; font-size:.9rem;}
    .col-exp span {display:block; font-size:.83rem; color:#455A64; line-height:1.4; margin-top:3px;}

    .caja-resumen {background:#F1F8E9; border:1px solid #C5E1A5; border-left:6px solid #2E7D32;
                   border-radius:12px; padding:14px 18px; margin:6px 0 14px 0; color:#263238;}
    .caja-resumen h3 {margin:0 0 6px 0; color:#1B5E20; font-size:1.05rem;}
    .caja-resumen ul {margin:0; padding-left:18px; line-height:1.6; font-size:0.92rem;}
    .caja-analogia {background:#FFF9E6; border:1px solid #FFE082; border-left:5px solid #C8A415;
                    border-radius:10px; padding:14px 18px; margin:14px 0; color:#37474F;}

    div[data-testid="stPlotlyChart"] .js-plotly-plot g.axis text,
    div[data-testid="stPlotlyChart"] .js-plotly-plot .ticktext {
        fill: var(--text-color, #37474F) !important;
    }

    .conf-lista {display:flex; flex-direction:column; gap:8px; margin-top:6px;}
    .conf-item {display:flex; align-items:center; gap:10px; height:58px;
                background:var(--secondary-background-color, #F7FBF7);
                border:1px solid rgba(128,128,128,.28); border-radius:10px; padding:6px 10px;}
    .conf-item .chip {min-width:76px; text-align:center; flex-shrink:0; margin-bottom:0;}
    .conf-desc {font-size:.8rem; line-height:1.35; color:var(--text-color, #455A64);}

    [role="tablist"] {gap:8px; margin-top: 8px;}
    [role="tab"] {border-radius:10px !important; padding:8px 18px !important; font-weight:600;}
    [role="tab"][aria-selected="true"] {background-color:#2E7D32 !important; color:#fff !important;}

    .footer-app {margin-top:28px; padding-top:14px; border-top:1px solid #DCE9DC;
                 color:#78909C; font-size:.8rem; text-align:center;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------------------
# Utilidades
# ------------------------------------------------------------------------------
def limpiar_html(txt: str) -> str:
    return "\n".join(linea.strip() for linea in txt.strip().splitlines())


def sin_acentos(t) -> str:
    t = unicodedata.normalize("NFD", str(t))
    return "".join(c for c in t if unicodedata.category(c) != "Mn").lower().strip()


def buscar_columna(df: pd.DataFrame, candidatos):
    mapa = {sin_acentos(c): c for c in df.columns}
    for cand in candidatos:
        if sin_acentos(cand) in mapa:
            return mapa[sin_acentos(cand)]
    return None


def mostrar_grafico(fig):
    for kwargs in (
        dict(width="stretch", theme="streamlit"),
        dict(use_container_width=True, theme="streamlit"),
        dict(width="stretch"),
        dict(use_container_width=True),
    ):
        try:
            st.plotly_chart(fig, **kwargs)
            return
        except (TypeError, ValueError):
            continue


def mostrar_tabla(obj):
    try:
        st.dataframe(obj, width="stretch", hide_index=True)
    except TypeError:
        st.dataframe(obj, use_container_width=True, hide_index=True)


def estilizar(fig):
    fig.update_layout(
        font=dict(family="'Segoe UI', 'Helvetica Neue', Arial, sans-serif", size=13),
        margin=dict(l=12, r=30, t=16, b=12),
    )
    return fig


def redondear_barras(fig, radio=6):
    try:
        fig.update_layout(barcornerradius=radio)
    except Exception:  # noqa: BLE001
        pass
    return fig


def pintar_confianza(df_o_styler):
    colores = {"Alta": "#E8F5E9", "Media": "#FFF3CD", "Baja": "#FFE0B2", "Muy baja": "#FFCDD2"}
    texto = {"Alta": "#1B5E20", "Media": "#7A5C00", "Baja": "#8C4A00", "Muy baja": "#9A0F0F"}

    def celda(v):
        return (f"background-color:{colores.get(v, 'transparent')}; color:{texto.get(v, 'inherit')}; "
                f"font-weight:600;")

    sty = df_o_styler.style if isinstance(df_o_styler, pd.DataFrame) else df_o_styler
    if hasattr(sty, "map"):
        return sty.map(celda, subset=["Confianza"])
    return sty.applymap(celda, subset=["Confianza"])


def formatear(df: pd.DataFrame, formatos: dict):
    fm = {c: f for c, f in formatos.items() if c in df.columns}
    return df.style.format(fm, na_rep="—")


def nivel_confianza(n, cv=None):
    if n >= 20:
        i = 0
    elif n >= 5:
        i = 1
    elif n >= 2:
        i = 2
    else:
        i = 3
    if cv is not None and n >= 2 and not np.isnan(cv) and cv > CV_ALTO:
        i = min(i + 1, 3)
    return ORDEN_CONF[i]


def intervalo(pred, residuos, nivel):
    a = (100 - nivel) / 2
    lo, hi = np.percentile(residuos, [a, 100 - a])
    return np.clip(pred + lo, 0, None), pred + hi


# ------------------------------------------------------------------------------
# BACKEND OFICIAL: primer programa (calculos.py)
# ------------------------------------------------------------------------------
@st.cache_data(show_spinner="Entrenando y validando el modelo (solo la primera vez)…")
def preparar():
    for f in ("dataset_maestro.csv", "modelo_agrocebada.py"):
        if not os.path.exists(f):
            return None, f"Falta el archivo **{f}** en la carpeta de la app."
    try:
        d = C.cargar()
        R = C.calcular_todo(d)
        R["val_k0"] = C.validar(R["train"], k=0)["por_semilla"].mean().to_dict()
        return R, None
    except Exception as e:  # noqa: BLE001
        return None, f"No pude ejecutar los cálculos oficiales: {e}"

@st.cache_data(show_spinner=False)
def leer_geojson():
    if not os.path.exists("parcelas.geojson"):
        return None
    try:
        with open("parcelas.geojson", encoding="utf-8") as fh:
            gj = json.load(fh)
        return {str(f["properties"].get("ID_POLIGON", "")).upper(): f["geometry"] for f in gj["features"]}
    except Exception:
        return None

R, error = preparar()
if error:
    st.title("🌾 AgroCebada FIRA 2026")
    st.error(error)
    st.stop()

train = R["train"].copy()
test = R["test"].copy()
tm = R["municipios"].copy()
M = R["metricas"]
val = R["val"]
poligonos = leer_geojson()

# Alias de compatibilidad: el frontend nuevo usa estos nombres, pero los valores
# proceden directamente del resultado oficial de calculos.py.
R["tabla_muni"] = tm.copy()
R["tabla_muni"]["prom_edo"] = R["tabla_muni"].get("media_estado", np.nan)
R["tabla_muni"]["cv"] = R["tabla_muni"]["desv"] / R["tabla_muni"]["prom_obs"].replace(0, np.nan)
R["medias_estado"] = {}
for edo, grupo in train.groupby(EDO):
    R["medias_estado"][str(edo)] = float(grupo[Y].mean())
R["media_global"] = float(train[Y].mean())
R["n_muni"] = dict(zip(tm[MUNI], tm["n"].astype(int)))
R["cv_muni"] = dict(zip(tm[MUNI], R["tabla_muni"]["cv"]))
R["muni_conocidos"] = set(tm[MUNI])
R["pred"] = test["pred"].to_numpy(dtype=float) if len(test) else np.array([])
R["pred_cv"] = np.asarray(val["pred_cv"], dtype=float)
R["residuos"] = np.asarray(val["residuos"], dtype=float)

# Las predicciones, rangos y confianza se toman del primer programa sin
# recalcularlos con otra implementación.
if len(test):
    pred_df = pd.DataFrame({
        "ID parcela": test[ID].astype(str).values,
        "Municipio": test[MUNI].values,
        "Estado": test[EDO].values,
        "Rendimiento estimado (t/ha)": np.round(test["pred"].astype(float).values, 4),
        "Mínimo esperado (t/ha)": np.round(test["min_esp"].astype(float).values, 4),
        "Máximo esperado (t/ha)": np.round(test["max_esp"].astype(float).values, 4),
        "Área (ha)": pd.to_numeric(test["area_ha"], errors="coerce").values,
        "Parcelas conocidas en el municipio": test[MUNI].map(R["n_muni"]).fillna(0).astype(int).values,
        "Confianza": test["confianza"].values,
        "Latitud": pd.to_numeric(test["lat"], errors="coerce").values,
        "Longitud": pd.to_numeric(test["lon"], errors="coerce").values,
    })
    if "toneladas" in test.columns:
        pred_df["Toneladas totales (*)"] = pd.to_numeric(test["toneladas"], errors="coerce").round(2).values
    else:
        pred_df["Toneladas totales (*)"] = (pred_df["Rendimiento estimado (t/ha)"] * pred_df["Área (ha)"]).round(2)
else:
    pred_df = pd.DataFrame()

# El primer programa usa k=1 y rangos oficiales de 80%; se mantienen fijos
# para que la nueva interfaz no altere los resultados de entrega.
k = 1.0
nivel = 80

# Métricas oficiales de calculos.py
mae = float(M["MAE"])
rmse = float(M["RMSE"])
r2 = float(M["R2"])
r2_pct = max(0.0, r2) * 100
error_rel = float(M["error_rel"]) * 100
error_rel_txt = f"{error_rel:.0f}%" if np.isfinite(error_rel) else "—"
y = train[Y].to_numpy(dtype=float)
residuos = R["residuos"]

FORMATO_PRED = {
    "Rendimiento estimado (t/ha)": "{:.2f}",
    "Mínimo esperado (t/ha)": "{:.2f}",
    "Máximo esperado (t/ha)": "{:.2f}",
    "Área (ha)": "{:.2f}",
    "Toneladas totales (*)": "{:,.1f}",
}

# Barra lateral del frontend nuevo, sin controles que puedan modificar el cálculo oficial.
st.sidebar.title("🌾 AgroCebada")
st.sidebar.caption(f"Reto AgroCebada FIRA 2026 · ciclo {CICLO}")
st.sidebar.markdown(
    limpiar_html(
        """
        <div style="margin-top:4px;"><b>¿Qué significa la confianza?</b></div>
        <div class="conf-lista">
          <div class="conf-item"><span class="chip chip-alta">Alta</span><span class="conf-desc">Estimación respaldada por el comportamiento de las parcelas del municipio.</span></div>
          <div class="conf-item"><span class="chip chip-media">Media</span><span class="conf-desc">Información suficiente, pero con mayor incertidumbre.</span></div>
          <div class="conf-item"><span class="chip chip-baja">Baja</span><span class="conf-desc">Mayor variación o información limitada.</span></div>
          <div class="conf-item"><span class="chip chip-muybaja">Muy baja</span><span class="conf-desc">Poca evidencia para respaldar la estimación.</span></div>
        </div>
        """
    ), unsafe_allow_html=True)

st.sidebar.divider()
if len(test):
    st.sidebar.download_button(
        "⬇️ Tabla oficial de predicciones",
        C.anexo_csv(R["anexo"]),
        "TABLA_ANEXO_predicciones.csv",
        "text/csv",
        help="Formato oficial: ID_parcela y rendimiento predicho en t/ha con 4 decimales.",
        width="stretch",
    )
    detalle = test[[ID, EDO, MUNI, "area_ha", "pred", "min_esp", "max_esp", "confianza"]].rename(columns={
        ID: "ID_parcela", "area_ha": "Area_poligono_ha", "pred": "Rendimiento_predicho_ton_ha",
        "min_esp": "Minimo_esperado_80", "max_esp": "Maximo_esperado_80", "confianza": "Confianza"})
    st.sidebar.download_button(
        "⬇️ Detalle con rangos y confianza",
        detalle.sort_values("ID_parcela").to_csv(index=False, float_format="%.4f").encode("utf-8-sig"),
        "predicciones_detalle.csv", "text/csv", width="stretch",
    )

# ------------------------------------------------------------------------------
# Encabezado + cifras clave
# ------------------------------------------------------------------------------
st.markdown(
    limpiar_html(
        """
        <div class="hero">
        <h1>AgroCebada · Mapa y predicción de rendimiento por parcela</h1>
        <p>Reto FIRA 2026 — explora en el mapa principal cuántas toneladas por hectárea se espera cosechar
        en cada parcela y consulta el detalle individual a la derecha.</p>
        </div>
        """
    ),
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Parcelas con dato real", f"{len(train):,}",
          help="Parcelas cuyo rendimiento ya se conoce (sirven para entrenar y validar).")
c2.metric("Parcelas por predecir", f"{len(test):,}",
          help="Parcelas nuevas a las que el modelo les estima el rendimiento.")
c3.metric("Rendimiento promedio", f"{y.mean():.2f} t/ha",
          help="Promedio de las parcelas conocidas; es la referencia general.")
c4.metric("Error típico del modelo", f"± {mae:.2f} t/ha", f"≈ {error_rel_txt} del promedio", delta_color="off",
          help="En promedio, la estimación se aleja del valor real por esta cantidad.")

if len(test):
    municipios_desconocidos = sorted(set(test.loc[~test["conocido"], MUNI]))
    if municipios_desconocidos:
        st.error(
            "**Advertencia:** el modelo solo está validado para los municipios conocidos. "
            f"Las parcelas en {', '.join(municipios_desconocidos)} usan la media del estado, "
            "lo cual no está validado. El error esperado es de ~0.5 t/ha o más."
        )

# ==============================================================================
# VISTA PRINCIPAL: MAPA INTERACTIVO + CONSULTA DE PARCELA AL LADO
# ==============================================================================
st.markdown("---")

col_mapa, col_detalle = st.columns([1.45, 1], gap="large")

with col_mapa:
    st.subheader("Mapa de rendimiento por parcela")

    # Filtros del mapa en 2D
    fm1, fm2, fm3 = st.columns([1.1, 1.2, 1.1])
    estados_disp = ["Todos"] + (
        sorted(pred_df["Estado"].unique().tolist())
        if not pred_df.empty else sorted(train[EDO].unique().tolist())
    )
    edo_mapa = fm1.selectbox("Estado", estados_disp, key="mapa_edo")

    if not pred_df.empty:
        base_muni_mapa = pred_df if edo_mapa == "Todos" else pred_df[pred_df["Estado"] == edo_mapa]
        munis_disp = ["Todos"] + sorted(base_muni_mapa["Municipio"].unique().tolist())
    else:
        base_muni_mapa = train if edo_mapa == "Todos" else train[train[EDO] == edo_mapa]
        munis_disp = ["Todos"] + sorted(base_muni_mapa[MUNI].unique().tolist())

    muni_mapa = fm2.selectbox("Municipio", munis_disp, key="mapa_muni")
    mostrar_reales = fm3.checkbox("Incluir parcelas con dato real", value=False)

    mapa = pd.DataFrame()
    if not pred_df.empty:
        mapa = pd.DataFrame({
            "ID parcela": pred_df["ID parcela"].astype(str),
            "Estado": pred_df["Estado"],
            "Municipio": pred_df["Municipio"],
            "Tipo": "Por estimar",
            "Rendimiento (t/ha)": pred_df["Rendimiento estimado (t/ha)"],
            "Área (ha)": pred_df["Área (ha)"],
            "Confianza": pred_df["Confianza"],
            "Latitud": pred_df["Latitud"],
            "Longitud": pred_df["Longitud"],
        })

    if mostrar_reales:
        reales = pd.DataFrame({
            "ID parcela": train[ID].astype(str),
            "Estado": train[EDO],
            "Municipio": train[MUNI],
            "Tipo": "Con dato real",
            "Rendimiento (t/ha)": train[Y],
            "Área (ha)": train["_area"] if "_area" in train.columns else np.nan,
            "Confianza": "Dato conocido",
            "Latitud": train["_lat"] if "_lat" in train.columns else np.nan,
            "Longitud": train["_lon"] if "_lon" in train.columns else np.nan,
        })
        mapa = pd.concat([mapa, reales], ignore_index=True)

    if not mapa.empty:
        if edo_mapa != "Todos":
            mapa = mapa[mapa["Estado"] == edo_mapa]
        if muni_mapa != "Todos":
            mapa = mapa[mapa["Municipio"] == muni_mapa]
        mapa = mapa.dropna(subset=["Latitud", "Longitud"]).copy()

    if mapa.empty:
        st.info("No hay parcelas con coordenadas para los filtros seleccionados.")
    else:
        rend_min, rend_max = VMIN, VMAX

        def calcular_color(val, tipo):
            if pd.isna(val):
                rgb = [158, 158, 158]
            else:
                ratio = float(np.clip((val - rend_min) / (rend_max - rend_min), 0, 1))
                # Rojo para rendimiento bajo, verde para rendimiento alto.
                rgb = [int(198 - ratio * 187), int(40 + ratio * 125), int(40 - ratio * 29)]
            return rgb + ([225] if tipo == "Por estimar" else [125])

        mapa["color_pdk"] = mapa.apply(
            lambda r: calcular_color(r["Rendimiento (t/ha)"], r["Tipo"]), axis=1
        )
        area = pd.to_numeric(mapa["Área (ha)"], errors="coerce")
        area_val = area.fillna(area.mean() if area.notna().any() else 2.0).clip(lower=0.1)
        mapa["radio_pdk"] = np.sqrt((area_val * 10000) / np.pi).clip(lower=15, upper=120)

        centro_lat = float(mapa["Latitud"].mean())
        centro_lon = float(mapa["Longitud"].mean())
        span = max(
            float(mapa["Latitud"].max() - mapa["Latitud"].min()),
            float(mapa["Longitud"].max() - mapa["Longitud"].min()),
            0.015,
        )
        zoom_inicial = float(np.clip(np.log2(360 / span) - 0.4, 4, 14))
        view_state = pdk.ViewState(
            latitude=centro_lat, longitude=centro_lon, zoom=zoom_inicial,
            pitch=0, bearing=0,
        )

        tooltip = {
            "html": (
                "<b>Parcela:</b> {ID parcela}<br/>"
                "<b>Tipo:</b> {Tipo}<br/>"
                "<b>Municipio:</b> {Municipio}, {Estado}<br/>"
                "<b>Rendimiento:</b> {Rendimiento (t/ha)} t/ha<br/>"
                "<b>Área:</b> {Área (ha)} ha<br/>"
                "<b>Confianza:</b> {Confianza}"
            ),
            "style": {"backgroundColor": "#1B5E20", "color": "white"},
        }

        capa_puntos = pdk.Layer(
            "ScatterplotLayer",
            data=mapa,
            get_position=["Longitud", "Latitud"],
            get_radius="radio_pdk",
            get_fill_color="color_pdk",
            get_line_color=[35, 35, 35, 220],
            get_line_width=1,
            stroked=True,
            filled=True,
            radius_min_pixels=4,
            radius_max_pixels=18,
            pickable=True,
            auto_highlight=True,
        )

        # Si existe el GeoJSON, dibuja la forma de las parcelas en 2D.
        features = []
        if poligonos:
            for _, r in mapa.iterrows():
                geom = poligonos.get(str(r["ID parcela"]).upper())
                if geom:
                    features.append({
                        "type": "Feature",
                        "geometry": geom,
                        "properties": {
                            "ID parcela": r["ID parcela"],
                            "Estado": str(r["Estado"]),
                            "Municipio": str(r["Municipio"]),
                            "Tipo": r["Tipo"],
                            "Rendimiento (t/ha)": float(r["Rendimiento (t/ha)"]),
                            "Área (ha)": float(r["Área (ha)"]) if pd.notna(r["Área (ha)"]) else 0,
                            "Confianza": str(r["Confianza"]),
                            "color_pdk": r["color_pdk"],
                        },
                    })

        capas = []
        if features:
            capas.append(pdk.Layer(
                "GeoJsonLayer",
                data={"type": "FeatureCollection", "features": features},
                pickable=True,
                stroked=True,
                filled=True,
                get_fill_color="properties.color_pdk",
                get_line_color=[35, 35, 35, 220],
                get_line_width=2,
                line_width_min_pixels=1,
                auto_highlight=True,
            ))
        else:
            capas.append(capa_puntos)

        deck = pdk.Deck(
            layers=capas,
            initial_view_state=view_state,
            tooltip=tooltip,
            map_style="light",
        )
        st.pydeck_chart(deck, use_container_width=True)
        st.caption(
            "Mapa plano 2D: rojo = rendimiento bajo y verde = alto. "
            "El tamaño de los puntos representa aproximadamente la superficie de la parcela. "
            "Las parcelas por estimar comparten la estimación de su municipio; el modelo combina "
            "el promedio municipal con el estatal."
            + ("" if features else " · Se muestran puntos porque no se encontraron geometrías de parcela en parcelas.geojson.")
        )

with col_detalle:
    st.subheader("Detalle de parcela y zona")
    if pred_df.empty:
        st.info("No hay parcelas por predecir en los datos cargados.")
    else:
        # Sincronizado con el filtro del mapa para que sea inmediato encontrar una parcela
        sub_pred = pred_df.copy()
        if edo_mapa != "Todos":
            sub_pred = sub_pred[sub_pred["Estado"] == edo_mapa]
        if muni_mapa != "Todos":
            sub_pred = sub_pred[sub_pred["Municipio"] == muni_mapa]
        if sub_pred.empty:
            sub_pred = pred_df.copy()

        id_sel = st.selectbox(
            "Selecciona una parcela de la zona mostrada",
            sub_pred["ID parcela"].tolist(),
            format_func=lambda x: f"{x} — {sub_pred.loc[sub_pred['ID parcela'] == x, 'Municipio'].iloc[0]} ({sub_pred.loc[sub_pred['ID parcela'] == x, 'Estado'].iloc[0]})",
            key="p_id_main",
        )
        fila = pred_df[pred_df["ID parcela"] == id_sel].iloc[0]
        edo_sel = fila["Estado"]
        muni_sel = fila["Municipio"]
        n_muni = int(fila["Parcelas conocidas en el municipio"])

        m1, m2 = st.columns(2)
        m1.metric("Rendimiento estimado", f"{fila['Rendimiento estimado (t/ha)']:.2f} t/ha")
        m2.metric(
            f"Rango esperado ({nivel}%)",
            f"{fila['Mínimo esperado (t/ha)']:.2f} – {fila['Máximo esperado (t/ha)']:.2f}",
        )

        m3, m4 = st.columns(2)
        area_v = fila["Área (ha)"]
        if pd.notna(area_v) and pd.notna(fila["Toneladas totales (*)"]):
            m3.metric(
                "Área y cosecha estimada",
                f"{area_v:.2f} ha",
                f"~{fila['Toneladas totales (*)']:.1f} t totales",
                delta_color="off",
            )
        else:
            m3.metric("Área de la parcela", "—")

        peso_muni = n_muni / (n_muni + k) if (n_muni + k) else 0.0
        m4.metric(
            "Confianza en la zona",
            fila["Confianza"],
            f"{n_muni} parcelas conocidas ({peso_muni * 100:.0f}% peso municipal)",
            delta_color="off",
        )

        tm_i = R["tabla_muni"].set_index(MUNI)
        prom_muni = float(tm_i["prom_obs"].get(muni_sel, np.nan))
        prom_edo = float(R["medias_estado"].get(edo_sel, R["media_global"]))
        comp = pd.DataFrame({
            "Referencia": [
                "Esta parcela (estimado)",
                f"Promedio de {muni_sel}",
                f"Promedio de {edo_sel}",
                "Promedio general",
            ],
            "t/ha": [
                fila["Rendimiento estimado (t/ha)"],
                prom_muni,
                prom_edo,
                R["media_global"],
            ],
            "color": [VERDE, DORADO, GRIS, "#CFD8DC"],
        }).dropna(subset=["t/ha"])

        fig_comp = go.Figure(go.Bar(
            x=comp["t/ha"],
            y=comp["Referencia"],
            orientation="h",
            marker_color=comp["color"].tolist(),
            text=comp["t/ha"].round(2),
            textposition="outside",
        ))
        fig_comp.update_yaxes(autorange="reversed")
        estilizar(fig_comp).update_layout(height=230, xaxis_title="Rendimiento (t/ha)")
        redondear_barras(fig_comp)
        mostrar_grafico(fig_comp)

        # ¿Qué significa en la práctica? (Q7)
        st.markdown("#### ¿Qué significa para el productor?")
        ton = float(fila["Toneladas totales (*)"]) if pd.notna(fila["Toneladas totales (*)"]) else 0.0
        area_parcela = float(area_v) if pd.notna(area_v) else 0.0
        err_ton = mae * area_parcela
        st.markdown(
            f"Con **{area_parcela:.2f} ha**, se esperan unas **{ton:,.1f} toneladas** de grano. "
            f"El error absoluto medio global del modelo es **{mae:.2f} t/ha**, equivalente a "
            f"unas **±{err_ton:,.1f} toneladas** para esta superficie."
        )
        with st.expander("Convertir a pesos"):
            precio = st.number_input(
                "Precio por tonelada (MXN)", min_value=0.0, value=0.0, step=100.0,
                help="Escribe el precio de contrato o de mercado que quieras usar.",
                key=f"precio_ton_{id_sel}",
            )
            if precio > 0:
                st.markdown(
                    f"Ingreso esperado: **${ton * precio:,.0f}** · "
                    f"margen orientativo de error: **±${err_ton * precio:,.0f}**"
                )
        st.caption(
            "Úsalo para planear o dimensionar un crédito por zona; no garantiza el rendimiento "
            "de una parcela individual. El margen usa el MAE global como referencia, no como intervalo garantizado."
        )

        # ¿De dónde sale este número? (Q13)
        st.markdown("#### ¿De dónde sale este número?")
        peso_muni = n_muni / (n_muni + k) if (n_muni + k) else 0.0
        st.progress(
            float(np.clip(peso_muni, 0, 1)),
            text=f"{peso_muni * 100:.1f}% promedio de {muni_sel} · "
                 f"{(1 - peso_muni) * 100:.1f}% promedio de {edo_sel}",
        )
        referencias = []
        if np.isfinite(prom_muni):
            referencias.append({
                "Referencia": f"Promedio de {muni_sel} ({n_muni} parcelas)",
                "t/ha": prom_muni,
                "color": "#78c679",
            })
        referencias.extend([
            {"Referencia": f"Promedio de {edo_sel}", "t/ha": prom_edo, "color": "#bdbdbd"},
            {"Referencia": "Esta parcela (estimado)",
             "t/ha": float(fila["Rendimiento estimado (t/ha)"]), "color": "#004529"},
        ])
        comp_origen = pd.DataFrame(referencias).dropna(subset=["t/ha"])
        fig_origen = go.Figure(go.Bar(
            x=comp_origen["t/ha"],
            y=comp_origen["Referencia"],
            orientation="h",
            marker_color=comp_origen["color"].tolist(),
            text=comp_origen["t/ha"],
            texttemplate="%{x:.2f}",
            textposition="outside",
        ))
        fig_origen.update_yaxes(autorange="reversed")
        estilizar(fig_origen).update_layout(
            height=190, showlegend=False, margin=dict(l=0, r=40, t=4, b=0),
            yaxis_title=None, xaxis_title=None, xaxis_range=[0, VMAX],
        )
        mostrar_grafico(fig_origen)
        st.caption(
            "La estimación combina el promedio municipal con el promedio estatal según la cantidad "
            "de parcelas conocidas. Las variables satelitales y de clima no modifican esta estimación."
        )

        vals = sorted(train.loc[train[MUNI] == muni_sel, Y].round(3).unique())
        if n_muni >= 6 and 0 < len(vals) <= 4:
            txt = ", ".join(f"{v:g}" for v in vals)
            st.warning(
                f"**Particularidad en {muni_sel}:** los rendimientos conocidos solo toman los valores **{txt} t/ha**."
            )
        elif n_muni == 0:
            st.warning(
                "De este municipio no hay parcelas con rendimiento conocido; la estimación usa solo el **promedio de su estado**."
            )
        elif fila["Confianza"] in ("Baja", "Muy baja"):
            st.warning("Pocas parcelas conocidas en este municipio o alta variación entre ellas.")
        else:
            st.success("Información suficiente en este municipio para una estimación sólida.")

# Resumen ejecutivo breve debajo del mapa
if len(pred_df):
    col_pred = "Rendimiento estimado (t/ha)"
    por_muni = pred_df.groupby("Municipio")[col_pred].mean()
    toneladas = pred_df["Toneladas totales (*)"].sum(skipna=True)
    area_tot = pred_df["Área (ha)"].sum(skipna=True)
    pct_alta = (pred_df["Confianza"].isin(["Alta", "Media"])).mean() * 100
    lineas = [
        f"Se estimó el rendimiento de <b>{len(pred_df):,} parcelas</b> en <b>{pred_df['Municipio'].nunique()} municipios</b>, con un promedio esperado de <b>{pred_df[col_pred].mean():.2f} t/ha</b>.",
    ]
    if area_tot > 0:
        lineas.append(f"En las <b>{area_tot:,.0f} ha</b> totales por predecir se esperan alrededor de <b>{toneladas:,.0f} toneladas</b> de cebada.")
    if len(por_muni) > 1:
        lineas.append(
            f"Municipio con mayor rendimiento estimado: <b>{escape(str(por_muni.idxmax()))}</b> ({por_muni.max():.2f} t/ha) · "
            f"Menor: <b>{escape(str(por_muni.idxmin()))}</b> ({por_muni.min():.2f} t/ha)."
        )
    lineas.append(f"El <b>{pct_alta:.0f}%</b> de las predicciones cuenta con confianza Alta o Media.")
    st.markdown(
        "<div class='caja-resumen'><h3>Resumen general de la cosecha estimada</h3><ul>"
        + "".join(f"<li>{x}</li>" for x in lineas)
        + "</ul></div>",
        unsafe_allow_html=True,
    )

# ==============================================================================
# SECCIONES COMPLEMENTARIAS (SOLO 3 PESTAÑAS)
# ==============================================================================
tab_pred, tab_zonas, tab_modelo = st.tabs(
    [
        "Tabla de predicciones y descargas",
        "Análisis por estado y municipio",
        "Cómo funciona y validación",
    ]
)

# ------------------------------------------------------------------------------
# PESTAÑA 1: TABLA DE PREDICCIONES Y DESCARGAS
# ------------------------------------------------------------------------------
with tab_pred:
    if pred_df.empty:
        st.info("No hay parcelas por predecir en los datos cargados.")
    else:
        st.markdown(
            "Filtra por estado, municipio o confianza y descarga los resultados. "
            f"El **rango esperado** indica dónde caerá el rendimiento real en **{nivel} de cada 100** parcelas."
        )
        st.markdown(
            limpiar_html(
                """
                <div class="leyenda">
                <div class="col-exp"><b>Rendimiento estimado</b><span>Toneladas por hectárea (t/ha) esperadas.</span></div>
                <div class="col-exp"><b>Mínimo / Máximo esperado</b><span>Intervalo esperado según el nivel elegido en la barra lateral.</span></div>
                <div class="col-exp"><b>Área (ha)</b><span>Superficie del polígono de la parcela en hectáreas.</span></div>
                <div class="col-exp"><b>Toneladas totales (*)</b><span>Rendimiento × área: cosecha total aproximada.</span></div>
                <div class="col-exp"><b>Parcelas conocidas</b><span>Parcelas con dato real en ese mismo municipio.</span></div>
                <div class="col-exp"><b>Confianza</b><span>Seguridad de la estimación: Alta, Media, Baja o Muy baja.</span></div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

        f1, f2, f3, f4 = st.columns([1, 1, 1, 1])
        sel_edo = f1.multiselect("Estado", sorted(pred_df["Estado"].unique()), placeholder="Todos")
        base = pred_df[pred_df["Estado"].isin(sel_edo)] if sel_edo else pred_df
        sel_muni = f2.multiselect("Municipio", sorted(base["Municipio"].unique()), placeholder="Todos")
        sel_conf = f3.multiselect("Confianza", ORDEN_CONF, placeholder="Todas")
        buscar = f4.text_input("Buscar ID de parcela", placeholder="Escribe parte del ID")

        vista = base
        if sel_muni:
            vista = vista[vista["Municipio"].isin(sel_muni)]
        if sel_conf:
            vista = vista[vista["Confianza"].isin(sel_conf)]
        if buscar.strip():
            vista = vista[vista["ID parcela"].str.contains(buscar.strip(), case=False, na=False, regex=False)]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Parcelas mostradas", f"{len(vista):,}")
        m2.metric("Rendimiento promedio estimado",
                  f"{vista['Rendimiento estimado (t/ha)'].mean():.2f} t/ha" if len(vista) else "—")
        m3.metric("Rango de estimaciones",
                  f"{vista['Rendimiento estimado (t/ha)'].min():.2f} – {vista['Rendimiento estimado (t/ha)'].max():.2f}"
                  if len(vista) else "—")
        m4.metric("Toneladas totales (*)",
                  f"{vista['Toneladas totales (*)'].sum(skipna=True):,.0f}" if len(vista) else "—")

        if vista.empty:
            st.warning("Ninguna parcela cumple con los filtros elegidos.")
        else:
            vista_tabla = vista.drop(columns=["Latitud", "Longitud"])
            mostrar_tabla(pintar_confianza(formatear(vista_tabla, FORMATO_PRED)))

        d1, d2 = st.columns(2)
        d1.download_button(
            "Descargar tabla filtrada (CSV)",
            vista.to_csv(index=False).encode("utf-8-sig"),
            "predicciones_filtradas.csv",
            "text/csv",
        )
        anexo = R["anexo"].copy()
        d2.download_button(
            "Descargar tabla anexo para entrega (CSV)",
            anexo.to_csv(index=False, float_format="%.4f").encode("utf-8"),
            "TABLA_ANEXO_predicciones.csv",
            "text/csv",
            help="Formato oficial con 4 decimales: ID_parcela, Rendimiento_predicho_ton_ha (todas las parcelas).",
        )

# ------------------------------------------------------------------------------
# PESTAÑA 2: ANALISIS POR ESTADO Y MUNICIPIO
# ------------------------------------------------------------------------------
with tab_zonas:
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.markdown("### Rendimiento promedio por estado")
        est = (
            train.groupby(EDO)[Y].agg(["mean", "count"]).reset_index()
            .rename(columns={"mean": "Promedio (t/ha)", "count": "Parcelas"})
            .sort_values("Promedio (t/ha)")
        )
        fig = px.bar(est, x="Promedio (t/ha)", y=EDO, orientation="h", text="Promedio (t/ha)",
                     hover_data={"Parcelas": True}, color_discrete_sequence=[VERDE])
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        estilizar(fig).update_layout(yaxis_title=None, height=max(260, 60 * len(est)))
        redondear_barras(fig)
        mostrar_grafico(fig)

    with col_e2:
        if len(pred_df) and pred_df["Toneladas totales (*)"].notna().any():
            st.markdown("### Cosecha total esperada por municipio (Top 12)")
            tt = (pred_df.groupby("Municipio")["Toneladas totales (*)"].sum().sort_values(ascending=False)
                  .head(12).iloc[::-1].reset_index())
            fig = px.bar(tt, x="Toneladas totales (*)", y="Municipio", orientation="h",
                         text="Toneladas totales (*)", color_discrete_sequence=[DORADO])
            fig.update_traces(texttemplate="%{x:,.0f}", textposition="outside")
            estilizar(fig).update_layout(yaxis_title=None, xaxis_title="Toneladas esperadas (aprox.)",
                                         height=max(260, 28 * len(tt)))
            redondear_barras(fig)
            mostrar_grafico(fig)

    st.markdown("---")
    st.markdown("### Estimación y nivel de confianza por municipio")
    tm_v = R["tabla_muni"].copy()
    if "confianza" in tm_v.columns:
        tm_v["Confianza"] = tm_v["confianza"]

    cf1, cf2, cf3 = st.columns([1, 1, 2])
    sel_e = cf1.multiselect("Estado", sorted(tm_v[EDO].unique()), placeholder="Todos", key="m_edo")
    if sel_e:
        tm_v = tm_v[tm_v[EDO].isin(sel_e)]
    n_tot = len(tm_v)
    if n_tot > 0:
        top = cf2.slider("Municipios a mostrar", 1, min(60, n_tot), min(20, n_tot)) if n_tot > 1 else 1
        orden = cf3.radio("Ordenar por", ["Mayor rendimiento", "Menor rendimiento", "Más parcelas conocidas"], horizontal=True)
        if orden == "Mayor rendimiento":
            tm_v = tm_v.sort_values("estimacion", ascending=False)
        elif orden == "Menor rendimiento":
            tm_v = tm_v.sort_values("estimacion", ascending=True)
        else:
            tm_v = tm_v.sort_values("n", ascending=False)
        tm_v = tm_v.head(top)

        tm_graf = tm_v.iloc[::-1]
        fig = px.bar(
            tm_graf, x="estimacion", y=MUNI, orientation="h", color="Confianza", text="estimacion",
            color_discrete_map={"Alta": VERDE, "Media": DORADO, "Baja": "#FB8C00", "Muy baja": ROJO},
            category_orders={"Confianza": ORDEN_CONF},
            hover_data={"n": True, "prom_obs": ":.2f", "media_estado": ":.2f", EDO: True, "estimacion": ":.2f"},
            labels={"estimacion": "Rendimiento estimado (t/ha)", "n": "Parcelas conocidas",
                    "prom_obs": "Promedio observado", "media_estado": "Promedio del estado"},
        )
        fig.update_traces(texttemplate="%{x:.2f}", textposition="outside")
        estilizar(fig).update_layout(yaxis_title=None, height=max(340, 28 * len(tm_graf)))
        redondear_barras(fig)
        mostrar_grafico(fig)

        st.markdown("#### Variación interna en los municipios mostrados")
        munis_graf = tm_v[MUNI].tolist()[:12]
        sub = train[train[MUNI].isin(munis_graf)]
        if len(sub):
            fig = px.box(sub, x=MUNI, y=Y, points="all", color_discrete_sequence=[VERDE],
                         category_orders={MUNI: munis_graf}, labels={Y: "Rendimiento (t/ha)", MUNI: ""})
            fig.add_trace(go.Scatter(x=tm_v[MUNI], y=tm_v["estimacion"], mode="markers", name="Estimación del modelo",
                                     marker=dict(symbol="diamond", size=11)))
            estilizar(fig)
            mostrar_grafico(fig)
# ------------------------------------------------------------------------------
# PESTAÑA 3: COMO FUNCIONA Y VALIDACION
# ------------------------------------------------------------------------------
with tab_modelo:
    st.markdown("### ¿Qué tan confiable es el modelo?")
    st.markdown(
        "Para evaluar el modelo de forma realista, se esconde el rendimiento de parcelas conocidas, "
        "se predice su valor y se mide el error cometido mediante la validación del primer programa."
    )

    if error_rel < 10:
        veredicto = "Muy bueno"
    elif error_rel < 20:
        veredicto = "Bueno"
    elif error_rel < 35:
        veredicto = "Moderado"
    else:
        veredicto = "Limitado"

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Error promedio (MAE)", f"{M['MAE']:.2f} t/ha")
    q2.metric("Error relativo", f"{error_rel:.1f}%")
    q3.metric("R² (variación explicada)", f"{M['R2'] * 100:.0f}%")
    q4.metric("Valoración", veredicto)

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("#### Real vs. estimado")
        dfv = pd.DataFrame({"Real": val["y"], "Estimado": val["pred_cv"], MUNI: val["municipio"]})
        fig = px.scatter(dfv, x="Real", y="Estimado", opacity=0.55, hover_name=MUNI,
                         color_discrete_sequence=[VERDE])
        mn = float(min(dfv.Real.min(), dfv.Estimado.min()))
        mx = float(max(dfv.Real.max(), dfv.Estimado.max()))
        fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines", name="Acierto perfecto",
                                 line=dict(color=ROJO, dash="dash")))
        estilizar(fig).update_layout(xaxis_title="Rendimiento real (t/ha)",
                                     yaxis_title="Rendimiento estimado (t/ha)",
                                     legend=dict(orientation="h", y=-0.25))
        mostrar_grafico(fig)
    with g2:
        st.markdown("#### Distribución de los errores")
        lo_r, hi_r = M["rango_global"]
        fig = px.histogram(x=val["residuos"], nbins=40, color_discrete_sequence=[DORADO])
        fig.add_vrect(x0=lo_r, x1=hi_r, fillcolor=VERDE, opacity=0.12, line_width=0,
                      annotation_text="80% de los casos", annotation_position="top left")
        fig.add_vline(x=0, line_color=ROJO, line_dash="dash")
        estilizar(fig).update_layout(xaxis_title="Error (real − estimado, t/ha)", yaxis_title="Parcelas")
        mostrar_grafico(fig)

    st.markdown("---")
    st.markdown("### Cómo calcula el modelo cada estimación (Simulador)")
    st.markdown(
        "El modelo combina el promedio del municipio con el promedio del estado según la cantidad de parcelas conocidas."
    )
    st.latex(r"\text{estimación} = \frac{n \cdot \text{prom. municipio} + k \cdot \text{prom. estado}}{n + k}")

    s1, s2 = st.columns(2)
    n_demo = s1.slider("Parcelas conocidas en el municipio (n)", 0, 50, 3, key="demo_n")
    m_demo = s1.slider("Promedio del municipio (t/ha)", 1.0, 8.0, 4.5, 0.1, key="demo_m")
    e_demo = s2.slider("Promedio del estado (t/ha)", 1.0, 8.0, 4.0, 0.1, key="demo_e")
    k_demo = s2.slider("Fuerza de contracción k", 0.0, 10.0, 1.0, 0.5, key="demo_k")

    if n_demo == 0:
        resultado, peso_m = e_demo, 0.0
    elif (n_demo + k_demo) == 0:
        resultado, peso_m = m_demo, 1.0
    else:
        resultado = (n_demo * m_demo + k_demo * e_demo) / (n_demo + k_demo)
        peso_m = n_demo / (n_demo + k_demo)

    r1, r2c, r3 = st.columns(3)
    r1.metric("Estimación resultante", f"{resultado:.2f} t/ha")
    r2c.metric("Peso del municipio", f"{peso_m * 100:.0f}%")
    r3.metric("Peso del estado", f"{(1 - peso_m) * 100:.0f}%")
    st.progress(float(peso_m), text=f"{peso_m * 100:.0f}% municipio · {(1 - peso_m) * 100:.0f}% estado")

    with st.expander("Validación y limitaciones"):
        st.markdown(
            f"""
- **k oficial = 1:** las predicciones de entrega no se modifican desde la interfaz.
- **MAE:** {M['MAE']:.3f} t/ha · **RMSE:** {M['RMSE']:.3f} t/ha · **R²:** {M['R2']:.3f}.
- **Rango esperado:** intervalo empírico oficial del primer programa; las predicciones de entrega usan 80%.
- **Datos del ciclo {CICLO}:** para otro ciclo se deben recalcular los promedios.
- **t/ha:** toneladas cosechadas por cada hectárea.
            """
        )

# ------------------------------------------------------------------------------
# Pie de pagina
# ------------------------------------------------------------------------------
st.markdown(
    f'<div class="footer-app">AgroCebada · Reto FIRA 2026 — modelo jerárquico municipio → estado · '
    f"datos del ciclo {CICLO} · error típico ±{mae:.2f} t/ha</div>",
    unsafe_allow_html=True,
)