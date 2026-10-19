"""
================================================================================
AgroCebada FIRA 2026 - Cosecha estimada de cebada por parcela (Streamlit)
================================================================================
Version unificada: diseno renovado, mapa satelital (Folium), descripciones
sencillas para el productor y seccion "Que significa para el productor" como
elemento principal de la ficha. Adaptada tambien para pantallas de celular.

Predicciones, validacion, confianza y rangos se calculan exclusivamente con
calculos.py (modelo jerarquico municipio -> estado, k = 1).

Archivos en la MISMA carpeta:
    app_agrocebada.py         esta app
    calculos.py               calculos oficiales (validacion, confianza, rangos)
    modelo_agrocebada.py      el modelo entregado
    dataset_maestro.csv       ID, estado, municipio, area y rendimiento
    parcelas_centroides.csv   centro de cada parcela (lon, lat)
    parcelas.geojson          OPCIONAL: poligonos reales

Ejecutar:
    pip install -r requirements.txt
    streamlit run app_agrocebada.py
================================================================================
"""
import json
import os
from html import escape

import branca.colormap as bcm
import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

import calculos as C
from calculos import ID, MUNI, EDO, Y

st.set_page_config(page_title="AgroCebada FIRA 2026", page_icon="🌾", layout="wide")

CICLO = "2025"
VMIN, VMAX = 2.0, 5.6  # misma escala de color para estimado y real
COLORES = ["#d73027", "#fc8d59", "#fee08b", "#91cf60", "#1a9850"]  # bajo -> alto
ORDEN_CONF = ["Alta", "Media", "Baja", "Muy baja"]
CLASE_CONF = {"Alta": "alta", "Media": "media", "Baja": "baja", "Muy baja": "muybaja"}
T_EST = "Con estimación"
T_REAL = "Ya cosechada y medida"
VERDE = "#2E7D32"
DORADO = "#C8A415"
GRIS = "#9E9E9E"
ROJO = "#C62828"

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem;}
    @media (max-width: 700px) {
      .block-container {padding: .7rem .75rem 1.5rem .75rem;}
      .hero {padding: 12px 14px;}
      .hero h1 {font-size: 1.22rem; line-height: 1.25;}
      .hero p {font-size: .88rem;}
      div[data-testid="stHorizontalBlock"] {flex-wrap: wrap; gap: .6rem;}
      div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {min-width: 100% !important; flex: 1 1 100% !important;}
    }
    div[data-testid="stMetric"] {
        background-color: rgba(46,125,50,0.07);
        border-left: 5px solid #2E7D32;
        padding: 10px 14px;
        border-radius: 10px;
    }
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] > div {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        overflow-wrap: anywhere;
    }
    div[data-testid="stNumberInput"] {width: 100% !important; max-width: none !important;}
    div[data-testid="stNumberInput"] > div {width: 100% !important; max-width: none !important;}
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

    .prod-titulo {background: linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%);
                  color:#fff; border-radius:12px 12px 0 0; padding:12px 18px; margin-top:10px;}
    .prod-titulo h3 {margin:0; font-size:1.15rem; color:#fff;}
    .prod-titulo p {margin:2px 0 0; font-size:.85rem; color:#E8F5E9;}
    .prod-texto {background:#F1F8E9; border:1px solid #C5E1A5; border-top:none;
                 padding:14px 18px; color:#263238; font-size:1rem; line-height:1.55;}
    .prod-texto b {color:#1B5E20;}
    .prod-pie {background:#FFF9E6; border:1px solid #FFE082; border-top:none;
               border-radius:0 0 12px 12px; padding:10px 18px; color:#5D4A00;
               font-size:.85rem; line-height:1.45; margin-bottom:10px;}

    div[data-testid="stPlotlyChart"] .js-plotly-plot g.axis text,
    div[data-testid="stPlotlyChart"] .js-plotly-plot .ticktext {
        fill: var(--text-color, #37474F) !important;
    }

    .conf-lista {display:flex; flex-direction:column; gap:8px; margin-top:6px;}
    .conf-item {display:flex; align-items:center; gap:10px; min-height:58px;
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


def chip(conf) -> str:
    clase = CLASE_CONF.get(conf, "media")
    return f'<span class="chip chip-{clase}">{escape(str(conf))}</span>'


def mostrar_grafico(fig, **extra):
    for kwargs in (
        dict(width="stretch", theme="streamlit"),
        dict(use_container_width=True, theme="streamlit"),
        dict(width="stretch"),
        dict(use_container_width=True),
    ):
        try:
            st.plotly_chart(fig, **kwargs, **extra)
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


# ------------------------------------------------------------------------------
# Datos y modelo oficial (calculos.py; se calculan una sola vez)
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
    except Exception:  # noqa: BLE001
        return None


R, error = preparar()
if error:
    st.title("AgroCebada FIRA 2026")
    st.error(error)
    st.stop()

train, test, tm, M = R["train"], R["test"], R["municipios"], R["metricas"]
info_muni = tm.set_index(MUNI)
poligonos = leer_geojson()

mae = float(M["MAE"])
error_rel = float(M["error_rel"]) * 100
error_rel_txt = f"{error_rel:.1f}%" if np.isfinite(error_rel) else "—"
media_global = float(train[Y].mean())
medias_estado = {str(e): float(g[Y].mean()) for e, g in train.groupby(EDO)}
n_muni_dict = dict(zip(tm[MUNI], tm["n"].astype(int)))

# Tabla unica de parcelas para el mapa y la ficha
parcelas = pd.concat([
    test.assign(tipo=T_EST, valor=test["pred"], ton_val=pd.to_numeric(test["toneladas"], errors="coerce")),
    train.assign(tipo=T_REAL, valor=train[Y], ton_val=train[Y] * pd.to_numeric(train["area_ha"], errors="coerce")),
], ignore_index=True)

# Tabla de predicciones para la pestana de resultados
if len(test):
    pred_df = pd.DataFrame({
        "ID parcela": test[ID].astype(str).values,
        "Municipio": test[MUNI].values,
        "Estado": test[EDO].values,
        "Rendimiento estimado (t/ha)": np.round(test["pred"].astype(float).values, 4),
        "Mínimo esperado (t/ha)": np.round(test["min_esp"].astype(float).values, 4),
        "Máximo esperado (t/ha)": np.round(test["max_esp"].astype(float).values, 4),
        "Área (ha)": pd.to_numeric(test["area_ha"], errors="coerce").values,
        "Toneladas totales (*)": pd.to_numeric(test["toneladas"], errors="coerce").round(2).values,
        "Margen de error (± t)": pd.to_numeric(test["error_ton"], errors="coerce").round(2).values,
        "Parcelas conocidas en el municipio": test[MUNI].map(n_muni_dict).fillna(0).astype(int).values,
        "Confianza": test["confianza"].values,
    })
else:
    pred_df = pd.DataFrame()

FORMATO_PRED = {
    "Rendimiento estimado (t/ha)": "{:.2f}",
    "Mínimo esperado (t/ha)": "{:.2f}",
    "Máximo esperado (t/ha)": "{:.2f}",
    "Área (ha)": "{:.2f}",
    "Toneladas totales (*)": "{:,.1f}",
    "Margen de error (± t)": "{:,.1f}",
}

# ------------------------------------------------------------------------------
# Barra lateral
# ------------------------------------------------------------------------------
st.sidebar.title("AgroCebada")
st.sidebar.caption(f"Reto AgroCebada FIRA 2026 · ciclo {CICLO}")
items_conf = "".join(
    f'<div class="conf-item">{chip(c)}<span class="conf-desc">{escape(C.EXPLICACION_CONF[c])}</span></div>'
    for c in ORDEN_CONF
)
st.sidebar.markdown(
    limpiar_html(f'<div style="margin-top:4px;"><b>¿Qué tan segura es la estimación?</b></div>'
                 f'<div class="conf-lista">{items_conf}</div>'),
    unsafe_allow_html=True,
)
st.sidebar.caption("La confianza depende de qué tanto se equivocó el modelo en cada municipio al revisarlo con "
                   "parcelas ya medidas, no de cuántas parcelas conocidas tiene.")

# ------------------------------------------------------------------------------
# Encabezado y cifras clave
# ------------------------------------------------------------------------------
st.markdown(
    limpiar_html(
        """
        <div class="hero">
        <h1>AgroCebada · Cosecha estimada</h1>
        <p>Encuentra tu parcela y consulta cuántas toneladas podrías cosechar.</p>
        </div>
        """
    ),
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Parcelas por estimar", f"{len(test):,}", help="Parcelas nuevas a las que el modelo les estima la cosecha.")
c2.metric("Parcelas con dato real", f"{len(train):,}",
          help="Parcelas cuyo rendimiento ya se conoce; con ellas se entrenó y revisó el modelo.")
c3.metric("Rendimiento promedio", f"{media_global:.2f} t/ha",
          help="Promedio de las parcelas conocidas; es la referencia general.")
c4.metric("Qué tanto se equivoca en promedio", f"± {mae:.2f} t/ha", f"≈ {error_rel_txt} del promedio",
          delta_color="off", delta_arrow="off",
          help="En promedio, la estimación se aleja del valor real por esta cantidad. No es un 'porcentaje de acierto'.")

desconocidos = sorted(set(test.loc[~test["conocido"], MUNI])) if len(test) else []
if desconocidos:
    st.error("**Advertencia:** estas parcelas están en municipios sin parcelas conocidas, así que usan el promedio "
             f"de su estado (no está validado; el error esperado es de ~0.5 t/ha o más): {', '.join(desconocidos)}.")

# ==============================================================================
# VISTA PRINCIPAL: MAPA SATELITAL + FICHA DE PARCELA / PRODUCTOR
# ==============================================================================
st.markdown("---")
st.info("**Cómo usarlo:** 1) Elige tu estado y municipio.  2) Toca tu parcela en el mapa o búscala en la lista.  "
        "3) Revisa cuántas toneladas podrías cosechar y cuánto podría variar.")

f1, f2, f3 = st.columns([1, 2, 1.2], vertical_alignment="bottom")
sel_edo_m = f1.multiselect("Estado", sorted(parcelas[EDO].unique()), placeholder="Todos", key="mapa_edo")
base = parcelas[parcelas[EDO].isin(sel_edo_m)] if sel_edo_m else parcelas
sel_muni_m = f2.multiselect("Municipio", sorted(base[MUNI].unique()), placeholder="Todos", key="mapa_muni")
if sel_muni_m:
    base = base[base[MUNI].isin(sel_muni_m)]
ver_reales = f3.toggle("Incluir parcelas ya cosechadas y medidas", value=True,
                       help="Ayudan a ver cuánto cambia el rendimiento de una parcela a otra dentro de cada municipio.")
visibles = base if ver_reales else base[base["tipo"] == T_EST]

if "parcela_sel" not in st.session_state:
    st.session_state["parcela_sel"] = sorted(test[ID])[0] if len(test) else sorted(train[ID])[0]
# Si los filtros dejan fuera a la parcela elegida, se elige la primera visible con estimacion
est_visibles = sorted(visibles.loc[visibles["tipo"] == T_EST, ID])
if st.session_state["parcela_sel"] not in set(visibles[ID]) and est_visibles:
    st.session_state["parcela_sel"] = est_visibles[0]

# Parcela elegida: la usan el bloque del productor (bajo el mapa) y la ficha (a la derecha)
_sel_df = parcelas.loc[parcelas[ID] == st.session_state["parcela_sel"]]
fila_sel = _sel_df.iloc[0] if len(_sel_df) else None

col_mapa, col_ficha = st.columns([1.45, 1], gap="large")

# ---------------------------------- mapa ----------------------------------
with col_mapa:
    st.subheader("Mapa de rendimiento por parcela")
    escala = bcm.LinearColormap(COLORES, vmin=VMIN, vmax=VMAX, caption="Rendimiento (t/ha)")

    def feature(fila, geometria):
        est = fila["tipo"] == T_EST
        ton_txt = f"{fila['ton_val']:,.1f} toneladas" if pd.notna(fila["ton_val"]) else "—"
        return {
            "type": "Feature", "geometry": geometria,
            "properties": {
                "ID": fila[ID], "Tipo": fila["tipo"], "Municipio": fila[MUNI],
                "Rendimiento": f"{fila['valor']:.2f} t/ha" + (" (estimado)" if est else " (real)"),
                "Cosecha": ton_txt + (" (estimada)" if est else " (real)"),
                "valor": float(fila["valor"]), "estimar": est,
            },
        }

    puntos = [feature(f, {"type": "Point", "coordinates": [f["lon"], f["lat"]]})
              for _, f in visibles.dropna(subset=["lon", "lat"]).iterrows()]

    def estilo(feat):
        p = feat["properties"]
        return {
            "fillColor": escala(min(max(p["valor"], VMIN), VMAX)),
            "color": "#1b1b1b" if p["estimar"] else "#6b6b6b",
            "weight": 2 if p["estimar"] else 1,
            "fillOpacity": 0.9 if p["estimar"] else 0.45,
            "dashArray": None if p["estimar"] else "3",
            "radius": 7 if p["estimar"] else 4,
        }

    campos_tt = dict(fields=["ID", "Tipo", "Municipio", "Cosecha", "Rendimiento"],
                     aliases=["Parcela", "", "Municipio", "Cosecha", "Rendimiento"])

    m = folium.Map(tiles=None, control_scale=True)
    folium.TileLayer("OpenStreetMap", name="Calles").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, Maxar, Earthstar Geographics", name="Satélite").add_to(m)

    if poligonos:
        poli = [feature(f, poligonos[f[ID]]) for _, f in visibles.iterrows() if f[ID] in poligonos]
        folium.GeoJson({"type": "FeatureCollection", "features": poli}, name="Parcelas (forma real)",
                       style_function=estilo, tooltip=folium.GeoJsonTooltip(**campos_tt),
                       highlight_function=lambda _: {"weight": 4, "color": "#d62728"}).add_to(m)

    folium.GeoJson({"type": "FeatureCollection", "features": puntos}, name="Parcelas (punto)",
                   marker=folium.CircleMarker(fill=True), style_function=estilo,
                   tooltip=folium.GeoJsonTooltip(**campos_tt),
                   highlight_function=lambda _: {"weight": 4, "color": "#d62728"}).add_to(m)

    escala.add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    geo = visibles.dropna(subset=["lon", "lat"])
    if len(geo):
        pad = 0.01
        m.fit_bounds([[geo["lat"].min() - pad, geo["lon"].min() - pad],
                      [geo["lat"].max() + pad, geo["lon"].max() + pad]])

    # La parcela elegida se dibuja en una capa aparte para no recargar todo el mapa
    resaltado = folium.FeatureGroup(name="Parcela elegida")
    sel = parcelas.loc[parcelas[ID] == st.session_state["parcela_sel"]]
    if len(sel) and pd.notna(sel["lat"].iloc[0]):
        folium.CircleMarker([sel["lat"].iloc[0], sel["lon"].iloc[0]], radius=14, color="#2b6cb0",
                            weight=4, fill=False, tooltip=f"Elegida: {sel[ID].iloc[0]}").add_to(resaltado)

    salida = st_folium(m, key="mapa", height=600, use_container_width=True,
                       returned_objects=["last_active_drawing"], feature_group_to_add=resaltado)

    clic = (salida or {}).get("last_active_drawing") or {}
    clic_id = (clic.get("properties") or {}).get("ID")
    if clic_id and clic_id != st.session_state.get("_ultimo_clic"):
        st.session_state["_ultimo_clic"] = clic_id
        st.session_state["parcela_sel"] = clic_id
        st.rerun()

    st.caption(
        "**Leyenda:** rojo = rinde menos · verde = rinde más. Las parcelas **con estimación** tienen borde oscuro; "
        "las **ya cosechadas y medidas**, borde punteado y color más tenue. La parcela elegida se marca con un "
        "círculo azul. Usa el control de capas del mapa (arriba a la derecha) para cambiar entre calles y satélite. "
        "Todas las parcelas de un municipio comparten la misma estimación: el modelo combina el promedio del "
        "municipio con el del estado."
        + ("" if poligonos else " · Se muestran puntos porque falta `parcelas.geojson` para dibujar la forma real de las parcelas.")
    )

    # ----------------------------------------------------------------------
    # QUE SIGNIFICA PARA EL PRODUCTOR (seccion prioritaria, bajo el mapa)
    # ----------------------------------------------------------------------
    if fila_sel is not None and fila_sel["tipo"] == T_EST:
        fp = fila_sel
        muni_p, edo_p = fp[MUNI], fp[EDO]
        area = float(fp["area_ha"])
        ton, err = float(fp["toneladas"]), float(fp["error_ton"])
        ton_min = float(fp["min_esp"]) * area
        ton_max = float(fp["max_esp"]) * area

        st.markdown(
            limpiar_html(
                f"""
                <div class="prod-titulo">
                <h3>¿Qué significa para el productor?</h3>
                <p>Parcela {escape(str(fp[ID]))} · {escape(str(muni_p))}, {escape(str(edo_p))} · tu estimación traducida a toneladas e ingreso.</p>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            limpiar_html(
                f"""
                <div class="prod-texto">
                Con <b>{area:.2f} ha</b>, podrías cosechar unas <b>{ton:,.1f} toneladas</b> de cebada.
                En {escape(str(muni_p))} el modelo se ha equivocado en promedio <b>{fp['mae_val']:.2f} t/ha</b>,
                es decir, unas <b>±{err:,.1f} toneladas</b> en una parcela de este tamaño.
                En 8 de cada 10 casos la cosecha quedaría entre
                <b>{ton_min:,.1f}</b> y <b>{ton_max:,.1f} toneladas</b>.
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

        p1, p2, p3 = st.columns(3)
        p1.metric("¿Cuántas toneladas podrías cosechar?", f"≈ {ton:,.1f} toneladas",
                  f"Superficie registrada: {area:.2f} ha", delta_color="off", delta_arrow="off")
        p2.metric("¿Cuánto podría desviarse?", f"± {err:,.1f} toneladas",
                  f"Error típico de {muni_p}", delta_color="off", delta_arrow="off")
        p3.metric("Rango probable (8 de cada 10 casos)", f"{ton_min:,.1f} – {ton_max:,.1f} t")

        with st.container(border=True):
            st.markdown("**Calculadora de ingreso**")
            precio = st.number_input(
                "Precio por tonelada (MXN)", min_value=0.0, value=0.0, step=100.0, key="precio_ton",
                help="Escribe el precio de contrato o de mercado que quieras usar para convertir a pesos.")
            if precio > 0:
                i1, i2 = st.columns(2)
                i1.metric("Ingreso esperado", f"${ton * precio:,.0f}")
                i2.metric("Margen de error", f"± ${err * precio:,.0f}")
                st.metric("Ingreso mínimo – máximo", f"\${ton_min * precio:,.0f} - ${ton_max * precio:,.0f}")
            else:
                st.caption("Escribe un precio por tonelada para ver cuánto podrías ingresar en pesos.")

        st.markdown(
            limpiar_html(
                """
                <div class="prod-pie">
                Útil para planear la venta o pedir un crédito. No considera tu variedad, fecha de siembra ni
                daños por granizo o helada, y no garantiza el rendimiento de una parcela en particular.
                Las toneladas se calculan con la superficie registrada, que puede no ser la sembrada.
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
    elif fila_sel is not None:
        st.info("La parcela elegida ya fue cosechada y medida. Elige una parcela **con estimación** en el mapa o en "
                "la lista para ver qué significa para el productor.")

# ---------------------------------- ficha ----------------------------------
with col_ficha:
    st.subheader("Tu ficha de cosecha")
    opciones = sorted(base.loc[base["tipo"] == T_EST, ID])
    if not opciones:
        st.info("No hay parcelas con estimación con estos filtros.")

    def _desde_lista():
        st.session_state["parcela_sel"] = st.session_state["sel_lista"]

    if opciones:
        if st.session_state.get("parcela_sel") in opciones:
            st.session_state["sel_lista"] = st.session_state["parcela_sel"]
        elif st.session_state.get("sel_lista") not in opciones:
            st.session_state["sel_lista"] = opciones[0]
        st.selectbox("Parcela", opciones, key="sel_lista", on_change=_desde_lista,
                     help="También puedes tocarla en el mapa.")
        if st.session_state.get("parcela_sel") not in set(parcelas[ID]):
            st.session_state["parcela_sel"] = st.session_state["sel_lista"]

        fila = parcelas.loc[parcelas[ID] == st.session_state["parcela_sel"]].iloc[0]
        muni, edo = fila[MUNI], fila[EDO]
        im = info_muni.loc[muni] if muni in info_muni.index else None

        st.markdown(f"#### Parcela {fila[ID]}")
        st.caption(f"{muni}, {edo} · {fila['area_ha']:.2f} ha (superficie del terreno registrado)")

        if fila["tipo"] == T_REAL:
            st.info("Esta parcela **ya fue cosechada y medida**: su rendimiento real se conoce y se usó para "
                    "ajustar el modelo.")
            a, b = st.columns(2)
            a.metric("Rendimiento real", f"{fila[Y]:.2f} t/ha")
            if pd.notna(fila["ton_val"]):
                b.metric("Cosecha real aproximada", f"{fila['ton_val']:,.1f} toneladas")
            if im is not None:
                dif = fila[Y] - im["estimacion"]
                st.metric("Estimación para su municipio", f"{im['estimacion']:.2f} t/ha",
                          f"{dif:+.2f} t/ha de diferencia con esta parcela", delta_color="off", delta_arrow="off")
            st.caption("La diferencia muestra cuánto puede alejarse una parcela del promedio de su zona.")
        else:
            st.caption("Las toneladas y el ingreso esperado de esta parcela están en la sección "
                       "**¿Qué significa para el productor?**, debajo del mapa.")

            # ------------------------------------------------------------------
            # Rendimiento, rango y confianza
            # ------------------------------------------------------------------
            n_muni = int(im["n"]) if im is not None else 0
            m1, m2 = st.columns(2)
            m1.metric("Rendimiento estimado", f"{fila['pred']:.2f} t/ha", f"valor oficial {fila['pred']:.4f}",
                      delta_color="off", delta_arrow="off")
            m2.metric("Rango esperado (8 de cada 10)", f"{fila['min_esp']:.2f} – {fila['max_esp']:.2f} t/ha")
            st.markdown(
                f"**Confianza en la zona:** {chip(fila['confianza'])}  \n"
                f"{escape(C.EXPLICACION_CONF[fila['confianza']])} ({n_muni} parcelas conocidas en {escape(str(muni))}).",
                unsafe_allow_html=True,
            )

            # ------------------------------------------------------------------
            # De donde sale el numero
            # ------------------------------------------------------------------
            st.markdown("##### ¿De dónde sale este número?")
            if im is not None:
                pm = float(im["peso_muni"])
                st.progress(pm, text=f"{pm * 100:.1f}% promedio de {muni} · {(1 - pm) * 100:.1f}% promedio de {edo}")
                comp = pd.DataFrame({
                    "Referencia": ["Esta parcela (estimado)", f"Promedio de {muni} ({int(im['n'])} parcelas)",
                                   f"Promedio de {edo}", "Promedio general"],
                    "t/ha": [fila["pred"], im["prom_obs"], medias_estado.get(str(edo), media_global), media_global],
                    "color": [VERDE, DORADO, GRIS, "#CFD8DC"],
                }).dropna(subset=["t/ha"])
                fig_comp = go.Figure(go.Bar(
                    x=comp["t/ha"], y=comp["Referencia"], orientation="h",
                    marker_color=comp["color"].tolist(), text=comp["t/ha"].round(2), textposition="outside",
                ))
                fig_comp.update_yaxes(autorange="reversed")
                estilizar(fig_comp).update_layout(height=230, xaxis_title="Rendimiento (t/ha)",
                                                  xaxis_range=[0, VMAX + 0.8])
                redondear_barras(fig_comp)
                mostrar_grafico(fig_comp, config={"displayModeBar": False})
                st.caption("La estimación mezcla el promedio del municipio con el del estado, según cuántas "
                           "parcelas se conocen. Las variables de satélite y clima no cambian este número.")

                if im["rango_propio"]:
                    st.caption(f"El rango sale de cómo se equivocó el modelo con las {int(im['n'])} parcelas "
                               f"conocidas de {muni}. Aplica a cada parcela, no al promedio de la zona.")
                else:
                    st.caption(f"{muni} tiene solo {int(im['n'])} parcela{'s' if im['n'] != 1 else ''} "
                               f"conocida{'s' if im['n'] != 1 else ''}; se muestra el rango general del modelo. "
                               "Aplica a cada parcela, no al promedio de la zona.")
            else:
                st.warning("Este municipio no tiene parcelas conocidas: se usa solo el promedio del estado.")

            # Avisos especiales
            if im is not None and im["n"] >= 6 and im["valores_distintos"] <= 4:
                vals = ", ".join(f"{v:g}" for v in sorted(train.loc[train[MUNI] == muni, Y].round(2).unique()))
                st.warning(f"En {muni} los rendimientos reportados solo toman los valores **{vals} t/ha**. "
                           "El rendimiento real de cada parcela probablemente sea uno de esos valores, "
                           "así que el error será de 0 o de un escalón completo.")
            if im is not None and im["n"] < 3:
                st.warning(f"{muni} tiene solo {int(im['n'])} parcela{'s' if im['n'] != 1 else ''} "
                           f"conocida{'s' if im['n'] != 1 else ''}: la estimación se apoya "
                           f"en {(1 - im['peso_muni']) * 100:.0f}% en el promedio del estado.")
            elif fila["confianza"] in ("Baja", "Muy baja"):
                st.warning("Las parcelas de este municipio rinden de forma muy distinta entre sí: la estimación es "
                           "menos segura para una parcela en particular.")
            elif fila["confianza"] in ("Alta", "Media"):
                st.success("Hay información suficiente en este municipio para una estimación sólida.")

    if len(test):
        st.markdown("**Descargas para técnicos**")
        dl1, dl2 = st.columns(2)
        dl1.download_button(
            "Tabla oficial de predicciones", C.anexo_csv(R["anexo"]),
            "TABLA_ANEXO_predicciones.csv", "text/csv",
            help=f"ID_parcela y rendimiento predicho en t/ha con 4 decimales ({len(test)} parcelas). Es la tabla de entrega.",
            width="stretch",
        )
        detalle = test[[ID, EDO, MUNI, "area_ha", "pred", "min_esp", "max_esp", "confianza"]].rename(columns={
            ID: "ID_parcela", "area_ha": "Area_ha", "pred": "Rendimiento_ton_ha",
            "min_esp": "Minimo_esperado_80", "max_esp": "Maximo_esperado_80", "confianza": "Confianza"})
        dl2.download_button(
            "Detalle, rangos y confianza",
            detalle.sort_values("ID_parcela").to_csv(index=False, float_format="%.4f").encode("utf-8-sig"),
            "predicciones_detalle.csv", "text/csv", width="stretch",
        )

# ------------------------------------------------------------------------------
# Resumen general
# ------------------------------------------------------------------------------
if len(pred_df):
    col_pred = "Rendimiento estimado (t/ha)"
    por_muni = pred_df.groupby("Municipio")[col_pred].mean()
    toneladas = pred_df["Toneladas totales (*)"].sum(skipna=True)
    area_tot = pred_df["Área (ha)"].sum(skipna=True)
    pct_alta = (pred_df["Confianza"].isin(["Alta", "Media"])).mean() * 100
    lineas = [
        f"Se estimó el rendimiento de <b>{len(pred_df):,} parcelas</b> en <b>{pred_df['Municipio'].nunique()} "
        f"municipios</b>, con un promedio esperado de <b>{pred_df[col_pred].mean():.2f} t/ha</b>.",
    ]
    if area_tot > 0:
        lineas.append(f"En las <b>{area_tot:,.0f} ha</b> totales se esperan alrededor de "
                      f"<b>{toneladas:,.0f} toneladas</b> de cebada.")
    if len(por_muni) > 1:
        lineas.append(
            f"Municipio con mayor rendimiento estimado: <b>{escape(str(por_muni.idxmax()))}</b> ({por_muni.max():.2f} t/ha) · "
            f"Menor: <b>{escape(str(por_muni.idxmin()))}</b> ({por_muni.min():.2f} t/ha)."
        )
    lineas.append(f"El <b>{pct_alta:.0f}%</b> de las estimaciones tiene confianza Alta o Media.")
    st.markdown(
        "<div class='caja-resumen'><h3>Resumen general de la cosecha estimada</h3><ul>"
        + "".join(f"<li>{x}</li>" for x in lineas)
        + "</ul></div>",
        unsafe_allow_html=True,
    )

# ==============================================================================
# SECCIONES COMPLEMENTARIAS
# ==============================================================================
tab_pred, tab_zonas, tab_modelo = st.tabs(
    ["Tabla de predicciones y descargas", "Análisis por estado y municipio", "Para técnicos: cómo funciona el modelo"]
)

# ------------------------------------------------------------------------------
# PESTANA 1: TABLA DE PREDICCIONES Y DESCARGAS
# ------------------------------------------------------------------------------
with tab_pred:
    if pred_df.empty:
        st.info("No hay parcelas por estimar en los datos cargados.")
    else:
        st.markdown(
            "Filtra por estado, municipio o confianza y descarga los resultados. "
            "El **rango esperado** indica dónde caería el rendimiento real en **8 de cada 10** parcelas."
        )
        st.markdown(
            limpiar_html(
                """
                <div class="leyenda">
                <div class="col-exp"><b>Rendimiento estimado</b><span>Toneladas por hectárea (t/ha) que se esperan.</span></div>
                <div class="col-exp"><b>Mínimo / Máximo esperado</b><span>Rango en el que caería el rendimiento real en 8 de cada 10 casos.</span></div>
                <div class="col-exp"><b>Área (ha)</b><span>Superficie del terreno registrado, en hectáreas.</span></div>
                <div class="col-exp"><b>Toneladas totales (*)</b><span>Rendimiento × área: cosecha total aproximada.</span></div>
                <div class="col-exp"><b>Margen de error (± t)</b><span>Cuántas toneladas se suele equivocar el modelo en una parcela de ese municipio y tamaño.</span></div>
                <div class="col-exp"><b>Confianza</b><span>Qué tan segura es la estimación: Alta, Media, Baja o Muy baja.</span></div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

        g1, g2, g3, g4 = st.columns(4)
        t_edo = g1.multiselect("Estado", sorted(pred_df["Estado"].unique()), placeholder="Todos", key="t_edo")
        base_t = pred_df[pred_df["Estado"].isin(t_edo)] if t_edo else pred_df
        t_muni = g2.multiselect("Municipio", sorted(base_t["Municipio"].unique()), placeholder="Todos", key="t_muni")
        t_conf = g3.multiselect("Confianza", ORDEN_CONF, placeholder="Todas", key="t_conf")
        buscar = g4.text_input("Buscar ID de parcela", placeholder="Escribe parte del ID")

        vista = base_t
        if t_muni:
            vista = vista[vista["Municipio"].isin(t_muni)]
        if t_conf:
            vista = vista[vista["Confianza"].isin(t_conf)]
        if buscar.strip():
            vista = vista[vista["ID parcela"].str.contains(buscar.strip(), case=False, na=False, regex=False)]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Parcelas mostradas", f"{len(vista):,}")
        k2.metric("Rendimiento promedio estimado",
                  f"{vista['Rendimiento estimado (t/ha)'].mean():.2f} t/ha" if len(vista) else "—")
        k3.metric("Rango de estimaciones",
                  f"{vista['Rendimiento estimado (t/ha)'].min():.2f} – {vista['Rendimiento estimado (t/ha)'].max():.2f}"
                  if len(vista) else "—")
        k4.metric("Toneladas totales (*)",
                  f"{vista['Toneladas totales (*)'].sum(skipna=True):,.0f}" if len(vista) else "—")

        if vista.empty:
            st.warning("Ninguna parcela cumple con los filtros elegidos.")
        else:
            mostrar_tabla(pintar_confianza(formatear(vista, FORMATO_PRED)))
            st.caption("(*) Toneladas = rendimiento estimado × área del terreno registrado; el área no necesariamente "
                       "es la superficie sembrada.")

        d1, d2 = st.columns(2)
        d1.download_button(
            "Descargar tabla filtrada (CSV)",
            vista.to_csv(index=False).encode("utf-8-sig"),
            "predicciones_filtradas.csv", "text/csv",
        )
        d2.download_button(
            "Descargar tabla anexo para entrega (CSV)",
            C.anexo_csv(R["anexo"]), "TABLA_ANEXO_predicciones.csv", "text/csv",
            help="Formato oficial con 4 decimales: ID_parcela y Rendimiento_predicho_ton_ha (todas las parcelas).",
        )

# ------------------------------------------------------------------------------
# PESTANA 2: ANALISIS POR ESTADO Y MUNICIPIO
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
    st.markdown(
        "Cada municipio tiene **una sola estimación** para todas sus parcelas. Lo que cambia es qué tanto se puede "
        "confiar en ella, y eso depende de **qué tan parecidas son sus parcelas entre sí**, no de cuántas hay."
    )
    st.info("Ejemplo: Chignahuapan tiene 71 parcelas conocidas, pero rinden desde 2.15 hasta 5.59 t/ha. "
            "Almoloya tiene solo 10, pero todas rinden entre 4.14 y 4.87 t/ha. Por eso Almoloya es más confiable "
            "parcela por parcela.")

    tabla = tm[tm["a_predecir"] > 0].copy()
    fmt = lambda v, d=2: "—" if pd.isna(v) else f"{v:.{d}f}"  # noqa: E731
    vista_m = pd.DataFrame({
        "Municipio": tabla[MUNI] + ", " + tabla[EDO],
        "Por estimar": tabla["a_predecir"],
        "Con dato real": tabla["n"],
        "Variación entre parcelas": [fmt(v) for v in tabla["desv"]],
        "Estimación (t/ha)": [fmt(v, 4) for v in tabla["estimacion"]],
        "Error típico": [fmt(v) for v in tabla["mae_val"]],
        "Rango 80% (t/ha)": [f"{e + lo:.2f} – {e + hi:.2f}" + ("" if p else " *") for e, lo, hi, p in
                             zip(tabla["estimacion"], tabla["rango_lo"], tabla["rango_hi"], tabla["rango_propio"])],
        "Confianza": tabla["confianza"],
    })
    mostrar_tabla(pintar_confianza(vista_m))
    otros = tm.loc[tm["a_predecir"] == 0, MUNI].tolist()
    st.caption("Variación entre parcelas y error típico en t/ha. Variación entre parcelas = qué tan distintos son "
               "los rendimientos reales del municipio; error típico = cuánto se equivocó en promedio el modelo en ese "
               "municipio al revisarlo. * Menos de 10 parcelas conocidas: se muestra el rango general del modelo."
               + (f" {', '.join(otros)} tienen parcelas con dato real pero ninguna por estimar." if otros else ""))

    tm_v = tm.copy()
    tm_v["Confianza"] = tm_v["confianza"]
    cf1, cf2, cf3 = st.columns([1, 1, 2])
    sel_e = cf1.multiselect("Estado", sorted(tm_v[EDO].unique()), placeholder="Todos", key="m_edo")
    if sel_e:
        tm_v = tm_v[tm_v[EDO].isin(sel_e)]
    n_tot = len(tm_v)
    if n_tot > 0:
        top = cf2.slider("Municipios a mostrar", 1, min(60, n_tot), min(20, n_tot)) if n_tot > 1 else 1
        orden = cf3.radio("Ordenar por", ["Mayor rendimiento", "Menor rendimiento", "Más parcelas conocidas"],
                          horizontal=True)
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

        st.markdown("#### Qué tan distinto rinden las parcelas dentro de cada municipio")
        munis_graf = tm_v[MUNI].tolist()[:12]
        sub = train[train[MUNI].isin(munis_graf)]
        if len(sub):
            fig = px.box(sub, x=MUNI, y=Y, points="all", color_discrete_sequence=[VERDE],
                         category_orders={MUNI: munis_graf}, labels={Y: "Rendimiento real (t/ha)", MUNI: ""})
            fig.add_trace(go.Scatter(x=tm_v[MUNI], y=tm_v["estimacion"], mode="markers",
                                     name="Estimación del modelo",
                                     marker=dict(symbol="diamond", size=12, color="#004529",
                                                 line=dict(width=1, color="white"))))
            estilizar(fig).update_layout(height=430, legend=dict(orientation="h", y=1.08))
            mostrar_grafico(fig)
            st.caption("Cada punto verde es una parcela con rendimiento real; el rombo es la estimación del municipio. "
                       "Cajas altas = parcelas muy distintas entre sí = estimación menos confiable por parcela.")

# ------------------------------------------------------------------------------
# PESTANA 3: PARA TECNICOS - COMO FUNCIONA Y VALIDACION
# ------------------------------------------------------------------------------
with tab_modelo:
    st.markdown("### La idea, sin fórmulas")
    st.markdown(
        "El análisis mostró que **el lugar (municipio) es lo que más define el rendimiento**. Probamos más de cien "
        "variables de satélite, clima y topografía, y ninguna mejoró la estimación una vez que se conoce el municipio. "
        "Por eso el modelo es deliberadamente simple:\n\n"
        "1. Se parte del **promedio de las parcelas conocidas del municipio**.\n"
        "2. Se mezcla un poco con el **promedio del estado**, para no confiar de más en un municipio con pocas parcelas.\n"
        "3. **Mientras más parcelas conocidas, más pesa el municipio.**\n"
        "4. Si el municipio no tiene parcelas conocidas, se usa el promedio del estado."
    )
    st.markdown(
        "<div class='caja-analogia'><b>Analogía:</b> si quieres saber cuánto rinde una parcela nueva, lo sensato es "
        "ver cuánto rindieron sus vecinas del mismo municipio. Pero si solo conoces una o dos, su promedio puede salir "
        "alto o bajo por puro azar; entonces lo moderas con el promedio del estado. Eso hace el modelo.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### Pruébalo tú")
    st.caption("Este simulador es solo para entender la fórmula. **No cambia las predicciones oficiales**, que siempre usan k = 1.")
    s1, s2 = st.columns(2)
    n_demo = s1.slider("Parcelas conocidas en el municipio (n)", 0, 80, 3, key="demo_n")
    m_demo = s1.slider("Promedio del municipio (t/ha)", 1.0, 6.0, 4.5, 0.05, key="demo_m")
    e_demo = s2.slider("Promedio del estado (t/ha)", 1.0, 6.0, 3.8, 0.05, key="demo_e")
    k_demo = s2.slider("Fuerza de contracción k", 0.0, 10.0, 1.0, 0.5, key="demo_k")
    peso = 0.0 if n_demo == 0 else (1.0 if n_demo + k_demo == 0 else n_demo / (n_demo + k_demo))
    res = peso * m_demo + (1 - peso) * e_demo
    r1, r2, r3 = st.columns(3)
    r1.metric("Estimación resultante", f"{res:.2f} t/ha")
    r2.metric("Peso del municipio", f"{peso * 100:.0f}%")
    r3.metric("Peso del estado", f"{(1 - peso) * 100:.0f}%")
    st.progress(float(peso), text=f"{peso * 100:.0f}% municipio · {(1 - peso) * 100:.0f}% estado")

    with st.expander("La fórmula exacta y la elección de k"):
        st.latex(r"\hat{y} = \frac{n \cdot \bar{y}_{municipio} + k \cdot \bar{y}_{estado}}{n + k}, \qquad k = 1")
        k0 = R["val_k0"]
        st.markdown(
            f"Con **k = 0** (usar solo el promedio del municipio) el error de validación es un poco menor en las tres "
            f"métricas: MAE {k0['MAE']:.4f} vs {M['MAE']:.4f}, RMSE {k0['RMSE']:.4f} vs {M['RMSE']:.4f}, "
            f"R² {k0['R2']:.4f} vs {M['R2']:.4f}. Elegimos **k = 1** a propósito: con k = 0, Nanacamilpa "
            f"(una sola parcela conocida) se estimaría copiando ese único valor. El costo es menor a 0.005 t/ha."
        )

    st.markdown("---")
    st.markdown("### ¿Qué tan bien predice?")
    st.markdown("Para medirlo, se **esconde** el rendimiento de parte de las parcelas conocidas, se estima con el resto "
                "y se compara con lo real. Se repitió 10 veces con distintas particiones (validación cruzada estratificada "
                "por municipio, 5 pliegues).")

    if error_rel < 10:
        veredicto = "Muy bueno"
    elif error_rel < 20:
        veredicto = "Bueno"
    elif error_rel < 35:
        veredicto = "Moderado"
    else:
        veredicto = "Limitado"

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("MAE (error promedio)", f"{M['MAE']:.3f} t/ha", f"± {M['MAE_sd']:.3f} entre repeticiones",
              delta_color="off", delta_arrow="off")
    q2.metric("RMSE", f"{M['RMSE']:.3f} t/ha", f"± {M['RMSE_sd']:.3f}", delta_color="off", delta_arrow="off",
              help="Penaliza más los errores grandes.")
    q3.metric("R²", f"{M['R2']:.3f}", f"± {M['R2_sd']:.3f}", delta_color="off", delta_arrow="off")
    q4.metric("Valoración", veredicto, f"error relativo {error_rel_txt}", delta_color="off", delta_arrow="off")

    st.markdown(
        f"- **R² = {M['R2']:.2f}**: el modelo explica el {M['R2'] * 100:.0f}% de la variación **total** del rendimiento, "
        f"y **toda** esa capacidad viene de distinguir un municipio de otro. El máximo posible para cualquier modelo "
        f"que solo use el municipio es {M['techo']:.3f}; alcanzamos el {M['R2'] / M['techo'] * 100:.0f}% de ese techo. "
        "Dentro de un mismo municipio, el modelo no distingue parcelas buenas de malas.\n"
        f"- **El error promedio de {M['MAE']:.2f} t/ha** equivale a ~{error_rel:.1f}% del rendimiento promedio "
        f"({M['media_train']:.2f} t/ha). En regresión no existe un \"porcentaje de acierto\": no significa "
        f"{100 - error_rel:.1f}% de precisión.\n"
        f"- Estas cifras son de **validación** con las {len(train)} parcelas conocidas. El error real sobre las "
        f"{len(test)} parcelas por estimar todavía no se conoce."
    )

    v = R["val"]
    gr1, gr2 = st.columns(2)
    with gr1:
        st.markdown("#### Real vs. estimado")
        dfv = pd.DataFrame({"Real (t/ha)": v["y"], "Estimado en validación (t/ha)": v["pred_cv"],
                            "Municipio": v["municipio"]})
        fig = px.scatter(dfv, x="Real (t/ha)", y="Estimado en validación (t/ha)", color="Municipio", opacity=0.75)
        fig.add_trace(go.Scatter(x=[2, 5.7], y=[2, 5.7], mode="lines", name="Acierto perfecto",
                                 line=dict(dash="dash", color=ROJO)))
        estilizar(fig).update_layout(height=400, legend=dict(font=dict(size=10)))
        mostrar_grafico(fig)
        st.caption("Las estimaciones forman líneas horizontales: dentro de un municipio, todas las parcelas reciben el mismo valor.")
    with gr2:
        st.markdown("#### Distribución de los errores")
        lo_g, hi_g = M["rango_global"]
        fig = px.histogram(x=v["residuos"], nbins=35, color_discrete_sequence=[DORADO],
                           labels={"x": "Error (real − estimado, t/ha)"})
        fig.add_vrect(x0=lo_g, x1=hi_g, fillcolor=VERDE, opacity=0.12, line_width=0,
                      annotation_text="80% de los casos", annotation_position="top left")
        fig.add_vline(x=0, line_dash="dash", line_color=ROJO)
        estilizar(fig).update_layout(height=400, xaxis_title="Error (real − estimado, t/ha)", yaxis_title="Parcelas")
        mostrar_grafico(fig)
        st.caption(f"En 8 de cada 10 parcelas conocidas, el error quedó entre {lo_g:+.2f} y {hi_g:+.2f} t/ha (rango general).")

    st.markdown("---")
    st.markdown("### ¿Cuándo se puede usar?")
    st.markdown(
        "Como el modelo solo usa el municipio y el estado, **puede dar una estimación desde antes de la siembra**, "
        "sin esperar imágenes de satélite ni datos de clima del ciclo. La condición: en un uso real, los promedios "
        f"municipales vendrían del ciclo anterior. Aquí se calcularon con el ciclo {CICLO}, el único disponible, así que "
        "no se pudo comprobar que se mantengan de un año a otro. Las imágenes de satélite sí sirven durante el ciclo "
        "para seguir el cultivo (emergencia, máximo verdor, maduración) y detectar anomalías."
    )

    with st.expander("Limitaciones"):
        st.markdown(
            "- No distingue parcelas del mismo municipio: no hay datos de variedad, fecha o dosis de siembra, rotación "
            "ni siniestros por parcela.\n"
            "- Las parcelas excepcionalmente altas o bajas se estiman hacia el promedio de su zona.\n"
            "- El rendimiento se pesó en báscula sin ajuste por humedad, impurezas ni mermas, y una parte se reporta "
            "redondeada a 0.5 t/ha. Parte del error no se puede eliminar con ningún modelo.\n"
            "- El área es la del terreno registrado, no necesariamente la sembrada; las toneladas son aproximadas.\n"
            f"- Solo hay datos del ciclo {CICLO}. Para otro ciclo hay que volver a calcular los promedios.\n"
            "- Chignahuapan concentra 29 de las 59 parcelas y es el municipio más variable: pesa mucho en el error final.\n"
            "- **k oficial = 1:** las predicciones de entrega no se modifican desde la interfaz.\n"
            "- **t/ha:** toneladas cosechadas por cada hectárea."
        )

# ------------------------------------------------------------------------------
# Pie de pagina
# ------------------------------------------------------------------------------
st.markdown(
    f'<div class="footer-app">AgroCebada · Reto FIRA 2026 — modelo jerárquico municipio → estado · '
    f"datos del ciclo {CICLO} · error típico ±{mae:.2f} t/ha</div>",
    unsafe_allow_html=True,
)