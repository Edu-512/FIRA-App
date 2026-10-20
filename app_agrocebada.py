"""
================================================================================
AgroCebada FIRA 2026 - Cosecha estimada de cebada por parcela (Streamlit)
================================================================================
Organizada por publico, en 4 pestanas:
    Mi parcela        (productor)  mapa + ficha de la parcela
    Panorama          (analista)   resumen, confianza por municipio, graficas
    Tabla y descargas              tabla de las 59 parcelas y archivo oficial
    Para tecnicos                  como funciona el modelo y su validacion

Predicciones, validacion, confianza y rangos salen exclusivamente de
calculos.py (modelo jerarquico municipio -> estado, k = 1).

Archivos en la MISMA carpeta:
    app_agrocebada.py         esta app
    calculos.py               calculos oficiales (validacion, confianza, rangos)
    modelo_agrocebada.py      el modelo entregado
    dataset_maestro.csv       ID, estado, municipio, area y rendimiento
    parcelas_centroides.csv   centro de cada parcela (lon, lat)
    parcelas.geojson          OPCIONAL: poligonos reales (preparar_datos.py)

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
from branca.element import MacroElement
from jinja2 import Template
from streamlit_folium import st_folium

import calculos as C
from calculos import ID, MUNI, EDO, Y

st.set_page_config(page_title="AgroCebada FIRA 2026", layout="wide")

CICLO = "2025"
VMIN, VMAX = 2.0, 5.6  # misma escala de color para estimado y real
COLORES = ["#d73027", "#fc8d59", "#fee08b", "#91cf60", "#1a9850"]  # bajo -> alto
ORDEN_CONF = ["Alta", "Media", "Baja", "Muy baja"]
CLASE_CONF = {"Alta": "alta", "Media": "media", "Baja": "baja", "Muy baja": "muybaja"}
T_EST = "Con estimación"
T_REAL = "Ya cosechada y medida"
VERDE, DORADO, GRIS, ROJO = "#2E7D32", "#C8A415", "#9E9E9E", "#C62828"
ALTO_MAPA = 680  # px; el panel flotante de la parcela usa la misma altura maxima

st.markdown(
    """
    <style>
    /* Deja libre la barra superior fija de Streamlit (~3.75rem) para que no tape el titulo */
    .block-container {padding-top: 4.5rem;}
    @media (max-width: 700px) {
      .block-container {padding: 4.25rem .75rem 1.5rem .75rem;}
      .hero {padding: 12px 14px;}
      .hero h1 {font-size: 1.22rem; line-height: 1.25;}
      .hero p {font-size: .88rem;}
    }
    div[data-testid="stMetric"] {
        background-color: rgba(46,125,50,0.07);
        border-left: 5px solid #2E7D32;
        padding: 10px 14px;
        border-radius: 10px;
    }
    div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] > div {
        white-space: normal !important; overflow: visible !important;
        text-overflow: clip !important; overflow-wrap: anywhere;
    }
    .hero {background: linear-gradient(135deg, #E8F5E9 0%, #FFF9E6 100%);
           border: 1px solid #C8E6C9; border-radius: 14px; padding: 18px 24px; margin-bottom: 8px;}
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
                  color:#fff; border-radius:12px 12px 0 0; padding:12px 18px; margin-top:6px;}
    .prod-titulo h3 {margin:0; font-size:1.15rem; color:#fff;}
    .prod-titulo p {margin:2px 0 0; font-size:.85rem; color:#E8F5E9;}
    .prod-texto {background:#F1F8E9; border:1px solid #C5E1A5; border-top:none; border-radius:0 0 12px 12px;
                 padding:14px 18px; color:#263238; font-size:1rem; line-height:1.55; margin-bottom:10px;}
    .prod-texto b {color:#1B5E20;}
    .prod-pie {background:#FFF9E6; border:1px solid #FFE082; border-radius:12px; padding:10px 16px;
               color:#5D4A00; font-size:.85rem; line-height:1.45; margin:10px 0;}

    .calc-res {display:grid; grid-template-columns:repeat(auto-fit, minmax(min(100%, 200px), 1fr));
               gap:10px; margin:4px 0 8px 0;}
    .calc-item {background-color: rgba(46,125,50,0.07); border-left: 5px solid #2E7D32;
                border-radius:10px; padding:10px 14px; min-width:0;}
    .calc-item .etq {font-size:.85rem; opacity:.8;}
    .calc-item .val {font-size:clamp(1.15rem, 1.1vw + .7rem, 1.6rem); font-weight:600; line-height:1.25;
                     margin-top:2px; overflow-wrap:anywhere; word-break:break-word;}

    /* ---------- Mi parcela: mapa a todo lo ancho con panel flotante ---------- */
    /* Fondo solido sin colores fijos: los ancestros transparentes heredan el fondo del tema de .stApp
       (no cambia nada visible) y asi los elementos flotantes pueden usar "inherit" en claro y oscuro. */
    .stApp *:has(.st-key-panel_parcela, .st-key-aviso_mapa, .st-key-leyenda_mapa) {background-color: inherit;}
    .st-key-zona_mapa {position: relative; gap: 0;}
    .st-key-panel_parcela, .st-key-aviso_mapa, .st-key-leyenda_mapa {
        position: absolute; z-index: 1000; background-color: inherit;
        border: 1px solid rgba(128,128,128,.25); box-shadow: 0 6px 24px rgba(0,0,0,.18);}
    .st-key-aviso_mapa [data-testid="stMarkdownContainer"],
    .st-key-leyenda_mapa [data-testid="stMarkdownContainer"],
    .st-key-panel_parcela [data-testid="stMarkdownContainer"] {margin-bottom: 0;}  /* Streamlit pone -1rem */
    .st-key-panel_parcela {top: 12px; right: 12px; width: 400px; max-height: calc(680px - 24px);
        overflow-y: auto; border-radius: 12px; padding: 12px 16px 14px 16px; gap: .55rem;}
    .st-key-panel_parcela h4 {font-size: 1.1rem; padding: 0;}
    .st-key-panel_parcela h5 {font-size: .95rem; padding: .2rem 0 0 0;}
    .st-key-panel_parcela .prod-titulo {margin-top: 0; padding: 10px 14px;}
    .st-key-panel_parcela .prod-texto {padding: 10px 14px; font-size: .93rem; margin-bottom: 0;}
    .st-key-panel_parcela .prod-pie {margin: 2px 0 0 0;}
    .st-key-panel_parcela div[data-testid="stMetric"] {padding: 8px 10px;}
    .st-key-panel_parcela div[data-testid="stMetricValue"] {font-size: 1.45rem;}
    .st-key-panel_parcela div[data-testid="stMetricLabel"] p {font-size: .82rem;}
    .st-key-panel_parcela div[data-testid="stMetricDelta"], .st-key-panel_parcela div[data-testid="stMetricDelta"] * {
        white-space: normal !important; overflow: visible !important; text-overflow: clip !important;}
    .panel-barra {font-size: .8rem; font-weight: 600; letter-spacing: .04em; text-transform: uppercase;
                  opacity: .65; padding-right: 40px; min-height: 26px; line-height: 26px;}
    .st-key-cerrar_panel {position: absolute; top: 9px; right: 10px; width: auto !important; z-index: 2;}
    .st-key-cerrar_panel button {min-height: 0; padding: 1px 9px; line-height: 1.4; border-radius: 8px;}
    .st-key-aviso_mapa {top: 12px; right: 12px; width: max-content; max-width: 400px; border-radius: 16px;
        padding: 7px 16px;}
    .st-key-aviso_mapa p {font-size: .88rem; margin: 0;}
    .st-key-leyenda_mapa {top: calc(680px - 165px); left: 12px; width: 200px; border-radius: 10px;
        padding: 7px 11px 6px 11px; gap: 0;}
    .lm-tit {font-size: .78rem; font-weight: 600; margin-bottom: 4px;}
    .lm-barra {height: 10px; border-radius: 5px;}
    .lm-esc {display: flex; justify-content: space-between; font-size: .72rem; margin-top: 2px;}
    .lm-pie {font-size: .7rem; opacity: .7; text-align: center; margin-top: 1px;}
    .st-key-barra_filtros {gap: .4rem;}
    .st-key-barra_filtros div[data-testid="stHorizontalBlock"] {gap: .6rem;}
    @media (max-width: 700px) {
      /* relative = en el flujo normal (debajo del mapa) como static, pero sigue anclando el boton cerrar */
      .st-key-panel_parcela, .st-key-aviso_mapa {position: relative; top: auto; right: auto; width: 100%; max-width: none;
          max-height: none; overflow: visible; box-shadow: none; margin-top: 10px;}
      .st-key-aviso_mapa {border-radius: 10px;}
    }

    [role="tablist"] {gap:8px; margin-top: 4px;}
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
    return f'<span class="chip chip-{CLASE_CONF.get(conf, "media")}">{escape(str(conf))}</span>'


def pesos(x) -> str:
    """Pesos sin el signo $ (dos $ en un texto se interpretan como formula matematica)."""
    return f"{x:,.0f} pesos"


def mostrar_grafico(fig, **extra):
    st.plotly_chart(fig, width="stretch", theme="streamlit", **extra)


def mostrar_tabla(obj):
    st.dataframe(obj, width="stretch", hide_index=True)


def estilizar(fig):
    fig.update_layout(font=dict(family="'Segoe UI', 'Helvetica Neue', Arial, sans-serif", size=13),
                      margin=dict(l=12, r=30, t=16, b=12))
    return fig


def redondear_barras(fig, radio=6):
    try:
        fig.update_layout(barcornerradius=radio)
    except Exception:  # noqa: BLE001 - versiones antiguas de plotly
        pass
    return fig


def pintar_confianza(df_o_styler):
    colores = {"Alta": "#E8F5E9", "Media": "#FFF3CD", "Baja": "#FFE0B2", "Muy baja": "#FFCDD2"}
    texto = {"Alta": "#1B5E20", "Media": "#7A5C00", "Baja": "#8C4A00", "Muy baja": "#9A0F0F"}
    sty = df_o_styler.style if isinstance(df_o_styler, pd.DataFrame) else df_o_styler
    return sty.map(lambda v: f"background-color:{colores.get(v, 'transparent')}; "
                             f"color:{texto.get(v, 'inherit')}; font-weight:600;", subset=["Confianza"])


class EncuadreConPanel(MacroElement):
    """fit_bounds que deja libre el lado derecho cuando el mapa es ancho (ahi flota el panel de la parcela)."""
    _template = Template("""
        {% macro script(this, kwargs) %}
        (function () {
            var mapa = {{ this._parent.get_name() }};
            var der = mapa.getSize().x > 900 ? 430 : 10;
            mapa.fitBounds({{ this.limites|tojson }}, {paddingTopLeft: [10, 10], paddingBottomRight: [der, 10]});
        })();
        {% endmacro %}""")

    def __init__(self, limites):
        super().__init__()
        self._name = "EncuadreConPanel"
        self.limites = [[float(a), float(b)] for a, b in limites]


def formatear(df: pd.DataFrame, formatos: dict):
    return df.style.format({c: f for c, f in formatos.items() if c in df.columns}, na_rep="—")


def plural(n, palabra):
    return f"{n} {palabra}{'' if n == 1 else 's'}"


# ------------------------------------------------------------------------------
# Datos y modelo oficial (calculos.py; se calculan una sola vez)
# ------------------------------------------------------------------------------
@st.cache_data(show_spinner="Entrenando y validando el modelo (solo la primera vez)…")
def preparar():
    for f in ("dataset_maestro.csv", "modelo_agrocebada.py"):
        if not os.path.exists(f):
            return None, f"Falta el archivo **{f}** en la carpeta de la app."
    try:
        R = C.calcular_todo(C.cargar())
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
media_global = float(train[Y].mean())
medias_estado = {str(e): float(g[Y].mean()) for e, g in train.groupby(EDO)}
n_muni_dict = dict(zip(tm[MUNI], tm["n"].astype(int)))
escala = bcm.LinearColormap(COLORES, vmin=VMIN, vmax=VMAX, caption="Rendimiento (t/ha)")

# Tabla unica de parcelas para el mapa y la ficha
parcelas = pd.concat([
    test.assign(tipo=T_EST, valor=test["pred"], ton_val=pd.to_numeric(test["toneladas"], errors="coerce")),
    train.assign(tipo=T_REAL, valor=train[Y], ton_val=train[Y] * pd.to_numeric(train["area_ha"], errors="coerce")),
], ignore_index=True)

# Tabla de predicciones (pestanas Panorama y Tabla)
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
}) if len(test) else pd.DataFrame()

FORMATO_PRED = {
    "Rendimiento estimado (t/ha)": "{:.2f}", "Mínimo esperado (t/ha)": "{:.2f}",
    "Máximo esperado (t/ha)": "{:.2f}", "Área (ha)": "{:.2f}",
    "Toneladas totales (*)": "{:,.1f}", "Margen de error (± t)": "{:,.1f}",
}

# ------------------------------------------------------------------------------
# Encabezado
# ------------------------------------------------------------------------------
st.markdown(limpiar_html("""
    <div class="hero">
    <h1>AgroCebada · Cosecha estimada</h1>
    <p>Encuentra tu parcela y consulta cuántas toneladas de cebada podrías cosechar.</p>
    </div>"""), unsafe_allow_html=True)

desconocidos = sorted(set(test.loc[~test["conocido"], MUNI])) if len(test) else []
if desconocidos:
    st.error("**Advertencia:** estas parcelas están en municipios sin parcelas conocidas y usan el promedio de su "
             f"estado (no validado; error esperado de ~0.5 t/ha o más): {', '.join(desconocidos)}.")

tab_parcela, tab_panorama, tab_tabla, tab_tecnicos = st.tabs(
    ["Mi parcela", "Panorama", "Tabla y descargas", "Para técnicos"])

# ==============================================================================
# 1. MI PARCELA (productor)
# ==============================================================================
with tab_parcela:
    st.caption("**Cómo usarlo:** 1) Elige tu estado y municipio.  2) Toca tu parcela en el mapa o búscala en la lista.  "
               "3) Revisa cuántas toneladas podrías cosechar y cuánto podría variar.")

    # parcela_sel = None -> panel cerrado. mapa_ver cambia la key de los mapas para reiniciarlos: asi el
    # ultimo clic que st_folium sigue devolviendo no vuelve a abrir el panel.
    st.session_state.setdefault("parcela_sel", None)
    st.session_state.setdefault("mapa_ver", 0)

    def reiniciar_mapa():
        st.session_state["_ultimo_clic"] = None
        st.session_state["mapa_ver"] += 1

    def cerrar_panel():
        st.session_state["parcela_sel"] = None
        st.session_state["sel_lista"] = None
        reiniciar_mapa()

    def _desde_lista():
        st.session_state["parcela_sel"] = st.session_state["sel_lista"]
        reiniciar_mapa()

    def elegir_desde_mapa(pid):
        if pid and pid != st.session_state.get("_ultimo_clic"):
            st.session_state["_ultimo_clic"] = pid
            st.session_state["parcela_sel"] = pid
            st.rerun()

    # ------------------------------ barra de filtros ------------------------------
    with st.container(key="barra_filtros"):
        f1, f2, f3, f4 = st.columns([1, 1.5, 1.3, 0.95], vertical_alignment="center")
        sel_edo = f1.multiselect("Estado", sorted(parcelas[EDO].unique()), placeholder="Estado: todos",
                                 key="mapa_edo", label_visibility="collapsed")
        base = parcelas[parcelas[EDO].isin(sel_edo)] if sel_edo else parcelas
        sel_muni = f2.multiselect("Municipio", sorted(base[MUNI].unique()), placeholder="Municipio: todos",
                                  key="mapa_muni", label_visibility="collapsed")
        if sel_muni:
            base = base[base[MUNI].isin(sel_muni)]
        ver_reales = f4.toggle("Ver ya cosechadas", value=True,
                               help="Ayudan a ver cuánto cambia el rendimiento de una parcela a otra dentro de cada municipio.")
        visibles = base if ver_reales else base[base["tipo"] == T_EST]

        if st.session_state["parcela_sel"] not in set(visibles[ID]):
            st.session_state["parcela_sel"] = None
        opciones = sorted(base.loc[base["tipo"] == T_EST, ID])
        st.session_state["sel_lista"] = (st.session_state["parcela_sel"]
                                         if st.session_state["parcela_sel"] in opciones else None)
        f3.selectbox("Parcela", opciones, index=None, key="sel_lista", on_change=_desde_lista,
                     placeholder="Buscar parcela por ID", label_visibility="collapsed",
                     help="También puedes tocarla en el mapa.")
    pid_sel = st.session_state["parcela_sel"]

    # ---------------------------------- ficha ----------------------------------
    def ficha_parcela(fila):
        muni, edo = fila[MUNI], fila[EDO]
        im = info_muni.loc[muni] if muni in info_muni.index else None
        area = float(fila["area_ha"])

        if fila["tipo"] == T_REAL:
            st.markdown(f"#### Parcela {fila[ID]}")
            st.caption(f"{muni}, {edo} · {area:.2f} ha (superficie del terreno registrado)")
            st.info("Esta parcela **ya fue cosechada y medida**: su rendimiento real se conoce y se usó para "
                    "ajustar el modelo. Elige una parcela **con estimación** para ver su cosecha esperada.")
            a, b = st.columns(2)
            a.metric("Rendimiento real", f"{fila[Y]:.2f} t/ha")
            if pd.notna(fila["ton_val"]):
                b.metric("Cosecha real aproximada", f"{fila['ton_val']:,.1f} toneladas")
            if im is not None:
                st.metric("Estimación para su municipio", f"{im['estimacion']:.2f} t/ha",
                          f"{fila[Y] - im['estimacion']:+.2f} t/ha de diferencia con esta parcela",
                          delta_color="off", delta_arrow="off")
            st.caption("La diferencia muestra cuánto puede alejarse una parcela del promedio de su zona.")
            return

        ton, err = float(fila["toneladas"]), float(fila["error_ton"])
        ton_min, ton_max = float(fila["min_esp"]) * area, float(fila["max_esp"]) * area

        # 1. Lo que mas le importa al productor
        st.markdown(limpiar_html(f"""
            <div class="prod-titulo">
            <h3>Parcela {escape(str(fila[ID]))}</h3>
            <p>{escape(str(muni))}, {escape(str(edo))} · {area:.2f} ha de terreno registrado</p>
            </div>
            <div class="prod-texto">
            Podrías cosechar unas <b>{ton:,.1f} toneladas</b> de cebada.
            En 8 de cada 10 casos, la cosecha quedaría entre <b>{ton_min:,.1f}</b> y
            <b>{ton_max:,.1f} toneladas</b>.
            </div>"""), unsafe_allow_html=True)
        p1, p2 = st.columns(2)
        p1.metric("Cosecha estimada", f"≈ {ton:,.1f} t",
                  f"{fila['pred']:.2f} t/ha", delta_color="off", delta_arrow="off")
        p2.metric("Cuánto podría desviarse", f"± {err:,.1f} t",
                  f"error típico en {muni}", delta_color="off", delta_arrow="off")

        # 2. Calculadora de ingreso
        with st.container(border=True):
            st.markdown("**Calculadora de ingreso**")
            precio = st.number_input("Precio por tonelada (pesos)", min_value=0.0, value=0.0, step=100.0,
                                     key="precio_ton",
                                     help="Escribe el precio de contrato o de mercado que quieras usar.")
            if precio > 0:
                # HTML propio en vez de st.metric: con montos grandes el texto baja de linea
                # en lugar de salirse de la caja
                st.markdown(limpiar_html(f"""
                    <div class="calc-res">
                    <div class="calc-item"><div class="etq">Ingreso esperado</div>
                    <div class="val">{pesos(ton * precio)}</div></div>
                    <div class="calc-item"><div class="etq">Margen</div>
                    <div class="val">± {pesos(err * precio)}</div></div>
                    </div>"""), unsafe_allow_html=True)
                st.caption(f"En 8 de cada 10 casos: entre {pesos(ton_min * precio)} y {pesos(ton_max * precio)}.")
            else:
                st.caption("Escribe un precio por tonelada para ver cuánto podrías ingresar.")

        # 3. Confianza en palabras
        n_m = int(im["n"]) if im is not None else 0
        st.markdown(f"**Qué tan segura es:** {chip(fila['confianza'])}  \n"
                    f"{escape(C.EXPLICACION_CONF[fila['confianza']])} "
                    f"({plural(n_m, 'parcela')} ya medida{'' if n_m == 1 else 's'} en {escape(str(muni))}).",
                    unsafe_allow_html=True)

        # 4. De donde sale el numero
        st.markdown("##### ¿De dónde sale este número?")
        if im is not None:
            pm = float(im["peso_muni"])
            st.progress(pm, text=f"{pm * 100:.1f}% promedio de {muni} · {(1 - pm) * 100:.1f}% promedio de {edo}")
            st.caption(f"Se combina lo que rindieron las parcelas ya medidas de {muni} "
                       f"({im['prom_obs']:.2f} t/ha en promedio) con el promedio de {edo} "
                       f"({im['media_estado']:.2f} t/ha). Mientras más parcelas medidas tiene el municipio, "
                       "más pesa su propio promedio. Satélite y clima no cambian este número.")
            st.caption(f"Rendimiento estimado: **{fila['pred']:.2f} t/ha** · en 8 de cada 10 casos entre "
                       f"{fila['min_esp']:.2f} y {fila['max_esp']:.2f} t/ha"
                       + ("." if im["rango_propio"] else
                          f" (rango general del modelo: {muni} tiene pocas parcelas medidas)."))
        else:
            st.warning("Este municipio no tiene parcelas medidas: se usa solo el promedio del estado.")

        # 5. Avisos
        if im is not None and im["n"] >= 6 and im["valores_distintos"] <= 4:
            vals = ", ".join(f"{v:g}" for v in sorted(train.loc[train[MUNI] == muni, Y].round(2).unique()))
            st.warning(f"En {muni} los rendimientos reportados solo toman los valores **{vals} t/ha**. "
                       "El de tu parcela probablemente sea uno de esos valores, así que la diferencia con la "
                       "estimación será de 0 o de un escalón completo.")
        if im is not None and im["n"] < 3:
            st.warning(f"{muni} tiene solo {plural(int(im['n']), 'parcela')} medida"
                       f"{'' if im['n'] == 1 else 's'}: la estimación se apoya en "
                       f"{(1 - im['peso_muni']) * 100:.0f}% en el promedio del estado.")
        elif fila["confianza"] == "Baja":
            st.warning("Las parcelas de este municipio rinden muy distinto entre sí: tu parcela podría "
                       "alejarse bastante de la estimación.")

        st.markdown(limpiar_html("""
            <div class="prod-pie">
            Útil para planear la venta o pedir un crédito. No considera tu variedad, fecha de siembra ni
            daños por granizo o helada, y no garantiza el rendimiento de una parcela en particular.
            Las toneladas se calculan con la superficie registrada, que puede no ser la sembrada.
            </div>"""), unsafe_allow_html=True)

    # ----------------------- mapa a todo lo ancho + panel flotante -----------------------
    with st.container(key="zona_mapa"):
        geo = visibles.dropna(subset=["lon", "lat"])
        # Encuadre: lo visible; si el filtro deja el mapa vacio, la zona filtrada (aunque esten ocultas)
        marco = geo if len(geo) else base.dropna(subset=["lon", "lat"])
        sel = parcelas.loc[parcelas[ID] == pid_sel]
        ver = st.session_state["mapa_ver"]

        def feature(fila, geometria):
            est = fila["tipo"] == T_EST
            ton_txt = f"{fila['ton_val']:,.1f} toneladas" if pd.notna(fila["ton_val"]) else "—"
            return {"type": "Feature", "geometry": geometria, "properties": {
                "ID": fila[ID], "Tipo": fila["tipo"], "Municipio": fila[MUNI],
                "Cosecha": ton_txt + (" (estimada)" if est else " (real)"),
                "Rendimiento": f"{fila['valor']:.2f} t/ha" + (" (estimado)" if est else " (real)"),
                "valor": float(fila["valor"]), "estimar": est}}

        def estilo(feat):
            p = feat["properties"]
            return {"fillColor": escala(min(max(p["valor"], VMIN), VMAX)),
                    "color": "#1b1b1b" if p["estimar"] else "#6b6b6b",
                    "weight": 2 if p["estimar"] else 1,
                    "fillOpacity": 0.9 if p["estimar"] else 0.45,
                    "dashArray": None if p["estimar"] else "3",
                    "radius": 7 if p["estimar"] else 4}

        campos_tt = dict(fields=["ID", "Tipo", "Municipio", "Cosecha", "Rendimiento"],
                         aliases=["Parcela", "", "Municipio", "Cosecha", "Rendimiento"])
        m = folium.Map(tiles=None, control_scale=True)
        folium.TileLayer("OpenStreetMap", name="Calles").add_to(m)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri, Maxar, Earthstar Geographics", name="Satélite").add_to(m)
        # Una capa sin parcelas rompe GeoJsonTooltip (folium exige que los campos existan),
        # por eso solo se agregan las capas que tienen algo que dibujar
        poli = ([feature(f, poligonos[f[ID]]) for _, f in visibles.iterrows() if f[ID] in poligonos]
                if poligonos else [])
        if poli:
            folium.GeoJson({"type": "FeatureCollection", "features": poli}, name="Parcelas (forma real)",
                           style_function=estilo, tooltip=folium.GeoJsonTooltip(**campos_tt),
                           highlight_function=lambda _: {"weight": 4, "color": "#d62728"}).add_to(m)
        puntos = [feature(f, {"type": "Point", "coordinates": [f["lon"], f["lat"]]}) for _, f in geo.iterrows()]
        if puntos:
            folium.GeoJson({"type": "FeatureCollection", "features": puntos}, name="Parcelas (punto)",
                           marker=folium.CircleMarker(fill=True), style_function=estilo,
                           tooltip=folium.GeoJsonTooltip(**campos_tt),
                           highlight_function=lambda _: {"weight": 4, "color": "#d62728"}).add_to(m)
        # Capas a la izquierda: la derecha la ocupa el panel de la parcela
        folium.LayerControl(collapsed=True, position="topleft").add_to(m)
        if len(marco):
            pad = 0.01
            EncuadreConPanel([[marco["lat"].min() - pad, marco["lon"].min() - pad],
                              [marco["lat"].max() + pad, marco["lon"].max() + pad]]).add_to(m)

        # La parcela elegida va en una capa aparte para no recargar todo el mapa
        resaltado = folium.FeatureGroup(name="Parcela elegida")
        if len(sel) and pd.notna(sel["lat"].iloc[0]):
            folium.CircleMarker([sel["lat"].iloc[0], sel["lon"].iloc[0]], radius=14, color="#2b6cb0",
                                weight=4, fill=False, tooltip=f"Elegida: {sel[ID].iloc[0]}").add_to(resaltado)
        salida = st_folium(m, key=f"mapa_{ver}", height=ALTO_MAPA, use_container_width=True,
                           returned_objects=["last_active_drawing"], feature_group_to_add=resaltado)
        clic = (salida or {}).get("last_active_drawing") or {}
        elegir_desde_mapa((clic.get("properties") or {}).get("ID"))

        # Leyenda de color flotante (abajo a la izquierda)
        with st.container(key="leyenda_mapa"):
            st.markdown(limpiar_html(f"""
                <div class="lm-tit">Rendimiento (t/ha)</div>
                <div class="lm-barra" style="background:linear-gradient(90deg, {', '.join(COLORES)});"></div>
                <div class="lm-esc"><span>{VMIN:.1f}</span><span>{(VMIN + VMAX) / 2:.1f}</span><span>{VMAX:.1f}</span></div>
                <div class="lm-pie">rinde menos · rinde más</div>"""), unsafe_allow_html=True)

        if pid_sel:
            with st.container(key="panel_parcela"):
                st.button("✕", key="cerrar_panel", on_click=cerrar_panel, help="Cerrar la ficha")
                st.markdown('<div class="panel-barra">Ficha de la parcela</div>', unsafe_allow_html=True)
                ficha_parcela(parcelas.loc[parcelas[ID] == pid_sel].iloc[0])
        else:
            with st.container(key="aviso_mapa"):
                if visibles.empty and not ver_reales and len(base):
                    st.markdown("En esta zona solo hay parcelas **ya cosechadas**, sin estimación. "
                                "Activa «Ver ya cosechadas» o elige otro municipio.")
                else:
                    st.markdown("No hay parcelas con estimación con estos filtros." if not opciones else
                                "Toca una parcela en el mapa para ver su cosecha estimada")

    st.caption(
        "**Leyenda:** rojo = rinde menos · verde = rinde más. Las parcelas **con estimación** tienen borde "
        "oscuro; las **ya cosechadas**, borde punteado y color más tenue. La parcela elegida se marca con un "
        "círculo azul. Con el botón de capas (arriba a la izquierda) cambias entre calles y satélite."
        + ("" if poligonos else " · Se muestran puntos porque falta `parcelas.geojson`."))

# ==============================================================================
# 2. PANORAMA (analista)
# ==============================================================================
with tab_panorama:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Parcelas con estimación", f"{len(test)}")
    k2.metric("Parcelas ya medidas", f"{len(train)}", help="Con ellas se entrenó y revisó el modelo.")
    k3.metric("Error promedio", f"± {mae:.2f} t/ha", f"≈ {error_rel:.1f}% del rendimiento promedio",
              delta_color="off", delta_arrow="off",
              help="Error absoluto medio en validación cruzada (10 repeticiones). No es un 'porcentaje de acierto'.")
    k4.metric("Cosecha total esperada", f"{pred_df['Toneladas totales (*)'].sum():,.0f} t",
              f"en {pred_df['Área (ha)'].sum():,.0f} ha", delta_color="off", delta_arrow="off")

    por_muni = pred_df.groupby("Municipio")["Rendimiento estimado (t/ha)"].mean()
    pct_am = pred_df["Confianza"].isin(["Alta", "Media"]).mean() * 100
    lineas = [
        f"Se estimó el rendimiento de <b>{len(pred_df)} parcelas</b> en <b>{pred_df['Municipio'].nunique()} "
        f"municipios</b>, con un promedio de <b>{pred_df['Rendimiento estimado (t/ha)'].mean():.2f} t/ha</b>.",
        f"Mayor rendimiento estimado: <b>{escape(str(por_muni.idxmax()))}</b> ({por_muni.max():.2f} t/ha) · "
        f"menor: <b>{escape(str(por_muni.idxmin()))}</b> ({por_muni.min():.2f} t/ha).",
        f"El <b>{pct_am:.0f}%</b> de las estimaciones tiene confianza Alta o Media.",
        "Todas las parcelas de un municipio reciben la misma estimación: el modelo usa solo el municipio y el estado.",
    ]
    st.markdown("<div class='caja-resumen'><h3>Resumen</h3><ul>" + "".join(f"<li>{x}</li>" for x in lineas)
                + "</ul></div>", unsafe_allow_html=True)

    st.markdown("### Estimación y confianza por municipio")
    st.markdown("Cada municipio tiene **una sola estimación**. Lo que cambia es qué tanto se puede confiar en ella, "
                "y eso depende de **qué tan parecidas son sus parcelas entre sí**, no de cuántas hay.")
    st.info("Ejemplo: Chignahuapan tiene 71 parcelas medidas, pero rinden desde 2.15 hasta 5.59 t/ha. "
            "Almoloya tiene solo 10, pero todas rinden entre 4.14 y 4.87 t/ha. Por eso Almoloya es más confiable "
            "parcela por parcela.")
    tabla = tm[tm["a_predecir"] > 0].copy()
    fmt = lambda v, d=2: "—" if pd.isna(v) else f"{v:.{d}f}"  # noqa: E731
    mostrar_tabla(pintar_confianza(pd.DataFrame({
        "Municipio": tabla[MUNI],
        "Por estimar": tabla["a_predecir"],
        "Medidas": tabla["n"],
        "Variación": [fmt(v) for v in tabla["desv"]],
        "Estimación": [fmt(v, 4) for v in tabla["estimacion"]],
        "Error": [fmt(v) for v in tabla["mae_val"]],
        "Rango 8 de 10": [f"{e + lo:.2f} – {e + hi:.2f}" + ("" if p else " *") for e, lo, hi, p in
                                 zip(tabla["estimacion"], tabla["rango_lo"], tabla["rango_hi"], tabla["rango_propio"])],
        "Confianza": tabla["confianza"],
    })))
    otros = tm.loc[tm["a_predecir"] == 0, MUNI].tolist()
    st.caption("Todo en t/ha. **Por estimar** = parcelas con estimación; **Medidas** = parcelas ya cosechadas y medidas. "
               "**Variación** = qué tan distinto rinden las parcelas medidas del municipio (desviación estándar). "
               "**Error** = cuánto se equivocó en promedio el modelo en ese municipio durante la validación. "
               "* Menos de 10 parcelas medidas: se muestra el rango general del modelo."
               + (f" {', '.join(otros)} tienen parcelas medidas pero ninguna con estimación." if otros else ""))

    g1, g2 = st.columns([1.4, 1])
    with g1:
        st.markdown("#### Qué tan distinto rinden las parcelas de cada municipio")
        orden = tm.sort_values("estimacion", ascending=False)[MUNI].tolist()
        fig = px.box(train, x=MUNI, y=Y, points="all", color_discrete_sequence=[VERDE],
                     category_orders={MUNI: orden}, labels={Y: "Rendimiento medido (t/ha)", MUNI: ""})
        fig.add_trace(go.Scatter(x=tm[MUNI], y=tm["estimacion"], mode="markers", name="Estimación del modelo",
                                 marker=dict(symbol="diamond", size=12, color="#004529",
                                             line=dict(width=1, color="white"))))
        estilizar(fig).update_layout(height=440, legend=dict(orientation="h", y=1.08))
        mostrar_grafico(fig)
        st.caption("Cada punto es una parcela ya medida; el rombo es la estimación del municipio. "
                   "Cajas altas = parcelas muy distintas entre sí = estimación menos segura por parcela.")
    with g2:
        st.markdown("#### Cosecha total esperada por municipio")
        tt = (pred_df.groupby("Municipio")["Toneladas totales (*)"].sum().sort_values().reset_index())
        fig = px.bar(tt, x="Toneladas totales (*)", y="Municipio", orientation="h", text="Toneladas totales (*)",
                     color_discrete_sequence=[DORADO])
        fig.update_traces(texttemplate="%{x:,.0f} t", textposition="outside")
        estilizar(fig).update_layout(yaxis_title=None, xaxis_title="Toneladas esperadas (aprox.)", height=440,
                                     xaxis_range=[0, float(tt["Toneladas totales (*)"].max()) * 1.3])
        redondear_barras(fig)
        mostrar_grafico(fig)
        st.caption("Suma de las parcelas con estimación (rendimiento × superficie registrada). Útil para planear "
                   "volumen de acopio o de crédito por zona.")

# ==============================================================================
# 3. TABLA Y DESCARGAS
# ==============================================================================
with tab_tabla:
    st.markdown("Filtra por estado, municipio o confianza y descarga los resultados. El **rango esperado** indica "
                "dónde caería el rendimiento real en **8 de cada 10** parcelas.")
    st.markdown(limpiar_html("""
        <div class="leyenda">
        <div class="col-exp"><b>Rendimiento estimado</b><span>Toneladas por hectárea (t/ha) que se esperan.</span></div>
        <div class="col-exp"><b>Mínimo / Máximo esperado</b><span>Rango en el que caería el rendimiento real en 8 de cada 10 casos.</span></div>
        <div class="col-exp"><b>Toneladas totales (*)</b><span>Rendimiento × superficie registrada: cosecha total aproximada.</span></div>
        <div class="col-exp"><b>Margen de error (± t)</b><span>Cuántas toneladas se suele equivocar el modelo en una parcela de ese municipio y tamaño.</span></div>
        </div>"""), unsafe_allow_html=True)

    t1, t2, t3, t4 = st.columns(4)
    t_edo = t1.multiselect("Estado", sorted(pred_df["Estado"].unique()), placeholder="Todos", key="t_edo")
    base_t = pred_df[pred_df["Estado"].isin(t_edo)] if t_edo else pred_df
    t_muni = t2.multiselect("Municipio", sorted(base_t["Municipio"].unique()), placeholder="Todos", key="t_muni")
    t_conf = t3.multiselect("Confianza", ORDEN_CONF, placeholder="Todas", key="t_conf")
    buscar = t4.text_input("Buscar ID de parcela", placeholder="Escribe parte del ID")
    vista = base_t
    if t_muni:
        vista = vista[vista["Municipio"].isin(t_muni)]
    if t_conf:
        vista = vista[vista["Confianza"].isin(t_conf)]
    if buscar.strip():
        vista = vista[vista["ID parcela"].str.contains(buscar.strip(), case=False, na=False, regex=False)]

    if vista.empty:
        st.warning("Ninguna parcela cumple con los filtros elegidos.")
    else:
        st.caption(f"{len(vista)} parcelas · {vista['Toneladas totales (*)'].sum():,.0f} toneladas esperadas en total")
        mostrar_tabla(pintar_confianza(formatear(vista, FORMATO_PRED)))
        st.caption("(*) Toneladas = rendimiento estimado × superficie del terreno registrado, que puede no ser la sembrada.")

    st.markdown("#### Descargas")
    d1, d2, d3 = st.columns(3)
    d1.download_button(
        "⬇️ Tabla oficial de predicciones", C.anexo_csv(R["anexo"]), "TABLA_ANEXO_predicciones.csv", "text/csv",
        help=f"Formato oficial: ID_parcela y Rendimiento_predicho_ton_ha con 4 decimales ({len(test)} parcelas).",
        width="stretch")
    detalle = test[[ID, EDO, MUNI, "area_ha", "pred", "min_esp", "max_esp", "confianza"]].rename(columns={
        ID: "ID_parcela", "area_ha": "Area_ha", "pred": "Rendimiento_ton_ha",
        "min_esp": "Minimo_esperado_80", "max_esp": "Maximo_esperado_80", "confianza": "Confianza"})
    d2.download_button(
        "⬇️ Detalle con rangos y confianza",
        detalle.sort_values("ID_parcela").to_csv(index=False, float_format="%.4f").encode("utf-8-sig"),
        "predicciones_detalle.csv", "text/csv", width="stretch")
    d3.download_button(
        "⬇️ Tabla filtrada (lo que ves arriba)", vista.to_csv(index=False, float_format="%.4f").encode("utf-8-sig"),
        "predicciones_filtradas.csv", "text/csv", width="stretch")
    st.caption("La **tabla oficial** es exactamente la que se entrega en el anexo del reporte.")

# ==============================================================================
# 4. PARA TECNICOS
# ==============================================================================
with tab_tecnicos:
    st.markdown("### La idea, sin fórmulas")
    st.markdown(
        "El análisis mostró que **el lugar (municipio) es lo que más define el rendimiento**. Probamos más de cien "
        "variables de satélite, clima y topografía, y ninguna mejoró la estimación una vez que se conoce el municipio. "
        "Por eso el modelo es deliberadamente simple:\n\n"
        "1. Se parte del **promedio de las parcelas conocidas del municipio**.\n"
        "2. Se mezcla un poco con el **promedio del estado**, para no confiar de más en un municipio con pocas parcelas.\n"
        "3. **Mientras más parcelas conocidas, más pesa el municipio.**\n"
        "4. Si el municipio no tiene parcelas conocidas, se usa el promedio del estado.")
    st.markdown("<div class='caja-analogia'><b>Analogía:</b> si quieres saber cuánto rinde una parcela nueva, lo "
                "sensato es ver cuánto rindieron sus vecinas del mismo municipio. Pero si solo conoces una o dos, su "
                "promedio puede salir alto o bajo por puro azar; entonces lo moderas con el promedio del estado. "
                "Eso hace el modelo.</div>", unsafe_allow_html=True)

    st.markdown("### Pruébalo tú")
    st.caption("Este simulador es solo para entender la fórmula. **No cambia las predicciones oficiales**, que siempre usan k = 1.")
    s1, s2 = st.columns(2)
    n_demo = s1.slider("Parcelas conocidas en el municipio (n)", 0, 80, 3, key="demo_n")
    m_demo = s1.slider("Promedio del municipio (t/ha)", 1.0, 6.0, 4.5, 0.05, key="demo_m")
    e_demo = s2.slider("Promedio del estado (t/ha)", 1.0, 6.0, 3.8, 0.05, key="demo_e")
    k_demo = s2.slider("Fuerza de contracción k", 0.0, 10.0, 1.0, 0.5, key="demo_k")
    peso = 0.0 if n_demo == 0 else (1.0 if n_demo + k_demo == 0 else n_demo / (n_demo + k_demo))
    r1, r2, r3 = st.columns(3)
    r1.metric("Estimación resultante", f"{peso * m_demo + (1 - peso) * e_demo:.2f} t/ha")
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
            f"(una sola parcela conocida) se estimaría copiando ese único valor. El costo es menor a 0.005 t/ha.")

    st.markdown("---")
    st.markdown("### ¿Qué tan bien predice?")
    st.markdown("Se **esconde** el rendimiento de parte de las parcelas conocidas, se estima con el resto y se compara "
                "con lo real. Se repitió 10 veces con distintas particiones (validación cruzada estratificada por "
                "municipio, 5 pliegues).")
    q1, q2, q3 = st.columns(3)
    q1.metric("MAE (error promedio)", f"{M['MAE']:.3f} t/ha", f"± {M['MAE_sd']:.3f} entre repeticiones",
              delta_color="off", delta_arrow="off")
    q2.metric("RMSE", f"{M['RMSE']:.3f} t/ha", f"± {M['RMSE_sd']:.3f}", delta_color="off", delta_arrow="off",
              help="Penaliza más los errores grandes.")
    q3.metric("R²", f"{M['R2']:.3f}", f"± {M['R2_sd']:.3f}", delta_color="off", delta_arrow="off")
    st.markdown(
        f"- **R² = {M['R2']:.2f}**: el modelo explica el {M['R2'] * 100:.0f}% de la variación **total** del rendimiento, "
        f"y **toda** esa capacidad viene de distinguir un municipio de otro. El máximo posible para cualquier modelo "
        f"que solo use el municipio es {M['techo']:.3f}; alcanzamos el {M['R2'] / M['techo'] * 100:.0f}% de ese techo. "
        "Dentro de un mismo municipio, el modelo no distingue parcelas buenas de malas.\n"
        f"- **El error promedio de {M['MAE']:.2f} t/ha** equivale a ~{error_rel:.1f}% del rendimiento promedio "
        f"({M['media_train']:.2f} t/ha). En regresión no existe un \"porcentaje de acierto\": es un error "
        "promedio en t/ha.\n"
        f"- Estas cifras son de **validación** con las {len(train)} parcelas conocidas. El error real sobre las "
        f"{len(test)} parcelas con estimación todavía no se conoce.")

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
                      annotation_text="8 de cada 10 casos", annotation_position="top left")
        fig.add_vline(x=0, line_dash="dash", line_color=ROJO)
        estilizar(fig).update_layout(height=400, xaxis_title="Error (real − estimado, t/ha)", yaxis_title="Parcelas")
        mostrar_grafico(fig)
        st.caption(f"En 8 de cada 10 parcelas conocidas, el error quedó entre {lo_g:+.2f} y {hi_g:+.2f} t/ha "
                   "(rango general). En la ficha de cada parcela se usa el rango de su municipio cuando tiene "
                   "10 o más parcelas medidas.")

    st.markdown("---")
    st.markdown("### ¿Cuándo se puede usar?")
    st.markdown(
        "Como el modelo solo usa el municipio y el estado, **puede dar una estimación desde antes de la siembra**, "
        "sin esperar imágenes de satélite ni datos de clima del ciclo. La condición: en un uso real, los promedios "
        f"municipales vendrían del ciclo anterior. Aquí se calcularon con el ciclo {CICLO}, el único disponible, así que "
        "no se pudo comprobar que se mantengan de un año a otro. Las imágenes de satélite sí sirven durante el ciclo "
        "para seguir el cultivo (emergencia, máximo verdor, maduración) y detectar anomalías.")

    with st.expander("Limitaciones"):
        st.markdown(
            "- No distingue parcelas del mismo municipio: no hay datos de variedad, fecha o dosis de siembra, rotación "
            "ni siniestros por parcela.\n"
            "- Las parcelas excepcionalmente altas o bajas se estiman hacia el promedio de su zona.\n"
            "- El rendimiento se pesó en báscula sin ajuste por humedad, impurezas ni mermas, y una parte se reporta "
            "redondeada a 0.5 t/ha. Parte del error no se puede eliminar con ningún modelo.\n"
            "- La superficie es la del terreno registrado, no necesariamente la sembrada; las toneladas son aproximadas.\n"
            f"- Solo hay datos del ciclo {CICLO}. Para otro ciclo hay que volver a calcular los promedios.\n"
            "- Chignahuapan concentra 29 de las 59 parcelas y es el municipio más variable: pesa mucho en el error final.\n"
            "- **k oficial = 1:** las predicciones de entrega no se modifican desde la interfaz.")

# ------------------------------------------------------------------------------
# Pie de pagina
# ------------------------------------------------------------------------------
st.markdown(f'<div class="footer-app">AgroCebada · Reto FIRA 2026 — modelo jerárquico municipio → estado (k = 1) · '
            f"datos del ciclo {CICLO} · error promedio ±{mae:.2f} t/ha</div>", unsafe_allow_html=True)
