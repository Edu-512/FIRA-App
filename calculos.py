"""
calculos.py - Reto AgroCebada FIRA 2026
Calculos de la app, separados de la interfaz para poder probarlos.

Todo lo OFICIAL usa k = 1 (el modelo entregado). Nada en este archivo
depende de controles de la interfaz.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from modelo_agrocebada import ModeloAgroCebada

K_OFICIAL = 1.0
N_SEMILLAS = 10          # igual que la validacion del reporte
N_PLIEGUES = 5
MIN_N_INTERVALO = 10     # parcelas conocidas minimas para un rango propio del municipio
NIVEL = 80               # rango esperado al 80%

Y, ID, MUNI, EDO = "RENDIMIENTO_T_HA", "ID_POLIGON", "Municipio", "Estado"


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def cargar(ruta_maestro="dataset_maestro.csv", ruta_centroides="parcelas_centroides.csv"):
    d = pd.read_csv(ruta_maestro)
    d[MUNI] = d[MUNI].astype(str).str.strip()
    d[EDO] = d[EDO].astype(str).str.strip()
    d[ID] = d[ID].astype(str).str.strip().str.upper()
    d[Y] = pd.to_numeric(d[Y], errors="coerce")
    try:
        c = pd.read_csv(ruta_centroides)
        c[ID] = c[ID].astype(str).str.strip().str.upper()
        d = d.merge(c[[ID, "lon", "lat"]], on=ID, how="left")
    except FileNotFoundError:
        d["lon"], d["lat"] = np.nan, np.nan
    return d


# ---------------------------------------------------------------------------
# Validacion cruzada (mismo esquema que modelo_agrocebada.calibrar_intervalo)
# ---------------------------------------------------------------------------
def validar(train: pd.DataFrame, k=K_OFICIAL, n_semillas=N_SEMILLAS):
    d = train.reset_index(drop=True)
    y = d[Y].to_numpy()
    cnt = d[MUNI].value_counts()
    estrato = d[MUNI].where(d[MUNI].map(cnt) >= 5, "otros")

    preds = []
    for s in range(n_semillas):
        skf = StratifiedKFold(N_PLIEGUES, shuffle=True, random_state=s)
        p = np.zeros(len(y))
        for i, j in skf.split(d, estrato):
            p[j] = ModeloAgroCebada(k).fit(d.iloc[i]).predict(d.iloc[j])
        preds.append(p)
    preds = np.array(preds)                      # (semillas, parcelas)

    # Metricas por semilla -> media y desviacion (asi se reportan en el reporte)
    filas = []
    for p in preds:
        e = y - p
        filas.append({
            "MAE": np.mean(np.abs(e)),
            "RMSE": np.sqrt(np.mean(e ** 2)),
            "R2": 1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2),
        })
    por_semilla = pd.DataFrame(filas)

    pred_prom = preds.mean(axis=0)
    resid = y - pred_prom                         # residuos promedio (para rangos)
    return {"por_semilla": por_semilla, "pred_cv": pred_prom, "residuos": resid, "y": y,
            "municipio": d[MUNI].to_numpy()}


def rango(residuos, nivel=NIVEL):
    a = (100 - nivel) / 2
    lo, hi = np.percentile(residuos, [a, 100 - a])
    return float(lo), float(hi)


def techo_municipio(train):
    """Proporcion de la varianza que esta ENTRE municipios (R2 maximo de un modelo por municipio)."""
    y = train[Y]
    dentro = ((y - train.groupby(MUNI)[Y].transform("mean")) ** 2).sum()
    total = ((y - y.mean()) ** 2).sum()
    return float(1 - dentro / total)


# ---------------------------------------------------------------------------
# Confianza: se basa en el ERROR REAL de validacion del municipio, no en n
# ---------------------------------------------------------------------------
def nivel_confianza(n, mae_muni):
    if n < 3 or mae_muni is None or np.isnan(mae_muni):
        return "Muy baja"
    if mae_muni < 0.25:
        return "Alta"
    if mae_muni < 0.35:
        return "Media"
    return "Baja"


EXPLICACION_CONF = {
    "Alta": "Las parcelas conocidas del municipio rinden muy parecido entre sí; el modelo acertó con errores pequeños.",
    "Media": "Las parcelas conocidas varían de forma moderada; el error típico es intermedio.",
    "Baja": "Las parcelas conocidas rinden muy distinto entre sí; cada parcela puede alejarse bastante del promedio.",
    "Muy baja": "Casi no hay parcelas conocidas del municipio; la estimación se apoya mucho en el promedio del estado.",
}


# ---------------------------------------------------------------------------
# Todo junto
# ---------------------------------------------------------------------------
def calcular_todo(d: pd.DataFrame):
    train = d[d[Y].notna()].reset_index(drop=True)
    test = d[d[Y].isna()].reset_index(drop=True)

    modelo = ModeloAgroCebada(k=K_OFICIAL).fit(train)
    val = validar(train)
    lo_g, hi_g = rango(val["residuos"])

    # --- tabla por municipio ---
    res = pd.DataFrame({MUNI: val["municipio"], "resid": val["residuos"], "y": val["y"]})
    mm = modelo.medias_municipio_
    filas = []
    for muni, g in res.groupby(MUNI):
        n = int(mm.loc[muni, "count"])
        edo = train.loc[train[MUNI] == muni, EDO].mode().iloc[0]
        media_edo = float(modelo.medias_estado_.get(edo, modelo.media_global_))
        est = float(modelo.predict(pd.DataFrame({MUNI: [muni], EDO: [edo]}))[0])
        mae_m = float(np.mean(np.abs(g["resid"])))
        if n >= MIN_N_INTERVALO:
            lo, hi, propio = *rango(g["resid"]), True
        else:
            lo, hi, propio = lo_g, hi_g, False
        filas.append({
            MUNI: muni, EDO: edo, "n": n,
            "a_predecir": int((test[MUNI] == muni).sum()),
            "prom_obs": float(mm.loc[muni, "mean"]),
            "desv": float(g["y"].std(ddof=1)) if n > 1 else np.nan,
            "min": float(g["y"].min()), "max": float(g["y"].max()),
            "media_estado": media_edo,
            "peso_muni": n / (n + K_OFICIAL),
            "estimacion": est,
            "mae_val": mae_m,
            "rango_lo": lo, "rango_hi": hi, "rango_propio": propio,
            "valores_distintos": int(g["y"].round(3).nunique()),
            "confianza": nivel_confianza(n, mae_m),
        })
    tm = pd.DataFrame(filas).sort_values("estimacion", ascending=False).reset_index(drop=True)

    # --- predicciones oficiales (k = 1) ---
    pred = modelo.predict(test)
    info = tm.set_index(MUNI)
    t = test.copy()
    t["pred"] = pred
    t["conocido"] = t[MUNI].isin(info.index)
    for col in ["n", "confianza", "rango_lo", "rango_hi", "rango_propio", "mae_val", "peso_muni"]:
        t[col] = t[MUNI].map(info[col])
    t["n"] = t["n"].fillna(0).astype(int)
    t["confianza"] = t["confianza"].fillna("Muy baja")
    t["rango_lo"] = t["rango_lo"].fillna(lo_g)
    t["rango_hi"] = t["rango_hi"].fillna(hi_g)
    t["rango_propio"] = t["rango_propio"].fillna(False)
    t["peso_muni"] = t["peso_muni"].fillna(0.0)
    t["min_esp"] = np.clip(t["pred"] + t["rango_lo"], 0, None)
    t["max_esp"] = t["pred"] + t["rango_hi"]
    t["toneladas"] = t["pred"] * t["area_ha"]
    t["error_ton"] = t["mae_val"].fillna(np.mean(np.abs(val["residuos"]))) * t["area_ha"]

    anexo = (pd.DataFrame({"ID_parcela": t[ID], "Rendimiento_predicho_ton_ha": np.round(t["pred"], 4)})
             .sort_values("ID_parcela").reset_index(drop=True))

    ps = val["por_semilla"]
    metricas = {
        "MAE": ps["MAE"].mean(), "MAE_sd": ps["MAE"].std(ddof=1),
        "RMSE": ps["RMSE"].mean(), "RMSE_sd": ps["RMSE"].std(ddof=1),
        "R2": ps["R2"].mean(), "R2_sd": ps["R2"].std(ddof=1),
        "media_train": float(train[Y].mean()),
        "techo": techo_municipio(train),
        "rango_global": (lo_g, hi_g),
    }
    metricas["error_rel"] = metricas["MAE"] / metricas["media_train"]
    return {"train": train, "test": t, "municipios": tm, "metricas": metricas,
            "val": val, "anexo": anexo, "modelo": modelo}


def anexo_csv(anexo: pd.DataFrame) -> bytes:
    return anexo.to_csv(index=False, float_format="%.4f").encode("utf-8")
