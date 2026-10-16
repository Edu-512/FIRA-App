"""
================================================================================
Reto AgroCebada FIRA 2026
MODELO PREDICTIVO DE RENDIMIENTO DE CEBADA
================================================================================

MODELO: Estimador jerarquico de medias con contraccion (shrinkage) empirica.

    prediccion = (n * promedio_municipio + k * promedio_estado) / (n + k)

    n = numero de parcelas conocidas en ese municipio
    k = 1  (fuerza de la contraccion, elegida por validacion cruzada)

JUSTIFICACION:
    El analisis exploratorio demostro que la variable determinante del
    rendimiento es la ZONA (municipio). Se probaron 79 variables derivadas
    de 23 indices satelitales, ademas de clima y topografia, y ninguna
    aporto senal medible una vez conocido el municipio. Se compararon
    Ridge, Lasso, Random Forest y Gradient Boosting: ninguno supero a
    este estimador. La contraccion hacia el estado protege a los
    municipios con muy pocas parcelas conocidas.

DESEMPENO (validacion cruzada estratificada por municipio, 10 repeticiones):
    MAE  = 0.332 +- 0.004 t/ha   (~8.4% de error relativo)
    RMSE = 0.499 +- 0.004 t/ha
    R2   = 0.654 +- 0.005        (explica el 65% de la variacion)
    Intervalo empirico 80%: [-0.520, +0.528] t/ha

USO:
    python modelo_agrocebada.py
    o bien:  from modelo_agrocebada import ModeloAgroCebada
================================================================================
"""
import pandas as pd
import numpy as np


class ModeloAgroCebada:
    """Estimador jerarquico municipio -> estado con contraccion."""

    def __init__(self, k: float = 1.0):
        self.k = k
        self.media_global_ = None
        self.medias_municipio_ = None   # media y conteo por municipio
        self.medias_estado_ = None
        self.residuos_ = None           # para los intervalos

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, col_y="RENDIMIENTO_T_HA",
            col_muni="Municipio", col_edo="Estado"):
        """Aprende las medias por municipio y por estado."""
        d = df[df[col_y].notna()].copy()
        d[col_muni] = d[col_muni].astype(str)
        d[col_edo] = d[col_edo].astype(str)

        self.media_global_ = d[col_y].mean()
        self.medias_municipio_ = d.groupby(col_muni)[col_y].agg(["mean", "count"])
        self.medias_estado_ = d.groupby(col_edo)[col_y].mean()
        self.col_y, self.col_muni, self.col_edo = col_y, col_muni, col_edo
        return self

    # ------------------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predice el rendimiento (t/ha) de cada parcela."""
        pred = []
        for _, fila in df.iterrows():
            muni = str(fila[self.col_muni])
            edo = str(fila[self.col_edo])
            media_edo = self.medias_estado_.get(edo, self.media_global_)

            if muni in self.medias_municipio_.index:
                m = self.medias_municipio_.loc[muni, "mean"]
                n = self.medias_municipio_.loc[muni, "count"]
                pred.append((n * m + self.k * media_edo) / (n + self.k))
            else:
                # municipio nunca visto -> se usa el promedio de su estado
                pred.append(media_edo)
        return np.array(pred)

    # ------------------------------------------------------------------
    def intervalo(self, pred: np.ndarray, nivel=80):
        """Intervalo empirico alrededor de la prediccion (requiere calibrar)."""
        if self.residuos_ is None:
            raise RuntimeError("Llama primero a calibrar_intervalo()")
        a = (100 - nivel) / 2
        lo, hi = np.percentile(self.residuos_, [a, 100 - a])
        return pred + lo, pred + hi

    def calibrar_intervalo(self, df, n_semillas=10):
        """Calcula los residuos por validacion cruzada para los intervalos."""
        from sklearn.model_selection import StratifiedKFold
        d = df[df[self.col_y].notna()].copy().reset_index(drop=True)
        y = d[self.col_y].values
        cnt = d[self.col_muni].value_counts()
        strat = d[self.col_muni].where(d[self.col_muni].map(cnt) >= 5, "otros")
        todos = []
        for s in range(n_semillas):
            skf = StratifiedKFold(5, shuffle=True, random_state=s)
            p = np.zeros(len(y))
            for i, j in skf.split(d, strat):
                p[j] = ModeloAgroCebada(self.k).fit(d.iloc[i]).predict(d.iloc[j])
            todos.append(p)
        self.residuos_ = y - np.mean(todos, axis=0)
        return self


# ======================================================================
if __name__ == "__main__":
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    d = pd.read_csv("dataset_maestro.csv")
    d["Municipio"] = d["Municipio"].astype(str)
    d["Estado"] = d["Estado"].astype(str)
    train = d[d["RENDIMIENTO_T_HA"].notna()].copy()
    test = d[d["RENDIMIENTO_T_HA"].isna()].copy()

    modelo = ModeloAgroCebada(k=1).fit(train).calibrar_intervalo(train)

    # desempeno en validacion
    y = train["RENDIMIENTO_T_HA"].values
    pred_cv = y - modelo.residuos_
    print("DESEMPENO (validacion cruzada estratificada, 10 repeticiones)")
    print(f"  MAE  = {mean_absolute_error(y, pred_cv):.4f} t/ha")
    print(f"  RMSE = {np.sqrt(mean_squared_error(y, pred_cv)):.4f} t/ha")
    print(f"  R2   = {r2_score(y, pred_cv):+.4f}")

    # predicciones finales
    pred = modelo.predict(test)
    lo, hi = modelo.intervalo(pred, nivel=80)

    entrega = pd.DataFrame({
        "ID_parcela": test["ID_POLIGON"].values,
        "Rendimiento_predicho_ton_ha": np.round(pred, 4),
    }).sort_values("ID_parcela").reset_index(drop=True)
    entrega.to_csv("TABLA_ANEXO_predicciones.csv", index=False)

    detalle = pd.DataFrame({
        "ID_parcela": test["ID_POLIGON"].values,
        "Rendimiento_predicho_ton_ha": np.round(pred, 4),
        "IC80_min": np.round(lo, 4), "IC80_max": np.round(hi, 4),
        "Municipio": test["Municipio"].values, "Estado": test["Estado"].values,
    }).sort_values("ID_parcela").reset_index(drop=True)
    detalle.to_csv("predicciones_detalle.csv", index=False)

    print(f"\nPredicciones generadas: {len(entrega)} parcelas")
    print(f"  Rango: {pred.min():.4f} a {pred.max():.4f} t/ha")
    print("  Archivos: TABLA_ANEXO_predicciones.csv / predicciones_detalle.csv")
    print("\nPrimeras filas:")
    print(detalle.head(6).to_string(index=False))
