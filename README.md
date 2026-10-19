# AgroCebada FIRA 2026 — App de rendimiento estimado de cebada

Mapa interactivo con el rendimiento estimado (t/ha) de las 59 parcelas de evaluación.
Usa el modelo jerárquico municipio → estado (`modelo_agrocebada.py`, k = 1).

## Archivos

| Archivo | Para qué |
|---|---|
| `app_agrocebada.py` | La app (Streamlit) |
| `calculos.py` | Validación, confianza, rangos y tabla oficial (sin interfaz) |
| `modelo_agrocebada.py` | El modelo entregado |
| `dataset_maestro.csv` | ID, estado, municipio, área y rendimiento de las 197 parcelas |
| `parcelas_centroides.csv` | Centro (lon, lat) de cada parcela |
| `parcelas.geojson` | Opcional: forma real de las parcelas (ver paso 2) |
| `preparar_datos.py` | Genera `parcelas.geojson` desde el shapefile oficial |
| `requirements.txt` | Librerías con las versiones probadas |

## 1. Correr la app

```bash
pip install -r requirements.txt
streamlit run app_agrocebada.py
```

## 2. (Recomendado) Dibujar la forma real de las parcelas

Se hace una sola vez, en una computadora con geopandas, con el shapefile del dataset
(`Parcelas_Reto_AGC_CONJUNTO.shp` junto con sus `.shx`, `.dbf` y `.prj`):

```bash
pip install geopandas
python preparar_datos.py Parcelas_Reto_AGC_CONJUNTO.shp
```

Genera `parcelas.geojson` (y regenera `parcelas_centroides.csv`). Súbelos junto con la app.
Sin `parcelas.geojson` la app funciona igual y dibuja cada parcela como un punto.

## 3. Publicar en Streamlit Community Cloud

1. Subir esta carpeta a un repositorio público de GitHub (incluida la carpeta `.streamlit`).
2. En share.streamlit.io: *New app* → elegir el repositorio → archivo principal `app_agrocebada.py`.
3. Abrir el enlace en una ventana de incógnito para confirmar que el jurado puede verla sin cuenta.

## Notas de interpretación

- Todo lo oficial (mapa, ficha, tabla descargable) usa **k = 1**. El simulador de "Cómo funciona" es solo didáctico.
- La tabla descargable *Tabla oficial de predicciones* es idéntica al anexo: `ID_parcela`, `Rendimiento_predicho_ton_ha`, 4 decimales.
- Las métricas (MAE 0.332, RMSE 0.499, R² 0.654) son el promedio de 10 repeticiones de validación cruzada estratificada, igual que en el reporte.
- **Confianza** = qué tanto se equivocó el modelo en cada municipio durante la validación (Alta < 0.25 t/ha; Media < 0.35; Baja ≥ 0.35; Muy baja con menos de 3 parcelas conocidas). No depende del número de parcelas.
- **Rango al 80%**: por municipio cuando hay 10 o más parcelas conocidas; si no, el rango general del modelo (−0.52 a +0.53 t/ha).
