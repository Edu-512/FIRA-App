# AgroCebada FIRA 2026 — App de rendimiento estimado de cebada

App con la cosecha estimada de las 59 parcelas de evaluación, organizada por público en 4 pestañas:

| Pestaña | Para quién | Qué tiene |
|---|---|---|
| Mi parcela | Productor | Mapa a todo lo ancho; al tocar una parcela se abre su ficha flotante (toneladas, rango, calculadora de ingreso, confianza) |
| Panorama | Analista / FIRA | Resumen, confianza por municipio, variación entre parcelas, cosecha total |
| Tabla y descargas | Todos | Tabla de las 59 parcelas y la tabla oficial del anexo |
| Para técnicos | Jurado técnico | Cómo funciona el modelo, simulador, métricas de validación, limitaciones |

Usa el modelo jerárquico municipio → estado (`modelo_agrocebada.py`, k = 1).

## Archivos

| Archivo | Para qué |
|---|---|
| `app_agrocebada.py` | La app (Streamlit) |
| `calculos.py` | Validación, confianza, rangos y tabla oficial (sin interfaz) |
| `modelo_agrocebada.py` | El modelo entregado |
| `dataset_maestro.csv` | ID, estado, municipio, área y rendimiento de las 197 parcelas |
| `parcelas_centroides.csv` | Centro (lon, lat) de cada parcela |
| `parcelas.geojson` | Forma real de las 197 parcelas (ya incluido; se generó con el paso 2) |
| `preparar_datos.py` | Genera `parcelas.geojson` desde el shapefile oficial |
| `requirements.txt` | Librerías con las versiones probadas |

## 1. Correr la app

```bash
pip install -r requirements.txt
streamlit run app_agrocebada.py
```

## 2. Regenerar la forma real de las parcelas (solo si cambia el shapefile)

`parcelas.geojson` ya está incluido. Este paso se hace una sola vez, en una computadora con geopandas, con el shapefile del dataset
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

- Todo lo oficial (mapa, ficha, tabla descargable) usa **k = 1**. El simulador de la pestaña "Para técnicos" es solo didáctico.
- La tabla descargable *Tabla oficial de predicciones* es idéntica al anexo: `ID_parcela`, `Rendimiento_predicho_ton_ha`, 4 decimales.
- Las métricas (MAE 0.332, RMSE 0.499, R² 0.654) son el promedio de 10 repeticiones de validación cruzada estratificada, igual que en el reporte.
- **Confianza** = qué tanto se equivocó el modelo en cada municipio durante la validación (Alta < 0.25 t/ha; Media < 0.35; Baja ≥ 0.35; Muy baja con menos de 3 parcelas conocidas). No depende del número de parcelas.
- **Rango al 80%**: por municipio cuando hay 10 o más parcelas conocidas; si no, el rango general del modelo (−0.52 a +0.53 t/ha).
