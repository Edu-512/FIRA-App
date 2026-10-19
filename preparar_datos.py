"""
preparar_datos.py - Reto AgroCebada FIRA 2026

Se corre UNA sola vez, en una computadora con geopandas, para convertir el
shapefile real de las parcelas en los dos archivos ligeros que lee la app:

    parcelas.geojson         poligonos reales de las 197 parcelas
    parcelas_centroides.csv  centro (lon, lat) de cada parcela

Asi la app desplegada no necesita geopandas (que suele fallar al instalarse
en la nube).

Uso:
    pip install geopandas
    python preparar_datos.py Parcelas_Reto_AGC_CONJUNTO.shp

El shapefile debe ir acompanado de sus archivos .shx, .dbf y .prj.
"""
import sys

import geopandas as gpd

ruta = sys.argv[1] if len(sys.argv) > 1 else "Parcelas_Reto_AGC_CONJUNTO.shp"
g = gpd.read_file(ruta)

# El formato shapefile corta los nombres de columna a 10 caracteres:
# ID_POLIGONO aparece como ID_POLIGON.
col_id = next((c for c in g.columns if c.upper() in ("ID_POLIGON", "ID_POLIGONO")), None)
if col_id is None:
    sys.exit(f"No encontre la columna de ID en el shapefile. Columnas: {list(g.columns)}")

g = g[[col_id, "geometry"]].rename(columns={col_id: "ID_POLIGON"})
g["ID_POLIGON"] = g["ID_POLIGON"].astype(str).str.strip().str.upper()

if g.crs is None:
    sys.exit("El shapefile no trae sistema de referencia (.prj). Revisa que el .prj este en la carpeta.")

# Centroides calculados en una proyeccion metrica de Mexico (EPSG:6372) y
# regresados a coordenadas geograficas
cent = g.to_crs(6372).geometry.centroid.to_crs(4326)
g = g.to_crs(4326)

g[["ID_POLIGON"]].assign(lon=cent.x.round(6), lat=cent.y.round(6)).to_csv(
    "parcelas_centroides.csv", index=False)

# Simplificacion ligera (~1 m) para que el archivo pese poco sin deformar las parcelas
g["geometry"] = g.geometry.simplify(0.00001, preserve_topology=True)
g.to_file("parcelas.geojson", driver="GeoJSON")

print(f"Listo: {len(g)} parcelas -> parcelas.geojson y parcelas_centroides.csv")
if len(g) != 197:
    print("AVISO: se esperaban 197 parcelas.")
