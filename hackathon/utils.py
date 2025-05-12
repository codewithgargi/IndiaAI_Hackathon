import os
import geopandas as gpd

def load_all_geospatial_files(base_path):
    geo_data = {}
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".shp") or file.endswith(".geojson") or file.endswith(".gpkg"):
                full_path = os.path.join(root, file)
                try:
                    gdf = gpd.read_file(full_path)
                    key = os.path.splitext(file)[0].lower()
                    geo_data[key] = gdf
                    print(f"✅ Loaded: {key} ({len(gdf)} features)")
                except Exception as e:
                    print(f"❌ Failed to load {file}: {e}")
    return geo_data
