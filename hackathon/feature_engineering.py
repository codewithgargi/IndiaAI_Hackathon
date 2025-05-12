import geopandas as gpd
from shapely.geometry import Point

def extract_all_features(mines, geo_data):
    print("🚀 Extracting features for mines...")

    # 🗺️ Reproject to UTM (EPSG:32643 covers most of South India)
    print("⚠️ Reprojecting to UTM for accurate distance calculations...")
    mines = mines.to_crs(epsg=32643)

    # ➕ Add placeholder columns
    mines["distance_to_fault"] = -1.0
    mines["distance_to_fold"] = -1.0
    mines["lithology"] = None
    mines["num_nearby_lineaments"] = 0
    mines["avg_chem_value"] = None

    # ⚡ Extract distances from fault and fold layers
    for name in geo_data:
        if "fault" in name.lower():
            fault = geo_data[name].to_crs(epsg=32643)
            mines["distance_to_fault"] = mines.geometry.apply(lambda x: fault.distance(x).min())
        elif "fold" in name.lower():
            fold = geo_data[name].to_crs(epsg=32643)
            mines["distance_to_fold"] = mines.geometry.apply(lambda x: fold.distance(x).min())

    # 🌍 Join lithology info if available
    for name in geo_data:
        if "lithology" in name.lower():
            litho = geo_data[name]
            if "lithology" in litho.columns:
                litho = litho.to_crs(epsg=32643)
                litho_subset = litho[["geometry", "lithology"]]
                mines = gpd.sjoin(mines, litho_subset, how="left", predicate="intersects").drop(columns=["index_right"])
                break
            else:
                print("⚠️ Warning: 'lithology' column not found in lithology layer. Using only geometry.")

    # ➕ Count nearby lineaments (within 1 km)
    for name in geo_data:
        if "lineament" in name.lower():
            lineaments = geo_data[name].to_crs(epsg=32643)
            buffer = mines.geometry.buffer(1000)
            mines["num_nearby_lineaments"] = buffer.apply(lambda buf: lineaments.intersects(buf).sum())
            break

    # 🧪 Add average geochemistry values nearby (within 1 km)
    for name in geo_data:
        if "geochem" in name.lower():
            geochem = geo_data[name]
            if geochem.geometry.geom_type.iloc[0] != "Point":
                geochem = geochem.copy()
                geochem["geometry"] = geochem.centroid
            geochem = geochem.to_crs(epsg=32643)
            buffer = mines.geometry.buffer(1000)

            avg_vals = []
            for buf in buffer:
                points_within = geochem[geochem.intersects(buf)]
                if not points_within.empty:
                    num_cols = points_within.select_dtypes(include='number')
                    avg_vals.append(num_cols.mean(axis=1).mean())
                else:
                    avg_vals.append(None)
            mines["avg_chem_value"] = avg_vals
            break

    return mines
