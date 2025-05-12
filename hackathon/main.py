import os
from utils import load_all_geospatial_files
from feature_engineering import extract_all_features

# 📁 Base directory containing your 4 folders
base_path = "C:/Users/Aligned Studios/Downloads/dataset"

# ✅ Load all geospatial files from the folder
geo_data = load_all_geospatial_files(base_path)
print("📦 Loaded layers:", list(geo_data.keys()))
print("🔍 Available layers:", list(geo_data.keys()))

# 🔍 Dynamically find mine/quarry layer
mine_layer_key = None
for key in geo_data.keys():
    if "mine" in key.lower() or "quarry" in key.lower():
        mine_layer_key = key
        break

if not mine_layer_key:
    raise ValueError("❌ No mine/quarry layer found!")

mines = geo_data[mine_layer_key]
print(f"✅ Using mine layer: {mine_layer_key} with {len(mines)} features")

# 🚀 Extract features
mines_with_features = extract_all_features(mines, geo_data)

# 💾 Save results
output_path = os.path.join(base_path, "mines_with_all_features.geojson")
mines_with_features.to_file(output_path, driver="GeoJSON")
print(f"✅ Feature extraction complete. Saved to: {output_path}")
