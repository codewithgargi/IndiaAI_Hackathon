import folium
import geopandas as gpd

# Load the enriched mine data
gdf = gpd.read_file("C:/Users/Aligned Studios/Downloads/dataset/mines_with_all_features.geojson")

# Calculate the center for the map
center = [gdf.geometry.y.mean(), gdf.geometry.x.mean()]

# Initialize the map
m = folium.Map(location=center, zoom_start=7, tiles="CartoDB positron")

# Add each mine as a marker
for _, row in gdf.iterrows():
    popup_info = "<br>".join([f"<b>{col}</b>: {row[col]}" for col in gdf.columns if col != "geometry"])
    folium.Marker(
        location=[row.geometry.y, row.geometry.x],
        popup=folium.Popup(popup_info, max_width=300),
        icon=folium.Icon(color='green', icon='info-sign')
    ).add_to(m)

# Save to HTML file
m.save("mines_map.html")
print("✅ Map saved as 'mines_map.html'")
