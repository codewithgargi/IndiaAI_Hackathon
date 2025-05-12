import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from feature_engineering import extract_all_features
from utils import load_all_geospatial_files

def generate_grid_points(bounds, resolution_meters=5000):
    """Generate a grid of points within the given bounds at specified resolution"""
    min_x, min_y, max_x, max_y = bounds
    
    # Create grid coordinates
    x_coords = np.arange(min_x, max_x, resolution_meters)
    y_coords = np.arange(min_y, max_y, resolution_meters)
    
    # Create all combinations of coordinates
    points = []
    for x in x_coords:
        for y in y_coords:
            points.append(Point(x, y))
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:32643")
    return gdf

def create_mineral_potential_map(base_path, model_path, output_path, resolution_km=5):
    """Create a mineral potential prediction map"""
    print("🌐 Loading geospatial data...")
    geo_data = load_all_geospatial_files(base_path)
    
    # Find a reference layer to determine the study area extent
    ref_layer = None
    for key in geo_data:
        if "lithology" in key.lower() or "geology" in key.lower():
            ref_layer = geo_data[key].to_crs(epsg=32643)
            break
    
    if ref_layer is None:
        print("⚠️ No reference layer found for extent. Using first available layer...")
        ref_layer = list(geo_data.values())[0].to_crs(epsg=32643)
    
    # Get bounds of the study area
    bounds = ref_layer.total_bounds
    
    # Generate grid points
    print(f"📍 Generating grid points at {resolution_km}km resolution...")
    resolution_meters = resolution_km * 1000
    grid = generate_grid_points(bounds, resolution_meters)
    
    # Extract features for each grid point
    print("🔍 Extracting features for grid points...")
    grid_with_features = extract_all_features(grid, geo_data)
    
    # Load the trained model
    print("🧠 Loading trained model...")
    model = joblib.load(model_path)
    
    # Prepare features for prediction
    feature_cols = [col for col in grid_with_features.columns 
                   if col not in ['geometry', 'id', 'name', 'mine_id']]
    
    # Handle all categorical features (not just lithology)
    categorical_cols = []
    for col in feature_cols:
        if col in grid_with_features.columns and grid_with_features[col].dtype == 'object':
            print(f"⚠️ Found categorical column: {col}")
            categorical_cols.append(col)
    
    for col in categorical_cols:
        if col in grid_with_features.columns:
            print(f"🔄 Converting {col} to dummy variables")
            grid_with_features[col] = grid_with_features[col].fillna('unknown')
            dummies = pd.get_dummies(grid_with_features[col], prefix=col[:5])
            grid_with_features = pd.concat([grid_with_features, dummies], axis=1)
            # Remove the original categorical column from feature list
            feature_cols.remove(col)
            # Add the new dummy columns to feature list
            feature_cols.extend(dummies.columns.tolist())
    
    # Handle missing values
    for col in grid_with_features.select_dtypes(include=[np.number]).columns:
        if col in feature_cols:
            grid_with_features[col] = grid_with_features[col].fillna(grid_with_features[col].median())
    
    # Prepare feature matrix for prediction - only numeric columns
    print("📊 Selecting numeric features only...")
    numeric_feature_cols = []
    for col in feature_cols:
        if col in grid_with_features.columns:
            if np.issubdtype(grid_with_features[col].dtype, np.number):
                numeric_feature_cols.append(col)
            else:
                print(f"⚠️ Skipping non-numeric column: {col}")
    
    print(f"🔢 Using {len(numeric_feature_cols)} numeric features")
    X = grid_with_features[numeric_feature_cols]
    
    # Get model's feature names
    model_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
    
    # If model expects specific features, align our feature matrix
    if model_features is not None:
        print("🔄 Aligning features with trained model...")
        # Add missing features as zeros
        for feature in model_features:
            if feature not in X.columns:
                print(f"⚠️ Adding missing feature: {feature}")
                X[feature] = 0
        
        # Select only features used in the model
        X = X[model_features]
    
    # Make predictions
    print("🔮 Predicting mineral potential...")
    # The key fix: Handle the prediction correctly
    try:
        # If the model returns probability estimates for binary classification
        proba = model.predict_proba(X)
        # Get the probability for the positive class (index 1)
        if proba.shape[1] >= 2:
            grid_with_features['potential_score'] = proba[:, 1]
        else:
            grid_with_features['potential_score'] = proba[:, 0]
    except (AttributeError, IndexError):
        # If predict_proba is not available or other issues
        print("⚠️ predict_proba not available, using predict instead")
        grid_with_features['potential_score'] = model.predict(X)
    
    # Save the grid with predictions
    grid_with_features.to_file(os.path.join(output_path, "mineral_potential_grid.geojson"), driver="GeoJSON")
    
    # Create a static map using matplotlib
    print("🗺️ Creating static potential map...")
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot base reference layer
    ref_layer.plot(ax=ax, color='lightgrey', edgecolor='grey', alpha=0.5)
    
    # Plot potential as points with color gradient
    grid_with_features.plot(
        column='potential_score',
        ax=ax,
        cmap='viridis',
        markersize=20,
        legend=True,
        legend_kwds={'label': "Mineral Potential Score"}
    )

    # Add known mines if available
    for key in geo_data:
        if "mine" in key.lower() or "quarry" in key.lower():
            mines = geo_data[key].to_crs(epsg=32643)
            mines.plot(ax=ax, color='red', markersize=30, marker='*', label='Known Mines')
            break
    
    plt.title('Mineral Potential Prediction Map')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "mineral_potential_map.png"), dpi=300)
    
    # Create interactive map using Plotly
    print("🌐 Creating interactive map with Plotly...")
    
    try:
        # Convert to WGS84 for mapping
        grid_wgs84 = grid_with_features.to_crs(epsg=4326)
        
        # Extract coordinates for plotting
        grid_wgs84['lon'] = grid_wgs84.geometry.x
        grid_wgs84['lat'] = grid_wgs84.geometry.y
        
        # Create a dataframe for plotting
        plot_df = pd.DataFrame({
            'latitude': grid_wgs84['lat'],
            'longitude': grid_wgs84['lon'],
            'potential_score': grid_wgs84['potential_score']
        })
        
        # Find mines if available
        mines_df = None
        for key in geo_data:
            if "mine" in key.lower() or "quarry" in key.lower():
                try:
                    mines = geo_data[key].to_crs(epsg=4326)
                    # Extract mine coordinates
                    mines_df = pd.DataFrame({
                        'latitude': [geom.y for geom in mines.geometry],
                        'longitude': [geom.x for geom in mines.geometry],
                        'name': mines['name'] if 'name' in mines.columns else ['Mine'] * len(mines)
                    })
                    print(f"✅ Found {len(mines_df)} mines/quarries")
                except Exception as e:
                    print(f"⚠️ Error processing mines: {e}")
                break
        
        # Using a more compatible approach: Creating a figure with two traces
        fig = go.Figure()
        
        # Add potential score points
        fig.add_trace(go.Scattergeo(
            lat=plot_df['latitude'],
            lon=plot_df['longitude'],
            mode='markers',
            marker=dict(
                size=10,
                color=plot_df['potential_score'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Potential Score"),
                cmin=0,
                cmax=1
            ),
            name='Potential Score'
        ))
        
        # Add mine locations if available
        if mines_df is not None and not mines_df.empty:
            fig.add_trace(go.Scattergeo(
                lat=mines_df['latitude'],
                lon=mines_df['longitude'],
                mode='markers',
                marker=dict(
                    size=15,
                    color='red',
                    symbol='star'
                ),
                name='Known Mines',
                text=mines_df['name'],
                hoverinfo='text'
            ))
            print(f"✅ Added {len(mines_df)} mines to the map")
        
        # Update layout
        fig.update_layout(
            title='Mineral Potential Prediction Map',
            title_x=0.5,
            height=800,
            geo=dict(
                scope='asia',  # Adjust based on your region
                showland=True,
                landcolor='rgb(243, 243, 243)',
                countrycolor='rgb(204, 204, 204)',
                showcoastlines=True,
                coastlinecolor='rgb(204, 204, 204)',
                center=dict(
                    lat=plot_df['latitude'].mean(),
                    lon=plot_df['longitude'].mean()
                ),
                projection_scale=8  # Zoom level
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            margin={"r": 0, "t": 50, "l": 0, "b": 0}
        )
        
        # Add color bar title
        fig.update_coloraxes(colorbar_title="Potential Score")
        
        # Save as HTML file
        plot_path = os.path.join(output_path, "interactive_potential_map_plotly.html")
        fig.write_html(plot_path)
        print(f"✅ Interactive map saved to {plot_path}")
        
    except Exception as e:
        print(f"❌ Error creating plotly map: {e}")
    
    print(f"✅ All maps saved to {output_path}")
    
    return grid_with_features

if __name__ == "__main__":
    base_path = "C:/Users/Aligned Studios/Downloads/hackathon"
    model_path = "best_mineral_potential_model.pkl"
    output_path = os.path.join(base_path, "results")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    try:
        # Check if plotly is installed
        try:
            import plotly
            print(f"✅ Plotly is installed (version {plotly.__version__})")
        except ImportError:
            print("❌ Plotly is not installed. Please install it with:")
            print("pip install plotly")
            exit(1)
            
        # Generate potential map
        potential_grid = create_mineral_potential_map(base_path, model_path, output_path)
        print("✅ Process completed successfully")
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {e}")