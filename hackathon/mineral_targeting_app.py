import os
import argparse
import time
from utils import load_all_geospatial_files
from feature_engineering import extract_all_features
from train_models import train_mineral_potential_models
from predict_potential import create_mineral_potential_map

def main():
    """
    India AI Hackathon - Mineral Targeting AI Application
    This script runs the entire pipeline for mineral potential prediction
    """
    parser = argparse.ArgumentParser(description="AI for Mineral Targeting - IndiaAI Hackathon")
    parser.add_argument("--data_path", type=str, default="./dataset", 
                        help="Path to the directory containing geospatial datasets")
    parser.add_argument("--output_path", type=str, default="./results", 
                        help="Path to save results")
    parser.add_argument("--resolution", type=float, default=2.0, 
                        help="Resolution for prediction grid in km")
    parser.add_argument("--skip_training", action="store_true", 
                        help="Skip model training if model already exists")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)
    
    print("=" * 60)
    print("🚀 AI for Mineral Targeting - IndiaAI Hackathon")
    print("=" * 60)
    
    start_time = time.time()
    
    # Step 1: Load all geospatial data
    print("\n📦 Step 1: Loading geospatial data...")
    geo_data = load_all_geospatial_files(args.data_path)
    print(f"✅ Loaded {len(geo_data)} geospatial layers")
    
    # Step 2: Find mine layer
    print("\n🔍 Step 2: Finding mine/quarry layer...")
    mine_layer_key = None
    for key in geo_data.keys():
        if "mine" in key.lower() or "quarry" in key.lower():
            mine_layer_key = key
            break
    
    if not mine_layer_key:
        print("❌ No mine/quarry layer found! Creating sample points...")
        # If no mine layer is found, we could create random sample points
        # for demonstration purposes, but this is not ideal
        from shapely.geometry import Point
        import geopandas as gpd
        import numpy as np
        
        # Use first layer to get extent
        ref_layer = list(geo_data.values())[0]
        bounds = ref_layer.total_bounds
        
        # Create random points
        np.random.seed(42)
        n_points = 100
        x = np.random.uniform(bounds[0], bounds[2], n_points)
        y = np.random.uniform(bounds[1], bounds[3], n_points)
        points = [Point(x[i], y[i]) for i in range(n_points)]
        
        # Create GeoDataFrame
        mines = gpd.GeoDataFrame(geometry=points, crs=ref_layer.crs)
        print(f"⚠️ Created {len(mines)} sample points for demonstration")
    else:
        mines = geo_data[mine_layer_key]
        print(f"✅ Using mine layer: {mine_layer_key} with {len(mines)} features")
    
    # Step 3: Extract features
    print("\n🧪 Step 3: Extracting features for mines...")
    mines_with_features = extract_all_features(mines, geo_data)
    
    # Save enriched mine data
    mines_output_path = os.path.join(args.output_path, "mines_with_all_features.geojson")
    mines_with_features.to_file(mines_output_path, driver="GeoJSON")
    print(f"✅ Feature extraction complete. Saved to: {mines_output_path}")
    
    # Step 4: Train models
    model_path = os.path.join(args.output_path, "best_mineral_potential_model.pkl")
    
    if args.skip_training and os.path.exists(model_path):
        print("\n⏩ Step 4: Skipping model training (using existing model)")
    else:
        print("\n🧠 Step 4: Training predictive models...")
        best_model, results, feature_names = train_mineral_potential_models(mines_output_path)
        print(f"✅ Model training complete")
    
    # Step 5: Generate potential maps
    print("\n🗺️ Step 5: Generating mineral potential maps...")
    potential_grid = create_mineral_potential_map(
        args.data_path, 
        model_path, 
        args.output_path,
        resolution_km=args.resolution
    )
    print("✅ Mineral potential maps generated")
    
    # Final step: Generate a report
    print("\n📊 Generating final report...")
    with open(os.path.join(args.output_path, "mineral_targeting_report.md"), "w") as f:
        f.write("# AI for Mineral Targeting - Results Report\n\n")
        f.write(f"**Generated on:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Dataset Information\n\n")
        f.write(f"- Number of geospatial layers: {len(geo_data)}\n")
        f.write(f"- Number of mines/points analyzed: {len(mines)}\n")
        
        f.write("\n## Features Used\n\n")
        for col in mines_with_features.columns:
            if col not in ['geometry', 'id', 'name', 'mine_id']:
                f.write(f"- {col}\n")
        
        f.write("\n## Results\n\n")
        f.write("The analysis identified areas with high mineral potential. ")
        f.write("The top 20 locations have been marked on the interactive map.\n\n")
        
        f.write("### Maps Generated\n\n")
        f.write("1. **Static Potential Map**: `mineral_potential_map.png`\n")
        f.write("2. **Interactive Web Map**: `interactive_potential_map.html`\n")
        f.write("3. **GeoJSON Prediction Grid**: `mineral_potential_grid.geojson`\n\n")
        
        f.write("### How to Interpret the Results\n\n")
        f.write("- **Red/Yellow Areas**: High potential for mineral deposits\n")
        f.write("- **Green/Blue Areas**: Lower potential for mineral deposits\n")
        f.write("- **Stars**: Top 20 predicted locations\n")
        f.write("- **Purple Markers**: Known mines/quarries\n\n")
        
        f.write("### Next Steps\n\n")
        f.write("1. Field verification of high-potential areas\n")
        f.write("2. Detailed geophysical surveys at top predicted locations\n")
        f.write("3. Integration with additional data sources (e.g., remote sensing)\n")
        f.write("4. Model refinement based on field verification results\n")
    
    print(f"✅ Report saved to: {os.path.join(args.output_path, 'mineral_targeting_report.md')}")
    
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"✨ All processing complete in {elapsed_time:.1f} seconds!")
    print(f"📂 Results saved to: {args.output_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()