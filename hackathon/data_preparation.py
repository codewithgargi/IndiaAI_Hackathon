import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.preprocessing import MinMaxScaler
import os

def preprocess_features(gdf):
    """
    Clean and prepare features for modeling
    
    Args:
        gdf: GeoDataFrame with features extracted by feature_engineering.py
        
    Returns:
        Processed GeoDataFrame ready for modeling
    """
    # Create a copy to avoid modifying the original
    processed_gdf = gdf.copy()
    
    # Handle missing values in numeric columns
    for col in processed_gdf.select_dtypes(include=[np.number]).columns:
        if col not in ['geometry']:  # Skip geometry column
            processed_gdf[col] = processed_gdf[col].fillna(processed_gdf[col].median())
    
    # Convert categorical features
    if 'lithology' in processed_gdf.columns:
        processed_gdf['lithology'] = processed_gdf['lithology'].fillna('unknown')
        # One-hot encode lithology
        lithology_dummies = pd.get_dummies(processed_gdf['lithology'], prefix='litho')
        processed_gdf = pd.concat([processed_gdf, lithology_dummies], axis=1)
    
    # Feature scaling for numeric columns (except certain columns)
    exclude_cols = ['geometry', 'id', 'name', 'mine_id', 'is_productive', 'productivity']
    numeric_cols = [col for col in processed_gdf.select_dtypes(include=[np.number]).columns 
                   if col not in exclude_cols]
    
    if numeric_cols:
        scaler = MinMaxScaler()
        processed_gdf[numeric_cols] = scaler.fit_transform(processed_gdf[numeric_cols])
    
    return processed_gdf

def create_training_dataset(mines_gdf, output_dir=None):
    """
    Create a labeled dataset for training mineral potential prediction models.
    If productivity/target variable doesn't exist, creates a proxy target.
    
    Args:
        mines_gdf: GeoDataFrame with mine points and extracted features
        output_dir: Directory to save the processed dataset (optional)
        
    Returns:
        X: Feature matrix
        y: Target variable
        feature_cols: List of feature column names
    """
    # Preprocess features
    processed_gdf = preprocess_features(mines_gdf)
    
    # Define columns to exclude from features
    exclude_cols = ['geometry', 'id', 'name', 'mine_id', 'is_productive', 'productivity']
    
    # Get feature columns
    feature_cols = [col for col in processed_gdf.columns if col not in exclude_cols]
    
    # Create feature matrix
    X = processed_gdf[feature_cols]
    
    # Check if we have a target variable
    if 'is_productive' in processed_gdf.columns:
        print("✅ Using existing 'is_productive' column as target")
        y = processed_gdf['is_productive']
    elif 'productivity' in processed_gdf.columns:
        print("✅ Using existing 'productivity' column as target")
        y = processed_gdf['productivity']
    else:
        # Create a proxy target based on available data
        print("⚠️ No target variable found. Creating proxy target based on available features...")
        
        # If we have geochemical values, use them as a proxy
        if 'avg_chem_value' in processed_gdf.columns and processed_gdf['avg_chem_value'].notna().any():
            print("  - Using 'avg_chem_value' as proxy for productivity")
            threshold = processed_gdf['avg_chem_value'].median()
            processed_gdf['is_productive'] = (processed_gdf['avg_chem_value'] > threshold).astype(int)
            
        # If we have distance to fault, closer mines might be more productive
        elif 'distance_to_fault' in processed_gdf.columns and processed_gdf['distance_to_fault'].notna().any():
            print("  - Using 'distance_to_fault' as proxy for productivity")
            threshold = processed_gdf['distance_to_fault'].median()
            processed_gdf['is_productive'] = (processed_gdf['distance_to_fault'] < threshold).astype(int)
            
        # If we have nearby lineaments, more lineaments might indicate better productivity
        elif 'num_nearby_lineaments' in processed_gdf.columns and processed_gdf['num_nearby_lineaments'].notna().any():
            print("  - Using 'num_nearby_lineaments' as proxy for productivity")
            threshold = processed_gdf['num_nearby_lineaments'].median()
            processed_gdf['is_productive'] = (processed_gdf['num_nearby_lineaments'] > threshold).astype(int)
            
        # If all else fails, assign random labels (for demonstration only!)
        else:
            print("⚠️ WARNING: No suitable proxy features. Creating random labels for demonstration only!")
            np.random.seed(42)
            processed_gdf['is_productive'] = np.random.binomial(1, 0.3, size=len(processed_gdf))
        
        y = processed_gdf['is_productive']
    
    # Save processed dataset if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "processed_training_data.geojson")
        processed_gdf.to_file(output_path, driver="GeoJSON")
        print(f"✅ Processed training dataset saved to: {output_path}")
    
    return X, y, feature_cols

def add_target_to_mines(mines_gdf, target_source=None):
    """
    Add a target variable to the mines GeoDataFrame.
    
    Args:
        mines_gdf: GeoDataFrame with mine points
        target_source: Path to a file containing target information (optional)
        
    Returns:
        GeoDataFrame with added target variable
    """
    if target_source and os.path.exists(target_source):
        # Load target data
        print(f"📊 Loading target data from: {target_source}")
        target_data = pd.read_csv(target_source)
        
        # Assuming there's a common ID column to join on
        if 'mine_id' in mines_gdf.columns and 'mine_id' in target_data.columns:
            print("✅ Joining target data based on 'mine_id'")
            # Merge target data with mines
            mines_with_target = mines_gdf.merge(target_data[['mine_id', 'is_productive']], 
                                               on='mine_id', how='left')
            return mines_with_target
        else:
            print("⚠️ No common ID column found for joining target data")
    
    # If no target source or joining failed, create proxy target
    print("⚠️ Creating proxy target variable...")
    mines_gdf = mines_gdf.copy()
    
    # Simple proxy: distance to fault (closer = more likely productive)
    if 'distance_to_fault' in mines_gdf.columns and mines_gdf['distance_to_fault'].notna().any():
        threshold = mines_gdf['distance_to_fault'].median()
        mines_gdf['is_productive'] = (mines_gdf['distance_to_fault'] < threshold).astype(int)
        print("✅ Created proxy target based on distance to fault")
    else:
        # Try geochemical values
        if 'avg_chem_value' in mines_gdf.columns and mines_gdf['avg_chem_value'].notna().any():
            threshold = mines_gdf['avg_chem_value'].median()
            mines_gdf['is_productive'] = (mines_gdf['avg_chem_value'] > threshold).astype(int)
            print("✅ Created proxy target based on geochemical values")
        else:
            # Random assignment for demonstration
            np.random.seed(42)
            mines_gdf['is_productive'] = np.random.binomial(1, 0.3, size=len(mines_gdf))
            print("⚠️ Created random proxy target (for demonstration only)")
    
    return mines_gdf

if __name__ == "__main__":
    # Example usage
    mines_path = "mines_with_all_features.geojson"
    output_dir = "./processed_data"
    
    if os.path.exists(mines_path):
        print(f"Loading mines data from: {mines_path}")
        mines_gdf = gpd.read_file(mines_path)
        
        # Add target variable (if not already present)
        if 'is_productive' not in mines_gdf.columns and 'productivity' not in mines_gdf.columns:
            mines_gdf = add_target_to_mines(mines_gdf)
        
        # Create training dataset
        X, y, feature_cols = create_training_dataset(mines_gdf, output_dir)
        
        print(f"✅ Created training dataset with {X.shape[1]} features and {len(y)} samples")
        print(f"Features: {feature_cols}")
    else:
        print(f"❌ Mine data file not found: {mines_path}")