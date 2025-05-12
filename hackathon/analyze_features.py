import geopandas as gpd
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_dataset(input_file):
    """
    Analyze feature types and distributions in the dataset
    """
    print(f"📊 Analyzing dataset: {input_file}")
    
    # Load the dataset
    gdf = gpd.read_file(input_file)
    
    # Display basic information
    print("\n=== Dataset Overview ===")
    print(f"Number of records: {len(gdf)}")
    print(f"Number of columns: {len(gdf.columns)}")
    
    # Analyze column types
    print("\n=== Column Types ===")
    col_types = gdf.dtypes
    print(col_types)
    
    # Count of each type
    type_counts = col_types.value_counts()
    print("\nCount of each data type:")
    print(type_counts)
    
    # Identify non-geometry columns
    non_geom_cols = [col for col in gdf.columns if col != 'geometry']
    
    # Identify numeric and categorical columns
    numeric_cols = [col for col in non_geom_cols if np.issubdtype(gdf[col].dtype, np.number)]
    categorical_cols = [col for col in non_geom_cols if gdf[col].dtype == 'object']
    
    print(f"\nNumeric columns ({len(numeric_cols)}): {numeric_cols}")
    print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols}")
    
    # Analyze missing values
    print("\n=== Missing Values ===")
    missing = gdf[non_geom_cols].isna().sum()
    missing_pct = (missing / len(gdf)) * 100
    missing_df = pd.DataFrame({'Count': missing, 'Percentage': missing_pct})
    missing_df = missing_df[missing_df['Count'] > 0].sort_values('Count', ascending=False)
    print(missing_df)
    
    # Detailed analysis of categorical columns
    print("\n=== Categorical Columns Analysis ===")
    for col in categorical_cols:
        unique_values = gdf[col].nunique()
        print(f"\n{col}: {unique_values} unique values")
        value_counts = gdf[col].value_counts().head(10)  # Show top 10 values
        print(value_counts)
    
    # Basic summary statistics for numeric columns
    print("\n=== Numeric Columns Statistics ===")
    numeric_stats = gdf[numeric_cols].describe().T
    print(numeric_stats)
    
    # Create output directory for plots
    plots_dir = "feature_analysis"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate histograms for numeric features
    print("\n=== Generating Plots ===")
    for col in numeric_cols:
        if gdf[col].notna().sum() > 0:  # Only plot if there's data
            plt.figure(figsize=(10, 6))
            sns.histplot(gdf[col].dropna(), kde=True)
            plt.title(f'Distribution of {col}')
            plt.savefig(os.path.join(plots_dir, f"{col.replace(' ', '_')}_histogram.png"))
            plt.close()
    
    # Generate bar plots for categorical features (top 10 values)
    for col in categorical_cols:
        if gdf[col].notna().sum() > 0:  # Only plot if there's data
            plt.figure(figsize=(12, 8))
            top_values = gdf[col].value_counts().head(10)
            sns.barplot(x=top_values.index, y=top_values.values)
            plt.title(f'Top 10 values for {col}')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"{col.replace(' ', '_')}_barplot.png"))
            plt.close()
    
    print(f"\n✅ Analysis complete! Plots saved to '{plots_dir}' directory")
    
    return gdf

if __name__ == "__main__":
    input_file = "mines_with_all_features.geojson"
    
    if os.path.exists(input_file):
        gdf = analyze_dataset(input_file)
    else:
        print(f"❌ File not found: {input_file}")