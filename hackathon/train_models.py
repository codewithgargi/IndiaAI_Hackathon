import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.inspection import permutation_importance
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def train_mineral_potential_models(input_file):
    """
    Train and evaluate multiple models for mineral potential prediction
    """
    print("🔍 Loading enriched mine data...")
    mines = gpd.read_file(input_file)
    
    # Data preparation
    print("🧹 Preprocessing features...")
    # Drop non-feature columns
    feature_cols = [col for col in mines.columns if col not in 
                   ['geometry', 'id', 'name', 'mine_id']]
    
    # Handle missing values in numeric columns
    for col in mines.select_dtypes(include=[np.number]).columns:
        if col in feature_cols:
            mines[col] = mines[col].fillna(mines[col].median())
    
    # Convert categorical columns to dummy variables
    categorical_cols = []
    for col in feature_cols:
        if col in mines.columns and mines[col].dtype == 'object':
            print(f"⚠️ Found categorical column: {col}")
            categorical_cols.append(col)
    
    # Handle all categorical columns (not just lithology)
    for col in categorical_cols:
        if col in mines.columns:
            print(f"🔄 Converting {col} to dummy variables")
            mines[col] = mines[col].fillna('unknown')
            dummies = pd.get_dummies(mines[col], prefix=col[:5])
            mines = pd.concat([mines, dummies], axis=1)
            # Remove the original categorical column from feature list
            feature_cols.remove(col)
            # Add the new dummy columns to feature list
            feature_cols.extend(dummies.columns.tolist())
    
    # Create target variable if not present
    # For this example, we'll use a proxy target based on geochemical values
    # In a real scenario, you'd want to use known productive vs non-productive mines
    if 'is_productive' not in mines.columns:
        print("⚠️ Creating proxy target variable based on geochemical values...")
        if 'avg_chem_value' in mines.columns and mines['avg_chem_value'].notna().any():
            # Use geochemical values as a proxy for productivity
            threshold = mines['avg_chem_value'].median()
            mines['is_productive'] = (mines['avg_chem_value'] > threshold).astype(int)
        elif 'distance_to_fault' in mines.columns and mines['distance_to_fault'].notna().any():
            # Use distance to fault as a proxy (closer = more productive)
            threshold = mines['distance_to_fault'].median()
            mines['is_productive'] = (mines['distance_to_fault'] < threshold).astype(int)
        else:
            # Create random labels for demonstration (you should NOT do this in practice)
            print("⚠️ WARNING: Creating random labels for demonstration only!")
            np.random.seed(42)
            mines['is_productive'] = np.random.binomial(1, 0.3, size=len(mines))
    
    # Prepare feature matrix - select only numeric columns for X
    print("📊 Selecting numeric features only...")
    numeric_feature_cols = [
        col for col in feature_cols
        if col in mines.columns and pd.api.types.is_numeric_dtype(mines[col])
    ]

    
    print(f"🔢 Using {len(numeric_feature_cols)} numeric features")
    if len(numeric_feature_cols) < 1:
        print("❌ Error: No numeric features available for training!")
        raise ValueError("No numeric features available for training")
        
    X = mines[numeric_feature_cols]
    
    # Display the first few rows of X to verify
    print("\n📝 Sample of feature matrix:")
    print(X.head())
    print(f"\n📊 Feature matrix shape: {X.shape}")
    
    # Make sure the target is numeric
    y = mines['is_productive'].astype(int)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"🧠 Training models using {len(feature_cols)} features and {len(X_train)} samples...")
    
    # Train multiple models
    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42)
    }
    
    best_models = {}
    results = {}
    
    for name, model in models.items():
        print(f"🔄 Training {name}...")
        
        # Define hyperparameter grid
        if name == "RandomForest":
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        else:  # GradientBoosting
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        
        # Perform grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_models[name] = grid_search.best_estimator_
        
        # Evaluate on test set
        y_pred = best_models[name].predict(X_test)
        y_proba = best_models[name].predict_proba(X_test)[:, 1]
        
        # Store results
        results[name] = {
            'best_params': grid_search.best_params_,
            'train_score': grid_search.best_score_,
            'test_score': roc_auc_score(y_test, y_proba),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        print(f"✅ {name} - Best CV Score: {grid_search.best_score_:.4f}, Test AUC: {results[name]['test_score']:.4f}")
        
        # Feature importance analysis
        if name == "RandomForest":
            importances = best_models[name].feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Plot top 10 features
            plt.figure(figsize=(10, 6))
            plt.title(f"Top 10 Feature Importances - {name}")
            plt.bar(range(min(10, len(indices))), 
                   importances[indices][:10], 
                   align='center')
            plt.xticks(range(min(10, len(indices))), 
                      [X.columns[i] for i in indices][:10], 
                      rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f"{name}_feature_importance.png")
            
            # Permutation importance (more reliable)
            perm_importance = permutation_importance(
                best_models[name], X_test, y_test, n_repeats=10, random_state=42
            )
            
            perm_indices = perm_importance.importances_mean.argsort()[::-1]
            plt.figure(figsize=(10, 6))
            plt.title(f"Top 10 Permutation Importances - {name}")
            plt.bar(range(min(10, len(perm_indices))), 
                   perm_importance.importances_mean[perm_indices][:10], 
                   yerr=perm_importance.importances_std[perm_indices][:10],
                   align='center')
            plt.xticks(range(min(10, len(perm_indices))), 
                      [X.columns[i] for i in perm_indices][:10], 
                      rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f"{name}_permutation_importance.png")
    
    # Save best model
    best_model_name = max(results, key=lambda k: results[k]['test_score'])
    best_model = best_models[best_model_name]
    joblib.dump(best_model, "best_mineral_potential_model.pkl")
    
    print(f"🏆 Best model: {best_model_name} with AUC: {results[best_model_name]['test_score']:.4f}")
    print("💾 Model saved as 'best_mineral_potential_model.pkl'")
    
    # Return the results and best model
    return best_model, results, X.columns

if __name__ == "__main__":
    best_model, results, feature_names = train_mineral_potential_models("mines_with_all_features.geojson")