**Mineral Potential Targeting** - Hackathon Project

**Overview**

This project focuses on predicting areas with high mineral potential using machine learning techniques. It was developed during a hackathon and includes data processing, feature engineering, model training, and visualization components.

Features

Data Preparation: Scripts to clean and prepare geospatial and tabular datasets.

Feature Engineering: Create meaningful features from raw data for modeling.

Modeling: Trained using Random Forest; the best model is saved as a .pkl file.

Visualization: Feature importance visualizations and predictions.

App: A simple app interface to run predictions on new data.

Project Structure

hackathon/
├── analyze_features.py
├── best_mineral_potential_model.pkl
├── data_preparation.py
├── feature_engineering.py
├── main.py
├── mineral_targeting_app.py
├── mines_with_all_features.geojson
├── predict_potential.py
├── RandomForest_feature_importance.png
├── RandomForest_permutation_importance.png

Requirements

Python 3.8+

pandas

geopandas

scikit-learn

matplotlib

joblib


Running the Project

Prepare data using data_preparation.py.

Generate features with feature_engineering.py.

Train or load model and predict with main.py or predict_potential.py.

Use mineral_targeting_app.py for app interface (if applicable).

Authors

Team Leader and Solo Contributor: Gargi Gupta

License

This project is open source and available under the MIT License.
