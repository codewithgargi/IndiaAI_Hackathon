**AI for Mineral Targeting 
IndiaAI Hackathon Project: Intelligent Mineral Exploration Platform**

**Overview**

An advanced AI-powered platform for predicting mineral potential using machine learning and geospatial analysis. This project aims to revolutionize mineral exploration by providing data-driven insights to help geologists and mining companies identify promising exploration targets more efficiently.

**Features**

Data Preparation: Scripts to clean and prepare geospatial and tabular datasets.
Feature Engineering: Create meaningful features from raw data for modeling.
Modeling: Trained using Random Forest; the best model is saved as a .pkl file.
Visualization: Feature importance visualizations and predictions.
App: A simple app interface to run predictions on new data.

**Project Structure**

hackathon/\n
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

**Requirements**

Python 3.8+
pandas
geopandas
scikit-learn
matplotlib
joblib


**Running the Project**

Prepare data using data_preparation.py.
Generate features with feature_engineering.py.
Train or load model and predict with main.py or predict_potential.py.
Use mineral_targeting_app.py for app interface (if applicable).

**Team Leader and Solo Contributor:** Gargi Gupta

**License**

This project is open source and available under the MIT License.
