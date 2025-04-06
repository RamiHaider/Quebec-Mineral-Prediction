# Quebec Mineral Potential Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Multi-Modal Deep Learning for Mineral Exploration

This repository contains the modeling code and notebooks for a mineral prospectivity prediction project using multi-modal deep learning. The model combines geochemical data, geological features, and magnetic imagery to predict the probability of anomalous values for Au, Ag, Cu, Co, and Ni across Quebec.

![Quebec Mineral Prediction Map](https://ramihaider.me/assets/images/quebec-prediction-map.jpg)

> **Note**: This repository contains only the modeling code and notebooks. For the complete project including data preprocessing steps in QGIS, image generation scripts, and web application implementation, visit the [full project page](https://ramihaider.me/portfolio/quebec-minerals.html).


## Data Availability

Taken offline until further notice

These files are available upon 

## Project Structure

```
Modeling/
├── data/
│   ├── output/
│   │   └── output_fused.csv           # Combined predictions from both models
│   └── preprocessed/
│       ├── grid_features.csv          # Features for prediction grid points
│       └── rock_features.csv          # Features for rock samples (training)
├── models/
│   ├── cnn/                           # Saved CNN model weights
│   └── gradient_boost/                # Saved GBT model weights
└── notebooks/
    ├── EDA&Preprocessing.ipynb        # Exploratory data analysis
    ├── Training_GBT.ipynb             # GBT model training
    ├── Training_CNN.ipynb             # CNN model training
    ├── Hyperparameter_CNN.ipynb       # CNN hyperparameter tuning
    ├── Predicting_GBT.ipynb           # Predictions using GBT models
    ├── Predicting_CNN.ipynb           # Predictions using CNN models
    └── Fusing_CSVs.ipynb              # Fusion of model predictions
```

## Key Features

- **Multi-modal approach**: Combines convolutional neural networks for magnetic imagery with gradient boosted trees for tabular geological data
- **Class imbalance handling**: Innovative approach using moderate fixed weights (5-12 range) instead of extreme dynamic weights
- **Memory-efficient implementation**: Optimized for MacBook M2 with 8GB RAM using custom data generators and batch processing
- **Model fusion strategy**: Weighted ensemble (CNN: 0.7, GBT: 0.3) with confidence scoring system
- **High performance**: AUC scores between 0.68-0.81 for CNN and 0.84-0.92 for GBT models

## Requirements

- Python 3.9+
- PyTorch 1.10+
- scikit-learn 1.0+
- pandas, numpy, matplotlib
- Jupyter notebooks

Installation:
```bash
pip install -r requirements.txt
```

## Model Descriptions

### Convolutional Neural Network (CNN)

- 3-layer CNN with batch normalization
- Optimized for magnetic imagery (170×170 pixels)
- Key hyperparameters:
  - Learning rate: 5e-4 (Adam optimizer)
  - Class weights: AU/AG: 12.0, CU/CO: 7.0, NI: 5.0
  - Dropout rates: 0.3, 0.2

### Gradient Boosted Trees (GBT)

- Scikit-learn's GradientBoostingClassifier
- Separate models for each mineral
- Key hyperparameters:
  - n_estimators: 100
  - learning_rate: 0.1
  - max_depth: 3
  - subsample: 0.8

### Ensemble Fusion Strategy

- CNN weight: 0.7
- GBT weight: 0.3
- Confidence scoring:
  - High confidence (2): Both models predict positive
  - Medium confidence (1): At least one model predicts positive
  - No prediction (0): Neither model predicts positive

## Results

### CNN Performance (AUC Scores)
- Gold (Au): 0.8118
- Silver (Ag): 0.7052
- Copper (Cu): 0.6871
- Cobalt (Co): 0.7736
- Nickel (Ni): 0.8039

### Feature Importance (GBT)
- Gold (Au): Proximity to geological contacts (37%), Lithology type (29%), Distance to faults (24%)
- Silver (Ag): Lithology type (41%), Proximity to geological contacts (32%), Stratigraphic position (18%)
- Copper (Cu): Lithology type (39%), Distance to faults (31%), Proximity to geological contacts (22%)

## Usage

1. **Exploratory Data Analysis**:
   ```bash
   jupyter notebook notebooks/EDA\&Preprocessing.ipynb
   ```

2. **Training the Models**:
   ```bash
   jupyter notebook notebooks/Training_GBT.ipynb
   jupyter notebook notebooks/Training_CNN.ipynb
   ```

3. **Making Predictions**:
   ```bash
   jupyter notebook notebooks/Predicting_GBT.ipynb
   jupyter notebook notebooks/Predicting_CNN.ipynb
   ```

4. **Fusing Predictions**:
   ```bash
   jupyter notebook notebooks/Fusing_CSVs.ipynb
   ```

## Complete Project

For the complete project including:
- QGIS preprocessing steps
- Spatial chunking for efficient image generation
- Web application implementation with Supabase and PostGIS
- Interactive visualization with Leaflet.js

Visit: [https://ramihaider.me/portfolio/quebec-minerals.html](https://ramihaider.me/portfolio/quebec-minerals.html)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**Rami Haider**  
[Website](https://ramihaider.me) | [LinkedIn](https://linkedin.com/in/ramihaider) | [GitHub](https://github.com/RamiHaider)

---

If you use this code or methodology in your research, please cite:
```
Haider, R. (2023). Multi-Modal Deep Learning for Mineral Potential Prediction: A Case Study in Quebec. 
GitHub repository: https://github.com/RamiHaider/Quebec-Mineral-Prediction
```
