# Machine Learning Approaches for Solar Irradiance Prediction in Bristol City

This repository contains the complete implementation of machine learning models for solar irradiance forecasting in Bristol, UK, developed as part of an MSc Data Science dissertation at the University of the West of England.

## Project Overview

This research compares three machine learning approaches—XGBoost, LSTM, and Support Vector Regression (SVR)—for predicting solar irradiance across two forecasting horizons:
- **Short-term**: 2-hour ahead predictions
- **Daily**: 24-hour ahead predictions

### Key Results
- **XGBoost** achieved superior performance: MAE = 14.17 W/m² (R² = 0.9816) for 2-hour forecasting and MAE = 18.37 W/m² (R² = 0.9699) for 24-hour forecasting
- **Feature importance analysis** revealed the critical role of lagged irradiance values and cloud cover variables
- **PCA comparison** showed that feature selection outperformed dimensionality reduction techniques

## Dataset

The research utilizes 10 years (2015-2024) of hourly meteorological data from:
- **Source**: [Renewables.ninja](https://www.renewables.ninja) platform
- **Underlying data**: NASA's MERRA-2 reanalysis dataset
- **Location**: Bristol, UK (51.4545°N, 2.5879°W)
- **Variables**: Solar irradiance (SWGDN), cloud cover, temperature, wind speed, precipitation

## Repository Structure
├── raw_data/                     # Original data files from Renewables.ninja
├── all_merged_data.csv          # Processed combined dataset
├── data_preparation.ipynb       # Data cleaning and preprocessing
├── solar_irradiance_prediction.ipynb           # 2-hour ahead forecasting models
├── solar_irradiance_prediction_24.ipynb       # 24-hour ahead forecasting models
├── solar_irradiance_prediction_feature_importance... # Feature analysis (hourly)
├── solar_irradiance_prediction_feature_importance... # Feature analysis (daily)
└── README.md                    # This file
## Interactive Notebooks (Google Colab)

The complete pipeline is available as interactive Google Colab notebooks:

1. **[Data Preparation](https://colab.research.google.com/drive/1Ca275L7wF5egbukspOJuVTUhVI7kTynw?usp=sharing)** - Data merging, cleaning, and exploratory analysis
2. **[Short-term Forecasting](https://colab.research.google.com/drive/1zyIhnsVkKvAKEOehW6vl1kbWxaHvEpuE?usp=sharing)** - 2-hour ahead prediction models
3. **[Daily Forecasting](https://colab.research.google.com/drive/1_5UTU-qGau4K1s8ZMleyYKET-0i2d4rA?usp=sharing)** - 24-hour ahead prediction models
4. **[Feature Analysis (Hourly)](https://colab.research.google.com/drive/1440i-TDRRU57Hq_Juvrfv7yVocGKAeJc?usp=sharing)** - Feature importance and PCA for short-term forecasting
5. **[Feature Analysis (Daily)](https://colab.research.google.com/drive/1MyxgfH5LMtFRGzb-Jm5ymydOT_nfSFIz?usp=sharing)** - Feature importance and PCA for daily forecasting

## Methodology

### Models Implemented
- **XGBoost**: Ensemble gradient boosting with hyperparameter optimization
- **LSTM**: Recurrent neural networks for temporal sequence modeling  
- **SVR**: Support Vector Regression with RBF kernel

### Feature Engineering
- Lagged irradiance values (1-48 hours)
- Cyclical time encodings (hour, day, month using sine/cosine transformations)
- Meteorological variables (cloud cover, temperature, wind, precipitation)
- Input window optimization (24h for hourly, 48h for daily forecasting)

### Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Coefficient of Determination (R²)

## Key Findings

1. **XGBoost consistently outperformed** LSTM and SVR across both forecasting horizons
2. **Feature importance revealed**:
   - `swgdn_lag_2` as the dominant predictor for hourly forecasting (importance ≈ 0.6)
   - `swgdn_lag_48` and `swgdn_lag_24` leading daily forecasting
   - Cloud cover variables consistently ranking in top predictors
3. **Feature selection proved superior to PCA** for dimensionality reduction
4. **Seasonal patterns clearly captured** with strong performance across Bristol's maritime climate conditions

## Requirements

### Python Libraries
- pandas, numpy
- scikit-learn
- xgboost
- tensorflow/keras (for LSTM)
- matplotlib, seaborn (visualization)
- joblib (model persistence)

### Installation
```bash
# Clone the repository
git clone https://github.com/Danial-Amiri/Machine-Learning-Approaches-for-Solar-Irradiance-Prediction-in-Bristol-City.git

# Install required packages
pip install pandas numpy scikit-learn xgboost tensorflow matplotlib seaborn joblib





###Usage

Data Preparation: Run data_preparation.ipynb to process raw data files
Model Training: Execute forecasting notebooks for desired time horizon
Feature Analysis: Run feature importance notebooks for model interpretability
Evaluation: Results include performance metrics and visualizations

###Applications
This forecasting system supports:

Grid operators: Improved stability through irradiance variability anticipation
Renewable energy providers: Optimized solar farm operations and cost reduction
Policy makers: Enhanced sustainable energy planning capabilities
Research community: Reproducible methodology for solar forecasting studies

###Data Attribution
This research uses data from Renewables.ninja under Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) licensing. Please cite:

Pfenninger, S. and Staffell, I. (2016) 'Long-term patterns of European PV output using 30 years of validated hourly reanalysis and satellite data', Energy, 114, pp. 1251-1265.

###License
This project is licensed under the MIT License - see the LICENSE file for details.
Citation
If you use this work in your research, please cite:
@mastersthesis{amiri2024solar,
  title={Machine Learning Approaches for Solar Irradiance Prediction in Bristol City},
  author={Amiri, Danial},
  year={2024},
  school={University of the West of England},
  type={MSc Data Science Dissertation}
}

###Contact

Author: Danial Amiri
Student ID: 24003829
Institution: University of the West of England
Supervisor: Dr. Neil Phillips

###Acknowledgments

University of the West of England for academic support
Renewables.ninja platform for providing high-quality meteorological data
NASA MERRA-2 project for underlying reanalysis datasets
