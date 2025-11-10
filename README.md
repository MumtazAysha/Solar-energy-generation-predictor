# Solar-energy-generation-predictor-and-the-relation-between-weather-and-the-solar-energy-generation
This ML model is designed to predict the future solar energy generations in different districts in Sri Lanka, at different times. In addition to that this, model helps the user to identify if there is a correlation between the solar energy generation and the weather factors such as rainfall, humidity, and cloud coverage. 

The data sets used for this project are strictly confidential and can only be used with the permission from Public Utilities Commission of Sri Lanka.  

8 essential libraries are used.
Data processing, Pandas(to load CSV,Excel files, and manipulate time series data), numpy(for numerical operations and array handling), and openpyxl(to read excel files in .xlsx format) are used. 
Statistical analysis: scipy(Pearson/Spearman correlation, and statistical tests), statsmodels(ARIMA,ARIMAX,SARIMAX time series models).
Machine Learning: scikit-learn(Random Forest,gradient boosting,metrics,preprocessing), xgboost(Extreme gradienr boosting(best performance)).
Visualization: matplotlib(Time series plots, and basic charts), seaboorn(correlation heatmaps,statistical plots)
Utilities: To save and load trained models.

The models used in this project.
Persistence Model : Mandatory reference for skill score calculation. 
                    Purpose: Simplest baseline - next value equals current value
                    Code: forecast[t] = actual[t-1]
                    No libraries are required.

ARIMA/ARIMAX : Library: statsmodels
               Code: from statsmodels.tsa.arima.model import ARIMA 
               Purpose: classical time series for forecasting.
               Use: short-term forecasting(5 min to 1 hour ahead)

Main moodels :
 XXGBoost: Library: xgboost
          Code: import xgboost as xgb; model = xgb.XGBRegressor(n_estimators=500,max_depth=8)
          Advantages: Best accuracy for solar forecasting.
                      Handles weather features excellently.
                      Built-in feature impoortance.
                      Fast training and prediction.
          Use: All forecast horizoons(5 min to day-ahead)

 Random Forest: Library: scikit-learn
                Code: from sklearn.ensemble import RandomForestRegressor
                Advantages: Robust to outliers
                            Good for feature importance analysis
                            Easy to interpret
                Use: Backup model, comparison with XGBoost

 Gradient Boosting: Library: scikit-learn
                    Code: from sklearn.ensemble import GradientBoostingRegressor
                    Use: Alternative t XGBoost, often similar performance


Correlation Analysis
 Pearson correlation: Library: scipy
                      Code: from scipy.stats import spearmanr; corr, p = spearman(x,y)
                      Purpose: Measure moonotonic(non-linear relationships)
                      
                    


# Solar Data Converter

## Purpose
Converts solar energy data from multiple monthly sheets to 2022 format with:
- Continuous day numbering (1-365/366)
- Sri Lanka irradiance constant (5.25)
- Normalized calculations

## Usage


## When to Use
- New data from 2025 or beyond
- Data from additional districts
- Re-processing with updated formulas

## Input Format
- Multiple sheets (one per month)
- Time columns as strings ('08:00', '08:05', etc.)
- Date/Month/Day columns

## Output Format
- Single sheet with all year data
- Days numbered 1-365/366
- Time columns as decimals (8, 8.05, etc.)
- Summary rows: Total kW, Daily Average, Energy kWh


This script:
- Converts multiple monthly sheets to single annual sheet
- Applies continuous day numbering
- Calculates normalized values using Sri Lanka's irradiance constant (5.25)

**Location:** `scripts/convert.py`
**Input:** `data/raw/*.xlsx`
**Output:** `data/processed/*-CONVERTED.xlsx`



