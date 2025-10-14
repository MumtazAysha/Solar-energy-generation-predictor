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
                      
                    





