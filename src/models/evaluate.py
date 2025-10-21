"""
Model Evaluation Module
Evaluates trained RandomForest model performance with plots and metrics
"""

import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

from src.common.config import load_config
from src.common.io_utils import read_parquet

logger = logging.getLogger(__name__)


def evaluate_district_performance(df, model, feature_cols, le):
    """Compute metrics per district."""
    logger.info("Evaluating performance per district...")
    results = []
    for district in sorted(df['District'].unique()):
        sub = df[df['District'] == district]
        X = sub[feature_cols].copy()
        if 'District_encoded' not in X:
            X['District_encoded'] = le.transform(sub['District'])
        y_true = sub['target_kw']
        y_pred = model.predict(X)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        results.append({'District': district, 'MAE': mae, 'RMSE': rmse, 'R2': r2})
    results_df = pd.DataFrame(results)
    logger.info("\n" + str(results_df.round(4)))
    return results_df

