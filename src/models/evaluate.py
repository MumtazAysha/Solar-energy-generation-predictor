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
import matplotlib
matplotlib.use('Agg')
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
        # Use exactly the same feature names as during training
        X = sub[[c for c in feature_cols if c in sub.columns]].copy()

        # Optional: if model was trained without District_encoded, don't add it
        if 'District_encoded' in X.columns and 'District_encoded' not in feature_cols:
            X = X.drop(columns=['District_encoded'], errors='ignore')

        y_true = sub['target_kw']
        y_pred = model.predict(X)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        results.append({'District': district, 'MAE': mae, 'RMSE': rmse, 'R2': r2})
    results_df = pd.DataFrame(results)
    logger.info("\n" + str(results_df.round(4)))
    return results_df

def plot_feature_importance(model, feature_names, output_dir):
    if not hasattr(model, "feature_importances_"):
        logger.warning("Model has no feature importances attribute.")
        return
    imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=imp.head(15), x='Importance', y='Feature', palette='viridis')
    plt.title("Top 15 Feature Importances")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "feature_importance_top15.png", dpi=200)
    plt.close()
    logger.info("Feature importance plot saved.")


def plot_actual_vs_pred(df, y_true, y_pred, output_dir, n=1000):
    """Plot Actual vs Predicted for sample points."""
    idx = np.random.choice(len(y_true), size=min(n, len(y_true)), replace=False)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true.iloc[idx], y_pred[idx], alpha=0.3)
    plt.xlabel("Actual Generation (kW)")
    plt.ylabel("Predicted Generation (kW)")
    line = [y_true.min(), y_true.max()]
    plt.plot(line, line, color="red", linestyle="--")
    plt.title("Actual vs Predicted (Sample)")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "actual_vs_predicted.png", dpi=200)
    plt.close()
    logger.info("Actual vs Predicted scatter plot saved.")


def evaluate_model():
    cfg = load_config()
    output_path = Path(cfg.output_paths['models'])
    gold_path = Path(cfg.data_paths['gold'])

    logger.info("=" * 60)
    logger.info("STEP 6: MODEL EVALUATION")
    logger.info("=" * 60)

    # Load model and artifacts
    model_file = output_path / "random_forest_model.pkl"
    encoder_file = output_path / "district_encoder.pkl"
    features_file = output_path / "feature_names.txt"

    model = joblib.load(model_file)
    le = joblib.load(encoder_file)
    feature_cols = Path(features_file).read_text().splitlines()
    logger.info("✅ Model and artifacts loaded")

    # Load Gold features data
    features_df = read_parquet(gold_path / "gold_features_all_years.parquet")
    logger.info(f"Loaded {len(features_df):,} feature records for evaluation")

    # Encode district
    # Encode if missing, but don't add to features if model wasn't trained with it
    if 'District_encoded' not in features_df:
        features_df['District_encoded'] = le.transform(features_df['District'])

# Only keep exactly the feature names the model was trained with
# (Training already saved feature_names.txt)

    feature_cols = Path(features_file).read_text().splitlines()
    X = features_df[feature_cols]   # Ensure identical columns


    # Predict
    logger.info("Generating predictions...")
    X = features_df[feature_cols]
    y_true = features_df['target_kw']
    y_pred = model.predict(X)

    # Compute global metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100

    logger.info(f"\nOverall Metrics:")
    logger.info(f"  MAE:  {mae:.2f} kW")
    logger.info(f"  RMSE: {rmse:.2f} kW")
    logger.info(f"  R²:   {r2:.4f}")
    logger.info(f"  MAPE: {mape:.2f}%")

    # Plots
    plot_dir = output_path / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_actual_vs_pred(features_df, y_true, y_pred, plot_dir)
    plot_feature_importance(model, feature_cols, plot_dir)

    # District‑wise metric summary
    district_results = evaluate_district_performance(features_df, model, feature_cols, le)
    district_results.to_csv(plot_dir / "district_metrics.csv", index=False)
    logger.info(f"District metrics saved: {plot_dir / 'district_metrics.csv'}")

    logger.info("\n" + "=" * 60)
    logger.info("✅ EVALUATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Plots and reports in {plot_dir}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    evaluate_model()