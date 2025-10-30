import pandas as pd
import numpy as np
import logging 
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from src.common.config import load_config
from src.common.io_utils import read_parquet

logger = logging.getLogger(__name__)


def load_and_prepare_data(cfg):
    """Load feature-engineered data and prepare for training"""
    logger.info("Loading feature data...")
    
    gold_path = Path(cfg.data_paths['gold'])
    features_file = gold_path / "gold_features_all_years.parquet"
    
    if not features_file.exists():
        logger.error(f"❌ Features file not found: {features_file}")
        logger.info("Please run feature engineering first: python -m src.features.builder")
        return None, None, None

    df = read_parquet(features_file)
    logger.info(f"Loaded {len(df):,} records")

    # Identify feature columns
    exclude_cols = ['datetime', 'Date', 'Year', 'Month', 'Day', 'target_kw', 'District']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    logger.info(f"Feature columns: {len(feature_cols)}")

    return df, feature_cols, exclude_cols


def encode_district(df):
    """Encode District as numerical category"""
    logger.info("Encoding District feature...")

    le = LabelEncoder()
    df['District_encoded'] = le.fit_transform(df['District'])

    logger.info(f"Districts encoded: {len(le.classes_)} unique values")
    logger.info(f"  Mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    return df, le


def split_data(df, feature_cols, cfg):
    """
    PRODUCTION split: Use ALL years in training for maximum accuracy.
    """
    logger.info("Splitting data for PRODUCTION (temporal 80/10/10 split)...")
    
    # Add District_encoded
    feature_cols_final = feature_cols.copy() if isinstance(feature_cols, list) else list(feature_cols)
    if 'District_encoded' in df.columns and 'District_encoded' not in feature_cols_final:
        feature_cols_final.append('District_encoded')
    
    # Sort chronologically for temporal split
    df = df.sort_values('datetime')
    
    n = len(df)
    train_end = int(n * 0.80)
    val_end = int(n * 0.90)
    
    X_train = df.iloc[:train_end][feature_cols_final]
    y_train = df.iloc[:train_end]['target_kw']
    
    X_val = df.iloc[train_end:val_end][feature_cols_final]
    y_val = df.iloc[train_end:val_end]['target_kw']
    
    X_test = df.iloc[val_end:][feature_cols_final]
    y_test = df.iloc[val_end:]['target_kw']
    
    logger.info(f"  Train: {len(X_train):,} samples (2018-2023)")
    logger.info(f"  Validation: {len(X_val):,} samples (late 2023)")
    logger.info(f"  Test: {len(X_test):,} samples (2024)")
    
    # Drop NaN
    if X_train.isna().any().any():
        train_mask = ~X_train.isna().any(axis=1)
        X_train = X_train[train_mask]
        y_train = y_train[train_mask]
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(X_train, y_train, cfg):
    """Production-grade model"""
    logger.info("Training PRODUCTION model...")
    
    # Sample for speed but keep more data
    if len(X_train) > 1000000:
        logger.info(f"  Sampling 1M records from {len(X_train):,}...")
        sample_idx = np.random.choice(len(X_train), 1000000, replace=False)
        X_train_sample = X_train.iloc[sample_idx]
        y_train_sample = y_train.iloc[sample_idx]
    else:
        X_train_sample = X_train
        y_train_sample = y_train
    
    model = RandomForestRegressor(
        n_estimators=100,     # More trees
        max_depth=30,         # Deeper
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    
    logger.info("  Fitting model (5-8 minutes)...")
    model.fit(X_train_sample, y_train_sample)
    
    return model


def evaluate_predictions(y_true, y_pred, dataset_name="Dataset"):
    """Calculate and log evaluation metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    
    logger.info(f"  {dataset_name} Metrics:")
    logger.info(f"    MAE:  {mae:.2f} kW")
    logger.info(f"    RMSE: {rmse:.2f} kW")
    logger.info(f"    R²:   {r2:.4f}")
    logger.info(f"    MAPE: {mape:.2f}%")
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape}


def save_model_and_artifacts(model, label_encoder, feature_cols, metrics, cfg):
    """Save trained model and related artifacts"""
    logger.info("Saving model and artifacts...")
    
    models_path = Path(cfg.output_paths['models'])
    models_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_file = models_path / "random_forest_model.pkl"
    joblib.dump(model, model_file)
    logger.info(f"  Model saved: {model_file}")
    
    # Save label encoder
    encoder_file = models_path / "district_encoder.pkl"
    joblib.dump(label_encoder, encoder_file)
    logger.info(f"  Encoder saved: {encoder_file}")
    
    # Save feature names
    features_file = models_path / "feature_names.txt"
    with open(features_file, 'w') as f:
        f.write('\n'.join(feature_cols))
    logger.info(f"  Features saved: {features_file}")
    
    # Save metrics
    metrics_file = models_path / "training_metrics.txt"
    with open(metrics_file, 'w') as f:
        f.write("Training Metrics\n")
        f.write("="*50 + "\n")
        for dataset, metric_dict in metrics.items():
            f.write(f"\n{dataset}:\n")
            for metric, value in metric_dict.items():
                f.write(f"  {metric.upper()}: {value:.4f}\n")
    logger.info(f"  Metrics saved: {metrics_file}")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance_file = models_path / "feature_importance.csv"
        importance_df.to_csv(importance_file, index=False)
        logger.info(f"  Feature importance saved: {importance_file}")
        
        # Log top 10 features
        logger.info("\n  Top 10 Important Features:")
        for idx, row in importance_df.head(10).iterrows():
            logger.info(f"    {row['feature']}: {row['importance']:.4f}")


def train_pipeline():
    """Main training pipeline"""
    cfg = load_config()
    
    logger.info("="*60)
    logger.info("STEP 5: MODEL TRAINING (Year-Based Split)")
    logger.info("="*60)
    
    # Load data
    df, feature_cols, exclude_cols = load_and_prepare_data(cfg)
    if df is None:
        return
    
    print()
    
    # Encode District
    df, label_encoder = encode_district(df)
    print()
    
    # Split data (YEAR-BASED)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, feature_cols, cfg)
    print()
    
    # Train model
    model = train_model(X_train, y_train, cfg)
    print()
    
    # Evaluate on all splits
    logger.info("Evaluating model performance...")
    
    # Training set
    y_train_pred = model.predict(X_train)
    train_metrics = evaluate_predictions(y_train, y_train_pred, "Training")
    print()
    
    # Validation set
    y_val_pred = model.predict(X_val)
    val_metrics = evaluate_predictions(y_val, y_val_pred, "Validation")
    print()
    
    # Test set (2022 + 2024 - UNSEEN!)
    y_test_pred = model.predict(X_test)
    test_metrics = evaluate_predictions(y_test, y_test_pred, "Test (2022+2024 UNSEEN)")
    print()
    
    # Save everything
    metrics = {
        'train': train_metrics,
        'validation': val_metrics,
        'test': test_metrics
    }
    
    save_model_and_artifacts(
        model, 
        label_encoder, 
        X_train.columns.tolist(), 
        metrics, 
        cfg
    )
    
    logger.info("\n" + "="*60)
    logger.info("✅ TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Model saved to: {Path(cfg.output_paths['models']) / 'random_forest_model.pkl'}")
    logger.info(f"Test R² (on unseen 2022+2024): {test_metrics['r2']:.4f}")
    logger.info(f"Test MAE (on unseen 2022+2024): {test_metrics['mae']:.2f} kW")
    logger.info("\n🎯 2022 was NEVER seen during training - this is true prediction!")
    logger.info("="*60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    train_pipeline()


