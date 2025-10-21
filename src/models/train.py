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

   #Identify feature columns
   exclude_cols = ['datetime', 'Date', 'Year', 'Month', 'Day', 'target_kw', 'District']
   feature_cols = [c for c in df.columns if c not in exclude_cols]

   logger.info(f"Feature columns: {len(feature_cols)}")

   return df, feature_cols, exclude_cols

def encode_district(df):
    """Encode District as numerical category"""
    logger.info("Encoding District feature...")

    le = LabelEncoder()
    df['Dstrict_encoded'] = le.fit_transform(df['District'])

    logger.info(f"Districts encoded: {len(le.classes_)} unique values")
    logger.info(f"  Mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    return df, le

def split_data(df, feature_cols, cfg):
    """Split data into train/validation/test sets"""
    logger.info("Splitting data into train/val/test sets...")
    
    # Work on a copy to avoid UnboundLocalError
    feature_cols_final = feature_cols.copy() if isinstance(feature_cols, list) else list(feature_cols)
    
    # Add District_encoded to features if it exists and isn't already included
    if 'District_encoded' in df.columns and 'District_encoded' not in feature_cols_final:
        feature_cols_final.append('District_encoded')
    
    X = df[feature_cols_final]
    y = df['target_kw']
    
    # Split: 70% train, 15% validation, 15% test
    # Use temporal split to avoid data leakage
    train_frac = cfg.train_fraction  # 0.7 from config
    val_frac = 0.15
    
    # Sort by datetime for temporal split
    df_sorted = df.sort_values('datetime')
    X = df_sorted[feature_cols_final]
    y = df_sorted['target_kw']
    
    n = len(df_sorted)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    
    X_val = X.iloc[train_end:val_end]
    y_val = y.iloc[train_end:val_end]
    
    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]
    
    logger.info(f"  Train: {len(X_train):,} samples ({train_frac*100:.0f}%)")
    logger.info(f"  Validation: {len(X_val):,} samples ({val_frac*100:.0f}%)")
    logger.info(f"  Test: {len(X_test):,} samples ({(1-train_frac-val_frac)*100:.0f}%)")
    
    # Check for NaN
    if X_train.isna().any().any():
        logger.warning("⚠️ Training data contains NaN values, dropping...")
        mask = ~X_train.isna().any(axis=1)
        X_train = X_train[mask]
        y_train = y_train[mask]
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_model(X_train, y_train, cfg):
    """Train RandomForest model"""
    logger.info("Training RandomForest model...")
    
    model_params = cfg.model['params']
    logger.info(f"  Parameters: {model_params}")
    
    model = RandomForestRegressor(**model_params)
    
    logger.info("  Fitting model (this may take a few minutes)...")
    model.fit(X_train, y_train)
    
    logger.info("  ✅ Model trained successfully")
    
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
    logger.info("STEP 5: MODEL TRAINING")
    logger.info("="*60)
    
    # Load data
    df, feature_cols, exclude_cols = load_and_prepare_data(cfg)
    if df is None:
        return
    
    print()
    
    # Encode District
    df, label_encoder = encode_district(df)
    print()
    
    # Split data
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
    
    # Test set
    y_test_pred = model.predict(X_test)
    test_metrics = evaluate_predictions(y_test, y_test_pred, "Test")
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
    logger.info(f"Test R²: {test_metrics['r2']:.4f}")
    logger.info(f"Test MAE: {test_metrics['mae']:.2f} kW")
    logger.info("="*60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    train_pipeline()




