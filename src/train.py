"""
train.py — Train, evaluate, and compare regression models for sales forecasting.

Models:
  1. Baseline: Linear Regression (minimal features)
  2. Advanced: XGBoost with rich feature engineering

The script prints metrics, calculates the improvement percentage, and saves the
best model to models/.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import json


def create_features(df):
    """Create comprehensive time-series features from the date column."""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Calendar features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['quarter'] = df['date'].dt.quarter

    # Cyclical encoding of month (captures Dec→Jan continuity)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Cyclical encoding of day-of-week
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Lag features
    for lag in [1, 2, 3, 7, 14, 28]:
        df[f'sales_lag_{lag}'] = df['sales'].shift(lag)

    # Rolling window features
    for window in [7, 14, 30]:
        df[f'sales_roll_mean_{window}'] = df['sales'].rolling(window).mean()
        df[f'sales_roll_std_{window}'] = df['sales'].rolling(window).std()

    # Expanding mean (cumulative average up to that point)
    df['sales_expanding_mean'] = df['sales'].expanding().mean()

    # Drop rows with NaN from lag/rolling
    df = df.dropna().reset_index(drop=True)
    return df


def evaluate_model(y_true, y_pred, model_name):
    """Compute and print regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"--- {model_name} ---")
    print(f"  MAE  : {mae:.2f}")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  R²   : {r2:.4f}")
    print(f"  MAPE : {mape:.2f}%\n")
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape}


def main():
    print("=" * 60)
    print(" Sales Forecasting — Model Training & Evaluation")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────
    data_path = 'data/historical_sales.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found at {data_path}. Run generate_data.py first.")

    df = pd.read_csv(data_path)
    print(f"\nLoaded {len(df)} rows from {data_path}")

    # ── Feature engineering ───────────────────────────────────────
    print("Engineering features...")
    df = create_features(df)
    print(f"After feature engineering: {len(df)} rows, {df.shape[1]} columns\n")

    # ── Baseline features (simple — what you'd get without ML expertise)
    baseline_features = ['month', 'day_of_week', 'is_weekend', 'promotion']

    # ── Advanced features (comprehensive)
    advanced_features = [
        'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
        'day_of_month', 'week_of_year', 'quarter', 'is_weekend', 'promotion',
        'sales_lag_1', 'sales_lag_2', 'sales_lag_3',
        'sales_lag_7', 'sales_lag_14', 'sales_lag_28',
        'sales_roll_mean_7', 'sales_roll_mean_14', 'sales_roll_mean_30',
        'sales_roll_std_7', 'sales_roll_std_14', 'sales_roll_std_30',
        'sales_expanding_mean',
    ]

    X_baseline = df[baseline_features]
    X_advanced = df[advanced_features]
    y = df['sales']

    # Time-based split (80/20)
    split_idx = int(len(df) * 0.8)
    X_bl_train, X_bl_test = X_baseline.iloc[:split_idx], X_baseline.iloc[split_idx:]
    X_adv_train, X_adv_test = X_advanced.iloc[:split_idx], X_advanced.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Train size: {len(y_train)}")
    print(f"Test size : {len(y_test)}\n")

    # ── 1. Baseline: Linear Regression with simple features ──────
    print("Training Baseline Model (Linear Regression, simple features)...")
    baseline = LinearRegression()
    baseline.fit(X_bl_train, y_train)
    baseline_preds = baseline.predict(X_bl_test)
    baseline_metrics = evaluate_model(y_test, baseline_preds, "Baseline: Linear Regression")

    # ── 2. Advanced: Gradient Boosting with rich features ─────────
    print("Training Advanced Model (Gradient Boosting, engineered features)...")
    advanced = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )
    advanced.fit(X_adv_train, y_train)
    advanced_preds = advanced.predict(X_adv_test)
    advanced_metrics = evaluate_model(y_test, advanced_preds, "Advanced: Gradient Boosting")

    # ── Improvement ───────────────────────────────────────────────
    improvement = ((baseline_metrics['mae'] - advanced_metrics['mae']) / baseline_metrics['mae']) * 100
    print("=" * 60)
    print(f"  FORECASTING ACCURACY IMPROVEMENT: {improvement:.1f}% MAE reduction")
    print(f"    Baseline MAE : {baseline_metrics['mae']:.2f}")
    print(f"    Advanced MAE : {advanced_metrics['mae']:.2f}")
    print("=" * 60)

    # ── Save best model & metadata ────────────────────────────────
    os.makedirs('models', exist_ok=True)
    model_path = 'models/gb_sales_model.pkl'
    joblib.dump(advanced, model_path)
    print(f"\nBest model saved -> {model_path}")

    # Save metrics for reproducibility
    results = {
        'baseline': baseline_metrics,
        'advanced': advanced_metrics,
        'improvement_pct': round(improvement, 2),
        'advanced_features': advanced_features,
    }
    with open('models/metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Metrics saved  -> models/metrics.json")

    # ── Feature importances ───────────────────────────────────────
    importances = advanced.feature_importances_
    print("\nTop Feature Importances:")
    for feat, imp in sorted(zip(advanced_features, importances),
                            key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {feat:25s} {imp:.4f}")


if __name__ == "__main__":
    main()
