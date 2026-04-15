"""
visualize.py - Generate all charts for the Sales Forecasting project.
Saves publication-quality plots to the outputs/ directory.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import os
import sys

# Import feature engineering from train module
# (use importlib to avoid conflict with pip 'train' package)
import importlib.util
_train_path = os.path.join(os.path.dirname(__file__), 'train.py')
_spec = importlib.util.spec_from_file_location('train_module', _train_path)
_train_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train_module)
create_features = _train_module.create_features

# -- Style --
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")
FIGSIZE = (14, 6)
OUTPUT_DIR = 'outputs'


def save_fig(name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved -> {path}")
    plt.close()


# ---- 1. Sales Time Series ------------------------------------------------
def plot_sales_timeseries(df_raw):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(df_raw['date'], df_raw['sales'], alpha=0.35, linewidth=0.8, label='Daily Sales')
    ax.plot(df_raw['date'], df_raw['sales'].rolling(30).mean(),
            color='crimson', linewidth=2, label='30-Day Moving Average')
    ax.set_title('Daily Sales with 30-Day Moving Average', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales ($)')
    ax.legend()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.autofmt_xdate()
    save_fig('01_sales_timeseries.png')


# ---- 2. Monthly Seasonality ----------------------------------------------
def plot_monthly_seasonality(df_raw):
    fig, ax = plt.subplots(figsize=(12, 6))
    df_plot = df_raw.copy()
    df_plot['month'] = df_plot['date'].dt.month
    month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']
    sns.boxplot(x='month', y='sales', data=df_plot, ax=ax, palette='coolwarm')
    ax.set_xticklabels(month_names)
    ax.set_title('Sales Distribution by Month (Seasonality)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Month')
    ax.set_ylabel('Sales ($)')
    save_fig('02_monthly_seasonality.png')


# ---- 3. Weekly Pattern ---------------------------------------------------
def plot_weekly_pattern(df_raw):
    fig, ax = plt.subplots(figsize=(10, 6))
    df_plot = df_raw.copy()
    df_plot['day_of_week'] = df_plot['date'].dt.dayofweek
    day_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    weekly_avg = df_plot.groupby('day_of_week')['sales'].mean()
    colors = ['#3498db'] * 5 + ['#e74c3c'] * 2
    ax.bar(day_names, weekly_avg.values, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_title('Average Sales by Day of Week', fontsize=16, fontweight='bold')
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Average Sales ($)')
    for i, v in enumerate(weekly_avg.values):
        ax.text(i, v + 2, f'${v:.0f}', ha='center', fontweight='bold', fontsize=10)
    save_fig('03_weekly_pattern.png')


# ---- 4. Actual vs Predicted ----------------------------------------------
def plot_actual_vs_predicted(df):
    baseline_features = ['month', 'day_of_week', 'is_weekend', 'promotion']
    advanced_features = [
        'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
        'day_of_month', 'week_of_year', 'quarter', 'is_weekend', 'promotion',
        'sales_lag_1', 'sales_lag_2', 'sales_lag_3',
        'sales_lag_7', 'sales_lag_14', 'sales_lag_28',
        'sales_roll_mean_7', 'sales_roll_mean_14', 'sales_roll_mean_30',
        'sales_roll_std_7', 'sales_roll_std_14', 'sales_roll_std_30',
        'sales_expanding_mean',
    ]

    y = df['sales']
    split_idx = int(len(df) * 0.8)
    test_dates = df['date'].iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Baseline
    X_bl = df[baseline_features]
    lr = LinearRegression().fit(X_bl.iloc[:split_idx], y_train)
    lr_preds = lr.predict(X_bl.iloc[split_idx:])
    lr_mae = mean_absolute_error(y_test, lr_preds)

    # Advanced
    X_adv = df[advanced_features]
    gb = GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.1,
                                   subsample=0.8, random_state=42)
    gb.fit(X_adv.iloc[:split_idx], y_train)
    gb_preds = gb.predict(X_adv.iloc[split_idx:])
    gb_mae = mean_absolute_error(y_test, gb_preds)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(test_dates, y_test.values, alpha=0.7, label='Actual', linewidth=1)
    axes[0].plot(test_dates, lr_preds, alpha=0.8, label='Linear Regression', linewidth=1.2, color='orange')
    axes[0].set_title(f'Baseline: Linear Regression  (MAE = {lr_mae:.2f})', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].set_ylabel('Sales ($)')

    axes[1].plot(test_dates, y_test.values, alpha=0.7, label='Actual', linewidth=1)
    axes[1].plot(test_dates, gb_preds, alpha=0.8, label='Gradient Boosting', linewidth=1.2, color='green')
    axes[1].set_title(f'Advanced: Gradient Boosting  (MAE = {gb_mae:.2f})', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].set_ylabel('Sales ($)')
    axes[1].set_xlabel('Date')

    fig.suptitle('Actual vs Predicted Sales (Test Set)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig('04_actual_vs_predicted.png')
    return lr_mae, gb_mae


# ---- 5. Feature Importance -----------------------------------------------
def plot_feature_importance(df):
    advanced_features = [
        'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
        'day_of_month', 'week_of_year', 'quarter', 'is_weekend', 'promotion',
        'sales_lag_1', 'sales_lag_2', 'sales_lag_3',
        'sales_lag_7', 'sales_lag_14', 'sales_lag_28',
        'sales_roll_mean_7', 'sales_roll_mean_14', 'sales_roll_mean_30',
        'sales_roll_std_7', 'sales_roll_std_14', 'sales_roll_std_30',
        'sales_expanding_mean',
    ]
    X = df[advanced_features]
    y = df['sales']
    split_idx = int(len(df) * 0.8)
    gb = GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.1,
                                   subsample=0.8, random_state=42)
    gb.fit(X.iloc[:split_idx], y.iloc[:split_idx])

    importances = gb.feature_importances_
    sorted_idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(advanced_features)))
    ax.barh(np.array(advanced_features)[sorted_idx], importances[sorted_idx], color=colors)
    ax.set_title('Feature Importance (Gradient Boosting)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Importance')
    save_fig('05_feature_importance.png')


# ---- 6. Model Comparison -------------------------------------------------
def plot_model_comparison(lr_mae, gb_mae):
    fig, ax = plt.subplots(figsize=(8, 6))
    models = ['Linear Regression\n(Baseline)', 'Gradient Boosting\n(Advanced)']
    maes = [lr_mae, gb_mae]
    colors = ['#e74c3c', '#2ecc71']
    bars = ax.bar(models, maes, color=colors, width=0.5, edgecolor='white', linewidth=1.5)

    for bar, mae in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'MAE: {mae:.2f}', ha='center', fontweight='bold', fontsize=13)

    improvement = ((lr_mae - gb_mae) / lr_mae) * 100
    ax.set_title(f'Model Comparison -- {improvement:.1f}% Error Reduction',
                 fontsize=16, fontweight='bold')
    ax.set_ylabel('Mean Absolute Error (MAE)')
    ax.set_ylim(0, max(maes) * 1.25)
    save_fig('06_model_comparison.png')


# ---- 7. Residual Analysis ------------------------------------------------
def plot_residuals(df):
    advanced_features = [
        'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
        'day_of_month', 'week_of_year', 'quarter', 'is_weekend', 'promotion',
        'sales_lag_1', 'sales_lag_2', 'sales_lag_3',
        'sales_lag_7', 'sales_lag_14', 'sales_lag_28',
        'sales_roll_mean_7', 'sales_roll_mean_14', 'sales_roll_mean_30',
        'sales_roll_std_7', 'sales_roll_std_14', 'sales_roll_std_30',
        'sales_expanding_mean',
    ]
    X = df[advanced_features]
    y = df['sales']
    split_idx = int(len(df) * 0.8)
    gb = GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.1,
                                   subsample=0.8, random_state=42)
    gb.fit(X.iloc[:split_idx], y.iloc[:split_idx])
    preds = gb.predict(X.iloc[split_idx:])
    residuals = y.iloc[split_idx:].values - preds

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(residuals, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
    axes[0].axvline(0, color='red', linestyle='--', linewidth=1.5)
    axes[0].set_title('Residual Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Residual (Actual - Predicted)')
    axes[0].set_ylabel('Frequency')

    axes[1].scatter(preds, residuals, alpha=0.4, s=15, color='steelblue')
    axes[1].axhline(0, color='red', linestyle='--', linewidth=1.5)
    axes[1].set_title('Residuals vs Predicted', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted Sales ($)')
    axes[1].set_ylabel('Residual')

    plt.suptitle('Gradient Boosting -- Residual Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig('07_residual_analysis.png')


# ---- Main ----------------------------------------------------------------
def main():
    print("=" * 60)
    print(" Sales Forecasting -- Generating Visualizations")
    print("=" * 60)

    data_path = 'data/historical_sales.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found at {data_path}. Run generate_data.py first.")

    df_raw = pd.read_csv(data_path, parse_dates=['date'])
    df = create_features(df_raw.copy())

    print("\n[1/7] Sales time series...")
    plot_sales_timeseries(df_raw)

    print("[2/7] Monthly seasonality...")
    plot_monthly_seasonality(df_raw)

    print("[3/7] Weekly pattern...")
    plot_weekly_pattern(df_raw)

    print("[4/7] Actual vs predicted...")
    lr_mae, gb_mae = plot_actual_vs_predicted(df)

    print("[5/7] Feature importance...")
    plot_feature_importance(df)

    print("[6/7] Model comparison...")
    plot_model_comparison(lr_mae, gb_mae)

    print("[7/7] Residual analysis...")
    plot_residuals(df)

    print("\nAll 7 visualizations saved to outputs/")


if __name__ == "__main__":
    main()
