import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sales_data(start_date='2021-01-01', end_date='2023-12-31'):
    np.random.seed(42)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n = len(dates)
    
    # 1. Trend: Steady growth
    trend = np.linspace(100, 300, n)
    
    # 2. Seasonality: Monthly cycles (higher in Dec/holiday season)
    month_effect = {
        1: 0.8, 2: 0.85, 3: 1.0, 4: 1.05, 5: 1.1, 6: 1.0,
        7: 0.95, 8: 1.0, 9: 1.1, 10: 1.2, 11: 1.4, 12: 1.8
    }
    seasonality = np.array([month_effect[d.month] for d in dates])
    
    # 3. Weekly pattern (higher on weekends)
    day_effect = {0: 1.0, 1: 0.9, 2: 0.9, 3: 0.95, 4: 1.1, 5: 1.3, 6: 1.3}
    weekly = np.array([day_effect[d.weekday()] for d in dates])
    
    # 4. Random Noise
    noise = np.random.normal(0, 15, n)
    
    # Combine
    sales = (trend * seasonality * weekly) + noise
    sales = np.maximum(sales, 0) # No negative sales
    
    df = pd.DataFrame({
        'date': dates,
        'sales': sales.round(2),
        'promotion': np.random.choice([0, 1], size=n, p=[0.9, 0.1])
    })
    
    # Add promotion boost
    df.loc[df['promotion'] == 1, 'sales'] *= 1.25
    
    return df

if __name__ == "__main__":
    data_path = 'data/historical_sales.csv'
    df = generate_sales_data()
    df.to_csv(data_path, index=False)
    print(f"Synthetic data generated and saved to {data_path}")
    print(df.head())
