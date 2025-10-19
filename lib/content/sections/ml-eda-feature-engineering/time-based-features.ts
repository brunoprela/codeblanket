/**
 * Time-Based Features Section
 */

export const timebasedfeaturesSection = {
  id: 'time-based-features',
  title: 'Time-Based Features',
  content: `# Time-Based Features

## Introduction

Time is one of the most informative features in many real-world applications - from predicting customer churn to forecasting stock prices. Proper time-based feature engineering can capture trends, seasonality, cycles, and temporal patterns that dramatically improve model performance.

**Why Time-Based Features Matter**:
- **Capture Seasonality**: Daily, weekly, monthly, yearly patterns
- **Encode Trends**: Long-term changes over time
- **Model Cycles**: Recurring patterns and rhythms
- **Handle Time Decay**: Recent events matter more
- **Enable Time Series Prediction**: Future forecasting

## Extracting Date Components

### Basic Temporal Features

\\\`\\\`\\\`python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

print("=" * 70)
print("TIME-BASED FEATURE ENGINEERING")
print("=" * 70)

# Create sample time series data
np.random.seed(42)
dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
n = len(dates)

df = pd.DataFrame({
    'date': dates,
    'sales': 1000 + np.random.randn(n) * 100 + 
             np.sin(np.arange(n) * 2 * np.pi / 365) * 200 +  # Yearly seasonality
             np.sin(np.arange(n) * 2 * np.pi / 7) * 50 +      # Weekly seasonality
             np.arange(n) * 0.5                                # Trend
})

print("\\nOriginal Data:")
print(df.head())

# Extract date components
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek  # Monday=0, Sunday=6
df['day_of_year'] = df['date'].dt.dayofyear
df['week_of_year'] = df['date'].dt.isocalendar().week
df['quarter'] = df['date'].dt.quarter
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)

print("\\nExtracted Date Components:")
print(df[['date', 'year', 'month', 'day_of_week', 'quarter', 'is_weekend']].head(10))

# Visualize weekly pattern
weekly_avg = df.groupby('day_of_week')['sales'].mean()
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

plt.figure(figsize=(10, 4))
plt.bar(range(7), weekly_avg.values)
plt.xticks(range(7), days)
plt.xlabel('Day of Week')
plt.ylabel('Average Sales')
plt.title('Weekly Seasonality Pattern')
plt.grid(True, alpha=0.3)
plt.show()

print("\\n✓ Basic temporal features extracted")
\\\`\\\`\\\`

## Cyclical Feature Encoding

### Handling Cyclical Time Features

\\\`\\\`\\\`python
def encode_cyclical_features(df, col, max_val):
    """Encode cyclical features using sine and cosine transformation"""
    
    print(f"\\nCYCLICAL ENCODING: {col}")
    print("=" * 70)
    
    # Problem: day 1 and day 31 of month are close but encoded as far apart
    # Solution: Use sine/cosine to preserve circular nature
    
    df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / max_val)
    df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / max_val)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Original encoding (problematic)
    axes[0].scatter(df[col], df['sales'], alpha=0.5)
    axes[0].set_xlabel(f'{col} (linear encoding)')
    axes[0].set_ylabel('Sales')
    axes[0].set_title(f'Linear Encoding (gap between {max_val} and 1)')
    axes[0].grid(True, alpha=0.3)
    
    # Sine component
    axes[1].scatter(df[f'{col}_sin'], df['sales'], alpha=0.5)
    axes[1].set_xlabel(f'{col}_sin')
    axes[1].set_ylabel('Sales')
    axes[1].set_title('Sine Component (preserves cyclical)')
    axes[1].grid(True, alpha=0.3)
    
    # Sine vs Cosine (circular pattern)
    axes[2].scatter(df[f'{col}_sin'], df[f'{col}_cos'], 
                   c=df[col], cmap='hsv', alpha=0.5)
    axes[2].set_xlabel(f'{col}_sin')
    axes[2].set_ylabel(f'{col}_cos')
    axes[2].set_title('Sine vs Cosine (forms circle)')
    axes[2].grid(True, alpha=0.3)
    plt.colorbar(axes[2].collections[0], ax=axes[2], label=col)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\\n✓ Created {col}_sin and {col}_cos")
    print(f"  Benefits: Preserves circular nature (December close to January)")
    
    return df

# Encode cyclical features
df = encode_cyclical_features(df, 'month', 12)
df = encode_cyclical_features(df, 'day_of_week', 7)
df = encode_cyclical_features(df, 'day_of_year', 365)
\\\`\\\`\\\`

## Lag Features

### Creating Historical Values

\\\`\\\`\\\`python
def create_lag_features(df, target_col, lags=[1, 7, 30]):
    """Create lag features (past values)"""
    
    print(f"\\nLAG FEATURES: {target_col}")
    print("=" * 70)
    
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    print(f"\\nCreated lag features: {lags}")
    print(f"\\nSample data:")
    print(df[['date', target_col] + [f'{target_col}_lag_{lag}' for lag in lags]].head(35))
    
    # Visualize lag correlation
    lag_cols = [f'{target_col}_lag_{lag}' for lag in lags]
    correlations = df[lag_cols + [target_col]].corr()[target_col][:-1]
    
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(lags)), correlations.values)
    plt.xticks(range(len(lags)), [f'Lag {lag}' for lag in lags])
    plt.xlabel('Lag Feature')
    plt.ylabel(f'Correlation with {target_col}')
    plt.title('Lag Feature Correlation with Target')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"\\n✓ Lag features capture temporal dependencies")
    print(f"  Example: Yesterday's sales predict today's sales")
    
    return df

# Create lag features
df = create_lag_features(df, 'sales', lags=[1, 7, 30, 365])
\\\`\\\`\\\`

## Rolling Window Statistics

### Aggregating Over Time Windows

\\\`\\\`\\\`python
def create_rolling_features(df, target_col, windows=[7, 30, 90]):
    """Create rolling window statistics"""
    
    print(f"\\nROLLING WINDOW FEATURES: {target_col}")
    print("=" * 70)
    
    for window in windows:
        # Mean
        df[f'{target_col}_rolling_mean_{window}'] = (
            df[target_col].rolling(window=window, min_periods=1).mean()
        )
        
        # Standard deviation
        df[f'{target_col}_rolling_std_{window}'] = (
            df[target_col].rolling(window=window, min_periods=1).std()
        )
        
        # Min and Max
        df[f'{target_col}_rolling_min_{window}'] = (
            df[target_col].rolling(window=window, min_periods=1).min()
        )
        df[f'{target_col}_rolling_max_{window}'] = (
            df[target_col].rolling(window=window, min_periods=1).max()
        )
    
    print(f"\\nCreated rolling features for windows: {windows}")
    
    # Visualize rolling means
    plt.figure(figsize=(14, 6))
    plt.plot(df['date'], df[target_col], label='Actual', alpha=0.5)
    
    for window in windows:
        plt.plot(df['date'], df[f'{target_col}_rolling_mean_{window}'],
                label=f'{window}-day MA', linewidth=2)
    
    plt.xlabel('Date')
    plt.ylabel(target_col)
    plt.title('Rolling Averages (Moving Averages)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"\\n✓ Rolling features smooth out noise and capture trends")
    
    return df

# Create rolling features
df = create_rolling_features(df, 'sales', windows=[7, 30, 90])
\\\`\\\`\\\`

## Time Since Events

### Elapsed Time Features

\\\`\\\`\\\`python
def create_time_since_features(df, date_col):
    """Create 'time since' features"""
    
    print("\\nTIME SINCE FEATURES")
    print("=" * 70)
    
    # Time since start of data
    df['days_since_start'] = (df[date_col] - df[date_col].min()).dt.days
    
    # Time since start of year
    df['days_since_year_start'] = (
        df[date_col] - pd.to_datetime(df[date_col].dt.year.astype(str) + '-01-01')
    ).dt.days
    
    # Time until end of year
    df['days_until_year_end'] = (
        pd.to_datetime(df[date_col].dt.year.astype(str) + '-12-31') - df[date_col]
    ).dt.days
    
    # Simulate event-based features (e.g., marketing campaign)
    campaign_dates = pd.to_datetime(['2021-06-01', '2022-06-01', '2023-06-01'])
    
    def days_since_last_campaign(date):
        past_campaigns = campaign_dates[campaign_dates <= date]
        if len(past_campaigns) == 0:
            return 9999  # No campaign yet
        return (date - past_campaigns.max()).days
    
    df['days_since_campaign'] = df[date_col].apply(days_since_last_campaign)
    
    print("\\nCreated time-since features:")
    print(df[['date', 'days_since_start', 'days_since_year_start', 
             'days_since_campaign']].head(10))
    
    # Visualize campaign effect
    plt.figure(figsize=(14, 6))
    plt.plot(df['date'], df['sales'], label='Sales', alpha=0.7)
    
    for campaign_date in campaign_dates:
        plt.axvline(campaign_date, color='r', linestyle='--', alpha=0.7, 
                   label='Campaign' if campaign_date == campaign_dates[0] else '')
    
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.title('Time Since Events (Campaigns)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\\n✓ Time-since features capture event impacts")
    
    return df

# Create time-since features
df = create_time_since_features(df, 'date')
\\\`\\\`\\\`

## Time-Based Aggregations

### Group-by Time Periods

\\\`\\\`\\\`python
def create_time_aggregations(df, date_col, target_col):
    """Create aggregated features by time periods"""
    
    print("\\nTIME-BASED AGGREGATIONS")
    print("=" * 70)
    
    # Monthly statistics
    df['year_month'] = df[date_col].dt.to_period('M')
    monthly_stats = df.groupby('year_month')[target_col].agg([
        'mean', 'std', 'min', 'max', 'sum'
    ]).reset_index()
    
    monthly_stats.columns = ['year_month', 'monthly_mean', 'monthly_std',
                            'monthly_min', 'monthly_max', 'monthly_sum']
    
    # Merge back
    df = df.merge(monthly_stats, on='year_month', how='left')
    
    # Same for weekly
    df['year_week'] = df[date_col].dt.to_period('W')
    weekly_stats = df.groupby('year_week')[target_col].agg(['mean', 'std']).reset_index()
    weekly_stats.columns = ['year_week', 'weekly_mean', 'weekly_std']
    df = df.merge(weekly_stats, on='year_week', how='left')
    
    print("\\nCreated aggregated features:")
    print(df[['date', 'sales', 'monthly_mean', 'weekly_mean']].head(10))
    
    # Visualize monthly aggregations
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    axes[0].plot(df['date'], df['sales'], alpha=0.5, label='Daily Sales')
    axes[0].plot(df['date'], df['monthly_mean'], linewidth=2, label='Monthly Mean')
    axes[0].set_ylabel('Sales')
    axes[0].set_title('Daily Sales vs Monthly Average')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(df['date'], df['monthly_std'], linewidth=2, color='orange')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Monthly Std Dev')
    axes[1].set_title('Monthly Volatility')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\\n✓ Aggregations capture period-level patterns")
    
    return df

# Create aggregations
df = create_time_aggregations(df, 'date', 'sales')
\\\`\\\`\\\`

## Holiday and Special Event Features

### Encoding Special Days

\\\`\\\`\\\`python
def create_holiday_features(df, date_col, country='US'):
    """Create features for holidays and special events"""
    
    print("\\nHOLIDAY AND SPECIAL EVENT FEATURES")
    print("=" * 70)
    
    # Major US holidays (simplified - use holidays library in production)
    holidays_dict = {
        '01-01': 'New Year',
        '07-04': 'Independence Day',
        '12-25': 'Christmas',
        '11-28': 'Thanksgiving',  # Approximate (4th Thursday of November)
    }
    
    df['month_day'] = df[date_col].dt.strftime('%m-%d')
    df['is_major_holiday'] = df['month_day'].isin(holidays_dict.keys()).astype(int)
    
    # Days before/after holiday
    df['days_to_holiday'] = 0
    df['days_from_holiday'] = 0
    
    for date in df[date_col]:
        # Find closest holiday
        holiday_dates = pd.to_datetime(
            [f"{date.year}-{h}" for h in holidays_dict.keys()]
        )
        
        future_holidays = holiday_dates[holiday_dates >= date]
        if len(future_holidays) > 0:
            df.loc[df[date_col] == date, 'days_to_holiday'] = (
                future_holidays.min() - date
            ).days
        
        past_holidays = holiday_dates[holiday_dates <= date]
        if len(past_holidays) > 0:
            df.loc[df[date_col] == date, 'days_from_holiday'] = (
                date - past_holidays.max()
            ).days
    
    # Paycheck days (typically 15th and last day of month)
    df['is_payday'] = (
        (df[date_col].dt.day == 15) | 
        df[date_col].dt.is_month_end
    ).astype(int)
    
    print("\\nHoliday features created:")
    holiday_samples = df[df['is_major_holiday'] == 1][['date', 'is_major_holiday']].head()
    print(holiday_samples)
    
    print("\\n✓ Holiday features capture special event patterns")
    
    return df

# Create holiday features
df = create_holiday_features(df, 'date')
\\\`\\\`\\\`

## Key Takeaways

1. **Extract date components**: year, month, day, day_of_week, quarter
2. **Use cyclical encoding**: sine/cosine for circular features (month, hour)
3. **Create lag features**: past values as predictors (yesterday, last week)
4. **Rolling statistics**: moving averages capture trends
5. **Time since events**: days since campaign, last purchase, etc.
6. **Holiday encoding**: capture special day effects
7. **Trend features**: days_since_start captures long-term trends
8. **Aggregations**: monthly/weekly means provide context
9. **Maintain temporal order**: no future information in features
10. **Handle missing values**: first few rows have NaN in lags/rolling

## Connection to Machine Learning

- **Lag features** essential for time series forecasting
- **Cyclical encoding** prevents model from treating December and January as far apart
- **Rolling features** smooth noise and capture medium-term trends
- **Holiday features** critical for retail and e-commerce predictions
- **Time-based features** often among top predictors in temporal data
- **Proper feature engineering** can improve time series models 30-70%
- **Avoid data leakage**: only use past information to predict future

Time-based feature engineering is critical for any dataset with temporal components - master these techniques for real-world applications.
`,
};
