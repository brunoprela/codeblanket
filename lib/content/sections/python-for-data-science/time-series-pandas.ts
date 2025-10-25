/**
 * Section: Time Series with Pandas
 * Module: Python for Data Science
 *
 * Covers DatetimeIndex, resampling, rolling windows, time-based indexing, and timezone handling
 */

export const timeSeriesPandas = {
  id: 'time-series-pandas',
  title: 'Time Series with Pandas',
  content: `
# Time Series with Pandas

## Introduction

Time series data is ubiquitous in real-world applications: stock prices, sensor readings, web traffic, sales data, and more. Pandas provides powerful tools for working with time series data, making it easy to resample, aggregate, and analyze temporal patterns.

**Key Concepts:**
- DatetimeIndex for time-aware indexing
- Resampling for changing frequency
- Rolling windows for moving calculations
- Time zones and daylight saving time
- Missing data handling in time series

\`\`\`python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
\`\`\`

## Creating DatetimeIndex

### From Strings

\`\`\`python
# Create DataFrame with date strings
df = pd.DataFrame({
    'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'],
    'value': [100, 105, 103, 108]
})

# Convert to datetime
df['date'] = pd.to_datetime (df['date'])
print(df['date'].dtype)  # datetime64[ns]

# Set as index
df = df.set_index('date')
print(df.index)
# DatetimeIndex(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'], 
#               dtype='datetime64[ns]', name='date', freq=None)
\`\`\`

### Using date_range

\`\`\`python
# Generate date range
dates = pd.date_range (start='2024-01-01', end='2024-12-31', freq='D')
print(f"Generated {len (dates)} dates")  # 366 dates (2024 is leap year)

# Different frequencies
hourly = pd.date_range('2024-01-01', periods=24, freq='h')
business_days = pd.date_range('2024-01-01', periods=20, freq='B')  # Business days
monthly = pd.date_range('2024-01-01', periods=12, freq='MS')  # Month start
quarterly = pd.date_range('2024-01-01', periods=4, freq='QS')  # Quarter start

# Create time series DataFrame
df = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=100, freq='D'),
    'value': np.random.randn(100).cumsum() + 100
})
df = df.set_index('date')

print(df.head())
#             value
# date             
# 2024-01-01  100.5
# 2024-01-02   99.8
# 2024-01-03  101.2
# ...
\`\`\`

### Common Frequencies

\`\`\`python
# Frequency codes:
# D  - calendar day
# B  - business day  
# W  - weekly
# M  - month end
# MS - month start
# Q  - quarter end
# QS - quarter start
# A  - year end
# AS - year start
# H  - hourly
# T, min - minutely
# S  - secondly

# Custom frequencies
every_3_days = pd.date_range('2024-01-01', periods=10, freq='3D')
every_4_hours = pd.date_range('2024-01-01 00:00', periods=10, freq='4h')
print(every_3_days)
print(every_4_hours)
\`\`\`

## Time-Based Indexing and Selection

### Selecting by Date

\`\`\`python
# Create sample data
dates = pd.date_range('2024-01-01', periods=365, freq='D')
df = pd.DataFrame({
    'value': np.random.randn(365).cumsum() + 100
}, index=dates)

# Select specific date
print(df.loc['2024-01-15'])

# Select date range
jan_data = df.loc['2024-01':'2024-01']
print(f"January data: {len (jan_data)} days")

# Select by partial date
q1_data = df.loc['2024-01':'2024-03']
print(f"Q1 data: {len (q1_data)} days")

# Select by year
year_2024 = df.loc['2024']
print(f"Year 2024: {len (year_2024)} days")

# Boolean indexing with dates
recent = df[df.index > '2024-06-01']
print(f"After June 1: {len (recent)} days")

# Between dates
summer = df[(df.index >= '2024-06-01') & (df.index <= '2024-08-31')]
print(f"Summer months: {len (summer)} days")
\`\`\`

### Datetime Properties

\`\`\`python
# Extract date components
df['year'] = df.index.year
df['month'] = df.index.month
df['day'] = df.index.day
df['dayofweek'] = df.index.dayofweek  # Monday=0, Sunday=6
df['quarter'] = df.index.quarter
df['dayofyear'] = df.index.dayofyear
df['week'] = df.index.isocalendar().week

# String representations
df['month_name'] = df.index.month_name()
df['day_name'] = df.index.day_name()

print(df.head())
#             value  year  month  day  dayofweek  quarter  month_name day_name
# date                                                                        
# 2024-01-01  100.5  2024      1    1          0        1     January   Monday
# 2024-01-02   99.8  2024      1    2          1        1     January  Tuesday
# ...

# Filter by day of week (weekdays only)
weekdays = df[df.index.dayofweek < 5]
print(f"Weekdays: {len (weekdays)} / {len (df)} days")

# Filter by month
summer_months = df[df.index.month.isin([6, 7, 8])]
print(f"Summer: {len (summer_months)} days")
\`\`\`

## Resampling

Resampling changes the frequency of time series data (upsampling or downsampling).

### Downsampling (Higher to Lower Frequency)

\`\`\`python
# Daily data
dates = pd.date_range('2024-01-01', periods=365, freq='D')
df = pd.DataFrame({
    'sales': np.random.randint(100, 500, 365),
    'customers': np.random.randint(10, 50, 365)
}, index=dates)

# Resample to weekly (sum)
weekly = df.resample('W').sum()
print(f"Daily: {len (df)} rows → Weekly: {len (weekly)} rows")
print(weekly.head())

# Resample to monthly (various aggregations)
monthly = df.resample('ME').agg({  # ME = Month End
    'sales': 'sum',
    'customers': 'mean'
})
print(monthly.head())
#             sales  customers
# 2024-01-31  12450       28.5
# 2024-02-29  11980       29.1
# ...

# Multiple aggregations
monthly_detailed = df.resample('ME').agg({
    'sales': ['sum', 'mean', 'std', 'min', 'max'],
    'customers': ['sum', 'mean']
})
print(monthly_detailed)

# Resample with custom function
def sales_volatility (x):
    return x.std() / x.mean() if x.mean() > 0 else 0

monthly_vol = df.resample('ME')['sales'].apply (sales_volatility)
print(f"Monthly sales volatility:\\n{monthly_vol.head()}")
\`\`\`

### Upsampling (Lower to Higher Frequency)

\`\`\`python
# Monthly data
dates = pd.date_range('2024-01-01', periods=12, freq='MS')
df_monthly = pd.DataFrame({
    'revenue': [100000, 110000, 105000, 115000, 120000, 125000,
                130000, 128000, 135000, 140000, 145000, 150000]
}, index=dates)

# Upsample to daily (forward fill)
daily_ffill = df_monthly.resample('D').ffill()
print(f"Monthly: {len (df_monthly)} → Daily (ffill): {len (daily_ffill)}")
print(daily_ffill.head(5))
#             revenue
# 2024-01-01   100000
# 2024-01-02   100000  # Forward filled
# 2024-01-03   100000  # Forward filled
# ...

# Upsample with interpolation
daily_interp = df_monthly.resample('D').interpolate (method='linear')
print(daily_interp.head(5))
#             revenue
# 2024-01-01  100000.00
# 2024-01-02  100322.58  # Linearly interpolated
# 2024-01-03  100645.16
# ...

# Backward fill
daily_bfill = df_monthly.resample('D').bfill()
print(daily_bfill.head(5))
\`\`\`

### Resampling with Offset

\`\`\`python
# Start week on Monday vs Sunday
weekly_mon = df.resample('W-MON').sum()
weekly_sun = df.resample('W-SUN').sum()

print(f"Week starting Monday: {len (weekly_mon)} weeks")
print(f"Week starting Sunday: {len (weekly_sun)} weeks")

# Custom period boundaries
# Business quarter end
quarterly = df.resample('BQ').sum()
print(f"Business quarters: {len (quarterly)}")
\`\`\`

## Rolling Windows

Rolling windows compute statistics over a moving window of data.

### Simple Rolling Calculations

\`\`\`python
# Stock price data
dates = pd.date_range('2024-01-01', periods=252, freq='B')  # Trading days
df = pd.DataFrame({
    'price': np.random.randn(252).cumsum() + 100
}, index=dates)

# Simple moving average (SMA)
df['SMA_20'] = df['price'].rolling (window=20).mean()
df['SMA_50'] = df['price'].rolling (window=50).mean()

# Exponential moving average (EMA)
df['EMA_20'] = df['price'].ewm (span=20, adjust=False).mean()

# Rolling standard deviation (volatility)
df['volatility_20'] = df['price'].rolling (window=20).std()

# Rolling min and max
df['high_20'] = df['price'].rolling (window=20).max()
df['low_20'] = df['price'].rolling (window=20).min()

print(df.tail())
#              price     SMA_20     SMA_50     EMA_20  volatility_20
# 2024-12-30  115.23     114.56     113.89     114.78           2.34
# ...
\`\`\`

### Advanced Rolling Operations

\`\`\`python
# Rolling correlation
df['returns'] = df['price'].pct_change()
df['returns_vol_corr'] = df['returns'].rolling (window=50).corr(
    df['volatility_20'].rolling (window=50).std()
)

# Custom rolling function
def sharpe_ratio (returns, window=252):
    """Annualized Sharpe ratio"""
    return np.sqrt (window) * returns.mean() / returns.std()

df['rolling_sharpe'] = df['returns'].rolling (window=50).apply(
    lambda x: sharpe_ratio (x, window=252)
)

# Rolling quantiles
df['price_75th'] = df['price'].rolling (window=20).quantile(0.75)
df['price_25th'] = df['price'].rolling (window=20).quantile(0.25)

# Bollinger Bands
window = 20
df['BB_middle'] = df['price'].rolling (window=window).mean()
df['BB_std'] = df['price'].rolling (window=window).std()
df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']

print(df[['price', 'BB_upper', 'BB_middle', 'BB_lower']].tail())
\`\`\`

### Centered Rolling Windows

\`\`\`python
# Center the window (equal weights before and after)
df['centered_ma'] = df['price'].rolling (window=5, center=True).mean()

# Compare
print(df[['price', 'SMA_20', 'centered_ma']].head(10))
\`\`\`

### Rolling with Custom Windows

\`\`\`python
# Variable-size window based on date range
df['rolling_7d'] = df['price'].rolling (window='7D').mean()

# Min periods (minimum observations required)
df['sma_min10'] = df['price'].rolling (window=20, min_periods=10).mean()
# First 10-19 observations will have values (instead of NaN)
\`\`\`

## Time Shifts and Lags

\`\`\`python
# Shift values forward (lag)
df['price_lag1'] = df['price'].shift(1)  # Yesterday\'s price
df['price_lag5'] = df['price'].shift(5)  # Price 5 days ago

# Shift values backward (lead)
df['price_lead1'] = df['price'].shift(-1)  # Tomorrow\'s price

# Calculate returns
df['return_1d'] = df['price'].pct_change(1)  # 1-day return
df['return_5d'] = df['price'].pct_change(5)  # 5-day return

# Difference
df['price_diff1'] = df['price'].diff(1)  # Price change from yesterday
df['price_diff5'] = df['price'].diff(5)  # Price change from 5 days ago

# Shift by time period
df['price_1week_ago'] = df['price'].shift (freq='7D')

print(df[['price', 'price_lag1', 'return_1d', 'price_diff1']].head(10))
#              price  price_lag1  return_1d  price_diff1
# 2024-01-01  100.00         NaN        NaN          NaN
# 2024-01-02  101.50      100.00     0.0150         1.50
# 2024-01-03   99.80      101.50    -0.0167        -1.70
# ...
\`\`\`

## Handling Missing Data in Time Series

\`\`\`python
# Create data with missing dates
dates = pd.date_range('2024-01-01', periods=100, freq='D')
df = pd.DataFrame({
    'value': np.random.randn(100).cumsum() + 100
}, index=dates)

# Remove random dates
df = df.drop (df.sample (n=10).index)
print(f"Missing {100 - len (df)} dates")

# Reindex to include all dates
df = df.reindex (pd.date_range('2024-01-01', periods=100, freq='D'))
print(f"After reindex: {df.isna().sum().values[0]} NaN values")

# Forward fill
df['ffill'] = df['value'].ffill()

# Backward fill
df['bfill'] = df['value'].bfill()

# Interpolate
df['interpolate_linear'] = df['value'].interpolate (method='linear')
df['interpolate_time'] = df['value'].interpolate (method='time')

# Fill with rolling mean
df['rolling_fill'] = df['value'].fillna(
    df['value'].rolling (window=7, min_periods=1, center=True).mean()
)

print(df[df['value'].isna()].head())
\`\`\`

## Timezone Handling

\`\`\`python
# Create timezone-naive datetime
df = pd.DataFrame({
    'value': range(24)
}, index=pd.date_range('2024-01-01', periods=24, freq='h'))

print(f"Timezone: {df.index.tz}")  # None

# Localize to timezone
df = df.tz_localize('UTC')
print(f"Timezone: {df.index.tz}")  # UTC

# Convert to different timezone
df_ny = df.tz_convert('America/New_York')
df_tokyo = df.tz_convert('Asia/Tokyo')

print("UTC vs NY vs Tokyo:")
print(pd.DataFrame({
    'UTC': df.index[:5],
    'NY': df_ny.index[:5],
    'Tokyo': df_tokyo.index[:5]
}))

# Handle daylight saving time
dates = pd.date_range('2024-03-09', periods=48, freq='h', tz='America/New_York')
df_dst = pd.DataFrame({'hour': dates.hour}, index=dates)
print("Around DST transition:")
print(df_dst.loc['2024-03-10'])  # Spring forward (skip hour)
\`\`\`

## Practical Examples

### Example 1: Stock Price Analysis

\`\`\`python
# Generate realistic stock data
np.random.seed(42)
dates = pd.date_range('2023-01-01', '2024-12-31', freq='B')
returns = np.random.normal(0.0005, 0.02, len (dates))
price = 100 * (1 + returns).cumprod()

df = pd.DataFrame({
    'price': price,
    'volume': np.random.randint(1000000, 10000000, len (dates))
}, index=dates)

# Technical indicators
df['SMA_20'] = df['price'].rolling(20).mean()
df['SMA_50'] = df['price'].rolling(50).mean()
df['SMA_200'] = df['price'].rolling(200).mean()

# Volatility (20-day)
df['returns'] = df['price'].pct_change()
df['volatility'] = df['returns'].rolling(20).std() * np.sqrt(252)

# RSI (Relative Strength Index)
delta = df['price'].diff()
gain = delta.where (delta > 0, 0).rolling(14).mean()
loss = -delta.where (delta < 0, 0).rolling(14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# MACD (Moving Average Convergence Divergence)
df['EMA_12'] = df['price'].ewm (span=12, adjust=False).mean()
df['EMA_26'] = df['price'].ewm (span=26, adjust=False).mean()
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['MACD_signal'] = df['MACD'].ewm (span=9, adjust=False).mean()
df['MACD_hist'] = df['MACD'] - df['MACD_signal']

# Trading signals
df['signal'] = 0
df.loc[df['SMA_20'] > df['SMA_50'], 'signal'] = 1  # Golden cross
df.loc[df['SMA_20'] < df['SMA_50'], 'signal'] = -1  # Death cross

print("Recent technical indicators:")
print(df[['price', 'SMA_20', 'SMA_50', 'RSI', 'signal']].tail())

# Monthly performance
monthly = df['returns'].resample('ME').agg([
    ('return', lambda x: (1 + x).prod() - 1),
    ('volatility', lambda x: x.std() * np.sqrt(21)),
    ('sharpe', lambda x: x.mean() / x.std() * np.sqrt(21) if x.std() > 0 else 0)
])
print("\\nMonthly performance:")
print(monthly.tail())
\`\`\`

### Example 2: Website Traffic Analysis

\`\`\`python
# Hourly website traffic
dates = pd.date_range('2024-01-01', '2024-12-31', freq='h')
# Simulate daily and weekly patterns
hour_pattern = np.sin (np.arange (len (dates)) * 2 * np.pi / 24) * 0.3 + 1
week_pattern = np.sin (np.arange (len (dates)) * 2 * np.pi / (24*7)) * 0.2 + 1
base_traffic = 1000
noise = np.random.normal(0, 100, len (dates))

df = pd.DataFrame({
    'visitors': (base_traffic * hour_pattern * week_pattern + noise).astype (int)
}, index=dates)

# Daily aggregation
daily = df.resample('D').agg({
    'visitors': ['sum', 'mean', 'max']
})
daily.columns = ['total_visitors', 'avg_visitors_per_hour', 'peak_visitors']

# Add day of week
daily['day_of_week'] = daily.index.day_name()
daily['is_weekend'] = daily.index.dayofweek >= 5

# Weekly patterns
weekly_pattern = daily.groupby('day_of_week')['total_visitors'].mean().sort_values (ascending=False)
print("Average visitors by day of week:")
print(weekly_pattern)

# Month-over-month growth
monthly = daily['total_visitors'].resample('ME').sum()
monthly_growth = monthly.pct_change() * 100
print("\\nMonth-over-month growth:")
print(monthly_growth)

# Anomaly detection (simple threshold)
daily['rolling_mean'] = daily['total_visitors'].rolling(7).mean()
daily['rolling_std'] = daily['total_visitors'].rolling(7).std()
daily['z_score'] = (daily['total_visitors'] - daily['rolling_mean']) / daily['rolling_std']
daily['anomaly'] = np.abs (daily['z_score']) > 3

print(f"\\nAnomalies detected: {daily['anomaly'].sum()} days")
print(daily[daily['anomaly']][['total_visitors', 'rolling_mean', 'z_score']].head())
\`\`\`

### Example 3: Sensor Data with Irregular Timestamps

\`\`\`python
# Simulate sensor readings with irregular intervals
base_time = pd.Timestamp('2024-01-01')
irregular_times = [base_time + pd.Timedelta (seconds=np.random.randint(0, 3600)) 
                   for _ in range(1000)]
irregular_times = sorted (irregular_times)

df = pd.DataFrame({
    'temperature': 20 + np.random.randn(1000) * 2,
    'humidity': 50 + np.random.randn(1000) * 10
}, index=irregular_times)

print(f"Original data: {len (df)} irregular timestamps")

# Resample to regular 5-minute intervals
regular = df.resample('5min').mean()
print(f"Resampled: {len (regular)} regular 5-min intervals")

# Interpolate missing values
regular_filled = regular.interpolate (method='time')
print(f"Missing values: {regular.isna().sum().sum()} → {regular_filled.isna().sum().sum()}")

# Rolling average (30 minutes)
regular_filled['temp_ma'] = regular_filled['temperature'].rolling (window='30min').mean()
regular_filled['humidity_ma'] = regular_filled['humidity'].rolling (window='30min').mean()

print(regular_filled.head(20))
\`\`\`

## Key Takeaways

1. **DatetimeIndex** enables powerful time-aware operations
2. **Resampling** changes frequency (upsample/downsample)
3. **Rolling windows** compute moving statistics
4. **Shift and lag** create lagged features
5. **Time-based indexing** uses partial dates and ranges
6. **Timezone handling** critical for global applications
7. **Missing data** requires special attention in time series
8. **Technical indicators** built from rolling operations

Pandas makes time series analysis intuitive and powerful!
`,
};
