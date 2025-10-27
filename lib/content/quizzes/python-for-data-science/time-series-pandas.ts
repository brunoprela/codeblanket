import { QuizQuestion } from '../../../types';

export const timeseriespandasQuiz: QuizQuestion[] = [
  {
    id: 'time-series-pandas-dq-1',
    question:
      'Explain the difference between resampling and rolling windows in time series analysis. When would you use each, and what are common pitfalls to avoid?',
    sampleAnswer: `Resampling and rolling windows are both fundamental time series operations, but they serve different purposes and produce different outputs.

**Resampling: Changing Frequency**

**What it does:**
- Changes the frequency of time series data
- Aggregates (downsampling) or expands (upsampling)
- Produces new index with different frequency

**Example - Downsampling:**
\`\`\`python
# Daily data
dates = pd.date_range('2024-01-01', periods=365, freq='D')
df = pd.DataFrame({'sales': np.random.randint(100, 500, 365)}, index=dates)

# Resample to monthly
monthly = df.resample('ME').sum()
# Result: 12 rows (one per month), each containing sum of daily sales

print(f"Daily: {len (df)} rows → Monthly: {len (monthly)} rows")
# Daily: 365 rows → Monthly: 12 rows
\`\`\`

**When to use resampling:**1. **Changing reporting period:**
\`\`\`python
# Convert hourly to daily reports
hourly_data.resample('D').agg({'sales': 'sum', 'customers': 'mean'})
\`\`\`

2. **Reducing data size:**
\`\`\`python
# 1-minute data too granular, aggregate to 15-minute
minute_data.resample('15min').mean()
\`\`\`

3. **Aligning different frequencies:**
\`\`\`python
# Align daily stock prices with monthly economic data
daily_prices.resample('ME').last()  # Last price of month
\`\`\`

4. **Handling irregular timestamps:**
\`\`\`python
# Sensor data with irregular intervals → regular 5-min grid
irregular_data.resample('5min').mean()
\`\`\`

**Rolling Windows: Moving Calculations**

**What it does:**
- Computes statistics over a moving window
- Maintains same frequency as input
- Each output value based on window of surrounding values

**Example:**
\`\`\`python
# Daily stock prices
df = pd.DataFrame({'price': range(100, 200)}, 
                  index=pd.date_range('2024-01-01', periods=100, freq='D'))

# 20-day moving average
df['MA_20',] = df['price',].rolling (window=20).mean()
# Result: 100 rows (same as input), each with 20-day average

print(f"Input: {len (df)} rows → Output: {len (df)} rows")
# Input: 100 rows → Output: 100 rows (same)
\`\`\`

**When to use rolling windows:**1. **Technical indicators:**
\`\`\`python
# Moving averages
df['SMA_50',] = df['price',].rolling(50).mean()
df['SMA_200',] = df['price',].rolling(200).mean()

# Volatility
df['volatility',] = df['returns',].rolling(20).std()

# Bollinger Bands
ma = df['price',].rolling(20).mean()
std = df['price',].rolling(20).std()
df['upper_band',] = ma + 2 * std
df['lower_band',] = ma - 2 * std
\`\`\`

2. **Smoothing noisy data:**
\`\`\`python
# Smooth sensor readings
df['temp_smooth',] = df['temperature',].rolling(10, center=True).mean()
\`\`\`

3. **Trend analysis:**
\`\`\`python
# Compare current value to recent average
df['above_average',] = df['sales',] > df['sales',].rolling(30).mean()
\`\`\`

4. **Feature engineering for ML:**
\`\`\`python
# Create lag features
df['price_ma_7',] = df['price',].rolling(7).mean()
df['price_ma_30',] = df['price',].rolling(30).mean()
df['momentum',] = df['price_ma_7',] / df['price_ma_30',]
\`\`\`

**Key Differences:**

| Aspect | Resampling | Rolling Windows |
|--------|------------|-----------------|
| **Output size** | Changes | Same as input |
| **Frequency** | Changes | Same as input |
| **Window** | Fixed (period) | Can be fixed or variable |
| **Purpose** | Change frequency | Moving statistics |
| **Result index** | New timestamps | Original timestamps |

**Common Pitfalls:**

**Pitfall 1: Look-ahead bias with rolling windows**
\`\`\`python
# WRONG: Center=True uses future data
df['ma_wrong',] = df['price',].rolling(20, center=True).mean()
# Uses 10 days before and 10 days after → can't use for prediction!

# CORRECT: Default (center=False) uses only past
df['ma_correct',] = df['price',].rolling(20).mean()
# Uses current day and 19 days before → safe for prediction
\`\`\`

**Pitfall 2: Forgetting to handle NaN at start of rolling**
\`\`\`python
# First 19 values are NaN
df['ma_20',] = df['price',].rolling(20).mean()

# Option 1: Use min_periods
df['ma_20',] = df['price',].rolling(20, min_periods=10).mean()
# Starts producing values after 10 observations

# Option 2: Drop NaN
df = df.dropna()
\`\`\`

**Pitfall 3: Resampling with wrong aggregation**
\`\`\`python
# WRONG: Mean of prices (not meaningful)
daily = hourly_prices.resample('D').mean()

# CORRECT: Last price of day (closing price)
daily = hourly_prices.resample('D').last()

# OR: OHLC (Open, High, Low, Close)
daily = hourly_prices.resample('D').agg({
    'price': ['first', 'max', 'min', 'last',]
})
\`\`\`

**Pitfall 4: Mixing up window sizes**
\`\`\`python
# WRONG: Window as number of periods (inconsistent for irregular data)
df.rolling (window=20).mean()  # 20 what? Days? Rows?

# CORRECT: Specify time-based window
df.rolling (window='20D').mean()  # Explicitly 20 days
\`\`\`

**Pitfall 5: Resampling loses information**
\`\`\`python
# Hourly data → Daily (24:1 compression)
daily = hourly.resample('D').mean()
# Lost information about intraday patterns!

# Better: Keep both levels
daily_agg = hourly.resample('D').agg(['mean', 'std', 'min', 'max',])
# Captures more information
\`\`\`

**Combined Usage:**

Often, you use both together:

\`\`\`python
# Example: Stock analysis
# 1. Resample to align different data sources
prices_daily = minute_prices.resample('D').last()
volume_daily = minute_volume.resample('D').sum()

# 2. Apply rolling windows for indicators
prices_daily['MA_50',] = prices_daily['close',].rolling(50).mean()
prices_daily['volume_avg',] = volume_daily.rolling(20).mean()

# 3. Resample for different timeframes
weekly_summary = prices_daily.resample('W').agg({
    'close': 'last',
    'MA_50': 'last',
    'volume_avg': 'mean'
})
\`\`\`

**Decision Framework:**

\`\`\`
Need to operate on time series?
│
├─ Want to change frequency?
│  ├─ Higher → Lower (downsample) → Use resample with aggregation
│  │  Example: Daily → Monthly
│  │
│  └─ Lower → Higher (upsample) → Use resample with fill/interpolate
│     Example: Monthly → Daily
│
└─ Want moving statistics at same frequency?
   └─ Use rolling windows
      ├─ Past data only (for prediction) → center=False
      └─ Smoothing (past & future OK) → center=True
\`\`\`

**Best Practices:**1. **Be explicit about aggregation:**
\`\`\`python
# Good: Clear intent
monthly = daily.resample('ME').agg({'sales': 'sum', 'price': 'mean'})

# Bad: Ambiguous
monthly = daily.resample('ME').mean()  # What about count? sum?
\`\`\`

2. **Use time-based windows:**
\`\`\`python
# Good: Time-aware
df.rolling (window='30D').mean()

# Bad: Row-based (inconsistent with irregular data)
df.rolling (window=30).mean()
\`\`\`

3. **Document window choices:**
\`\`\`python
# Why 20 days? Because typical trading month
df['ma_20',] = df['price',].rolling(20).mean()
\`\`\`

4. **Check for look-ahead bias:**
\`\`\`python
# For ML features, never use center=True
features = df.rolling(20).mean()  # Safe
\`\`\`

**Key Takeaway:**

- **Resampling**: Changes frequency (compress or expand timeline)
- **Rolling**: Moving statistics (same frequency, smooth or aggregate)

Use resampling when you need different time granularity. Use rolling windows when you need moving statistics at the same granularity. Both are essential tools in time series analysis!`,
    keyPoints: [
      'Resampling changes frequency (daily→monthly) and requires aggregation function',
      'Rolling windows compute moving statistics over fixed window size',
      'Resample output has new DatetimeIndex at target frequency',
      'Rolling output maintains original index and length',
      'Common pitfalls: forward-looking bias in rolling, choosing wrong aggregation',
    ],
  },
  {
    id: 'time-series-pandas-dq-2',
    question:
      'Discuss strategies for handling missing data in time series, particularly irregular timestamps and gaps. How do forward fill, backward fill, and interpolation differ, and when is each appropriate?',
    sampleAnswer: `Missing data in time series requires special handling because temporal ordering matters. The choice of strategy depends on the data's nature and your analysis goals.

**Types of Missing Data in Time Series:**1. **Irregular timestamps** (data arrives at non-uniform intervals)
2. **Gaps** (missing observations in regular series)
3. **Data quality issues** (sensors failures, transmission errors)
4. **Non-trading periods** (weekends, holidays in financial data)

**Three Main Strategies:**

**1. Forward Fill (ffill)**

**How it works:**
Propagates last valid observation forward

\`\`\`python
dates = pd.date_range('2024-01-01', periods=10, freq='D')
df = pd.DataFrame({'price': [100, 101, np.nan, np.nan, 105, 
                               np.nan, 107, 108, np.nan, 110]}, index=dates)

df['ffill',] = df['price',].ffill()
print(df)
#             price  ffill
# 2024-01-01  100.0  100.0
# 2024-01-02  101.0  101.0
# 2024-01-03    NaN  101.0  # Last known value
# 2024-01-04    NaN  101.0  # Still 101.0
# 2024-01-05  105.0  105.0
# 2024-01-06    NaN  105.0
# ...
\`\`\`

**When to use:**

✅ **Values persist until changed:**
\`\`\`python
# Inventory levels
inventory.ffill()  # Level stays same until restocking

# Account balances  
balances.ffill()  # Balance unchanged until transaction

# Status flags
status.ffill()  # Status persists until update
\`\`\`

✅ **Real-time systems:**
\`\`\`python
# Latest sensor reading
sensor_data.ffill()  # Use last known value

# Market prices (during non-trading hours)
prices.ffill()  # Price doesn't change when market closed
\`\`\`

❌ **Inappropriate for:**
- Trending data (temperature, sales) - creates artificial plateaus
- Flow measurements (traffic, throughput) - assumes flow continues
- Event counts - events don't repeat

**2. Backward Fill (bfill)**

**How it works:**
Uses next valid observation to fill backward

\`\`\`python
df['bfill',] = df['price',].bfill()
print(df[['price', 'ffill', 'bfill',]])
#             price  ffill  bfill
# 2024-01-01  100.0  100.0  100.0
# 2024-01-02  101.0  101.0  101.0
# 2024-01-03    NaN  101.0  105.0  # Next known value
# 2024-01-04    NaN  101.0  105.0  # Still 105.0
# 2024-01-05  105.0  105.0  105.0
# ...
\`\`\`

**When to use:**

✅ **Retrospective analysis:**
\`\`\`python
# Fill gaps with future known values (non-real-time)
historical_data.bfill()
\`\`\`

✅ **Planned events:**
\`\`\`python
# Election results (use final result for missing exit polls)
exit_polls.bfill()
\`\`\`

❌ **Never use for:**
- Real-time prediction (look-ahead bias!)
- Machine learning features (uses future information)

**3. Interpolation**

**How it works:**
Estimates intermediate values based on surrounding data

\`\`\`python
# Linear interpolation
df['interpolate_linear',] = df['price',].interpolate (method='linear')
print(df[['price', 'ffill', 'interpolate_linear',]])
#             price  ffill  interpolate_linear
# 2024-01-01  100.0  100.0             100.0
# 2024-01-02  101.0  101.0             101.0
# 2024-01-03    NaN  101.0             102.0  # (101+105)/3 * 1
# 2024-01-04    NaN  101.0             103.0  # (101+105)/3 * 2
# 2024-01-05  105.0  105.0             105.0
# ...

# Time-aware interpolation
df['interpolate_time',] = df['price',].interpolate (method='time')
# Accounts for actual time distances

# Polynomial interpolation (smoother)
df['interpolate_poly',] = df['price',].interpolate (method='polynomial', order=2)

# Spline interpolation (very smooth)
df['interpolate_spline',] = df['price',].interpolate (method='spline', order=3)
\`\`\`

**When to use:**

✅ **Physical/Natural phenomena:**
\`\`\`python
# Temperature (gradual changes)
temperature.interpolate (method='linear')

# Water levels, air pressure
sensor_readings.interpolate (method='time')
\`\`\`

✅ **Upsampling:**
\`\`\`python
# Monthly to daily (create intermediate values)
monthly_data.resample('D').interpolate (method='linear')
\`\`\`

✅ **Small gaps in smooth data:**
\`\`\`python
# Few missing points in continuous measurement
continuous_data.interpolate (method='cubic')
\`\`\`

❌ **Inappropriate for:**
- Discrete data (counts, events)
- Large gaps (unreliable estimates)
- Non-smooth data (stock prices have jumps)
- Categorical data

**Comparison:**

\`\`\`python
# Visualize differences
dates = pd.date_range('2024-01-01', periods=20, freq='D')
prices = [100, 105, np.nan, np.nan, np.nan, 115, 118, np.nan, 120, 119,
          np.nan, np.nan, 125, 130, np.nan, 128, 126, np.nan, np.nan, 132]
df = pd.DataFrame({'price': prices}, index=dates)

df['ffill',] = df['price',].ffill()
df['bfill',] = df['price',].bfill()
df['interpolate',] = df['price',].interpolate (method='linear')
df['mean',] = df['price',].fillna (df['price',].mean())

print(df[5:15])
#             price  ffill  bfill  interpolate    mean
# 2024-01-06  115.0  115.0  115.0       115.0  115.0
# 2024-01-07  118.0  118.0  118.0       118.0  118.0
# 2024-01-08    NaN  118.0  120.0       119.0  119.5  # See difference
# 2024-01-09  120.0  120.0  120.0       120.0  120.0
# ...
\`\`\`

**Advanced Strategies:**

**1. Rolling Fill (Group-based)**
\`\`\`python
# Fill with rolling average
df['rolling_fill',] = df['price',].fillna(
    df['price',].rolling (window=7, min_periods=1, center=True).mean()
)
\`\`\`

**2. Seasonal Interpolation**
\`\`\`python
# Use same day last week/month/year
df['seasonal_fill',] = df['price',].fillna(
    df['price',].shift (freq='7D')  # Use value from 7 days ago
)
\`\`\`

**3. Model-Based Imputation**
\`\`\`python
from sklearn.linear_model import LinearRegression

# Use other features to predict missing values
X_train = df[df['price',].notna()][['feature1', 'feature2',]]
y_train = df[df['price',].notna()]['price',]

model = LinearRegression()
model.fit(X_train, y_train)

X_missing = df[df['price',].isna()][['feature1', 'feature2',]]
df.loc[df['price',].isna(), 'price_filled',] = model.predict(X_missing)
\`\`\`

**Handling Irregular Timestamps:**

\`\`\`python
# Sensor data with irregular intervals
irregular_data = pd.DataFrame({
    'value': np.random.randn(100)
}, index=pd.to_datetime([
    '2024-01-01 00:00:00',
    '2024-01-01 00:03:15',  # 3m 15s later
    '2024-01-01 00:08:42',  # 5m 27s later
    # ...
]))

# Strategy 1: Resample to regular grid
regular = irregular_data.resample('5min').mean()
regular_filled = regular.interpolate (method='time')

# Strategy 2: Keep irregular, interpolate at query points
def get_value_at (timestamp):
    # Find surrounding values
    before = irregular_data[irregular_data.index <= timestamp].iloc[-1]
    after = irregular_data[irregular_data.index >= timestamp].iloc[0]
    
    # Linear interpolation
    time_ratio = (timestamp - before.name) / (after.name - before.name)
    return before['value',] + time_ratio * (after['value',] - before['value',])
\`\`\`

**Best Practices:**

**1. Understand your data:**
\`\`\`python
# Analyze gap patterns
missing = df['price',].isna()
print(f"Missing: {missing.sum()} / {len (df)} ({missing.sum()/len (df)*100:.1f}%)")

# Gap sizes
gaps = missing.astype (int).groupby((missing != missing.shift()).cumsum()).sum()
print(f"Average gap size: {gaps[gaps > 0].mean():.1f}")
print(f"Max gap size: {gaps.max()}")
\`\`\`

**2. Set maximum gap size:**
\`\`\`python
# Only fill small gaps
def fill_small_gaps (series, max_gap=3, method='linear'):
    """Fill gaps only if ≤ max_gap consecutive missing"""
    filled = series.copy()
    gaps = series.isna().astype (int).groupby((series.notna()).cumsum()).cumsum()
    
    # Only interpolate where gap ≤ max_gap
    mask = (series.isna()) & (gaps <= max_gap)
    filled[mask] = series.interpolate (method=method)[mask]
    
    return filled

df['smart_fill',] = fill_small_gaps (df['price',], max_gap=3)
\`\`\`

**3. Document filling strategy:**
\`\`\`python
# Create indicator of filled values
df['was_filled',] = df['price',].isna()
df['price_filled',] = df['price',].ffill()

# For analysis, you can filter or weight by this
reliable_data = df[~df['was_filled',]]
\`\`\`

**4. Consider domain constraints:**
\`\`\`python
# Prices can't be negative
df['price_filled',] = df['price',].interpolate().clip (lower=0)

# Inventory is integer
df['inventory_filled',] = df['inventory',].interpolate().round().astype (int)

# Percentages are [0, 100]
df['pct_filled',] = df['percent',].interpolate().clip(0, 100)
\`\`\`

**5. Validate filling quality:**
\`\`\`python
# Compare to known values (train/test split)
known = df[df['price',].notna()]
test_idx = known.sample (frac=0.2).index

# Remove test values, fill, compare
df_test = df.copy()
df_test.loc[test_idx, 'price',] = np.nan
df_test['filled',] = df_test['price',].interpolate()

# Measure error
mae = np.abs (df_test.loc[test_idx, 'filled',] - df.loc[test_idx, 'price',]).mean()
print(f"Fill MAE: {mae:.2f}")
\`\`\`

**Decision Framework:**

\`\`\`
Missing data in time series?
│
├─ Real-time system (can't use future)?
│  ├─ Values persist? → Forward fill
│  ├─ Small gaps in smooth data? → Interpolate (carefully)
│  └─ Default → Forward fill (safest)
│
├─ Retrospective analysis (can use future)?
│  ├─ Very small gaps (<3 points)? → Interpolate
│  ├─ Smooth continuous data? → Time-aware interpolate
│  ├─ Discrete/categorical? → Don't fill or use mode
│  └─ Large gaps → Consider if data is usable
│
└─ For prediction/ML features?
   └─ NEVER use backward fill or center interpolation
      (creates look-ahead bias)
\`\`\`

**Key Takeaway:**

- **Forward fill**: Safe for real-time, assumes value persists
- **Backward fill**: Only for retrospective, uses future info
- **Interpolation**: Best for smooth data, small gaps
- **Domain knowledge**: Essential for choosing strategy
- **Validation**: Always check fill quality
- **Documentation**: Mark which values were filled

The right strategy depends on data characteristics, temporal constraints, and use case. When in doubt, forward fill is safest for real-time systems, and time-aware interpolation is best for retrospective analysis of smooth data!`,
    keyPoints: [
      'Forward fill (ffill) propagates last known value - good for prices',
      'Backward fill (bfill) uses next known value - rare use case',
      'Interpolation estimates values using mathematical methods',
      'Time-aware interpolation considers temporal distance between points',
      'Choose method based on data characteristics and domain knowledge',
    ],
  },
  {
    id: 'time-series-pandas-dq-3',
    question:
      'Describe how to build technical indicators (moving averages, RSI, MACD, Bollinger Bands) using Pandas rolling and exponential operations. Discuss the mathematical foundations and common implementation pitfalls.',
    sampleAnswer: `Technical indicators are mathematical calculations based on price, volume, or open interest of a security. Understanding their implementation in Pandas requires knowledge of both the math and potential pitfalls.

**1. Simple Moving Average (SMA)**

**Mathematical foundation:**
\`\`\`
SMA = (P₁ + P₂ + ... + Pₙ) / n

Where P = price, n = window size
\`\`\`

**Implementation:**
\`\`\`python
# Generate sample price data
np.random.seed(42)
dates = pd.date_range('2023-01-01', '2024-12-31', freq='B')
returns = np.random.normal(0.0005, 0.02, len (dates))
df = pd.DataFrame({
    'price': 100 * (1 + returns).cumprod()
}, index=dates)

# Simple Moving Average
df['SMA_20',] = df['price',].rolling (window=20).mean()
df['SMA_50',] = df['price',].rolling (window=50).mean()
df['SMA_200',] = df['price',].rolling (window=200).mean()

# Trading signals
df['golden_cross',] = (df['SMA_50',] > df['SMA_200',]).astype (int).diff()  # 1 when crosses up
df['death_cross',] = (df['SMA_50',] < df['SMA_200',]).astype (int).diff()  # 1 when crosses down

print(df[['price', 'SMA_20', 'SMA_50',]].tail())
\`\`\`

**Pitfall 1: Look-ahead bias**
\`\`\`python
# WRONG: Center=True uses future data
df['SMA_wrong',] = df['price',].rolling(20, center=True).mean()

# CORRECT: Default uses only past
df['SMA_correct',] = df['price',].rolling(20).mean()
\`\`\`

**2. Exponential Moving Average (EMA)**

**Mathematical foundation:**
\`\`\`
EMA_today = Price_today × α + EMA_yesterday × (1 - α)

Where α = 2 / (period + 1)  # Smoothing factor

For 20-period EMA: α = 2 / 21 ≈ 0.095
\`\`\`

**Implementation:**
\`\`\`python
# Pandas EMA
df['EMA_20',] = df['price',].ewm (span=20, adjust=False).mean()
df['EMA_50',] = df['price',].ewm (span=50, adjust=False).mean()

# Manual implementation (for understanding)
def calculate_ema (prices, period):
    alpha = 2 / (period + 1)
    ema = [prices.iloc[0]]  # Start with first price
    
    for price in prices.iloc[1:]:
        ema.append (price * alpha + ema[-1] * (1 - alpha))
    
    return pd.Series (ema, index=prices.index)

df['EMA_20_manual',] = calculate_ema (df['price',], 20)

# Verify they match
print(f"Match: {np.allclose (df['EMA_20',].dropna(), df['EMA_20_manual',].dropna())}")
\`\`\`

**Pitfall 2: adjust parameter**
\`\`\`python
# adjust=False: Standard EMA (weighted recursive)
df['EMA_standard',] = df['price',].ewm (span=20, adjust=False).mean()

# adjust=True: Simple weighted average (not standard EMA)
df['EMA_adjusted',] = df['price',].ewm (span=20, adjust=True).mean()

# They differ, especially at start
print(df[['EMA_standard', 'EMA_adjusted',]].head(25))
# Use adjust=False for standard EMA!
\`\`\`

**3. Relative Strength Index (RSI)**

**Mathematical foundation:**
\`\`\`
1. Calculate price changes: Change = Price_today - Price_yesterday

2. Separate gains and losses:
   Gain = Change if Change > 0, else 0
   Loss = -Change if Change < 0, else 0

3. Calculate average gain and loss (14 periods):
   Avg_Gain = EMA(Gains, 14)
   Avg_Loss = EMA(Losses, 14)

4. Calculate RS and RSI:
   RS = Avg_Gain / Avg_Loss
   RSI = 100 - (100 / (1 + RS))

Range: 0-100 (typically >70 = overbought, <30 = oversold)
\`\`\`

**Implementation:**
\`\`\`python
def calculate_rsi (prices, period=14):
    """Calculate RSI"""
    # Calculate price changes
    delta = prices.diff()
    
    # Separate gains and losses
    gain = delta.where (delta > 0, 0)
    loss = -delta.where (delta < 0, 0)
    
    # Calculate average gains and losses (EMA)
    avg_gain = gain.ewm (alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm (alpha=1/period, adjust=False).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

df['RSI_14',] = calculate_rsi (df['price',], period=14)

# Trading signals
df['RSI_overbought',] = df['RSI_14',] > 70
df['RSI_oversold',] = df['RSI_14',] < 30

print(df[['price', 'RSI_14', 'RSI_overbought', 'RSI_oversold',]].tail())
\`\`\`

**Pitfall 3: Division by zero**
\`\`\`python
# If avg_loss = 0, RS = infinity → RSI = 100
# Handle in calculation:
rs = avg_gain / avg_loss.replace(0, np.nan)  # NaN instead of inf
rsi = 100 - (100 / (1 + rs))
\`\`\`

**4. MACD (Moving Average Convergence Divergence)**

**Mathematical foundation:**
\`\`\`
MACD Line = EMA_12 - EMA_26
Signal Line = EMA_9(MACD Line)
Histogram = MACD Line - Signal Line

Signals:
- MACD crosses above signal = bullish
- MACD crosses below signal = bearish
- Histogram growing = momentum increasing
\`\`\`

**Implementation:**
\`\`\`python
# MACD components
df['EMA_12',] = df['price',].ewm (span=12, adjust=False).mean()
df['EMA_26',] = df['price',].ewm (span=26, adjust=False).mean()
df['MACD',] = df['EMA_12',] - df['EMA_26',]
df['MACD_signal',] = df['MACD',].ewm (span=9, adjust=False).mean()
df['MACD_histogram',] = df['MACD',] - df['MACD_signal',]

# Trading signals
df['MACD_bullish',] = (df['MACD',] > df['MACD_signal',]).astype (int).diff() == 1
df['MACD_bearish',] = (df['MACD',] < df['MACD_signal',]).astype (int).diff() == 1

print(df[['price', 'MACD', 'MACD_signal', 'MACD_histogram',]].tail())

# Visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax1.plot (df.index, df['price',], label='Price')
ax1.plot (df.index, df['EMA_12',], label='EMA 12', alpha=0.7)
ax1.plot (df.index, df['EMA_26',], label='EMA 26', alpha=0.7)
ax1.legend()
ax1.set_title('Price and EMAs')

ax2.plot (df.index, df['MACD',], label='MACD', linewidth=2)
ax2.plot (df.index, df['MACD_signal',], label='Signal', linewidth=2)
ax2.bar (df.index, df['MACD_histogram',], label='Histogram', alpha=0.3)
ax2.axhline (y=0, color='black', linestyle='--', alpha=0.3)
ax2.legend()
ax2.set_title('MACD')

plt.tight_layout()
plt.show()
\`\`\`

**5. Bollinger Bands**

**Mathematical foundation:**
\`\`\`
Middle Band = SMA_20
Upper Band = Middle Band + (2 × STD_20)
Lower Band = Middle Band - (2 × STD_20)

Interpretation:
- Price touching upper band = potentially overbought
- Price touching lower band = potentially oversold
- Bands squeeze = low volatility (potential breakout coming)
- Bands widen = high volatility
\`\`\`

**Implementation:**
\`\`\`python
def calculate_bollinger_bands (prices, period=20, num_std=2):
    """Calculate Bollinger Bands"""
    middle = prices.rolling (period).mean()
    std = prices.rolling (period).std()
    
    upper = middle + (num_std * std)
    lower = middle - (num_std * std)
    
    return middle, upper, lower

df['BB_middle',], df['BB_upper',], df['BB_lower',] = calculate_bollinger_bands (df['price',])

# Additional indicators
df['BB_width',] = (df['BB_upper',] - df['BB_lower',]) / df['BB_middle',]  # Volatility
df['BB_position',] = (df['price',] - df['BB_lower',]) / (df['BB_upper',] - df['BB_lower',])  # 0-1

# Signals
df['BB_squeeze',] = df['BB_width',] < df['BB_width',].rolling(50).quantile(0.25)  # Low volatility
df['above_upper',] = df['price',] > df['BB_upper',]
df['below_lower',] = df['price',] < df['BB_lower',]

print(df[['price', 'BB_upper', 'BB_middle', 'BB_lower', 'BB_width',]].tail())

# Visualization
plt.figure (figsize=(12, 6))
plt.plot (df.index, df['price',], label='Price', linewidth=2)
plt.plot (df.index, df['BB_middle',], label='Middle Band', linestyle='--')
plt.fill_between (df.index, df['BB_upper',], df['BB_lower',], alpha=0.2, label='Bollinger Bands')
plt.legend()
plt.title('Bollinger Bands')
plt.show()
\`\`\`

**Pitfall 4: Insufficient data**
\`\`\`python
# Need enough data for accurate calculation
print(f"First valid MACD: {df['MACD',].first_valid_index()}")
# Needs at least 26 periods for EMA_26, then 9 more for signal
# Total: 35 periods minimum

# Use min_periods to get earlier (but less reliable) values
df['SMA_flexible',] = df['price',].rolling(20, min_periods=10).mean()
\`\`\`

**6. Complete Trading System Example**

\`\`\`python
def add_all_indicators (df):
    """Add comprehensive technical indicators"""
    # Moving Averages
    df['SMA_20',] = df['price',].rolling(20).mean()
    df['SMA_50',] = df['price',].rolling(50).mean()
    df['SMA_200',] = df['price',].rolling(200).mean()
    df['EMA_12',] = df['price',].ewm (span=12, adjust=False).mean()
    df['EMA_26',] = df['price',].ewm (span=26, adjust=False).mean()
    
    # RSI
    df['RSI',] = calculate_rsi (df['price',], 14)
    
    # MACD
    df['MACD',] = df['EMA_12',] - df['EMA_26',]
    df['MACD_signal',] = df['MACD',].ewm (span=9, adjust=False).mean()
    df['MACD_hist',] = df['MACD',] - df['MACD_signal',]
    
    # Bollinger Bands
    df['BB_mid',], df['BB_upper',], df['BB_lower',] = calculate_bollinger_bands (df['price',])
    df['BB_width',] = (df['BB_upper',] - df['BB_lower',]) / df['BB_mid',]
    
    # Volatility
    df['returns',] = df['price',].pct_change()
    df['volatility',] = df['returns',].rolling(20).std() * np.sqrt(252)
    
    return df

# Apply indicators
df = add_all_indicators (df)

# Generate trading signals
def generate_signals (df):
    """Generate composite trading signals"""
    signals = pd.DataFrame (index=df.index)
    
    # Trend signals
    signals['trend',] = (df['SMA_20',] > df['SMA_50',]).astype (int)
    
    # Momentum signals
    signals['momentum',] = ((df['RSI',] > 50) & (df['MACD',] > df['MACD_signal',])).astype (int)
    
    # Volatility signals
    signals['low_vol',] = (df['BB_width',] < df['BB_width',].quantile(0.25)).astype (int)
    
    # Combined signal
    signals['composite',] = (signals['trend',] + signals['momentum',]).clip(0, 2)
    # 0 = bearish, 1 = neutral, 2 = bullish
    
    return signals

signals = generate_signals (df)
print(signals.tail())
\`\`\`

**Common Pitfalls Summary:**1. **Look-ahead bias**: Never use \`center=True\` or future data
2. **Insufficient data**: Indicators need minimum periods
3. **adjust parameter**: Use \`adjust=False\` for standard EMA
4. **Division by zero**: Handle zero denominators
5. **Data alignment**: Ensure all indicators use same index
6. **Overfitting**: Don't optimize on same data you test on
7. **Transaction costs**: Paper trading profits disappear with real costs

**Best Practices:**1. **Vectorize calculations**:
\`\`\`python
# Good: Vectorized
df['signal',] = (df['SMA_20',] > df['SMA_50',]).astype (int)

# Bad: Loop
for i in range (len (df)):
    df.loc[df.index[i], 'signal',] = 1 if df.iloc[i]['SMA_20',] > df.iloc[i]['SMA_50',] else 0
\`\`\`

2. **Handle NaN properly**:
\`\`\`python
# Drop rows where any indicator is NaN
df_clean = df.dropna()

# Or: Only use after warmup period
warmup = 200  # Longest indicator period
df_ready = df.iloc[warmup:]
\`\`\`

3. **Validate indicators**:
\`\`\`python
# Compare with established library
import ta

df['RSI_ta',] = ta.momentum.RSIIndicator (df['price',], window=14).rsi()
df['RSI_custom',] = calculate_rsi (df['price',], 14)

# Should be nearly identical
assert np.allclose (df['RSI_ta',].dropna(), df['RSI_custom',].dropna(), rtol=0.01)
\`\`\`

4. **Document parameters**:
\`\`\`python
# Clear parameter documentation
PARAMS = {
    'SMA_short': 20,
    'SMA_long': 50,
    'RSI_period': 14,
    'RSI_overbought': 70,
    'RSI_oversold': 30,
    'BB_period': 20,
    'BB_std': 2
}

df['SMA_short',] = df['price',].rolling(PARAMS['SMA_short',]).mean()
\`\`\`

**Key Takeaway:**

Technical indicators are powerful tools but require careful implementation:
- Understand the math to avoid errors
- Watch for look-ahead bias (deadly for backtesting)
- Handle edge cases (NaN, division by zero)
- Validate against known implementations
- Remember: Indicators describe past, don't predict future!`,
    keyPoints: [
      'Datetime components: year, month, day, hour, dayofweek via .dt accessor',
      'Lag features shift values back in time for autoregressive patterns',
      'Rolling statistics capture trends and volatility over windows',
      'Cyclical encoding (sin/cos) preserves periodicity for ML models',
      'Feature engineering from dates often more impactful than algorithm choice',
    ],
  },
];
