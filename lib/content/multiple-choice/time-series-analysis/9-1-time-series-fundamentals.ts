export const timeSeriesFundamentalsMultipleChoice = [
    {
        id: 1,
        question:
            'A quantitative analyst calculates that a trading strategy had the following monthly returns: Month 1: +5%, Month 2: +5%, Month 3: +5%. They report the total return as 15% (5% + 5% + 5%). What is the ACTUAL total return, and what is the error introduced by their incorrect calculation?',
        options: [
            'The calculation is correct: 15.0% total return',
            'Actual return is 15.76%; error of 0.76 percentage points due to compounding',
            'Actual return is 14.25%; error of 0.75 percentage points due to time-decay',
            'Actual return is 16.2%; error of 1.2 percentage points due to geometric mean',
            'Cannot calculate without knowing the initial capital',
        ],
        correctAnswer: 1,
        explanation:
            "Simple returns CANNOT be added over time due to compounding. Correct calculation: (1.05 × 1.05 × 1.05) - 1 = 1.157625 - 1 = 15.76%. The analyst's method (summing) gives 15%, understating returns by 0.76 percentage points. This error grows dramatically over longer periods. For example, 12 months of 5% returns: Incorrect sum = 60%, Correct compound = 79.6% (19.6 point error!). This is why quantitative finance uses log returns (which ARE additive over time) internally and converts to simple returns only for reporting.",
        difficulty: 'intermediate',
    },
    {
        id: 2,
        question:
            'You are backtesting a trading strategy and need to calculate rolling 30-day volatility for 500 stocks every day for 5 years (1,250 trading days). Which data structure and approach is MOST appropriate for this operation?',
        options: [
            'Python lists with nested loops to calculate each volatility value individually',
            'Pandas DataFrame with DatetimeIndex, using .rolling() method with vectorized operations',
            'Dictionary of dictionaries storing each stock-date-volatility combination',
            'SQL database with SELECT queries for each 30-day window',
            'NumPy 3D array (stocks × days × prices) with manual window slicing',
        ],
        correctAnswer: 1,
        explanation:
            "Pandas DataFrame with DatetimeIndex and .rolling() is optimal for this use case because: (1) DatetimeIndex enables automatic date alignment and resampling, (2) .rolling() provides optimized sliding window calculations, (3) Vectorized operations are orders of magnitude faster than loops (500 stocks × 1,250 days = 625,000 calculations), (4) Built-in handling of missing data and irregular dates, (5) Memory efficient with view-based operations. Example: `volatilities = returns_df.rolling(window=30).std()` calculates all volatilities in milliseconds. Lists would take minutes, SQL queries would be inefficient for computation-heavy operations, and NumPy 3D arrays lose the datetime functionality critical for financial time series.",
        difficulty: 'intermediate',
    },
    {
        id: 3,
        question:
            'A time series of daily stock returns shows: Mean = 0.05%, Std Dev = 2.0%, Skewness = -0.8, Kurtosis = 12.0. What does this tell you about the distribution, and what are the implications for risk management?',
        options: [
            'Returns are normally distributed with slight negative bias; standard risk metrics are appropriate',
            'Left-skewed with fat tails: large negative returns occur more often than normal distribution predicts; VaR underestimates risk',
            'Right-skewed with thin tails: positive returns dominate; can use more leverage safely',
            'The high kurtosis (12) exactly offsets the negative skew (-0.8), resulting in normal distribution',
            'The negative skew indicates the strategy is losing money consistently',
        ],
        correctAnswer: 1,
        explanation:
            "This distribution is left-skewed (skew = -0.8 < 0) with extremely fat tails (kurtosis = 12 >> 3). Interpretation: (1) Negative skew means left tail is longer - large LOSSES occur more frequently than large gains ('blow-up risk'), (2) Kurtosis of 12 (excess kurtosis = 9) means extreme events happen 3-4x more often than normal distribution, (3) Risk implications: Parametric VaR/CVaR based on normal distribution will severely UNDERESTIMATE tail risk. For example, a '1-in-100 day' loss might actually occur 1-in-20 days. (4) This distribution is typical of short-volatility strategies (like selling options) that collect small premiums but occasionally suffer large losses. Risk management must use: historical simulation VaR, stress testing, and tail-risk hedging. The mean of 0.05% is irrelevant for assessing distribution shape.",
        difficulty: 'advanced',
    },
    {
        id: 4,
        question:
            'When constructing 1-minute OHLCV bars from irregular tick data, a stock trades at $100.00 at 9:30:15 AM, $100.50 at 9:30:45 AM, and $99.75 at 9:31:10 AM. What should the 9:30-9:31 AM bar show as the Close price?',
        options: [
            '$100.50 (the highest price during the period)',
            '$100.00 (the first price, since it opened the bar)',
            '$100.08 (the volume-weighted average price)',
            '$100.50 (the last price BEFORE 9:31:00 AM)',
            '$99.75 (the most recent price, even though after 9:31)',
        ],
        correctAnswer: 3,
        explanation:
            "Close price for a time-based bar (9:30-9:31 AM) should be the LAST price that occurred BEFORE the bar boundary (9:31:00). The trade at $100.50 (9:30:45 AM) is the last one within the 9:30-9:31 window, so Close = $100.50. The $99.75 trade at 9:31:10 AM belongs to the NEXT bar (9:31-9:32 AM) and must not be included - doing so would create look-ahead bias. Common mistakes: (1) Using highest price (that's High, not Close), (2) Using VWAP (that's a derived metric, not OHLC), (3) Including trades from next period (look-ahead bias). Proper bar construction is critical for backtesting validity. For bars with NO trades (illiquid stocks), best practice is to forward-fill from previous close or mark as NaN rather than inventing a price.",
        difficulty: 'intermediate',
    },
    {
        id: 5,
        question:
            'You discover your time series has these issues: (1) 5% of dates are missing (market holidays), (2) One suspicious price spike (stock went from $50 to $5000 for one minute), (3) Returns show autocorrelation of 0.85, (4) Timestamps are in different timezones (EST and UTC mixed). What is the CORRECT order to address these issues?',
        options: [
            'Fill missing dates → Fix timezone → Remove spike → Ignore autocorrelation (it is what it is)',
            'Fix timezone → Remove spike → Fill missing dates → Investigate autocorrelation cause',
            'Remove spike → Fill missing dates → Fix timezone → Ignore autocorrelation',
            'Fix timezone → Fill missing dates → Remove spike → Transform to log returns to reduce autocorrelation',
            'Remove spike → Ignore missing dates → Fix timezone → Autocorrelation is not an issue',
        ],
        correctAnswer: 1,
        explanation:
            "Correct order: (1) FIX TIMEZONE FIRST - Mixed timezones create wrong temporal ordering. If EST and UTC are mixed, events appear out of sequence. Must convert all to common timezone (usually UTC) before any analysis. (2) REMOVE SPIKE - The $50→$5000 spike is data error (100x move). Must clean bad data before imputation or modeling. Check exchange records to find true price or mark as missing. (3) FILL MISSING DATES - After cleaning, address missing data. For market holidays, usually forward-fill previous close (price doesn't change when market closed). (4) INVESTIGATE AUTOCORRELATION - 0.85 autocorrelation is EXTREMELY high and indicates: bad data (stale prices), wrong calculation (using overlapping windows), or fundamental issue (illiquid stock with delayed price discovery). This is not normal and cannot be 'fixed' by log transform. Why this order? Each step depends on previous being correct. Can't properly detect spikes if timestamps are wrong. Can't impute missing data if bad data still present. Can't analyze autocorrelation until clean, aligned data exists.",
        difficulty: 'advanced',
    },
];

