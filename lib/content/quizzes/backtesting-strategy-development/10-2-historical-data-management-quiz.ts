export const historicalDataManagementQuiz = [
  {
    id: 1,
    question:
      'A stock trading at $200 per share undergoes a 2-for-1 stock split. You have historical data showing it traded at $400 before the split. What should the adjusted historical price be?',
    options: [
      '$400 (no adjustment needed - that was the actual price)',
      '$200 (adjust down by the split ratio)',
      '$800 (adjust up by the split ratio)',
      '$300 (adjust partially)',
      'Depends on whether you owned the stock or not',
    ],
    correctAnswer: 1,
    explanation:
      "Historical prices should be adjusted BACKWARDS to maintain consistent price scale. A 2-for-1 split doubles shares and halves price. To keep charts meaningful, divide all pre-split prices by 2. So $400 becomes $200 (split-adjusted). This ensures: 1) Charts don't show artificial 50% drops, 2) Returns calculations are accurate, 3) Indicators work correctly. **Without adjustment**: Chart shows crash from $400 to $200. **With adjustment**: Chart shows smooth continuity. Volume should also be multiplied by 2 for pre-split dates. This is why financial websites show 'adjusted close' - it includes all splits and dividends.",
    difficulty: 'easy',
  },
  {
    id: 2,
    question:
      'Your backtest of a long-only equity strategy on S&P 500 stocks from 2010-2023 shows 18% annual returns. You used the current 500 constituents (as of 2024). What is the MOST likely problem?',
    options: [
      'The strategy is overfit to recent market conditions',
      'Transaction costs were not included',
      'Survivorship bias - the dataset excludes delisted stocks that would have caused losses',
      'The backtest period is too long',
      'Nothing is wrong - this is a proper backtest',
    ],
    correctAnswer: 2,
    explanation:
      "This is classic **survivorship bias**. Using current S&P 500 constituents means you're only testing on stocks that survived and succeeded. This excludes:\n- Companies that went bankrupt (Lehman Brothers 2008, Enron 2001)\n- Companies removed for poor performance\n- Companies acquired/delisted\n\n**Impact**: Studies show survivorship bias inflates returns by 30-60% for long-only strategies. Your 18% might actually be 9-12% with proper data.\n\n**Solution**: Use point-in-time constituent data. On Jan 1 2010, use the S&P 500 members as of that date (including future bankruptcies). Data sources: Bloomberg Terminal, Norgate Data, or Compustat Point-in-Time database.\n\n**Real example**: A 'winning' strategy tested on current stocks might have actually lost money by holding Lehman Brothers through 2008.",
    difficulty: 'intermediate',
  },
  {
    id: 3,
    question:
      'You notice your backtested strategy generated a 35% return on a single trade: buying AAPL at $100 and selling at $135 one month later. However, your data shows no major news during that period. What should you investigate first?',
    options: [
      'Whether your strategy correctly identified a market inefficiency',
      'Whether there was a stock split that was not properly adjusted in your data',
      'Whether this was during a general market rally',
      'Whether the volumes were sufficient for execution',
      'Nothing - 35% monthly returns are possible',
    ],
    correctAnswer: 1,
    explanation:
      "A 35% move in one month with no news is a red flag for a **data quality issue**, most likely an unadjusted stock split. Here's what probably happened:\n\n**Scenario**: AAPL did a 7-for-1 split during that month.\n- Pre-split price: $700\n- Post-split price: $100 (same economic value)\n- Unadjusted data shows: $700 → $100 = -86% crash\n- Or if reversed: $100 → $700 = +600% gain\n\n**Your case**: Data might show $100 → $135, but this could be:\n1. Missing split adjustment\n2. Wrong split ratio applied\n3. Data vendor error\n\n**How to verify**:\n```python\n# Check for split\nif daily_return > 0.3:  # 30%+ move\n    print('Possible data error or split')\n    verify_corporate_actions(ticker, date)\n```\n\n**Real impact**: Many backtest failures are due to missing corporate action adjustments. Always validate unusual returns against corporate actions and news.",
    difficulty: 'advanced',
  },
  {
    id: 4,
    question:
      'What is the primary benefit of storing historical market data in Parquet format on S3 compared to CSV files?',
    options: [
      'Parquet files are easier to read and edit manually',
      'Parquet provides better compression and faster query performance due to columnar storage',
      'CSV files cannot store stock data accurately',
      'Parquet files are compatible with more software',
      'There is no significant benefit - CSV is just as good',
    ],
    correctAnswer: 1,
    explanation:
      "**Parquet format** offers major advantages for financial data:\n\n**1. Columnar Storage**\n- Stores data by column, not row\n- Query only needed columns (e.g., just 'Close' prices)\n- Much faster for analysis\n\n**2. Compression**\n- Typical compression: 5-10x vs CSV\n- 5GB CSV → 0.5-1GB Parquet\n- Lower storage costs\n\n**3. Performance**\n- 10-100x faster queries\n- Predicate pushdown (filter before loading)\n- Efficient for time-series data\n\n**Example**:\n```python\n# CSV: Load entire file, filter in memory\ndf = pd.read_csv('aapl_10_years.csv')  # 1GB, 30 seconds\ndf = df[df['date'] > '2023-01-01']\n\n# Parquet: Filter during read\ndf = pd.read_parquet(\n    'aapl_10_years.parquet',\n    filters=[('date', '>', '2023-01-01')]  # 100MB, 2 seconds\n)\n```\n\n**Cost**: S3 storage at $0.023/GB/month: 5GB CSV = $0.12/month, 0.5GB Parquet = $0.01/month + faster queries.\n\n**Production standard**: Use Parquet (or similar columnar formats like ORC) for financial data.",
    difficulty: 'intermediate',
  },
  {
    id: 5,
    question:
      'Your data validation pipeline flags 15 trading days with zero volume for a stock. What is the MOST appropriate action?',
    options: [
      'Delete these rows from your dataset - they are obviously errors',
      'Fill zero volume with the average volume from surrounding days',
      'Investigate the cause (trading halt, holidays, data error) before deciding on treatment',
      'Ignore it - volume data is not important for backtesting',
      'Replace zero with 1 to avoid division errors',
    ],
    correctAnswer: 2,
    explanation:
      'Zero volume days can have **multiple causes**, each requiring different treatment:\n\n**Legitimate Reasons**:\n1. **Trading Halts**: Stock halted due to news pending, circuit breaker, or regulatory action\n   → Keep the data, mark as halted, adjust strategy logic\n\n2. **Exchange Holidays**: Market closed (Christmas, Thanksgiving, etc.)\n   → Remove these dates from dataset\n\n3. **Illiquid Stocks**: Small-cap stocks may have no trades on some days\n   → Keep data, but note low liquidity in analysis\n\n4. **IPO/Delisting Dates**: First/last trading days\n   → Keep for historical accuracy\n\n**Data Errors**:\n5. **Data Feed Issues**: Vendor data missing\n   → Fill from alternative source or interpolate\n\n**Investigation Process**:\n```python\ndef investigate_zero_volume(ticker, dates):\n    for date in dates:\n        # Check if market was open\n        if not is_trading_day(date):\n            continue  # Expected\n        \n        # Check for halt\n        if check_trading_halt(ticker, date):\n            mark_as_halted(ticker, date)\n        \n        # Check alternative data sources\n        alt_volume = check_alternative_source(ticker, date)\n        if alt_volume > 0:\n            # Data feed error - use alternative\n            update_volume(ticker, date, alt_volume)\n```\n\n**Never automatically fill/delete** without understanding why. Wrong treatment can create false signals in your backtest.',
    difficulty: 'advanced',
  },
];
