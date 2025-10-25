export const timeSeriesFundamentalsQuiz = [
  {
    id: 'tsf-q-1',
    question:
      'You have 5 years of daily stock price data and want to build a predictive model. Explain the complete process of testing for stationarity and applying appropriate transformations. Include: (1) which statistical tests to use and why, (2) how to interpret the results, (3) what transformations to apply for non-stationary series, (4) how to validate that transformations worked. Why is stationarity critical for financial modeling?',
    sampleAnswer:
      'Complete stationarity analysis: (1) Visual inspection: Plot the series, rolling mean, and rolling std. Non-stationary series show trending mean or changing variance. (2) Statistical tests: Run both ADF (null: non-stationary) and KPSS (null: stationary). If ADF p-value > 0.05 and KPSS p-value < 0.05 → non-stationary. Use both tests because they test opposite hypotheses. (3) ACF analysis: Non-stationary series show slowly decaying ACF with high values at all lags. (4) Transformations: For prices, apply log returns: r_t = log(P_t/P_{t-1}). This achieves two goals: removes trend (differencing) and stabilizes variance (log). Alternative: simple returns r_t = (P_t - P_{t-1})/P_{t-1}. (5) Validation: Re-run ADF and KPSS on transformed series. Expect: ADF p-value < 0.05 (reject non-stationarity) and KPSS p-value > 0.05 (fail to reject stationarity). Check rolling statistics are stable. (6) Why critical: Non-stationary series have time-varying parameters. A model trained on 2020 data (low volatility) will fail on 2022 data (high volatility). Stationarity ensures statistical properties remain constant, making future predictions valid. Example: SPY prices have ADF p-value = 0.65 (non-stationary), but log returns have p-value = 0.0001 (stationary). Always model returns, not prices.',
    keyPoints: [
      'Use both ADF (null: non-stationary) and KPSS (null: stationary) tests',
      'Log returns achieve two goals: remove trend and stabilize variance',
      'Non-stationary parameters change over time, invalidating predictions',
      'ACF of non-stationary series decays slowly with high persistence',
      'Always validate transformations by re-testing stationarity',
    ],
  },
  {
    id: 'tsf-q-2',
    question:
      'Financial returns exhibit several stylized facts: fat tails, volatility clustering, leverage effect, and minimal autocorrelation. For each: (1) explain what it means, (2) how to test for it statistically, (3) implications for modeling and trading. Why do these characteristics make financial forecasting challenging?',
    sampleAnswer:
      "Stylized facts analysis: (1) Fat tails: Returns have higher probability of extreme events than normal distribution predicts. Test: Calculate kurtosis (normal = 3, fat tails > 3). Measure: scipy.stats.kurtosis (returns). SPY returns have kurtosis ~ 6-8. Implication: Value-at-Risk models using normal distribution underestimate tail risk. Use Student-t or empirical distributions instead. Trading: Size positions smaller to account for larger-than-expected drawdowns. (2) Volatility clustering: High volatility follows high volatility. Test: Calculate ACF of squared returns. If ACF(1) > 0.1, clustering present. Plot: |returns| over time shows visible clusters. Implication: Need GARCH-type models to forecast volatility. Trading: Reduce position sizes during high volatility periods, increase during calm periods. Volatility mean-reverts. (3) Leverage effect: Negative returns increase future volatility more than positive returns. Test: Correlation between lagged returns and forward volatility (expect negative). Implication: Asymmetric GARCH models (GJR-GARCH, EGARCH). Trading: Expect volatility spikes after selloffs, position for mean reversion. (4) Minimal autocorrelation: Today\'s return doesn't predict tomorrow's. Test: ACF of returns, Ljung-Box test. Implication: Weak-form market efficiency. Simple momentum strategies have low edge. Need sophisticated features (orderflow, sentiment, alternative data). Challenge: These facts contradict classical models. Normal distribution + constant volatility + autocorrelation assumptions all fail. Must use: Student-t distributions, GARCH volatility, non-linear models.",
    keyPoints: [
      'Fat tails: kurtosis > 3, extreme events more likely than normal distribution',
      'Volatility clustering: use ACF of squared returns, GARCH models needed',
      'Leverage effect: negative returns increase volatility asymmetrically',
      'Minimal autocorrelation: returns appear random, weak-form efficiency',
      'Classical models fail: need robust distributions and volatility models',
    ],
  },
  {
    id: 'tsf-q-3',
    question:
      'Design a time series preprocessing pipeline for algorithmic trading. Given raw OHLCV data, what steps would you take to prepare features for modeling? Address: (1) handling missing data and gaps (weekends, holidays), (2) outlier detection and treatment, (3) determining appropriate transformation (returns, log returns, differences), (4) checking for lookahead bias, (5) creating rolling windows for cross-validation. Include code structure and reasoning.',
    sampleAnswer:
      "Preprocessing pipeline: (1) Data loading and validation: Load OHLCV data, check for duplicates (df.duplicated()), verify timestamp ordering, ensure no future timestamps. (2) Missing data: Financial markets closed weekends/holidays. Options: (a) Forward fill (ffill) for short gaps (1-2 days), (b) Remove long gaps (> 7 days), (c) For intraday: interpolate within trading hours only, never across days. Code: df = df.resample('1D').ffill (limit=2). Don't use mean/median imputation—introduces unrealistic values. (3) Outlier detection: Calculate z-scores of returns: z = (r - mean) / std. Flag if |z| > 5 (likely data error). Flash crashes (real outliers): Keep but robust winsorization (cap at 99th percentile). Code: returns.clip (lower=returns.quantile(0.01), upper=returns.quantile(0.99)). (4) Transformation: Always use returns for modeling: Simple returns: (P_t - P_{t-1}) / P_{t-1}, Log returns: log(P_t / P_{t-1}). Log returns are additive (easier math) and symmetric. Validate stationarity with ADF test. (5) Feature engineering: Create lags without lookahead: features at time t must use data only from t-1 and before. Code: df['return_lag1',] = df['return',].shift(1). Roll window features: df['ma_20',] = df['close',].rolling(20).mean(). Never use .rolling().mean() without shift! (6) Cross-validation: Time-based splits, never random. Walk-forward: train on [t-252:t], test on [t:t+21], slide by 21 days. Code: for i in range(252, len (df), 21): train = df[i-252:i], test = df[i:i+21]. (7) Scaling: Fit scaler only on train set, transform train and test. Code: scaler.fit(X_train), X_train_scaled = scaler.transform(X_train), X_test_scaled = scaler.transform(X_test). Critical: No lookahead bias. Each feature at time t must be calculable with data available at t.",
    keyPoints: [
      'Forward fill short gaps (1-2 days), remove long gaps, never interpolate across market close',
      'Log returns preferred: additive, symmetric, validate stationarity with ADF',
      'Winsorize outliers at 1st/99th percentile, keep flash crashes (real events)',
      'Feature engineering: shift all features by 1 period, no lookahead bias',
      'Walk-forward validation: train on past, test on future, never random split',
    ],
  },
];
