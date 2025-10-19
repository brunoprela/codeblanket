/**
 * Discussion Questions for Time Series Forecasting
 */

import { QuizQuestion } from '../../../types';

export const timeseriesforecastingQuiz: QuizQuestion[] = [
  {
    id: 'timeseries-forecasting-q1',
    question:
      'Explain what stationarity means in time series and why it is important for classical forecasting models. How can you make a non-stationary series stationary?',
    hint: 'Think about constant statistical properties and what models assume.',
    sampleAnswer:
      'Stationarity means a time series has constant statistical properties (mean, variance, autocorrelation) over time. No trend, no changing variance, no seasonality. Important because classical models (ARIMA, exponential smoothing) assume stationarity - their parameters are estimated assuming consistent patterns. Non-stationary series violates this, leading to unreliable forecasts. Test stationarity: Augmented Dickey-Fuller (ADF) test - null hypothesis is non-stationary, reject if p<0.05. Visual: plot rolling mean/std, should be flat. Making stationary: (1) Differencing: Y_t - Y_{t-1} removes trend; seasonal differencing Y_t - Y_{t-s} removes seasonality; (2) Log transform: stabilizes variance; (3) Detrending: subtract trend component. First-order differencing usually sufficient; if not, apply second-order. Check stationarity after each transformation. Once stationary, fit ARIMA. After forecasting, reverse transformations to get actual predictions (integrate back if differenced).',
    keyPoints: [
      'Stationarity: constant mean, variance, autocorrelation',
      'Classical models assume stationarity',
      'ADF test: p<0.05 → stationary',
      'Differencing removes trend/seasonality',
      'Log transform stabilizes variance',
    ],
  },
  {
    id: 'timeseries-forecasting-q2',
    question:
      'Compare ARIMA and Exponential Smoothing methods. When would you choose each? What are their strengths and limitations?',
    hint: 'Consider model complexity, interpretability, and data requirements.',
    sampleAnswer:
      'Exponential Smoothing (ES): Weighted averages with exponentially decaying weights. Simple ES: no trend/seasonality. Holt: adds trend. Holt-Winters: adds seasonality. Strengths: Simple, intuitive, fast, requires less data, automatic parameter optimization. Limitations: Less flexible, assumes specific patterns (additive/multiplicative), harder to incorporate external variables. ARIMA: Combines autoregression (past values) + moving average (past errors). ARIMA(p,d,q) with differencing d. Strengths: More flexible, handles complex patterns, theoretical foundation, incorporates external variables (ARIMAX), seasonal extensions (SARIMA). Limitations: Requires more data, needs stationarity, parameter selection tricky, computationally heavier. When to choose: Use ES for quick forecasts, simple patterns, limited data, need speed. Use ARIMA for complex patterns, sufficient data, need flexibility, incorporate external variables. In practice: try both, compare performance. ES often sufficient for business forecasting; ARIMA when accuracy critical. Modern: combine with ML/deep learning for best results.',
    keyPoints: [
      'ES: weighted averages, simple, fast, less data',
      'ARIMA: AR + MA, flexible, needs stationarity',
      'ES for simple patterns and speed',
      'ARIMA for complex patterns and accuracy',
      'Try both, validate performance',
    ],
  },
  {
    id: 'timeseries-forecasting-q3',
    question:
      'Why is the train/test split for time series different from standard ML? What are the key considerations for evaluating time series models?',
    hint: 'Think about temporal order and data leakage.',
    sampleAnswer:
      'Standard ML: random shuffle train/test OK because samples independent. Time series: samples NOT independent - temporal order matters! Data leakage: if we randomly split, training data could include future values relative to test data - model learns from future, unrealistic. Correct approach: time-based split - training = earlier data, test = later data. Never shuffle. Example: 80% training (Jan-Oct), 20% test (Nov-Dec). Cross-validation: use "rolling window" or "time series split" - train on [1:100], test [101:110], train [1:110], test [111:120], etc. Always respects temporal order. Evaluation considerations: (1) Multiple horizons - evaluate 1-step, 7-step, 30-step forecasts separately; (2) Walk-forward validation - retrain at each step using all past data; (3) Metrics - RMSE, MAE, MAPE commonly used; (4) Business context - accuracy at specific horizons (e.g., monthly forecasts); (5) Forecast intervals - provide uncertainty. Key: respect temporal causality - past predicts future, not vice versa!',
    keyPoints: [
      'Must respect temporal order (no random shuffle)',
      'Time-based split: train earlier, test later',
      'Rolling window cross-validation',
      'Evaluate multiple forecast horizons',
      'Walk-forward validation for realistic performance',
    ],
  },
];
