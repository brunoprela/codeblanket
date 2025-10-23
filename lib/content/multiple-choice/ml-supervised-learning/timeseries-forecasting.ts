/**
 * Multiple Choice Questions for Time Series Forecasting
 */

import { MultipleChoiceQuestion } from '../../../types';

export const timeseriesforecastingMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'timeseries-forecasting-mc-1',
    question: 'What does the "d" parameter represent in ARIMA(p, d, q)?',
    options: [
      'The number of autoregressive terms',
      'The order of differencing to make the series stationary',
      'The number of moving average terms',
      'The seasonal period',
    ],
    correctAnswer: 1,
    explanation:
      'In ARIMA(p,d,q), d represents the order of differencing. d=1 means first-order differencing (Y_t - Y_{t-1}), d=2 means second-order differencing. Differencing is used to make a non-stationary series stationary by removing trend. p is autoregressive order, q is moving average order.',
  },
  {
    id: 'timeseries-forecasting-mc-2',
    question:
      'You split a time series randomly into train/test sets and achieve excellent test performance. What is likely wrong?',
    options: [
      'Nothing is wrong',
      'Data leakage - future data in training set, violating temporal order',
      'The model is overfitting',
      'Feature scaling was not applied',
    ],
    correctAnswer: 1,
    explanation:
      'Random splitting violates temporal causality in time series. Training data could contain future values relative to test data, causing data leakage. The model unrealistically learns from the future. Always use time-based splits: train on earlier data, test on later data.',
  },
  {
    id: 'timeseries-forecasting-mc-3',
    question:
      'Which method is most appropriate for a time series with both trend and seasonality?',
    options: [
      'Simple Exponential Smoothing',
      "Holt's Linear Trend",
      'Holt-Winters (Triple Exponential Smoothing)',
      'Simple Moving Average',
    ],
    correctAnswer: 2,
    explanation:
      'Holt-Winters (Triple Exponential Smoothing) handles both trend and seasonality. Simple ES handles neither, Holt handles only trend, and Simple MA is just an average. Holt-Winters has three smoothing parameters: level (alpha), trend (beta), and seasonal (gamma).',
  },
  {
    id: 'timeseries-forecasting-mc-4',
    question:
      'What does a significant spike at lag 1 in the PACF plot suggest?',
    options: [
      'The series needs seasonal differencing',
      'An AR(1) model might be appropriate',
      'The series is non-stationary',
      'There is no autocorrelation',
    ],
    correctAnswer: 1,
    explanation:
      'A significant spike at lag 1 in the Partial Autocorrelation Function (PACF) suggests that Y_t is directly correlated with Y_{t-1} after controlling for intermediate lags. This indicates an AR(1) model (autoregressive order p=1) might be appropriate. PACF helps determine the AR order (p) in ARIMA.',
  },
  {
    id: 'timeseries-forecasting-mc-5',
    question:
      'You apply first-order differencing to a time series and the ADF test p-value is 0.03. What should you do next?',
    options: [
      'Apply second-order differencing',
      'The series is now stationary, proceed with ARIMA modeling',
      'Apply log transformation',
      'Collect more data',
    ],
    correctAnswer: 1,
    explanation:
      'ADF test p-value of 0.03 < 0.05 means we reject the null hypothesis of non-stationarity. The series is stationary after first-order differencing (d=1). You can now proceed with fitting ARIMA, using d=1 in the model. No need for additional differencing.',
  },
];
