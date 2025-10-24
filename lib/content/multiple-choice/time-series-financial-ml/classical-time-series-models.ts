import { MultipleChoiceQuestion } from '@/lib/types';

export const classicalTimeSeriesModelsMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'ctsm-mc-1',
      question: 'When does the PACF cut off sharply but ACF decays gradually?',
      options: [
        'MA(q) process',
        'AR(p) process',
        'ARMA(p,q) process',
        'Non-stationary process',
      ],
      correctAnswer: 1,
      explanation:
        'AR(p) process has PACF that cuts off at lag p (only first p lags significant) and ACF that decays exponentially/gradually. This is the signature pattern for identifying AR order. Opposite for MA: ACF cuts off, PACF decays. ARMA: both decay. Use PACF to identify AR order.',
    },
    {
      id: 'ctsm-mc-2',
      question:
        'For stock prices that are non-stationary, what ARIMA order is typically used?',
      options: ['ARIMA(0,0,0)', 'ARIMA(1,0,1)', 'ARIMA(1,1,1)', 'ARIMA(5,0,5)'],
      correctAnswer: 2,
      explanation:
        'ARIMA(1,1,1) is typical for stock prices: d=1 for differencing (prices non-stationary), p=1 for AR component (weak momentum), q=1 for MA component (shock absorption). The d=1 transforms prices to returns, achieving stationarity. Financial returns rarely need p,q > 2.',
    },
    {
      id: 'ctsm-mc-3',
      question:
        'What does it mean if model residuals fail the Ljung-Box test (p-value < 0.05)?',
      options: [
        'Model is perfect',
        'Residuals show autocorrelation, model is inadequate',
        'Residuals are normally distributed',
        'Model is overfitting',
      ],
      correctAnswer: 1,
      explanation:
        "Ljung-Box tests null hypothesis: no autocorrelation in residuals. p-value < 0.05 → reject null → residuals ARE autocorrelated → model hasn't captured all patterns → model inadequate. Good model has white noise residuals (p > 0.05). Solution: increase p or q, or add exogenous variables.",
    },
    {
      id: 'ctsm-mc-4',
      question:
        'When comparing ARIMA models, which criterion penalizes complexity more?',
      options: [
        'AIC (Akaike Information Criterion)',
        'BIC (Bayesian Information Criterion)',
        'R-squared',
        'MSE (Mean Squared Error)',
      ],
      correctAnswer: 1,
      explanation:
        'BIC penalizes complexity more than AIC: BIC = -2*log(L) + k*log(n), AIC = -2*log(L) + 2*k. BIC penalty grows with sample size n. Use BIC to avoid overfitting (prefers simpler models), use AIC for better fit (allows complexity). For time series with 1000+ observations, BIC strongly prefers simpler models.',
    },
    {
      id: 'ctsm-mc-5',
      question:
        'Why use walk-forward validation instead of random train-test split for ARIMA?',
      options: [
        'Walk-forward is faster',
        'Random split respects temporal ordering',
        'Walk-forward respects temporal ordering and avoids lookahead bias',
        'Walk-forward gives higher accuracy',
      ],
      correctAnswer: 2,
      explanation:
        'Walk-forward validation respects time: train on past (t-252 to t), test on future (t to t+21), slide forward. Random split uses future data in training (lookahead bias), unrealistic for trading. ARIMA assumes temporal dependency—random split violates this. Walk-forward simulates real deployment where you only have past data.',
    },
  ];
