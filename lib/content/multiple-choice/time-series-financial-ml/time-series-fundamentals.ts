import { MultipleChoiceQuestion } from '@/lib/types';

export const timeSeriesFundamentalsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'tsf-mc-1',
    question:
      'What is the correct interpretation when ADF test p-value = 0.03 and KPSS test p-value = 0.08?',
    options: [
      'Series is non-stationary, apply differencing',
      'Series is stationary, no transformation needed',
      'Tests are contradictory, cannot determine stationarity',
      'Apply detrending instead of differencing',
    ],
    correctAnswer: 1,
    explanation:
      'ADF null hypothesis: non-stationary. p-value < 0.05 → reject null → series IS stationary. KPSS null hypothesis: stationary. p-value > 0.05 → fail to reject null → series IS stationary. Both tests agree: series is stationary. No transformation needed for modeling.',
  },
  {
    id: 'tsf-mc-2',
    question:
      'For financial modeling, which transformation is most appropriate for stock prices?',
    options: [
      'No transformation, use prices directly',
      'First difference: P_t - P_{t-1}',
      'Log returns: log(P_t / P_{t-1})',
      'Standardization: (P_t - mean) / std',
    ],
    correctAnswer: 2,
    explanation:
      "Log returns are preferred because: (1) Achieve stationarity by differencing, (2) Stabilize variance with log transform, (3) Additive across time: log(P_t/P_0) = sum of log returns, (4) Symmetric: +10% and -10% have equal magnitude. Simple differences don't stabilize variance. Standardization doesn't remove trend. Prices are non-stationary.",
  },
  {
    id: 'tsf-mc-3',
    question:
      'What does high autocorrelation in squared returns (ACF of r_t^2) indicate?',
    options: [
      'Returns are predictable',
      'Series is non-stationary',
      'Volatility clustering (ARCH effects)',
      'Mean reversion in prices',
    ],
    correctAnswer: 2,
    explanation:
      'High ACF of squared returns indicates volatility clustering: periods of high volatility follow high volatility. This is an ARCH/GARCH effect. Returns themselves may have low autocorrelation (unpredictable), but squared returns (volatility proxy) are highly autocorrelated. Requires GARCH-type models for volatility forecasting.',
  },
  {
    id: 'tsf-mc-4',
    question:
      'Why is random train-test split inappropriate for financial time series?',
    options: [
      'Random splits are fine for all data types',
      'It violates temporal dependency and creates lookahead bias',
      'It makes the model train too slowly',
      'Random splits reduce model accuracy',
    ],
    correctAnswer: 1,
    explanation:
      "Random splits create lookahead bias: training data contains future information relative to test data. This violates temporal causality—you can't use tomorrow's data to predict yesterday. Time series requires temporal ordering: train on past, test on future. Example: train on 2020-2022, test on 2023. Random split would train on scattered 2023 days, predicting 2020, which is unrealistic.",
  },
  {
    id: 'tsf-mc-5',
    question:
      'What does a slowly decaying ACF with high values at all lags indicate?',
    options: [
      'Series has strong seasonality',
      'Series is white noise',
      'Series is non-stationary',
      'Series follows ARMA process',
    ],
    correctAnswer: 2,
    explanation:
      'Slowly decaying ACF with persistent high values is the signature of non-stationarity. Stationary series have ACF that drops quickly to near-zero. Non-stationary series (like prices) show ACF ≈ 0.9+ at many lags because past values highly predictive of future (trending). Apply differencing to achieve stationarity. ARMA has exponential decay, seasonal has periodic spikes.',
  },
];
