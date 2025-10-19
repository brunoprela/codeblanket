/**
 * Multiple choice questions for Time Series Statistics section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const timeseriesstatisticsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What characterizes a stationary time series?',
    options: [
      'Values remain constant over time',
      'Mean, variance, and autocorrelation structure are constant over time',
      'No trend or seasonality',
      'All values are positive',
    ],
    correctAnswer: 1,
    explanation:
      "A stationary time series has constant mean, variance, and autocorrelation structure over time. This doesn't mean values are constant - they vary randomly around a stable mean with stable variance.",
  },
  {
    id: 'mc2',
    question: 'The Augmented Dickey-Fuller (ADF) test has null hypothesis:',
    options: [
      'The series is stationary',
      'The series has a unit root (is non-stationary)',
      'The series has no trend',
      'The series has autocorrelation',
    ],
    correctAnswer: 1,
    explanation:
      'ADF tests H₀: unit root (non-stationary). If p<0.05, reject H₀ and conclude the series is stationary. This is opposite to KPSS where H₀ is stationarity.',
  },
  {
    id: 'mc3',
    question: '"X Granger-causes Y" means:',
    options: [
      'X causes Y',
      "Past values of X help predict Y beyond what Y's own past predicts",
      'X and Y are correlated',
      'Y causes X',
    ],
    correctAnswer: 1,
    explanation:
      "Granger causality means past X provides useful information for predicting Y beyond Y's own history. This is about predictive power, not true causation. Confounders can create Granger causality without true causal relationships.",
  },
  {
    id: 'mc4',
    question: 'Volatility clustering in financial returns means:',
    options: [
      'Returns are clustered around zero',
      'Large price changes tend to be followed by large changes',
      'All stocks move together',
      'Volatility is constant over time',
    ],
    correctAnswer: 1,
    explanation:
      'Volatility clustering means periods of high volatility tend to follow periods of high volatility, and calm periods follow calm periods. This shows up as positive autocorrelation in absolute or squared returns, even when returns themselves show little autocorrelation.',
  },
  {
    id: 'mc5',
    question: 'To make a non-stationary time series stationary, you typically:',
    options: [
      'Remove the mean',
      'Take the first difference (subtract previous value)',
      'Square all values',
      'Remove outliers',
    ],
    correctAnswer: 1,
    explanation:
      'First differencing (Δy_t = y_t - y_{t-1}) removes trends and often achieves stationarity. For series with unit roots, differencing converts them to stationary series. Some series may require second differencing or seasonal differencing.',
  },
];
