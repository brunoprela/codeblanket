/**
 * Multiple choice questions for Time-Based Features section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const timebasedfeaturesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'For a "month" feature (1-12), what is the correct way to encode it for machine learning?',
    options: [
      'Label encoding (1,2,3,...,12)',
      'One-hot encoding (12 binary columns)',
      'Sine and cosine transformation',
      'Leave as integer',
    ],
    correctAnswer: 2,
    explanation:
      'Month is cyclical (December→January are adjacent). Sine/cosine transformation preserves this circular relationship: month_sin=sin(2π*m/12), month_cos=cos(2π*m/12). Label encoding treats December(12) and January(1) as far apart.',
  },
  {
    id: 'mc2',
    question: 'What is a lag feature in time series?',
    options: [
      'The time delay in model training',
      'Past values of a variable shifted forward as predictive features',
      'The difference between predicted and actual values',
      'Features that slow down the model',
    ],
    correctAnswer: 1,
    explanation:
      "Lag features are past values used as predictors. sales_lag_1 is yesterday's sales, sales_lag_7 is sales from 7 days ago. They capture autocorrelation and temporal dependencies in time series data.",
  },
  {
    id: 'mc3',
    question:
      'Why do the first few rows have missing values when creating lag features or rolling windows?',
    options: [
      "It\'s a bug in the code",
      'There is no past data available for those rows',
      "The scaler hasn't been fitted yet",
      'The data is corrupted',
    ],
    correctAnswer: 1,
    explanation:
      "For lag_7 on the first row, there is no data from 7 days ago. For rolling_mean_30 on row 15, there aren't 30 previous days yet. This is expected behavior. Handle by: dropping initial rows, imputing, or using min_periods parameter.",
  },
  {
    id: 'mc4',
    question:
      'Which feature would likely cause data leakage in a churn prediction model?',
    options: [
      'days_since_last_login',
      'months_since_signup',
      'days_until_churn',
      'average_session_duration',
    ],
    correctAnswer: 2,
    explanation:
      "days_until_churn uses future information (when the customer will churn), which isn't available at prediction time. days_since_last_login uses only past data and is safe. Time-until features cause leakage; time-since features are safe.",
  },
  {
    id: 'mc5',
    question:
      'What is the primary purpose of creating rolling window features (e.g., 7-day moving average)?',
    options: [
      'To create more features and improve accuracy',
      'To smooth out noise and capture trends over time periods',
      'To reduce dataset size',
      'To handle missing values',
    ],
    correctAnswer: 1,
    explanation:
      'Rolling windows aggregate data over time periods, smoothing out daily noise while preserving medium-term trends. A 7-day moving average shows weekly patterns without daily volatility. They help models focus on meaningful patterns rather than random fluctuations.',
  },
];
