import { MultipleChoiceQuestion } from '../../../types';

export const timeseriespandasMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'time-series-pandas-mc-1',
    question:
      'What is the difference between resample("M") and resample("MS") in Pandas?',
    options: [
      'M is monthly, MS is milliseconds',
      'M is month end, MS is month start',
      'They are identical',
      'M is mean, MS is sum',
    ],
    correctAnswer: 1,
    explanation:
      'In Pandas time series, "M" (or "ME" in newer versions) refers to month end dates, while "MS" refers to month start dates. This affects which date each aggregated value is labeled with.',
  },
  {
    id: 'time-series-pandas-mc-2',
    question: 'What does df["price",].shift(1) do in a time series DataFrame?',
    options: [
      'Moves all values forward by 1 position (creates lag)',
      'Adds 1 to all values',
      'Shifts the index by 1 day',
      'Removes the first row',
    ],
    correctAnswer: 0,
    explanation:
      "shift(1) moves all values forward by one position, creating a lagged version. The first value becomes NaN. This is commonly used to create yesterday's price for calculating returns: returns = price / price.shift(1) - 1.",
  },
  {
    id: 'time-series-pandas-mc-3',
    question:
      'In a rolling window calculation, what does the min_periods parameter control?',
    options: [
      'The minimum size of the rolling window',
      'The minimum number of observations required to compute a result',
      'The minimum value in the window',
      'The minimum time period',
    ],
    correctAnswer: 1,
    explanation:
      'min_periods specifies the minimum number of observations required to return a value. For example, rolling(window=20, min_periods=10) will start returning values after 10 observations instead of waiting for the full 20-observation window.',
  },
  {
    id: 'time-series-pandas-mc-4',
    question:
      'What is the difference between forward fill (ffill) and interpolation when upsampling?',
    options: [
      'ffill repeats the last value, interpolation estimates intermediate values',
      'They are identical',
      'ffill is faster',
      'interpolation only works with numeric data',
    ],
    correctAnswer: 0,
    explanation:
      'Forward fill (ffill) repeats the last known value for all new periods, while interpolation estimates intermediate values based on surrounding data points. For example, ffill of [100, 200] daily to hourly repeats 100 for the first day, while interpolation creates values like 100, 104.17, 108.33, etc.',
  },
  {
    id: 'time-series-pandas-mc-5',
    question:
      'Why might you use freq="B" instead of freq="D" in pd.date_range()?',
    options: [
      'B is faster than D',
      'B generates business days only (excluding weekends), D generates all calendar days',
      'B is for binary data',
      'D is deprecated',
    ],
    correctAnswer: 1,
    explanation:
      'freq="B" generates business days (Monday-Friday), excluding weekends, which is appropriate for financial data like stock prices that don\'t trade on weekends. freq="D" generates all calendar days including weekends.',
  },
];
