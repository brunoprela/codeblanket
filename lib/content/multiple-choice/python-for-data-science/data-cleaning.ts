import { MultipleChoiceQuestion } from '../../../types';

export const datacleaningMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'data-cleaning-mc-1',
    question: 'What is the difference between df.dropna() and df.fillna()?',
    options: [
      'dropna() is faster than fillna()',
      'dropna() removes rows/columns with missing values, fillna() replaces them',
      'They do the same thing, just different syntax',
      'fillna() only works with numeric data',
    ],
    correctAnswer: 1,
    explanation:
      'dropna() removes rows or columns containing missing values (reducing dataset size), while fillna() replaces missing values with specified values (preserving dataset size). The choice depends on whether you can afford to lose data or need to impute it.',
  },
  {
    id: 'data-cleaning-mc-2',
    question:
      'When using pd.to_numeric(series, errors="coerce"), what happens to values that cannot be converted to numbers?',
    options: [
      'They are removed from the Series',
      'They are converted to NaN',
      'They raise an exception',
      'They are left unchanged as strings',
    ],
    correctAnswer: 1,
    explanation:
      'With errors="coerce", values that cannot be converted to numeric become NaN. Other options: errors="raise" throws exception, errors="ignore" returns original Series unchanged.',
  },
  {
    id: 'data-cleaning-mc-3',
    question:
      'What is the IQR (Interquartile Range) method for outlier detection?',
    options: [
      'Values more than 3 standard deviations from the mean',
      'Values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR',
      'The 5 highest and 5 lowest values',
      'Values that appear only once in the dataset',
    ],
    correctAnswer: 1,
    explanation:
      'IQR method defines outliers as values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR, where IQR = Q3 - Q1. This is more robust to outliers than the Z-score method and is the basis for box plot whiskers.',
  },
  {
    id: 'data-cleaning-mc-4',
    question: 'What is "winsorization" in the context of outlier treatment?',
    options: [
      'Removing all outliers from the dataset',
      'Capping extreme values at a specified percentile',
      'Replacing outliers with the mean',
      'Transforming data with logarithm',
    ],
    correctAnswer: 1,
    explanation:
      'Winsorization caps extreme values at specified percentiles (e.g., 5th and 95th), replacing outliers with the boundary values rather than removing them. This preserves sample size while reducing the impact of extreme values.',
  },
  {
    id: 'data-cleaning-mc-5',
    question:
      'Why might you use forward fill (ffill) for missing values in time series data?',
    options: [
      'It is the fastest method',
      'It assumes the last known value persists until the next observation',
      'It removes all missing values',
      'It works only with numeric data',
    ],
    correctAnswer: 1,
    explanation:
      'Forward fill propagates the last valid observation forward, which is appropriate for time series when you assume values remain constant until a new measurement is available (e.g., inventory levels, status flags).',
  },
];
