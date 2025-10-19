/**
 * Multiple choice questions for Univariate Analysis section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const univariateanalysisMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'A feature has mean = 150 and median = 100. What does this suggest about the distribution?',
    options: [
      'The distribution is perfectly symmetric',
      'The distribution is right-skewed (positive skew) with high outliers',
      'The distribution is left-skewed (negative skew) with low outliers',
      'The feature has no outliers',
    ],
    correctAnswer: 1,
    explanation:
      'When mean > median, it indicates right-skewness. Extreme high values (outliers on the right) pull the mean upward, while the median (middle value) remains relatively unaffected. This is common in income, property prices, and other naturally right-skewed variables.',
  },
  {
    id: 'mc2',
    question:
      'You have a feature with skewness = 2.5. Which transformation is MOST likely to help normalize this distribution?',
    options: [
      'Square transformation (x²)',
      'No transformation needed',
      'Log transformation (log(x) or log(x+1))',
      'Standardization (z-score)',
    ],
    correctAnswer: 2,
    explanation:
      "Positive skewness (2.5 is quite high) indicates right-skewness. Log transformation is the most effective for right-skewed data as it compresses the right tail. Standardization doesn't change skewness, and squaring would make it worse.",
  },
  {
    id: 'mc3',
    question:
      'Using the IQR method for outlier detection, a value is considered an outlier if it falls outside which range?',
    options: [
      '[Mean - 2×Std, Mean + 2×Std]',
      '[Q1 - 1.5×IQR, Q3 + 1.5×IQR]',
      '[Min, Max]',
      '[1st percentile, 99th percentile]',
    ],
    correctAnswer: 1,
    explanation:
      "The IQR (Interquartile Range) method defines outliers as values below Q1 - 1.5×IQR or above Q3 + 1.5×IQR, where IQR = Q3 - Q1. This method is robust and doesn't assume any particular distribution.",
  },
  {
    id: 'mc4',
    question:
      'What does a Coefficient of Variation (CV) close to zero indicate about a feature?',
    options: [
      'The feature is highly variable and informative',
      'The feature has many outliers',
      'The feature has low variance relative to its mean and may be uninformative',
      'The feature follows a normal distribution',
    ],
    correctAnswer: 2,
    explanation:
      'Coefficient of Variation (CV = std/mean) measures relative variability. A CV near zero means the feature varies very little relative to its mean - essentially constant. Such features provide little information for ML models and are candidates for removal.',
  },
  {
    id: 'mc5',
    question:
      'Which normality test is generally considered most appropriate for sample sizes less than 5,000?',
    options: [
      'Kolmogorov-Smirnov test',
      'Chi-square test',
      'Shapiro-Wilk test',
      'F-test',
    ],
    correctAnswer: 2,
    explanation:
      'The Shapiro-Wilk test is considered the most powerful normality test for sample sizes less than 5,000. It has better statistical power than Kolmogorov-Smirnov for detecting departures from normality in smaller samples. For larger samples, Anderson-Darling or Kolmogorov-Smirnov are more appropriate.',
  },
];
