/**
 * Multiple choice questions for Descriptive Statistics section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const descriptivestatisticsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'For a right-skewed distribution (e.g., income data), which relationship is typically true?',
    options: [
      'Mean < Median < Mode',
      'Mode < Median < Mean',
      'Median < Mode < Mean',
      'Mean = Median = Mode',
    ],
    correctAnswer: 1,
    explanation:
      'In a right-skewed distribution, outliers on the right pull the mean upward, making it the largest. The median is less affected, and the mode (peak) is typically lowest. This gives: Mode < Median < Mean.',
  },
  {
    id: 'mc2',
    question:
      'Which measure of central tendency is most appropriate for categorical data?',
    options: ['Mean', 'Median', 'Mode', 'Standard deviation'],
    correctAnswer: 2,
    explanation:
      'The mode is the only measure of central tendency that works for categorical data, as it simply identifies the most frequent category. Mean and median require numerical ordering, which doesn\'t apply to categories like "red", "blue", "green".',
  },
  {
    id: 'mc3',
    question:
      'A dataset has Q1 = 50, Q3 = 80, and a value of 140. Using the IQR method with factor 1.5, is 140 an outlier?',
    options: [
      'No, because 140 < Q3 + 1.5×IQR = 125',
      'Yes, because 140 > Q3 + 1.5×IQR = 125',
      'No, because 140 < Q3 + 1.5×IQR = 170',
      'Cannot determine without knowing the mean',
    ],
    correctAnswer: 1,
    explanation:
      'IQR = Q3 - Q1 = 80 - 50 = 30. The upper outlier bound is Q3 + 1.5×IQR = 80 + 1.5×30 = 80 + 45 = 125. Since 140 > 125, it is considered an outlier by the IQR method.',
  },
  {
    id: 'mc4',
    question:
      'When should you use the coefficient of variation (CV) instead of standard deviation?',
    options: [
      'When comparing variability across variables with different units or scales',
      'When the data is normally distributed',
      'When you want an absolute measure of spread',
      'When the mean is negative',
    ],
    correctAnswer: 0,
    explanation:
      "The coefficient of variation (CV = σ/μ × 100%) is a relative measure of variability that allows comparison across different scales or units. For example, you can compare the CV of height (cm) with weight (kg), which wouldn't be meaningful with standard deviation alone.",
  },
  {
    id: 'mc5',
    question:
      'A feature has a skewness of 2.5. What transformation is most likely to help normalize this distribution?',
    options: [
      'Square the values',
      'Take the logarithm',
      'Standardize (z-score)',
      'No transformation needed',
    ],
    correctAnswer: 1,
    explanation:
      "A skewness of 2.5 indicates strong right-skewness. The logarithm transformation compresses large values more than small values, reducing right-skew. Squaring would increase skewness, standardization doesn't change the shape, and the high skewness definitely needs addressing.",
  },
];
