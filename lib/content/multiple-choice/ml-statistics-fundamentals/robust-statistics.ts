/**
 * Multiple choice questions for Robust Statistics section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const robuststatisticsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the "breakdown point" of a statistic?',
    options: [
      'The point where the statistic equals zero',
      'The minimum fraction of outliers that can make the statistic arbitrarily large',
      'The computational limit of the algorithm',
      'The standard error of the estimate',
    ],
    correctAnswer: 1,
    explanation:
      'Breakdown point is the minimum fraction of outliers needed to make a statistic arbitrarily large or small. The mean has 0% breakdown (one outlier can ruin it), while the median has 50% breakdown (up to half the data can be outliers).',
  },
  {
    id: 'mc2',
    question:
      'Which regression method is most robust to a large proportion (30%) of outliers?',
    options: [
      'Ordinary Least Squares (OLS)',
      'Ridge Regression',
      'RANSAC Regression',
      'Lasso Regression',
    ],
    correctAnswer: 2,
    explanation:
      'RANSAC can handle up to 50% outliers by fitting to random subsets and finding the largest set of inliers. OLS is very sensitive to outliers. Ridge and Lasso provide regularization but not outlier robustness.',
  },
  {
    id: 'mc3',
    question: 'The Median Absolute Deviation (MAD) is a robust alternative to:',
    options: ['The mean', 'The standard deviation', 'The mode', 'The range'],
    correctAnswer: 1,
    explanation:
      "MAD is a robust measure of spread/dispersion, making it a robust alternative to standard deviation. It's calculated as the median of absolute deviations from the median, and has a 50% breakdown point unlike the standard deviation which has 0%.",
  },
  {
    id: 'mc4',
    question: 'Z-score outlier detection (|Z| > 3) assumes the data is:',
    options: [
      'Uniformly distributed',
      'Approximately normally distributed',
      'Exponentially distributed',
      'Binary',
    ],
    correctAnswer: 1,
    explanation:
      'Z-score outlier detection assumes normality. Under normality, ~99.7% of data is within 3 standard deviations. This method fails for heavy-tailed or skewed distributions where legitimate values can exceed 3σ.',
  },
  {
    id: 'mc5',
    question:
      'Which outlier detection method makes NO distributional assumptions?',
    options: [
      'Z-score (>3σ)',
      'Elliptic Envelope',
      'Isolation Forest',
      'Mahalanobis distance',
    ],
    correctAnswer: 2,
    explanation:
      'Isolation Forest is distribution-free and makes no assumptions about the data distribution. It identifies outliers by how easily they can be isolated in random partitions. Z-score assumes normality, Elliptic Envelope assumes Gaussian, Mahalanobis assumes multivariate normality.',
  },
];
