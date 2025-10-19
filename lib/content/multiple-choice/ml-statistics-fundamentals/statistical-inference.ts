/**
 * Multiple choice questions for Statistical Inference section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const statisticalinferenceMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'If you double the sample size from 100 to 200, how does the standard error of the mean change?',
    options: [
      'It is cut in half',
      'It is reduced by a factor of √2 (about 1.41)',
      'It stays the same',
      'It doubles',
    ],
    correctAnswer: 1,
    explanation:
      'Standard error SE = σ/√n. If n doubles, √n increases by √2, so SE decreases by a factor of √2 ≈ 1.41. To cut SE in half, you would need to quadruple the sample size (4x), not double it.',
  },
  {
    id: 'mc2',
    question:
      'A 95% confidence interval for model accuracy is [0.80, 0.90]. What is the correct interpretation?',
    options: [
      'There is a 95% probability that the true accuracy is between 80% and 90%',
      '95% of the predictions will have accuracy between 80% and 90%',
      'If we repeated this process many times, 95% of the resulting intervals would contain the true accuracy',
      'The model will be 95% accurate on average',
    ],
    correctAnswer: 2,
    explanation:
      'A confidence interval describes the long-run behavior of the procedure, not probability about this specific interval. If we repeated the entire process (sampling, training, testing) many times, about 95% of the resulting CIs would contain the true accuracy.',
  },
  {
    id: 'mc3',
    question:
      'The Central Limit Theorem states that as sample size increases, the sampling distribution of the sample mean approaches:',
    options: [
      'The distribution of the population',
      'A normal distribution',
      'A uniform distribution',
      'An exponential distribution',
    ],
    correctAnswer: 1,
    explanation:
      "The Central Limit Theorem states that the distribution of sample means approaches a normal distribution, regardless of the population's distribution shape. This is why we can use normal-based inference even when the underlying data is not normal.",
  },
  {
    id: 'mc4',
    question:
      'What is the primary advantage of bootstrap methods for statistical inference?',
    options: [
      'They are faster to compute than traditional methods',
      'They require fewer data points',
      'They make no assumptions about the underlying distribution',
      'They always produce narrower confidence intervals',
    ],
    correctAnswer: 2,
    explanation:
      'Bootstrap is a non-parametric method that makes no distributional assumptions. It works by resampling from your data, so it adapts to whatever distribution you have. This makes it very flexible and widely applicable, though it can be computationally intensive.',
  },
  {
    id: 'mc5',
    question:
      'You have a test set of 100 samples. Your model achieves 85% accuracy. The 95% confidence interval is approximately:',
    options: ['[0.78, 0.92]', '[0.80, 0.90]', '[0.83, 0.87]', '[0.75, 0.95]'],
    correctAnswer: 0,
    explanation:
      'For a proportion, SE ≈ √(p(1-p)/n) = √(0.85×0.15/100) ≈ 0.036. The 95% CI is approximately 0.85 ± 1.96×0.036 = 0.85 ± 0.07 = [0.78, 0.92]. This wide interval reflects the small test set size - more test data would narrow it.',
  },
];
