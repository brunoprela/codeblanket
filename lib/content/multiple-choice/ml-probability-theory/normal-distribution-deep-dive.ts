/**
 * Multiple choice questions for Normal Distribution Deep Dive section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const normaldistributiondeepdiveMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'mc1',
      question:
        'For a standard normal distribution N(0,1), what is P(-1 ≤ X ≤ 1)?',
      options: [
        'Approximately 50%',
        'Approximately 68%',
        'Approximately 95%',
        'Approximately 99.7%',
      ],
      correctAnswer: 1,
      explanation:
        'By the 68-95-99.7 rule, approximately 68% of data falls within one standard deviation of the mean. For N(0,1), this is the interval [-1, 1].',
    },
    {
      id: 'mc2',
      question: 'If X ~ N(10, 25), what are the mean and standard deviation?',
      options: [
        'Mean = 10, SD = 25',
        'Mean = 10, SD = 5',
        'Mean = 25, SD = 10',
        'Mean = 5, SD = 10',
      ],
      correctAnswer: 1,
      explanation:
        'Normal distribution notation N(μ, σ²) uses VARIANCE, not standard deviation. So N(10, 25) has mean μ=10 and variance σ²=25, giving standard deviation σ=√25=5.',
    },
    {
      id: 'mc3',
      question:
        'If X₁ ~ N(3, 4) and X₂ ~ N(5, 9) are independent, what is X₁ + X₂?',
      options: ['N(8, 13)', 'N(8, 5)', 'N(8, √13)', 'N(15, 36)'],
      correctAnswer: 0,
      explanation:
        'For independent normals: means add and variances add. μ = 3+5 = 8, σ² = 4+9 = 13. So X₁+X₂ ~ N(8, 13). Note: standard deviations do NOT simply add!',
    },
    {
      id: 'mc4',
      question: 'A data point has a Z-score of -2.5. What does this mean?',
      options: [
        'The point is 2.5 units below the mean',
        'The point is 2.5 standard deviations below the mean',
        'The point has value -2.5',
        'The point is an error',
      ],
      correctAnswer: 1,
      explanation:
        'Z-score = (X - μ)/σ measures how many standard deviations a point is from the mean. Z = -2.5 means the point is 2.5 standard deviations below the mean. This is somewhat unusual (outside 95% interval).',
    },
    {
      id: 'mc5',
      question:
        'What happens to a normal distribution when you multiply all values by 3?',
      options: [
        'Mean is multiplied by 3, variance is multiplied by 3',
        'Mean is multiplied by 3, variance is multiplied by 9',
        'Mean is multiplied by 3, variance stays the same',
        'Both mean and variance are multiplied by 3',
      ],
      correctAnswer: 1,
      explanation:
        'For Y = aX where X ~ N(μ, σ²): E[Y] = aμ and Var(Y) = a²σ². With a=3: mean becomes 3μ, variance becomes 9σ². Standard deviation becomes 3σ (not 9σ).',
    },
  ];
