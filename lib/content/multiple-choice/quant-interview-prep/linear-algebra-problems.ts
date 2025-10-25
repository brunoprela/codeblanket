import { MultipleChoiceQuestion } from '@/lib/types';

export const linearAlgebraProblemsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'lap-mc-1',
    question: 'What is the determinant of [[2,3],[1,4]]?',
    options: ['5', '8', '11', '-5'],
    correctAnswer: 0,
    explanation:
      'For 2×2 matrix [[a,b],[c,d]], det = ad - bc. Here: det = 2(4) - 3(1) = 8 - 3 = 5. The determinant represents the signed area scaling factor of the linear transformation.',
  },
  {
    id: 'lap-mc-2',
    question:
      'Matrix A has eigenvalues [3, 1, -2]. What is tr(A) (trace of A)?',
    options: ['2', '3', '6', '-6'],
    correctAnswer: 0,
    explanation:
      'The trace equals the sum of eigenvalues. tr(A) = 3 + 1 + (-2) = 2. This is a fundamental property: tr(A) = Σλᵢ = sum of diagonal elements. Also, det(A) = Πλᵢ = 3(1)(-2) = -6.',
  },
  {
    id: 'lap-mc-3',
    question:
      'A covariance matrix has eigenvalues [10, 3, 2]. What percentage of variance is explained by the first principal component?',
    options: ['33%', '50%', '67%', '75%'],
    correctAnswer: 2,
    explanation:
      'Total variance = 10 + 3 + 2 = 15. First PC variance = 10. Percentage = 10/15 = 2/3 ≈ 67%. The first eigenvalue represents the variance along the direction of maximum variance.',
  },
  {
    id: 'lap-mc-4',
    question:
      'For data matrix X with 50 samples and 100 features, what is the maximum possible rank of the sample covariance matrix S = XᵀX?',
    options: ['50', '100', '150', '5000'],
    correctAnswer: 0,
    explanation:
      'Rank of XᵀX ≤ rank(X) ≤ min (n_samples, n_features) = min(50, 100) = 50. Since we have only 50 observations, the covariance matrix (100×100) can have at most rank 50, making it singular (non-invertible) with at least 50 zero eigenvalues.',
  },
  {
    id: 'lap-mc-5',
    question: 'What is the rank of [[1,2,3],[2,4,6]]?',
    options: ['1', '2', '3', '0'],
    correctAnswer: 0,
    explanation:
      'Row 2 = 2 × Row 1, so the rows are linearly dependent. The matrix has only 1 linearly independent row, giving rank = 1. This means the row space is 1-dimensional (a line).',
  },
];
