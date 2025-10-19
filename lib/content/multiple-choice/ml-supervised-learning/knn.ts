/**
 * Multiple Choice Questions for k-Nearest Neighbors
 */

import { MultipleChoiceQuestion } from '../../../types';

export const knnMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'knn-mc-1',
    question:
      'What is the main disadvantage of using a very small k (e.g., k=1) in kNN?',
    options: [
      'The model will underfit and have high bias',
      'The model will overfit and be sensitive to noise',
      'The model will be too slow to train',
      'The model cannot make probabilistic predictions',
    ],
    correctAnswer: 1,
    explanation:
      'A very small k (especially k=1) makes the model highly sensitive to noise and outliers in the training data, leading to overfitting. The decision boundary becomes very jagged, fitting to individual noisy points rather than the underlying pattern. A larger k smooths the decision boundary and reduces sensitivity to individual points, though too large k can lead to underfitting.',
    difficulty: 'medium',
  },
  {
    id: 'knn-mc-2',
    question:
      'You have two features: income ($20k-$200k) and age (18-80 years). You apply kNN without scaling. What happens?',
    options: [
      'Both features contribute equally to distance calculations',
      'Income will dominate distances due to its larger scale, making age nearly irrelevant',
      'Age will dominate because it has fewer unique values',
      "Scaling doesn't matter for kNN",
    ],
    correctAnswer: 1,
    explanation:
      'Income has a much larger numeric range (180k spread) compared to age (62 year spread), so income differences will dominate Euclidean distance calculations by roughly 3x in magnitude. This means kNN will base decisions primarily on income similarity regardless of whether age is actually more predictive. Feature scaling (e.g., StandardScaler) is critical to give both features equal opportunity to influence predictions.',
    difficulty: 'easy',
  },
  {
    id: 'knn-mc-3',
    question:
      'What is the computational complexity of making predictions with kNN for n training samples and d features?',
    options: [
      'O(1) - constant time',
      'O(d) - linear in features',
      'O(n·d) - must compute distance to all training points',
      'O(n log n) - using efficient search',
    ],
    correctAnswer: 2,
    explanation:
      'For each prediction, kNN must compute the distance (O(d) operations) to every training point (n points), resulting in O(n·d) complexity. This makes prediction slow for large datasets. KD-trees or ball trees can improve this to O(d log n) in low dimensions, but degrade to O(n·d) in high dimensions. This is why kNN is called "lazy learning" - all computation happens at prediction time.',
    difficulty: 'medium',
  },
  {
    id: 'knn-mc-4',
    question:
      'In high-dimensional spaces (e.g., 1000+ features), what problem does kNN face?',
    options: [
      'It runs out of memory',
      'The curse of dimensionality makes distances meaningless',
      'It cannot handle categorical features',
      'Training becomes too slow',
    ],
    correctAnswer: 1,
    explanation:
      'The curse of dimensionality causes all points to become roughly equidistant in high dimensions, making it impossible to identify truly "similar" neighbors. The ratio of nearest to farthest distance approaches 1. This fundamentally breaks kNN\'s assumption that similar inputs (by distance) should have similar outputs. Solutions include dimensionality reduction (PCA) or using algorithms that don\'t rely on distances.',
    difficulty: 'hard',
  },
  {
    id: 'knn-mc-5',
    question:
      "What is the advantage of using weighted kNN (weights='distance') over uniform weights?",
    options: [
      "It's faster to compute",
      'It gives more influence to closer neighbors',
      'It requires less memory',
      'It handles missing data better',
    ],
    correctAnswer: 1,
    explanation:
      'Distance-weighted kNN assigns higher weight to closer neighbors (typically weight = 1/distance), making them more influential in the prediction. This is beneficial when you have a larger k but want nearby neighbors to dominate the decision. It allows using larger k values (for stability) while still being sensitive to local patterns. Uniform weights treat all k neighbors equally regardless of distance.',
    difficulty: 'medium',
  },
];
