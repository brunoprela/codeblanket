/**
 * Multiple Choice Questions for Support Vector Machines
 */

import { MultipleChoiceQuestion } from '../../../types';

export const svmMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'svm-mc-1',
    question: 'What are support vectors in SVM?',
    options: [
      'All training data points',
      'The data points closest to the decision boundary',
      'The data points with the highest prediction confidence',
      'The centers of each class',
    ],
    correctAnswer: 1,
    explanation:
      'Support vectors are the training points that lie closest to the decision boundary - specifically, they are on the margin boundaries. These points "support" the hyperplane and are the only points that determine its position. Removing other points doesn\'t change the decision boundary.',
    difficulty: 'easy',
  },
  {
    id: 'svm-mc-2',
    question:
      'Which kernel should you use for data that is already linearly separable?',
    options: [
      'RBF (Radial Basis Function)',
      'Polynomial with degree 5',
      'Linear kernel',
      'Sigmoid kernel',
    ],
    correctAnswer: 2,
    explanation:
      "For linearly separable data, use the linear kernel. It's faster, has fewer hyperparameters to tune, and avoids overfitting. More complex kernels (RBF, polynomial) are unnecessary and can actually hurt performance by creating overly complex boundaries.",
    difficulty: 'easy',
  },
  {
    id: 'svm-mc-3',
    question: 'What happens when you increase the C parameter in SVM?',
    options: [
      'The margin increases and the model becomes simpler',
      'The margin decreases and the model fits training data more closely',
      'The kernel changes automatically',
      'The number of support vectors increases',
    ],
    correctAnswer: 1,
    explanation:
      'Increasing C makes the penalty for misclassification stronger, causing the model to fit training data more closely (smaller margin, more complex boundary). This can lead to overfitting. Small C allows larger margin with more training errors (simpler model).',
    difficulty: 'medium',
  },
  {
    id: 'svm-mc-4',
    question: 'Why is feature scaling critical for SVM?',
    options: [
      'SVM cannot handle different scales',
      'Features with larger scales will dominate the distance calculations in the kernel',
      'It makes training faster',
      'It increases the number of support vectors',
    ],
    correctAnswer: 1,
    explanation:
      'SVM uses distance calculations in kernels (especially RBF). Features with larger numeric ranges dominate these distances, effectively making smaller-scale features irrelevant. StandardScaler should always be applied before training SVM to ensure all features contribute fairly.',
    difficulty: 'medium',
  },
  {
    id: 'svm-mc-5',
    question:
      'What is the main computational limitation of SVM for large datasets?',
    options: [
      'It requires too much memory to store all data',
      'Training complexity is O(n²) to O(n³), making it slow for millions of samples',
      'It cannot handle more than 1000 features',
      'Prediction is too slow',
    ],
    correctAnswer: 1,
    explanation:
      'SVM training has quadratic to cubic time complexity in the number of samples, making it impractical for datasets with millions of examples. While prediction is relatively fast (depends only on support vectors), training large models is prohibitively expensive. For large datasets, consider linear SVM variants or other algorithms.',
    difficulty: 'hard',
  },
];
