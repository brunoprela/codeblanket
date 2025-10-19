/**
 * Multiple Choice Questions for Machine Learning Fundamentals
 */

import { MultipleChoiceQuestion } from '../../../types';

export const mlfundamentalsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'ml-fundamentals-mc-1',
    question:
      "You have a dataset of house features and prices. You want to predict the price of houses you've never seen before. Which type of machine learning problem is this?",
    options: [
      'Unsupervised learning - clustering',
      'Supervised learning - regression',
      'Supervised learning - classification',
      'Reinforcement learning',
    ],
    correctAnswer: 1,
    explanation:
      'This is supervised learning (we have labeled data: features and prices) and specifically regression (predicting a continuous numerical value: price). Classification would involve predicting discrete categories (e.g., expensive/affordable), while unsupervised learning would find patterns without labeled data, and reinforcement learning involves learning through trial and error with rewards.',
    difficulty: 'easy',
  },
  {
    id: 'ml-fundamentals-mc-2',
    question:
      'A model achieves 99% accuracy on the training set but only 65% accuracy on the test set. What problem does this indicate, and what would be the most appropriate first step to address it?',
    options: [
      'Underfitting; increase model complexity',
      'Overfitting; collect more training data or add regularization',
      'Data leakage; check preprocessing steps',
      'Perfect performance; deploy the model',
    ],
    correctAnswer: 1,
    explanation:
      'The large gap between training (99%) and test (65%) accuracy is a classic sign of overfitting. The model has learned to memorize the training data rather than generalizing. Solutions include collecting more training data, adding regularization (L1/L2, dropout), reducing model complexity, or using data augmentation. Underfitting would show poor performance on both sets, and data leakage would typically show unrealistically high performance on both sets.',
    difficulty: 'medium',
  },
  {
    id: 'ml-fundamentals-mc-3',
    question:
      'When splitting data for machine learning, what is the primary purpose of the validation set?',
    options: [
      'To train the model and learn patterns from the data',
      'To provide final, unbiased estimate of model performance',
      'To tune hyperparameters and select between different models',
      'To augment the training set when data is limited',
    ],
    correctAnswer: 2,
    explanation:
      'The validation set is used to tune hyperparameters (like learning rate, regularization strength, or number of trees) and to select between different model architectures without touching the test set. The training set is used to learn patterns, and the test set provides the final unbiased performance estimate. Using the test set for hyperparameter tuning would bias its performance estimate.',
    difficulty: 'medium',
  },
  {
    id: 'ml-fundamentals-mc-4',
    question:
      'Which of the following scenarios represents correct practice in preventing data leakage?',
    options: [
      'Normalize all features using mean and standard deviation of the entire dataset, then split into train/test',
      'Split data into train/test, then fit StandardScaler on training data only and apply it to both sets',
      'Use cross-validation on the entire dataset to select features, then evaluate on a test set',
      'Train multiple models and select the best performing one based on test set accuracy',
    ],
    correctAnswer: 1,
    explanation:
      'Option 2 is correct: split first, then fit the scaler ONLY on training data, and apply the learned transformation to both training and test sets. Option 1 causes leakage by using test set statistics. Option 3 causes leakage by using test data for feature selection. Option 4 turns the test set into a validation set by using it for model selection, losing the unbiased performance estimate.',
    difficulty: 'hard',
  },
  {
    id: 'ml-fundamentals-mc-5',
    question:
      'A linear model has 60% training accuracy and 58% test accuracy on a binary classification task. What is the most likely problem and solution?',
    options: [
      'Overfitting; add regularization',
      'Underfitting; use a more complex model or add features',
      'Perfect balance; deploy immediately',
      'Data leakage; recheck preprocessing',
    ],
    correctAnswer: 1,
    explanation:
      "Both training and test accuracies are low and similar, indicating underfitting (high bias). The model is too simple to capture the underlying patterns. Solutions include using a more complex model architecture, adding relevant features, using polynomial features, or reducing regularization. If the model were overfitting, we'd see high training accuracy with low test accuracy.",
    difficulty: 'medium',
  },
];
