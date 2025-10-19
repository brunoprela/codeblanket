import { MultipleChoiceQuestion } from '../../../types';

export const crossValidationTechniquesMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'cross-validation-techniques-mc-1',
      question:
        'What is the primary advantage of cross-validation over a single train-test split?',
      options: [
        'Cross-validation is faster to compute',
        'Cross-validation requires less data',
        'Cross-validation provides a more robust performance estimate by averaging across multiple splits',
        'Cross-validation always gives better model performance',
      ],
      correctAnswer: 2,
      explanation:
        'Cross-validation provides multiple train-test splits and averages their performance, reducing the variance in performance estimates caused by the randomness of a single split. This gives a more reliable estimate of how well the model will generalize.',
    },
    {
      id: 'cross-validation-techniques-mc-2',
      question:
        'In 5-fold cross-validation, how many times is each data point used for training?',
      options: [
        '1 time',
        '4 times',
        '5 times',
        'Never (only used for testing)',
      ],
      correctAnswer: 1,
      explanation:
        "In 5-fold CV, the data is split into 5 folds. Each fold serves as the test set once while the other 4 folds are used for training. Therefore, each data point appears in the training set 4 times (when it's not in the test fold).",
    },
    {
      id: 'cross-validation-techniques-mc-3',
      question:
        'Which cross-validation technique should you use for an imbalanced classification dataset with 95% class 0 and 5% class 1?',
      options: [
        'Regular K-Fold',
        'Stratified K-Fold',
        'Leave-One-Out CV',
        'Time Series Split',
      ],
      correctAnswer: 1,
      explanation:
        'Stratified K-Fold maintains the class proportions (95%-5%) in each fold, ensuring that every fold is representative of the original dataset. Regular K-Fold might create folds with no class 1 samples by chance.',
    },
    {
      id: 'cross-validation-techniques-mc-4',
      question:
        'For a stock price prediction problem, which is the CORRECT cross-validation approach?',
      options: [
        'Regular K-Fold with shuffle=True',
        'Stratified K-Fold',
        'TimeSeriesSplit (sequential splits preserving temporal order)',
        'Leave-One-Out Cross-Validation',
      ],
      correctAnswer: 2,
      explanation:
        'Time series data must use TimeSeriesSplit which preserves temporal order—training data always comes before test data. Regular K-Fold with shuffling would create data leakage by allowing the model to use future information to predict the past.',
    },
    {
      id: 'cross-validation-techniques-mc-5',
      question: 'What is the purpose of nested cross-validation?',
      options: [
        'To make cross-validation faster by using fewer folds',
        'To provide an unbiased estimate of model performance when tuning hyperparameters',
        'To handle time series data correctly',
        'To increase the size of the training set',
      ],
      correctAnswer: 1,
      explanation:
        'Nested CV uses an outer loop for performance estimation and an inner loop for hyperparameter tuning. This ensures the test sets in the outer loop never participate in hyperparameter selection, providing an unbiased performance estimate. Non-nested CV is biased because hyperparameters are selected based on the same folds used for evaluation.',
    },
  ];
