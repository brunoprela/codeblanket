export const crossValidationTechniquesMultipleChoice = {
  title: 'Cross-Validation Techniques - Multiple Choice Questions',
  questions: [
    {
      id: 1,
      question:
        'What is the main advantage of k-fold cross-validation over a single train-test split?',
      options: [
        "It's faster to compute",
        'It provides more reliable performance estimates by averaging over k different splits',
        'It requires less data',
        'It always gives higher accuracy scores',
      ],
      correctAnswer: 1,
      explanation:
        'K-fold CV provides k different train-test splits, and averaging the results gives a more robust and reliable estimate of model performance with confidence intervals, reducing variance from a single lucky or unlucky split.',
      difficulty: 'beginner' as const,
      category: 'Concepts',
    },
    {
      id: 2,
      question:
        'For a dataset with 1000 samples, what is the main disadvantage of Leave-One-Out Cross-Validation (LOOCV)?',
      options: [
        "It doesn't provide reliable estimates",
        'It requires 1000 model trainings, making it computationally expensive',
        "It can't be used for classification problems",
        'It always gives worse results than k-fold CV',
      ],
      correctAnswer: 1,
      explanation:
        "LOOCV requires training the model N times (once per sample), which becomes prohibitively expensive for large datasets. With 1000 samples, you'd need 1000 model trainings vs 5-10 for k-fold CV.",
      difficulty: 'intermediate' as const,
      category: 'Computational Cost',
    },
    {
      id: 3,
      question:
        'Why is standard k-fold cross-validation inappropriate for time series data?',
      options: [
        'Time series data has too many features',
        'It would use future data to predict past data, creating data leakage',
        "K-fold CV doesn't work with continuous target variables",
        "Time series models don't need validation",
      ],
      correctAnswer: 1,
      explanation:
        'Random k-fold CV shuffles data, putting future observations in training and past observations in test set. This violates temporal ordering and creates lookahead bias. Use TimeSeriesSplit instead.',
      difficulty: 'intermediate' as const,
      category: 'Time Series',
    },
    {
      id: 4,
      question: 'What is nested cross-validation used for?',
      options: [
        'Making models train faster',
        'Getting unbiased performance estimates when tuning hyperparameters',
        'Reducing the amount of data needed',
        'Making the code more complex for no reason',
      ],
      correctAnswer: 1,
      explanation:
        'Nested CV uses an outer loop for performance estimation and inner loop for hyperparameter tuning. This prevents optimistic bias from selecting hyperparameters that happen to work well on a specific validation set.',
      difficulty: 'advanced' as const,
      category: 'Methodology',
    },
    {
      id: 5,
      question:
        'In 5-fold stratified cross-validation for a classification problem, what is preserved across folds?',
      options: [
        'The exact same samples in each fold',
        'The class distribution (proportion of each class)',
        'The order of the samples',
        'The feature correlations',
      ],
      correctAnswer: 1,
      explanation:
        'Stratified CV ensures each fold maintains the same class distribution as the original dataset. For example, if the original data has 70% class A and 30% class B, each fold will have approximately the same 70/30 split.',
      difficulty: 'beginner' as const,
      category: 'Stratification',
    },
  ],
};
