export const trainTestSplitValidationMultipleChoice = {
  title: 'Train-Test Split & Validation - Multiple Choice Questions',
  questions: [
    {
      id: 1,
      question:
        'Why is it critical to split data into train/test sets BEFORE any preprocessing steps like scaling or normalization?',
      options: [
        'To make the code run faster',
        'To prevent data leakage where test set information influences training',
        'To ensure both sets have the same number of samples',
        'To make the model more complex',
      ],
      correctAnswer: 1,
      explanation:
        'Preprocessing fitted on all data (including test) leaks information about test distribution into the model, leading to optimistically biased performance estimates. Always split first, then fit preprocessing only on training data.',
      difficulty: 'intermediate' as const,
      category: 'Best Practices',
    },
    {
      id: 2,
      question:
        'For a binary classification problem with 95% negative and 5% positive examples, what splitting strategy is most appropriate?',
      options: [
        'Random split without any special consideration',
        'Stratified split to maintain 95/5 ratio in both train and test',
        'Put all positive examples in training set',
        'Use 50/50 split to balance the classes',
      ],
      correctAnswer: 1,
      explanation:
        'Stratified splitting maintains the original class distribution (95/5) across train and test sets, ensuring both sets are representative. This prevents scenarios where test set might have 98% negative by random chance.',
      difficulty: 'beginner' as const,
      category: 'Splitting Strategies',
    },
    {
      id: 3,
      question:
        'In a time series forecasting problem for stock prices, which split strategy is correct?',
      options: [
        'Random 80/20 split',
        'Stratified split based on price levels',
        'Time-based split where training data comes before test data chronologically',
        'K-fold cross-validation with random folds',
      ],
      correctAnswer: 2,
      explanation:
        'Time series data requires temporal ordering. Training must always precede testing chronologically to avoid using future information to predict the past (lookahead bias). Random splitting would create severe data leakage.',
      difficulty: 'intermediate' as const,
      category: 'Time Series',
    },
    {
      id: 4,
      question:
        'What is the purpose of having a separate validation set in addition to train and test sets?',
      options: [
        'To have more data for training',
        'To make the evaluation process longer',
        'To tune hyperparameters and make model selection decisions without touching the test set',
        'To confuse other data scientists',
      ],
      correctAnswer: 2,
      explanation:
        'The validation set is used during development for hyperparameter tuning and model selection. The test set remains untouched until final evaluation to provide an unbiased estimate of real-world performance.',
      difficulty: 'beginner' as const,
      category: 'Validation',
    },
    {
      id: 5,
      question:
        'A model achieves 95% accuracy on training set and 70% on test set. What is the most likely issue?',
      options: [
        'The model is underfitting',
        'The test set is too small',
        'The model is overfitting to the training data',
        'The model is perfect and ready for production',
      ],
      correctAnswer: 2,
      explanation:
        "The large gap between training (95%) and test (70%) performance indicates overfitting. The model has learned patterns specific to the training data that don't generalize to new data.",
      difficulty: 'beginner' as const,
      category: 'Diagnosis',
    },
  ],
};
