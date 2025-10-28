export const modelDebuggingMultipleChoice = {
  title: 'Model Debugging - Multiple Choice Questions',
  questions: [
    {
      id: 1,
      question:
        'Your model achieves 98% training accuracy but 65% test accuracy. What is the most likely cause?',
      options: [
        'The test set is corrupted',
        'The model is overfitting to the training data',
        'The model is underfitting',
        'This is normal and expected',
      ],
      correctAnswer: 1,
      explanation:
        "Large gap between training (98%) and test (65%) performance is a classic sign of overfitting. The model has memorized training data specifics that don't generalize. Solutions: reduce complexity, add regularization, get more data, or improve feature engineering.",
      difficulty: 'easy' as const,
      category: 'Diagnosis',
    },
    {
      id: 2,
      question:
        'What is the most common cause of unexpectedly poor model performance in production?',
      options: [
        'The algorithm is wrong',
        'Data leakage or preprocessing done incorrectly',
        'The model is too simple',
        'The computer is too slow',
      ],
      correctAnswer: 1,
      explanation:
        'Data leakage (using test data in training or having target information in features) is the most common cause of the train-test performance gap. Other common issues: preprocessing fitted on all data, temporal data not split correctly, or features computed using future information.',
      difficulty: 'intermediate' as const,
      category: 'Production',
    },
    {
      id: 3,
      question:
        "In a trading backtest, your strategy shows Sharpe ratio of 2.0 but loses money in live trading. What\'s the most likely issue?",
      options: [
        'Bad luck in live trading',
        'Lookahead bias, transaction costs not modeled, or survivorship bias',
        'The market changed overnight',
        'The strategy is perfect, just needs more time',
      ],
      correctAnswer: 1,
      explanation:
        'Backtest-to-live degradation in trading usually comes from: (1) Lookahead bias (using future data), (2) Ignoring transaction costs and slippage, (3) Survivorship bias (only backtesting surviving stocks), (4) Overfitting to specific historical periods, or (5) Implementation differences between backtest and live code.',
      difficulty: 'advanced' as const,
      category: 'Trading',
    },
    {
      id: 4,
      question:
        'Learning curves show train and validation errors are high and close together. What action should you take?',
      options: [
        'Get more data',
        "Increase model complexity - it's underfitting",
        "Reduce model complexity - it's overfitting",
        'The model is optimal',
      ],
      correctAnswer: 1,
      explanation:
        'High errors that are close together (small gap) indicate underfitting/high bias. The model is too simple to capture patterns. Solution: increase complexity (more features, deeper trees, more layers), reduce regularization, or try a more powerful model class.',
      difficulty: 'intermediate' as const,
      category: 'Learning Curves',
    },
    {
      id: 5,
      question:
        'When debugging a model, what should you check FIRST before diving into complex analysis?',
      options: [
        'Hyperparameter tuning',
        'Data quality - missing values, correct labels, distribution shifts',
        'Try a different algorithm',
        'Add more features',
      ],
      correctAnswer: 1,
      explanation:
        "80% of ML problems are data problems! Always check data quality first: missing values, incorrect labels, data leakage, train-test distribution match, feature preprocessing, and temporal ordering. Most 'model' problems are actually data problems in disguise.",
      difficulty: 'easy' as const,
      category: 'Best Practices',
    },
  ],
};
