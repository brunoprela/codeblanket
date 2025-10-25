export const regressionMetricsMultipleChoice = {
  title: 'Regression Metrics - Multiple Choice Questions',
  questions: [
    {
      id: 1,
      question: 'Which regression metric is most sensitive to outliers?',
      options: [
        'MAE (Mean Absolute Error)',
        'RMSE (Root Mean Squared Error)',
        'Median Absolute Error',
        'MAPE (Mean Absolute Percentage Error)',
      ],
      correctAnswer: 1,
      explanation:
        'RMSE squares errors before averaging, which heavily penalizes large errors. A single large outlier error will dominate the RMSE. MAE treats all errors equally, making it robust to outliers.',
      difficulty: 'intermediate' as const,
      category: 'Metrics',
    },
    {
      id: 2,
      question: 'An R² (R-squared) value of 0.85 means:',
      options: [
        'The model is 85% accurate',
        'The model explains 85% of the variance in the target variable',
        '85% of predictions are correct',
        'The error is 15%',
      ],
      correctAnswer: 1,
      explanation:
        'R² represents the proportion of variance in the dependent variable that is explained by the independent variables. R²=0.85 means the model explains 85% of the variance, with 15% remaining unexplained.',
      difficulty: 'beginner' as const,
      category: 'Interpretation',
    },
    {
      id: 3,
      question:
        'When would you prefer MAE over RMSE as your evaluation metric?',
      options: [
        'When you want to heavily penalize large errors',
        'When you want all errors to be weighted equally and be robust to outliers',
        'When the target variable is categorical',
        'When you want the metric in different units than the target',
      ],
      correctAnswer: 1,
      explanation:
        "MAE treats all errors equally (linear penalty) and is robust to outliers. Use it when you don't want outliers to dominate your metric or when all errors should be weighted equally regardless of magnitude.",
      difficulty: 'intermediate' as const,
      category: 'Metric Selection',
    },
    {
      id: 4,
      question:
        'What is a major limitation of MAPE (Mean Absolute Percentage Error)?',
      options: [
        "It can't be used for regression problems",
        "It\'s undefined when actual values are zero and asymmetric (penalizes over-predictions more)",
        "It's too slow to compute",
        'It requires more data than other metrics',
      ],
      correctAnswer: 1,
      explanation:
        "MAPE divides by actual values, so it's undefined when y=0. It\'s also asymmetric: a $50 error on $100 (50%) is treated worse than a $50 error on $200 (25%), even though absolute error is the same.",
      difficulty: 'advanced' as const,
      category: 'Limitations',
    },
    {
      id: 5,
      question:
        'If your model has Train RMSE=10 and Test RMSE=15, and baseline RMSE (predicting mean)=40, how would you interpret this?',
      options: [
        'The model is terrible and should be discarded',
        'The model is excellent with no issues',
        'The model shows some overfitting but significantly beats baseline (62.5% improvement)',
        'The model is underfitting',
      ],
      correctAnswer: 2,
      explanation:
        'Test RMSE=15 is much better than baseline=40 (62.5% reduction), showing the model learns useful patterns. The train/test gap (10→15) indicates some overfitting but not severe. Overall, this is a decent model with room for improvement.',
      difficulty: 'advanced' as const,
      category: 'Interpretation',
    },
  ],
};
