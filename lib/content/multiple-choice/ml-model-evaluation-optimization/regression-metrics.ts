import { MultipleChoiceQuestion } from '../../../types';

export const regressionMetricsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'regression-metrics-mc-1',
    question:
      'You have two house price prediction models with the following performance: Model A has MAE=$10,000 and RMSE=$15,000. Model B has MAE=$10,000 and RMSE=$10,500. What can you conclude?',
    options: [
      'Both models have identical performance',
      'Model A has more uniform errors, while Model B has some large outliers',
      'Model B has more uniform errors, while Model A has some large outliers',
      'The RMSE values are invalid since RMSE should equal MAE',
    ],
    correctAnswer: 2,
    explanation:
      'RMSE is always ≥ MAE, with the ratio RMSE/MAE revealing error distribution. Model A has ratio 1.5 ($15k/$10k) while Model B has ratio 1.05 ($10.5k/$10k). A higher ratio indicates presence of outliers or high variance in errors. Model A has more variable errors (some large outliers), while Model B has more consistent errors.',
  },
  {
    id: 'regression-metrics-mc-2',
    question:
      'A model achieves R² = 0.85 on a house price prediction task. What does this mean?',
    options: [
      'The model is 85% accurate',
      'The model predicts prices within 85% of the true value on average',
      'The model explains 85% of the variance in house prices',
      'The model has 15% error rate',
    ],
    correctAnswer: 2,
    explanation:
      'R² (R-squared) represents the proportion of variance in the target variable that is explained by the model. R²=0.85 means the model explains 85% of the variance in house prices compared to simply predicting the mean. It does NOT mean 85% accuracy or that predictions are within 85% of true values.',
  },
  {
    id: 'regression-metrics-mc-3',
    question:
      'Which metric would be MOST appropriate for evaluating a medical dosage prediction model where being off by 2× is much worse than twice the error of being off by 1×?',
    options: [
      'MAE (Mean Absolute Error)',
      'MAPE (Mean Absolute Percentage Error)',
      'RMSE (Root Mean Squared Error)',
      'R-squared',
    ],
    correctAnswer: 2,
    explanation:
      "RMSE squares the errors before averaging, which means large errors are penalized disproportionately—exactly what we want when large errors are catastrophically bad. RMSE penalizes a 2× error much more than twice as much as a 1× error due to the quadratic penalty. MAE treats all errors equally, MAPE has issues with scale, and R² doesn't directly address error magnitude.",
  },
  {
    id: 'regression-metrics-mc-4',
    question:
      'You want to compare the performance of a sales prediction model across different product categories that have vastly different sales volumes (e.g., $100 vs $100,000). Which metric is MOST appropriate?',
    options: [
      'MAE (Mean Absolute Error)',
      'MSE (Mean Squared Error)',
      'RMSE (Root Mean Squared Error)',
      'MAPE (Mean Absolute Percentage Error) or R²',
    ],
    correctAnswer: 3,
    explanation:
      'MAPE and R² are scale-independent metrics, making them suitable for comparing across different scales. MAE, MSE, and RMSE are scale-dependent—a $1,000 error would seem large for a $100 product but small for a $100,000 product. MAPE expresses errors as percentages, allowing fair comparison across scales (assuming no zero or near-zero values).',
  },
  {
    id: 'regression-metrics-mc-5',
    question:
      'Your model has MAPE = 15% for predicting product demand. For which scenario would MAPE be INAPPROPRIATE?',
    options: [
      'Predicting demand for products with sales ranging from 1,000 to 10,000 units',
      'Predicting demand for products where some products have zero sales (new launches)',
      'Predicting demand across different product categories with different scales',
      'Communicating performance to business stakeholders',
    ],
    correctAnswer: 1,
    explanation:
      'MAPE fails catastrophically when true values are zero (division by zero) or near-zero (extremely high percentage errors that dominate the metric). Products with zero sales make MAPE undefined. MAPE works fine for positive values away from zero and is excellent for scale-independent comparison and business communication.',
  },
];
