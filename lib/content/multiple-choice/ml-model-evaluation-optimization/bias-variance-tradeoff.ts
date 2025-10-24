export const biasVarianceTradeoffMultipleChoice = {
  title: 'Bias-Variance Tradeoff - Multiple Choice Questions',
  questions: [
    {
      id: 1,
      question:
        'A model has high training error and high test error that are close to each other. This indicates:',
      options: [
        'Perfect model - ready for production',
        'High variance (overfitting)',
        'High bias (underfitting)',
        'Data leakage',
      ],
      correctAnswer: 2,
      explanation:
        'High errors on both training and test (with small gap) indicates high bias/underfitting. The model is too simple to capture the underlying patterns. Solution: increase model complexity, add features, or reduce regularization.',
      difficulty: 'beginner' as const,
      category: 'Diagnosis',
    },
    {
      id: 2,
      question:
        'Which of the following techniques is most effective at reducing variance (overfitting)?',
      options: [
        'Adding more features',
        'Increasing model complexity',
        'Adding regularization (L1/L2)',
        'Removing training data',
      ],
      correctAnswer: 2,
      explanation:
        'Regularization (L1/L2) directly reduces variance by constraining model parameters and preventing overfitting to noise. Other variance reduction techniques include: getting more data, feature selection, dropout, and early stopping.',
      difficulty: 'beginner' as const,
      category: 'Solutions',
    },
    {
      id: 3,
      question:
        'As you increase the polynomial degree in polynomial regression from 1 to 15, what happens to bias and variance?',
      options: [
        'Both bias and variance increase',
        'Both bias and variance decrease',
        'Bias decreases, variance increases',
        'Bias increases, variance decreases',
      ],
      correctAnswer: 2,
      explanation:
        'Higher degree polynomials can fit more complex patterns (lower bias) but are more sensitive to training data fluctuations (higher variance). This is the fundamental bias-variance tradeoff: reducing one increases the other.',
      difficulty: 'intermediate' as const,
      category: 'Theory',
    },
    {
      id: 4,
      question:
        'You have learning curves where training and validation errors have converged but remain high. What should you do?',
      options: [
        'Get more data - it will help reduce the errors',
        "Reduce model complexity - it's overfitting",
        "Increase model complexity - it's underfitting",
        'The model is optimal, no changes needed',
      ],
      correctAnswer: 2,
      explanation:
        "Converged curves with high errors indicate high bias (underfitting). More data won't help since curves have converged. Need to increase model capacity: add features, increase complexity, or reduce regularization.",
      difficulty: 'advanced' as const,
      category: 'Learning Curves',
    },
    {
      id: 5,
      question:
        'What is the effect of increasing the regularization parameter λ (lambda) in Ridge regression?',
      options: [
        'Increases both bias and variance',
        'Increases bias, decreases variance',
        'Decreases bias, increases variance',
        'No effect on bias or variance',
      ],
      correctAnswer: 1,
      explanation:
        'Increasing λ strengthens regularization, constraining parameters more. This increases bias (model becomes simpler, may underfit) but decreases variance (less sensitive to training data, better generalization).',
      difficulty: 'intermediate' as const,
      category: 'Regularization',
    },
  ],
};
