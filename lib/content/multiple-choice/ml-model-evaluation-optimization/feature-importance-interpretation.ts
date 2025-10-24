export const featureImportanceInterpretationMultipleChoice = {
  title: 'Feature Importance & Interpretation - Multiple Choice Questions',
  questions: [
    {
      id: 1,
      question:
        'What is the main limitation of built-in feature importance from Random Forest models?',
      options: [
        "It's too slow to compute",
        "It's biased toward high-cardinality features and doesn't work for non-tree models",
        'It only works for classification',
        'It requires labeled data',
      ],
      correctAnswer: 1,
      explanation:
        "Built-in tree importance (Gini/impurity reduction) is biased toward features with more unique values (high cardinality) and can be misleading with correlated features. It's also specific to tree models. Permutation importance and SHAP are more robust alternatives.",
      difficulty: 'intermediate' as const,
      category: 'Limitations',
    },
    {
      id: 2,
      question: 'How does permutation importance determine feature importance?',
      options: [
        'By removing the feature entirely',
        'By shuffling the feature values and measuring performance drop',
        'By looking at feature correlations',
        'By counting how many times a feature is used',
      ],
      correctAnswer: 1,
      explanation:
        "Permutation importance randomly shuffles each feature's values (breaking its relationship with target) and measures how much model performance drops. Large drop = important feature. It's model-agnostic and accounts for feature interactions.",
      difficulty: 'beginner' as const,
      category: 'Methods',
    },
    {
      id: 3,
      question:
        'What key advantage do SHAP values provide over other feature importance methods?',
      options: [
        'They are faster to compute',
        'They provide directional explanations for individual predictions with theoretical guarantees',
        'They only work for simple models',
        "They don't require a trained model",
      ],
      correctAnswer: 1,
      explanation:
        "SHAP values show both magnitude and direction of each feature's contribution to individual predictions, based on game theory (Shapley values). They have theoretical guarantees (consistency, local accuracy) and can explain individual predictions, not just global importance.",
      difficulty: 'advanced' as const,
      category: 'SHAP',
    },
    {
      id: 4,
      question:
        'Your model shows feature X as most important, but domain experts disagree. What should you investigate first?',
      options: [
        'Ignore the experts - trust the model',
        'Check for data leakage, feature correlations, and data quality issues',
        'Remove feature X immediately',
        'Retrain with a different algorithm',
      ],
      correctAnswer: 1,
      explanation:
        'Discrepancy between model importance and domain knowledge is a red flag. Investigate: (1) Data leakage (X contains target information), (2) X correlated with truly important feature (proxy), (3) Data quality (X clean while important features have missing values), (4) Non-linear relationships model captures.',
      difficulty: 'intermediate' as const,
      category: 'Debugging',
    },
    {
      id: 5,
      question:
        'For a production system requiring GDPR Article 22 compliance (right to explanation), which interpretation method is most appropriate?',
      options: [
        'No explanation needed',
        'Built-in tree importance only',
        'SHAP values with individual prediction explanations',
        'Just show the model accuracy',
      ],
      correctAnswer: 2,
      explanation:
        'GDPR Article 22 requires the right to explanation for automated decisions. SHAP values provide rigorous, individual prediction explanations showing which features influenced each decision and by how much. Essential for regulatory compliance and building user trust.',
      difficulty: 'advanced' as const,
      category: 'Compliance',
    },
  ],
};
