import { MultipleChoiceQuestion } from '../../../types';

export const featureImportanceInterpretationMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'feature-importance-interpretation-mc-1',
      question:
        'What does it mean if "customer_id" has the highest feature importance in your model?',
      options: [
        'Customer ID is genuinely predictive',
        'The model is working correctly',
        'There is likely data leakage or overfitting',
        'You should use customer ID in production',
      ],
      correctAnswer: 2,
      explanation:
        'High importance for identifier columns like customer_id is a red flag indicating data leakage. The model has memorized specific customers rather than learning generalizable patterns. This will fail on new customers. Remove the feature and investigate why it had signal.',
    },
    {
      id: 'feature-importance-interpretation-mc-2',
      question:
        'Which feature importance method is model-agnostic and works with any ML model?',
      options: [
        'Built-in feature importance from Random Forest',
        'Coefficients from Logistic Regression',
        'Permutation importance',
        'Gini importance from Decision Trees',
      ],
      correctAnswer: 2,
      explanation:
        'Permutation importance is model-agnostic—it works with any model by shuffling feature values and measuring performance drop. Built-in importance is specific to tree models, and coefficients only apply to linear models.',
    },
    {
      id: 'feature-importance-interpretation-mc-3',
      question:
        'What do SHAP values provide that other importance methods do not?',
      options: [
        'Faster computation time',
        'Global feature ranking only',
        'Individual prediction explanations with direction and magnitude',
        'Works only with tree-based models',
      ],
      correctAnswer: 2,
      explanation:
        'SHAP values uniquely provide individual prediction explanations, showing how much each feature contributed to a specific prediction (with direction: +/-) and magnitude. This is essential for explaining decisions to stakeholders and regulatory compliance.',
    },
    {
      id: 'feature-importance-interpretation-mc-4',
      question:
        'If two features are highly correlated, what happens with permutation importance?',
      options: [
        'Both features show high importance',
        'Importance might be underestimated because shuffling one still leaves the other',
        'One feature gets all the importance',
        'The model will fail',
      ],
      correctAnswer: 1,
      explanation:
        "With highly correlated features, permutation importance can underestimate their importance. When you shuffle one feature, the correlated feature still provides similar information, so performance doesn't drop as much. This is a known limitation of permutation importance.",
    },
    {
      id: 'feature-importance-interpretation-mc-5',
      question:
        'A feature has high SHAP importance but low permutation importance. What might this indicate?',
      options: [
        'The feature is important for specific predictions but not globally',
        'There is a bug in the SHAP calculation',
        'You should remove the feature',
        'The model is overfitting',
      ],
      correctAnswer: 0,
      explanation:
        'High SHAP importance but low permutation importance can indicate that the feature is important for specific subsets of predictions (high local importance) but not important on average across all predictions (low global importance). This highlights the difference between local and global explanations.',
    },
  ];
