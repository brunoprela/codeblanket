/**
 * Discussion Questions for Random Forests
 */

import { QuizQuestion } from '../../../types';

export const randomforestsQuiz: QuizQuestion[] = [
  {
    id: 'random-forests-q1',
    question:
      'Explain how Random Forests reduce variance compared to single decision trees. What roles do bagging and random feature selection play?',
    hint: 'Think about why averaging multiple models helps and how randomness decorrelates predictions.',
    sampleAnswer:
      'Random Forests reduce variance through two randomness mechanisms. Bagging (bootstrap aggregating) trains each tree on a random sample with replacement, creating diverse trees that make different errors. Averaging predictions reduces variance: if trees are independent, variance decreases by factor of 1/n_trees. However, bagged trees from same data would be highly correlated (similar splits). Random feature selection solves this by considering only random subset of features at each split (sqrt(p) for classification), forcing trees to use different features. This decorrelates trees, making ensemble more powerful. Final prediction: classification uses majority vote, regression uses mean. Result: Random Forest maintains low bias of deep trees but dramatically reduces variance through diversified ensemble. This is why RF rarely overfits even with many trees - more trees just reduces variance further without increasing bias.',
    keyPoints: [
      'Bagging: train trees on bootstrap samples',
      'Averaging diverse predictions reduces variance',
      'Random features decorrelate trees',
      'More trees → lower variance, no overfitting',
      'Maintains low bias of individual trees',
    ],
  },
  {
    id: 'random-forests-q2',
    question:
      'What is Out-of-Bag (OOB) error and why is it useful? How does it compare to traditional cross-validation?',
    hint: "Consider what data each tree doesn't see during training.",
    sampleAnswer:
      "Out-of-Bag (OOB) error provides free validation without separate test set. Bootstrap sampling selects ~63% of data for each tree, leaving ~37% out-of-bag. For each sample, we average predictions from trees that didn't see it during training - this is OOB prediction. OOB error is computed as accuracy of OOB predictions across all samples. Advantages: (1) No separate validation set needed; (2) Uses all data for training and testing; (3) Nearly unbiased estimate of test error; (4) Computationally free (byproduct of training). Comparison to CV: OOB is faster (single model fit vs k fits for k-fold CV), but CV is slightly more accurate and works with any model. OOB is specific to bagging/RF. For Random Forest, OOB error closely approximates test error, making it excellent for quick model assessment and hyperparameter tuning without splitting data.",
    keyPoints: [
      'Bootstrap uses ~63% data, ~37% out-of-bag',
      "OOB prediction: average from trees that didn't see sample",
      'OOB error ≈ test error',
      'Free validation (no data splitting)',
      'Faster than CV but less precise',
    ],
  },
  {
    id: 'random-forests-q3',
    question:
      'Discuss the interpretability tradeoff: Random Forests are less interpretable than single trees but provide feature importance. How should feature importance be used and what are its limitations?',
    hint: 'Think about what feature importance measures and when it might be misleading.',
    sampleAnswer:
      "Random Forest sacrifices the flowchart interpretability of single trees but gains robust feature importance: average decrease in impurity when feature is used for splitting, averaged across all trees. This is more stable than single-tree importance. Uses: (1) Feature selection - drop low-importance features; (2) Domain insight - understand what drives predictions; (3) Feature engineering guidance. Limitations: (1) Biased toward high-cardinality features (more split opportunities); (2) Correlated features split importance between them (both show low importance even if jointly important); (3) Shows importance not direction (unlike regression coefficients); (4) Can be misleading with extrapolation. Best practices: Use permutation importance (shuffle feature, measure performance drop) for more robust estimates. For true interpretability, use SHAP values which explain individual predictions. Bottom line: RF importance is great for feature ranking but don't over-interpret absolute values. Combine with domain knowledge.",
    keyPoints: [
      'Feature importance: average impurity decrease',
      'More stable than single-tree importance',
      'Biased toward high-cardinality features',
      'Correlated features split importance',
      'Use permutation importance or SHAP for robustness',
    ],
  },
];
