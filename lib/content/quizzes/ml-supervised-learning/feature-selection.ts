/**
 * Discussion Questions for Feature Selection
 */

import { QuizQuestion } from '../../../types';

export const featureselectionQuiz: QuizQuestion[] = [
  {
    id: 'feature-selection-q1',
    question:
      'Compare filter, wrapper, and embedded feature selection methods. When would you use each approach? What are the computational and performance tradeoffs?',
    hint: 'Think about speed, accuracy, and whether the method considers feature interactions.',
    sampleAnswer:
      'Filter methods (correlation, ANOVA, mutual information) rank features by statistical scores independently of the model. Fast and scalable but ignore feature interactions - a feature useless alone might be valuable with others. Best for: high-dimensional data (>10K features), initial screening, model-agnostic selection. Wrapper methods (RFE, forward/backward selection) evaluate feature subsets using model performance. They capture interactions but are computationally expensive (O(n²) for n features). Best for: small-medium datasets, maximum accuracy, when interactions critical. Embedded methods (Lasso, tree importance) perform selection during model training. Balance speed and accuracy - faster than wrappers, capture some interactions. Best for: production pipelines, most practical choice. Tradeoffs: Filters fastest but least accurate. Wrappers most accurate but prohibitively slow for large p. Embedded methods practical middle ground. Recommendation: start with filter for initial pruning, use embedded for final selection, reserve wrappers for small datasets or competitions.',
    keyPoints: [
      'Filter: fast, statistical scores, ignores interactions',
      'Wrapper: slow, model-based, captures interactions',
      'Embedded: moderate speed, integrated with training',
      'Filter for high-dim screening',
      'Embedded for production pipelines',
    ],
  },
  {
    id: 'feature-selection-q2',
    question:
      'Explain data leakage in feature selection and how to prevent it. Why must feature selection be done within cross-validation?',
    hint: 'Consider what happens when you select features on the full dataset before splitting.',
    sampleAnswer:
      'Data leakage occurs when feature selection uses information from test set. Common mistake: (1) Select features on full dataset; (2) Split train/test; (3) Train model. Problem: feature selection "saw" test data, so test performance is overly optimistic. Features were chosen because they correlate with full dataset including test set - this inflates importance. Correct approach: Select features only on training data. In CV: feature selection must happen inside each fold, not before. Use Pipeline to ensure safety - it re-does selection in each fold automatically. Example impact: without proper CV, test accuracy might show 95% but true performance is 85%. Subtle but critical. Prevention: (1) Always use Pipeline for feature selection; (2) Never fit selector on test data; (3) In manual CV, re-select features in each fold. Real-world: leakage is one of most common mistakes causing production models to underperform. Test accuracy looks great in development, fails in production.',
    keyPoints: [
      'Leakage: feature selection uses test data',
      'Causes overly optimistic test performance',
      'Must select features only on training data',
      'Use Pipeline to prevent leakage automatically',
      'Re-select features in each CV fold',
    ],
  },
  {
    id: 'feature-selection-q3',
    question:
      'Discuss the advantages and limitations of tree-based feature importance vs permutation importance. When might feature importance be misleading?',
    hint: 'Think about correlated features and how importance is calculated.',
    sampleAnswer:
      "Tree-based importance (Gini/impurity decrease) measures average impurity reduction when feature used for splits. Fast (byproduct of training) but has biases: (1) Favors high-cardinality features (more split opportunities); (2) Correlated features split importance (both show low importance even if jointly important); (3) Can be misleading with extrapolation. Permutation importance shuffles feature values and measures performance drop. More reliable: directly measures predictive power, handles correlations better, consistent across models. But slower (requires multiple predictions). Misleading cases: (1) Correlated features - removing one doesn't hurt if others remain; (2) Feature interactions - permuting one might not matter if interaction partner still works; (3) Non-linear relationships - importance might underestimate complex patterns. Best practice: Use tree importance for quick screening, permutation importance for reliable ranking, SHAP values for interpretability. Don't over-interpret absolute values - focus on relative rankings. Always combine with domain knowledge.",
    keyPoints: [
      'Tree importance: fast, biased toward high-cardinality',
      'Permutation importance: reliable, directly measures prediction power',
      'Correlated features can split importance',
      'Permutation more robust but slower',
      'Use SHAP for detailed interpretability',
    ],
  },
];
