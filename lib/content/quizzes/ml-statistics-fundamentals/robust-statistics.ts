/**
 * Quiz questions for Robust Statistics section
 */

export const robuststatisticsQuiz = [
  {
    id: 'q1',
    question:
      'Why is the median more robust than the mean? Give an example where this matters in ML.',
    hint: 'Think about the breakdown point and influence of extreme values.',
    sampleAnswer:
      '**Median robustness**: Median has 50% breakdown point - up to 50% of data can be arbitrarily large without affecting it. Mean has 0% breakdown - one extreme value can pull it arbitrarily far. **Example**: Feature scaling with outliers. User age: [20, 22, 25, 28, 999] (data error). Mean = 218.8 (useless!), Median = 25 (sensible). **In ML**: Use median for outlier-resistant: (1) Feature scaling (robust scaling), (2) Missing value imputation, (3) Evaluation metrics (median absolute error), (4) Anomaly detection baselines. **Tradeoff**: Median less efficient than mean when data is truly normal.',
    keyPoints: [
      'Median: 50% breakdown point',
      'Mean: 0% breakdown (one outlier affects)',
      'Use median for feature scaling with outliers',
      'Robust imputation and metrics',
      'Less efficient under normality',
    ],
  },
  {
    id: 'q2',
    question:
      'When would you choose RANSAC over Huber regression? What are the tradeoffs?',
    hint: 'Consider the percentage of outliers and computational cost.',
    sampleAnswer:
      '**Huber**: Down-weights outliers but includes all data. Good for <10% outliers. Fast, smooth objective. **RANSAC**: Fits to random subsets, finds best inlier set. Good for 10-50% outliers. Slower, more robust. **Choose RANSAC when**: (1) High outlier percentage (>10%), (2) Outliers are gross errors (very far off), (3) Need to identify inliers explicitly. **Choose Huber when**: (1) Few outliers (<10%), (2) Computational efficiency matters, (3) All data somewhat informative. **Example**: RANSAC for image stitching (many wrong matches), Huber for financial regression (occasional fat-tail events).',
    keyPoints: [
      'Huber: <10% outliers, fast',
      'RANSAC: 10-50% outliers, slower',
      'RANSAC explicitly identifies inliers',
      'Huber smooth, RANSAC robust',
      'Choice depends on outlier proportion',
    ],
  },
  {
    id: 'q3',
    question:
      'You detect outliers using Z-score (>3σ). What assumption does this make, and when might it fail?',
    hint: 'Think about the underlying distribution assumption.',
    sampleAnswer:
      '**Z-score assumption**: Data is approximately normal. Under normality, ~99.7% within 3σ, so |Z|>3 is rare → outlier. **Fails when**: (1) **Heavy-tailed distribution**: Finance, web traffic. Many legitimate 3σ+ events (not outliers!). Z-score overdetects. (2) **Skewed data**: Income, property values. Upper tail naturally extends far. (3) **Multimodal**: Multiple clusters, legitimate values far from overall mean. **Better methods for non-normal**: (1) IQR method (distribution-free), (2) Isolation Forest (no assumptions), (3) Domain-specific thresholds. **Use Z-score only if**: Data approximately normal or quick rough check.',
    keyPoints: [
      'Z-score assumes normality',
      'Fails for heavy-tailed, skewed data',
      'Overdetects in financial data',
      'Use IQR, Isolation Forest instead',
      'Distribution-free methods more robust',
    ],
  },
];
