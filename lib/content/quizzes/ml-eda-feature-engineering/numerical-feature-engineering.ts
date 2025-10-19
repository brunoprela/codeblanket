/**
 * Quiz questions for Numerical Feature Engineering section
 */

export const numericalfeatureengineeringQuiz = [
  {
    id: 'q1',
    question:
      'Explain when and why you would use StandardScaler vs MinMaxScaler vs RobustScaler. Provide specific use cases for each.',
    sampleAnswer:
      'STANDARDSCALER (Z-score normalization): (x - mean) / std. Results in mean=0, std=1. WHEN: Features approximately normally distributed, most common default choice. USE CASES: Linear models (linear/logistic regression), SVM, neural networks, PCA (required!), k-means clustering. WHY: Preserves distribution shape, handles negative values, well-understood statistically. PROS: Standard practice, works with gradient descent. CONS: Sensitive to outliers (mean and std affected by extremes). MINMAXSCALER: (x - min) / (max - min). Results in range [0, 1]. WHEN: Need specific bounded range, neural networks with sigmoid/tanh activation. USE CASES: Image pixel normalization (already 0-255), neural networks with bounded activations, when zero has meaning, features that should stay positive. WHY: Bounded output guaranteed, preserves relationships. PROS: Interpretable range, preserves zero. CONS: VERY sensitive to outliers (min/max shift entire distribution). ROBUSTSCALER: (x - median) / IQR. Uses median and interquartile range. WHEN: Data has significant outliers that are valid (not errors). USE CASES: Financial data (extreme values common), sensor data with anomalies, after deciding outliers are valid, when StandardScaler fails due to outliers. WHY: Median and IQR robust to outliers, stable transformation. PROS: Handles outliers gracefully, stable. CONS: Less common, can lose information about extreme values. DECISION TREE: Check for outliers → Many valid outliers? Use RobustScaler. Need bounded range [0,1]? Use MinMaxScaler. Otherwise → Use StandardScaler (default). CRITICAL: Tree-based models (Random Forest, XGBoost) dont need ANY scaling!',
    keyPoints: [
      'StandardScaler: normal data, most common, mean=0 std=1',
      'MinMaxScaler: bounded range [0,1], very sensitive to outliers',
      'RobustScaler: data with outliers, uses median/IQR',
      'StandardScaler default unless specific reason for others',
      'Tree models dont need scaling',
      'Always fit on train, apply to test',
    ],
  },
  {
    id: 'q2',
    question:
      'Discuss the purpose of mathematical transformations (log, sqrt, Box-Cox) in feature engineering. How do you choose which transformation to apply?',
    sampleAnswer:
      'Mathematical transformations modify feature distributions to improve model performance, primarily for linear models. PURPOSE: (1) REDUCE SKEWNESS: Make distributions more symmetric/normal. Linear models assume normal distributions. (2) STABILIZE VARIANCE: Transform to constant variance across range (homoscedasticity). (3) LINEARIZE RELATIONSHIPS: Make non-linear relationships linear. (4) HANDLE MULTIPLICATIVE EFFECTS: Convert to additive (important for linear models). TRANSFORMATIONS: LOG (log(x) or log(1+x)): For RIGHT-SKEWED data (long tail on right). WHEN: Income, prices, counts, anything following power law. EFFECT: Compresses high values, expands low values. HANDLES: Multiplicative relationships → additive. EXAMPLE: Income distribution (few billionaires pull mean right). SQUARE ROOT (sqrt(x)): Milder than log, for moderate right skew. WHEN: Count data, moderate skewness. EFFECT: Less aggressive compression than log. EXAMPLE: Number of purchases, website visits. CUBE ROOT (cbrt(x)): Can handle negative values unlike sqrt. WHEN: Data with negative values, mild skewness. BOX-COX: Automatically finds optimal power transformation (x^λ). WHEN: Not sure which transformation, want data-driven choice. FINDS: Best λ parameter (λ=0 is log, λ=0.5 is sqrt, λ=1 is no transform). LIMITATION: Requires x > 0. YEO-JOHNSON: Like Box-Cox but handles negative values. WHEN: Data has negatives, want automatic transformation. CHOOSING PROCESS: (1) Plot histogram - identify skewness direction. (2) Right-skewed (mean > median, long right tail) → Try log or sqrt. (3) Left-skewed (mean < median) → Try square or exponential. (4) Uncertain → Use Box-Cox/Yeo-Johnson (automatic). (5) Compare skewness after transformation - closest to 0 is best. (6) Check if model performance improves with transformation. WHEN NOT TO TRANSFORM: Tree-based models handle non-linearity naturally - transformation often unnecessary. Neural networks with enough capacity learn transformations.',
    keyPoints: [
      'Transformations reduce skewness and linearize relationships',
      'Log: right-skewed data (income, prices)',
      'Sqrt: moderate skewness, count data',
      'Box-Cox: automatic transformation selection',
      'Choose based on skewness direction and model type',
      'Tree models dont need transformations',
    ],
  },
  {
    id: 'q3',
    question:
      'Explain the difference between equal-width binning and equal-frequency binning. When would you use each approach, and what are the trade-offs?',
    sampleAnswer:
      'Binning converts continuous features to categorical by dividing range into intervals. EQUAL-WIDTH BINNING: Divides range into bins of equal size. FORMULA: bin_width = (max - min) / n_bins. Each bin covers same value range. RESULT: Bins can have very different counts. EXAMPLE: Income 0-100K divided into 5 bins: [0-20K], [20-40K], [40-60K], [60-80K], [80-100K]. PROBLEM: If most people earn 20-40K, that bin has 80% of data, others nearly empty. EQUAL-FREQUENCY BINNING (Quantile Binning): Divides data into bins with equal number of samples. FORMULA: Uses quantiles (percentiles). Each bin has ~same count. RESULT: Bin ranges can be very different. EXAMPLE: Same income, 5 equal-frequency bins might be: [0-22K], [22K-28K], [28K-35K], [35K-45K], [45K-100K]. Each bin has 20% of samples, but ranges vary wildly. WHEN TO USE EACH: USE EQUAL-WIDTH WHEN: (1) Values uniformly distributed across range. (2) Interpretability matters (nice round numbers). (3) Domain knowledge suggests natural breakpoints. (4) Comparing bins across different datasets. (5) Bins should represent equal value ranges. EXAMPLE: Temperature ranges (0-20C, 20-40C logical). USE EQUAL-FREQUENCY WHEN: (1) Distribution is skewed (most common case!). (2) Want statistical power in each bin (need samples). (3) Using bin as categorical feature (need balance). (4) Detecting patterns at different quantiles. (5) Ranking/percentile matters. EXAMPLE: Customer segmentation by spending (bottom 20%, next 20%, etc). TRADE-OFFS: Equal-width: Interpretable ranges, but can create empty/sparse bins. Equal-frequency: Balanced sample sizes, but ranges can be unintuitive. BEST PRACTICE: Try both, evaluate model performance. Often equal-frequency performs better (avoids sparse bins). Consider domain-driven bins (custom ranges based on business logic).',
    keyPoints: [
      'Equal-width: equal value ranges, can have unbalanced counts',
      'Equal-frequency: equal sample counts, can have varying ranges',
      'Equal-width better for interpretability and uniform distributions',
      'Equal-frequency better for skewed data and statistical power',
      'Often equal-frequency performs better in practice',
      'Consider custom domain-driven bins',
    ],
  },
];
