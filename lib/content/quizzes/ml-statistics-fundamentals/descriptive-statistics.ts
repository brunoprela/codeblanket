/**
 * Quiz questions for Descriptive Statistics section
 */

export const descriptivestatisticsQuiz = [
  {
    id: 'q1',
    question:
      'When analyzing a dataset for machine learning, you find that the mean is significantly higher than the median. What does this tell you about your data, and how might this affect your choice of models and preprocessing steps?',
    hint: 'Think about the shape of the distribution and which models make assumptions about normality.',
    sampleAnswer:
      'When mean > median, the data is right-skewed with a long tail of high values (outliers pulling the mean up). This has several implications: (1) The mean is not representative of typical values - median is better for central tendency. (2) Linear models may struggle because outliers have high leverage. (3) Distance-based algorithms (KNN, K-means) will be distorted by the scale difference. (4) Tree-based models (Random Forest, XGBoost) will handle this better as they use splits, not distances. Preprocessing steps to consider: (a) Log transformation to reduce skewness, (b) Outlier removal or capping (winsorization), (c) Robust scaling using median and IQR instead of mean and std, (d) Check if skewness is meaningful (e.g., income is naturally skewed) or due to data errors. Always visualize with histograms and box plots before deciding.',
    keyPoints: [
      'Right-skewed distribution indicated by mean > median',
      'Outliers pull the mean up, median is more robust',
      'Linear and distance-based models sensitive to skewness',
      'Consider log transformation to normalize',
      'Tree-based models handle skewness better',
      'Use robust scaling (median/IQR) instead of standard scaling',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain the difference between population variance (using n) and sample variance (using n-1). Why does the n-1 correction matter, and when should you use each in machine learning?',
    hint: 'Consider bias in estimation and whether you have the full population or just a sample.',
    sampleAnswer:
      "Population variance uses n: σ² = Σ(x-μ)²/n, assuming you have ALL data from the population. Sample variance uses n-1: s² = Σ(x-x̄)²/(n-1), which is Bessel\'s correction for unbiased estimation. The reason: when we estimate variance from a sample, we first estimate the mean (x̄), which is closer to our sample points than the true population mean (μ). This makes our squared deviations systematically smaller, underestimating the variance. Dividing by n-1 instead of n corrects this bias. In ML: (1) Almost always use sample variance (n-1) because we never have the full population, only samples. (2) StandardScaler in sklearn uses n-1 by default. (3) numpy uses ddof=0 by default (population), so specify ddof=1 for samples. (4) For large datasets (n > 30), the difference becomes negligible. (5) For small datasets, using n instead of n-1 will make you overconfident (underestimate uncertainty). The correction is crucial for proper statistical inference, confidence intervals, and hypothesis testing.",
    keyPoints: [
      'Population variance (n): when you have ALL data',
      'Sample variance (n-1): corrects bias from estimating mean',
      "Bessel\'s correction makes estimator unbiased",
      'ML almost always uses sample variance (n-1)',
      'Difference negligible for large n',
      'Critical for proper uncertainty quantification',
    ],
  },
  {
    id: 'q3',
    question:
      "You're analyzing two features for a machine learning model: income (ranging from $20k to $500k) and age (ranging from 20 to 70 years). The coefficient of variation (CV) for income is 45% and for age is 18%. What does this tell you, and how should you handle these features before training?",
    hint: 'Think about relative variability and feature scaling.',
    sampleAnswer:
      "The coefficient of variation (CV = std/mean × 100%) measures relative variability independent of units. A CV of 45% for income means income is highly variable relative to its mean - there's substantial spread in income levels. An 18% CV for age means age is more consistent relative to its mean. Key insights: (1) Income has 2.5x more relative variability than age, making it potentially more informative. (2) Without scaling, income will dominate distance calculations (e.g., in KNN) simply due to its larger scale, not its importance. (3) Gradient-based optimization (neural networks, logistic regression) will struggle with features at different scales. Handling: (a) Standard Scaling: Transform both to mean=0, std=1, putting them on equal footing. (b) MinMax Scaling: Scale to [0,1] range. (c) Robust Scaling: Use median and IQR if there are outliers. (d) After scaling, both features contribute equally to distance metrics. (e) For tree-based models (Random Forest), scaling is less critical as they use splits, not distances. (f) Consider interaction features: income × age might be informative. Always check distributions with histograms and consider transformations (log for income if right-skewed) before scaling.",
    keyPoints: [
      'CV measures relative variability (scale-independent)',
      'Income has 2.5x higher relative variability',
      'Unscaled features bias distance-based algorithms',
      'Standard scaling puts features on equal footing',
      'Tree-based models less sensitive to scale',
      'Check distributions before choosing scaling method',
    ],
  },
];
