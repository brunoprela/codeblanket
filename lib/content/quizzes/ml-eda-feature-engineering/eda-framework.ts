/**
 * Quiz questions for EDA Framework section
 */

export const edaframeworkQuiz = [
  {
    id: 'q1',
    question:
      'Why is Exploratory Data Analysis (EDA) considered a critical first step in machine learning projects? Discuss what can go wrong if EDA is skipped or done superficially.',
    hint: 'Think about data quality, feature understanding, and downstream model performance.',
    sampleAnswer:
      'EDA is critical because it prevents the "garbage in, garbage out" problem and ensures you truly understand your data before building models. Without proper EDA, several issues arise: (1) Data quality problems go undetected - missing values, duplicates, incorrect dtypes, and outliers can severely degrade model performance. (2) Poor feature understanding - you might include irrelevant features or miss important ones, leading to suboptimal models. (3) Inappropriate model selection - without understanding data distributions and relationships, you might choose models poorly suited to the problem. (4) Missed feature engineering opportunities - EDA reveals transformations (log, scaling) and new feature ideas that dramatically improve performance. (5) Biased evaluation - if you don\'t understand data distribution, you might create unrealistic train/test splits. (6) Business misalignment - superficial EDA means you miss domain inconsistencies and might build technically sound but business-irrelevant models. Real-world example: A model might achieve 95% accuracy but fail because EDA would have revealed that 95% of samples are one class (class imbalance), making the metric misleading. EDA is detective work that saves enormous time and resources downstream.',
    keyPoints: [
      'Prevents data quality issues from corrupting models',
      'Enables informed feature engineering and selection',
      'Guides appropriate model selection based on data characteristics',
      'Reveals data distributions critical for proper evaluation',
      'Builds domain understanding and business alignment',
      'Catches issues early before expensive modeling iterations',
    ],
  },
  {
    id: 'q2',
    question:
      'Walk through the complete EDA workflow for a new machine learning project. What are the key stages, and what specific analyses would you perform at each stage?',
    sampleAnswer:
      'A comprehensive EDA workflow has 8 key stages: (1) UNDERSTAND THE PROBLEM: Define business objective, identify target variable, define success metrics, understand constraints, and document data sources. This ensures alignment before analysis. (2) INITIAL INSPECTION: Load data, check shape (rows × columns), examine first/last rows, review column names and data types, check memory usage, and generate basic statistics (describe()). This gives a high-level overview. (3) DATA QUALITY ASSESSMENT: Analyze missing values (counts and patterns), check for duplicates, validate data types, identify potential outliers, and check consistency. This catches data issues early. (4) UNIVARIATE ANALYSIS: Examine distribution of each feature independently - histograms for continuous, bar charts for categorical, summary statistics, and skewness/kurtosis checks. Understand each feature individually. (5) BIVARIATE ANALYSIS: Analyze relationships between features and target - scatter plots, correlation coefficients, box plots by category, and statistical tests. Identify predictive features. (6) MULTIVARIATE ANALYSIS: Examine feature interactions - correlation matrices, pair plots, dimensionality reduction visualizations (PCA), and multicollinearity checks. Understand complex relationships. (7) DOMAIN VALIDATION: Validate findings against domain knowledge, identify data errors or anomalies, check business rule violations, and analyze temporal patterns. Ensure business relevance. (8) DOCUMENTATION: Document all findings, data quality issues, transformations needed, feature engineering opportunities, and share insights. Create reproducible analysis. Each stage informs the next and you often iterate back.',
    keyPoints: [
      'Start with problem understanding and data source documentation',
      'Initial inspection provides high-level overview of data structure',
      'Data quality assessment catches errors early',
      'Univariate → Bivariate → Multivariate analysis builds understanding progressively',
      'Domain validation ensures business relevance',
      'Documentation creates reproducible, shareable insights',
    ],
  },
  {
    id: 'q3',
    question:
      'EDA often reveals data quality issues. Describe the main categories of data quality problems you should look for, and explain how each can impact machine learning models if left unaddressed.',
    hint: 'Consider missing values, duplicates, incorrect types, outliers, and inconsistencies.',
    sampleAnswer:
      "Data quality issues fall into several critical categories: (1) MISSING VALUES: Can range from random (easy to handle) to systematic (introducing bias). If not addressed, most ML algorithms will fail or drop valuable samples. Impact: Reduced sample size, biased models if missing-not-at-random, and algorithm failures. (2) DUPLICATES: Exact duplicate rows can artificially inflate model confidence and cause data leakage if same data appears in train and test sets. Impact: Overly optimistic evaluation metrics and potential overfitting. (3) INCORRECT DATA TYPES: Numeric fields stored as strings can't be used in models; dates stored as strings lose temporal information. Impact: Features become unusable or lose critical information. (4) OUTLIERS: Extreme values can be real (need to handle) or errors (need to remove). Impact: Models sensitive to outliers (linear regression, KNN) perform poorly; statistics like mean become misleading. (5) INCONSISTENCIES: Different formats for same entity (USA vs United States), varying scales (dollars vs cents), or impossible values (age = 150). Impact: Creates artificial categories, scaling issues, and nonsensical predictions. (6) IMBALANCED DATA: Severe class imbalance (1% positive class) without recognition leads to models that ignore minority class. Impact: High accuracy but zero recall on rare but important cases. (7) TEMPORAL ISSUES: Data leakage from future information or concept drift over time. Impact: Unrealistically good validation performance that doesn't generalize. Each issue requires specific handling strategies discovered during EDA.",
    keyPoints: [
      'Missing values reduce sample size and can introduce bias',
      'Duplicates inflate metrics and cause data leakage',
      'Incorrect dtypes make features unusable',
      'Outliers skew models and statistics',
      'Inconsistencies create artificial patterns',
      'Imbalance causes models to ignore rare classes',
      'Temporal issues lead to data leakage and drift',
    ],
  },
];
