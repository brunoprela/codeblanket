/**
 * Quiz questions for Feature Engineering Fundamentals section
 */

export const featureengineeringfundamentalsQuiz = [
  {
    id: 'q1',
    question:
      'Explain what data leakage is in feature engineering and provide three specific examples. Why is data leakage one of the most dangerous mistakes in machine learning?',
    hint: "Think about using information that wouldn't be available at prediction time.",
    sampleAnswer:
      'Data leakage occurs when information from outside the training dataset (particularly from the future or from the target variable itself) is used to create features, leading to unrealistically optimistic performance that doesn\'t generalize. EXAMPLES OF DATA LEAKAGE: (1) TEMPORAL LEAKAGE: Creating "days_until_customer_churns" feature when predicting churn. At prediction time, you don\'t know when/if they\'ll churn! Correct: Use "days_since_last_purchase", "engagement_trend". (2) TARGET LEAKAGE: Using "hospital_readmission_count" to predict first admission outcome. Readmissions happen after the target event! Correct: Only use information before first admission. (3) TRAIN/TEST CONTAMINATION: Computing mean income from entire dataset (including test set) and using it for imputation. Test data leaks into training. Correct: Compute mean from training set only, apply to test. (4) PREPROCESSING LEAKAGE: Scaling features using StandardScaler().fit() on combined train+test data. Test statistics leak into training. Correct: Fit scaler on training data only. WHY IT\'S DANGEROUS: (1) FALSE CONFIDENCE: Models appear perfect in development (99% accuracy!) but fail miserably in production (55% accuracy). (2) WRONG DECISIONS: Business decisions made based on inflated metrics lead to costly failures. (3) HARD TO DETECT: Especially with complex pipelines, leakage can hide in subtle ways. (4) REPUTATION DAMAGE: Deploying a leaky model damages data science team credibility. (5) WASTED RESOURCES: Time and money spent on model that doesn\'t work. DETECTING LEAKAGE: (1) Suspiciously high performance (too good to be true usually is). (2) Feature importance shows unexpected features as top predictors. (3) Performance drops dramatically in production vs validation. (4) Careful review: "Would this information be available at prediction time?" PREVENTION: (1) Maintain strict temporal order in time series. (2) Only use information from before prediction time. (3) Fit all transformations on training data only. (4) Use proper cross-validation (time-based for temporal data). (5) Have someone else review features for leakage. Real example: Predicting hospital readmissions, someone created "total_hospital_stays" feature including the current stay plus future stays. Model was 95% accurate in validation but 62% in production. Leakage made it useless.',
    keyPoints: [
      'Data leakage: using information not available at prediction time',
      'Types: temporal, target, train/test contamination, preprocessing',
      'Causes unrealistically high validation performance',
      'Results in dramatic failure in production',
      'Detect via suspicious performance and careful feature review',
      'Prevent by strictly using only pre-prediction information',
    ],
  },
  {
    id: 'q2',
    question:
      'Compare domain-driven feature engineering with automated feature generation. What are the trade-offs, and when would you use each approach?',
    sampleAnswer:
      'Domain-driven and automated feature engineering serve different purposes with distinct trade-offs: DOMAIN-DRIVEN FEATURE ENGINEERING: APPROACH: Human experts create features based on domain knowledge. Example: Financial analyst creates "debt_to_income_ratio" for loan default prediction. PROS: (1) Interpretable: "debt_to_income_ratio" is immediately understandable. (2) Efficient: Creates few but highly predictive features. (3) Expert knowledge: Incorporates decades of domain wisdom. (4) Generalizes well: Based on causal understanding, not spurious correlations. (5) Regulatory compliant: Can explain features to regulators. CONS: (1) Requires domain expertise (expensive, time-consuming). (2) May miss non-obvious patterns. (3) Limited by human imagination. (4) Doesn\'t scale to many domains. (5) Subject to human bias and assumptions. AUTOMATED FEATURE GENERATION: APPROACH: Algorithms create features automatically (polynomial features, Deep Feature Synthesis, AutoML). Example: PolynomialFeatures creates x², x³, xy, x²y, etc. PROS: (1) Scalable: Can create thousands of features quickly. (2) Discovers unexpected patterns humans miss. (3) No domain expertise required. (4) Consistent and reproducible. (5) Can combine with domain features. CONS: (1) Creates many irrelevant features (need feature selection). (2) Often uninterpretable (what is "feature_847"?). (3) Risk of overfitting without proper regularization. (4) Computationally expensive. (5) May not capture domain nuances. WHEN TO USE EACH: USE DOMAIN-DRIVEN WHEN: (1) Domain expertise is available and valuable. (2) Interpretability is critical (healthcare, finance, legal). (3) Regulatory requirements demand explainability. (4) Limited data (domain features more efficient). (5) Need features that generalize across time/contexts. USE AUTOMATED WHEN: (1) No domain expertise available. (2) Exploring new domain where patterns unknown. (3) Have large datasets (can support many features). (4) Using tree-based or neural network models (handle complexity). (5) Rapid prototyping phase. BEST PRACTICE: HYBRID APPROACH: (1) Start with domain features (foundation). (2) Add automated features (discovery). (3) Use feature selection to combine best of both. (4) Validate that automated features make business sense. Real example: Credit scoring. Domain features: "debt_to_income_ratio", "payment_history_12mo", "credit_utilization". Automated: polynomial interactions discovered "income × payment_history" matters. Hybrid achieved 15% better performance than either alone.',
    keyPoints: [
      'Domain-driven: interpretable, efficient, requires expertise',
      'Automated: scalable, discovers patterns, creates many features',
      'Domain-driven best when interpretability and expertise available',
      'Automated best for exploration and when expertise lacking',
      'Hybrid approach often optimal: domain foundation + automated discovery',
      'Always validate that features make business sense',
    ],
  },
  {
    id: 'q3',
    question:
      'Explain the train/test consistency principle in feature engineering. What problems occur when this principle is violated, and how do you maintain consistency?',
    sampleAnswer:
      "Train/test consistency means features must be engineered identically on training and test data, using ONLY information from training data to define transformations. PRINCIPLE: Any statistic, threshold, or parameter learned for feature transformation must be computed from training data only, then applied to both training and test data. VIOLATIONS AND PROBLEMS: (1) SCALING WITH TRAIN+TEST DATA: WRONG: StandardScaler().fit(X_train_and_test) - test data influences scaling parameters. Leaks test statistics into training. PROBLEM: Optimistic validation performance, poor generalization. CORRECT: scaler.fit(X_train), then apply scaler.transform(X_train) and scaler.transform(X_test). (2) IMPUTATION WITH OVERALL MEAN: WRONG: Fill missing with mean of entire dataset. PROBLEM: Test data's mean leaks into training decisions. CORRECT: Compute mean from X_train only, use that mean for both train and test. (3) ENCODING WITH FULL DATASET: WRONG: LabelEncoder().fit(all_categories_including_test). PROBLEM: Test categories influence encoding scheme. CORRECT: Fit encoder on training categories only, handle unseen test categories properly. (4) BINNING WITH GLOBAL QUANTILES: WRONG: Create bins using quantiles from train+test combined. PROBLEM: Test distribution influences where bins are cut. CORRECT: Define bins from training data quantiles only. (5) FEATURE SELECTION WITH FULL DATA: WRONG: Select features using correlation with target on train+test. PROBLEM: Test target values influence which features are selected. CORRECT: Select features using only training data. MAINTAINING CONSISTENCY: (1) SCIKIT-LEARN PIPELINES: Use Pipeline to ensure fit() happens on train only, transform() applied to both. (2) SEPARATE FIT AND TRANSFORM: Always: preprocessor.fit(X_train), then preprocessor.transform(X_train) and preprocessor.transform(X_test). Never: preprocessor.fit_transform(X_all). (3) SAVE TRANSFORMATION PARAMETERS: Store means, stds, encodings, bins learned from training data. Apply same parameters to test and production data. (4) CROSS-VALIDATION DONE RIGHT: Each CV fold must fit transformations on its training portion only. Use Pipeline in cross_val_score() to ensure this. (5) TIME SERIES SPECIAL CASE: For time series, use only past data. Never: Include future dates in rolling window calculations. Always: Compute features using only historical data available at that point. (6) FEATURE ENGINEERING FUNCTIONS: Write functions that take train data as input, return both transformed train data AND fitted transformer object. PRODUCTION IMPLICATIONS: In production, you must apply THE EXACT SAME transformations (same means, same bins, same encodings) learned during training. Store these parameters (pickle, joblib, or database). DETECTION: If test performance is suspiciously close to training performance, or feature distributions differ dramatically between train and test, investigate consistency.",
    keyPoints: [
      'Train/test consistency: fit transformations on training data only',
      'Apply same learned parameters to both train and test',
      'Violations cause data leakage and optimistic performance',
      'Use scikit-learn Pipelines to maintain consistency',
      'Save transformation parameters for production deployment',
      'Critical for scaling, imputation, encoding, and feature selection',
    ],
  },
];
