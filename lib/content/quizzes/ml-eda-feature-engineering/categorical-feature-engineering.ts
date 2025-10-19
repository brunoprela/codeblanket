/**
 * Quiz questions for Categorical Feature Engineering section
 */

export const categoricalfeatureengineeringQuiz = [
  {
    id: 'q1',
    question:
      'Explain the difference between label encoding and one-hot encoding. When should you use each, and what problems can occur if you use the wrong one?',
    sampleAnswer:
      'LABEL ENCODING: Assigns integer to each category (Red=0, Blue=1, Green=2). Creates single column. WHEN TO USE: ONLY for ordinal features with natural ordering. Examples: [Small, Medium, Large] → [0,1,2], [Low, Medium, High] → [0,1,2], Education level, T-shirt sizes. PROBLEM IF MISUSED: Using label encoding on nominal features (no natural order) makes model assume ordering exists! Example: Encoding colors [Red=0, Blue=1, Green=2] makes model think Green(2) > Blue(1) > Red(0), which is meaningless. Linear models will treat "distance" between colors as meaningful. Result: Poor performance, nonsensical coefficients. ONE-HOT ENCODING: Creates binary column for each category. [Red, Blue, Green] becomes 3 columns. WHEN TO USE: For nominal features (no natural ordering). Examples: Color, Country, Product Category, Gender. Each category becomes independent feature. ADVANTAGES: No false ordering implied, each category independent. PROBLEMS: (1) High dimensionality: 1000 categories → 1000 columns. (2) Sparse matrices. (3) Multicollinearity: columns sum to 1 (dummy variable trap). FIX: drop_first=True for linear models. (4) Computationally expensive with high cardinality. REAL EXAMPLE DISASTER: Encoding "country" (200 countries) with label encoding (0-199) for linear regression. Model learns "USA(150) is closer to Canada(75) than to UK(175)" based on arbitrary numbers, not geography. Nonsensical. CORRECT: One-hot encode country (or use target encoding for high cardinality). SUMMARY: Ordinal (natural order) → Label encoding. Nominal (no order) → One-hot encoding. Using label encoding on nominal features is a common beginner mistake that destroys model performance!',
    keyPoints: [
      'Label: ordinal features with natural order only',
      'One-hot: nominal features without natural order',
      'Label encoding nominal creates false ordering',
      'One-hot creates sparse high-dimensional data',
      'Use drop_first=True to avoid multicollinearity',
      'Tree models handle both better than linear models',
    ],
  },
  {
    id: 'q2',
    question:
      'Describe target encoding and explain why it risks data leakage. How can you use target encoding safely in production machine learning?',
    sampleAnswer:
      'TARGET ENCODING (Mean Encoding): Replace category with mean target value for that category. Example: If category A has average target value 0.8, encode all A→0.8. WHY POWERFUL: Captures relationship between category and target in single number. Handles high cardinality (1000 categories → 1 column). Often best single encoding for tree models. WHY LEAKAGE RISK: Using target variable information to create features violates independence assumption! PROBLEM: If you compute category means from entire dataset (train+test), test target values leak into training features. Example: Category A has 100 samples: 50 in train (mean=0.6), 50 in test (mean=0.9). If you use overall mean (0.75), test information leaked into train! Model performs unrealistically well in validation, fails in production. SAFE USAGE - K-FOLD TARGET ENCODING: (1) Split data into K folds. (2) For each fold: compute category means from OTHER folds only. (3) Encode current fold using out-of-fold means. (4) For test data: use means from ALL training data. This ensures no sample sees its own target value in encoding. PRODUCTION PIPELINE: (1) During training: Use K-fold target encoding on train data. (2) Compute global category means from full training set. (3) Save these global means (in database or pickle). (4) In production: Use saved means to encode new data. (5) For unseen categories: use global mean as fallback. SMOOTHING/REGULARIZATION: Add smoothing to avoid overfitting rare categories. Smoothed = (n_cat * cat_mean + m * global_mean) / (n_cat + m). Balances category-specific information with global mean. ADVANCED: Cross-validated target encoding in scikit-learn pipeline using category_encoders library handles this automatically. KEY INSIGHT: Target encoding is powerful but dangerous. Must use proper cross-validation to prevent leakage. Many competitions won by proper target encoding!',
    keyPoints: [
      'Target encoding: replace category with mean target value',
      'Risks leakage if train/test computed together',
      'Safe usage: K-fold encoding with out-of-fold means',
      'Production: save training means, apply to new data',
      'Add smoothing for rare categories',
      'Very powerful for high-cardinality features in trees',
    ],
  },
  {
    id: 'q3',
    question:
      'You have a categorical feature with 10,000 unique categories. Discuss your options for encoding this feature and the trade-offs of each approach.',
    sampleAnswer:
      'High-cardinality features (many categories) require special handling. 10,000 categories is extreme! OPTIONS: (1) ONE-HOT ENCODING: Creates 10,000 columns. PROS: Captures all information, simple. CONS: Extreme dimensionality (curse of dimensionality), memory explosion, sparse matrix (most values 0), slow training. VERDICT: Usually impractical for >100 categories. (2) TOP-N + OTHER: Keep top N most frequent categories, group rest as "Other". Example: Top 50 + Other = 51 categories, then one-hot. PROS: Reduces dimensionality, focuses on important categories. CONS: Loses information for rare categories, arbitrary cutoff. WORKS WHEN: Clear distinction between frequent and rare. (3) FREQUENCY ENCODING: Replace category with its frequency count. PROS: Single column, no dimensionality issues, fast. CONS: Different categories with same frequency get same encoding, loses category identity. WORKS WHEN: Frequency is predictive, tree-based models. (4) TARGET ENCODING: Replace with mean target value per category. PROS: Single column, captures predictive power, handles high cardinality perfectly. CONS: Requires target variable, leakage risk (need K-fold), overfits rare categories (need smoothing). WORKS WHEN: Supervised learning, proper CV used, tree models. BEST for high cardinality! (5) HASH ENCODING: Use hash function to map to fixed number of buckets. PROS: Fixed dimensionality, handles unseen categories. CONS: Hash collisions (different categories same bucket), loses interpretability. WORKS WHEN: Online learning, streaming data. (6) ENTITY EMBEDDINGS: Neural network learns dense vector representation. Example: 10,000 categories → 50D embedding. PROS: Learns similarity, dramatic dimensionality reduction, captures complex patterns. CONS: Requires neural network, computationally expensive training, black box. WORKS WHEN: Using deep learning, large dataset, performance critical. BEST for deep learning! (7) HIERARCHICAL GROUPING: If categories have structure, create hierarchy. Example: Product IDs → Category → Department. PROS: Uses domain knowledge, interpretable, reduces cardinality. CONS: Requires domain expertise, may lose specificity. RECOMMENDATION FOR 10K CATEGORIES: TREE MODELS: Target encoding (with proper CV and smoothing) or frequency encoding. BEST performance. LINEAR MODELS: Target encoding or hash encoding or top-N + one-hot. DEEP LEARNING: Entity embeddings. Learn representation during training. PRACTICAL: Often combine approaches! Target encode + frequency encode + top-10 one-hot for most frequent.',
    keyPoints: [
      '10K categories too many for simple one-hot encoding',
      'Top-N + Other reduces dimensionality but loses information',
      'Target encoding best for trees if properly cross-validated',
      'Frequency encoding simple but loses category identity',
      'Embeddings best for neural networks (learn representation)',
      'Hash encoding for streaming/online learning',
    ],
  },
];
