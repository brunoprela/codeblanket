/**
 * Quiz questions for Multivariate Analysis section
 */

export const multivariateanalysisQuiz = [
  {
    id: 'q1',
    question:
      'Explain multicollinearity and why it matters for machine learning models. How does it affect different types of models differently?',
    hint: 'Think about linear models vs tree-based models.',
    sampleAnswer:
      'Multicollinearity occurs when two or more predictor features are highly correlated with each other. WHY IT MATTERS: In linear models (linear regression, logistic regression), multicollinearity causes: (1) Unstable coefficients - small changes in data lead to large changes in coefficients. (2) Difficult interpretation - can\'t isolate individual feature effects. (3) Inflated standard errors - reduces statistical power. (4) Wrong signs on coefficients - positive correlation feature gets negative coefficient. Example: If "square feet" and "number of rooms" are correlated at r=0.95, linear regression can\'t tell which actually drives price. Coefficients become unreliable. DETECTION: (1) High correlation (|r| > 0.8) between features. (2) Variance Inflation Factor (VIF) > 10. (3) Condition number > 30. IMPACT BY MODEL TYPE: LINEAR MODELS (high impact): Coefficients unreliable, interpretation impossible, predictions still okay but confidence intervals wrong. TREE-BASED MODELS (low impact): No problem! Trees naturally handle it by selecting one feature at splits. Predictions unaffected. NEURAL NETWORKS (medium impact): Can learn to route around it but training may be less efficient. SOLUTIONS: (1) Remove one of correlated pair (keep more interpretable or predictive one). (2) Combine correlated features (PCA, average, domain-specific combination). (3) Regularization (Ridge/Lasso) helps but doesn\'t eliminate. (4) Use tree-based models if multicollinearity severe. KEY INSIGHT: Multicollinearity is primarily an interpretability problem, not a prediction problem. If you only care about predictions (not understanding feature effects), it matters less.',
    keyPoints: [
      'Multicollinearity: high correlation between predictor features',
      'Makes linear model coefficients unstable and uninterpretable',
      'Severely impacts linear models, minimal impact on tree-based models',
      'Detect with correlation matrix, VIF, or condition number',
      "Predictions still work, but interpretation and confidence intervals don't",
      'Solutions: remove features, combine with PCA, or use tree models',
    ],
  },
  {
    id: 'q2',
    question:
      'You perform PCA and find that the first 3 components explain 85% of variance, while your dataset has 20 original features. What does this tell you, and how would you decide whether to use PCA for your model?',
    sampleAnswer:
      "Finding that 3 PCs explain 85% of variance in 20-feature dataset tells us: (1) HIGH REDUNDANCY: Original features contain significant redundant information. Many features likely correlated. (2) INTRINSIC DIMENSIONALITY: Data essentially lives in 3-dimensional space despite 20 features. (3) NOISE REDUCTION OPPORTUNITY: Remaining 15% might be noise that can be discarded. WHEN TO USE PCA FOR MODELING: PROS: (1) Dimensionality reduction: 3 features instead of 20 - faster training, less memory. (2) Multicollinearity elimination: PCs are orthogonal (uncorrelated). (3) Noise reduction: Dropping low-variance components can improve generalization. (4) Visualization: Can plot 3D data easily. CONS: (1) INTERPRETABILITY LOSS: PC1, PC2, PC3 are linear combinations - can't explain \"What drives the outcome?\" in business terms. Critical if stakeholders need explanations. (2) Information loss: Lost 15% of variance - might contain important patterns. (3) Test set consistency: Must apply same transformation (store scaler and PCA object). DECISION FRAMEWORK: USE PCA WHEN: (1) Many highly correlated features (multicollinearity). (2) Interpretability not critical (just need predictions). (3) High-dimensional data causing computational issues. (4) Data visualization needed. (5) Using linear models sensitive to multicollinearity. DON'T USE PCA WHEN: (1) Interpretability is critical (need to explain feature importance to business). (2) Tree-based models (they handle high dimensions and correlations well). (3) Features already meaningful and distinct. (4) Small number of features already (<20). IN THIS CASE: If using linear regression and don't need interpretability → YES, use 3 PCs. If using Random Forest or XGBoost → NO, keep original features. If need to explain to business → NO, feature selection better than PCA. ALTERNATIVE: Instead of PCA, use feature selection to keep most important original features - maintains interpretability.",
    keyPoints: [
      'Few PCs explaining most variance indicates high redundancy',
      'PCA benefits: dimensionality reduction, removes multicollinearity',
      'PCA drawbacks: loses interpretability, information loss',
      'Use PCA for linear models with many correlated features',
      'Avoid PCA when interpretability critical or using tree models',
      'Feature selection maintains interpretability better than PCA',
    ],
  },
  {
    id: 'q3',
    question:
      'Describe how pair plots help identify feature interactions and non-linear relationships. What patterns should you look for, and how do they inform feature engineering?',
    sampleAnswer:
      'Pair plots display scatter plots for all feature pairs, revealing patterns invisible in univariate or simple bivariate analysis. PATTERNS TO LOOK FOR: (1) NON-LINEAR RELATIONSHIPS: Scatter plots showing curves (polynomial, exponential, logarithmic) instead of straight lines. Action: Create polynomial features (x², x³) or apply transformations (log, sqrt). Example: If income vs spending is exponential, create log (income) feature. (2) INTERACTION EFFECTS: Pattern in X1 vs Y depends on value of X2. Example: Age vs insurance cost relationship differs for smokers vs non-smokers. Action: Create interaction features (age × smoker). (3) DISTINCT CLUSTERS: Scatter plots showing separate groups. Action: Try clustering or create categorical features. Example: Customer data showing distinct segments. (4) OUTLIERS: Points far from main cluster, visible across multiple plots. Action: Investigate - errors or valid extreme cases? (5) CORRELATION PATTERNS: Strong linear relationships between features. Action: Check for multicollinearity, consider removing redundant features. (6) HETEROSCEDASTICITY: Spread of Y increases with X (fan shape). Action: Log transform Y or use robust regression. (7) BOUNDARY EFFECTS: Data clustered at edges (e.g., all salaries at minimum wage or cap). Action: Create binary indicators for boundary values. FEATURE ENGINEERING INSIGHTS: Example from housing data pair plot: (1) Square feet vs price: exponential curve → create log (square_feet). (2) Location effects: Lat/Long show distinct clusters → create neighborhood features. (3) Age × renovated: Interaction pattern → create age_renovated_interaction. (4) Bedrooms vs bathrooms: Strong correlation → create bed_bath_ratio. BEST PRACTICES: (1) Use hue/color by target variable to see predictive patterns. (2) Sample large datasets (plot max 1000-2000 points for clarity). (3) Focus on features with domain meaning for interpretable engineering. (4) Look diagonal (distributions) and off-diagonal (relationships). Pair plots are EDA gold - they surface opportunities for feature engineering that dramatically improve model performance.',
    keyPoints: [
      'Pair plots show all pairwise scatter plots and distributions',
      'Reveal non-linear relationships, interactions, and clusters',
      'Non-linear patterns suggest polynomial features or transformations',
      'Distinct groups suggest clustering or categorical features',
      'Strong correlations indicate multicollinearity to address',
      'Use color by target to identify predictive patterns',
    ],
  },
];
