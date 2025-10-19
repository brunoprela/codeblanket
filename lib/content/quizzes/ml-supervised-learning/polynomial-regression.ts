/**
 * Discussion Questions for Polynomial and Non-linear Regression
 */

import { QuizQuestion } from '../../../types';

export const polynomialregressionQuiz: QuizQuestion[] = [
  {
    id: 'polynomial-regression-q1',
    question:
      'Explain the bias-variance tradeoff specifically in the context of polynomial regression. How does increasing the polynomial degree affect bias and variance? What practical strategies would you use to find the optimal degree?',
    hint: 'Consider what happens to training vs. test error as degree increases, and think about validation techniques.',
    sampleAnswer:
      "In polynomial regression, the bias-variance tradeoff manifests clearly through the polynomial degree parameter. Low-degree polynomials (e.g., degree 1) have high bias - they make strong assumptions about the relationship being simple (linear), which leads to systematic errors when the true relationship is more complex. The model underfits, showing poor performance on both training and test data. As we increase polynomial degree, we reduce bias by allowing the model to fit more complex curves, improving training performance.\n\nHowever, high-degree polynomials introduce high variance - the model becomes overly sensitive to small fluctuations in the training data. While training error continues to decrease (the model fits training data better and better), test error begins to increase after a certain point. The model memorizes noise and training-specific patterns that don't generalize. For example, a degree-15 polynomial through 20 points can fit perfectly (zero training error) but will oscillate wildly and perform terribly on new data.\n\nPractical strategies for finding optimal degree: (1) Cross-validation - train models with different degrees and select the one with best cross-validation score, not training score; (2) Learning curves - plot training and validation errors vs. degree to visualize the sweet spot where test error is minimized; (3) Regularization - use Ridge or Lasso with high-degree polynomials, letting the algorithm automatically suppress unnecessary terms; (4) Domain knowledge - consider what relationships make physical/business sense; (5) Hold-out validation - always evaluate final model on untouched test set; (6) Visual inspection - plot fitted curves to catch obviously unrealistic behavior like wild oscillations; (7) Start simple - begin with low degrees and increase only if needed. In practice, degrees 2-3 often work well; degrees above 5-6 usually indicate you should consider a different modeling approach entirely.",
    keyPoints: [
      'Low degree: high bias, systematic errors, underfitting on both train and test',
      'High degree: high variance, fits noise, excellent train but poor test performance',
      'Training error always decreases with degree; test error U-shaped (decreases then increases)',
      'Cross-validation is gold standard for selecting degree objectively',
      'Learning curves visualize bias-variance tradeoff across degrees',
      'Regularization allows using high degrees while controlling overfitting',
      'Domain knowledge and visual inspection catch unrealistic fits',
      'Degrees above 5-6 suggest need for different modeling approach',
    ],
  },
  {
    id: 'polynomial-regression-q2',
    question:
      'Discuss the dangers of extrapolation with polynomial regression. Why are polynomials particularly problematic for making predictions outside the training data range? Provide examples of how this could lead to serious errors in real-world applications.',
    hint: 'Think about the behavior of polynomial functions at extreme values and what happens beyond your data range.',
    sampleAnswer:
      "Polynomial regression is notorious for dangerous extrapolation behavior - making predictions outside the training data range can produce wildly inaccurate, even nonsensical results. The core issue is that polynomials have no bounded behavior: they inevitably tend toward +∞ or -∞ at extreme values. A polynomial that fits beautifully within your data range can shoot off catastrophically just beyond it. For example, a degree-4 polynomial fitted to house prices vs. size (1000-3000 sq ft) might predict negative prices for 500 sq ft or billion-dollar prices for 5000 sq ft - clearly absurd.\n\nReal-world example 1 (Financial): A hedge fund fits a polynomial to model VIX (market volatility) versus S&P 500 returns using historical data from moderate market conditions. The model works well in-sample. But during a market crash (outside training range), the polynomial extrapolates volatility to impossibly high values, causing the fund to massively overhedge, realizing huge losses. The polynomial had no inherent knowledge that volatility is bounded or mean-reverting.\n\nReal-world example 2 (Healthcare): Researchers fit a polynomial to model drug efficacy vs. dosage using data from 10-100mg. Based on the curve, they extrapolate that 300mg would be highly effective. In reality, at high doses, the drug becomes toxic (a relationship the polynomial cannot capture). Clinical trials at 300mg cause serious adverse events. The polynomial fitted a smooth curve but had no understanding of toxicity thresholds.\n\nReal-world example 3 (Marketing): An e-commerce company models conversion rate vs. ad spend using a degree-3 polynomial on data from $10k-$100k monthly spend. The curve shows diminishing returns within this range. But when they extrapolate to justify $500k spend, the polynomial predicts conversion rates above 100% (mathematically possible for polynomials, physically impossible). Acting on this prediction leads to massive wasted spend.\n\nWhy polynomials are uniquely dangerous: Unlike domain-specific models with bounded outputs or asymptotic behavior, polynomials have no inherent constraints. They're purely mathematical curve-fitting tools without physical meaning. Solutions: (1) Never extrapolate - restrict predictions to training range; (2) Use domain-appropriate models (exponential, logistic, etc.) that capture known behavior; (3) Add uncertainty estimates that grow rapidly outside training range; (4) Combine with domain constraints (e.g., prices must be positive); (5) If extrapolation is necessary, use extreme caution, collect more data at those ranges, and validate extensively.",
    keyPoints: [
      'Polynomials are unbounded: tend toward ±∞ at extremes, no asymptotic behavior',
      'Can fit perfectly in-range but produce nonsensical predictions out-of-range',
      'Financial example: overestimating volatility/risk in extreme market conditions',
      'Healthcare example: missing toxicity thresholds or saturation effects',
      'Marketing example: predicting impossible conversion rates, wasted spend',
      'Polynomials are mathematical tools without physical meaning or constraints',
      'Solutions: avoid extrapolation, use domain-appropriate models, add uncertainty bounds',
      'If extrapolation needed: extreme caution, validation, collect data in new range',
    ],
  },
  {
    id: 'polynomial-regression-q3',
    question:
      'Explain multicollinearity in the context of polynomial regression and why it becomes severe with high-degree polynomials. How does it affect model interpretation and stability? What techniques can address this issue?',
    hint: 'Consider what happens when you include x, x², x³, etc. as features - how correlated are these features with each other?',
    sampleAnswer:
      'Multicollinearity occurs when features are highly correlated with each other, and it becomes particularly severe in polynomial regression because powers of the same variable are inherently correlated. If x ranges from 1 to 10, then x, x², and x³ are all monotonically related - when x increases, so do x² and x³. The correlation between x and x² might be 0.95+, making it difficult to isolate their individual effects. For degree-10 polynomials, you might have features with 0.999+ correlation.\n\nEffects on model interpretation: When features are highly correlated, coefficient estimates become unstable and unreliable. Small changes in training data can cause dramatic swings in coefficients. You might see a huge positive coefficient for x² and equally huge negative coefficient for x³, not because these are the true relationships, but because the algorithm is trying to disambiguate between nearly identical features. Individual coefficients become uninterpretable - you can\'t say "x² has this effect" because its coefficient is entangled with x³, x⁴, etc. The model as a whole might predict well, but you can\'t understand which terms matter.\n\nEffects on model stability: Multicollinearity causes numerical instability in the matrix inversion step ((X^T X)^(-1)). The matrix becomes nearly singular (determinant close to zero), making inversion unreliable. Small rounding errors in computation can lead to vastly different solutions. The model becomes sensitive to: adding/removing a single training point, slight changes in feature values, and random initialization differences. Standard errors of coefficients become inflated, making hypothesis tests unreliable. Variance Inflation Factors (VIFs) skyrocket above 10-100.\n\nTechniques to address multicollinearity: (1) **Feature scaling** - center and standardize features before creating polynomials, reducing correlation somewhat; (2) **Orthogonal polynomials** - use orthogonal basis (Legendre, Chebyshev) instead of raw powers, ensuring features are uncorrelated; (3) **Regularization** - Ridge regression is specifically designed to handle multicollinearity by adding a penalty term that shrinks correlated coefficients; Lasso can zero out redundant terms; (4) **Reduce degree** - use lower-degree polynomials or select specific terms instead of including all up to degree d; (5) **Splines** - use piecewise polynomials with fewer features per region; (6) **Feature selection** - use domain knowledge to include only meaningful interactions; (7) **PCA** - transform features to uncorrelated principal components. In practice, regularization (especially Ridge) is the most common and effective solution, allowing high-degree polynomials while maintaining stability.',
    keyPoints: [
      'Polynomial features (x, x², x³) are inherently highly correlated - multicollinearity guaranteed',
      'Correlation increases with degree; degree-10 polynomials can have 0.999+ correlations',
      'Coefficients become unstable: small data changes cause large coefficient changes',
      'Individual coefficients uninterpretable due to entanglement',
      'Numerical instability in matrix inversion, sensitive to rounding errors',
      'VIFs (Variance Inflation Factors) become very large (>10-100)',
      'Solutions: feature scaling, orthogonal polynomials, regularization (Ridge/Lasso)',
      'Ridge regression specifically designed to handle multicollinearity',
      'Alternative: use splines or reduce degree instead of high-degree globals',
    ],
  },
];
