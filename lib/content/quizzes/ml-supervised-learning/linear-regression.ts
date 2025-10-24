/**
 * Discussion Questions for Linear Regression
 */

import { QuizQuestion } from '../../../types';

export const linearregressionQuiz: QuizQuestion[] = [
  {
    id: 'linear-regression-q1',
    question:
      'Explain why the assumptions of linear regression (linearity, independence, homoscedasticity, and normality of errors) are important. What happens if each assumption is violated, and how can you detect and address violations in practice?',
    hint: "Consider how each assumption affects the model's predictions, coefficient estimates, and statistical inference.",
    sampleAnswer:
      'The assumptions of linear regression are fundamental to ensuring the model provides reliable predictions and valid statistical inference. Linearity assumes that the relationship between features and target is linear; if violated, the model systematically under or overestimates predictions in certain regions. Detection: scatter plots showing non-linear patterns and residual plots showing curved patterns. Solution: transform features (log, polynomial), add interaction terms, or use non-linear models.\\n\\nIndependence assumes observations are not correlated with each other; violation (common in time series or clustered data) causes standard errors to be underestimated, making coefficients appear more significant than they are. Detection: Durbin-Watson test for time series, plots of residuals over time. Solution: use time series models (ARIMA), account for clustering, or add lagged variables.\\n\\nHomoscedasticity (constant variance) assumes errors have the same spread across all feature values; heteroscedasticity causes inefficient parameter estimates and unreliable confidence intervals. Detection: residual plot showing funnel shape (variance increasing/decreasing). Solution: transform target variable (log), use weighted least squares, or robust standard errors.\\n\\nNormality of errors is needed for valid hypothesis tests and confidence intervals (though less critical with large samples due to Central Limit Theorem); if violated, p-values and confidence intervals may be inaccurate. Detection: histogram of residuals, Q-Q plot showing deviations from diagonal line. Solution: transform variables, use bootstrapping for inference, or use robust regression methods. In practice, check assumptions using diagnostic plots after fitting and iterate on feature engineering or model choice.',
    keyPoints: [
      'Linearity violation: systematic prediction errors; fix with transformations or non-linear models',
      'Independence violation: underestimated standard errors; common in time series and clustered data',
      'Homoscedasticity violation: inefficient estimates, unreliable confidence intervals',
      'Normality violation: incorrect p-values and confidence intervals (less critical with large samples)',
      'Use diagnostic plots: scatter plots, residual plots, Q-Q plots, Durbin-Watson test',
      'Solutions vary: transformations, robust methods, different model types',
    ],
  },
  {
    id: 'linear-regression-q2',
    question:
      'Compare and contrast solving linear regression using the Normal Equation (matrix inversion) versus Gradient Descent. What are the computational tradeoffs, and when would you choose one over the other?',
    hint: 'Think about computational complexity, dataset size, feature dimensionality, and practical considerations.',
    sampleAnswer:
      "The Normal Equation provides a closed-form, analytical solution using matrix operations: β = (X^T X)^(-1) X^T y. Its main advantage is finding the exact optimal solution in one step with no hyperparameters to tune. However, computing (X^T X)^(-1) has O(n³) complexity where n is the number of features, making it impractical for high-dimensional data (thousands of features). Additionally, matrix inversion requires the matrix to be invertible (full rank), which fails when features are perfectly collinear. Memory requirements also scale with feature dimensionality, potentially causing issues.\n\nGradient Descent is an iterative optimization algorithm that updates parameters step by step: β = β - α∇J(β). Its key advantage is O(k*m*n) complexity where k is iterations, m is samples, and n is features - much better for high-dimensional data. It works well with millions of samples and can handle streaming data. Stochastic and mini-batch variants further improve efficiency and enable online learning. However, it requires tuning learning rate and iterations, doesn't find the exact optimum (depends on convergence criteria), and can be slower than Normal Equation for small datasets.\n\nPractical guidelines: Use Normal Equation for small to medium datasets (< 10,000 features) when you want the exact solution and features aren't highly correlated. Use Gradient Descent for large-scale problems, high-dimensional data, when you need online learning, or when combined with regularization (which often makes gradient descent more natural). In production ML systems, gradient descent and its variants (SGD, Adam) dominate because they scale to massive datasets and integrate seamlessly with neural network frameworks. For financial applications with moderate feature counts and batch processing, Normal Equation often suffices and provides exact solutions quickly.",
    keyPoints: [
      'Normal Equation: O(n³) complexity, exact solution, no hyperparameters, fails with non-invertible matrices',
      'Gradient Descent: O(k*m*n) complexity, scales to large datasets, requires tuning, approximate solution',
      'Normal Equation better for: small-medium datasets, low-dimensional features, need exact solution',
      'Gradient Descent better for: large datasets, many features, online learning, regularization',
      'Modern ML systems favor gradient descent for scalability and integration with deep learning',
      'Consider dataset size, feature count, and infrastructure when choosing',
    ],
  },
  {
    id: 'linear-regression-q3',
    question:
      'In financial modeling, linear regression is often used for factor models (e.g., predicting stock returns from market factors). Discuss the challenges and considerations specific to financial applications, including issues like autocorrelation, regime changes, and the difference between in-sample and out-of-sample performance.',
    hint: 'Consider the non-stationary nature of financial data, lookahead bias, and the difficulty of prediction in efficient markets.',
    sampleAnswer:
      "Linear regression in financial modeling faces unique challenges that differ from typical ML applications. First, financial time series violate the independence assumption due to autocorrelation - today's returns are correlated with recent past returns. This inflates significance tests and creates misleading confidence intervals. Solutions include using Newey-West standard errors for robust inference, incorporating lagged variables explicitly, or using time series models like ARIMA. Ignoring autocorrelation leads to overconfident predictions and poor risk estimates.\n\nRegime changes are another critical issue - financial relationships are non-stationary and change over time. A model trained during bull markets may fail completely during bear markets or crashes. The correlation structure between factors shifts, volatility regimes change, and historical relationships break down. This manifests as severe degradation in out-of-sample performance compared to in-sample fits. Solutions include: regime detection models (Hidden Markov Models), rolling window estimation, adaptive learning rates, or ensemble models trained on different market conditions.\n\nThe in-sample vs out-of-sample gap is particularly large in finance because markets are highly efficient - most predictable patterns are quickly arbitraged away. A model might show high R² in-sample by fitting noise, but near-zero R² out-of-sample. Rigorous validation requires: walk-forward analysis (train on past, test on future), no lookahead bias (only use information available at prediction time), transaction costs in evaluation, and realistic trading constraints. The Sharpe ratio, maximum drawdown, and actual P&L are more relevant than R² for trading strategies.\n\nAdditional financial-specific issues include: heteroscedasticity (volatility clustering), fat tails (extreme events more common than normal distribution), microstructure noise, survivorship bias in historical data, and the cost of being wrong (losses are real). Best practices: use robust regression methods, validate on multiple time periods, implement position sizing and risk management, combine with domain knowledge, and maintain healthy skepticism about seemingly strong predictive relationships.",
    keyPoints: [
      'Autocorrelation common in financial data, violates independence assumption, inflates significance',
      'Regime changes cause non-stationarity, relationships break down over time',
      'Large in-sample to out-of-sample gap due to market efficiency and noise fitting',
      'Must avoid lookahead bias, use walk-forward validation, include transaction costs',
      'Financial-specific issues: volatility clustering, fat tails, survivorship bias',
      'Practical metrics: Sharpe ratio, maximum drawdown, real P&L more important than R²',
      'Require robust methods, multiple validation periods, and strict risk management',
    ],
  },
];
