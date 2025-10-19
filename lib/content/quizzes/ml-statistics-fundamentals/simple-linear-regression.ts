/**
 * Quiz questions for Simple Linear Regression section
 */

export const simplelinearregressionQuiz = [
  {
    id: 'q1',
    question:
      'Your linear regression model has R²=0.85 and all residual diagnostic plots look good, but the slope coefficient has p=0.12. Should you use this model for predictions? Explain the apparent contradiction and what it tells you about your data.',
    hint: 'Consider sample size, power, and the difference between predictive performance and statistical significance.',
    sampleAnswer:
      "This apparent contradiction reveals an important distinction between **predictive performance** and **statistical significance**. Analysis: **High R² (0.85)** means: The model explains 85% of variance - excellent predictive performance! The linear relationship is strong in your data. **Non-significant slope (p=0.12)** means: We don't have strong statistical evidence that the true population slope is non-zero. High uncertainty in the slope estimate. **Why this happens**: (1) **Small sample size**: With n=10-15, even strong relationships may not reach p<0.05. The model fits well but we lack statistical power. (2) **Large standard error**: High variability in slope estimate despite good fit. (3) **Outliers/influential points**: A few points drive the fit but make inference unreliable. **Should you use it for predictions?** (1) **If n is small**: Be cautious. The high R² might be overfitting. The model works on THIS data but generalization is uncertain. Use cross-validation to check. (2) **If prediction is the goal**: R² matters more than p-value. Go ahead if cross-validation confirms performance. (3) **If inference/interpretation is the goal**: The p=0.12 is problematic. Can't confidently say X affects Y in the population. **Best actions**: (1) Check sample size - if n<30, collect more data. (2) Run cross-validation to verify R² isn't inflated. (3) Look for influential points (Cook's distance). (4) Bootstrap confidence intervals for more robust inference. (5) Report both: \"Model explains 85% of variance but relationship not statistically significant (possibly due to small sample).\" **Lesson**: Statistical significance ≠ Practical importance. R² measures fit quality, p-value measures certainty. Need both!",
    keyPoints: [
      'R² measures predictive fit, p-value measures statistical certainty',
      'Small sample → good fit but low power to detect significance',
      'For prediction: R² matters more; use cross-validation',
      'For inference: p-value matters; need statistical significance',
      'Check sample size and run diagnostics',
      'Bootstrap CIs for robust inference with small samples',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain the difference between confidence intervals and prediction intervals in regression. Why is the prediction interval always wider? In what ML scenarios would you use each?',
    hint: 'Think about what each interval is estimating and sources of uncertainty.',
    sampleAnswer:
      '**Confidence Interval (CI)**: Estimates uncertainty in the **mean** response. Formula: ŷ ± t* × SE(mean). SE(mean) = σ√(1/n + (x₀-x̄)²/Σ(x-x̄)²). Smaller SE because averaging reduces variance. **Prediction Interval (PI)**: Estimates uncertainty for an **individual** response. Formula: ŷ ± t* × SE(prediction). SE(prediction) = σ√(1 + 1/n + (x₀-x̄)²/Σ(x-x̄)²). Note the extra "1" under the square root! **Why PI is always wider**: PI uncertainty = CI uncertainty + Individual variation. Even if we knew the true mean perfectly (CI=0), individuals still vary around that mean. The "1" in SE(prediction) represents this irreducible individual variance. **Example**: Predicting exam scores based on study hours: CI (80%): Mean score for ALL students studying 5 hours is between 78-82. PI (80%): A SPECIFIC student studying 5 hours will score between 65-95. **ML Scenarios**: **Use Confidence Intervals when**: (1) Estimating population averages: "What\'s the average customer lifetime value for users with feature X=5?" (2) A/B testing: Comparing mean responses between groups. (3) Policy decisions based on average outcomes. (4) Reporting model performance: Average prediction error. **Use Prediction Intervals when**: (1) Individual predictions: "Will this specific customer churn?" Need uncertainty for this one person. (2) Risk assessment: "What\'s the worst-case prediction for safety-critical application?" (3) Outlier detection: Points outside PI are unusual. (4) Decision-making with individuals: Medical diagnosis, loan approval for specific person. **Common mistake**: Using CI when you should use PI! "Model predicts 100 with 95% CI [98, 102]" is wrong if predicting individuals - should be PI [70, 130]. **Width comparison**: At x̄ (mean of X): PI/CI ratio ≈ σ/σ√(1/n) ≈ √n. With n=100, PI is ~10× wider!',
    keyPoints: [
      'CI: uncertainty in mean response (narrower)',
      'PI: uncertainty in individual response (wider)',
      'PI = CI + individual variation',
      'CI for: averages, A/B testing, policy decisions',
      'PI for: individual predictions, risk assessment',
      'Common error: using CI for individual predictions',
    ],
  },
  {
    id: 'q3',
    question:
      'You fit a linear regression and notice the Q-Q plot shows residuals deviating from the diagonal line at both tails. What does this indicate about your model assumptions? Should you be concerned, and what remedial actions could you take?',
    hint: 'Think about the normality assumption and when it matters.',
    sampleAnswer:
      "Q-Q plot showing deviation at tails indicates **non-normal residuals**, specifically suggesting: **Heavy tails** (points curve away at both ends): More extreme values than normal distribution predicts. Possibly outliers or a heavy-tailed distribution (e.g., t-distribution). **Should you be concerned?** It depends: **When it matters (be concerned)**: (1) **Small samples (n<30)**: Confidence intervals and p-values rely on normality. May be inaccurate. (2) **Extreme deviations**: Large departures suggest serious assumption violation. (3) **Inference is goal**: Testing hypotheses about coefficients requires normality for validity. (4) **Prediction intervals**: Will be inaccurate if residuals non-normal. **When it matters less**: (1) **Large samples (n>100)**: Central Limit Theorem - inference becomes robust to non-normality. (2) **Prediction only**: Point predictions unaffected; only intervals affected. (3) **Mild deviations**: Small departures often negligible. **Remedial actions**: (1) **Check for outliers**: Cook's distance, leverage plots. Remove or investigate influential points. (2) **Transform Y**: Log(Y), √Y, Box-Cox transformation. Often fixes right-skewness and heavy tails. (3) **Robust regression**: Huber regression, RANSAC - less sensitive to outliers. (4) **Non-parametric methods**: Quantile regression, bootstrap CIs. (5) **Different model**: Maybe relationship isn't linear - try GAMs, polynomial regression. (6) **Accept it**: If n is large and prediction is goal, might be fine. Report robust standard errors. **Example decision tree**: Tails deviate + n<30 → Transform Y or use bootstrap. Tails deviate + n>100 + prediction focus → Acceptable, report robust SEs. Tails deviate + outliers present → Remove outliers or robust regression. Tails deviate + all else fails → Non-parametric or different model family. **Bottom line**: Normality assumption is for inference (CIs, p-values), not predictions. Large samples make it less critical. Always visualize and use appropriate methods for your goals.",
    keyPoints: [
      'Q-Q tail deviations = heavy-tailed residuals, possible outliers',
      'Matters most for: small samples, inference, prediction intervals',
      'Less critical for: large samples, point predictions',
      'Remedies: transform Y, remove outliers, robust regression',
      'Large n → inference robust to modest non-normality (CLT)',
      'Always report diagnostics and methods used',
    ],
  },
];
