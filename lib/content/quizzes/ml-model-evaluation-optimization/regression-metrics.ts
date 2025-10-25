export const regressionMetricsQuiz = {
  title: 'Regression Metrics - Discussion Questions',
  questions: [
    {
      id: 1,
      question: `You're building a house price prediction model for a real estate company. The prices range from $100K to $10M with a heavy right skew (most houses $200K-$500K, few luxury homes $5M+). Compare MAE, RMSE, and MAPE as evaluation metrics. Which would you choose and why? How would outlier luxury homes affect each metric?`,
      expectedAnswer: `**MAE (Mean Absolute Error)**: Treats all errors equally, robust to outliers. $100K error on $200K house = same as $100K error on $5M house. Outlier luxury homes won't dominate. **RMSE**: Squares errors, heavily penalizes large errors. A single $2M error on luxury home >> many small errors. Outliers will dominate metric. **MAPE**: Percentage-based, but asymmetric - penalizes under-predictions more. $100K error on $200K house (50%) worse than $500K error on $5M house (10%). **Choice**: MAE for this use case because: 1) Absolute dollar error matters most to business, 2) Don't want luxury homes to dominate optimization, 3) RMSE would make model overfit to luxury segment, 4) MAPE asymmetry problematic. Alternative: Use RMSE but log-transform target to handle skew, or report MAE for business, RMSE for model optimization.`,
      difficulty: 'intermediate' as const,
      category: 'Metric Selection',
    },
    {
      id: 2,
      question: `Your regression model achieves Train RMSE=10, Test RMSE=15, R²=0.65 on test set. Your manager asks "Is this good?" Walk through how you would answer this, including what additional analysis you'd perform and what comparisons you'd make.`,
      expectedAnswer: `Cannot judge "good" in isolation - need context: **1) Baseline comparison**: What\'s RMSE of predicting mean? If baseline RMSE=40, our 15 is good (62.5% improvement). If baseline=16, we barely beat it. **2) R² interpretation**: 0.65 means we explain 65% of variance, leaving 35% unexplained - moderate but acceptable for many problems. **3) Business context**: What's cost of prediction error? For $1M contracts, $15K error may be acceptable. For $1K products, it's not. **4) Overfitting analysis**: Train=10, Test=15 shows some overfitting (50% gap). Not severe but room for improvement. **5) Residual analysis**: Check if errors are unbiased, homoscedastic, normally distributed. **6) Error distribution**: What\'s median error? Max error? Are there systematic patterns? **7) Competitor comparison**: Industry benchmarks or previous models. **Recommendation**: Report multiple metrics (MAE, RMSE, R², Max Error), compare to baselines, show business impact ("reduces error by X%, saving $Y"), and present error distribution to stakeholders.`,
      difficulty: 'intermediate' as const,
      category: 'Interpretation',
    },
    {
      id: 3,
      question: `Design a comprehensive regression evaluation framework for a critical application (e.g., medical dosage prediction, infrastructure load forecasting). What metrics would you track, how would you analyze residuals, and what safety checks would you implement before deployment?`,
      expectedAnswer: `**Core Metrics**: 1) **Accuracy**: MAE (interpretable), RMSE (penalizes large errors), R² (variance explained), MAPE (percentage errors), Max Error (worst case), 2) **Reliability**: Confidence intervals on predictions, prediction intervals (uncertainty), calibration plots. **Residual Analysis**: 1) Check for bias (mean residual ≈ 0), 2) Homoscedasticity (constant variance across prediction range), 3) Normality (Q-Q plot), 4) No patterns vs. predictions or time, 5) Outlier investigation. **Safety Checks**: 1) **Validation**: Out-of-sample test on completely different data source, 2) **Boundary testing**: Performance at extreme values, 3) **Stability**: Consistent performance across subgroups (demographics, time periods), 4) **Error modes**: Identify when model fails catastrophically, 5) **Human review**: Domain expert validation of predictions, 6) **Monitoring**: Real-time prediction tracking, alert on distribution shifts, 7) **Fallback**: Default to conservative baseline when uncertainty high. **Documentation**: Clear documentation of limitations, known failure modes, and appropriate use cases.`,
      difficulty: 'advanced' as const,
      category: 'Production',
    },
  ],
};
