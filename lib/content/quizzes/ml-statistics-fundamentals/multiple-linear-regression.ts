/**
 * Quiz questions for Multiple Linear Regression section
 */

export const multiplelinearregressionQuiz = [
  {
    id: 'q1',
    question:
      'In your multiple regression model predicting salary from years_experience and education_level, the coefficient for years_experience is 5000. What does this mean? Why is the phrase "holding other variables constant" crucial, and what happens if you ignore it?',
    hint: 'Think about confounding and partial effects.',
    sampleAnswer:
      'The coefficient of 5000 means: "**Holding education_level constant**, each additional year of experience increases salary by $5,000." **Why "holding constant" matters**: Multiple regression estimates the **partial effect** - the effect of experience AFTER accounting for education. Without this context: (1) **Misleading interpretation**: "5000 per year" sounds like the total effect, but it\'s only the direct effect. (2) **Ignores confounding**: Experience and education are likely correlated (more experience → more education or vice versa). The 5000 is the effect of experience BEYOND what\'s explained by education. (3) **Different from univariate**: If you ran simple regression (experience only), you\'d get a DIFFERENT coefficient because it would include both direct effect AND indirect effect through education. **Example scenario**: Person A: 10 years exp, Bachelor\'s → $60k. Person B: 11 years exp, Bachelor\'s → $65k (same education!). The $5k difference is the coefficient - pure experience effect with education held constant. **What if you ignore "holding constant"**: You might wrongly conclude the total impact of experience is $5k/year, when actually: Total impact = Direct (5k) + Indirect through education (maybe 2k) = 7k/year. Or you\'d miss that comparing people with different education levels confounds the relationship. **In ML**: This is why feature importance changes when you add/remove features - coefficients represent partial, not total, effects. Always interpret as "**conditional on** other features in the model."',
    keyPoints: [
      'Coefficient is partial effect holding other variables constant',
      'Different from univariate (total) effect',
      'Accounts for confounding between predictors',
      'Crucial for correct interpretation',
      'Total effect = direct + indirect through other variables',
      'Feature importance is conditional on model features',
    ],
  },
  {
    id: 'q2',
    question:
      'Your model has R²=0.82 with 10 predictors. You add 5 more random noise features and R² increases to 0.85. Should you keep the new model? Explain using adjusted R² and the bias-variance tradeoff.',
    hint: 'Consider overfitting and model complexity.',
    sampleAnswer:
      "NO - this is a classic case of overfitting! Analysis: **R² always increases** with more features (even noise) because: R² measures fit on training data. More parameters = more flexibility = better fit to noise. Going from 0.82 to 0.85 seems like improvement, but it's illusory. **Adjusted R² tells the truth**: Formula: R²_adj = 1 - (1-R²)(n-1)/(n-p-1). Penalizes model complexity (p). If noise features added: R² up slightly (0.82→0.85), but p increases significantly (10→15). R²_adj likely DECREASES despite R² increasing! **Bias-Variance Tradeoff**: Original model (10 features): Some bias, moderate variance, good generalization. New model (15 features with noise): Lower bias on training data, HIGH variance, poor generalization. The noise features add variance without reducing bias meaningfully. **Test set performance**: Original: R²_test ≈ 0.80 (slight drop from 0.82). New: R²_test ≈ 0.70 (large drop from 0.85!). The 0.85 was overfitting - doesn't generalize. **Correct approach**: (1) Use **cross-validation**: Compare CV scores, not training R². (2) Use **adjusted R²**: Penalizes complexity. (3) **Feature selection**: Remove features with p>0.05 or low importance. (4) **Regularization**: Ridge/Lasso automatically handles this. (5) **Information criteria**: AIC/BIC penalize complexity. **Example calculation**: Suppose n=100, p=10: R²_adj = 1-(1-0.82)(99)/(89) = 0.80. With p=15: R²_adj = 1-(1-0.85)(99)/(84) = 0.82. Adjusted R² barely changed despite 50% more features! Not worth the complexity. **Bottom line**: More features ≠ better model. Always validate on hold-out data and use complexity-adjusted metrics.",
    keyPoints: [
      'R² always increases with more features (even noise)',
      'Adjusted R² penalizes complexity - use for comparison',
      'Adding noise features increases variance (overfitting)',
      'Test set performance will be worse despite higher training R²',
      'Use cross-validation, not training R²',
      'Regularization or feature selection to prevent overfitting',
    ],
  },
  {
    id: 'q3',
    question:
      'Two of your features have VIF > 15. Your model has R²=0.90 but the coefficients have huge standard errors and flip signs when you slightly change the data. Should you remove one of the collinear features? What are the tradeoffs?',
    hint: 'Consider prediction vs interpretation goals.',
    sampleAnswer:
      "VIF > 15 indicates severe multicollinearity - the features are nearly redundant. The symptoms (huge SE, unstable coefficients) confirm this is problematic for interpretation. **Should you remove one?** It depends on your goal: **For INTERPRETATION (remove one)**: Multicollinearity makes coefficients meaningless. You can't interpret \"effect of X1 holding X2 constant\" when X1 and X2 always move together! Coefficients are unstable - small data changes cause large swings. Hypothesis tests unreliable - can't determine which features are truly important. **Solution**: Remove one of the correlated features. Coefficients become stable and interpretable. For PREDICTION (maybe keep both): R²=0.90 is excellent for prediction. Multicollinearity doesn't hurt prediction accuracy on similar data. Combined effect of both features is well-estimated, even if individual effects aren't. **However, risks even for prediction**: (1) **Overfitting**: Model may have learned spurious patterns from the correlation. (2) **Extrapolation danger**: Predictions unreliable if new data has different correlation structure. (3) **Model instability**: Small changes in training data cause different models. **Best approaches**: (1) **Remove one**: Choose based on domain knowledge, data availability, or cost. Check R² barely drops. (2) **Combine features**: Create single feature (e.g., average, PCA). Reduces dimensionality, preserves information. (3) **Regularization**: Ridge regression handles multicollinearity beautifully! Shrinks correlated coefficients. (4) **Use tree models**: Random Forests, XGBoost don't care about multicollinearity. **Example decision**: Linear regression for interpretation → Remove one feature. Random Forest for prediction → Keep both (doesn't matter). Ridge regression → Keep both with regularization. **My recommendation**: Given unstable coefficients, remove the feature with: Lower univariate correlation with target, or Higher measurement cost/noise, or Less theoretical importance. Check that R² stays high (likely drops <0.01 since features are redundant). You'll get stable, interpretable model with minimal prediction loss.",
    keyPoints: [
      'VIF > 10-15 indicates severe multicollinearity',
      'Causes unstable, uninterpretable coefficients',
      'For interpretation: remove one feature',
      'For prediction: may keep both (but risky)',
      'Better: combine features, PCA, or Ridge regression',
      'Tree models unaffected by multicollinearity',
    ],
  },
];
