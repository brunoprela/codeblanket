export const biasVarianceTradeoffQuiz = {
  title: 'Bias-Variance Tradeoff - Discussion Questions',
  questions: [
    {
      id: 1,
      question: `Your model achieves Train Error=2%, Validation Error=15%. A colleague suggests "just add more data" while another suggests "reduce model complexity." Using bias-variance analysis, determine which suggestion is correct and explain your reasoning. What additional diagnostics would you perform to confirm your decision?`,
      expectedAnswer: `**Analysis**: Train=2%, Val=15% indicates **high variance (overfitting)** - large gap between train and validation. Model fits training data too well but doesn't generalize. **Correct suggestion**: BOTH have merit but for different reasons. **Reduce complexity** (immediate fix): Lower model capacity, add regularization, feature selection, early stopping. This directly addresses overfitting. **More data** (if possible): Can help reduce variance by providing more examples to learn robust patterns, but may not fully solve if model is fundamentally too complex. **Additional Diagnostics**: 1) **Learning curves**: Plot train/val error vs training set size. If curves haven't converged and val error decreasing → more data helps. If converged with large gap → complexity issue, 2) **Validation curve**: Plot error vs model complexity (e.g., tree depth). Find optimal complexity, 3) **Feature importance**: Are low-importance features causing overfitting?, 4) **Cross-validation**: Is 15% consistent across folds? **Decision**: Start with reduce complexity (faster, cheaper). If that underperforms, then add data.`,
      difficulty: 'advanced' as const,
      category: 'Diagnosis',
    },
    {
      id: 2,
      question: `Explain how regularization (L1, L2) addresses the bias-variance tradeoff. Why does increasing regularization strength reduce variance but increase bias? Provide a mathematical and intuitive explanation, including when you might prefer L1 vs L2 regularization.`,
      expectedAnswer: `**Effect**: Regularization adds penalty term to loss function: Loss = DataError + λ * Penalty. As λ increases: bias↑, variance↓. **Mathematical**: Regularization constrains parameter space, forcing parameters toward zero. **L2 (Ridge)**: Penalty=||w||², shrinks all weights proportionally, **L1 (Lasso)**: Penalty=||w||₁, drives weights exactly to zero. **Why Variance Decreases**: 1) Smaller parameter values = less sensitivity to training data fluctuations, 2) Simpler effective model (especially L1 with zero weights), 3) Smoother decision boundaries, 4) Reduced capacity to memorize noise. **Why Bias Increases**: 1) Constrained parameters can't fit training data perfectly, 2) May underfit true relationship, 3) Trades fitting accuracy for generalization. **Intuition**: Unregularized = memorize every detail (high variance). Regularized = learn general patterns (high bias but better generalization). **L1 vs L2**: Use L1 when: 1) Want feature selection (sparse solutions), 2) Many irrelevant features, 3) Interpretability important. Use L2 when: 1) All features relevant, 2) Better for correlated features, 3) Generally more stable. **Optimal**: Elastic Net (combines both).`,
      difficulty: 'intermediate' as const,
      category: 'Theory',
    },
    {
      id: 3,
      question: `Design a systematic approach to finding the optimal model complexity for a regression problem. Your approach should include specific steps, diagnostic plots, and decision criteria. How would you communicate the final model choice to non-technical stakeholders?`,
      expectedAnswer: `**Systematic Approach**: **Step 1 - Baseline**: Train simplest model (linear regression), measure performance. **Step 2 - Complexity Sweep**: Train models of increasing complexity (polynomial degree 1→15, or tree depth 1→20), track train and val error for each. **Step 3 - Generate Diagnostics**: 1) **Model Complexity Curve**: Plot train/val error vs complexity. Identify where val error minimizes, 2) **Learning Curves**: For top 3 complexities, plot train/val vs dataset size, 3) **Bias-Variance Decomposition**: Empirically measure bias² and variance via bootstrap, 4) **Residual Analysis**: Check patterns in errors. **Step 4 - Select**: Choose model where: 1) Val error is minimized, 2) Small gap between train/val, 3) Learning curves have converged, 4) Residuals show no patterns. **Step 5 - Validate**: Confirm on held-out test set. **Non-Technical Communication**: "We tested models from simple to complex. Too simple: misses patterns (65% accuracy). Too complex: memorizes noise (62% on new data). Sweet spot: Medium complexity (78% accurate, stable across time). Think Goldilocks - not too simple, not too complex, just right. This balances fitting your data with predicting future cases."`,
      difficulty: 'advanced' as const,
      category: 'Methodology',
    },
  ],
};
