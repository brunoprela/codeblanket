export const crossValidationTechniquesQuiz = {
  title: 'Cross-Validation Techniques - Discussion Questions',
  questions: [
    {
      id: 1,
      question: `Design a cross-validation strategy for a trading algorithm that predicts daily stock returns. Explain why standard k-fold CV would be inappropriate and detail the specific CV approach you would use, including how many folds, what type of split, and how to handle market regime changes.`,
      expectedAnswer: `Standard k-fold is inappropriate because it violates temporal ordering (trains on future, tests on past). Proper approach: 1) **TimeSeriesSplit with expanding window**: Start with 6 months training, test on next month, expand training window each fold, 2) **Number of folds**: 5-10 depending on data length, 3) **Purging/Embargo**: Add 5-10 day gap between train/test to prevent autocorrelation leakage, 4) **Regime handling**: Ensure each fold contains different market conditions (bull/bear), or use regime-specific CV, 5) **Walk-forward**: Retrain model at each time step to simulate production, 6) **Metrics**: Track Sharpe ratio, max drawdown, and win rate across folds. This ensures realistic performance estimates that account for temporal dependencies.`,
      difficulty: 'advanced' as const,
      category: 'Time Series',
    },
    {
      id: 2,
      question: `You have 500 samples for a binary classification problem with 10% positive class. Compare the pros and cons of using 5-fold stratified CV versus Leave-One-Out CV (LOOCV). Which would you choose and why? What if you had 10,000 samples instead?`,
      expectedAnswer: `**5-fold Stratified CV**: Pros - Computationally efficient (5 model trainings), maintains 10% positive rate in each fold, lower variance across runs, practical for hyperparameter tuning. Cons - Each fold has limited positive examples (~10), estimates may be noisy. **LOOCV**: Pros - Maximum training data per iteration (499 samples), nearly unbiased, deterministic. Cons - Computationally expensive (500 trainings), high variance in estimates, each test is single sample. **For 500 samples**: Choose 10-fold stratified CV (better than 5-fold here) - balances bias-variance tradeoff, ensures ~50 positive examples across validation. **For 10,000 samples**: Definitely 5-fold stratified CV - LOOCV becomes prohibitively expensive, and with larger dataset the bias reduction from LOOCV isn't worth the computational cost.`,
      difficulty: 'intermediate' as const,
      category: 'Strategy',
    },
    {
      id: 3,
      question: `Explain the concept of nested cross-validation and when it's necessary. Provide a specific example where not using nested CV would give optimistically biased performance estimates, and show how nested CV solves this problem.`,
      expectedAnswer: `Nested CV provides unbiased performance estimates when tuning hyperparameters. **Problem**: Standard approach - split data into train/test, use CV on train for hyperparameter tuning, report CV performance. This is biased because hyperparameters were chosen to maximize CV performance on that specific data. **Nested CV solution**: Outer loop for performance estimation, inner loop for hyperparameter tuning. **Example**: Random Forest with 100 hyperparameter combinations. Without nesting: Try all 100 on 5-fold CV, pick best (e.g., AUC=0.85), report 0.85. This is optimistic! With nesting: Outer 5-fold, each fold independently runs inner 5-fold CV to select best hyperparameters, then tests on outer fold. Average outer scores (e.g., AUC=0.82) is unbiased. Difference shows optimism from hyperparameter selection. Essential when: comparing models with tuning, reporting to stakeholders, estimating real-world performance.`,
      difficulty: 'advanced' as const,
      category: 'Methodology',
    },
  ],
};
