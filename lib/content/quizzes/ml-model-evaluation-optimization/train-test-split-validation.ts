export const trainTestSplitValidationQuiz = {
  title: 'Train-Test Split & Validation - Discussion Questions',
  questions: [
    {
      id: 1,
      question: `Explain why simply splitting data randomly is insufficient for time series financial data. Design a proper validation strategy for a stock price prediction model that will be deployed in production, including how you would handle walk-forward validation and prevent data leakage.`,
      expectedAnswer: `For time series financial data, random splitting violates temporal ordering and creates data leakage (using future information to predict the past). A proper strategy includes: 1) Time-based split where training always precedes validation/test chronologically, 2) Walk-forward validation with expanding or rolling windows to simulate real trading, 3) Gap between train and test to prevent autocorrelation leakage, 4) Multiple time periods to test across different market regimes, 5) Strict feature computation using only past data (no lookahead). Example: Train on 2018-2020, validate on 2021, test on 2022, with a 5-day gap between periods.`,
      difficulty: 'advanced' as const,
      category: 'System Design',
    },
    {
      id: 2,
      question: `You're building a medical diagnosis model where missing a positive case (cancer) is far more critical than a false alarm. How would you design your train-validation-test split strategy, and what additional considerations would you include beyond standard splitting?`,
      expectedAnswer: `Key considerations: 1) Use stratified splitting to maintain the (likely imbalanced) disease prevalence across splits, 2) Ensure sufficient positive cases in validation set for reliable evaluation (minimum 30-50), 3) Consider temporal validation if treatment protocols change over time, 4) Include diverse patient demographics in all splits, 5) Use separate hospital/geographic test set to check generalization, 6) Implement nested CV for hyperparameter tuning to avoid optimistic bias. Additionally: track both sensitivity (recall) as primary metric, set different decision thresholds on validation set, and validate on external dataset before deployment.`,
      difficulty: 'advanced' as const,
      category: 'Production',
    },
    {
      id: 3,
      question: `A data scientist reports 99% accuracy on their model, but when deployed to production, performance drops to 60%. Walk through a systematic debugging process to identify potential issues related to data splitting and evaluation strategy.`,
      expectedAnswer: `Systematic debugging checklist: 1) **Data Leakage**: Check if test data was used in preprocessing (scaling fitted on all data, feature engineering using future information), 2) **Distribution Shift**: Verify train and test come from same distribution (check feature distributions, temporal changes), 3) **Label Leakage**: Ensure no target information in features, 4) **Evaluation Metric**: Confirm metric matches business goal (accuracy misleading for imbalanced data), 5) **Test Set Size**: Check if test set is too small or unrepresentative, 6) **Overfitting to Validation**: Model tuned extensively on validation set without fresh test evaluation, 7) **Implementation Bugs**: Verify production preprocessing matches training. Most common culprit: data leakage through improper preprocessing.`,
      difficulty: 'intermediate' as const,
      category: 'Debugging',
    },
  ],
};
