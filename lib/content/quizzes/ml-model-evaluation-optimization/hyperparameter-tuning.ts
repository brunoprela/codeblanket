export const hyperparameterTuningQuiz = {
  title: 'Hyperparameter Tuning - Discussion Questions',
  questions: [
    {
      id: 1,
      question: `You have a limited budget to tune hyperparameters for a Random Forest model with 10 hyperparameters. Compare Grid Search, Random Search, and Bayesian Optimization for this scenario. Which would you choose and why? Include computational cost analysis and expected quality of results.`,
      expectedAnswer: `**Grid Search**: Tries every combination. With 10 params and 5 values each = 5¹⁰ = 9,765,625 combinations! Prohibitively expensive. Even with 3 values: 3¹⁰ = 59,049 - still too expensive. **Random Search**: Samples N random combinations (e.g., N=100-200). **Pros**: 1) Explores more values of important params, 2) Can stop anytime, 3) Parallelizable, 4) Often finds good solutions in 50-100 iterations. **Cons**: No learning from results, may miss optimal region. **Bayesian Optimization**: Builds probabilistic model of objective function, intelligently samples promising regions. **Pros**: 1) Very sample efficient (50-100 iterations competitive with 1000s of random), 2) Focuses on promising regions, 3) Can handle expensive evaluations. **Cons**: Sequential (less parallelizable), overhead of building surrogate model. **Choice for Limited Budget**: **Bayesian Optimization** because: 1) 10 hyperparams = large search space where intelligence helps, 2) RF training expensive → minimize evaluations, 3) Budget-constrained → need sample efficiency, 4) Can stop early with reasonable solution. **Practical**: Use Random Search (100 iterations) first to explore, then Bayesian Optimization (50 iterations) to refine. Total 150 evaluations vs 59,049 for grid search!`,
      difficulty: 'advanced' as const,
      category: 'Strategy',
    },
    {
      id: 2,
      question: `Explain why tuning hyperparameters on the test set is a critical error that leads to overfitting. Provide a concrete example with numbers showing how this can give misleading results, and explain the proper workflow using train/validation/test splits or nested cross-validation.`,
      expectedAnswer: `**Why It\'s Wrong**: Test set meant to estimate performance on unseen data. Using it for tuning "leaks" information about test set into model selection, making test performance optimistically biased. **Example**: 1) Try 100 hyperparameter combinations, 2) Each evaluated on test set, 3) Choose best: Test Accuracy = 92%, 4) **Problem**: We picked the combination that happened to work best on *this specific* test set, maybe just by luck, 5) True unseen performance might be only 85%, 6) We've "overfit" to the test set without training on it! **Concrete Numbers**: Say true performance is 85% ±3% (noise). After trying 100 combinations, some will score 88-92% just by random variation. We pick 92%, thinking we're great, but it's 85% + lucky noise. **Proper Workflow**: 1) **Three-way split**: Train 60%, Validation 20%, Test 20%, 2) **Tune on validation**: Try all hyperparams, evaluate on validation, pick best, 3) **Final eval on test**: Report test performance ONCE at very end, 4) Or **Nested CV**: Outer loop for testing, inner loop for tuning. **Key**: Test set is sacred - touch it only once at the end, after all development decisions are final.`,
      difficulty: 'intermediate' as const,
      category: 'Methodology',
    },
    {
      id: 3,
      question: `Design a hyperparameter tuning strategy for a production trading algorithm where: (1) model retrains daily with new data, (2) hyperparameters need updating monthly, (3) you have limited compute budget, (4) performance must be stable across market conditions. What approach would you use and why?`,
      expectedAnswer: `**Challenge**: Need automated, efficient, robust tuning for production system. **Strategy**: **1) Coarse-to-Fine Schedule**: Monthly: broad search (20-30 configs), Weekly: local refinement (5-10 configs around current best), Daily: use current best. **2) Online/Adaptive Approach**: **Multi-Armed Bandit**: Treat hyperparameter configs as "arms", allocate compute to promising configurations, explore vs exploit tradeoff. **Thompson Sampling** to balance exploration. **3) Efficient Search**: **Successive Halving**: Start with 30 configs on 1 week of data, eliminate worst 2/3, evaluate remaining on 2 weeks, repeat. Finds good config with <1/10th compute of full grid search. **4) Regime-Aware**: **Market Segmentation**: Detect market regime (bull/bear/volatile), maintain separate hyperparams per regime, switch based on current regime. **5) Safety Constraints**: **Performance Gates**: New hyperparams must beat current by >5% on validation before deployment, **Walk-Forward Validation**: Test on multiple out-of-sample periods, **Ensemble Fallback**: If single model fails, have ensemble ready. **6) Monitoring**: Track hyperparameter performance drift over time, alert if degradation. **Result**: Robust automated system that adapts to markets while minimizing compute and risk.`,
      difficulty: 'advanced' as const,
      category: 'Production',
    },
  ],
};
