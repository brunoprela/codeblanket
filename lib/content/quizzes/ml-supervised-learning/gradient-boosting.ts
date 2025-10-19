/**
 * Discussion Questions for Gradient Boosting
 */

import { QuizQuestion } from '../../../types';

export const gradientboostingQuiz: QuizQuestion[] = [
  {
    id: 'gradient-boosting-q1',
    question:
      'Explain the fundamental difference between boosting (Gradient Boosting) and bagging (Random Forests). Why does boosting typically achieve higher accuracy?',
    hint: 'Think about how trees are built and what each tree learns.',
    sampleAnswer:
      'Bagging (Random Forests) builds trees independently in parallel, each on random bootstrap sample. Trees are decorrelated through random sampling, and averaging reduces variance. Boosting (Gradient Boosting) builds trees sequentially, with each tree fitting the residuals (errors) of the current ensemble. This is fundamentally different: boosting trees explicitly target mistakes. Why boosting wins: it reduces both bias and variance. Initial trees reduce bias by capturing signal. Later trees reduce variance by smoothing predictions (via learning rate). Boosting adaptively focuses on hard examples (large residuals), while bagging treats all samples equally. However, boosting requires careful tuning (learning rate, depth) to prevent overfitting, whereas Random Forest is more robust with defaults. Boosting is sequential (slower), bagging is parallel (faster). For maximum accuracy, use boosting. For fast training and robustness, use Random Forest.',
    keyPoints: [
      'Bagging: independent trees, averaging reduces variance',
      'Boosting: sequential trees, each fits residuals',
      'Boosting adaptively focuses on hard examples',
      'Boosting reduces bias and variance',
      'Boosting requires more tuning than Random Forest',
    ],
  },
  {
    id: 'gradient-boosting-q2',
    question:
      'Discuss the role of learning rate (shrinkage) in Gradient Boosting. Why is the tradeoff between learning rate and number of trees important?',
    hint: 'Consider what happens with very small vs very large learning rates.',
    sampleAnswer:
      'Learning rate ν controls how much each tree contributes: F_m = F_{m-1} + ν·h_m. Small ν (e.g., 0.01) means each tree makes tiny corrections, requiring many trees but generalizing better. Large ν (e.g., 1.0) means aggressive updates, needing fewer trees but risking overfitting. The key insight: small learning rate with many trees performs better than large learning rate with few trees, even with same training time. Why? Slow learning allows the model to explore solution space more carefully, avoiding local optima. Practical strategy: use ν=0.1 as default, then tune. For production, use ν=0.01-0.05 with 500-2000 trees and early stopping. The tradeoff: computation time vs accuracy. Small ν needs more trees (longer training) but achieves better test performance. Modern implementations (XGBoost, LightGBM) are fast enough that small learning rates are feasible. Always pair low learning rate with early stopping to find optimal number of trees automatically.',
    keyPoints: [
      'Learning rate controls tree contribution',
      'Small LR + many trees > large LR + few trees',
      'Small LR explores solution space carefully',
      'Typical: 0.01-0.1, tune with early stopping',
      'Tradeoff: training time vs accuracy',
    ],
  },
  {
    id: 'gradient-boosting-q3',
    question:
      'Compare XGBoost, LightGBM, and CatBoost. When would you choose each implementation?',
    hint: 'Think about their different strengths and use cases.',
    sampleAnswer:
      'XGBoost is most popular and well-tested: excellent general-purpose choice, strong regularization (L1/L2), good documentation, wide adoption (Kaggle). Best for: most applications, moderate-sized datasets (<1M rows), when you need proven reliability. LightGBM is fastest: uses leaf-wise tree growth (vs level-wise), gradient-based sampling, exclusive feature bundling. Can be 10x faster than XGBoost on large data. Best for: large datasets (>1M rows), when training speed matters, production systems needing frequent retraining. However, leaf-wise growth can overfit small datasets. CatBoost handles categorical features natively: automatic categorical encoding using target statistics, prevents overfitting via ordered boosting, good defaults. Best for: datasets with many categorical features (no manual encoding needed), small datasets (less prone to overfitting), when you want minimal preprocessing. Practical advice: Start with XGBoost for reliability. Switch to LightGBM if training is too slow. Use CatBoost if many categorical features. All three typically achieve similar accuracy with proper tuning; choice depends on data characteristics and computational constraints.',
    keyPoints: [
      'XGBoost: general-purpose, most popular, proven',
      'LightGBM: fastest, best for large datasets',
      'CatBoost: handles categorical features, good defaults',
      'All achieve similar accuracy when tuned',
      'Choose based on data size and feature types',
    ],
  },
];
