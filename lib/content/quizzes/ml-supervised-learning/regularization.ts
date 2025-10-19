/**
 * Discussion Questions for Regularization
 */

import { QuizQuestion } from '../../../types';

export const regularizationQuiz: QuizQuestion[] = [
  {
    id: 'regularization-q1',
    question:
      'Explain why Lasso (L1 regularization) can set coefficients exactly to zero while Ridge (L2 regularization) only shrinks them toward zero. Use both geometric intuition and mathematical reasoning in your explanation.',
    hint: 'Think about the shapes of L1 and L2 penalty regions in coefficient space and where they intersect with the loss function contours.',
    sampleAnswer:
      "The difference between Lasso and Ridge in producing exact zeros stems from the geometric shape of their penalty regions in coefficient space. The L2 penalty (\\(\\sum \\beta_j^2\\)) creates a circular (in 2D) or spherical (in higher dimensions) constraint region - smooth and differentiable everywhere. The L1 penalty (\\(\\sum |\\beta_j|\\)) creates a diamond shape (in 2D) or polytope (in higher dimensions) with corners at the axes - these corners are non-differentiable points where coordinates equal zero.\n\nWhen minimizing the regularized loss function, we're finding where the original loss function's contours (ellipses for least squares) first touch the penalty region. For Ridge, the smooth circular boundary means this contact almost never occurs exactly at a coordinate axis (where a coefficient would be zero) - instead, it happens at some point where all coefficients are small but non-zero. The gradient of the L2 penalty (2β) is continuous and can balance the loss gradient at any point on the sphere.\n\nFor Lasso, the diamond shape has corners extending along the coordinate axes. These corners are highly likely to be where the loss function contours first touch the constraint region, especially in high dimensions. At a corner on an axis (say β₁=0, β₂≠0), the L1 penalty is non-differentiable, allowing it to exactly zero out β₁ while keeping β₂ active. Mathematically, the subgradient of |β| at zero is the interval [-1, 1], which can exactly balance the loss gradient, producing a stable solution with that coefficient at precisely zero.\n\nAnother perspective: during gradient descent, the L2 penalty adds a term -2αβ to the gradient, which shrinks β proportionally - the closer to zero, the smaller the shrinkage, approaching but never reaching zero. The L1 penalty adds -α·sign(β), a constant magnitude regardless of β's size. This constant push can drive small coefficients to exactly zero and keep them there, as the L1 penalty provides no distinction between small and zero values once |β| < threshold.\n\nPractical implication: Lasso performs automatic feature selection by producing sparse solutions, while Ridge keeps all features with small weights.",
    keyPoints: [
      'L2 penalty creates smooth circular/spherical constraint region; L1 creates diamond with corners at axes',
      'Loss contours typically touch L1 region at corners (axes) where coefficients are exactly zero',
      'L2 smooth everywhere: coefficients shrink proportionally toward zero but rarely reach it',
      'L1 non-differentiable at zero: subgradient interval [-1,1] allows stable zero solutions',
      'Gradient descent: L2 adds -2αβ (proportional), L1 adds -α·sign(β) (constant)',
      'L1 constant push can drive small coefficients to exactly zero and maintain them there',
      'Practical result: Lasso produces sparse models (many zeros), Ridge does not',
    ],
  },
  {
    id: 'regularization-q2',
    question:
      'When would you prefer Ridge over Lasso, and vice versa? Discuss the scenarios, data characteristics, and practical considerations that would guide your choice. Include examples from different domains.',
    hint: 'Consider factors like multicollinearity, interpretability needs, true underlying sparsity, and stability requirements.',
    sampleAnswer:
      'The choice between Ridge and Lasso depends critically on the problem structure and practical requirements. Use Ridge when: (1) **All or most features are relevant** - if the true model includes contributions from many features, Lasso\'s aggressive feature elimination could discard useful information. Ridge keeps all features with appropriate shrinkage. (2) **Features are highly correlated** - with multicollinearity, Lasso arbitrarily picks one feature from a correlated group and zeros the others, leading to unstable selections across different data samples. Ridge handles correlated features more gracefully by shrinking their coefficients together. (3) **Stability is critical** - Ridge produces more stable coefficient estimates that don\'t change dramatically with small data perturbations. Example: In macroeconomic forecasting with many correlated indicators (GDP, employment, manufacturing), Ridge maintains contributions from all economic factors rather than arbitrarily selecting one.\n\nUse Lasso when: (1) **Sparse ground truth** - if truly only a few features matter among many candidates, Lasso\'s feature selection identifies them efficiently. (2) **Interpretability is paramount** - a model with 10 non-zero coefficients out of 1000 features is far easier to explain to stakeholders than one with 1000 tiny coefficients. (3) **Computational efficiency for deployment** - fewer features means faster inference. (4) **Features are largely independent** - reduces the arbitrary selection problem. Example: In genomics, predicting disease from 20,000 genes where perhaps 10-50 are causal. Lasso identifies the relevant genes for further biological investigation, providing interpretable results for researchers.\n\nDomain-specific examples:\n\n**Finance - Ridge**: Building a multi-factor model for stock returns using various market factors, fundamentals, and technical indicators. Many factors have some predictive power (market beta, size, value, momentum, volatility). Ridge incorporates all signals with appropriate weights rather than arbitrarily excluding some. The ensemble of weak predictors often outperforms aggressive selection.\n\n**Medicine - Lasso**: Predicting disease from patient symptoms and test results. With 100 potential predictors but perhaps only 5-10 truly diagnostic factors, Lasso creates an interpretable rule: "Patient likely has condition X if they have symptoms A, B, and D with these lab values." Doctors can understand and verify the logic, unlike a model using all 100 features with tiny weights.\n\n**Text Classification - Ridge**: Sentiment analysis where many words contribute to overall sentiment. Words like "excellent," "good," "terrible," "disappointing" all matter. Ridge keeps the full vocabulary with appropriate weights, capturing nuanced sentiment from multiple signals.\n\n**High-dimensional sensor data - Lasso**: With 1000 sensors monitoring a manufacturing process, perhaps only 20 are actually informative about product quality. Lasso identifies the critical sensors, enabling cheaper monitoring systems and focusing maintenance efforts.\n\n**Practical hybrid**: Use ElasticNet when you want feature selection (L1) but have correlated features requiring stability (L2). It combines benefits by using both penalties.',
    keyPoints: [
      'Ridge: use when most features relevant, high multicollinearity, need stability',
      'Lasso: use when sparse ground truth, need interpretability, features independent',
      'Finance example (Ridge): multi-factor models with many weak predictors',
      'Medicine example (Lasso): diagnostic models needing interpretable rules',
      'Ridge handles correlated features gracefully; Lasso arbitrarily picks one',
      'Lasso enables feature selection for computational efficiency and understanding',
      'ElasticNet combines both when you need selection despite correlations',
      'Consider domain requirements: explainability vs. predictive power',
    ],
  },
  {
    id: 'regularization-q3',
    question:
      "Explain why feature scaling is crucial before applying regularization. What happens if you don't scale features? How does this relate to the interpretation of the regularization strength parameter α?",
    hint: 'Consider what regularization penalizes and how feature scales affect coefficient magnitudes.',
    sampleAnswer:
      "Feature scaling is absolutely critical before regularization because regularization penalizes coefficient magnitudes, and coefficient magnitudes are directly determined by feature scales. Without scaling, features measured in different units receive unfair penalty amounts, completely breaking the regularization mechanism.\n\nConsider a concrete example: predicting house prices with features [square footage, number of bedrooms]. Square footage ranges 1000-5000, while bedrooms range 1-5. If we fit without scaling, the coefficient for square footage might be β₁=100 (price increases $100 per sq ft), while bedrooms gets β₂=50000 (price increases $50k per bedroom). The coefficients have these magnitudes purely due to the feature scales, not their true importance. Ridge penalty would be α(100² + 50000²) ≈ α(2.5×10⁹), dominated by the bedroom coefficient. Ridge would heavily shrink β₂ (bedrooms) to reduce the penalty, even though bedrooms might be more important! Lasso would almost certainly zero out β₁ first, eliminating square footage information.\n\nAfter standardization (mean=0, std=1), both features have comparable scales. Now coefficients reflect true relative importance rather than arbitrary measurement units. A coefficient of 0.5 for square footage and 0.8 for bedrooms genuinely indicates bedrooms have stronger effect. The regularization penalty treats both fairly: α(0.5² + 0.8²) = α(0.89), with shrinkage proportional to actual importance.\n\nMathematically: For a feature xⱼ with scale sⱼ, the corresponding coefficient βⱼ scales as 1/sⱼ to maintain the same prediction xⱼβⱼ. The L2 penalty then becomes βⱼ² ∝ 1/sⱼ², meaning features with smaller scales get larger penalties purely due to units! This makes the regularization strength α impossible to interpret - α=1 might be strong for large-scale features but weak for small-scale features.\n\nWith standardization (all features have std=1), α has consistent meaning across features. An α that shrinks one coefficient by 50% applies roughly 50% shrinkage to all, based on their relationship with the target, not their measurement units. This makes α interpretable and comparable across different datasets and problems.\n\nPractical implications:\n1. **Always use StandardScaler or similar before Ridge/Lasso** - sklearn does NOT do this automatically for LinearRegression\n2. **Fit scaler on training data only** - using test set info would be data leakage\n3. **Apply same scaling to new data** - use scaler.transform(), not fit_transform()\n4. **Check feature distributions** - extreme outliers can still affect scaling\n5. **Intercept handling** - don't regularize the intercept term (sklearn handles this)\n\nCommon mistake: Fitting regularization on raw features, then wondering why seemingly important features get eliminated. The model wasn't eliminating importance - it was eliminating arbitrary unit choices! Financial features are particularly prone to this: a feature measured in millions vs. another in percentages will have vastly different coefficient scales.",
    keyPoints: [
      'Regularization penalizes coefficient magnitudes; magnitudes depend on feature scales',
      'Without scaling: coefficients reflect measurement units, not true importance',
      'Example: sq. footage (1000s) vs. bedrooms (1-5) have vastly different coefficient scales',
      'L2 penalty dominated by large-scale features, unfairly shrinking them regardless of importance',
      'Standardization (mean=0, std=1) puts features on equal footing',
      'After scaling: α has consistent, interpretable meaning across all features',
      'Always: fit scaler on training data only, apply to train and test',
      'Common mistake: raw features → misleading feature importance and selection',
    ],
  },
];
