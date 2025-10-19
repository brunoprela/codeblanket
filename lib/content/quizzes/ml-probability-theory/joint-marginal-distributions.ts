/**
 * Quiz questions for Joint & Marginal Distributions section
 */

export const jointmarginaldistributionsQuiz = [
  {
    id: 'q1',
    question:
      'Explain the relationship between joint, marginal, and conditional distributions. How do you derive each from the others?',
    hint: 'Think about the chain rule and marginalization.',
    sampleAnswer:
      'These three distributions are interconnected through fundamental probability rules. Joint distribution P(X,Y) is the foundation - it specifies probability for every combination of X and Y values. From joint, we get marginal by summing/integrating: P(X) = Σ_y P(X,Y) or ∫ P(X,Y)dy. This "marginalizes out" Y. From joint, we get conditional: P(X|Y) = P(X,Y)/P(Y). Conversely, joint from conditional and marginal: P(X,Y) = P(X|Y)P(Y) (chain rule). All three perspectives are equivalent: (1) Joint tells you everything. (2) Marginal + Conditional also tell you everything via chain rule. (3) In ML: Joint P(X,Y) models features and label together. Marginal P(X) is feature distribution. Conditional P(Y|X) is our prediction model. We typically model P(Y|X) directly in supervised learning, rather than full joint P(X,Y), because we only need predictions, not feature generation.',
    keyPoints: [
      'Joint P(X,Y): probability of both together',
      'Marginal P(X) = Σ_y P(X,Y): integrate out other variable',
      'Conditional P(X|Y) = P(X,Y)/P(Y): probability given information',
      'Chain rule: P(X,Y) = P(X|Y)P(Y)',
      'ML: Usually model P(Y|X) not full joint P(X,Y)',
    ],
  },
  {
    id: 'q2',
    question:
      'If covariance is zero, does that mean the variables are independent? Explain with a concrete counterexample.',
    sampleAnswer:
      "No! Zero covariance only means no LINEAR relationship. Variables can have zero covariance but still be dependent through nonlinear relationships. Counterexample: Let X ~ Uniform(-1, 1), and Y = X². Then X and Y are clearly dependent - knowing X perfectly determines Y. However, Cov(X,Y) = E[XY] - E[X]E[Y] = E[X³] - E[X]E[X²]. Since X is symmetric around 0, E[X] = 0 and E[X³] = 0 (odd function). Therefore Cov(X,Y) = 0 - 0 = 0, despite complete dependence! Why? Covariance measures linear relationship. The relationship Y=X² is perfectly symmetric: positive and negative deviations cancel out in the covariance calculation. In ML implications: (1) Correlation/covariance matrices miss nonlinear dependencies. (2) Need other measures for nonlinear relationships (mutual information, etc.). (3) Zero correlation ≠ irrelevant feature. (4) Scatter plots reveal nonlinear relationships that correlation misses. Always visualize data, don't just rely on correlation!",
    keyPoints: [
      'Zero covariance ≠ independence',
      'Covariance only measures LINEAR relationships',
      'Example: Y = X² has Cov(X,Y) = 0 despite dependence',
      'Symmetric nonlinear relationships have zero covariance',
      'Must check for nonlinear dependencies separately',
    ],
  },
  {
    id: 'q3',
    question:
      'In machine learning, we typically model P(Y|X) rather than the full joint P(X,Y). Explain why this is more practical and what we lose by not modeling the joint.',
    sampleAnswer:
      "Modeling P(Y|X) (discriminative approach) is more practical than full joint P(X,Y) (generative approach) because: (1) Dimension reduction - if X has 100 features, joint P(X,Y) requires modeling 101-dimensional space. Conditional P(Y|X) only models how Y depends on X. (2) We only need predictions - don't care about generating X, just predicting Y given X. (3) Fewer parameters - conditional model is simpler, easier to train. (4) Computational efficiency - discriminative models (logistic regression, neural networks) scale better. What we lose: (1) Can't generate synthetic data - joint P(X,Y) allows sampling both X and Y together. (2) No uncertainty about X - discriminative assumes X is given, doesn't model X uncertainty. (3) Missing data handling - generative models handle missing features better. (4) Semi-supervised learning - generative can leverage unlabeled data via P(X). Trade-off: Discriminative (P(Y|X)) is usually better for pure prediction. Generative (P(X,Y)) better when you need synthesis, missing data handling, or semi-supervised learning. Most modern ML uses discriminative for efficiency.",
    keyPoints: [
      'Discriminative P(Y|X): model predictions only',
      'Generative P(X,Y): model full joint distribution',
      'Discriminative advantages: simpler, fewer parameters, efficient',
      'Generative advantages: can generate data, handle missing values',
      'Trade-off based on task requirements',
    ],
  },
];
