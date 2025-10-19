/**
 * Quiz questions for Information Theory Basics section
 */

export const informationtheorybasicsQuiz = [
  {
    id: 'q1',
    question:
      'Explain what entropy measures and why a uniform distribution has maximum entropy. What does this tell us about information and uncertainty?',
    hint: 'Think about predictability and surprise.',
    sampleAnswer:
      'Entropy H(X) measures average uncertainty or "surprise" in a random variable. It quantifies the average number of bits needed to encode outcomes. H(X) = -Σ P(x)log₂P(x). Uniform distribution has maximum entropy because all outcomes are equally likely - we have maximum uncertainty about which will occur. Example: fair die (uniform) has H = log₂(6) ≈ 2.58 bits. Loaded die favoring 6 has lower entropy because outcomes are more predictable. Why maximum? When one outcome is certain (P=1), entropy is 0 - no surprise, no information. When all equally likely, every observation provides maximum information. Information theory principle: uniform distribution is "maximally random" - no outcome is favored. In ML implications: (1) Maximum entropy distributions are least biased given constraints. (2) Regularization toward uniform prevents overfitting. (3) Uniform priors in Bayesian methods express maximum ignorance. (4) Feature with uniform distribution (all values equally common) provides less information than skewed distribution.',
    keyPoints: [
      'Entropy measures average uncertainty/surprise',
      'Uniform distribution has maximum entropy',
      'Maximum entropy = maximum unpredictability',
      'More predictable → lower entropy',
      'Uniform = least biased, maximum information per observation',
    ],
  },
  {
    id: 'q2',
    question:
      'Why do we minimize cross-entropy loss in classification instead of minimizing entropy? What is the relationship between cross-entropy and KL divergence?',
    sampleAnswer:
      "We minimize cross-entropy H(P,Q) = -Σ P(x)log Q(x) where P is true distribution (data labels) and Q is model distribution (predictions). Why not minimize entropy H(P)? Because H(P) is the entropy of true labels, which is constant (doesn't depend on model parameters) - we can't change it! Cross-entropy relationship to KL divergence: D_KL(P||Q) = H(P,Q) - H(P). Since H(P) is constant, minimizing H(P,Q) is equivalent to minimizing D_KL(P||Q). We're making our model distribution Q as close as possible to true distribution P. Why this works: (1) Cross-entropy is differentiable w.r.t. model parameters. (2) Minimizing it maximizes log-likelihood (same as MLE). (3) Convex in many cases (logistic regression). (4) Penalizes confident wrong predictions heavily. In practice: Cross-entropy preferred over accuracy because it's differentiable and provides gradient information about how wrong predictions are, not just that they're wrong.",
    keyPoints: [
      'Minimize H(P,Q) not H(P) because H(P) is constant',
      'H(P,Q) = D_KL(P||Q) + H(P)',
      'Minimizing cross-entropy = minimizing KL divergence',
      'Equivalent to maximum likelihood estimation',
      'Differentiable and provides rich gradient information',
    ],
  },
  {
    id: 'q3',
    question:
      'Mutual information I(X;Y) measures how much information X and Y share. Explain how this is used for feature selection in machine learning.',
    sampleAnswer:
      'Mutual information I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) measures how much knowing Y reduces uncertainty about X (or vice versa). If X and Y are independent, I(X;Y) = 0 - they share no information. If perfectly dependent, I(X;Y) = H(X) = H(Y). For feature selection: We want features X with high I(X;Y) where Y is the label. High MI means feature is informative for prediction. Algorithm: (1) Compute I(X_i; Y) for each feature. (2) Rank features by MI. (3) Select top-k features. (4) Check for redundancy: if I(X_i; X_j) is high, features are redundant - keep only one. Advantages over correlation: (1) Captures nonlinear relationships (correlation only linear). (2) Works for categorical variables. (3) Information-theoretic foundation. Example: In spam detection, I("contains_free"; spam) might be 0.8 bits - very informative. I("email_length"; spam) might be 0.1 bits - less useful. MI-based selection keeps most informative, non-redundant features.',
    keyPoints: [
      'I(X;Y) = information shared between X and Y',
      'High I(feature;label) = informative feature',
      'I(X;Y) = 0 iff independent',
      'Use for feature selection and redundancy detection',
      'Captures nonlinear relationships unlike correlation',
    ],
  },
];
