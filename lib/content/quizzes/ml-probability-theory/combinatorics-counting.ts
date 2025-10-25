/**
 * Quiz questions for Combinatorics & Counting section
 */

export const combinatoricscountingQuiz = [
  {
    id: 'q1',
    question:
      'Explain the fundamental difference between permutations and combinations. When would you use each in a machine learning context?',
    hint: 'Think about whether the order of selection matters.',
    sampleAnswer:
      "The key difference is whether order matters. Permutations count arrangements where order is significant (ABC ≠ BAC), while combinations count selections where order doesn't matter ({A,B,C} = {C,B,A}). Formula-wise: C(n,r) = P(n,r)/r! because we divide out the r! orderings that combinations don't distinguish. In ML contexts: (1) Use permutations when order matters - e.g., sequential feature selection in RNNs, ordering of layers in a pipeline, or ranking features by importance. (2) Use combinations when order doesn't matter - e.g., selecting a subset of features for training, choosing which samples to include in a batch, or selecting hyperparameters to tune (where {learning_rate, batch_size} = {batch_size, learning_rate}). Most feature engineering uses combinations since we typically don't care about the order we add features.",
    keyPoints: [
      'Permutations: order matters, P(n,r) = n!/(n-r)!',
      "Combinations: order doesn't matter, C(n,r) = n!/[r!(n-r)!]",
      'Use permutations for sequential/ordered problems',
      'Use combinations for subset selection',
      'Most ML feature selection uses combinations',
    ],
  },
  {
    id: 'q2',
    question:
      'Why does combinatorial explosion make exhaustive search impractical in machine learning? Give specific examples and explain how ML practitioners address this.',
    sampleAnswer:
      "Combinatorial explosion occurs because the number of possibilities grows exponentially or factorially with the problem size. Examples: (1) Neural architecture search with 5 layers and 15 choices per layer = 15^5 = 759,375 architectures. Testing one per minute = 1.4 years! (2) Feature subset selection with 50 features choosing 10 = C(50,10) ≈ 10 billion subsets. (3) Hyperparameter grid search with 5 params × 10 values each = 100,000 combinations. ML practitioners address this through: (1) Random search - sample randomly instead of exhaustively, (2) Bayesian optimization - use a probabilistic model to guide search intelligently, (3) Gradient-based methods - use gradients instead of discrete search, (4) Early stopping - eliminate poor candidates quickly, (5) Transfer learning - start from known good configurations. The key insight: you can't try everything, so you need smart search strategies.",
    keyPoints: [
      'Combinatorial explosion: possibilities grow exponentially/factorially',
      'Exhaustive search becomes computationally infeasible',
      'Architecture search, feature selection, hyperparameter tuning all affected',
      'Solutions: random search, Bayesian optimization, gradient methods',
      'Smart search strategies are essential in modern ML',
    ],
  },
  {
    id: 'q3',
    question:
      "Pascal\'s Triangle shows binomial coefficients. Explain the property that the sum of row n equals 2^n and what this means for subset counting.",
    hint: 'Think about all possible subsets of a set with n elements.',
    sampleAnswer:
      "Row n of Pascal\'s Triangle contains all binomial coefficients C(n,0), C(n,1), ..., C(n,n). Their sum equals 2^n, which represents the total number of subsets of an n-element set. Here's why: C(n,k) counts subsets of size exactly k. When we sum over all k from 0 to n, we're counting ALL possible subsets (empty set, size 1, size 2, ..., full set). For each of the n elements, we have 2 choices (include it or don't), giving 2^n total subsets. In ML: This is the size of the feature subset search space. With 20 features, there are 2^20 ≈ 1 million possible feature subsets! This is why we can't exhaustively try all subsets. The formula also appears in: (1) Neural network dropout (each neuron on/off), (2) Ensemble methods (subsets of models), (3) Data augmentation (combinations of transformations).",
    keyPoints: [
      'Sum of row n: C(n,0) + C(n,1) + ... + C(n,n) = 2^n',
      '2^n = total number of subsets of n-element set',
      'Each element has 2 choices: include or exclude',
      'In ML: feature subset search space size',
      'Appears in dropout, ensembles, and augmentation',
    ],
  },
];
