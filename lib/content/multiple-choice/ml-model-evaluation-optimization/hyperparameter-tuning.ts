export const hyperparameterTuningMultipleChoice = {
  title: 'Hyperparameter Tuning - Multiple Choice Questions',
  questions: [
    {
      id: 1,
      question:
        'What is the main advantage of Random Search over Grid Search for hyperparameter tuning?',
      options: [
        'Random Search always finds better parameters',
        'Random Search explores more values of important parameters with same budget',
        'Random Search is more systematic',
        "Random Search doesn't need cross-validation",
      ],
      correctAnswer: 1,
      explanation:
        "Random Search samples random combinations, so it tries more distinct values of each parameter compared to Grid Search which discretizes the space. If one parameter is unimportant, Random Search doesn't waste iterations on it while exploring more values of important parameters.",
      difficulty: 'intermediate' as const,
      category: 'Search Methods',
    },
    {
      id: 2,
      question: 'Why should you never tune hyperparameters using the test set?',
      options: [
        'It takes too long to compute',
        "It creates optimistic bias - you're selecting parameters that work well on that specific test set",
        'Test sets are too small',
        'It violates mathematical laws',
      ],
      correctAnswer: 1,
      explanation:
        "Tuning on test set means selecting hyperparameters that happen to work well on that specific test set, possibly by chance. This 'leaks' information about the test set into your model selection, giving optimistically biased performance estimates. Always tune on validation set, test once at the end.",
      difficulty: 'beginner' as const,
      category: 'Methodology',
    },
    {
      id: 3,
      question:
        'For a Random Forest with 10 hyperparameters and limited compute budget, which approach would be most efficient?',
      options: [
        'Grid Search with 5 values per parameter (9.7M combinations)',
        'Bayesian Optimization with 50-100 trials',
        'Try each parameter individually',
        'Use default parameters without tuning',
      ],
      correctAnswer: 1,
      explanation:
        "Bayesian Optimization builds a probabilistic model of the objective function and intelligently samples promising regions. It's highly sample-efficient, often finding good solutions in 50-100 trials vs thousands for Grid/Random Search. Perfect for limited budgets and many hyperparameters.",
      difficulty: 'advanced' as const,
      category: 'Strategy',
    },
    {
      id: 4,
      question:
        'What is the purpose of Successive Halving in hyperparameter search?',
      options: [
        'To try only half of the possible combinations',
        'To quickly eliminate poor configurations by training on small data first',
        'To use half the training data',
        'To reduce the number of hyperparameters',
      ],
      correctAnswer: 1,
      explanation:
        'Successive Halving starts with many candidates on small data samples, eliminates worst performers, gives remaining candidates more data, and repeats. This quickly discards bad configs without wasting compute, focusing resources on promising candidates.',
      difficulty: 'advanced' as const,
      category: 'Optimization',
    },
    {
      id: 5,
      question:
        'Which hyperparameter typically has the most significant impact on neural network training?',
      options: ['Random seed', 'Batch size', 'Learning rate', 'Number of GPUs'],
      correctAnswer: 2,
      explanation:
        'Learning rate is typically the most critical hyperparameter for neural networks. Too high causes divergence, too low results in slow convergence or getting stuck. It affects training dynamics more than most other hyperparameters and often requires careful tuning.',
      difficulty: 'beginner' as const,
      category: 'Neural Networks',
    },
  ],
};
