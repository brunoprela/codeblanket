export const modelSelectionMultipleChoice = {
  title: 'Model Selection - Multiple Choice Questions',
  questions: [
    {
      id: 1,
      question:
        'Three models achieve similar validation F1 scores (0.84, 0.85, 0.84). Model A predicts in 0.1s, Model B in 2s, Model C in 10s. Your application requires <1s response time. Which model should you deploy?',
      options: [
        'Model B - it has the highest F1 score',
        'Model A - only option meeting latency requirement',
        'Model C - more time means better predictions',
        'Use all three in an ensemble',
      ],
      correctAnswer: 1,
      explanation:
        'Model A is the only option meeting the hard constraint of <1s response time. In production, hard constraints (latency, memory, cost) often dominate selection. A 1% accuracy gain means nothing if the system is too slow for users.',
      difficulty: 'intermediate' as const,
      category: 'Production',
    },
    {
      id: 2,
      question:
        'According to the No Free Lunch theorem, which statement is true?',
      options: [
        'Neural networks are always the best choice',
        'Random Forest always outperforms Logistic Regression',
        'No single algorithm performs best across all possible problems',
        'Gradient Boosting solves all machine learning problems',
      ],
      correctAnswer: 2,
      explanation:
        'The No Free Lunch theorem states that averaged over ALL possible problems, every algorithm performs equally well. Algorithm success depends on match between algorithm assumptions and problem structure. Always try multiple approaches for your specific problem.',
      difficulty: 'intermediate' as const,
      category: 'Theory',
    },
    {
      id: 3,
      question:
        'For a medical diagnosis system that must be interpretable for regulatory compliance, which model is most appropriate?',
      options: [
        'Deep neural network with 100 layers',
        'Logistic regression or decision tree with max_depth=5',
        'Random forest with 1000 trees',
        'Black-box ensemble of all models',
      ],
      correctAnswer: 1,
      explanation:
        'Logistic regression (interpretable coefficients) and shallow decision trees (visible decision rules) provide clear explanations for predictions. Essential for regulatory compliance, stakeholder trust, and debugging. Deep learning and large ensembles are black boxes.',
      difficulty: 'beginner' as const,
      category: 'Interpretability',
    },
    {
      id: 4,
      question:
        "You've compared 5 models using 5-fold cross-validation. Model A and B have similar mean scores but B has much lower standard deviation. What does this suggest?",
      options: [
        'Model A is better',
        'Model B is more stable and robust across different data splits',
        'The models are identical',
        "Standard deviation doesn't matter",
      ],
      correctAnswer: 1,
      explanation:
        'Lower standard deviation across CV folds indicates the model performs consistently regardless of the specific train-test split. This suggests better generalization and more reliable performance. Stability is valuable in production where data distribution may vary.',
      difficulty: 'advanced' as const,
      category: 'Evaluation',
    },
    {
      id: 5,
      question:
        'When comparing models with statistical significance testing, what does a p-value < 0.05 mean?',
      options: [
        'The first model is 5% better',
        'The difference in performance is statistically significant, unlikely due to random chance',
        'The second model is definitely worse',
        'You should always choose the first model',
      ],
      correctAnswer: 1,
      explanation:
        "P-value < 0.05 means there's less than 5% probability the observed difference occurred by random chance. The performance difference is statistically significant. However, still consider practical significance (is the difference large enough to matter?) and other factors (speed, interpretability).",
      difficulty: 'intermediate' as const,
      category: 'Statistics',
    },
  ],
};
