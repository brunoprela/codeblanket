import { MultipleChoiceQuestion } from '@/lib/types';

export const predictiveModelingTradingMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'pmt-mc-1',
      question: 'What is walk-forward validation?',
      options: [
        'Random train-test split',
        'Train on past data, test on future data, slide forward',
        'Validate by walking through office',
        'Test on same data as training',
      ],
      correctAnswer: 1,
      explanation:
        'Walk-forward: Train on [t-252:t], test on [t:t+21], slide by 21 days. Respects temporal ordering, prevents lookahead bias. Essential for time series. Random split uses future to predict past (unrealistic).',
    },
    {
      id: 'pmt-mc-2',
      question:
        'What is typical directional accuracy for stock prediction models?',
      options: [
        '90-95% (highly accurate)',
        '70-80% (good accuracy)',
        '52-58% (slightly better than random)',
        '30-40% (worse than random)',
      ],
      correctAnswer: 2,
      explanation:
        'Stock returns nearly random walk. Best models: 52-58% directional accuracy. 55% accuracy with proper position sizing generates positive Sharpe. 70%+ accuracy suggests overfitting or lookahead bias. Markets are efficient—huge alpha hard to find.',
    },
    {
      id: 'pmt-mc-3',
      question: 'Why retrain models regularly (e.g., every 21 days)?',
      options: [
        'To waste computing power',
        'Markets change—model parameters need updating',
        'Models get bored',
        'No reason',
      ],
      correctAnswer: 1,
      explanation:
        'Markets evolve: volatility changes, correlations shift, regimes change. Model trained on 2020 (COVID) performs poorly in 2023 (normal). Regular retraining adapts to current conditions. 21 days balances adaptation (catch changes) vs stability (avoid noise). Monthly retraining typical.',
    },
    {
      id: 'pmt-mc-4',
      question: 'What is ensemble modeling in trading?',
      options: [
        'Using one model',
        'Combining multiple models with weighted average',
        'Trading with friends',
        'Using only technical indicators',
      ],
      correctAnswer: 1,
      explanation:
        'Ensemble: Train multiple models (XGBoost, RF, Logistic), combine predictions with weights (e.g., 0.5, 0.3, 0.2). Reduces overfitting, improves robustness. Typical improvement: 2-5% accuracy vs single model. Diversification principle applies to models like assets.',
    },
    {
      id: 'pmt-mc-5',
      question: 'How many features should you use to avoid overfitting?',
      options: [
        '5-10 (too few)',
        '50-100 (reasonable)',
        '500-1000 (too many)',
        'As many as possible',
      ],
      correctAnswer: 1,
      explanation:
        'Rule: Features < samples/10. With 2000 samples, max 200 features. Practical: 50-100 features work well. More features = overfitting risk. Use feature importance, remove low-importance features. Quality > quantity. 50 good features beat 500 random features.',
    },
  ];
