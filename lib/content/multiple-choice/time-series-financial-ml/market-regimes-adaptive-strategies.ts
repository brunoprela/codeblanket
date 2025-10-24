import { MultipleChoiceQuestion } from '@/lib/types';

export const marketRegimesAdaptiveStrategiesMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'mras-mc-1',
      question: 'What is a market regime?',
      options: [
        'Random behavior',
        'Persistent market condition (trending, ranging, volatile)',
        'Single day',
        'No pattern',
      ],
      correctAnswer: 1,
      explanation:
        'Market regime: Persistent state with characteristic behavior. Main regimes: (1) Trending: Momentum works, ADX>25. (2) Ranging: Mean reversion works, ADX<20. (3) High volatility: Reduce risk, VIX>30. Markets cycle through regimes. Detecting regime improves trading 20-30%.',
    },
    {
      id: 'mras-mc-2',
      question: 'What is ADX used for?',
      options: [
        'Predict price',
        'Measure trend strength (>25 = trending, <20 = ranging)',
        'Calculate returns',
        'Random metric',
      ],
      correctAnswer: 1,
      explanation:
        'ADX (Average Directional Index): Measures trend strength, not direction. ADX>25 = strong trend (momentum works), ADX<20 = weak trend (ranging, mean reversion works). ADX 50+ = very strong trend. Used for regime detection. Combine with +DI/-DI for direction.',
    },
    {
      id: 'mras-mc-3',
      question: 'What is an adaptive trading strategy?',
      options: [
        'One strategy forever',
        'Switches strategies based on market regime',
        'Never changes',
        'Random strategy',
      ],
      correctAnswer: 1,
      explanation:
        'Adaptive: Changes strategy by market condition. Trending → use momentum, Ranging → use mean reversion, High-vol → reduce risk/cash. Improves Sharpe 20-30% vs single strategy. Key: Accurate regime detection + smooth transitions (avoid whipsaw). Better than one-size-fits-all.',
    },
    {
      id: 'mras-mc-4',
      question: 'What is HMM for regime detection?',
      options: [
        'Random model',
        'Hidden Markov Model: learns hidden market states from returns',
        'No model',
        'Stock ticker',
      ],
      correctAnswer: 1,
      explanation:
        'HMM: Statistical model assuming N hidden states (regimes), each with return distribution. Learns from data: State 0 = bull (high return, low vol), State 1 = bear (negative return, high vol), State 2 = sideways. Predicts current regime. Unsupervised, data-driven. Used by quant funds.',
    },
    {
      id: 'mras-mc-5',
      question: 'What strategy to use in high volatility regime?',
      options: [
        'Increase leverage 10x',
        'Reduce risk: smaller positions or go to cash',
        'Ignore volatility',
        'Random trades',
      ],
      correctAnswer: 1,
      explanation:
        'High volatility (VIX>30, vol>30%): Reduce risk. Strategies: (1) Reduce position size to 25-50%, (2) Go to cash, (3) Tighter stops, (4) Avoid new positions. High vol = higher risk of large losses. Preserve capital during chaos. VIX>40 = crisis mode (2008, 2020 COVID).',
    },
  ];
