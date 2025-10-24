import { MultipleChoiceQuestion } from '@/lib/types';

export const reinforcementLearningTradingMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'rlt-mc-1',
      question: 'What is the state in RL trading?',
      options: [
        'Only current price',
        'Price + indicators + position + balance',
        'Nothing',
        'Random values',
      ],
      correctAnswer: 1,
      explanation:
        'State: All information agent needs to decide. Includes: price, volume, RSI, MACD, position (shares owned), balance (cash), recent returns. Typically 10-20 features. Good state design critical—too little info (bad decisions), too much (slow learning).',
    },
    {
      id: 'rlt-mc-2',
      question: 'What are actions in trading RL?',
      options: ['Do nothing', 'Buy, Sell, Hold', 'Only buy', 'Random'],
      correctAnswer: 1,
      explanation:
        'Actions: Discrete (Buy/Sell/Hold) or Continuous (position size -100% to +100%). Discrete simpler, continuous more flexible. Can also: Buy/Sell 25%/50%/100%. More actions = more flexibility but slower learning. Start with 3 actions.',
    },
    {
      id: 'rlt-mc-3',
      question: 'What is a good reward for RL trading?',
      options: [
        'Only final profit',
        'Sharpe ratio (risk-adjusted return)',
        'Number of trades',
        'Random',
      ],
      correctAnswer: 1,
      explanation:
        'Best reward: Sharpe ratio (return / volatility). Encourages profit AND risk management. Bad: only final PnL (sparse, hard to learn). Also good: incremental PnL per step. Add penalties: large drawdowns, excessive trading. Reward engineering crucial for RL success.',
    },
    {
      id: 'rlt-mc-4',
      question: 'What is DQN?',
      options: [
        'Deep Q-Network: RL with neural network Q-function',
        'Stock ticker',
        'Random algorithm',
        'No training',
      ],
      correctAnswer: 0,
      explanation:
        'DQN: Deep Q-Network. Uses neural network to approximate Q(state, action) = expected return. Combines Q-learning + deep learning. Breakthrough for RL (Atari games). For trading: learns which action (buy/sell/hold) maximizes future returns. More stable than policy gradient methods.',
    },
    {
      id: 'rlt-mc-5',
      question: 'What is the main challenge of RL for trading?',
      options: [
        'Too easy',
        'Sample efficiency: limited historical data, expensive to collect',
        'Always works',
        'No challenges',
      ],
      correctAnswer: 1,
      explanation:
        'Main challenge: Sample efficiency. RL needs millions of experiences. Trading: limited historical data (years not millions), expensive/risky to collect new data (real money). Markets non-stationary (change). Solutions: Careful reward design, transfer learning, combine with supervised. Most firms use supervised, not pure RL.',
    },
  ];
