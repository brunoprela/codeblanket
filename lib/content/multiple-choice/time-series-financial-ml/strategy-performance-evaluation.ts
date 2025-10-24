import { MultipleChoiceQuestion } from '@/lib/types';

export const strategyPerformanceEvaluationMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'spe-mc-1',
      question: 'What is a good Sharpe ratio?',
      options: ['<0', '>1.0', '>10.0', 'Negative'],
      correctAnswer: 1,
      explanation:
        'Sharpe > 1.0 good, > 1.5 excellent, > 2.0 exceptional. S&P 500 ≈ 0.5. Most strategies: 0.8-1.5. HFT can hit 2-3. Sharpe > 3 likely overfit. In-sample often 2x out-of-sample. Target: 1.5 in-sample → 0.75-1.0 live.',
    },
    {
      id: 'spe-mc-2',
      question: 'What is maximum drawdown?',
      options: [
        'Highest price',
        'Largest peak-to-trough decline',
        'Average loss',
        'Random metric',
      ],
      correctAnswer: 1,
      explanation:
        'Max DD: Largest peak-to-trough loss. Example: $100k → $85k → $110k. Max DD = -15%. Critical metric—shows worst loss period. Good: <15%, Acceptable: <20%, Bad: >30%. Deep DD hard to recover (need 43% gain to recover 30% loss). Investors care more about DD than Sharpe.',
    },
    {
      id: 'spe-mc-3',
      question: 'What is profit factor?',
      options: [
        'Random number',
        'Gross profit / Gross loss',
        'Win rate',
        'Sharpe ratio',
      ],
      correctAnswer: 1,
      explanation:
        'Profit Factor: Gross profits / Gross losses. PF = 2.0 means $2 profit for every $1 loss. Good: >1.5, Excellent: >2.0. PF < 1 = losing strategy. Combines win rate and win/loss magnitude. Better than win rate alone (can have 60% win rate but lose money if losses huge).',
    },
    {
      id: 'spe-mc-4',
      question: 'What is alpha?',
      options: [
        'Random letter',
        'Excess return vs risk-adjusted benchmark expectation',
        'Stock ticker',
        'Total return',
      ],
      correctAnswer: 1,
      explanation:
        'Alpha: Excess return after adjusting for market risk (beta). Positive alpha = outperformance/skill. α = R - [Rf + β(Rm-Rf)]. Example: 15% return, β=1.2, market 12%, Rf=2% → α = 15% - 14% = 1%. Hard to generate consistently. Most mutual funds have negative alpha (after fees).',
    },
    {
      id: 'spe-mc-5',
      question: 'What is Calmar ratio?',
      options: [
        'Random metric',
        'Annual return / Maximum drawdown',
        'Win rate',
        'Volatility',
      ],
      correctAnswer: 1,
      explanation:
        'Calmar: Annual return / Max DD. Shows return per unit of worst loss. Example: 15% return, 10% DD → Calmar = 1.5. Good: >1.0, Excellent: >2.0. Focuses on drawdown (key for investors). Hedge funds often optimize for Calmar > Sharpe. Investors tolerate losses better if returns high relative to max loss.',
    },
  ];
