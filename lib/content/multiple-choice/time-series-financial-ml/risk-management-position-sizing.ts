import { MultipleChoiceQuestion } from '@/lib/types';

export const riskManagementPositionSizingMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'rmps-mc-1',
      question: 'What is Kelly criterion?',
      options: [
        'Random sizing',
        'Optimal position size for max growth: f = (pW-q)/L',
        'Always use 100% capital',
        'Never risk anything',
      ],
      correctAnswer: 1,
      explanation:
        'Kelly: f = (p*W - q)/L where p=win rate, W=avg win, L=avg loss. Maximizes long-term growth. Example: 60% win, 2:1 ratio → Kelly = 0.4 (40%). But use 1/2 Kelly (20%) to reduce volatility. Full Kelly too aggressive.',
    },
    {
      id: 'rmps-mc-2',
      question: 'What is a typical risk per trade?',
      options: [
        '100% of capital',
        '2% of capital',
        '0.01% of capital',
        'Random amount',
      ],
      correctAnswer: 1,
      explanation:
        '2% per trade is standard. Allows 50 consecutive losses before ruin (unlikely). Aggressive: 5%, Conservative: 1%. Higher risk = higher volatility and drawdown. 2% balances growth and safety. With 2%, 10 positions = 20% max portfolio risk.',
    },
    {
      id: 'rmps-mc-3',
      question: 'What is ATR-based stop loss?',
      options: [
        'Random stop',
        'Stop = Entry - (ATR × Multiplier)',
        'No stop loss',
        'Stop at yesterday low',
      ],
      correctAnswer: 1,
      explanation:
        'ATR stop: Adaptive to volatility. Stop = Entry - 2×ATR (long). High volatility → wider stop (avoid whipsaw). Low volatility → tighter stop. Dynamic, not fixed %. Example: ATR=$2, entry=$100 → stop=$96. Better than fixed 2% in volatile markets.',
    },
    {
      id: 'rmps-mc-4',
      question: 'What is portfolio heat?',
      options: [
        'Temperature of computer',
        'Total % of capital at risk across all positions',
        'One position only',
        'Ignore risk',
      ],
      correctAnswer: 1,
      explanation:
        "Portfolio heat: Sum of all position risks. 10 positions × 2% each = 20% heat. Limit: 20-25% max. Prevents over-leverage. If at limit, can't add new positions. Reduces correlated risks, prevents catastrophic loss. Essential for multi-position strategies.",
    },
    {
      id: 'rmps-mc-5',
      question: 'What is take profit strategy "scale out at 1R, 2R, 3R"?',
      options: [
        'Close entire position at once',
        'Take partial profits at 1×risk, 2×risk, 3×risk',
        'Never take profit',
        'Random exits',
      ],
      correctAnswer: 1,
      explanation:
        'Scale out: If risking $1 (1R), take profit at +$1 (1R), +$2 (2R), +$3 (3R). Example: 50% at 1R, 30% at 2R, 20% at 3R. Locks in gains, lets winners run. Better than all-or-nothing. Balances "cut losses, let winners run" with "take profit before reversal."',
    },
  ];
