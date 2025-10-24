import { MultipleChoiceQuestion } from '@/lib/types';

export const orderExecutionTradingInfrastructureMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'oeti-mc-1',
      question: 'What is slippage?',
      options: [
        'Falling down',
        'Difference between expected and actual fill price',
        'Commission',
        'Random metric',
      ],
      correctAnswer: 1,
      explanation:
        'Slippage: Actual fill price vs expected. Example: Want to buy at $100, filled at $100.10 → slippage $0.10 (0.1%). Costs money. Caused by: market orders, illiquid stocks, large size. Minimize: use limit orders, trade liquid stocks. Typical: 0.05-0.1% per trade.',
    },
    {
      id: 'oeti-mc-2',
      question: 'What is TWAP execution?',
      options: [
        'Random execution',
        'Time-Weighted Average Price: split order evenly over time',
        'One large order',
        'Never execute',
      ],
      correctAnswer: 1,
      explanation:
        'TWAP: Split large order into small pieces over time. Example: Buy 10,000 shares → 100 shares every minute for 100 minutes. Reduces market impact, avoids slippage. Used by institutions for large orders. Alternative: VWAP (volume-weighted). Prevents moving market with single large order.',
    },
    {
      id: 'oeti-mc-3',
      question: 'What is a kill switch?',
      options: [
        'Power button',
        'Emergency stop for all trading (daily loss limit hit)',
        'Start trading',
        'Random feature',
      ],
      correctAnswer: 1,
      explanation:
        'Kill switch: Emergency stop. Triggers when: (1) Daily loss > 5%, (2) System error, (3) Manual override. Cancels all orders, exits positions, stops new trades. Prevents catastrophic loss. Example: Flash crash, system bug → kill switch protects capital. Essential risk control.',
    },
    {
      id: 'oeti-mc-4',
      question: 'What is paper trading?',
      options: [
        'Trading paper stocks',
        'Simulated trading with fake money (no real risk)',
        'Only live trading',
        'Printing money',
      ],
      correctAnswer: 1,
      explanation:
        'Paper trading: Simulated trading, no real money. Test strategy without risk. Pros: Safe, learn system. Cons: No slippage, no emotions, unrealistic fills. Use to: (1) Test code, (2) Validate strategy, (3) Learn platform. Do 1-3 months paper before live. Most brokers offer paper accounts (Alpaca, TD Ameritrade).',
    },
    {
      id: 'oeti-mc-5',
      question: 'What is order reconciliation?',
      options: [
        'Random process',
        'Comparing internal positions vs broker positions to detect errors',
        'Ignoring positions',
        'No checking needed',
      ],
      correctAnswer: 1,
      explanation:
        "Reconciliation: Compare your system's positions vs broker's actual positions. Detects: failed orders, partial fills, system bugs. Run every 1-5 minutes. If mismatch → alert, investigate. Example: System thinks owns 100 shares, broker shows 90 → error. Critical for production trading to avoid drift.",
    },
  ];
