import { MultipleChoiceQuestion } from '@/lib/types';

export const liveTradingPaperTradingMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'ltpt-mc-1',
    question: 'What is paper trading?',
    options: [
      'Trading paper',
      'Simulated trading with fake money to test strategy',
      'Only live trading',
      'Random trading',
    ],
    correctAnswer: 1,
    explanation:
      'Paper trading: Simulated with fake money, no real risk. Test strategy and system without losing capital. Pros: Safe, learn system. Cons: No emotions, perfect fills, unrealistic. Do 1-3 months paper before live. All serious traders paper trade first. Most brokers offer paper accounts.',
  },
  {
    id: 'ltpt-mc-2',
    question: 'How much capital to start live trading?',
    options: [
      '100% immediately',
      '10-20% to start, scale gradually',
      '0% never trade',
      'Borrow money',
    ],
    correctAnswer: 1,
    explanation:
      'Start small: 10-20% of planned capital. Test live execution, emotions, slippage. If profitable 1-3 months → increase to 50%. Then 100% over 6-12 months. Never risk all capital immediately. Gradual scaling essential. Most failures from starting too large.',
  },
  {
    id: 'ltpt-mc-3',
    question: 'What is a daily loss limit?',
    options: [
      'No limit',
      'Stop trading if down X% in one day (e.g., 5%)',
      'Unlimited losses',
      'Random rule',
    ],
    correctAnswer: 1,
    explanation:
      'Daily loss limit: Stop trading if down 5% in one day. Prevents revenge trading, catastrophic loss. Example: $100k account, down $5k → stop. Review what went wrong. Essential risk control. Many blow-ups from ignoring daily limits, trying to recover losses.',
  },
  {
    id: 'ltpt-mc-4',
    question:
      'What typically happens to strategy performance in live trading vs paper?',
    options: [
      'Improves 50%',
      'Degrades 20-30% due to slippage, costs, emotions',
      'Exactly the same',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Live performance 20-30% worse than paper. Causes: (1) Slippage 0.1% per trade, (2) Commissions, (3) Emotions (panic exits), (4) Partial fills. Example: Paper Sharpe 1.5 → Live 1.0. Expect this. Plan for degradation. If much worse → fix strategy or execution.',
  },
  {
    id: 'ltpt-mc-5',
    question: 'What is the hardest part of live trading?',
    options: [
      'Technical setup',
      'Psychology: managing fear and greed',
      'Buying computer',
      'Nothing is hard',
    ],
    correctAnswer: 1,
    explanation:
      'Psychology hardest: Fear (panic sell at bottom), Greed (oversize positions), Revenge trading (try to recover losses). Paper trading: no emotion. Live: every loss hurts. Solution: (1) Strict rules, (2) Small size, (3) Accept losses. Many profitable paper traders fail live due to emotions. Mental discipline > technical skills.',
  },
];
