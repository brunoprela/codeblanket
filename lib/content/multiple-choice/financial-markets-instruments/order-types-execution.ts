import { MultipleChoiceQuestion } from '@/lib/types';

export const orderTypesExecutionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'fm-1-11-mc-1',
    question:
      'You need to buy 10,000 shares urgently. Stock trades at $100.00 bid / $100.05 ask with 5,000 shares available at ask. If you use a market order, approximately what price will you pay for all shares?',
    options: [
      '$100.00 (bid)',
      '$100.05 (full ask)',
      '$100.08 (above ask)',
      '$101.00 (much higher)',
    ],
    correctAnswer: 2,
    explanation:
      'Market order "walks the book": First 5,000 shares at $100.05, next level might be $100.10 for 3,000 shares, final 2,000 at $100.15. Weighted average: ~$100.08. This is slippage - paying progressively worse prices to fill entire order. For large orders, use limit or iceberg to avoid walking book.',
  },
  {
    id: 'fm-1-11-mc-2',
    question:
      'An iceberg order to buy 100,000 shares shows only 5,000 shares visible at any time. What is the primary advantage?',
    options: [
      'Lower fees',
      'Faster execution',
      'Hides true order size from HFTs',
      'Better price improvement',
    ],
    correctAnswer: 2,
    explanation:
      'Iceberg advantage: Hides large size. If HFTs see "Buy 100K", they front-run (buy ahead, sell higher). Iceberg shows only 5K → looks like small order → no front-running → better average price. Trade-off: Slower execution (must wait for book to refill between clips).',
  },
  {
    id: 'fm-1-11-mc-3',
    question:
      'A VWAP algorithm targets executing 100,000 shares over the trading day. Historical volume shows 30% trades in first hour, 20% mid-day, 50% in final hour. How many shares should execute in the first hour?',
    options: [
      '20,000 (even split)',
      '30,000 (match volume)',
      '33,333 (equal thirds)',
      '50,000 (half)',
    ],
    correctAnswer: 1,
    explanation:
      'VWAP executes proportional to volume: 30% of volume → execute 30% of order = 30,000 shares in first hour. This matches market rhythm, minimizes market impact. TWAP would do 33,333 (even split). VWAP is better for liquid stocks where volume patterns are predictable.',
  },
  {
    id: 'fm-1-11-mc-4',
    question:
      'During the 2010 Flash Crash, many stop-loss orders at 5% below market triggered, selling into the crash. Dow dropped 9% before recovering. An investor with $100K position and 5% stop-loss got stopped out at:',
    options: [
      '$95,000 (5% loss as intended)',
      '$91,000 (9% loss at bottom)',
      '$100,000 (no loss)',
      '$105,000 (made money)',
    ],
    correctAnswer: 1,
    explanation:
      '5% stop triggered at $95K, but became market order in illiquid conditions → executed at crash bottom (\$91K, 9% loss). Recovered to $98K within minutes. Problem: Stop-loss becomes market order → walks book in illiquid crash → worse price than stop level. Solution: Use stop-limit (but risk no fill) or volatility-adjusted stops.',
  },
  {
    id: 'fm-1-11-mc-5',
    question:
      'A participation rate algorithm is set to trade 10% of market volume. Current market volume: 1,000 shares/minute. How many shares/minute should the algorithm execute?',
    options: [
      '100 shares/minute (10% of 1,000)',
      '1,000 shares/minute (match market)',
      '500 shares/minute (half)',
      '10 shares/minute (1%)',
    ],
    correctAnswer: 0,
    explanation:
      'Participation rate: Execute as % of real-time market volume. Market: 1,000 shares/min → algorithm: 10% × 1,000 = 100 shares/min. Advantage: Adapts to changing liquidity (if volume drops to 500, algo slows to 50). Disadvantage: Slow in illiquid periods. Good for large orders without urgency.',
  },
];
