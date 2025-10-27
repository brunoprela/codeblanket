import { MultipleChoiceQuestion } from '@/lib/types';

export const trendFollowingStrategiesMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'ats-2-1-mc-1',
      question:
        'A 50-day/200-day MA crossover strategy on SPY has a Sharpe ratio of 0.8 in backtest. What is the MOST likely problem reducing live performance?',
      options: [
        'Transaction costs (commission + slippage) not properly modeled',
        'Overfitting to historical data',
        'Look-ahead bias in backtest code',
        'Insufficient capital for position sizing',
      ],
      correctAnswer: 0,
      explanation:
        'MA crossover strategies are well-known and simple (hard to overfit, no look-ahead bias if coded correctly). The main killer is transaction costs: (1) MA systems generate signals infrequently but at key inflection points where slippage is high, (2) Crossing 200-day MA often coincides with high volatility/spreads, (3) SPY is liquid so commission small, but slippage 5-10 bps adds up. A 0.8 Sharpe can easily become 0.3-0.5 after realistic 10 bps total costs per trade. Option B unlikely (simple strategy), C unlikely (straightforward logic), D irrelevant to Sharpe.',
    },
    {
      id: 'ats-2-1-mc-2',
      question:
        'ADX reads 18, price breaks above 20-day high. According to trend-following best practices, what action should you take?',
      options: [
        'Enter long immediately (breakout confirmed)',
        'Wait for ADX to exceed 25 before entering',
        'Enter short (low ADX indicates false breakout)',
        'Enter half position now, add when ADX > 25',
      ],
      correctAnswer: 1,
      explanation:
        "ADX < 20 indicates choppy, non-trending market where breakouts frequently fail (whipsaws). Best practice: only take trend-following signals when ADX > 25 (confirms trend strength). Entering at ADX 18 will likely result in false breakout and stop-out. Option A ignores trend filter (mistake). Option C is backwards (low ADX doesn't predict direction). Option D (half position) is compromise but still violates filter - better to wait for confirmation than suffer multiple whipsaws.",
    },
    {
      id: 'ats-2-1-mc-3',
      question:
        'Turtle Trading rule: Risk 1% per trade, stop at 2× ATR. If ATR = $4 and capital = $100K, how many shares of a $100 stock should you trade?',
      options: ['100 shares', '125 shares', '250 shares', '500 shares'],
      correctAnswer: 1,
      explanation:
        'Risk amount = $100K × 1% = $1,000. Stop distance = 2× ATR = 2 × $4 = $8 per share. Position size = Risk amount / Stop distance = $1,000 / $8 = 125 shares. Verification: 125 shares × $8 stop = $1,000 risk = 1% of capital ✓. Option A (100) risks only $800 (0.8%). Option C (250) risks $2,000 (2%, too much). Option D (500) risks $4,000 (4%, way too much, violates risk management).',
    },
    {
      id: 'ats-2-1-mc-4',
      question:
        'A trend-following strategy has 35% win rate, average winner $1000, average loser $300. What is the profit factor?',
      options: ['1.2x', '1.8x', '2.4x', '3.0x'],
      correctAnswer: 1,
      explanation:
        'Profit Factor = Gross Profit / Gross Loss. Per 100 trades: Winners = 35 trades × $1000 = $35,000. Losers = 65 trades × $300 = $19,500. Profit Factor = $35,000 / $19,500 = 1.79 ≈ 1.8x. This is marginal (barely profitable after costs). Good trend following should have 2.5-3.5x profit factor. This strategy needs improvement: either increase win rate, increase avg winner size, or decrease avg loser size.',
    },
    {
      id: 'ats-2-1-mc-5',
      question:
        'Donchian breakout system: Long when price > 20-day high. Price breaks out on day 21 at $105. On day 22, the 20-day high becomes $106 (day 21 now in lookback). What happens?',
      options: [
        'Entry signal repeats (enter again if not already in)',
        'No new signal (already in position from day 21)',
        'Exit signal (price no longer above 20-day high)',
        'Add to position (pyramiding opportunity)',
      ],
      correctAnswer: 1,
      explanation:
        "This is a common implementation question. Proper Donchian system: (1) Day 21: Price breaks above 20-day high ($105) → Enter long, (2) Day 22: 20-day high updates to $106 (includes day 21), (3) Price still above NEW 20-day high? If yes, stay in position. If no (price < $106), no action yet - wait for exit signal (10-day low break). The system doesn't re-enter (you're already in). Option A wrong (no double entry). Option C wrong (exit only on 10-day low, not entry channel). Option D possible if price > entry + 0.5× ATR, but question doesn't specify pyramiding rules.",
    },
  ];
