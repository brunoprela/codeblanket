import { MultipleChoiceQuestion } from '@/lib/types';

export const tradingStrategyDevelopmentMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'tsd-mc-1',
      question: 'What is a momentum strategy?',
      options: [
        'Buy losers, sell winners',
        'Buy winners, sell losers (trend-following)',
        'Never trade',
        'Random trades',
      ],
      correctAnswer: 1,
      explanation:
        'Momentum: Buy recent winners (stocks up 20% in 3 months), sell/short losers. "Trend is your friend." Works because trends persist due to herding, underreaction to news. Typical holding: 1-6 months. Opposite of mean reversion.',
    },
    {
      id: 'tsd-mc-2',
      question: 'What is mean reversion?',
      options: [
        'Follow trends forever',
        'Prices return to average after extremes',
        'Always hold',
        'Buy high, sell low',
      ],
      correctAnswer: 1,
      explanation:
        'Mean Reversion: Prices oscillate around mean. When too high (overbought) → sell, too low (oversold) → buy. Example: Price 2 std dev above 20-day MA → short. Works in range-bound markets. Opposite of momentum.',
    },
    {
      id: 'tsd-mc-3',
      question: 'How do you filter noisy signals?',
      options: [
        'Trade every signal',
        'Apply volatility filter, volume filter, smoothing',
        'Ignore all signals',
        'Flip a coin',
      ],
      correctAnswer: 1,
      explanation:
        'Signal filters: (1) Volatility: only trade when vol < threshold (avoid chaos), (2) Volume: only trade liquid stocks (>1M shares/day), (3) Smoothing: 5-day MA of signals (reduce noise). Improves Sharpe 10-20%. Prevents overtrading in bad conditions.',
    },
    {
      id: 'tsd-mc-4',
      question: 'What is relative strength in multi-asset strategies?',
      options: [
        'Absolute returns',
        'Rank assets, long strongest, short weakest',
        'Trade one asset only',
        'Random selection',
      ],
      correctAnswer: 1,
      explanation:
        'Relative strength: Rank 10 assets by 3-month return. Long top 2 (strongest), short bottom 2 (weakest). Market-neutral (long + short balance). Works because cross-sectional momentum persists. Used by hedge funds. Better than absolute momentum in sideways markets.',
    },
    {
      id: 'tsd-mc-5',
      question: 'What is a regime-based strategy?',
      options: [
        'One strategy for all markets',
        'Switch strategies based on market conditions (trend vs range)',
        'Never change strategy',
        'Random strategy selection',
      ],
      correctAnswer: 1,
      explanation:
        'Regime detection: Identify market state (trending, range-bound, volatile). Use momentum in trends (ADX>25), mean reversion in ranges (ADX<20). Improves Sharpe 20-30%. Methods: HMM, volatility clustering, rolling correlation. Adapts to changing markets.',
    },
  ];
