import { MultipleChoiceQuestion } from '@/lib/types';

export const technicalIndicatorsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'ti-mc-1',
    question: 'What does RSI > 70 typically indicate?',
    options: [
      'Price will definitely fall',
      'Overbought condition, potential reversal but not guaranteed',
      'Bullish signal to buy',
      'Indicator is broken',
    ],
    correctAnswer: 1,
    explanation:
      'RSI > 70 indicates overbought condition: recent gains have been strong. This SUGGESTS potential reversal but is NOT guaranteed. In strong uptrends, RSI can stay >70 for weeks. Always confirm with other indicators (trend, volume). RSI is most reliable in ranging markets, less useful in strong trends. Use RSI with ADX: if ADX >25 (strong trend), ignore overbought/oversold signals.',
  },
  {
    id: 'ti-mc-2',
    question: 'What is a Golden Cross?',
    options: [
      'Price crosses above resistance',
      'RSI crosses 50',
      'SMA(50) crosses above SMA(200)',
      'Volume increases 2x',
    ],
    correctAnswer: 2,
    explanation:
      'Golden Cross: SMA(50) crosses above SMA(200), bullish long-term signal. Indicates uptrend beginning. Historical accuracy: ~55-60% (slightly better than random). Issue: Lagging indicator, often late to the move (10-20% already gained). Use for trend confirmation, not early entry. Opposite: Death Cross (SMA50 < SMA200), bearish signal. Combine with other indicators for better accuracy.',
  },
  {
    id: 'ti-mc-3',
    question: 'What does a MACD bullish crossover indicate?',
    options: [
      'MACD line crosses above Signal line, potential upward momentum',
      'Price crosses above MACD',
      'MACD reaches 100',
      'Signal line turns red',
    ],
    correctAnswer: 0,
    explanation:
      'MACD bullish crossover: MACD line (12-26 EMA difference) crosses above Signal line (9-period EMA of MACD). Indicates momentum shifting upward. Histogram turns positive. Best used with trend confirmation (price above MA200). In choppy markets, generates false signals. Accuracy improves 20-30% when combined with price above long-term MA and volume confirmation. Wait for histogram to stay positive for 2-3 days for confirmation.',
  },
  {
    id: 'ti-mc-4',
    question: 'Why use ATR (Average True Range) in trading?',
    options: [
      'To predict future prices',
      'To measure volatility and set appropriate stop-losses',
      'To identify overbought conditions',
      'To calculate moving averages',
    ],
    correctAnswer: 1,
    explanation:
      'ATR measures volatility, NOT direction. Used for: (1) Stop-loss placement: Stop = entry ± 2×ATR (accounts for normal fluctuation). (2) Position sizing: Higher ATR = higher risk = smaller position. (3) Breakout validation: Move > 1.5×ATR = significant, < 1×ATR = noise. (4) Volatility regimes: ATR increasing = volatility expanding, reduce position size. Example: SPY ATR = $4, set stops at entry ± $8 to avoid getting stopped out by normal moves.',
  },
  {
    id: 'ti-mc-5',
    question: 'What is the advantage of combining multiple indicators?',
    options: [
      'Makes trading 100% profitable',
      'Reduces false signals by requiring confirmation from multiple sources',
      'Allows trading without stop-losses',
      'Increases the number of trades',
    ],
    correctAnswer: 1,
    explanation:
      "Multiple indicators provide confirmation, reducing false signals by 20-40%. Example: RSI <30 (oversold) alone = 50% win rate. RSI <30 + MACD bullish cross + price above MA200 = 60-65% win rate. Tradeoff: Fewer signals (miss some opportunities) but higher quality. Common approach: Trend filter (MA) + Momentum (RSI/MACD) + Volume confirmation. Never rely on single indicator. But don't use too many (>5 indicators = overfitting).",
  },
];
