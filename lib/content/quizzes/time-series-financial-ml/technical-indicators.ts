export const technicalIndicatorsQuiz = [
  {
    id: 'ti-q-1',
    question:
      'Design a multi-indicator trading system combining trend, momentum, and volatility signals. How do you weight different indicators, handle conflicting signals, optimize parameters without overfitting, and validate effectiveness?',
    sampleAnswer:
      'Multi-indicator system: (1) Indicators: Trend: SMA(50)/SMA(200) cross, ADX>25. Momentum: RSI(14) <30 (oversold) >70 (overbought), MACD cross. Volatility: Bollinger Band %B, ATR for stops. (2) Weighting: Majority voting: 3/5 indicators agree = trade. Weighted: Trend weight=0.4, Momentum=0.3, Volatility=0.3 (trend most important). Hierarchical: Only trade if trend confirmed, then check momentum. (3) Conflicting signals: RSI overbought but trend up: Follow trend (trend > mean-reversion). MACD bearish but approaching BB lower: Wait for alignment. Use confidence score: If 2/5 agree = low confidence (skip), 4/5 = high (trade). (4) Parameter optimization: Walk-forward: Optimize on [t-504:t-252], validate [t-252:t], test [t:t+63]. Grid search RSI period [10,14,18], MACD [8-12, 21-26, 6-9]. Constraint: Max 5 parameters to optimize (prevent overfitting). Use BIC penalty for complexity. (5) Validation: Out-of-sample Sharpe > 0.8, Win rate >50%, Max drawdown <20%. Monte Carlo: Randomize trade order, 1000 simulations, check if Sharpe stays positive in 95% of simulations. Reality check: Test on different asset (if works on SPY, test on QQQ). Typical performance: Multi-indicator better than single by 10-20% Sharpe.',
    keyPoints: [
      'Combine trend (SMA, ADX), momentum (RSI, MACD), volatility (BB, ATR)',
      'Weight: trend 40%, momentum 30%, volatility 30% OR use majority voting',
      'Optimize with walk-forward, max 5 parameters, use BIC penalty',
      'Validate: Sharpe >0.8, Monte Carlo 1000 sims, test on different assets',
      'Multi-indicator improves Sharpe 10-20% vs single indicator',
    ],
  },
  {
    id: 'ti-q-2',
    question:
      'RSI shows overbought (>70) but price continues rising. Explain when technical indicators fail, how to identify false signals, and strategies to improve reliability (divergences, multi-timeframe analysis).',
    sampleAnswer:
      'Technical indicator failures: (1) Why indicators fail: RSI overbought in strong uptrend: Price can stay "overbought" for weeks during bull markets. Indicator shows extreme but trend continues. Reason: RSI measures recent momentum, not trend strength. In strong trends, "oversold" and "overbought" lose meaning. (2) False signals: MACD crossover in ranging market: Generates whipsaws (buy/sell/buy repeatedly). Bollinger Band touches in trending market: Price bounces off band but doesn\'t reverse. Moving average crosses in choppy periods: Multiple crosses with small gains. (3) Identifying false signals: Check trend strength first: ADX <20 = weak trend = indicators unreliable. Use ADX >25 filter before trading RSI/MACD. Volume confirmation: RSI overbought + volume declining = weak signal (likely reversal). RSI overbought + volume increasing = strong momentum (trend continues). (4) Divergences: Bullish divergence: Price makes lower low, RSI makes higher low = reversal signal. Bearish divergence: Price makes higher high, RSI makes lower high = weakness. Divergences improve reliability by 20-30%. (5) Multi-timeframe: Daily RSI overbought but 4-hour RSI neutral: Daily shows bigger picture, wait for 4H confirmation. Trade when indicators align across timeframes (daily + 4H + 1H all agree). (6) Improvements: Combine with price action: RSI overbought + price forms bearish engulfing = strong signal. Dynamic parameters: In high volatility, use RSI(10), in low use RSI(18). Regime filtering: Only use RSI in mean-reverting regimes, ignore in strong trends.',
    keyPoints: [
      'Indicators fail in strong trends: RSI overbought but price rises for weeks',
      "False signals: MACD whipsaws in range, BB touches don't reverse in trends",
      'Divergences: price vs RSI disagreement = reversal signal (20-30% better)',
      'Multi-timeframe: align daily + 4H + 1H for confirmation',
      'Improvements: Add volume, price action, dynamic parameters, regime filters',
    ],
  },
  {
    id: 'ti-q-3',
    question:
      'Compare lagging indicators (MA, MACD) vs leading indicators (RSI, Stochastic). Which are better for different strategies (trend-following, mean-reversion)? How to combine for optimal results?',
    sampleAnswer:
      'Lagging vs Leading indicators: (1) Lagging (MA, MACD, ADX): Definition: Based on past prices, confirm existing trends. Advantage: Less false signals, follow established trends. Disadvantage: Late entries, miss initial moves. Example: SMA(50) cross SMA(200) confirms trend but entry is 10% from bottom. Use case: Trend-following strategies. Want confirmation before entering. Willing to miss first 20% for lower risk. Performance: Lower win rate (45-50%) but large wins when trend established. (2) Leading (RSI, Stochastic, Volume): Definition: Predict potential reversals before they happen. Advantage: Early entries, catch moves from beginning. Disadvantage: Many false signals, whipsaws in trends. Example: RSI <30 signals oversold, but price can fall another 20% (2020 crash). Use case: Mean-reversion strategies in ranging markets. Try to catch bottoms/tops. Accept higher risk for early entry. Performance: Higher win rate (55-60%) but small wins, frequent trades. (3) Strategy selection: Trend-following: Use lagging indicators (MA, MACD, ADX). Enter on confirmation, ride trend. Sharpe: 0.8-1.2. Mean-reversion: Use leading indicators (RSI, Stochastic). Enter on extremes in ranges. Sharpe: 0.6-1.0. (4) Optimal combination: Lagging for direction + Leading for timing: MA(50) > MA(200) (uptrend confirmed) AND RSI <30 (pullback). Gives trend confirmation + timing. Best of both: trend safety + good entry. Leading to spot + Lagging to confirm: RSI showing bullish divergence (leading) + MACD crosses up (lagging). Reduces false signals by 30-40%. (5) Practical example: Setup: Price above SMA(200) (lagging trend filter). Wait: RSI drops to 30-35 (leading timing). Enter: When RSI turns up + MACD histogram goes positive (confirmation). This combines all three: trend (lagging), reversal (leading), confirmation (lagging).',
    keyPoints: [
      'Lagging: MA, MACD confirm trends, late entry but safer (trend-following)',
      'Leading: RSI, Stochastic predict reversals, early entry but false signals (mean-reversion)',
      'Trend-following: use lagging (45-50% win rate, large wins)',
      'Mean-reversion: use leading (55-60% win rate, small wins)',
      'Optimal: lagging for direction + leading for timing (reduces false signals 30-40%)',
    ],
  },
];
