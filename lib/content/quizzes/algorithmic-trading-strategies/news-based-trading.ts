export const newsBasedTradingQuiz = [
  {
    id: 'ats-9-1-q-1',
    question:
      'Design earnings surprise trading strategy: (1) Surprise calculation, (2) Entry rules, (3) Exit rules. Historical data: 5% surprise → +2% drift over 3 days. Calculate expected return and Sharpe ratio.',
    sampleAnswer: `**Earnings Surprise Strategy**: (1) Surprise = (actual - expected) / |expected|; (2) Entry: if |surprise| > 5%, enter direction of surprise (beat→long, miss→short); (3) Exit: after 3 days or if reverse 1%. Expected: 2% return per trade, 60% win rate, volatility 3%, Sharpe = (2% × 0.60 - 0.5% × 0.40) / 3% = 0.33 per trade, annualized ≈ 2.0 (60 earnings/year).`,
    keyPoints: [
      'Surprise calculation: (actual - expected) / |expected|; threshold 5% to filter significant surprises',
      'Entry: immediate market-on-open after earnings (pre-market if available); position size 5% of portfolio',
      'Exit: hold 3 days (post-earnings announcement drift), exit earlier if gap-fill (reverse 1%)',
      'Expected return: 2% drift × 60% win rate = 1.2% per trade; with 60 trades/year = 72% annual (before costs)',
      'Sharpe calculation: 1.2% mean per trade, 3% std per trade → 0.40 Sharpe per trade; 60 trades → √60 = 7.75 → Sharpe 3.1 annualized',
    ],
  },
  {
    id: 'ats-9-1-q-2',
    question:
      'Explain latency arbitrage in news trading. Your system has 100μs latency, competitor has 1ms. Calculate profit advantage on $1M order if price moves 10bps in 1ms. How do HFT firms achieve microsecond latency?',
    sampleAnswer: `**Latency Arbitrage**: 100μs vs 1ms (1000μs) = 900μs advantage. In 900μs, you execute before competitor reacts. Profit: price moves 10bps (0.10%) in 1ms; you execute at t=100μs (moved 1bp), competitor at t=1000μs (moved 10bps). Advantage: 9bps × $1M = $900. HFT latency: co-location (exchange servers), FPGA (hardware), microwave/laser (data transmission), kernel bypass networking.`,
    keyPoints: [
      'Latency advantage: 900μs (you 100μs vs competitor 1000μs); in this time, price moves linearly ~9bps',
      'Profit calculation: $1M order × 9bps advantage = $900 per trade; with 100 trades/day = $90K/day',
      'HFT infrastructure: co-location (<1ms to exchange), FPGAs (10-50μs processing), microwave towers (Chicago-NY in 4ms vs 7ms fiber)',
      'Example 2: Fed announcement → price moves 50bps in 10ms; 100μs advantage captures 0.5bps × $10M = $5K',
      'Cost-benefit: latency infrastructure costs $1M+/year, but generates $10M+/year from arbitrage',
    ],
  },
  {
    id: 'ats-9-1-q-3',
    question:
      'Design NLP-based news trading system: (1) News sources (Bloomberg, Twitter), (2) Sentiment analysis (keywords vs BERT), (3) Signal generation, (4) Risk management. How do you avoid false signals?',
    sampleAnswer: `**NLP News Trading System**: (1) Sources: Bloomberg terminal (fastest), Twitter (real-time), SEC filings; (2) Sentiment: BERT/FinBERT (accuracy 85%) vs keywords (70%); (3) Signals: sentiment >0.7 → long, <-0.7 → short, |sentiment|<0.5 → ignore; (4) Risk: position limits 2% per trade, verify with multiple sources, human override for extreme moves. False signal filters: require 2+ sources, minimum sentiment confidence 80%, volume confirmation (volume spike).`,
    keyPoints: [
      'News sources: Bloomberg/Reuters (low latency, high quality), Twitter (fast but noisy), earnings transcripts (delayed but accurate)',
      'Sentiment analysis: FinBERT (transformer, 85% accuracy) better than keywords (70%), but slower (10ms vs 1ms)',
      'Signal generation: sentiment threshold 0.7 (high confidence), require news freshness <1 min, combine with price action confirmation',
      'False signal filters: (1) multi-source verification, (2) sentiment confidence >80%, (3) volume spike confirmation (2x avg volume)',
      'Risk management: max 2% position size per signal, stop-loss 1%, aggregate news exposure <20% of portfolio',
    ],
  },
];
