export const liveTradingPaperTradingQuiz = [
  {
    id: 'ltpt-q-1',
    question:
      'Design complete transition plan from backtest to live trading. What are the key stages?',
    sampleAnswer:
      'Transition: (1) Backtest: Validate on 5+ years, Sharpe>1, DD<20%. (2) Walk-forward: Out-of-sample test, verify no overfitting. (3) Paper trading: 1-3 months live simulation, track slippage. (4) Small live: Start 10% capital, 1 month validation. (5) Scale gradually: If profitable, increase to 25%, 50%, 100% over 3-6 months. (6) Monitor: Daily review, adjust if Sharpe drops. Never skip stages—most failures from rushing to live. Psychology hardest: paper feels easy, live is stressful. Expect 20-30% worse performance live.',
    keyPoints: [
      'Backtest (5+ years) → Walk-forward → Paper (1-3 months)',
      'Start live with 10% capital, scale gradually',
      'Monitor daily, expect 20-30% degradation',
      'Never skip paper trading stage',
      'Psychology biggest challenge live vs paper',
    ],
  },
  {
    id: 'ltpt-q-2',
    question:
      'What are the main differences between paper and live trading? What surprises most traders?',
    sampleAnswer:
      'Differences: (1) Execution: Paper = perfect fills, Live = slippage (0.1%), partial fills, rejections. (2) Costs: Paper ignores costs, Live = commissions + data fees. (3) Psychology: Paper = no emotion, Live = fear (panic exits), greed (oversize). (4) Errors: Broker API failures, internet outages. (5) Market impact: Large orders move price. Surprises: (1) Performance 20-30% worse, (2) Emotional stress unbearable, (3) Stupid mistakes (wrong quantity, direction), (4) Draw downs feel much worse. Most traders quit after first live drawdown. Preparation and small size crucial.',
    keyPoints: [
      'Execution: slippage, partial fills, rejections',
      'Psychology: fear and greed (biggest factor)',
      'Performance: 20-30% worse than paper',
      'Surprises: emotional stress, stupid mistakes',
      'Start small, expect drawdowns',
    ],
  },
  {
    id: 'ltpt-q-3',
    question:
      'Design risk control system for live trading. What limits and failsafes are essential?',
    sampleAnswer:
      'Multi-layer controls: (1) Position: Max 20% per stock, 2% risk per trade. (2) Portfolio: Max 10 positions, total exposure <100%. (3) Daily: Stop if down 5%, review strategy. (4) Drawdown: Halt at -15% from peak, reassess. (5) Kill switch: Manual emergency stop. (6) Time-based: No trading first/last 5 minutes (volatile). (7) Order validation: Confirm quantity, price before submit. (8) Reconciliation: Check positions every 5 minutes. (9) Alerts: Email on errors, large losses. (10) Independent review: Weekly performance review. Test all failsafes in paper. One mistake can blow account—paranoid is good.',
    keyPoints: [
      'Position: max 20% per stock, 2% risk per trade',
      'Daily: stop at -5% loss',
      'Drawdown: halt at -15%, reassess',
      'Kill switch, order validation, reconciliation',
      'Test all failsafes, paranoia is good',
    ],
  },
];
