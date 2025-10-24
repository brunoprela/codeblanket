export const buildingFinancialAiApplicationsQuiz = [
  {
    id: 'bcap-bfaa-q-1',
    question:
      'Design an AI stock analysis system that combines: technical indicators, fundamental data, news sentiment, and insider trading. How do you: (1) Integrate multiple data sources, (2) Weight different signals, (3) Generate actionable insights, (4) Handle market hours vs after-hours, (5) Avoid liability ("not financial advice")? What accuracy can you realistically achieve?',
    sampleAnswer:
      'Multi-signal analysis system: (1) Data sources: yfinance (price/fundamentals), News API (sentiment), SEC Edgar (insider trades), Alpha Vantage (technical). (2) Signal processing: Technical - calculate RSI, MACD, moving averages (programmatic). Fundamental - extract P/E, debt ratio, revenue growth (from financial statements). Sentiment - LLM analyzes news headlines: positive/negative/neutral + confidence. Insider trading - flag if execs buying/selling (strong signal). (3) Weighting: Ensemble approach: Technical (30%), Fundamentals (30%), Sentiment (20%), Insider (20%). Adjust weights by market conditions: bull market → more technical, recession → more fundamental. (4) LLM synthesis: Feed all signals to Claude: "Analyze AAPL: Technical: RSI=68 (overbought), MACD=positive. Fundamentals: P/E=28 (above sector avg). Sentiment: 70% positive. Insider: CFO bought $1M shares. Generate analysis." (5) Actionable insights: "Bullish: Insider buying + positive sentiment. Risk: Overbought technical. Recommendation: Watch for pullback to $165 entry." (6) Market hours: Real-time during market, cache after-hours, clearly label data staleness. (7) Liability: Every response includes: "This is not financial advice. For informational purposes only. Consult a licensed advisor." Accuracy: Realistic expectation: 55-60% directional accuracy (better than random 50%, worse than perfect). Track record publicly, adjust models based on outcomes. Avoid: Guarantees, specific price targets, urgency ("buy now").',
    keyPoints: [
      'Multi-signal: technical (30%), fundamentals (30%), sentiment (20%), insider (20%)',
      'LLM synthesizes signals into narrative analysis with risks and opportunities',
      'Adjust weights by market conditions (bull vs bear)',
      'Liability: prominent disclaimers, no guarantees, "for informational purposes only"',
      'Realistic accuracy: 55-60% directional, track record publicly',
    ],
  },
  {
    id: 'bcap-bfaa-q-2',
    question:
      "You're building an AI portfolio rebalancing system. User has $100k portfolio (60% stocks, 30% bonds, 10% cash). Target allocation is 70/20/10. AI needs to: (1) Calculate rebalancing trades, (2) Minimize taxes (wash sales, capital gains), (3) Consider transaction costs, (4) Time trades optimally. How do you approach this? Should AI execute trades automatically or require approval?",
    sampleAnswer:
      'Portfolio optimization with constraints: (1) Calculate drift: Current: 60/30/10, Target: 70/20/10. Need: +$10k stocks, -$10k bonds. (2) Tax optimization: Query user\'s cost basis for each position. Prioritize selling: (a) Positions at loss (tax-loss harvesting), (b) Long-term gains (lower tax rate), (c) Avoid short-term gains (<1 year). Check wash sale rule: don\'t buy same security 30 days before/after loss sale. (3) Transaction costs: If using broker API, factor commission ($0-10 per trade). Batch small positions (avoid $5 trade on $50 position). (4) Trade timing: During market hours only, avoid: first/last 30min (high volatility), low liquidity (penny stocks). Use limit orders (not market) to control price. (5) LLM for strategy: Feed portfolio + constraints to Claude: "Generate rebalancing plan minimizing taxes. Portfolio: {positions with cost basis}. Target: 70/20/10." LLM suggests: "Sell Bond A (long-term gain, low tax), Stock B (at loss, harvest), Buy Stock C." (6) Execution: NEVER auto-execute. Always show proposed trades, tax impact ($), user must approve. Use two-factor auth for execution. (7) Validation: Check if proposed trades meet target (within 1%), calculate total tax impact, estimate total cost (commission + slippage). Regulatory: Requires investment advisor license in many jurisdictions. Alternative: Show trade suggestions only, user executes manually via their broker.',
    keyPoints: [
      'Optimize for: target allocation, minimize taxes, reduce transaction costs',
      'Tax optimization: prioritize losses (harvesting), long-term gains, avoid wash sales',
      'Use LLM to suggest trades considering cost basis and constraints',
      'NEVER auto-execute: show plan, tax impact, require user approval + 2FA',
      'Regulatory: may require investment advisor license, consider suggestions-only approach',
    ],
  },
  {
    id: 'bcap-bfaa-q-3',
    question:
      'Design a compliance monitoring system for financial AI. Requirements: detect suspicious trading patterns, flag large transactions (SAR threshold $10k), identify potential insider trading, prevent market manipulation. How do you: (1) Define suspicious patterns, (2) Balance false positives vs false negatives, (3) Handle reporting obligations, (4) Maintain audit trail? What are the legal risks?',
    sampleAnswer:
      'Multi-layer compliance system: (1) Rule-based detection: Hard rules: Transaction >$10k → auto-flag for SAR (Suspicious Activity Report). Wash sale detection: Same security ±30 days. Structuring: Multiple $9k transactions (avoid $10k threshold). (2) ML-based anomaly detection: Train model on historical transactions, flag: unusual amount relative to account history, rapid buy/sell (churning), trades before earnings/news (potential insider). (3) LLM pattern recognition: Feed transaction history to Claude: "Analyze for suspicious patterns: {transactions}. Consider: timing, frequency, amounts." LLM identifies: "3 trades of $9,500 within 10 days (possible structuring)." (4) Balance: False positive tolerance depends on risk: For SAR (legal requirement): Low threshold (flag more), accept 30% false positives. For internal monitoring: Higher threshold, avoid alert fatigue. (5) Reporting: If SAR triggered, generate report within 30 days (legal requirement), file with FinCEN. Log: transaction details, why flagged, resolution. (6) Audit trail: Immutable log (blockchain or append-only DB), track: all transactions, flags raised, investigations, reports filed. Retain 5+ years (regulatory requirement). (7) Human review: All ML flags go to compliance officer, they decide: investigate, file SAR, clear. AI assists, human decides. Legal risks: Failure to file SAR = fines ($25k-100k), criminal charges possible. Over-reporting (too aggressive) = operational burden but legally safe. Recommendation: Err on side of caution, robust compliance program, legal counsel review.',
    keyPoints: [
      'Multi-layer: rule-based (hard rules), ML (anomaly detection), LLM (pattern recognition)',
      'SAR threshold $10k is legal requirement, auto-flag, file within 30 days',
      'Balance: accept higher false positives for regulatory compliance',
      'Human-in-the-loop: AI flags, compliance officer investigates and decides',
      'Legal risks: failure to file SAR = fines/criminal charges, maintain 5+ year audit trail',
    ],
  },
];
