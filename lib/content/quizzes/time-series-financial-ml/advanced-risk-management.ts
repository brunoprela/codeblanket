export const advancedRiskManagementQuiz = [
  {
    id: 'arm-q-1',
    question:
      'Compare VaR vs CVaR (Expected Shortfall). Which is better for risk management?',
    sampleAnswer:
      'VaR: Maximum loss at confidence (e.g., 95% VaR = worst case 95% of time). Problem: Ignores tail beyond VaR. CVaR (ES): Average loss beyond VaR (in worst 5%). Better because: (1) Captures tail severity, (2) Coherent risk measure, (3) Actionable. Example: VaR = -$5k, CVaR = -$8k. In crisis, lose $8k on average, not just $5k. Use CVaR for: position limits, stress tests, capital requirements. Industry standard (Basel III).',
    keyPoints: [
      'VaR: max loss at confidence, ignores tail',
      'CVaR: average loss beyond VaR (captures severity)',
      'CVaR coherent (sub-additive), VaR not',
      'Example: VaR -$5k, CVaR -$8k (tail worse)',
      'Use CVaR for limits, Basel III standard',
    ],
  },
  {
    id: 'arm-q-2',
    question:
      'Design comprehensive stress testing framework for trading portfolio.',
    sampleAnswer:
      'Framework: (1) Historical scenarios: 2008 crisis (-50%), COVID (-35%), 2011 flash crash (-20%). (2) Hypothetical shocks: Market -30%, Volatility 3x, Correlations → 1. (3) Factor stress: Interest rates +200bp, USD +20%, Oil -50%. (4) Reverse stress: What causes -50% loss? (5) Frequency: Daily monitoring, weekly detailed. (6) Actions: If scenario loss > 20%, reduce positions 50%. Test: each stock, sector, total portfolio. Document assumptions. Present to risk committee. Protects from tail events.',
    keyPoints: [
      'Historical: 2008, COVID, flash crashes',
      'Hypothetical: -30% shock, vol spike, corr→1',
      'Factor stress: rates, FX, commodities',
      'Reverse: what causes catastrophic loss?',
      'Actions: reduce positions if scenario loss > threshold',
    ],
  },
  {
    id: 'arm-q-3',
    question: 'What is Sortino ratio? When is it better than Sharpe?',
    sampleAnswer:
      'Sortino: (Return - risk_free) / Downside deviation. Only penalizes downside volatility, not upside. Better when: (1) Asymmetric returns (crypto, options), (2) Care about losses not gains, (3) Returns non-normal. Example: Hedge fund with volatile gains, steady losses → Sortino >> Sharpe. Sharpe penalizes both up and down. For long-only equity: Sharpe sufficient. For strategies with positive skew: Sortino better reflects risk. Institutional investors prefer Sortino.',
    keyPoints: [
      'Sortino = return / downside deviation (not total)',
      'Only penalizes losses, not gains',
      'Better for: asymmetric returns, positive skew',
      'Example: volatile upside → Sortino > Sharpe',
      'Preferred by institutional investors',
    ],
  },
];
