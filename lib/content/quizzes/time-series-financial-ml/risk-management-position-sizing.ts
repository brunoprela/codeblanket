export const riskManagementPositionSizingQuiz = [
  {
    id: 'rmps-q-1',
    question:
      'Explain Kelly criterion and why you should use fractional Kelly in practice.',
    sampleAnswer:
      'Kelly: f = (p*W - q)/L where p=win rate, W=avg win, L=avg loss, q=1-p. Maximizes long-term growth. Example: 60% win rate, $2 avg win, $1 avg loss → Kelly = (0.6*2 - 0.4)/1 = 0.8 (80% of capital!). Problem: Full Kelly extremely volatile, one bad streak ruins you. Solution: Use 1/4 to 1/2 Kelly (20-40%). Reduces volatility 50%, slightly lower growth. Practical and safe.',
    keyPoints: [
      'Kelly = (p*W - q)/L, maximizes growth',
      'Full Kelly too aggressive (high volatility)',
      'Use 1/4 to 1/2 Kelly in practice',
      'Example: 60% win, 2:1 ratio → 20% position (half Kelly)',
      'Balances growth and risk of ruin',
    ],
  },
  {
    id: 'rmps-q-2',
    question: 'Design complete risk management system for a trading strategy.',
    sampleAnswer:
      'Multi-layer risk: (1) Position: 2% risk per trade, ATR-based stops. (2) Portfolio: Max 20% total risk (10 positions max). (3) Daily: Stop trading if down 5% in one day. (4) Drawdown: Halt if down 15% from peak, reassess. (5) Correlation: Reduce size if positions >50% correlated. (6) Position sizing: Kelly or fixed fraction. (7) Take profit: Scale out at 1R, 2R, 3R. Protects capital, prevents catastrophic loss.',
    keyPoints: [
      'Position: 2% risk, ATR stops',
      'Portfolio: <20% total heat',
      'Daily: Stop if down 5%',
      'Drawdown: Halt at -15%, reassess strategy',
      'Correlation adjustment, scale out profits',
    ],
  },
  {
    id: 'rmps-q-3',
    question: 'What is Value at Risk (VaR) and how do you use it in trading?',
    sampleAnswer:
      'VaR: Maximum expected loss at confidence level. "95% VaR = -3% means 95% of days, portfolio loses < 3%." Calculation: 5th percentile of return distribution. CVaR (conditional VaR): Average loss beyond VaR (tail risk). Use: (1) Set risk limits (VaR < 5%), (2) Position sizing, (3) Stress testing. Example: VaR -$5k → worst expected daily loss. If exceeded, reduce positions. Banks use for capital requirements.',
    keyPoints: [
      'VaR: max loss at confidence level (e.g., 95%)',
      'CVaR: average loss in worst 5% (tail risk)',
      'Use: set risk limits, position sizing, stress tests',
      'Example: 95% VaR = -$5k → typical worst day',
      'Limit: assumes normal distribution (underestimates fat tails)',
    ],
  },
];
