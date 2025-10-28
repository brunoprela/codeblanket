export default {
  id: 'fin-m15-s10-quiz',
  title: 'Risk Budgeting - Quiz',
  questions: [
    {
      id: 1,
      question:
        'A firm has $100M VaR budget. Equity desk has Sharpe ratio 1.5, Fixed Income desk has Sharpe ratio 1.2. How should risk be allocated?',
      options: [
        'Equally ($50M each) for fairness',
        'More to Equity (higher Sharpe = better return per unit risk)',
        'More to Fixed Income (more stable returns)',
        'Based on assets under management',
      ],
      correctAnswer: 1,
      explanation:
        "Risk budgets should be allocated to maximize firm-wide Sharpe ratio, which means allocating more risk to strategies with higher Sharpe ratios. Equity desk generates 1.5 units of return per unit risk vs. 1.2 for Fixed Income, so equity should get larger allocation. Option A (equal allocation) is suboptimal—ignores performance. Option C contradicts the data. Option D (AUM-based) is wrong—AUM doesn't reflect risk/return efficiency. Optimal allocation: Give desk with Sharpe 1.5 roughly 60% of risk budget, Sharpe 1.2 gets 40% (exact allocation depends on correlation). Reallocate quarterly based on realized Sharpe ratios. This is why high-performing desks get more risk capacity—they use it more efficiently. Firms that allocate risk equally are leaving money on table by underfunding best performers.",
    },
    {
      id: 2,
      question:
        'What is the key difference between risk parity and mean-variance optimization?',
      options: [
        'Risk parity uses volatility only; mean-variance uses returns and volatility',
        'Risk parity equalizes risk contributions; mean-variance maximizes return/risk',
        'Risk parity is for stocks only; mean-variance is multi-asset',
        'Both A and B are correct',
      ],
      correctAnswer: 3,
      explanation:
        "Both A and B are correct. Risk parity allocates so each asset contributes equally to portfolio risk (25% stocks, 75% bonds if bonds have 1/3 the vol of stocks). It only uses volatility/correlation, avoiding the need to forecast returns (the hard part). Mean-variance optimization maximizes return/risk by taking positions based on expected returns, volatilities, AND correlations. Option A alone is insufficient—doesn't explain the goal difference. Option B alone is insufficient—doesn't explain the input difference. Option C is wrong—both apply to any assets. Risk parity advantage: Avoids return forecasting (garbage in, garbage out problem). Disadvantage: Ignores valuations (will allocate to bonds at 0% yield). Mean-variance advantage: Optimizes for expected return. Disadvantage: Extremely sensitive to return inputs—small forecast changes cause massive allocation swings.",
    },
    {
      id: 3,
      question:
        'A volatility-targeting strategy has 10% vol target. Realized vol increases from 15% to 30% (doubles). What should the strategy do?',
      options: [
        'Maintain current leverage—vol will revert',
        'Reduce leverage by 50% (from 0.67x to 0.33x)',
        'Increase leverage to take advantage of high vol',
        'Exit all positions until vol normalizes',
      ],
      correctAnswer: 1,
      explanation:
        'Vol targeting maintains constant volatility by adjusting leverage inversely to realized vol. Target leverage = Target Vol / Realized Vol. Initially: 10% / 15% = 0.67x. After vol doubles to 30%: 10% / 30% = 0.33x (cut leverage in half). This automatically deleverages in high-vol periods (crises) and leverages in low-vol periods (calm). Option A (maintain leverage) would result in 20% realized vol (2× target)—violating the strategy. Option C is backwards—vol targeting reduces in high vol. Option D is too extreme. Benefits: (1) Drawdown control (auto-deleverage in crisis), (2) Constant risk budget, (3) Improved Sharpe ratio. Drawbacks: (1) Procyclical (sells into selloffs), (2) Transaction costs from rebalancing, (3) May miss V-shaped recovery. Used by risk parity funds, volatility-targeted ETFs, CTA funds.',
    },
    {
      id: 4,
      question:
        'Marginal contribution to risk (MCTR) for Asset A is 0.15, for Asset B is 0.20. Portfolio weights are 50/50. What action improves risk-adjusted return?',
      options: [
        'Increase Asset A (lower MCTR = adds less risk)',
        'Increase Asset B (higher MCTR = more diversifying)',
        'Depends on expected returns—cannot determine from MCTR alone',
        'Reduce both assets to lower overall risk',
      ],
      correctAnswer: 2,
      explanation:
        "MCTR alone doesn't determine optimal action—need Return/MCTR ratio for both assets. If Asset A has 10% return: 10%/0.15 = 0.67. If Asset B has 12% return: 12%/0.20 = 0.60. Asset A has better return per marginal risk, so increase A. But if Asset B had 15% return: 15%/0.20 = 0.75 > 0.67, increase B instead. Option A assumes lower MCTR is better (wrong—depends on return). Option B assumes higher MCTR indicates diversification (wrong—could indicate correlation). Option D reduces risk but ignores return optimization. Optimal portfolio has equal Return/MCTR ratios across all assets. If Asset A's ratio exceeds Asset B's, shift weight from B to A until ratios equalize. This is foundation of risk budgeting: allocate risk to assets with best return per marginal risk.",
    },
    {
      id: 5,
      question:
        'A portfolio has risk budget of $50M VaR. Current VaR is $45M (90% utilized). The CIO wants to add a new strategy with $8M standalone VaR. What should happen?',
      options: [
        'Approve—firm has $5M VaR capacity remaining',
        'Reject—standalone VaR of $8M exceeds available capacity',
        'Calculate marginal VaR of new strategy to determine impact on portfolio VaR',
        'Request increase in risk budget to $58M',
      ],
      correctAnswer: 2,
      explanation:
        "Standalone VaR is irrelevant—must calculate marginal VaR (incremental impact on portfolio VaR). The new strategy might have $8M standalone VaR but only add $3M to portfolio VaR if negatively correlated with existing positions (diversification). Or it might add $6M if highly correlated. Option A is naive—assumes $5M capacity based on current VaR ignoring new position. Option B uses wrong metric (standalone). Option D is premature—don't know yet if need more budget. Correct process: (1) Calculate portfolio VaR with new strategy included, (2) Marginal VaR = (New portfolio VaR) - (Current VaR) = ($48M - $45M) = $3M, (3) If $48M < $50M budget, approve, else reject or reduce size. This is why pre-trade risk checks use marginal VaR, not standalone VaR. Diversification benefits only realized at portfolio level.",
    },
  ],
} as const;
