export const strategyPerformanceEvaluationQuiz = [
  {
    id: 'spe-q-1',
    question:
      'Design complete performance evaluation framework for trading strategy. What metrics matter most?',
    sampleAnswer:
      'Framework: (1) Returns: Total, annual, monthly. (2) Risk-adjusted: Sharpe > 1, Sortino, Calmar > 1. (3) Risk: Max DD < 20%, volatility, ulcer index. (4) Consistency: Win rate > 50%, profit factor > 1.5, avg win/loss > 1.5. (5) vs Benchmark: Alpha > 0, IR > 0.5, beta. Most important: Sharpe (risk-adjusted), Max DD (survivability), Consistency (repeatable). Not just returns—risk matters. 15% return with 30% DD worse than 12% with 10% DD.',
    keyPoints: [
      'Core: Sharpe>1, Max DD<20%, Calmar>1',
      'Consistency: Win rate>50%, Profit factor>1.5',
      'Benchmark: Alpha>0, IR>0.5',
      'Risk-adjusted > absolute returns',
      'Test: multiple periods, robustness checks',
    ],
  },
  {
    id: 'spe-q-2',
    question: 'What is alpha? How do you calculate and interpret it?',
    sampleAnswer:
      'Alpha: Excess return vs benchmark after adjusting for risk (beta). Formula: α = R_strategy - [R_f + β(R_market - R_f)]. Example: Strategy 15%, market 12%, β=1.2, R_f=2% → Expected = 2% + 1.2(12%-2%) = 14%. Alpha = 15% - 14% = 1%. Positive alpha = skill/outperformance. Negative = underperformance. α > 2% excellent, > 5% exceptional. Combine with IR: α=2%, IR=0.8 → consistent skill. α alone can be luck; IR confirms consistency.',
    keyPoints: [
      'Alpha = excess return adjusted for beta',
      'Formula: R - [Rf + β(Rm - Rf)]',
      'Positive alpha = skill/outperformance',
      'α>2% excellent, >5% exceptional',
      'Combine with IR for consistency validation',
    ],
  },
  {
    id: 'spe-q-3',
    question:
      'Compare two strategies: A) Sharpe 1.5, Max DD 15%. B) Sharpe 1.2, Max DD 8%. Which is better?',
    sampleAnswer:
      'Depends on preferences: Strategy A: Higher risk-adjusted returns (1.5 Sharpe), but deeper drawdown (15%). More aggressive. Strategy B: Lower Sharpe (1.2) but shallower DD (8%). More defensive. Choose A if: High risk tolerance, longer horizon, can stomach 15% loss. Choose B if: Conservative, shorter horizon, cannot tolerate 15% loss. Practical: Most investors prefer B—smaller DD = easier to hold, less likely to panic sell. Calmar matters: A = Annual/0.15, B = Annual/0.08 → B likely higher Calmar. Consider: recovery time, investor psychology.',
    keyPoints: [
      'A: Higher Sharpe (1.5), deeper DD (15%) - aggressive',
      'B: Lower Sharpe (1.2), shallower DD (8%) - defensive',
      'Most prefer B: easier to hold, less panic',
      'Consider: Calmar ratio, recovery time',
      'Drawdown often more important than Sharpe for investors',
    ],
  },
];
