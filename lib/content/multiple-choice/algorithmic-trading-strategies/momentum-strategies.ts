export const momentumStrategiesMC = [
    {
        id: 'ats-6-mc-1',
        question: 'Why do momentum strategies skip the last month when calculating returns?',
        options: ['To reduce transaction costs', 'To avoid short-term reversal effect', 'To increase holding period', 'To reduce volatility'],
        correctAnswer: 1,
        explanation: `**Correct: To avoid short-term reversal effect.** 

Research shows 1-month reversal: last month's winners mean-revert. Using t-12 to t-2 (skip t-1) improves Sharpe by 0.2-0.3. This is the standard approach in academic research and industry practice.`,
    },
    {
        id: 'ats-6-mc-2',
        question: 'Cross-sectional momentum typically has Sharpe ratio of:',
        options: ['0.3-0.5 (poor)', '0.8-1.2 (good)', '2.0-3.0 (excellent)', '4.0+ (exceptional)'],
        correctAnswer: 1,
        explanation: `**Correct: 0.8-1.2 (good).**

Historical performance (1927-2024): Sharpe 0.8-1.2, annual return 10-15%, max drawdown -50% (2009). Momentum is robust but has crash risk.`,
    },
    {
        id: 'ats-6-mc-3',
        question: 'Momentum crashes occur when:',
        options: ['Market goes up', 'Winners reverse to losers overnight', 'Transaction costs increase', 'Correlation decreases'],
        correctAnswer: 1,
        explanation: `**Correct: Winners reverse to losers overnight.**

2009 Q1: Biggest losers (financials -80% in 2008) became biggest winners (+100% in Q1 2009). Momentum portfolios (short financials) crashed -50%.`,
    },
    {
        id: 'ats-6-mc-4',
        question: 'Time-series momentum differs from cross-sectional by:',
        options: ['Uses shorter lookback period', 'Trades each asset independently (not relative)', 'Only works in equities', 'Has lower Sharpe ratio'],
        correctAnswer: 1,
        explanation: `**Correct: Trades each asset independently (not relative).**

Cross-sectional: rank assets, long top vs short bottom (relative). Time-series: long if asset up, short if down (absolute). Time-series has market beta, cross-sectional is market-neutral.`,
    },
    {
        id: 'ats-6-mc-5',
        question: 'Volatility scaling for momentum means:',
        options: ['Trade only high-volatility stocks', 'Reduce positions when volatility increases', 'Increase positions when volatility increases', 'Avoid volatility completely'],
        correctAnswer: 1,
        explanation: `**Correct: Reduce positions when volatility increases.**

Target constant volatility (e.g., 15% annual). When realized vol > target, scale down positions. Protects against crashes (high vol periods). Improves risk-adjusted returns.`,
    },
];

