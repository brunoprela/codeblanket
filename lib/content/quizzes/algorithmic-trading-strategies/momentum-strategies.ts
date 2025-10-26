export const momentumStrategiesQuiz = [
    {
        id: 'ats-6-1-q-1',
        question: "Design cross-sectional momentum strategy for S&P 500: (1) Ranking period, (2) Holding period, (3) Long/short percentiles. Why skip last month in ranking? Calculate expected Sharpe ratio and explain momentum crashes (2009).",
        sampleAnswer: `**Complete Cross-Sectional Momentum Strategy**`,
        keyPoints: ['Ranking: 12 months (t-12 to t-2), skip last month to avoid short-term reversal; holding: 1 month rebalance', 'Long top 10%, short bottom 10%; equal-weighted within deciles; expect Sharpe 0.8-1.2, 12-15% annual return', 'Skip last month: short-term reversal effect (winners mean-revert over days); skipping improves Sharpe by 0.2-0.3', 'Momentum crashes (2009 Q1): -50% when winners become losers overnight; all correlations spike to 1.0; risk management: volatility scaling, crash detection', 'Implementation: monthly rebalance costs 20-30bps, use limit orders, manage turnover (~200%/year)'],
    },
    {
        id: 'ats-6-1-q-2',
        question: "Compare cross-sectional vs time-series momentum. Which works better in trending markets vs range-bound markets? Design a hybrid strategy combining both.",
        sampleAnswer: `**Cross-Sectional vs Time-Series Momentum**`,
        keyPoints: ['Cross-sectional: relative (long winners vs losers), market-neutral, works in any market; Time-series: absolute (long/short each asset), directional exposure', 'Trending markets: time-series wins (captures absolute moves); Range-bound: cross-sectional wins (captures relative moves)', 'Hybrid strategy: 50% cross-sectional + 50% time-series; diversification improves Sharpe from 0.9 to 1.3', 'Performance: cross-sectional Sharpe 1.0, time-series 0.8, combined 1.3 (diversification benefit)', 'Risk: time-series has market beta (+0.3), cross-sectional is market-neutral (0.0 beta); combined beta +0.15'],
    },
    {
        id: 'ats-6-1-q-3',
        question: "Momentum crashes: Explain 2009 Q1 (-50% drawdown). Design crash protection system: (1) Detection indicators, (2) Position sizing rules, (3) Exit triggers. How would this have helped in 2009?",
        sampleAnswer: `**Momentum Crash Protection**`,
        keyPoints: ['2009 Q1: winners (financials down -80%) became biggest gainers (+100%); losers reversed; momentum portfolios -50% in 3 months', 'Detection: (1) VIX >30, (2) market down >10% in month, (3) correlation >0.7 (everything moving together), (4) momentum portfolio down >15%', 'Position sizing: reduce exposure 50% when crash risk detected; increase when VIX <20; volatility targeting (15% annual)', 'Exit triggers: trailing stop 20%, absolute stop 30%, exit all positions if VIX spikes >50', '2009 protection: would have reduced exposure in Dec 2008, exited in Feb 2009, avoided 30% of losses'],
    },
];

