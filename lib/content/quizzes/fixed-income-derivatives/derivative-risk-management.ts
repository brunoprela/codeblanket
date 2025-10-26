export const derivativeRiskManagementQuiz = [
    {
        id: 'drm-q-1',
        question: 'Build a comprehensive VaR calculation engine that: (1) Implements all three methodologies (historical simulation, parametric, Monte Carlo), (2) Handles multi-asset portfolios (bonds, options, futures, swaps), (3) Calculates incremental VaR (marginal contribution of each position), (4) Performs backtesting (compare VaR predictions to actual P&L), (5) Generates breach reports when limits exceeded. Include: confidence levels (95%, 99%), time horizons (1-day, 10-day), component VaR. How do you handle: non-normal distributions (fat tails)? Non-linear instruments (options)? Correlation breakdowns in stress?',
        sampleAnswer: 'VaR engine: Historical simulation: Apply 250 days of returns to current positions, 5th percentile = 95% VaR, Parametric: VaR = Portfolio × σ × 1.65, use covariance matrix, Monte Carlo: Simulate 10K paths, reprice all instruments, Incremental VaR: Calculate VaR with/without position, difference = contribution, Backtesting: Count exceedances (actual loss > VaR), target <5% for 95% VaR, Component VaR: ∂VaR/∂position_i × position_i = contribution of position i.',
        keyPoints: ['Three VaR methods', 'Incremental VaR', 'Backtesting', 'Component VaR', 'Non-linear handling'],
    },
    {
        id: 'drm-q-2',
        question: 'Design a stress testing framework that: (1) Defines historical and hypothetical scenarios, (2) Reprices entire portfolio under each scenario, (3) Identifies largest losses and key risk factors, (4) Performs reverse stress testing (find breaking point), (5) Reports results to risk committee. Include: scenarios (2008, 2020, hypothetical +200bp rates), sensitivity to assumptions. How do you handle: liquidity stress (can\'t exit positions)? Correlation changes (→1 in crisis)? Multiple simultaneous shocks?',
        sampleAnswer: 'Stress framework: Scenarios: Historical (1987, 2008, 2020), Hypothetical (rates +200bp, spreads +150bp, equity -30%), Reverse (start with -$100M loss, solve for scenario), Repricing: Full revaluation of all positions (bonds, options, swaps), Risk factors: Identify top contributors (duration, spread, equity), Liquidity: Haircut prices (bid side only, widen spreads), Reporting: Summary table (scenario, loss, VaR exceedance, breaches).',
        keyPoints: ['Historical + hypothetical scenarios', 'Full repricing', 'Reverse stress testing', 'Liquidity stress', 'Risk factor identification'],
    },
    {
        id: 'drm-q-3',
        question: 'Implement a real-time risk limit monitoring system that: (1) Tracks Greeks (DV01, delta, gamma, vega) across all positions, (2) Enforces hard limits (reject trades if exceeded), (3) Generates alerts at threshold levels (80% of limit), (4) Aggregates exposure by desk, region, asset class, (5) Produces daily risk reports. Include: pre-trade checks, post-trade reconciliation. How do you handle: limit breaches (force unwind vs exception approval)? Intraday monitoring vs end-of-day? Limit utilization optimization?',
        sampleAnswer: 'Limit monitoring: Real-time Greeks: Calculate on every trade, aggregate by hierarchy (trader → desk → region), Hard limits: Pre-trade check, reject if limit exceeded (API returns 400), Alerts: Email/SMS at 80%, 90%, 100% utilization, Exposure aggregation: Sum DV01 across all bonds/swaps, sum delta across all options/equity, Reporting: Daily summary (current, limit, utilization %), breaches flagged, Breach handling: Automatic rejection (hard limit), or escalate to CRO for approval (soft limit).',
        keyPoints: ['Real-time Greeks tracking', 'Pre-trade checks', 'Alert thresholds', 'Aggregation by hierarchy', 'Breach handling'],
    },
];

