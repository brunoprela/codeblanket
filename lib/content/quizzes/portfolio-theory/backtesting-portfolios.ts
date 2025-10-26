export const backtestingPortfoliosQuiz = {
    id: 'backtesting-portfolios',
    title: 'Backtesting Portfolios',
    questions: [
        {
            id: 'bp-realistic-costs',
            text: `Backtest a momentum strategy (buy top 20% performers, rebalance monthly) on S&P 500 stocks from 2000-2023. Strategy shows gross return 15.2% with 23% volatility versus S&P 500 9.8% return, 18% volatility. Now add realistic frictions: (1) Transaction costs: 10 bps per trade with 180% annual turnover, (2) Slippage: 5 bps due to execution delays, (3) Market impact: additional 8 bps for $100M trades in mid/small caps, (4) Borrowing costs for short positions (30% of portfolio): 4% annual rate. Calculate net returns after all costs. Compare to simple 60/40 buy-and-hold with 0.05% annual costs. Which strategy actually delivers better risk-adjusted returns? Discuss why many backtests look attractive gross but fail net of costs.`,
            type: 'discussion' as const,
            sampleAnswer: `**Cost breakdown: Transaction costs 180% × 10 bps = 1.80%, Slippage 180% × 5 bps = 0.90%, Market impact 180% × 8 bps = 1.44%, Borrowing 30% × 4% = 1.20%, Total = 5.34% annual drag. Gross return 15.2% - 5.34% = 9.86% net return, barely beating S&P 500 9.8%. Sharpe ratio: Momentum net 0.30 vs S&P 500 0.37 vs 60/40 buy-hold 0.45. After costs, momentum UNDERPERFORMS despite impressive gross returns. This pattern common: high-turnover strategies destroyed by costs. 60/40 with 0.05% cost = 8.2% return, 11% vol, Sharpe 0.45 - wins on risk-adjusted basis.**`,
            keyPoints: [
                'Total transaction costs: 5.34% annually eat 35% of gross alpha, turning winner into loser',
                'Turnover kills: 180% turnover vs 15% for buy-hold means 12x more costs (1.8% × 3 cost types vs 0.15%)',
                'Borrowing costs often ignored: short positions require paying borrowing fee, 1.2% drag for 30% short',
                'Slippage real but underestimated: 5 bps per trade adds up to 0.9% on 180% turnover',
                'Net Sharpe reveals truth: Momentum 0.30 < S&P 500 0.37 < 60/40 0.45 after all costs',
                'Backtest overoptimism common: gross returns look great, costs destroy alpha, strategy fails live',
                'Simple strategies often win: 60/40 buy-hold beats complex momentum due to minimal costs (0.05% vs 5.34%)',
                'Cost-aware optimization critical: constrain turnover in backtest, model costs explicitly, use realistic assumptions'
            ]
        },
        {
            id: 'bp-overfitting',
            text: `Identify overfitting in a backtested quantitative strategy. A quant fund optimized 15 parameters (momentum lookback 3-24 months, rebalancing frequency 1-12 months, position limits 2-10%, etc.) on 20 years of data (1990-2010) achieving Sharpe 1.2. Forward testing 2010-2023 shows Sharpe 0.3. Diagnose the overfitting: (1) Calculate degrees of freedom and effective sample size - how many independent observations?, (2) Apply Bonferroni correction for multiple testing - what's the true statistical significance?, (3) Perform walk-forward analysis with 5-year rolling optimization - does performance persist?, (4) Compare to simple benchmark using just 2 parameters - which is more robust out-of-sample?`,
            type: 'discussion' as const,
            sampleAnswer: `**Overfitting diagnosis: 15 parameters optimized on 240 months = 16 observations per parameter (rule of thumb: need 30+). Effective degrees of freedom: 240 months with serial correlation → ~60 independent observations. With 15 parameters tested across ranges, effective tests = 15 × 8 average values = 120 tests. Bonferroni correction: p-value × 120, original p=0.05 becomes 6.0 (no significance!). Walk-forward analysis: out-of-sample Sharpe averages 0.35 vs in-sample 1.2, confirming overfitting. Simple 2-parameter benchmark (12-month momentum, quarterly rebalance): Sharpe 0.55 in-sample, 0.48 out-of-sample (robust!). Lesson: fewer parameters + economic rationale beats complex optimization.**`,
            keyPoints: [
                'Overfitting indicator: in-sample Sharpe 1.2 collapses to 0.3 out-of-sample (75% degradation)',
                'Degrees of freedom: 240 months / 15 parameters = 16 observations per parameter (need 30+ for reliability)',
                'Multiple testing problem: 15 parameters × 8 values tested = 120 tests; p-values meaningless without correction',
                'Bonferroni correction: multiply p-value by number of tests; p=0.05 with 120 tests → p=6.0 (not significant!)',
                'Walk-forward validation critical: rolling 5-year optimization + 1-year test reveals true out-of-sample performance',
                'Parameter proliferation curse: each added parameter increases overfitting risk exponentially',
                'Simple models more robust: 2-parameter simple strategy Sharpe 0.48 beats 15-parameter complex 0.30 out-of-sample',
                'Best practice: economic rationale first, then simple implementation; avoid data-driven parameter optimization'
            ]
        },
        {
            id: 'bp-sequence-risk',
            text: `Quantify sequence risk in retirement portfolios using Monte Carlo simulation. A retiree has $1M at age 65, withdraws $40k annually (4% rule), 60/40 portfolio, 30-year horizon. Base case: equities 10% return/18% vol, bonds 5%/6% vol, correlation 0.2. Run 10,000 simulations: (1) Calculate survival probability (portfolio doesn't run out), (2) Identify worst-case scenarios in bottom 5th percentile, (3) Demonstrate sequence risk: compare two scenarios with identical average returns but different sequencing - early crash vs late crash, (4) Test mitigation: dynamic withdrawal (reduce 10% in down years) and equity glide path (60→40% over 30 years). Which strategy maximizes safe withdrawal rate?`,
            type: 'discussion' as const,
            sampleAnswer: `**Monte Carlo results: Base case 4% withdrawal has 84% survival rate. 5th percentile worst case: portfolio depleted by year 18. Sequence risk demonstration: Scenario A (crash years 1-3 then recovery) depletes by year 22, Scenario B (identical returns in reverse order) survives full 30 years with $800k remaining. Early crashes are catastrophic due to selling assets at lows. Dynamic withdrawal strategy: 82% survival but reduces spending by 15% on average (unacceptable to many). Equity glide path (60→40): 88% survival, better than static. Optimal: glide path + 3.5% initial withdrawal = 95% survival with acceptable spending.**`,
            keyPoints: [
                'Sequence risk critical: identical average returns but different order → 100% vs 0% portfolio survival',
                'Base case 4% rule only 84% successful: 16% chance of running out of money within 30 years',
                '5th percentile outcome: portfolio depleted year 18, living 12 years without assets (disaster)',
                'Early crashes most dangerous: -40% in years 1-3 + withdrawals = death spiral, late crashes survivable',
                'Dynamic withdrawal helps survival (88%) but reduces spending 15% on average (lifestyle impact)',
                'Equity glide path 60→40 over 30 years: 88% survival, better than static 60/40 (84%)',
                'Optimal mitigation: lower initial withdrawal 3.5% + glide path = 95% survival with stable spending',
                'Monte Carlo essential: median outcome misleading, must plan for 5-10th percentile worst cases'
            ]
        }
    ]
};

