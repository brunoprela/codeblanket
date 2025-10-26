export const portfolioConstraintsQuiz = {
    id: 'portfolio-constraints',
    title: 'Portfolio Construction Constraints',
    questions: [
        {
            id: 'pc-position-sector',
            text: `Optimize a 50-stock portfolio with multiple constraint layers: (1) Position limits: min 0.5%, max 5% per stock, (2) Sector limits: Technology ≤30%, Financials ≤25%, match other sectors to S&P 500 ±5%, (3) Factor constraints: Market beta 0.9-1.1, no factor loading >0.5, (4) Turnover limit: ≤50% from current portfolio. Current portfolio has Tech 35%, Financials 20%, beta 1.15. Expected returns favor Tech heavily (12% vs 8% market). Using CVXPY, solve the optimization and analyze: how much does each constraint reduce expected portfolio return versus unconstrained optimum? Which constraint binds most tightly? What's the shadow price (Lagrange multiplier) of the Tech sector limit?`,
            type: 'discussion' as const,
            sampleAnswer: `**Unconstrained optimum: 18% expected return, 95% in top 5 Tech stocks, 22% volatility, impractical. With constraints: Position limits reduce return to 14.2% (-3.8%), sector limits to 13.1% (-1.1%), factor constraints to 12.8% (-0.3%), turnover limit to 12.3% (-0.5%). Tech sector constraint binds most tightly (reduces from 95% to 30%). Shadow price of Tech limit: 0.12 (relaxing Tech limit by 1% would increase return by 0.12%). Optimal constrained portfolio: 30% Tech (at limit), 25% Financials (at limit), beta 1.0, 48% turnover (just under 50% limit), final return 12.3%, volatility 17%, Sharpe 0.54.**`,
            keyPoints: [
                'Position limits most restrictive: reduce concentrated 95% positions to diversified 5% max, lose 3.8% return',
                'Sector limits prevent overconcentration: cap Tech at 30% despite high expected returns (12% vs 8% market)',
                'Factor constraints ensure style consistency: prevent portfolio drift from beta 1.15 to 1.0 target range',
                'Turnover limit reduces transaction costs: 48% turnover vs unconstrained would be 85%',
                'Binding constraints have positive shadow prices: Tech limit λ=0.12 means 1% relaxation → 0.12% return gain',
                'Hierarchical constraint value: regulatory > risk management > client preferences > operational',
                'Optimization trades off expected return for risk control: lose 5.7% return to achieve prudent diversification',
                'Real institutions typical return sacrifice: 2-4% expected return to achieve compliant, implementable portfolios'
            ]
        },
        {
            id: 'pc-esg-constraints',
            text: `Incorporate ESG constraints into portfolio optimization. Investment mandate requires: (1) Portfolio ESG score ≥8/10 (weighted average of stock scores), (2) Carbon intensity ≤50 tons CO2/$M revenue (vs benchmark 75), (3) Exclude tobacco, weapons, fossil fuel extraction, (4) Overweight ESG leaders (top quartile) by 20% vs benchmark weight. Starting universe: S&P 500, benchmark: market-cap weighted. Given these constraints: how many stocks are excluded by screens? What's the resulting tracking error vs S&P 500? Calculate the "ESG cost" - difference in expected Sharpe ratio between ESG-constrained and unconstrained portfolios. Is the 0.05-0.15 Sharpe reduction justified by ESG benefits?`,
            type: 'discussion' as const,
            sampleAnswer: `**Exclusionary screens eliminate 78 of 500 stocks (15.6%): tobacco 5, weapons 12, fossil fuels 61. Remaining universe: 422 stocks, representing 84% of S&P 500 market cap. ESG ≥8 constraint reduces universe to 127 stocks (top 25%). Optimal ESG portfolio: tracking error 4.2% vs S&P 500, expected return 9.8% vs 10.4% benchmark, Sharpe 0.48 vs 0.53 unconstrained. ESG cost: -0.05 Sharpe, -0.6% return annually. Benefits: 35% lower carbon intensity, 8.7 weighted ESG score vs 6.2 benchmark, alignment with values. Justified for investors who value ESG outcomes beyond financial returns, or believe ESG factors will drive long-term outperformance (debated).**`,
            keyPoints: [
                'Exclusionary screens remove 15-20% of universe: tobacco, weapons, fossil fuels eliminate major market cap',
                'ESG score constraint creates concentration: limits to top 25% of stocks, increases tracking error to 4-5%',
                'ESG cost quantified: -0.6% return, -0.05 Sharpe annually versus unconstrained benchmark',
                'Carbon reduction achieved: 35% lower intensity through exclusions and overweighting clean tech/services',
                'Tracking error tradeoff: tighter ESG constraints → higher TE → greater active risk vs benchmark',
                'ESG benefit depends on beliefs: (1) intrinsic value alignment, or (2) ESG = future alpha (controversial)',
                'Empirical evidence mixed: some studies show no ESG cost, others show 0.5-2% drag, depends on period',
                'Implementation challenge: ESG data quality variable, scores differ across providers, greenwashing risks'
            ]
        },
        {
            id: 'pc-transaction-costs',
            text: `Model realistic transaction costs in portfolio optimization. For a $500M portfolio rebalancing across 100 stocks: (1) Estimate costs with breakdown: commissions ($0 post-2019), bid-ask spreads (5-15 bps based on liquidity), market impact (Almgren-Chriss model: proportional to sqrt(volume)), opportunity cost. (2) Solve for optimal rebalancing: how much deviation from target justifies trading cost? Use quadratic program penalizing both tracking error and transaction costs. (3) Compare discrete rebalancing rules: don't trade if position within ±3% of target vs continuous optimization. (4) Calculate annual performance drag from different strategies: monthly rebalancing, quarterly, annual, and 5% threshold-based.`,
            type: 'discussion' as const,
            sampleAnswer: `**Transaction cost model: Liquid stocks 8 bps (5 spread + 3 impact), Mid-cap 15 bps, Small-cap 30 bps. Portfolio-weighted avg: 12 bps per trade. Optimal rebalancing: trade if |current - target| > 2.8% (where marginal benefit = marginal cost). Discrete ±3% bands vs continuous optimization: discrete is 95% as effective but simpler. Annual drag by frequency: Monthly 0.40% (trade 85% of portfolio annually), Quarterly 0.18% (40% traded), Annual 0.08% (15% traded), 5% threshold 0.10% (22% traded). Optimal: 5% threshold bands with annual review, captures 92% of benefit at 25% of cost.**`,
            keyPoints: [
                'Transaction costs for institutional portfolios: 8-15 bps average per trade including spread, impact, timing costs',
                'Optimal rebalancing threshold: |deviation| > 2-3% justifies trading cost; within band, don't trade',
        'Quadratic cost model: penalize sqrt(trade size / daily volume) captures market impact non-linearity',
                'Monthly rebalancing costs 0.40% annually through excessive turnover; quarterly much lower at 0.18%',
                '5% threshold approach optimal: adapts to market conditions, trades only when deviation meaningful',
                'Discrete bands (±3%) nearly as effective as continuous optimization but far simpler to implement',
                'Liquidity-adjusted costs: small-cap 30 bps vs large-cap 8 bps; adjust rebalancing bands accordingly',
                'Annual drag calculation critical: 0.20-0.40% cost can eliminate active management alpha entirely'
            ]
        }
    ]
};

