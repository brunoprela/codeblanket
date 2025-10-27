export const modernPortfolioTheoryQuiz = {
  id: 'modern-portfolio-theory',
  title: 'Modern Portfolio Theory',
  questions: [
    {
      id: 'mpt-diversification',
      text: `Explain the mathematical foundation of diversification benefits in Modern Portfolio Theory. Why does combining two assets with perfect positive correlation (+1.0) provide no diversification benefit, while combining assets with low or negative correlation provides substantial risk reduction? Include the portfolio variance formula and demonstrate with a numerical example comparing three scenarios: perfect positive correlation (+1.0), zero correlation (0.0), and perfect negative correlation (-1.0).`,
      type: 'discussion' as const,
      sampleAnswer: `**Mathematical Foundation of Diversification**

Portfolio variance depends not only on individual asset variances but critically on their covariances:

**Portfolio Variance Formula (2-asset case):**

σ²ₚ = w₁²σ₁² + w₂²σ₂² + 2w₁w₂σ₁σ₂ρ₁₂

Where:
- w₁, w₂ = weights (sum to 1)
- σ₁, σ₂ = individual standard deviations
- ρ₁₂ = correlation coefficient

The key insight is the correlation term: 2w₁w₂σ₁σ₂ρ₁₂

**Numerical Example:**

Two assets, equal weighted (50/50):
- Asset A: σ = 20%
- Asset B: σ = 20%

**Scenario 1: Perfect Positive Correlation (ρ = +1.0)**

σ²ₚ = 0.5²(0.20²) + 0.5²(0.20²) + 2(0.5)(0.5)(0.20)(0.20)(1.0)
    = 0.01 + 0.01 + 0.02 = 0.04
σₚ = 20%

**No diversification benefit!** Portfolio volatility equals weighted average of individual volatilities.

**Scenario 2: Zero Correlation (ρ = 0.0)**

σ²ₚ = 0.5²(0.20²) + 0.5²(0.20²) + 2(0.5)(0.5)(0.20)(0.20)(0.0)
    = 0.01 + 0.01 + 0 = 0.02
σₚ = 14.14%

**Substantial benefit!** Portfolio volatility reduced by 29% compared to individual assets.

**Scenario 3: Perfect Negative Correlation (ρ = -1.0)**

σ²ₚ = 0.5²(0.20²) + 0.5²(0.20²) + 2(0.5)(0.5)(0.20)(0.20)(-1.0)
    = 0.01 + 0.01 - 0.02 = 0
σₚ = 0%

**Perfect hedging!** Portfolio has zero volatility. This is the theoretical maximum diversification benefit.

**Why Correlation Matters:**

1. **Perfect Positive (+1.0):** Assets move in lockstep. When one zigs, the other zigs. The covariance term 2w₁w₂σ₁σ₂ρ adds fully to portfolio variance.

2. **Zero (0.0):** Assets move independently. The covariance term disappears (multiplied by 0), leaving only individual variances. This reduces total portfolio variance.

3. **Negative (<0):** Assets move in opposite directions. When one zigs, the other zags. The covariance term subtracts from portfolio variance, providing maximum diversification.

**Practical Implications:**

- Diversification is "free lunch" - reduces risk without necessarily reducing return
- Seek low-correlation assets: stocks + bonds (ρ ≈ 0.2), stocks + gold (ρ ≈ -0.1)
- International diversification reduces correlation (though globalization has increased correlations)
- In crisis, correlations often spike toward +1, reducing diversification benefits when most needed

**Key Insight:** The power of diversification comes from the covariance term in the portfolio variance formula. Lower correlation → smaller (or negative) covariance → lower portfolio variance → better risk-adjusted returns.`,
      keyPoints: [
        'Portfolio variance formula includes individual variances AND covariances between assets',
        'Correlation coefficient (ρ) ranges from -1 to +1 and determines the sign and magnitude of covariance contribution',
        'Perfect positive correlation (+1) provides no diversification; portfolio risk is weighted average of individual risks',
        'Zero correlation (0) provides substantial diversification; portfolio risk is lower than weighted average',
        'Negative correlation (<0) provides maximum diversification; assets partially or fully hedge each other',
        'Real-world diversification benefits depend on finding asset pairs with low or negative correlations',
        'Crisis periods often see correlation convergence toward +1, reducing diversification benefits',
      ],
    },
    {
      id: 'mpt-efficient-set',
      text: `A portfolio manager is constructing an optimal portfolio using three assets with the following characteristics:

Asset A: E(R) = 10%, σ = 15%, weight = 40%
Asset B: E(R) = 12%, σ = 20%, weight = 35%
Asset C: E(R) = 8%, σ = 10%, weight = 25%

Correlation matrix:
       A     B     C
A    1.0   0.6   0.3
B    0.6   1.0   0.4
C    0.3   0.4   1.0

Calculate: (1) the expected portfolio return, (2) the portfolio variance and standard deviation, (3) whether this portfolio is on the efficient frontier if the minimum variance portfolio has σ = 9.5% with E(R) = 8.5%, and (4) how the manager could improve the portfolio if it's not efficient. Show all calculations and explain the economic intuition behind your recommendations.`,
      type: 'discussion' as const,
      sampleAnswer: `**Solution:**

**1. Expected Portfolio Return**

E(Rₚ) = Σwᵢ E(Rᵢ)
      = 0.40(10%) + 0.35(12%) + 0.25(8%)
      = 4.0% + 4.2% + 2.0%
      = 10.2%

**2. Portfolio Variance and Standard Deviation**

First, construct the covariance matrix from correlations:

Cov(i,j) = ρᵢⱼ × σᵢ × σⱼ

Covariance Matrix:
              A           B           C
A    0.0225 (15²)   0.018      0.0045
B    0.018          0.04 (20²)  0.008
C    0.0045         0.008      0.01 (10²)

Where:
- Cov(A,B) = 0.6 × 15% × 20% = 0.018
- Cov(A,C) = 0.3 × 15% × 10% = 0.0045
- Cov(B,C) = 0.4 × 20% × 10% = 0.008

Portfolio variance:
σ²ₚ = Σᵢ Σⱼ wᵢwⱼCov(i,j)

Breaking down by components:

**Individual variances:**
- w²ₐσ²ₐ = 0.40² × 0.0225 = 0.0036
- w²ᵦσ²ᵦ = 0.35² × 0.04 = 0.0049
- w²ᴄσ²ᴄ = 0.25² × 0.01 = 0.000625

**Covariances (doubled for i≠j):**
- 2wₐwᵦCov(A,B) = 2 × 0.40 × 0.35 × 0.018 = 0.00504
- 2wₐwᴄCov(A,C) = 2 × 0.40 × 0.25 × 0.0045 = 0.0009
- 2wᵦwᴄCov(B,C) = 2 × 0.35 × 0.25 × 0.008 = 0.0014

**Total variance:**
σ²ₚ = 0.0036 + 0.0049 + 0.000625 + 0.00504 + 0.0009 + 0.0014
    = 0.0165

**Portfolio standard deviation:**
σₚ = √0.0165 = 12.85%

**3. Is This Portfolio Efficient?**

To determine if the portfolio is on the efficient frontier, we need to check if there exists another portfolio with:
- Same risk (12.85%) but higher return, OR
- Same return (10.2%) but lower risk

**Comparison with Minimum Variance Portfolio:**
- Minimum Variance Portfolio: E(R) = 8.5%, σ = 9.5%
- Current Portfolio: E(R) = 10.2%, σ = 12.85%

**Analysis:**

The current portfolio has higher return (10.2% vs 8.5%) and higher risk (12.85% vs 9.5%). This is consistent with the risk-return tradeoff.

However, we need to check if the portfolio is **dominated** - i.e., if another feasible portfolio exists with better risk-adjusted returns.

**Calculating Sharpe Ratio (assuming Rf = 4%):**
- Current Portfolio: Sharpe = (10.2% - 4%) / 12.85% = 0.482
- Min Variance Portfolio: Sharpe = (8.5% - 4%) / 9.5% = 0.474

The current portfolio has a slightly higher Sharpe ratio, suggesting it's reasonably efficient.

**However, we should check dominance more rigorously:**

Looking at the weights: 40% A, 35% B, 25% C

Asset B has the highest return (12%) but also highest risk (20%). Asset C has lowest return (8%) and lowest risk (10%).

**Key test:** Can we increase expected return without increasing risk by shifting weight from C to B?

Let's test shifting 5% from C to B: wₐ=40%, wᵦ=40%, wᴄ=20%

New E(R) = 0.40(10%) + 0.40(12%) + 0.20(8%) = 10.4% (increased!)

New variance calculation would show if risk increased proportionally or less.

**Conclusion:** The portfolio is **likely NOT on the efficient frontier** because Asset C (lowest return) has substantial allocation (25%), suggesting we could achieve higher returns for similar risk by reallocating toward higher-return assets.

**4. Improvement Recommendations**

**Option 1: Mean-Variance Optimization**

Run quadratic programming to find the true efficient portfolio at σ = 12.85%:

Maximize: E(Rₚ) = Σwᵢ E(Rᵢ)
Subject to:
- σₚ = 12.85% (target current risk level)
- Σwᵢ = 1
- wᵢ ≥ 0

Expected result: Shift weight from low-return Asset C to high-return Asset B, keeping overall portfolio risk constant.

**Option 2: Target Higher Sharpe Ratio**

Maximize Sharpe ratio directly:

Maximize: [E(Rₚ) - Rf] / σₚ

This finds the tangency portfolio (highest Sharpe ratio on efficient frontier).

Expected weights shift toward Asset B (highest return) balanced against correlation structure to minimize risk.

**Option 3: Practical Reallocation**

Without optimization, a simple improvement:
- Reduce Asset C from 25% to 15% (lowest return asset)
- Increase Asset B from 35% to 45% (highest return asset)
- Keep Asset A at 40% (middle ground)

New weights: 40% A, 45% B, 15% C

New E(R) = 0.40(10%) + 0.45(12%) + 0.15(8%) = 10.6% (vs 10.2%)
New risk: ~13.2% (slightly higher, but better risk-adjusted return)

**Economic Intuition:**

1. **Underweighting low-return assets:** Asset C contributes 25% of portfolio but only 8% expected return. This drags down overall returns.

2. **Moderate correlation structure:** The correlations (0.3-0.6) suggest decent diversification benefits, but not enough to justify overweighting the lowest-return asset.

3. **Risk-return tradeoff:** By accepting slightly higher risk (~0.3-0.5% more volatility), we can achieve meaningfully higher returns (~0.4-0.6% more return).

4. **Efficient frontier position:** The current portfolio sits below the efficient frontier, meaning investors are not being adequately compensated for the risk they're taking.

**Recommendation:** Use mean-variance optimization to find the efficient portfolio at current risk level (12.85%) or target the maximum Sharpe ratio portfolio. Expect to increase Asset B allocation and decrease Asset C allocation significantly.`,
      keyPoints: [
        'Portfolio return is weighted average of individual returns: E(Rₚ) = Σwᵢ E(Rᵢ)',
        'Portfolio variance requires covariance matrix: σ²ₚ = Σᵢ Σⱼ wᵢwⱼCov(i,j)',
        'Efficient frontier contains all portfolios with maximum return for given risk level',
        'Dominated portfolios exist below efficient frontier with inferior risk-return tradeoffs',
        'Sharpe ratio helps compare risk-adjusted returns across portfolios',
        'Mean-variance optimization finds efficient portfolios by maximizing return for given risk constraint',
        'Practical portfolio improvement involves shifting weights toward higher-return, appropriately-correlated assets',
        'Low-return assets should only receive substantial allocation if they provide significant diversification benefits through low correlation',
      ],
    },
    {
      id: 'mpt-limitations',
      text: `Modern Portfolio Theory revolutionized investing but has several well-documented limitations. Discuss at least five major limitations of MPT in real-world portfolio management, explaining: (1) the theoretical assumption MPT makes, (2) why this assumption fails in practice, (3) the practical consequences for investors, and (4) modern approaches that address each limitation. Include specific examples of how these limitations manifested during the 2008 financial crisis and the 2020 COVID-19 crash.`,
      type: 'discussion' as const,
      sampleAnswer: `**Major Limitations of Modern Portfolio Theory**

**1. Normal Distribution Assumption (Fat Tails)**

**MPT Assumption:** Returns follow a normal distribution, characterized solely by mean and variance. Extreme events are rare (3-sigma events should occur 0.3% of the time).

**Why It Fails:** Real asset returns exhibit:
- **Fat tails:** Extreme events occur far more frequently than normal distribution predicts
- **Negative skewness:** Large negative returns more common than large positive returns
- **Kurtosis:** Distribution is more peaked with fatter tails (leptokurtic)

**Practical Consequences:**

*2008 Financial Crisis Example:*
- S&P 500 fell 38.5% in 2008
- This represents an ~8-sigma event under normal distribution assumptions
- Should occur once in 6.6 billion years
- Reality: Major crashes happen every 10-20 years

*2020 COVID Crash:*
- S&P 500 fell 34% in 23 days (Feb-March 2020)
- VIX spiked to 82 (fear index)
- MPT models predicted far lower probabilities of such rapid declines

**Impact on Investors:**
- Underestimation of tail risk
- Insufficient portfolio protection
- Overreliance on diversification during crisis periods

**Modern Approaches:**
- **VaR and CVaR:** Focus on tail risk explicitly
- **Extreme Value Theory:** Model tail distributions separately
- **Risk parity strategies:** More robust to non-normal distributions
- **Options-based protection:** Tail hedging through puts
- **Robust optimization:** Optimize under worst-case scenarios

**2. Static Covariances (Correlation Breakdown)**

**MPT Assumption:** Correlations are stable and can be estimated from historical data.

**Why It Fails:** Correlations are time-varying and increase dramatically during crises ("correlation breakdown").

**Evidence:**

*2008 Crisis:*
- Pre-crisis: US stocks-international stocks correlation ~0.5-0.6
- During crisis: Correlation spiked to 0.85-0.95
- Diversification benefits evaporated when needed most

*2020 COVID:*
- March 2020: Almost all asset classes declined simultaneously
- Stocks, bonds, commodities, REITs all fell together
- Only gold and Treasuries provided safe haven

**Practical Consequences:**
- **Diversification illusion:** Portfolio appears diversified in calm markets but acts concentrated in crisis
- **Risk underestimation:** Models show lower risk than actual during stress periods
- **Rebalancing problems:** Optimal allocations change rapidly

**Modern Approaches:**
- **Dynamic correlation models:** GARCH, DCC-GARCH to model time-varying correlations
- **Regime-switching models:** Different correlations for bull/bear markets
- **Stress testing:** Test portfolio performance under high-correlation scenarios
- **True diversification:** Include truly uncorrelated assets (trend-following strategies, long volatility)
- **Dynamic risk budgeting:** Adjust allocations as correlations change

**3. Single-Period Framework (Myopic Optimization)**

**MPT Assumption:** Investors make one-time allocation decision for a single period.

**Why It Fails:** Real investors:
- Have multi-period horizons (saving for retirement over 30+ years)
- Face time-varying investment opportunities
- Must rebalance periodically
- Experience path-dependent risks (sequence of returns matters)

**Practical Consequences:**

*Example: Retirement Timing*
- Investor A retires 2007 (before crash): Portfolio falls 40%
- Investor B retires 2010 (after crash): Portfolio starts recovering
- Same long-run returns, drastically different outcomes

**Impact:**
- **Sequence risk:** Order of returns matters more than average return for multi-period investors
- **Timing risk:** When you invest/withdraw critically affects outcomes
- **Rebalancing costs:** Frequent trading erodes returns
- **Tax implications:** Capital gains from rebalancing

**Modern Approaches:**
- **Multi-period optimization:** Dynamic programming to find optimal intertemporal allocation
- **Lifecycle funds:** Adjust allocation based on age/time horizon
- **Monte Carlo simulation:** Test portfolio survival over thousands of paths
- **Dynamic asset allocation:** Adjust based on market conditions and valuations
- **Goals-based investing:** Optimize for specific future liabilities, not just maximum Sharpe ratio

**4. Known Returns and Covariances (Estimation Error)**

**MPT Assumption:** Expected returns, variances, and covariances are known with certainty.

**Why It Fails:** All inputs must be estimated from noisy historical data:

**Estimation Challenges:**
- Expected returns: Very noisy, low signal-to-noise ratio
- Standard errors on mean returns often larger than the means themselves
- Requires 25+ years of data for reliable estimates
- Past may not predict future (structural breaks)

**Practical Consequences:**

*Example: Optimization Sensitivity*
- Change expected return estimate for one asset by 1%
- Optimal portfolio weights can swing wildly (e.g., 30% → 5%)
- "Garbage in, garbage out" problem

*2008 Example:*
- Pre-crisis models estimated low volatility (VIX ~15)
- Used to optimize portfolios for "normal" market conditions
- Actual crisis volatility spiked to VIX ~80
- Portfolios were completely misallocated for realized risk environment

**Error Amplification:**
- MVO is extremely sensitive to input estimates
- Small errors in inputs → large errors in optimal weights
- Optimizer exploits estimation errors, treating noise as signal

**Modern Approaches:**
- **Bayesian methods:** Incorporate prior beliefs and uncertainty
- **Black-Litterman model:** Combine market equilibrium with investor views
- **Robust optimization:** Optimize under input uncertainty (worst-case optimization)
- **Resampled efficiency:** Average across multiple simulated inputs
- **Constrained optimization:** Limit position sizes to reduce impact of estimation error
- **Factor models:** Use systematic factors instead of individual asset estimates
- **Simple heuristics:** 1/N equal weight, inverse volatility weighting (often outperform MVO out-of-sample!)

**5. Ignores Transaction Costs and Taxes**

**MPT Assumption:** Trading is costless and frictionless; no taxes.

**Why It Fails:** Real-world frictions are substantial:

**Transaction Costs:**
- Commissions: $0 now for stocks, but spreads remain (1-10 bps)
- Market impact: Large trades move prices (especially illiquid assets)
- Slippage: 5-50 bps depending on liquidity
- Total: 10-100+ bps per round-trip trade

**Taxes:**
- Short-term capital gains: Up to 37% (ordinary income)
- Long-term capital gains: 0-20%
- Dividend taxes: 0-20% (qualified) or ordinary income rates
- State taxes: Additional 0-13%

**Practical Consequences:**

*High-Turnover Example:*
- Strategy suggests 100% annual turnover (rebalance entire portfolio)
- Transaction costs: 100% × 20 bps = 0.20% drag
- Taxes (assuming 50% gains, 20% rate): 100% × 50% × 20% = 10% drag
- Total: >10% per year lost to frictions!

*Real Example:*
- Managed futures funds often show 15% gross returns
- After fees (2/20), commissions, slippage: 8% net returns
- Frictions eat ~7% annually

**Impact on Optimal Portfolios:**
- High-frequency rebalancing becomes suboptimal
- Buy-and-hold may beat "optimal" dynamic strategies after costs
- Tax-loss harvesting becomes valuable
- Asset location matters (stocks in taxable, bonds in IRA)

**Modern Approaches:**
- **Transaction cost optimization:** Explicitly model costs in optimization
- **Turnover constraints:** Limit rebalancing frequency and amount
- **Tax-aware optimization:** Maximize after-tax returns
- **Asset location optimization:** Place assets in optimal account types
- **Tax-loss harvesting:** Systematically realize losses for tax benefits
- **Lower-frequency rebalancing:** Quarterly or annual instead of daily/weekly
- **Threshold-based rebalancing:** Only rebalance when drift exceeds X%

**6. BONUS: Leverage and Short-Selling Constraints Ignored**

**MPT Assumption:** Investors can borrow and lend at risk-free rate; can short any asset.

**Why It Fails:**
- Retail investors: Can't short easily; leverage limited to 2:1 margin
- Leverage costs: Borrow at 5-8%, not risk-free rate (1-4%)
- Short selling: Expensive (borrow costs), risky (unlimited loss)
- Regulatory constraints: Many funds prohibited from leverage/shorting

**Modern Approaches:**
- **Long-only optimization:** Add non-negativity constraints
- **Leverage cost modeling:** Use realistic borrowing rates
- **Alternatives:** Options, futures for leverage instead of margin
- **130/30 strategies:** Limited short exposure (30% short, 130% long)

**Summary Table: MPT Limitations and Solutions**

| Limitation | MPT Assumes | Reality | Consequence | Modern Solution |
|------------|-------------|---------|-------------|-----------------|
| Fat tails | Normal distribution | Leptokurtic, fat tails | Underestimate crash risk | CVaR, tail hedging, robust optimization |
| Static correlations | Stable correlations | Time-varying, crisis spikes | Diversification breakdown | DCC-GARCH, regime models, stress testing |
| Single period | One-shot decision | Multi-period horizon | Path dependence, sequence risk | Dynamic allocation, Monte Carlo |
| Known inputs | Perfect knowledge | Estimation error | Optimizer exploits noise | Black-Litterman, robust optimization, constraints |
| No frictions | Costless trading | Costs + taxes significant | High turnover destroys returns | Transaction cost models, turnover limits, tax optimization |

**Conclusion:**

MPT remains foundational but must be augmented with:
1. **Realistic distributional assumptions** (fat tails, skewness)
2. **Dynamic models** (time-varying parameters, regime switching)
3. **Multi-period frameworks** (intertemporal optimization)
4. **Robust estimation** (Black-Litterman, Bayesian methods)
5. **Friction models** (transaction costs, taxes)
6. **Stress testing** (crisis scenarios)

Modern portfolio management uses MPT as starting point but incorporates these extensions to address real-world complexities. The 2008 and 2020 crises highlighted these limitations dramatically, pushing the industry toward more robust, realistic approaches.`,
      keyPoints: [
        'Fat tail events occur far more frequently than normal distribution assumes; crashes are 5-10 sigma events that happen regularly',
        'Correlations spike during crises, causing diversification breakdown when protection is most needed',
        'Single-period optimization ignores sequence risk, path dependence, and rebalancing costs in multi-period investing',
        'Estimation error in returns/covariances is massive; small input errors lead to drastically suboptimal portfolios',
        'Transaction costs and taxes can consume 0.5-10%+ annually, making theoretically optimal strategies inferior after frictions',
        'Real-world constraints (no shorting, leverage limits) change the shape of efficient frontier dramatically',
        'Modern approaches include CVaR, dynamic correlation models, Black-Litterman, robust optimization, and explicit friction modeling',
        '2008 and 2020 crises demonstrated all these limitations simultaneously: fat tails, correlation spikes, and liquidity crises',
      ],
    },
  ],
};
