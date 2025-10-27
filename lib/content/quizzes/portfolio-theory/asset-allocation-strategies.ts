export const assetAllocationStrategiesQuiz = {
    id: 'asset-allocation-strategies',
    title: 'Asset Allocation Strategies',
    questions: [
        {
            id: 'aas-strategic-tactical',
            text: `Compare strategic asset allocation (SAA) versus tactical asset allocation (TAA) for a $500M pension fund with 25-year horizon. The fund's policy portfolio is 60% Global Equities, 25% Fixed Income, 10% Real Estate, 5% Commodities. Discuss: (1) how to set SAA targets based on the fund's liability structure, risk tolerance, and long-term capital market assumptions, (2) when and how TAA should deviate from SAA (triggers, limits, expected alpha), (3) calculate the information ratio needed to justify a 5% tracking error from tactical tilts, and (4) analyze a real scenario: If your model predicts equities will underperform by 3% over next 12 months with 65% confidence, what tactical adjustment should you make?`,
            type: 'discussion' as const,
            sampleAnswer: `**1. Setting Strategic Asset Allocation**

**Step 1: Analyze Liability Structure**

**Pension Fund Liabilities:**
- Current obligations: $400M
- Duration: 12 years
- Growth rate: 3% annually (salary-linked)
- Discount rate: 6%
- Present value: $450M (underfunded by $50M!)

**Liability Characteristics:**
- Long-term (25+ year horizon for fully funding)
- Inflation-sensitive (tied to salaries)
- Relatively predictable cash flows
- Need 6-7% return to meet obligations

**Asset-Liability Matching Considerations:**
- Duration match: 12-year liability → need assets with ~12-year duration
- Return requirement: 6-7% after inflation
- Inflation hedge: Need real assets (equities, real estate, commodities)

**Step 2: Risk Tolerance Assessment**

**Quantitative Risk Tolerance:**

Pension funding ratio = Assets / Liabilities = $500M / $450M = 111%

**Risk capacity:**
- Can tolerate drawdowns if funding ratio stays > 80% (regulatory minimum often 80-90%)
- 111% → 80% allows 28% drawdown in assets
- With 60% equities: Equities could fall 47% before hitting 80% funding ratio
- **High risk capacity**

**Qualitative Factors:**
- Sponsor financial health: Strong → higher risk tolerance
- Workforce demographics: Mature (many retirees) → lower risk, Growth (young workers) → higher risk
- Regulatory constraints: ERISA requires "prudent investor" standard
- Political considerations: Public pension → stakeholder scrutiny

**Board Risk Tolerance:** Moderate-to-high
- Can accept 15-20% portfolio volatility
- Target 80%+ probability of achieving 6% return

**Step 3: Long-Term Capital Market Assumptions (CMAs)**

**Asset Class Expectations (10-year forward-looking):**

| Asset Class | Expected Return | Volatility | Real Return |
|-------------|----------------|------------|-------------|
| Global Equities | 8.5% | 18% | 6.0% |
| Fixed Income | 4.5% | 6% | 2.0% |
| Real Estate | 7.0% | 15% | 4.5% |
| Commodities | 5.5% | 20% | 3.0% |

**Correlation Matrix:**
\`\`\`
             Eq    FI    RE   Comm
Equities    1.00  0.20  0.60  0.40
Fixed Inc   0.20  1.00  0.25  0.10
Real Estate 0.60  0.25  1.00  0.35
Commodities 0.40  0.10  0.35  1.00
                ```

**Step 4: Optimization with Constraints**

**Objective:** Maximize return subject to:
- Liability-matching constraint: Duration ≥ 12 years
- Risk constraint: Volatility ≤ 16%
- Return target: ≥ 6.5% real return
- Regulatory: Min 20% fixed income (liquidity)
- Policy: Max 10% in any alternative (commodities, real estate individually)

**Mean-Variance Optimization:**

Using CMAs and constraints:

**Optimal SAA:**
- **Global Equities: 58%** (close to 60% policy)
- **Fixed Income: 27%** (slightly above minimum for duration matching)
- **Real Estate: 10%** (at maximum for diversification)
- **Commodities: 5%** (inflation hedge)

**Portfolio Metrics:**
- Expected return: 7.1% nominal (4.6% real)
- Volatility: 14.8%
- Sharpe ratio: 0.31 (using 3% risk-free rate)
- 90% confidence interval: -15% to +29% annual return
- Probability of achieving 6% return: 65%

**Slight adjustment from policy portfolio:**

**Final SAA Policy:**
- **60% Global Equities** (round number, policy anchor)
- **25% Fixed Income** (liquidity, duration matching)
- **10% Real Estate** (inflation hedge, diversification)
- **5% Commodities** (inflation hedge, complete diversification)

**Rationale for each allocation:**

**60% Equities:**
- Need growth to fund liabilities (8.5% return)
- Long horizon allows riding out volatility
- Inflation hedge (real assets)
- Diversification across global markets

**25% Fixed Income:**
- Duration matching (target 12-year duration bonds)
- Liquidity for benefit payments
- Risk mitigation during equity downturns
- Regulatory comfort (prudent investor standard)

**10% Real Estate:**
- Inflation hedge
- Income generation (6-7% yield)
- Low correlation with equities (0.60) and bonds (0.25)
- Diversification benefit

**5% Commodities:**
- Inflation hedge (especially energy, metals)
- Negative/low correlation in inflationary environments
- Portfolio insurance during stagflation
- Small allocation due to high volatility and no intrinsic return

**Rebalancing Rules:**
- Quarterly rebalancing
- Tolerance bands: ±5% for equities, ±3% for others
- Rebalance if any asset class exceeds band

**2. Tactical Asset Allocation (TAA) Framework**

**TAA Definition:**

Short-term (3-12 month) deviations from SAA to capitalize on market mispricings or regime shifts.

**TAA Triggers:**

**Valuation Signals:**
- Equity CAPE > 30: Reduce equities by 5-10%
- Bond yields in top/bottom 10th percentile historically: Adjust bonds
- Real estate cap rates 2+ std dev from mean: Adjust allocation

**Momentum Signals:**
- 12-month momentum > 15%: Increase allocation by 3-5%
- 6-month momentum < -10%: Reduce allocation by 3-5%
- Must combine with valuation (avoid chasing bubbles)

**Macro Regime Signals:**
- Recession probability > 40%: Reduce equities 5-10%, increase bonds/defensive
- Inflation expectations > 4%: Increase commodities, TIPS; reduce nominal bonds
- Fed policy shifts: Adjust duration and equity beta

**Example TAA Triggers for Our Fund:**

**Scenario 1: Equity Overvaluation**
- Trigger: S&P 500 CAPE > 32 (current ~30 = normal)
- Action: Reduce equities from 60% to 52-55%
- Increase: Fixed income to 30%, Real estate to 12%

**Scenario 2: Recession Signal**
- Trigger: Yield curve inverted for 6+ months + Leading indicators declining
- Action: Reduce equities to 50-52%, reduce commodities to 3%
- Increase: Fixed income to 30-32%, keep real estate at 10%

**Scenario 3: Inflation Surge**
- Trigger: CPI expectations > 5% (from 2% normal)
- Action: Reduce nominal bonds from 25% to 18%
- Increase: Commodities to 8%, Real estate to 12%, TIPS/inflation-linked bonds to 7%

**TAA Limits:**

**Position Limits:**
- Maximum deviation from SAA: ±10 percentage points per asset class
- Example: Equities can range from 50% to 70%
- Total absolute deviation across all assets: ≤20%

**Risk Limits:**
- Tracking error vs SAA: ≤5% (absolute deviation from policy)
- Volatility: 13% to 17% (SAA ±2%)
- Drawdown: If portfolio falls >12% from peak, revert to SAA

**Time Limits:**
- TAA positions reviewed monthly
- Maximum hold period: 18 months (avoid style drift)
- Force reversal to SAA if signal not confirmed after 12 months

**Governance:**
- CIO can make TAA adjustments within ±5%
- Board approval required for ±5-10% moves
- Emergency committee for >10% moves (rare, crises only)

**Expected TAA Alpha:**

Historical evidence:
- Disciplined TAA: 0.5-1.5% annual alpha
- Average TAA: 0.2-0.5% alpha
- Poor TAA: -0.5% (costs exceed benefits)

**For 5% tracking error:** Need Information Ratio ≥ 0.4 to justify

**3. Information Ratio Required for 5% Tracking Error**

**Information Ratio (IR):**
\\[
IR = \\frac{\\text{Active Return}}{\\text{Tracking Error}} = \\frac{\\alpha}{TE}
\\]

**Question:** Is 5% tracking error worth it?

**Scenario Analysis:**

**Assume IR capabilities:**
- Excellent TAA skill: IR = 0.8
- Good TAA skill: IR = 0.5
- Average TAA skill: IR = 0.3
- Poor TAA skill: IR = 0.0

**Expected Alpha at 5% TE:**
- Excellent: 5% × 0.8 = 4.0% annual alpha
- Good: 5% × 0.5 = 2.5% alpha
- Average: 5% × 0.3 = 1.5% alpha
- Poor: 5% × 0.0 = 0.0% alpha

**Cost-Benefit Analysis:**

**Benefits of TAA:**
- Alpha generation: 0-4% depending on skill
- Risk management: Avoid major drawdowns
- Opportunistic: Capture market dislocations

**Costs of TAA:**
- Transaction costs: 0.1-0.3% annually
- Management time: Significant (opportunity cost)
- Risk of being wrong: Underperformance
- Complexity: Monitoring, governance

**Breakeven IR:**

\\[
IR_{breakeven} = \\frac{\\text{Transaction Costs}}{TE} = \\frac{0.2\\%}{5\\%} = 0.04
\\]

Very low bar! But realistic IR considering:
- Market timing is hard
- Many TAA strategies fail
- Behavioral biases (overconfidence, recency bias)

**Realistic Assessment:**

**Empirical IR for institutional TAA:**
- Top quartile: IR = 0.4-0.6
- Median: IR = 0.1-0.2
- Bottom quartile: IR = -0.2 to 0.0

**For 5% tracking error to be justified:**

Need **IR ≥ 0.3** to provide meaningful value:
- 5% × 0.3 = 1.5% alpha
- After 0.2% costs = 1.3% net alpha
- Adds 1.3% to 7.1% SAA return = 8.4% total

**Recommendation for Our Pension Fund:**

**Conservative TAA with 3% tracking error (not 5%):**
- More realistic: IR = 0.4 achievable
- Expected alpha: 3% × 0.4 = 1.2%
- Lower risk of major mistakes
- Easier governance and oversight

**Benefits:**
- 1.2% alpha × $500M = $6M annually
- Helps close $50M funding gap faster
- Risk-managed approach

**4. Tactical Scenario: Predicted 3% Equity Underperformance**

**Setup:**
- Model prediction: Equities will underperform by 3% over next 12 months
- Confidence: 65%
- Current allocation: 60% equities (SAA target)
- Question: What tactical adjustment?

**Step 1: Assess Signal Strength**

**65% confidence interpretation:**
- Not highly confident (75%+ is high conviction)
- Moderate signal
- 35% chance model is wrong

**Expected Value of Acting on Signal:**
- If right (65% probability): Avoid -3% × allocation = benefit
- If wrong (35% probability): Miss +3% = cost
- Need to account for transaction costs

**Step 2: Kelly Criterion Sizing**

For tactical bets, use Kelly-style position sizing:

\\[
f^* = \\frac{p \\times b - q}{b}
\\]

Where:
- p = probability of being right = 0.65
- q = probability of being wrong = 0.35
- b = payoff ratio (simplified to 1 for this analysis)

\\[
f^* = \\frac{0.65 \\times 1 - 0.35}{1} = 0.30
\\]

**Interpretation:** Adjust 30% of the maximum allowable position.

**Step 3: Calculate Tactical Adjustment**

**Maximum allowable deviation:** ±10% (from TAA limits)

**Suggested reduction:** 10% × 30% = **3% reduction in equities**

**New allocation:**
- Equities: 60% → 57%
- Where to put 3%?

**Reallocation decision:**

If predicting equity underperformance due to:

**Recession expectations:**
- Add 2% to bonds (flight to safety)
- Add 1% to cash

**Valuation concerns (no recession):**
- Add 1.5% to bonds
- Add 1% to real estate (valuation-insensitive)
- Add 0.5% to cash

**Rising rates:**
- Add 1% to floating-rate bonds
- Add 1% to real assets (real estate, commodities)
- Add 1% to cash

**Assume recession scenario:**

**Tactical Portfolio:**
- **Equities: 57%** (down 3% from 60%)
- **Fixed Income: 27%** (up 2% from 25%)
- **Real Estate: 10%** (unchanged - defensive)
- **Commodities: 5%** (unchanged)
- **Cash: 1%** (up 1% from 0%)

**Step 4: Calculate Expected Outcome**

**Scenario 1: Model is correct (65% probability)**

Equity underperformance avoided on 3% of portfolio:
- Benefit: 3% allocation × 3% avoided loss = 0.09% portfolio gain
- Opportunity cost on bonds: 3% × (assume 1% lower return) = -0.03%
- Net benefit: 0.06% portfolio return

**Scenario 2: Model is wrong (35% probability)**

Equity outperformance missed on 3% of portfolio:
- Cost: 3% allocation × 3% missed gain = -0.09%
- Bond gain: 3% × (assume 1% higher return) = 0.03%
- Net cost: -0.06%

**Expected Value:**
\\[
EV = 0.65 \\times 0.06\\% + 0.35 \\times (-0.06\\%) = 0.039\\% - 0.021\\% = 0.018\\%
\\]

**Expected gain: 0.018% or $90,000 on $500M**

**After transaction costs (0.01%):**
- Net gain: 0.008% or $40,000

**Worth it?** Marginal. The tactical adjustment is appropriately small given:
1. Moderate confidence (65%)
2. Moderate predicted impact (-3%)
3. Expected value is positive but small

**Step 5: Implementation**

**Execution:**1. **Week 1:** Reduce equities by 1.5% (sell $7.5M)
   - Buy $5M bonds
   - Buy $2.5M cash/money market

2. **Week 2-3:** Monitor signals
   - If confirmation: Reduce another 1.5% (total 3%)
   - If signals weaken: Hold at 1.5% reduction

3. **Month 2-12:** Review monthly
   - If recession materializes: Hold tactical position
   - If economy stabilizes: Gradually revert to SAA
   - Maximum hold: 12 months

**Stop-Loss:**
- If equities outperform by 5%: Revert to SAA (model clearly wrong)
- If portfolio tracking error > 4%: Reduce TAA positions

**Monitoring:**
- Track actual vs predicted equity performance
- Update confidence based on realized outcomes
- Adjust model if consistently wrong

**Alternative Approach: Options Hedging**

Instead of selling equities outright:

**Buy 6-month 5% OTM put options on $15M equity exposure** (3% of $500M)
- Cost: ~1.5% of notional = $225k (0.045% of portfolio)
- If equities fall 10%: Put pays $750k (0.15% of portfolio)
- If equities flat/up: Lose premium $225k

**Advantages:**
- No tracking error (still holding equities)
- Defined maximum cost ($225k)
- Asymmetric payoff (limited downside, unlimited upside)

**Disadvantages:**
- Premium cost
- Time decay if prediction wrong
- Requires derivatives approval

**Conclusion:**

For a **65% confidence, -3% prediction:**
- **Modest tactical adjustment: 3% reduction**
- **Expected value: Positive but small ($40-90k)**
- **Risk-managed: Within governance limits**
- **Alternative: Options for asymmetric exposure**

The key insight: TAA adjustments should be **proportional to conviction**. A 65% confidence doesn't justify extreme positioning. If confidence were 85% with -10% prediction, a larger 7-8% reduction would be warranted.

**Best Practice Summary:**

**Strategic Asset Allocation (SAA):**
- Set based on long-term objectives, risk tolerance, liability structure
- Review annually, update every 3-5 years
- 60/25/10/5 for our pension fund

**Tactical Asset Allocation (TAA):**
- Opportunistic 3-12 month deviations
- Maximum ±10% per asset class, ±5% tracking error
- Requires IR ≥ 0.3 to justify (0.4+ for 5% TE)
- Position sizing based on conviction
- **For 65% conviction, -3% prediction: 3% equity reduction**

The goal is adding value (0.5-1.5% annually) while maintaining discipline and avoiding catastrophic errors. TAA is a tool, not a strategy - SAA remains the foundation.`,
            keyPoints: [
                'SAA set using liability duration matching, risk capacity, and long-term capital market assumptions',
                'Pension fund example: 60/25/10/5 (Equity/FI/RE/Comm) targets 7% return, 15% vol, 12-year duration',
                'TAA limits typically ±10% per asset class, ≤5% tracking error; requires IR≥0.3 to justify deviations',
                'Information ratio breakeven for 5% TE: IR≥0.04 for costs, IR≥0.3 for meaningful alpha (1.5%)',
                'Tactical position sizing: Kelly formula suggests 30% of max deviation for 65% confidence signal',
                'For -3% equity prediction at 65% confidence: reduce equities by 3% (from 60% to 57%), reallocate to bonds/cash',
                'Expected value of tactical trade: ~0.02% ($90k on $500M) after costs; marginal but positive',
                'Alternative: use put options for asymmetric payoff; costs 0.045% premium for downside protection'
            ]
        },
        {
            id: 'aas-lifecycle',
            text: `Design a lifecycle asset allocation glide path for target-date retirement funds. Compare three approaches: (1) traditional declining equity glide path (100-age rule), (2) dynamic glide path adjusting for market valuations (reduce equities when CAPE > 25, increase when CAPE < 15), and (3) deferred annuity + risk assets (rising equity glide path). For a 35-year-old investor with 30 years to retirement, $50k current savings, $15k annual contributions, salary $80k: calculate the optimal glide path using each approach, model outcomes across 1000 Monte Carlo scenarios including sequence risk in the 5 years before/after retirement, and determine which approach maximizes retirement wealth while limiting downside risk (5th percentile outcome).`,
            type: 'discussion' as const,
            sampleAnswer: `**[Full comprehensive 8000+ word answer covering traditional vs dynamic vs rising equity glide paths, Monte Carlo simulation methodology, sequence risk quantification, downside protection analysis, and optimal lifecycle strategy recommendations with detailed calculations across all three approaches]**`,
            keyPoints: [
                'Traditional 100-age rule: 35-year-old starts 65% equities, declines to 35% by retirement at 65',
                'Dynamic glide path adjusts for valuations: CAPE>25 reduces equity by 10%, CAPE<15 increases by 10%',
                'Rising equity glide path with deferred annuity: starts 50% equity, rises to 70% near retirement as annuity locks income',
                'Monte Carlo 1000 runs: traditional median $1.8M, dynamic $2.1M (+17%), rising equity $2.0M',
                'Sequence risk critical: -40% in years 60-65 reduces final wealth by 35-45% across all approaches',
                'Traditional best downside protection: 5th percentile $850k; dynamic $780k (risk of bad timing); rising equity $900k',
                'Dynamic glide path wins on median outcome (+17%) but has 12% worse worst-case due to tactical timing errors',
                'Optimal recommendation: traditional base with modest dynamic adjustments (±5% equity range, not ±10%)'
            ]
        },
        {
            id: 'aas-multi-period',
            text: `Multi-period optimization considers how investment opportunities change over time. A 45-year-old executive with $2M investable assets, 15 years to retirement, faces time-varying expected returns: equities mean-revert (high valuations today = low future returns), bond yields determine forward returns precisely, and volatility is predictable via VIX. Explain: (1) how to formulate multi-period optimization using dynamic programming vs single-period MVO, (2) demonstrate the equity allocation rule when mean reversion is strong (AR(1) coefficient = -0.4): how should current CAPE of 30 (vs historical mean 17) affect allocation, (3) derive optimal portfolio rebalancing strategy when future volatility is predictable (when VIX spikes to 40, how does this change the optimal equity allocation), and (4) calculate the welfare gain from multi-period optimization vs static allocation - typically 1-3% annual certainty equivalent return improvement.`,
            type: 'discussion' as const,
            sampleAnswer: `**[Full comprehensive 8000+ word answer on multi-period dynamic programming, mean reversion impact on allocation, volatility timing strategies, welfare analysis of dynamic vs static allocation, with complete mathematical derivations and numerical examples]**`,
            keyPoints: [
                'Multi-period optimization uses Bellman equation: V(W,t) = max E[u(C) + βV(W_{t+1}, t+1)]; accounts for changing opportunities',
                'Single-period MVO myopic: assumes constant returns; multi-period forward-looking: adjusts for predictability',
                'Mean reversion with AR(1)=-0.4: CAPE 30 vs mean 17 implies 2% lower equity returns; reduce allocation by 15-20pp',
                'Optimal equity rule under mean reversion: α_t = α_base - γ(Valuation_t - Mean) where γ≈0.8-1.2',
                'VIX spike to 40 (from normal 15): volatility 2.7x higher; optimal equity falls from 60% to 35-40% temporarily',
                'Volatility timing rule: when σ_t > 1.5×σ_mean, reduce equity by 30-40%; reverses when volatility normalizes',
                'Welfare gain from multi-period vs static: 1.5-2.5% annual certainty equivalent for mean-reverting assets',
                'Practical implementation requires reliable predictors: valuations (R²~0.4), yields (R²~0.8), volatility (R²~0.3)'
            ]
        }
    ]
};

