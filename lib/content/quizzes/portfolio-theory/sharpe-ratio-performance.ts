export const sharpeRatioPerformanceQuiz = {
    id: 'sharpe-ratio-performance',
    title: 'Sharpe Ratio and Performance Metrics',
    questions: [
        {
            id: 'srp-comparison',
            text: `Three fund managers present their 5-year track records:

Manager A: 18% return, 22% volatility, 15% downside deviation
Manager B: 14% return, 16% volatility, 10% downside deviation  
Manager C: 12% return, 12% volatility, 8% downside deviation

Risk-free rate: 4%, S&P 500 benchmark: 11% return, 18% volatility

Calculate and compare: (1) Sharpe ratio, Sortino ratio, and Treynor ratio (assume betas: A=1.2, B=0.9, C=0.7), (2) explain which manager performed best and why your answer depends on the metric chosen, (3) calculate the M² (Modigliani-Modigliani) measure to compare on a common risk basis, and (4) discuss why Sharpe ratio can be misleading for hedge funds with non-normal return distributions.`,
            type: 'discussion' as const,
            sampleAnswer: `**Complete Risk-Adjusted Performance Analysis**

**1. Sharpe Ratio Calculations:**

Sharpe = (Return - Risk-free) / Volatility

- **Manager A**: (18% - 4%) / 22% = 14% / 22% = **0.636**
- **Manager B**: (14% - 4%) / 16% = 10% / 16% = **0.625**
- **Manager C**: (12% - 4%) / 12% = 8% / 12% = **0.667**
- **Benchmark**: (11% - 4%) / 18% = 7% / 18% = **0.389**

**Winner: Manager C** (highest risk-adjusted return per unit of total risk)

**2. Sortino Ratio Calculations:**

Sortino = (Return - Risk-free) / Downside Deviation

- **Manager A**: 14% / 15% = **0.933**
- **Manager B**: 10% / 10% = **1.000**
- **Manager C**: 8% / 8% = **1.000**

**Winner: Tie between B and C** (both achieve 1.0, indicating excellent downside risk management)

The Sortino ratios are notably higher than Sharpe ratios because downside deviation < total volatility, indicating that all managers have positive skew (larger upside moves than downside).

**3. Treynor Ratio Calculations:**

Treynor = (Return - Risk-free) / Beta

- **Manager A**: 14% / 1.2 = **11.67%**
- **Manager B**: 10% / 0.9 = **11.11%**
- **Manager C**: 8% / 0.7 = **11.43%**
- **Benchmark**: 7% / 1.0 = **7.00%**

**Winner: Manager A** (highest return per unit of systematic risk)

All managers significantly outperform benchmark on Treynor basis (11+% vs 7%), indicating genuine alpha generation beyond market exposure.

**4. M² (Modigliani-Modigliani) Measure:**

M² adjusts each portfolio to match benchmark volatility (18%), then compares returns directly.

Formula: M² = Rf + Sharpe_portfolio × σ_benchmark

- **Manager A**: 4% + 0.636 × 18% = 4% + 11.45% = **15.45%**
- **Manager B**: 4% + 0.625 × 18% = 4% + 11.25% = **15.25%**
- **Manager C**: 4% + 0.667 × 18% = 4% + 12.01% = **16.01%**
- **Benchmark**: 11.00% (by definition)

**Winner: Manager C** (16.01% vs benchmark 11%, outperformance of 5.01%)

M² makes performance directly comparable: if Manager C were levered to 18% volatility (1.5x leverage since current vol is 12%), expected return would be 16.01%, crushing the benchmark.

**5. Which Manager Performed Best? It Depends!**

**For a Well-Diversified Investor** (pension fund, endowment):
- **Best: Manager A** (Treynor 11.67%)
- Reason: Treynor measures return per unit of *systematic risk*. Since idiosyncratic risk is diversified away in large portfolios, systematic risk (beta) is what matters. Manager A generates most alpha per unit of market exposure.

**For an Individual Investor** (single-fund holder):
- **Best: Manager C** (Sharpe 0.667, M² 16.01%)
- Reason: Sharpe measures return per unit of *total risk*. For concentrated investors, total volatility matters. Manager C provides best risk-adjusted returns on standalone basis.

**For Risk-Averse Investors** (concerned about downside):
- **Best: Manager B or C** (Sortino 1.000)
- Reason: Sortino focuses on downside risk. Both B and C manage downside exceptionally well relative to upside potential.

**Summary Table:**

| Metric | Manager A | Manager B | Manager C | Winner |
|--------|-----------|-----------|-----------|--------|
| Sharpe | 0.636 | 0.625 | 0.667 | C |
| Sortino | 0.933 | 1.000 | 1.000 | B/C |
| Treynor | 11.67% | 11.11% | 11.43% | A |
| M² | 15.45% | 15.25% | 16.01% | C |

**6. Why Sharpe Ratio Misleads for Hedge Funds**

**Assumption Violations:**

Sharpe ratio assumes returns are **normally distributed** (symmetric bell curve). Hedge funds often violate this:

**Problem 1: Negative Skewness**
- Strategy: Selling out-of-money puts (collecting premium)
- Returns: Small gains 95% of time (+2% monthly), rare huge losses (-40% once)
- Sharpe: Looks great! High average, low measured volatility (most months calm)
- Reality: Hidden tail risk. Strategy is "picking up pennies in front of steamroller"
- Real example: Long-Term Capital Management (LTCM) had Sharpe > 2.0 until 1998 implosion

**Problem 2: Positive Skewness**
- Strategy: Trend-following / tail hedging (buying protection)
- Returns: Small losses frequently (-0.5% monthly), rare huge gains (+50% in crashes)
- Sharpe: Looks terrible! Negative most months, high volatility
- Reality: Valuable insurance. Pays off when needed most (2008, 2020)
- Real example: Universa had negative Sharpe but gained 4000% in March 2020

**Problem 3: Fat Tails (High Kurtosis)**
- Normal distribution: 99.7% of observations within 3 standard deviations
- Hedge fund reality: "6-sigma events" happen frequently
- Sharpe: Underestimates probability of extreme losses
- Example: "Black Monday" 1987 was theoretically a 20-sigma event (should occur once per billion years), yet markets have multiple 5+ sigma events per decade

**Problem 4: Autocorrelation**
- Some strategies smooth returns (illiquid positions, marking practices)
- Measured volatility artificially low
- Sharpe overstates risk-adjusted performance
- Example: Madoff reported impossibly consistent returns with Sharpe > 2.5 (fraud, but red flag)

**Better Alternatives for Hedge Funds:**

1. **Omega Ratio**: Ratio of probability-weighted gains to losses. Captures full distribution, not just first two moments.

2. **Sortino Ratio**: Penalizes only downside volatility. Better for asymmetric strategies.

3. **Calmar Ratio**: Return / Maximum Drawdown. Focuses on worst-case outcomes.

4. **Tail Risk Measures**: Conditional Value at Risk (CVaR), Expected Shortfall. Quantify extreme losses.

**Institutional Practice:**
- Sophisticated investors (CalPERS, Yale Endowment) use **multiple metrics**
- Sharpe as starting point, supplemented with skewness, kurtosis, max drawdown, tail ratios
- **Stress testing** critical: How does fund perform in 2008-style crisis?

**Key Insight:** Sharpe ratio is excellent for traditional long-only investments (stocks, bonds, 60/40). For complex strategies with options, leverage, or dynamic allocation, it's necessary but insufficient. Always complement with distribution-aware metrics and scenario analysis.`,
            keyPoints: [
                'Sharpe ratio = (Return - Rf) / Volatility; measures return per unit of total risk',
                'Sortino ratio uses downside deviation instead of total volatility; better for asymmetric returns',
                'Treynor ratio = (Return - Rf) / Beta; measures return per unit of systematic risk',
                'M² adjusts portfolios to benchmark volatility for direct return comparison',
                'Different metrics can rank managers differently based on what risk is measured',
                'Sharpe ratio misleading for hedge funds due to fat tails, skewness, autocorrelation',
                'Manager selection depends on investor context: diversified vs concentrated portfolio',
                'Multiple metrics needed for comprehensive performance evaluation: no single "best" measure'
            ]
        },
        {
            id: 'srp-information-ratio',
            text: `An active equity fund reports: portfolio return 13.5%, benchmark return 11%, tracking error 4.2%. Calculate: (1) the information ratio and interpret what it means for the fund's active management quality, (2) decompose the fund's total Sharpe ratio into benchmark Sharpe ratio and information ratio contributions, (3) determine how much tracking error is "worth it" given the information ratio, and (4) compare this fund to another with 15% return, 7% tracking error at the same benchmark - which demonstrates better active management skill?`,
            type: 'discussion' as const,
            sampleAnswer: `**Comprehensive Information Ratio Analysis**

**1. Information Ratio Calculation:**

Information Ratio (IR) = Active Return / Tracking Error

Active Return = Portfolio Return - Benchmark Return
= 13.5% - 11.0% = **2.5%**

Tracking Error (TE) = Standard deviation of active returns = **4.2%** (given)

**Information Ratio = 2.5% / 4.2% = 0.595**

**Interpretation:**

- **IR = 0.595 is EXCELLENT** (top quartile active management)
- Benchmarks: IR > 0.5 = excellent, 0.25-0.5 = good, <0.25 = marginal
- Meaning: Fund generates 0.595% excess return for each 1% of tracking error
- Consistency: High IR indicates skill, not luck. The 2.5% alpha is consistent relative to 4.2% active risk.

**What This Tells Us About Manager Skill:**

- **Predictability**: 0.595 IR suggests active bets are consistently rewarded
- **Risk-Adjusted Alpha**: The 2.5% outperformance is substantial given moderate 4.2% tracking error
- **Sustainability**: High IR (>0.5) is rare and difficult to maintain; likely indicates genuine skill
- **Fee Justification**: Even after 1% management fee, delivers 1.5% net alpha

**Statistical Significance:**

To verify significance, calculate t-statistic = IR × √n
Assuming 5 years (60 months): t = 0.595 × √60 = 4.61

With t > 2.0, the IR is highly statistically significant (p < 0.001). This is not luck!

**2. Sharpe Ratio Decomposition:**

**Fundamental Law of Active Management:**

Portfolio Sharpe ≈ Benchmark Sharpe + Information Ratio

Assume:
- Risk-free rate = 3%
- Benchmark volatility = 16%

**Benchmark Sharpe = (11% - 3%) / 16% = 0.50**

**Portfolio Sharpe:**
- Need portfolio volatility. Using approximation: σ²_portfolio ≈ σ²_benchmark + TE²
- σ_portfolio = √(16² + 4.2²) = √(256 + 17.64) = √273.64 = 16.54%
- Portfolio Sharpe = (13.5% - 3%) / 16.54% = **0.635**

**Decomposition:**
- Baseline (from benchmark exposure): 0.50
- Added value (from active management): 0.135
- Total Sharpe: 0.635

**Verification:** 0.50 + 0.135 ≈ 0.635 ✓ (close, small differences due to approximation)

**Insight:** The Information Ratio (0.595) contributes approximately 0.135 to the Sharpe ratio. This is substantial! The fund improved risk-adjusted returns by 27% (0.635 vs 0.50) through active management.

**3. Is 4.2% Tracking Error "Worth It"?**

**Break-Even Analysis:**

Minimum required alpha to justify TE: Must cover fees + opportunity cost

- Management fee: ~1.0% typically
- Opportunity cost: Could earn IR × TE from active management
- Break-even: Alpha ≥ Fee

This fund: 2.5% alpha > 1.0% fee → **YES, tracking error is worth it**

**Optimal Tracking Error:**

Theory: Optimal TE* = IR / λ, where λ = investor's risk aversion (typically 2-4)

With IR = 0.595 and λ = 3:
TE* = 0.595 / 3 = **19.8%** !

Wait, this suggests optimal TE is much higher than actual 4.2%. This means the fund could take MORE active risk!

**Corrected Formula:** TE* = (IR × σ_benchmark) / (λ × Sharpe_benchmark)

TE* = (0.595 × 16%) / (3 × 0.50) = 9.52% / 1.5 = **6.35%**

**Conclusion:** Optimal TE ≈ 6.35%, but fund uses only 4.2%. The fund is being conservative! It could increase active risk to ~6% and (if IR holds) increase alpha to ~3.8%. However, prudent managers often stay below optimal to avoid career risk.

**4. Comparison with Alternative Fund:**

**Fund A (Original):**
- Return: 13.5%, Benchmark: 11%
- Active Return: 2.5%
- Tracking Error: 4.2%
- **Information Ratio: 0.595**

**Fund B (Alternative):**
- Return: 15%, Benchmark: 11%
- Active Return: 4.0%
- Tracking Error: 7.0%
- **Information Ratio: 4.0% / 7.0% = 0.571**

**Which Shows Better Skill?**

**Fund A wins on Information Ratio (0.595 > 0.571)**

**Analysis:**

- Fund B has higher absolute alpha (4.0% vs 2.5%) ✓
- But Fund B takes 67% more tracking error (7.0% vs 4.2%) ✗
- Fund A is more efficient: generates 0.595% per unit of TE vs 0.571% for Fund B
- **Fund A demonstrates better active management skill** (more consistent, less risky)

**Decision Depends on Investor Type:**

**For Conservative Investors (low risk tolerance):**
- **Choose Fund A**: Lower tracking error (4.2% vs 7%), still good alpha (2.5%)
- Blend with index for total portfolio: 70% index + 30% Fund A = 11.75% return, 1.3% TE

**For Aggressive Investors (high risk tolerance):**
- **Choose Fund B**: Higher absolute alpha (4.0%), can tolerate 7% TE
- Or leverage Fund A: 1.67× Fund A ≈ 4.2% alpha with 7% TE (same risk, better return!)

**For Return-Maximizers:**
- Fund B delivers more alpha (4.0% > 2.5%), if fees are equal

**For Risk-Adjusted Return Seekers:**
- Fund A has better efficiency (IR 0.595 > 0.571)

**Practical Considerations:**

1. **Capacity**: Fund A likely has higher capacity (lower TE = less extreme positions)
2. **Stability**: IR 0.595 more likely to persist than high-alpha-high-TE strategies
3. **Career Risk**: Fund B manager takes bigger bets; riskier for their job
4. **Fees**: If Fund B charges higher fees, Fund A clearly wins

**Statistical Robustness:**

Over 5 years (60 months):
- Fund A: t-stat = 0.595 × √60 = 4.61 (highly significant)
- Fund B: t-stat = 0.571 × √60 = 4.42 (highly significant)

Both are statistically significant, but Fund A is slightly more robust.

**Final Verdict:**

**Fund A demonstrates superior active management skill** due to higher Information Ratio. While Fund B delivers more alpha, it does so at a higher risk level. Fund A is more efficient, consistent, and likely sustainable. For most institutional investors, Fund A is the better choice unless they have specific reasons to target higher tracking error.

**Key Principle:** In active management, it's not about how much you make, but how efficiently you make it relative to the active risk taken. Information Ratio is the gold standard for measuring manager skill, more important than absolute alpha.`,
            keyPoints: [
                'Information ratio = Active return / Tracking error; measures consistency of outperformance',
                'IR > 0.5 is excellent, 0.25-0.5 is good, <0.25 is marginal active management',
                'Portfolio Sharpe ≈ Benchmark Sharpe + IR (fundamental law of active management)',
                'Optimal tracking error increases with higher information ratio but constrained by risk aversion',
                'High tracking error only justified if IR is proportionally high (efficient alpha generation)',
                'IR more stable than raw alpha; better measure of manager skill over time',
                'Fund A: IR=0.595 vs Fund B: IR=0.571; Fund A shows better skill despite lower absolute alpha',
                'Consistency matters more than magnitude for long-term active management success and sustainability'
            ]
        },
        {
            id: 'srp-advanced-metrics',
            text: `You're evaluating a market-neutral hedge fund with the following 3-year statistics: Annual return 8%, volatility 12%, skewness -0.8, kurtosis 6.0, maximum drawdown -18%, worst month -8%, average month 0.6%, correlation with S&P 500: 0.15. The fund charges 2% management fee and 20% performance fee. Analyze: (1) why traditional Sharpe ratio understates the fund's risk given its negative skewness and high kurtosis, (2) calculate alternative risk-adjusted metrics (Omega ratio, Sortino ratio, Calmar ratio) that better capture tail risk, (3) adjust returns for fees and calculate investor Sharpe ratio vs gross Sharpe ratio, and (4) determine if the fund's low market correlation justifies its high fees for portfolio diversification.`,
            type: 'discussion' as const,
            sampleAnswer: `**Comprehensive Hedge Fund Performance Evaluation with Non-Normal Distribution Analysis**

**Given Statistics:**
- Annual return: 8%
- Volatility: 12%
- Skewness: -0.8 (negative = left tail risk)
- Kurtosis: 6.0 (excess kurtosis 3.0 = fat tails)
- Maximum drawdown: -18%
- Worst month: -8%
- Average month: 0.6% (× 12 = 7.2% simple, 8% compound)
- Correlation with S&P 500: 0.15
- Fees: 2% + 20% performance

**1. Why Sharpe Ratio Understates Risk:**

**Traditional Sharpe Ratio (Risk-free = 3%):**
Sharpe = (8% - 3%) / 12% = 0.417

This looks respectable! Above 0.40 is considered good. But this massively understates the true risk.

**Problem: Negative Skewness (-0.8)**

Normal distribution: skewness = 0 (symmetric)
This fund: skewness = -0.8 (strongly negatively skewed)

**Interpretation:**
- Returns are NOT symmetric
- Mean return (8%) > Median return (likely ~6%)
- Left tail is longer and fatter than right tail
- Rare large losses, frequent small gains
- **"Picking up pennies in front of steamroller" pattern**

**Visual Distribution:**
```
Normal(skew = 0): Bell curve symmetric
This fund(skew = -0.8):  ⟋ Compressed right, extended left
```

Most months: +0.6% to +2% (small gains)
Rare months: -8% to worse (large losses)

**The Danger:** Sharpe uses standard deviation, which treats upside and downside volatility equally. But investors hate downside! A portfolio with volatility from all losses is worse than one with volatility from all gains, yet Sharpe treats them the same.

**Problem: High Kurtosis (6.0)**

Normal distribution: kurtosis = 3 (baseline)
This fund: kurtosis = 6.0 (excess kurtosis = 3.0)

**Interpretation:**
- Excess kurtosis = 3.0 means **fat tails** (leptokurtic)
- Extreme events happen 3× more often than normal distribution predicts
- Normal: 99.7% within 3σ; This fund: ~95% within 3σ
- "6-sigma" events occur with disturbing frequency

**Concrete Risk:**

Under normal distribution with 12% volatility:
- Monthly vol = 12% / √12 = 3.46%
- 3σ monthly move = 10.4%
- P(loss > 10%) ≈ 0.13% (once per 64 years)

With kurtosis = 6 and negative skew:
- P(loss > 10%) ≈ 1-2% (once per 4-8 years)
- **10× more likely than Sharpe suggests!**

**Real-World Example:** Fund looks stable 95% of time, then delivers -15% in single month during crisis. Investors think they have 12% vol "moderate risk" fund, actually have significant tail risk.

**Sharpe's Fatal Flaw:** It summarizes entire distribution with 2 numbers (mean, std dev). Ignores shape (skewness, kurtosis). For hedge funds, the *shape* is the risk!

**2. Alternative Risk-Adjusted Metrics:**

**A. Sortino Ratio**

Sortino = (Return - MAR) / Downside Deviation

MAR (Minimum Acceptable Return) = 3% (risk-free)

Downside deviation = std dev of returns below MAR

**Approximation:** With negative skew, downside returns ~60-70% of months
Downside vol ≈ 12% × 1.2 (skew adjustment) = **14.4%**

Sortino = (8% - 3%) / 14.4% = **0.347**

**Comparison:**
- Sharpe: 0.417
- Sortino: 0.347 (16% lower)

Sortino properly penalizes the fat left tail. The fund's risk-adjusted performance looks worse when we focus on downside.

**B. Calmar Ratio**

Calmar = Annual Return / Maximum Drawdown

Calmar = 8% / 18% = **0.444**

**Interpretation:**
- Calmar 0.444 means fund earns 0.44% annually per 1% of max drawdown
- Good hedge funds target Calmar > 0.5
- This fund's 0.444 is borderline acceptable

**Comparison to peers:**
- Renaissance Medallion: Calmar > 2.0 (extraordinary)
- Bridgewater Pure Alpha: Calmar ~0.8 (excellent)
- This fund: Calmar 0.444 (mediocre)

**Key Insight:** The -18% max drawdown over just 3 years is concerning. Extrapolated over 10+ years, max drawdown could reach -30% or worse.

**C. Omega Ratio**

Omega = Probability-weighted gains / Probability-weighted losses (above/below threshold)

Omega(threshold = 0%) = ∫P(R>threshold) / ∫P(R<threshold)

**Approximation:** 
- % positive months: ~70% (implied by 0.6% avg, 12% vol, negative skew)
- Avg gain when positive: +2.5%
- Avg loss when negative: -4.5% (negative skew = larger losses)
- Omega ≈ (0.70 × 2.5%) / (0.30 × 4.5%) = 1.75% / 1.35% = **1.296**

**Interpretation:**
- Omega > 1 means gains outweigh losses (good!)
- Omega = 1.296 is modest
- Hedge funds typically target Omega > 1.5 at 0% threshold

**Comparison:**
- Sharpe (normalized): 0.417 looks decent
- Omega: 1.296 reveals the asymmetry penalty

**3. Fee-Adjusted Returns:**

**Fee Structure: 2% + 20% performance above hurdle**

Assume 0% hurdle (typical):
- Gross return: 8.0%
- Performance fee: 20% × 8.0% = 1.6%
- Management fee: 2.0%
- **Total fees: 3.6%**

**Net return: 8.0% - 3.6% = 4.4%**

**Net Sharpe Ratio:**
Net Sharpe = (4.4% - 3.0%) / 12% = 1.4% / 12% = **0.117**

**Dramatic Decline:**
- Gross Sharpe: 0.417
- Net Sharpe: 0.117 (72% reduction!)

**Investor Reality Check:**

With net Sharpe 0.117, the fund barely beats risk-free after fees!

**Comparison to Alternatives:**
- 60/40 stock/bond portfolio: 7% return, 10% vol, 0.05% fees → Sharpe 0.39
- This hedge fund net: 4.4% return, 12% vol → Sharpe 0.117

**The 60/40 has 3.3× better risk-adjusted performance!**

**Fee Impact Visualization:**
```
Gross: 8 % return → Sharpe 0.417 ✓
    After mgmt fee: 6 % → Sharpe 0.250 ✗
    After perf fee: 4.4 % → Sharpe 0.117 ✗✗
```

**Is Fee Structure Fair?**

Management fee (2%) on gross return = 25% of net return (2% / 8%)
Performance fee (20%) = another 20%
**Total: Managers take 45% of gross profits!**

This is only justified if fund provides unique alpha or diversification value...

**4. Does Low Correlation Justify High Fees?**

**Diversification Value Analysis:**

**Given:**
- Fund correlation with S&P 500: 0.15 (very low!)
- S&P 500: assume 10% return, 18% vol, Sharpe 0.39

**Portfolio Optimization:**

Investor holds 80% S&P 500, 20% hedge fund:

Portfolio return = 0.80 × 10% + 0.20 × 4.4% = 8.0% + 0.88% = **8.88%**

Portfolio variance = w₁²σ₁² + w₂²σ₂² + 2w₁w₂ρσ₁σ₂
= 0.80²(0.18²) + 0.20²(0.12²) + 2(0.80)(0.20)(0.15)(0.18)(0.12)
= 0.0207 + 0.0006 + 0.0010 = 0.0223

Portfolio vol = √0.0223 = **14.93%**

**Portfolio Sharpe = (8.88% - 3%) / 14.93% = 5.88% / 14.93% = 0.394**

**Comparison:**
- 100% S&P 500: 10% return, 18% vol, Sharpe 0.39
- 80/20 mix: 8.88% return, 14.93% vol, Sharpe 0.394

**Result:** 
- Return drops 11% (10% → 8.88%)
- Risk drops 17% (18% → 14.93%)
- Sharpe roughly constant (0.39 → 0.394)

**Diversification Benefit:** Very modest. The low correlation helps reduce vol from 18% to 14.93%, but the low net return drags down portfolio performance.

**Better Alternatives:**

**80/20 S&P/Bonds (corr = 0.2, bond return 4%, vol 6%):**
- Portfolio return: 0.80 × 10% + 0.20 × 4% = 8.8%
- Portfolio vol: ~15% (similar calculation)
- Sharpe: ~0.39
- **Cost: 0.05% fees vs 3.6%**

**Conclusion:** Bonds provide similar diversification at 1/72nd the cost!

**When Would Hedge Fund Be Justified?**

The fund needs to provide:
1. **Crisis alpha**: Gains during market crashes (like March 2020)
2. **Unique exposures**: Access to strategies unavailable elsewhere
3. **Convexity**: Non-linear payoffs that protect in tail events

**Does this fund qualify?**

- Correlation 0.15 suggests some independence ✓
- But max drawdown -18% in normal times suggests it won't protect in crisis ✗
- Negative skew suggests it may suffer MORE in crashes ✗✗

**Final Verdict:**

**Fund Does NOT Justify High Fees:**

1. **Risk Understatement:** Sharpe 0.417 looks okay, but negative skew + high kurtosis = hidden tail risk
2. **Poor Net Performance:** Net Sharpe 0.117 is abysmal after 3.6% fees
3. **Alternative Metrics:** Sortino 0.347, Calmar 0.444, Omega 1.296 all reveal mediocre risk-adjusted performance
4. **Diversification Failure:** Low correlation helps modestly, but bonds deliver same benefit at 1/72nd the cost
5. **Fee Extraction:** Managers keep 45% of gross profits; investors get 55%

**Recommendation:**
- **Pass on this fund**
- Alternative: 60/40 stocks/bonds cheaper, similar risk-adjusted returns, no tail risk
- Or: Seek hedge funds with positive skew (tail hedge), or much higher Sharpe (>1.0 gross)

**Red Flags:**
- Negative skew + fees = investor gets worst of both worlds (limited upside, full downside)
- 3-year track record too short to evaluate skill vs luck
- -18% drawdown in benign period suggests worse to come

**Key Lesson:** Always adjust for fees and non-normal distributions when evaluating hedge funds. Sharpe ratio alone is dangerously misleading. Most hedge funds don't justify 2/20 fees after proper risk adjustment.`,
    keyPoints: [
        'Negative skewness (-0.8) and high kurtosis (6.0) indicate fat left tail; Sharpe ratio ignores this tail risk',
        'Omega ratio captures full return distribution; better for non-normal returns and asymmetric strategies',
        'Sortino (0.347) and Calmar (0.444) ratios focus on downside; more appropriate for tail risk assessment',
        'Fee drag: 2/20 structure reduces investor returns by 3.6% annually (45% of gross profits to managers)',
        'Gross Sharpe 0.417 vs Net Sharpe 0.117 after fees; 72% degradation reveals fee impact',
        'Low correlation (0.15) adds modest diversification value but bonds achieve similar benefit at far lower cost',
        'Hedge fund justification requires crisis alpha or unique exposures; this fund fails both tests',
        'Most hedge funds fail to justify fees after adjusting for non-normal distribution risks and comparing alternatives'
    ]
    }
  ]
};
