export const riskReturnMetricsQuiz = {
  id: 'risk-return-metrics',
  title: 'Risk and Return Metrics',
  questions: [
    {
      id: 'rrm-sharpe-sortino',
      text: `Compare and contrast the Sharpe Ratio and Sortino Ratio as measures of risk-adjusted return. A hedge fund reports the following 5-year track record:

- Annualized Return: 15%
- Total Volatility: 18%
- Downside Deviation (below 0%): 12%
- Risk-Free Rate: 3%

Calculate both the Sharpe and Sortino ratios. Explain why they differ, discuss what each ratio tells us about the fund's performance, and describe investment scenarios where Sortino Ratio would be more appropriate than Sharpe Ratio. Include at least three specific fund strategies where this distinction matters significantly.`,
      type: 'discussion' as const,
      sampleAnswer: `**Calculations:**

**Sharpe Ratio:**
Sharpe = (Rₚ - Rf) / σₚ
       = (15% - 3%) / 18%
       = 12% / 18%
       = 0.667

**Sortino Ratio:**
Sortino = (Rₚ - Rf) / σdownside
        = (15% - 3%) / 12%
        = 12% / 12%
        = 1.00

**Key Difference: Sortino (1.00) is 50% higher than Sharpe (0.667)**

**Why They Differ:**1. **Different Risk Measures:**
   - Sharpe uses total volatility (18%) - penalizes both upside and downside volatility
   - Sortino uses downside deviation (12%) - only penalizes downside volatility

2. **Implied Return Distribution:**
   - Total volatility = 18%, downside deviation = 12%
   - Ratio: 12/18 = 0.67
   - This suggests returns are **positively skewed** (more upside volatility than downside)
   - The fund has "good volatility" on the upside that Sharpe penalizes unfairly

3. **Investor Perspective:**
   - Investors care about downside risk (losses), not upside volatility (gains)
   - Sortino better captures investor's true risk tolerance
   - Fund with positive skew looks better under Sortino than Sharpe

**What Each Ratio Tells Us:**

**Sharpe Ratio (0.667):**
- Moderate risk-adjusted return
- For every unit of total risk, the fund generates 0.667 units of excess return
- Comparison: S&P 500 long-term Sharpe ≈ 0.5, so fund outperforms on this metric
- **Limitation:** Treats upside and downside volatility the same

**Sortino Ratio (1.00):**
- Strong risk-adjusted return when focusing on downside
- For every unit of downside risk, generates 1.00 units of excess return
- The fund delivers attractive returns while limiting downside
- **Advantage:** Aligns with investor preference (asymmetric risk aversion)
- **Key Insight:** Fund's actual risk (what investors care about) is lower than total volatility suggests

**Performance Interpretation:**

The 50% difference between Sharpe and Sortino reveals:
1. Fund has **positive skewness** - more big up months than big down months
2. Downside protection is better than total volatility implies
3. Strategy likely includes some downside hedging or asymmetric payoffs
4. More attractive to risk-averse investors than Sharpe ratio alone suggests

**When Sortino Is More Appropriate:**

**Scenario 1: Options-Based Strategies**

*Example: Covered Call Writing*
- Strategy: Long S&P 500 + sell call options
- Return profile: Limited upside (capped at strike), full downside exposure
- Metrics:
  - Total volatility: ~15% (from underlying stocks)
  - Downside volatility: ~14% (similar to total, as downside not protected)
  - Sharpe: Moderate (~0.4)
  - Sortino: Similar to Sharpe (~0.45)
  - **Conclusion:** Sortino doesn't show advantage because strategy doesn't provide downside protection

*Example: Put-Protected Equity (Long stocks + long puts)*
- Strategy: S&P 500 + protective put options
- Return profile: Limited downside (floor at put strike), full upside
- Metrics:
  - Total volatility: ~12% (reduced from 18% by put protection)
  - Downside volatility: ~6% (significantly protected)
  - Sharpe: ~0.5 (penalized for put premium cost)
  - Sortino: ~1.2 (reflects asymmetric protection)
  - **Conclusion:** Sortino much more appropriate! Shows the true value of downside protection

**Scenario 2: Trend-Following / Managed Futures**

*Example: CTA Strategy*
- Strategy: Follow momentum in multiple asset classes with stop-losses
- Return profile: Small frequent losses (stopped out), occasional large gains (riding trends)
- Positive skew: Many small losses, few large gains
- Metrics:
  - Total volatility: 20% (high due to momentum whipsaws)
  - Downside volatility: 10% (stops limit losses)
  - Sharpe: 0.35 (looks mediocre due to high total vol)
  - Sortino: 0.90 (looks much better reflecting limited downside)
  - **Conclusion:** Sortino better captures strategy's risk management effectiveness

**Scenario 3: Long-Volatility Strategies**

*Example: Tail-Risk Hedging Fund*
- Strategy: Long out-of-the-money puts, other convex strategies
- Return profile: Small steady losses in calm markets, massive gains in crashes
- Extreme positive skew
- Metrics:
  - Annual return: 2% (due to option decay in normal years)
  - Total volatility: 35% (huge spikes in crash years)
  - Downside volatility: 8% (limited downside, as strategy gains in worst markets)
  - Sharpe: -0.05 (looks terrible! Negative Sharpe due to steady decay)
  - Sortino: 0.50 (looks reasonable when considering it's insurance)
  - **Conclusion:** Sortino essential for evaluating this type of strategy. Sharpe completely misleading.

**Additional Scenarios:**

**4. Private Equity / Venture Capital:**
- Return profile: Total losses on failed investments, 10-100x returns on winners
- Extreme positive skew
- Sortino better captures the asymmetric payoff structure

**5. Value Investing with Downside Focus:**
- Strategy: Buy distressed value stocks with margin of safety
- Focus on downside protection through low valuation entry
- Sortino better reflects the downside-focused risk management approach

**6. Dynamic Hedging Strategies:**
- Adjust hedge ratios based on market conditions
- Increase protection in volatile markets, reduce in calm markets
- Sortino captures effectiveness of dynamic downside management

**When Sharpe Is Still Appropriate:**1. **Long-only equity strategies** without asymmetric features
2. **Balanced funds** (60/40) with symmetric return distributions
3. **Index funds** tracking broad markets
4. **Market-neutral strategies** with symmetric bets
5. **Situations where total volatility matters** (e.g., leveraged investors facing margin calls from any volatility)

**Practical Recommendation:**

**Report Both Ratios:**
- Sharpe: Industry standard, allows comparison across all strategies
- Sortino: Provides additional insight into downside risk management
- **Sortino/Sharpe ratio:** Indicator of return distribution skewness
  - Ratio ≈ 1.0: Symmetric distribution (normal)
  - Ratio > 1.4: Positive skew (more upside than downside volatility)
  - Ratio < 0.7: Negative skew (more downside than upside volatility - avoid!)

**For the hedge fund in question:**
- Sortino/Sharpe = 1.00 / 0.667 = 1.50
- **Interpretation:** Significantly positive skew, attractive asymmetric return profile
- **Investor conclusion:** More attractive than Sharpe alone suggests, especially for risk-averse investors

**Key Takeaway:** Sortino Ratio is superior for any strategy with asymmetric return profiles (options, trend-following, tail hedging). It better aligns with investor preferences by focusing on downside risk. Always consider both metrics together to understand the full risk-return profile.`,
      keyPoints: [
        'Sharpe Ratio uses total volatility; penalizes both upside and downside deviation from mean',
        'Sortino Ratio uses downside deviation; only penalizes returns below target (typically 0% or risk-free rate)',
        'Sortino > Sharpe indicates positive skewness; strategy has asymmetric upside or downside protection',
        'Options strategies, trend-following, and tail hedging strategies should be evaluated primarily with Sortino',
        'Sortino better aligns with investor psychology: downside risk matters more than upside volatility',
        'Report both ratios: Sortino/Sharpe ratio > 1.4 indicates strong positive skew',
        'Strategies with protective features (puts, stop-losses, dynamic hedging) look better under Sortino than Sharpe',
        'For symmetric return distributions (index funds, balanced portfolios), both ratios converge and Sharpe suffices',
      ],
    },
    {
      id: 'rrm-drawdown-analysis',
      text: `Maximum drawdown is often called the "most important risk metric investors ignore." Analyze two funds with identical 10% annualized returns and 15% volatility but different drawdown profiles:

Fund A: Max Drawdown = -25%, Recovery Time = 18 months, 3 drawdowns >15% over 10 years
Fund B: Max Drawdown = -45%, Recovery Time = 42 months, 1 drawdown >15% over 10 years

Discuss: (1) why these funds can have identical Sharpe ratios but vastly different drawdown experiences, (2) the psychological and practical impacts of deep drawdowns on investor behavior, (3) how to calculate recovery time and why it matters for portfolio compounding, and (4) which metrics besides Sharpe ratio should be used to capture drawdown risk (provide formulas and thresholds).`,
      type: 'discussion' as const,
      sampleAnswer: `**1. Why Identical Sharpe Ratios, Different Drawdown Profiles**

**Sharpe Ratio Calculation (both funds):**
- Sharpe = (10% - 3%) / 15% = 0.467 (identical)

**Why This Happens:**

Sharpe ratio is a **mean-variance** metric that ignores:
- **Path dependence:** Order and clustering of returns
- **Higher moments:** Skewness and kurtosis
- **Tail risk:** Distribution of extreme negative returns
- **Serial correlation:** Whether bad months cluster together

**Fund A vs Fund B Return Paths:**

**Fund A:** "Steady erosion"
- Three separate -25% drawdowns over 10 years
- Drawdowns spread out (years 2, 5, 8)
- Quick recoveries (18 months average)
- Small, frequent negative returns
- Negative skewness: -0.3

**Fund B:** "Black swan"
- One catastrophic -45% drawdown (year 5)
- Long recovery period (42 months)
- Otherwise smooth positive returns
- Returns concentrated in single event
- High kurtosis: 8.0 (fat tails)

**Mathematical Proof:**

Both have same mean (10%) and standard deviation (15%), but different return sequences:

*Example Monthly Returns:*

Fund A: 1%, -2%, 1%, 1%, -3%, 1%, 1%, -2%, ... (frequent small losses)
Fund B: 1%, 1%, 1%, 1%, 1%, 1%, -20%, 1%, 1%, ... (rare large loss)

Both average 10% annually with 15% volatility, but Fund B's returns are negatively skewed and leptokurtic.

**Key Insight:** Sharpe ratio is **moment-blind beyond mean and variance**. It can't distinguish between:
- Many small drawdowns vs. one large drawdown
- Quick recovery vs. slow recovery
- Steady decline vs. sudden crash

**2. Psychological and Practical Impacts of Deep Drawdowns**

**Psychological Impact: Behavioral Finance**

**Pain Asymmetry (Prospect Theory):**
- Loss aversion: Losses hurt ~2.5x more than equivalent gains feel good
- -45% loss psychologically feels like -112% equivalent pain
- -25% loss feels like -62% equivalent pain
- **Fund B creates 80% more psychological pain despite same returns**

**Recency Bias:**
- Investors anchor on recent performance
- After -45% drawdown, investors traumatized and sell at bottom
- Fund A's smaller drawdowns less likely to trigger panic selling

**Investor Behavior Data:**

*Morningstar Studies:*
- Funds with max drawdown >40%: 60% outflow rate during drawdown
- Funds with max drawdown <30%: 30% outflow rate
- **Fund B would lose 2x more investors, forcing sales at worst time**

**Practical Impacts:**

**1. Margin Calls and Forced Liquidation:**
- Leveraged investor with 2:1 leverage
- -45% drawdown on underlying = -90% on equity (wiped out!)
- -25% drawdown = -50% on equity (severe but survivable)
- **Fund B would trigger margin call, Fund A might not**

**2. Withdrawal Needs (Retirees):**
- Retiree withdrawing 4% annually
- After -45% loss: withdrawing 7.4% of remaining portfolio (unsustainable!)
- After -25% loss: withdrawing 5.3% of remaining (painful but manageable)
- **Fund B could force retiree back to work or drastic lifestyle changes**

**3. Opportunity Cost:**
- 42 months underwater (Fund B) = 3.5 years of zero gain
- Could have earned 3.5 × 10% = 35% in risk-free assets
- 18 months underwater (Fund A) = 1.5 years, 15% opportunity cost
- **Fund B has 2.3x greater opportunity cost**

**4. Career Risk (Professional Managers):**
- Fund manager experiencing -45% drawdown likely fired
- Institutional mandates often have -30% loss trip wires
- **Fund B manager loses job despite identical 10-year returns**

**3. Recovery Time Analysis**

**Recovery Time Formula:**

Time to recover from drawdown of magnitude D with return R:

Recovery Time = ln(1 / (1 - D)) / ln(1 + R)

Where:
- D = drawdown (as decimal, e.g., 0.45 for -45%)
- R = annual return going forward

**Fund A Calculation:**
Recovery from -25% at 10% annual return:
Time = ln(1 / 0.75) / ln(1.10)
     = ln(1.333) / ln(1.10)
     = 0.2877 / 0.0953
     = 3.0 years (36 months, close to 18-month actual suggests recovery was faster than average return)

**Fund B Calculation:**
Recovery from -45% at 10% annual return:
Time = ln(1 / 0.55) / ln(1.10)
     = ln(1.818) / ln(1.10)
     = 0.5978 / 0.0953
     = 6.3 years (75 months, actual was 42 months suggesting strong recovery period)

**Why Recovery Time Matters for Compounding:**

**Compounding Loss:**

Fund B investor:
- Starts: $100,000
- After crash: $55,000 (down 45%)
- Needs +82% return just to break even! (55 × 1.82 = 100)
- At 10% annual: 6.3 years to recover

**Sequence Risk:**

Two investors, both in Fund B:

*Investor 1:* Invests before crash
- Year 0: $100k
- Year 1: $55k (-45%)
- Year 7: $100k (finally recovered)
- Year 10: $133k (cumulative 10% annualized)

*Investor 2:* Invests after crash
- Year 0: $100k  
- Year 10: $259k (pure 10% annual compounding, no drawdown)

**Same fund, same period, Investor 2 ends with 95% more money due to avoiding drawdown!**

**Path Dependence Example:**

Compare two return sequences, both with 10% annualized return:

*Sequence A (Fund A-like):* 15%, 5%, 15%, 5%, 15%, 5%, ... (smooth)
Starting $100k → $259k after 10 years

*Sequence B (Fund B-like):* -45%, 25%, 25%, 12%, 12%, 8%, 8%, 5%, 5%, 5% (crash then recovery)
Starting $100k → $245k after 10 years (6% less despite identical arithmetic mean!)

**4. Metrics to Capture Drawdown Risk**

**Calmar Ratio:**

Formula: Calmar = Annualized Return / |Max Drawdown|

**Fund A:**
Calmar = 10% / 25% = 0.40

**Fund B:**
Calmar = 10% / 45% = 0.22

**Fund A is 80% better on Calmar ratio!**

**Interpretation:**
- Calmar > 0.5: Excellent drawdown-adjusted return
- Calmar 0.3-0.5: Good
- Calmar < 0.3: Poor (Fund B)

**Advantage:** Directly incorporates worst-case drawdown into risk-adjusted return.

**Ulcer Index:**

Measures not just magnitude but also duration of drawdowns.

Formula: UI = √(Σ D²ᵢ / N)

Where Dᵢ = drawdown percentage on day i, N = number of days

**Fund A:**
- Three 18-month drawdowns averaging -15%
- UI ≈ 8-10

**Fund B:**
- One 42-month drawdown at -35% average depth
- UI ≈ 25-30 (much worse!)

**Ulcer Performance Index:**

UPI = (Return - Risk-Free Rate) / Ulcer Index

Better metric than Sharpe for drawdown-focused investors.

**Sterling Ratio:**

Sterling = Annualized Return / (Average of N Worst Drawdowns)

Looks at average of worst drawdowns, not just single worst.

**Fund A:**
Average of 3 drawdowns: (-25% + -18% + -16%) / 3 = -19.7%
Sterling = 10% / 19.7% = 0.51

**Fund B:**
Only has one major drawdown: -45%
Sterling = 10% / 45% = 0.22

**Pain Ratio:**

Pain Ratio = Return / (Sum of Squared Drawdowns)

Heavily penalizes large drawdowns due to squaring.

**Martin Ratio (Ulcer Performance Index):**

Martin = (Return - Rf) / Ulcer Index

**Conditional Drawdown at Risk (CDaR):**

Average drawdown in worst α% of scenarios.

CDaR 95% = average of worst 5% drawdown scenarios

**Recommended Thresholds:**

| Metric | Excellent | Good | Acceptable | Poor |
|--------|-----------|------|------------|------|
| Max Drawdown | <15% | 15-25% | 25-35% | >35% |
| Recovery Time | <12 mo | 12-24 mo | 24-36 mo | >36 mo |
| Calmar Ratio | >0.5 | 0.3-0.5 | 0.2-0.3 | <0.2 |
| Ulcer Index | <5 | 5-10 | 10-15 | >15 |

**Fund Comparison Summary:**

| Metric | Fund A | Fund B | Winner |
|--------|--------|--------|--------|
| Sharpe Ratio | 0.467 | 0.467 | Tie |
| Max Drawdown | -25% | -45% | A (by far) |
| Recovery Time | 18 mo | 42 mo | A |
| Calmar Ratio | 0.40 | 0.22 | A (+80%) |
| Ulcer Index | ~10 | ~28 | A |
| # Severe DDs | 3 | 1 | ? |
| Psychological Impact | Moderate | Severe | A |

**Conclusion:**

Despite identical Sharpe ratios and final returns:
- **Fund A is vastly superior** from risk management perspective
- Fund B's -45% drawdown would cause most investors to sell at bottom
- Recovery time and psychological factors make Fund B's profile unacceptable for most investors
- **Always look beyond Sharpe ratio:** Calmar, Ulcer Index, and max drawdown are critical

**Investment Decision:** Choose Fund A. The 2x smaller drawdown and 2.3x faster recovery make it far more investable despite identical Sharpe ratios. Fund B's profile is only acceptable for investors who:
1. Never check portfolio (extreme discipline)
2. Have no liquidity needs for 5+ years
3. Have iron stomach for -45% losses
4. Are institutions with long time horizons

For 95% of investors, Fund A's drawdown profile is far more appropriate.`,
      keyPoints: [
        'Sharpe ratio is moment-blind beyond mean and variance; cannot distinguish drawdown patterns with same volatility',
        'Maximum drawdown has asymmetric psychological impact due to loss aversion (losses hurt 2.5x more than gains feel good)',
        'Deep drawdowns (-40%+) trigger investor panic selling, margin calls, and can derail retirement plans',
        'Recovery time calculation: ln(1/(1-D)) / ln(1+R); matters because time underwater is opportunity cost',
        'Calmar Ratio = Return/|Max Drawdown| directly captures drawdown-adjusted performance',
        'Ulcer Index measures both magnitude and duration of drawdowns; superior to simple max drawdown',
        'Recovery from -45% requires +82% gain; compounding asymmetry makes deep losses catastrophic',
        'Professional investors should report Calmar Ratio, max drawdown, and recovery time alongside Sharpe ratio',
      ],
    },
    {
      id: 'rrm-risk-metrics-portfolio',
      text: `You are evaluating a multi-asset portfolio with the following annual performance data:

Returns by year (%): 12, -8, 15, 22, -5, 18, -12, 25, 8, 11
Benchmark returns (%): 10, -6, 12, 18, -3, 15, -8, 20, 10, 9

Calculate and interpret: (1) arithmetic vs geometric mean returns and explain which is more appropriate for long-term evaluation, (2) tracking error and information ratio, (3) beta and alpha relative to benchmark, (4) maximum drawdown and time to recovery, (5) VaR and CVaR at 95% confidence level, and (6) based on all metrics, provide a comprehensive assessment of this portfolio's risk-return profile and whether it justifies active management fees of 1% annually.`,
      type: 'discussion' as const,
      sampleAnswer: `**Data Summary:**

Portfolio: 12, -8, 15, 22, -5, 18, -12, 25, 8, 11 (%)
Benchmark: 10, -6, 12, 18, -3, 15, -8, 20, 10, 9 (%)

**1. Arithmetic vs Geometric Mean Returns**

**Arithmetic Mean (Simple Average):**

Portfolio: (12 - 8 + 15 + 22 - 5 + 18 - 12 + 25 + 8 + 11) / 10 = 86 / 10 = **8.6%**

Benchmark: (10 - 6 + 12 + 18 - 3 + 15 - 8 + 20 + 10 + 9) / 10 = 77 / 10 = **7.7%**

**Geometric Mean (Compound Annual Growth Rate - CAGR):**

Portfolio:
Final Value = 1.12 × 0.92 × 1.15 × 1.22 × 0.95 × 1.18 × 0.88 × 1.25 × 1.08 × 1.11 = 2.173

CAGR = (2.173)^(1/10) - 1 = 1.0805 - 1 = **8.05%**

Benchmark:
Final Value = 1.10 × 0.94 × 1.12 × 1.18 × 0.97 × 1.15 × 0.92 × 1.20 × 1.10 × 1.09 = 2.041

CAGR = (2.041)^(1/10) - 1 = 1.0742 - 1 = **7.42%**

**Comparison:**

| Metric | Portfolio | Benchmark | Difference |
|--------|-----------|-----------|------------|
| Arithmetic Mean | 8.60% | 7.70% | +0.90% |
| Geometric Mean | 8.05% | 7.42% | +0.63% |
| Volatility Drag | 0.55% | 0.28% | - |

**Why They Differ (Volatility Drag):**

Relationship: Geometric Mean ≈ Arithmetic Mean - (σ²/2)

Portfolio volatility: σₚ = 12.26% → Drag ≈ 0.75%
Benchmark volatility: σᵦ = 9.06% → Drag ≈ 0.41%

Portfolio has higher volatility → larger gap between arithmetic and geometric mean.

**Which Is More Appropriate?**

**Geometric Mean (CAGR) is correct for long-term evaluation** because:

1. **Actual realized returns:** Geometric mean represents the actual annualized return an investor earned
2. **Compounding effect:** Accounts for path dependence and sequence of returns
3. **Terminal wealth:** CAGR × n years = actual portfolio growth
4. **Volatility drag:** Incorporates the cost of volatility on compounding

**Example:**
- $100,000 invested for 10 years
- Arithmetic mean (8.6%) would suggest: $100k × 1.086^10 = $227k
- Geometric mean (8.05%) gives actual: $100k × 1.0805^10 = $217k
- **Arithmetic overstates by $10k!**

**When to use each:**
- **Geometric:** Long-term performance reporting, terminal wealth calculations, investor returns
- **Arithmetic:** Expected future returns (if conditions remain similar), portfolio optimization inputs

**2. Tracking Error and Information Ratio**

**Tracking Error (TE):**

Active returns (Portfolio - Benchmark):
2, -2, 3, 4, -2, 3, -4, 5, -2, 2 (%)

TE = Standard deviation of active returns
   = σ(2, -2, 3, 4, -2, 3, -4, 5, -2, 2)
   
Mean active return: (2-2+3+4-2+3-4+5-2+2) / 10 = 9/10 = 0.9%

Variance = [(2-0.9)² + (-2-0.9)² + ... + (2-0.9)²] / 9
        = [1.21 + 8.41 + 4.41 + 9.61 + 8.41 + 4.41 + 24.01 + 16.81 + 8.41 + 1.21] / 9
        = 86.9 / 9 = 9.66

TE = √9.66 = **3.11%**

**Information Ratio (IR):**

IR = Average Active Return / Tracking Error
   = 0.9% / 3.11%
   = **0.29**

**Interpretation:**

**Tracking Error (3.11%):**
- Low TE (< 5%): Enhanced indexing, closet indexing
- Medium TE (5-10%): Moderate active management
- High TE (> 10%): Aggressive active management

**Our portfolio (3.11%)** = Low to moderate tracking error, suggesting somewhat constrained active management.

**Information Ratio (0.29):**
- IR > 0.5: Excellent active management
- IR 0.25-0.5: Good active management
- IR 0-0.25: Marginal active management
- IR < 0: Destroying value

**Our portfolio (0.29)** = Borderline good active management. Generating positive excess returns, but not exceptional.

**Quality of Active Management:**
- 0.9% annual outperformance is modest
- Given 3.11% tracking error, hitting outperformance requires skill (not luck)
- But magnitude of outperformance isn't spectacular

**3. Beta and Alpha**

**Beta Calculation:**

β = Cov(Rₚ, Rᵦ) / Var(Rᵦ)

Covariance calculation:
Cov = Σ[(Rₚᵢ - μₚ)(Rᵦᵢ - μᵦ)] / (n-1)

Using arithmetic means: μₚ = 8.6%, μᵦ = 7.7%

Cov = [(12-8.6)(10-7.7) + (-8-8.6)(-6-7.7) + ... + (11-8.6)(9-7.7)] / 9
    = [7.82 + 227.28 + 14.69 + 135.98 + 14.69 + 12.03 + 75.41 + 214.54 + -0.37 + 5.52] / 9
    = 707.59 / 9
    = 78.62

Variance of benchmark:
Var(Rᵦ) = Σ(Rᵦᵢ - μᵦ)² / (n-1)
        = [(10-7.7)² + (-6-7.7)² + ... + (9-7.7)²] / 9
        = [5.29 + 187.69 + 18.49 + 106.09 + 114.49 + 53.29 + 247.69 + 150.89 + 5.29 + 1.69] / 9
        = 890.9 / 9
        = 99.0

**Beta:**
β = 78.62 / 99.0 = **0.79**

**Interpretation:** Portfolio is 21% less volatile than benchmark (defensive positioning).

**Alpha (Jensen's Alpha):**

α = μₚ - [Rf + β(μᵦ - Rf)]

Assuming Rf = 3%:

α = 8.6% - [3% + 0.79(7.7% - 3%)]
  = 8.6% - [3% + 0.79(4.7%)]
  = 8.6% - [3% + 3.71%]
  = 8.6% - 6.71%
  = **1.89%** annual alpha

**Interpretation:**

**Beta (0.79):**
- Less volatile than benchmark
- Defensive portfolio (lower market sensitivity)
- During bull markets: likely underperforms (captures only 79% of upside)
- During bear markets: likely outperforms (suffers only 79% of downside)

**Alpha (1.89%):**
- Strong positive alpha!
- Outperforms risk-adjusted expected return by 1.89% annually
- If statistically significant, indicates genuine manager skill
- **Key question:** Is this sustainable or lucky?

**Statistical Significance (t-test):**

t-statistic = α / (σ_α / √n)
where σ_α = tracking error = 3.11%

t = 1.89% / (3.11% / √10) = 1.89 / 0.98 = 1.93

With 9 degrees of freedom, critical t-value at 95% confidence = 2.26

**t-statistic (1.93) < critical value (2.26)**

**Alpha is NOT statistically significant at 95% level**, but close! Suggests promising but not definitive skill.

**4. Maximum Drawdown and Recovery Time**

**Calculating Cumulative Returns:**

Year 0: $100
Year 1: $112 (×1.12)
Year 2: $103 (×0.92) ← Drawdown
Year 3: $118 (×1.15) ← New high
Year 4: $144 (×1.22)
Year 5: $137 (×0.95) ← Drawdown begins
Year 6: $162 (×1.18) ← New high
Year 7: $142 (×0.88) ← Drawdown begins
Year 8: $178 (×1.25) ← New high
Year 9: $192 (×1.08)
Year 10: $213 (×1.11)

**Drawdown Analysis:**

**Drawdown 1:** Year 2
- Peak: $112 (Year 1)
- Trough: $103 (Year 2)
- Drawdown: (103-112)/112 = **-8.0%**
- Recovery: Year 3 (new high at $118)
- Recovery time: **1 year**

**Drawdown 2:** Year 5
- Peak: $144 (Year 4)  
- Trough: $137 (Year 5)
- Drawdown: (137-144)/144 = **-4.9%**
- Recovery: Year 6 (new high at $162)
- Recovery time: **1 year**

**Drawdown 3:** Year 7-8 (MAXIMUM)
- Peak: $162 (Year 6)
- Trough: $142 (Year 7)
- Drawdown: (142-162)/162 = **-12.3%**
- Recovery: Year 8 (new high at $178)
- Recovery time: **1 year**

**Maximum Drawdown = -12.3%**

**Benchmark Max Drawdown:**
Similar calculation: **-14.4%** (slightly worse)

**Interpretation:**

**Max Drawdown (-12.3%):**
- Excellent risk management!
- For equity portfolio, <15% max DD over 10 years is outstanding
- Shows effective downside protection during difficult years (Year 2: -8%, Year 7: -12%)
- Better than benchmark (-14.4%), confirming defensive beta

**Recovery Time (1 year):**
- Very fast recovery
- All three drawdowns recovered within 1 year
- Shows portfolio resilience and momentum post-decline
- No extended underwater periods (important for investor psychology)

**Comparison to Benchmarks:**
- Typical equity fund: -30 to -50% max DD over 10 years
- S&P 500 (2000-2010): -56% max DD
- Our portfolio: -12.3% (exceptional!)

**5. Value at Risk (VaR) and Conditional VaR (CVaR)**

**95% VaR (Parametric Method):**

Assuming normal distribution (approximate):

VaR₉₅% = μ - 1.65σ

σ = 12.26% (portfolio standard deviation)
μ = 8.6% (arithmetic mean)

VaR₉₅% = 8.6% - 1.65(12.26%)
        = 8.6% - 20.23%
        = **-11.63%**

**Interpretation:** With 95% confidence, annual losses will not exceed 11.63%.

**95% VaR (Historical Method - more accurate for small samples):**

Sort returns: -12, -8, -5, 8, 11, 12, 15, 18, 22, 25

5th percentile (95% VaR) = 5% × 10 = 0.5th observation

Interpolate between -12% and -8%: **VaR₉₅% = -12%**

**Conditional VaR (Expected Shortfall - CVaR):**

CVaR = Expected loss given that VaR threshold is exceeded

Average of worst 5% scenarios:
CVaR₉₅% = average of worst return = **-12%** (only one observation in worst 5%)

For more data, would average: (-12% + -8%) / 2 = **-10%**

**Interpretation:**

**VaR (95%) = -12%:**
- Excellent! Low downside risk
- 95% of the time, annual losses < 12%
- Only 1 in 20 years expect to lose more than 12%

**CVaR (95%) = -10 to -12%:**
- When things go bad, expect ~-10% loss
- CVaR is close to VaR, suggesting loss distribution is not extremely fat-tailed
- No extreme tail risk

**Comparison:**
- Typical equity portfolio: VaR₉₅% ≈ -20 to -25%
- Our portfolio: -12% (much better!)
- Confirms defensive positioning and strong risk management

**6. Comprehensive Assessment**

**Summary of All Metrics:**

| Metric | Portfolio | Benchmark | Assessment |
|--------|-----------|-----------|------------|
| CAGR | 8.05% | 7.42% | +0.63% outperformance |
| Volatility | 12.26% | 9.06% | Higher (cost of active mgmt) |
| Sharpe Ratio | 0.41 | 0.49 | Slightly worse |
| Tracking Error | 3.11% | - | Moderate active management |
| Information Ratio | 0.29 | - | Borderline good |
| Beta | 0.79 | 1.00 | Defensive positioning |
| Alpha | 1.89% | 0% | Strong (but not stat significant) |
| Max Drawdown | -12.3% | -14.4% | Excellent |
| Recovery Time | 1 year | - | Very fast |
| VaR (95%) | -12% | - | Low downside risk |

**Strengths:**1. **Consistent outperformance:** Beat benchmark 9 out of 10 years on risk-adjusted basis
2. **Excellent risk management:** Max DD only -12.3%, recovered quickly
3. **Positive alpha:**1.89% annual alpha suggests skill (though not yet statistically significant over 10 years)
4. **Defensive positioning:** Beta 0.79 provides downside protection
5. **Low tail risk:** VaR and CVaR indicate controlled extreme risk

**Weaknesses:**1. **Higher volatility:**12.26% vs 9.06% benchmark (35% higher)
2. **Lower Sharpe ratio:**0.41 vs 0.49 (due to higher volatility)
3. **Modest outperformance:**0.63% annualized may not justify fees
4. **Information ratio:**0.29 is borderline, not exceptional

**Does It Justify 1% Annual Fee?**

**Analysis:**

Net return after 1% fee: 8.05% - 1.00% = **7.05%**
Benchmark: **7.42%**

**Post-fee, portfolio UNDERPERFORMS benchmark by -0.37%!**

**Verdict:**

**NO, does not justify 1% fee** based purely on returns.

**However, consider:**1. **Risk-adjusted basis:** 
   - Portfolio has -12.3% max DD vs -14.4% benchmark
   - VaR better, recovery times better
   - For risk-averse investors, the improved risk profile might justify 0.5-0.7% fee

2. **Early track record:**
   - 10 years is borderline for statistical significance
   - Alpha t-stat of 1.93 (vs 2.26 needed) suggests promising skill
   - Longer track record might show statistically significant alpha

3. **Fee negotiation:**
   - At 0.5% fee: Net = 7.55% > 7.42% benchmark ✓
   - At 0.75% fee: Net = 7.30% ≈ 7.42% benchmark (break-even)
   - **Maximum justifiable fee: 0.63%** (to match benchmark return)

**Recommendation:**

**For aggressive growth investors:** Do NOT pay 1% fee. Higher volatility and modest outperformance don't justify the cost. Use low-cost index fund.

**For risk-averse investors:** Consider paying 0.5-0.6% fee for superior risk management (lower drawdowns, faster recovery, better VaR profile). The defensive positioning and risk controls add value beyond raw returns.

**For the manager:** Need to either:
1. Reduce fees to 0.5-0.7% to be competitive, OR
2. Improve outperformance to 1.5%+ to justify current fees, OR
3. Market to risk-averse clients who value the superior risk profile

**Overall Grade: B+**

Strong risk management and promising alpha, but needs longer track record or lower fees to be truly compelling.`,
      keyPoints: [
        'Geometric mean (CAGR) is correct for long-term performance; arithmetic mean overstates actual returns due to volatility drag',
        'Tracking error measures active risk; information ratio = active return / tracking error (>0.5 is excellent)',
        'Beta measures systematic risk vs benchmark; alpha measures risk-adjusted outperformance',
        'Maximum drawdown and recovery time are critical for assessing investor experience and portfolio resilience',
        'VaR estimates maximum loss at given confidence level; CVaR estimates expected loss beyond VaR threshold',
        'Statistical significance testing essential: alpha t-stat must exceed ~2.0 for 95% confidence over 10 years',
        'Comprehensive evaluation requires multiple metrics: Sharpe, IR, alpha, max DD, VaR - no single metric sufficient',
        'Fee justification requires post-fee returns to exceed benchmark, or superior risk management to justify value-add',
      ],
    },
  ],
};
