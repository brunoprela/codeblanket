export const statisticalArbitrageMC = [
  {
    id: 'ats-4-mc-1',
    question:
      'Two assets have correlation 0.90 but cointegration p-value 0.15. Should you trade this pair?',
    options: [
      'Yes, 0.90 correlation is strong enough',
      'No, p-value > 0.05 means not cointegrated (spread not stationary)',
      'Yes, but only with tight stop losses',
      'Yes, but use Kalman filter instead of OLS',
    ],
    correctAnswer: 1,
    explanation: `**Correct: No, p-value > 0.05 means not cointegrated.**

**Why correlation ≠ cointegration:**

**Correlation (0.90)**: Assets move together
- Both trending up → correlation 0.90
- But spread can diverge permanently

**Cointegration (p > 0.05)**: Spread is NOT stationary
- Spread = price_A - β*price_B
- If p-value > 0.05, spread is not mean-reverting
- Will experience persistent divergences

**Example:**
- Stock A: $100 → $150 (+50%)
- Stock B: $50 → $60 (+20%)
- Correlation: 0.90 (both went up)
- But spread: $50 → $90 (widened by $40)
- Spread did NOT revert → not cointegrated

**What Happens If You Trade Anyway:**
1. Enter long spread at z-score -2 (spread "low")
2. Spread keeps widening (relationship broke)
3. Hit stop loss at z-score -3
4. Large loss

**Rule**: ONLY trade cointegrated pairs (p < 0.05). High correlation without cointegration = disaster.

**Why Other Answers Wrong:**
- "Tight stop losses": Won't help if spread diverges permanently
- "Kalman filter": Can't fix broken cointegration, will adapt to wrong relationship`,
  },
  {
    id: 'ats-4-mc-2',
    question:
      'Your pairs strategy has half-life 80 days. What does this mean and should you trade it?',
    options: [
      'Spread reverts to mean in 80 days on average; too slow, do not trade',
      'Strategy holds positions for 80 days; acceptable',
      'Spread volatility doubles every 80 days; too risky',
      'Half-life measures correlation; 80 days is good',
    ],
    correctAnswer: 0,
    explanation: `**Correct: Spread reverts to mean in 80 days on average; too slow, do not trade.**

**What Is Half-Life?**

Half-life = time for spread to revert halfway to mean.

**Example:**
- Spread currently at +$4 (mean = $0)
- Half-life 80 days
- After 80 days, expect spread = +$2
- After 160 days, expect spread = +$1
- After 240 days, expect spread = +$0.50

**Why 80 Days Is Too Slow:**

**Rule of Thumb:** Half-life should be < 60 days, ideally < 30 days.

**Problems with 80-Day Half-Life:**
1. **Long Holding Periods**: Will hold 80+ days per trade
2. **Capital Efficiency**: Capital locked up for months
3. **Regime Risk**: Market can change in 80 days
4. **Opportunity Cost**: Better opportunities elsewhere

**Calculation:**
\`\`\`python
# Calculate half-life from Ornstein-Uhlenbeck process
lag = spread.shift(1)
delta = spread.diff()

# Regression: Δspread = λ*spread + ε
model = LinearRegression()
model.fit(lag.values.reshape(-1, 1), delta.values)

lambda_param = model.coef_[0]
half_life = -np.log(2) / lambda_param

if half_life > 60:
    print("Too slow, do not trade")
\`\`\`

**Good Half-Life Examples:**
- 15 days: Excellent (reverts quickly)
- 30 days: Good
- 45 days: Acceptable
- 60 days: Maximum
- 80 days: Too slow ❌

**Why Other Answers Wrong:**
- "Holds 80 days": Not acceptable, too long
- "Volatility doubles": Half-life measures mean reversion speed, not volatility
- "Measures correlation": Half-life measures mean reversion, not correlation

**Bottom Line**: Reject pairs with half-life > 60 days. They tie up capital too long and have regime risk.`,
  },
  {
    id: 'ats-4-mc-3',
    question: 'Kalman filter hedge ratio vs OLS: which statement is correct?',
    options: [
      'OLS always performs better because it uses more data',
      'Kalman filter adapts to time-varying relationships but adds complexity',
      'Kalman filter only works for high-frequency trading',
      'OLS and Kalman produce identical results in practice',
    ],
    correctAnswer: 1,
    explanation: `**Correct: Kalman filter adapts to time-varying relationships but adds complexity.**

**Key Differences:**

**OLS (Static Hedge Ratio):**
\`\`\`python
# Calculate once on full history
model = LinearRegression()
model.fit(price_B, price_A)
hedge_ratio = model.coef_[0]  # Single value, used forever
\`\`\`

**Kalman Filter (Dynamic Hedge Ratio):**
\`\`\`python
# Updates with each new data point
kf = KalmanFilter(...)
hedge_ratios = kf.filter(price_A)  # Time series of hedge ratios
\`\`\`

**When Kalman Wins:**

**Scenario: Relationship Evolves**
- 2020: AAPL = 2.0 × MSFT
- 2021: AAPL = 2.2 × MSFT (AAPL grew faster)
- 2022: AAPL = 2.4 × MSFT (continued divergence)

**OLS**: Uses average β = 2.2 for entire period (wrong at all times)
**Kalman**: Adapts: 2.0 → 2.2 → 2.4 (correct at each time)

**Performance Comparison:**

| Metric | OLS | Kalman | Improvement |
|--------|-----|--------|-------------|
| Sharpe Ratio | 1.5 | 2.0 | +0.5 |
| Win Rate | 58% | 62% | +4% |
| Max Drawdown | -15% | -10% | +5% |

**Trade-Off:**

**Added Complexity:**
- Code: 50 lines (OLS) vs 500 lines (Kalman)
- Parameters: None (OLS) vs 2 (transition/observation covariance)
- Understanding: Simple (OLS) vs state-space knowledge (Kalman)

**Justification:**
- Performance gain: +0.5 Sharpe
- On $1M: +$40K/year
- Worth it for professional funds
- Maybe not for retail

**When to Use Each:**

**Use OLS:**
- Stable relationships (utilities, staples)
- Short backtests (<1 year)
- Simplicity preferred
- Small capital

**Use Kalman:**
- Evolving relationships (tech, growth)
- Long periods (multi-year)
- Volatile markets (2020+)
- Professional funds

**Why Other Answers Wrong:**
- "OLS always better": False, Kalman adapts to regime changes
- "Only high-frequency": Kalman works at any frequency (daily, weekly)
- "Identical results": Different; Kalman has time-varying hedge ratio

**Bottom Line**: Kalman filter improves performance (+0.5 Sharpe) but adds complexity (10x code). Worth it when relationships evolve, not worth it for stable pairs.`,
  },
  {
    id: 'ats-4-mc-4',
    question:
      'PCA stat arb extracts 5 factors from 20 stocks. Factor 1 explains 60% of variance. What should you do?',
    options: [
      'Increase to 10 factors to capture more variance',
      'Trade the residuals (40% unexplained variance), which is idiosyncratic risk',
      'Only trade Factor 1 since it explains most variance',
      'Reduce to 3 factors to avoid overfitting',
    ],
    correctAnswer: 1,
    explanation: `**Correct: Trade the residuals (40% unexplained variance), which is idiosyncratic risk.**

**PCA Decomposition:**

**Total Return = Factor Returns + Residual Returns**

\`\`\`python
# Example: AAPL daily return
aapl_return = 2.5%

# Decomposition:
factor_returns = 2.0%  # Explained by 5 common factors
residual = 0.5%        # AAPL-specific (unexplained)

# PCA tells us:
# - 60% of variance from Factor 1 (market/tech sector)
# - 20% from Factors 2-5 (other systematic factors)
# - 20% from residuals (stock-specific)
\`\`\`

**What Are Factors?**

**Factor 1 (60% variance)**: Market/sector movement
- All 20 tech stocks move together
- Systematic risk (can't diversify away)
- Examples: Fed announcement, tech earnings

**Factors 2-5 (20% variance)**: Other common drivers
- Growth vs value
- Large cap vs small cap
- Momentum

**Residuals (20% variance)**: Stock-specific
- Company earnings beat/miss
- Product announcements
- Management changes
- This is what we trade!

**Why Trade Residuals?**

**Residual = Mispricing**

If AAPL residual = +0.5% (outperformed factors):
- AAPL went up more than factors predicted
- Likely overvalued temporarily
- **Action**: Short AAPL (expect reversion)

If MSFT residual = -0.5% (underperformed factors):
- MSFT went up less than factors predicted
- Likely undervalued temporarily
- **Action**: Long MSFT (expect reversion)

**Complete Strategy:**

\`\`\`python
class PCAStatArb:
    def trade_residuals(self, returns):
        # 1. Extract factors
        factor_returns = self.pca.transform(returns)
        
        # 2. Reconstruct returns from factors
        reconstructed = self.pca.inverse_transform(factor_returns)
        
        # 3. Calculate residuals
        residuals = returns - reconstructed
        
        # 4. Standardize residuals
        z_scores = (residuals - mean) / std
        
        # 5. Trade extreme residuals
        signals = []
        if z_score < -2.0:
            signals.append('LONG')   # Underperformed, expect reversion
        elif z_score > 2.0:
            signals.append('SHORT')  # Outperformed, expect reversion
        
        # 6. Portfolio is factor-neutral
        # Long 5 stocks (negative residuals)
        # Short 5 stocks (positive residuals)
        # Net exposure to factors = 0
\`\`\`

**Why NOT Trade Factors?**

**Factor 1 (60% variance)** = Market beta
- All stocks move together
- Can't arbitrage (everyone knows this)
- Just directional bet on market

**Trading factors = trend following, not stat arb**

**Why 5 Factors Is Right:**

**Too Few (3 factors):**
- Captures only 70% variance
- Residuals have systematic risk (some factors in residuals)
- Not truly idiosyncratic

**Just Right (5 factors):**
- Captures 80% variance
- Residuals are idiosyncratic
- Clean separation

**Too Many (10 factors):**
- Captures 95% variance
- Residuals only 5% (too small to trade)
- Overfitting (fitting to noise)

**Rule of Thumb**: Use factors that explain 70-85% of variance, trade residuals (15-30%).

**Why Other Answers Wrong:**
- "Increase to 10 factors": Overfitting, residuals too small
- "Trade Factor 1": That's just market beta, not stat arb
- "Reduce to 3 factors": Residuals still have systematic risk

**Bottom Line**: Trade the residuals (idiosyncratic risk), not the factors (systematic risk). 5 factors explaining 80% variance is ideal, leaves 20% residual variance to trade.`,
  },
  {
    id: 'ats-4-mc-5',
    question:
      'Your stat arb portfolio has 20 pairs. Pair correlation matrix shows average 0.60 correlation. What is the risk?',
    options: [
      'No risk; pairs are diversified',
      'High concentration risk; pairs are correlated and will lose together',
      'Low risk; 0.60 correlation is acceptable',
      'Need more pairs to achieve diversification',
    ],
    correctAnswer: 1,
    explanation: `**Correct: High concentration risk; pairs are correlated and will lose together.**

**Problem: Correlated Pairs**

**Scenario:**
- You have 20 pairs (40 stocks)
- Average pair correlation: 0.60
- Think you're diversified (20 positions)
- **Reality**: You have ~5 independent bets

**Why This Matters:**

**Example: Tech Sector Crash**
- You trade 20 pairs, all tech stocks
- Pairs: AAPL/MSFT, GOOGL/META, NVDA/AMD, etc.
- Correlation: 0.60

**Normal times**: Pairs uncorrelated, diversification works
**Crisis (2022 tech crash)**: ALL pairs correlate → all lose simultaneously

**Calculation:**

\`\`\`python
# Effective number of independent bets
avg_correlation = 0.60
n_pairs = 20

# Formula: Effective N = N / (1 + (N-1) × ρ)
effective_n = n_pairs / (1 + (n_pairs - 1) * avg_correlation)
# = 20 / (1 + 19 × 0.60)
# = 20 / 12.4
# ≈ 1.6 independent bets

print(f"Think you have: {n_pairs} pairs")
print(f"Actually have: {effective_n:.1f} independent bets")
\`\`\`

**Result**: 20 pairs with 0.60 correlation = 1.6 independent bets (not 20!)

**Real-World Example:**

**2022 Tech Crash:**
- All growth stocks crashed together
- Pairs traders with tech-only portfolios:
  - Thought: 20 pairs = diversified
  - Reality: 1 bet on "tech growth" = concentrated
  - Result: -30% drawdowns (expected -10%)

**Pairs Simultaneously Failing:**
- AAPL/MSFT: Spread widened (both fell, but MSFT fell more)
- GOOGL/META: Spread widened (META crashed harder)
- NVDA/AMD: Spread widened (AMD fell more)
- All 20 pairs hit stop losses within 2 weeks

**How to Fix:**

**1. Diversify Across Sectors**
\`\`\`python
sectors = {
    'tech': 5 pairs,
    'energy': 5 pairs,
    'financials': 5 pairs,
    'healthcare': 5 pairs
}

# Now sector correlation ≈ 0.2
# Effective N = 20 / (1 + 19 × 0.2) = 20 / 4.8 ≈ 4.2 bets
\`\`\`

**2. Calculate Pair Correlation Matrix**
\`\`\`python
def calculate_pair_correlations(pairs, returns):
    """
    Calculate correlation between pair returns
    """
    pair_returns = pd.DataFrame()
    
    for pair in pairs:
        spread_return = returns[pair.asset_1] - pair.hedge_ratio * returns[pair.asset_2]
        pair_returns[pair.name] = spread_return
    
    correlation_matrix = pair_returns.corr()
    avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
    
    return avg_correlation
\`\`\`

**3. Target Correlation < 0.3**

| Avg Pair Correlation | Effective Bets (20 pairs) | Diversification |
|---------------------|--------------------------|-----------------|
| 0.10 | 8.3 | Good |
| 0.20 | 4.8 | Acceptable |
| 0.30 | 3.2 | Mediocre |
| 0.60 | 1.6 | **DANGEROUS** |
| 0.80 | 1.3 | Concentrated |

**4. Stress Test**
\`\`\`python
# Simulate crisis scenario
crisis_correlation = 0.90  # All pairs correlate in crisis

# Your 20 pairs become:
effective_n_crisis = 20 / (1 + 19 × 0.90) = 1.1 bets

# If one pair has -10% max drawdown:
# Portfolio max drawdown ≈ -10% / sqrt(1.1) ≈ -9.5%
# (Almost no diversification!)
\`\`\`

**Why Other Answers Wrong:**
- "No risk": Correlation 0.60 means concentrated
- "0.60 acceptable": Should target <0.3
- "Need more pairs": 100 pairs with 0.60 correlation still concentrated

**Bottom Line**: 20 pairs with 0.60 correlation = 1.6 independent bets (not 20). High concentration risk. Diversify across sectors, target avg correlation <0.3. In crisis, all pairs correlate → simultaneous losses.`,
  },
];
