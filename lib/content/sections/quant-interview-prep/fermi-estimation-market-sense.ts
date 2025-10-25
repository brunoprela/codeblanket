export const fermiEstimation = {
  title: 'Fermi Estimation & Market Sense',
  id: 'fermi-estimation-market-sense',
  content: `
# Fermi Estimation & Market Sense

## Introduction

Fermi estimation problems test your ability to make reasonable approximations with limited information—a critical skill for traders who must make quick decisions under uncertainty. Named after physicist Enrico Fermi, these problems require:

**Core Skills:**
- Breaking complex problems into manageable parts
- Making reasonable assumptions
- Sanity-checking estimates
- Communicating your reasoning clearly
- Demonstrating market intuition

**Why Trading Firms Test This:**
1. **Real-world trading** requires constant estimation (order size, market impact, risk)
2. **Quick decision-making** under incomplete information
3. **Structured thinking** - can you decompose problems?
4. **Market sense** - do you understand how markets work?
5. **Communication** - can you explain your reasoning?

**Interview Format:**
- 10-15 minutes per problem
- Interviewer wants to see your thought process, not just the answer
- Think out loud!
- Reasonable approximations are fine (within order of magnitude)

This section covers 30+ Fermi problems specific to trading and finance, with detailed solution frameworks.

---

## Category 1: Market Size Estimation

### Problem 1.1: Daily Trading Volume of S&P 500

**Question:** Estimate the total daily trading volume (in dollars) of all S&P 500 stocks.

**Framework:**

**Step 1: Clarify**
- Do we mean notional value (shares × price)?
- Do we include pre-market and after-hours?
- Let\'s assume regular trading hours, NYSE/NASDAQ

**Step 2: Break Down**

Need to estimate:
1. Number of stocks: 500 (given)
2. Average stock price
3. Average shares outstanding per company
4. Average daily turnover rate

**Step 3: Estimate**

\`\`\`
Average market cap per S&P 500 company:
- Total S&P 500 market cap ≈ $40 trillion
- Per company: $40T / 500 = $80 billion

Average daily turnover:
- Typical turnover ≈ 0.5% - 1% of market cap per day
- Let's use 0.75%

Daily volume per company:
- $80B × 0.0075 = $600 million

Total daily volume:
- $600M × 500 = $300 billion
\`\`\`

**Step 4: Sanity Check**
- NYSE daily volume ≈ $100-200B (reasonable)
- NASDAQ daily volume ≈ $150-250B (reasonable)
- Total ≈ $300B (seems reasonable!)

**Actual Answer:** ~$300-400 billion (our estimate is in the right ballpark!)

**Key Insight:** Breaking down by market cap and turnover rate is more reliable than trying to estimate share counts.

---

### Problem 1.2: Number of Retail Traders in the US

**Question:** How many retail traders (individuals who trade stocks) are there in the US?

**Solution Framework:**

\`\`\`
US population: 330 million

Working-age adults (18-65): ~60% = 200 million

Households with brokerage accounts: ~30% = 60 million households

Active traders (trade at least once per quarter):
- Of account holders, maybe 20% are active
- 60M × 0.20 = 12 million active traders

Very active (trade weekly): maybe 10% of active
- 12M × 0.10 = 1.2 million very active traders
\`\`\`

**Answer:** ~10-15 million retail traders, with ~1-2 million very active

**Follow-up:** "How much capital do they have?"
\`\`\`
Average account size for active traders: ~$50,000
Total retail capital: 12M × $50K = $600 billion
\`\`\`

---

### Problem 1.3: High-Frequency Trading Market Share

**Question:** What percentage of US equity trading volume comes from high-frequency trading (HFT)?

**Solution:**

\`\`\`
Total daily volume: ~$300B (from Problem 1.1)

HFT characteristics:
- Very short holding periods (seconds to minutes)
- High turnover (same capital traded many times)
- Market making, arbitrage, momentum strategies

HFT firms: ~30-50 major players

Typical HFT shares:
- Market making: ~30-40% of volume
- Arbitrage/momentum: ~10-20%
- Total: ~40-60% of volume

Let's estimate: 50% of volume is HFT
- $300B × 0.50 = $150 billion per day
\`\`\`

**Answer:** ~40-60% of US equity volume

**Actual:** Studies show 50-60% (spot on!)

---

## Category 2: Trading Economics

### Problem 2.1: Transaction Costs for a Large Order

**Question:** You need to buy $100 million of a stock trading at $50/share with average daily volume of 5 million shares. Estimate your total transaction costs.

**Solution:**

\`\`\`
Order size: $100M / $50 = 2 million shares

Daily volume: 5 million shares

Participation rate: 2M / 5M = 40% of daily volume
(This is LARGE - will have significant market impact!)

Transaction cost components:

1. Bid-Ask spread:
   - Typical spread for liquid stock: ~0.05% = 5 bps
   - Cost: $100M × 0.0005 = $50,000

2. Market impact (more significant for large orders):
   - Rule of thumb: impact ∝ sqrt (order_size / daily_volume)
   - Impact ≈ 0.5 × volatility × sqrt (participation_rate)
   - If volatility = 2% daily
   - Impact ≈ 0.5 × 0.02 × sqrt(0.40) ≈ 0.63%
   - Cost: $100M × 0.0063 = $630,000

3. Timing risk / slippage:
   - Executing over multiple hours
   - Price might move against you
   - Estimate: ~0.1% = $100,000

Total cost: $50K + $630K + $100K = $780,000
          = 0.78% of order value = 78 bps
\`\`\`

**Answer:** ~$750,000 - $1,000,000 (0.75% - 1% of order)

**Key Insight:** Market impact dominates for large orders!

---

### Problem 2.2: Market Maker Daily P&L

**Question:** Estimate the daily P&L of a market maker in SPY (S&P 500 ETF).

**Solution:**

\`\`\`
SPY statistics:
- Price: ~$450
- Daily volume: ~60 million shares
- Bid-ask spread: ~$0.01 (very tight!)

Market maker assumptions:
- Captures 10% of volume: 6 million shares
- Captures half the spread: $0.005 per share
- But has to hedge, manage inventory risk

Revenue from spread:
- 6M shares × $0.005 = $30,000 per day

Costs:
- Technology/infrastructure: ~$5,000/day
- Risk (inventory fluctuations): ~$10,000/day
- Exchange fees: ~$2,000/day

Net profit: $30K - $17K = $13,000 per day

Annualized (250 trading days): ~$3.25 million per year
\`\`\`

**Answer:** ~$10,000 - $20,000 per day for one market maker

**Follow-up:** "How many market makers?"
- Maybe 5-10 major ones in SPY
- Total market-making profit in SPY: ~$50K-150K/day

---

## Category 3: Company Valuation & Metrics

### Problem 3.1: Starbucks Revenue

**Question:** Estimate Starbucks' annual revenue.

**Solution:**

\`\`\`
Approach 1: Bottom-up from stores

Number of Starbucks stores:
- US: ~15,000 (major presence)
- International: ~20,000
- Total: ~35,000 stores

Revenue per store:
- Average customers per day: 500
- Average ticket: $6
- Daily revenue: 500 × $6 = $3,000
- Annual revenue per store: $3,000 × 365 = $1.1 million

Total revenue: 35,000 × $1.1M = $38.5 billion

Approach 2: Top-down from market

US coffee market: ~$80 billion
Starbucks market share: ~40%
US revenue: $80B × 0.40 = $32 billion
International (40% of total): ~$21 billion
Total: $53 billion

Average of approaches: ~$45 billion
\`\`\`

**Answer:** ~$40-50 billion

**Actual (2023):** $35.9 billion (pretty close!)

---

### Problem 3.2: Netflix Subscriber Count

**Question:** Estimate Netflix\'s global subscriber count.

**Solution:**

\`\`\`
Regional breakdown:

US & Canada:
- Households: 140 million
- Penetration: ~45%
- Subscribers: 63 million

Europe:
- Households: 200 million
- Penetration: ~30%
- Subscribers: 60 million

Latin America:
- Households: 150 million
- Penetration: ~20%
- Subscribers: 30 million

Asia-Pacific:
- Households: 800 million
- Penetration: ~8%
- Subscribers: 64 million

Total: 63M + 60M + 30M + 64M = 217 million
\`\`\`

**Answer:** ~200-250 million subscribers

**Actual:** ~247 million (excellent estimate!)

---

## Category 4: Trading Scenarios

### Problem 4.1: Optimal Position Size

**Question:** You have $10 million to trade. A strategy has 55% win rate, average win of $1000, average loss of $800. What's your optimal position size per trade?

**Solution:**

\`\`\`
Expected value per trade:
- E[win] = 0.55 × $1,000 = $550
- E[loss] = 0.45 × $800 = $360
- E[profit] = $550 - $360 = $190 per trade

Kelly Criterion for optimal sizing:
- f* = (p × b - q) / b
where:
- p = win probability = 0.55
- q = loss probability = 0.45
- b = win/loss ratio = $1,000/$800 = 1.25

f* = (0.55 × 1.25 - 0.45) / 1.25
   = (0.6875 - 0.45) / 1.25
   = 0.19 = 19% of capital

Optimal position size: $10M × 0.19 = $1.9 million per trade

But in practice, use fractional Kelly (e.g., 25-50% of Kelly):
- Conservative: $1.9M × 0.25 = $475,000
- Moderate: $1.9M × 0.50 = $950,000
\`\`\`

**Answer:** ~$500,000 - $1,000,000 per trade

**Key Insight:** Full Kelly is often too aggressive; fractional Kelly provides better risk management.

---

### Problem 4.2: Pairs Trading Opportunity Size

**Question:** You find a pairs trading opportunity between two stocks that are usually 95% correlated but currently diverged 2 standard deviations. How much should you trade?

**Solution:**

\`\`\`
Assumptions:
- Your capital: $100 million
- Typical spread mean reversion: 3-5 days
- Z-score: 2.0 (current divergence)
- Expected profit when mean-reverts: ~2 std devs

Risk assessment:
- Max loss if divergence continues: 1 more std dev = 50% of expected profit
- Probability of profit: ~95% (2-sigma event should revert)

Historical volatility:
- Each stock: 2% daily
- Spread volatility: sqrt(2 × (1-0.95)) × 2% ≈ 0.63% daily

Position size calculation:
- Want to risk max 2% of capital: $2 million
- With 0.63% daily vol, 1 std dev move = 0.63% of position
- For 2% capital at risk over 1 std dev:
- Position size = $2M / 0.0063 = $317 million notional

But with leverage constraints and liquidity:
- Practical size: $20-30 million per leg
- Total exposure: $40-60 million
\`\`\`

**Answer:** ~$40-60 million total notional (20-30% of capital)

---

## Category 5: Market Microstructure

### Problem 5.1: Order Book Depth

**Question:** An order book shows 10,000 shares bid at $50.00 and 8,000 shares offered at $50.01. You want to buy 5,000 shares immediately. What's your execution price?

**Solution:**

\`\`\`
Best ask: $50.01 with 8,000 shares available

Your order: 5,000 shares (market buy)

Execution:
- First 8,000 shares available at $50.01
- You only need 5,000
- All filled at $50.01

Average execution price: $50.01

Cost vs mid-price:
- Mid: ($50.00 + $50.01) / 2 = $50.005
- You paid: $50.01
- Slippage: $0.005 per share = 0.01% = 1 basis point

Total cost: 5,000 × $50.01 = $250,050
Slippage cost: 5,000 × $0.005 = $25
\`\`\`

**Answer:** $50.01 per share, $25 slippage cost

**Follow-up:** "What if you needed 15,000 shares?"
- You'd sweep through $50.01 level (8,000 shares)
- Then need to take from next level (probably $50.02)
- Average price would be higher, more slippage

---

## Category 6: Risk Management

### Problem 6.1: VaR for a Portfolio

**Question:** You have a $50 million portfolio with 2% daily volatility. What\'s your 1-day 95% VaR?

**Solution:**

\`\`\`
VaR = Value × Volatility × Z-score

Where:
- Value = $50 million
- Daily volatility = 2%
- Z-score for 95% confidence = 1.65 (one-tailed)

VaR = $50M × 0.02 × 1.65
    = $1.65 million

Interpretation: 
95% of days, we won't lose more than $1.65 million
(Or: 5% of days, we'll lose more than this)
\`\`\`

**Answer:** $1.65 million 1-day 95% VaR

**Follow-up questions:**
- **"What\'s 99% VaR?"** Z-score = 2.33, VaR = $2.33M
- **"What's 10-day VaR?"** Multiply by √10: $5.22M
- **"What assumptions are you making?"** Normal returns, constant volatility, no autocorrelation

---

## Estimation Tips & Tricks

### Useful Numbers to Remember

**US Statistics:**
- Population: 330 million
- Households: 130 million
- Median income: $75,000
- GDP: $25 trillion

**Market Statistics:**
- S&P 500 market cap: $40 trillion
- US bond market: $50 trillion
- Daily FX volume: $7 trillion
- Daily equity volume: $300-500 billion

**Common Ratios:**
- Price/Earnings: 15-25 (typical)
- Profit margin: 5-15% (varies by industry)
- Revenue growth: 5-15% for mature companies

### Rounding Strategies

**For multiplication:**
- Round to make mental math easier
- 17 × 23 ≈ 20 × 20 = 400 (actual: 391)

**For powers:**
- Use log approximations
- 2^10 = 1,024 ≈ 1,000
- e ≈ 2.7, ln(2) ≈ 0.7

### Sanity Checks

Always ask:
1. **Is this the right order of magnitude?**
2. **Does this pass common sense?**
3. **Can I verify with a different approach?**
4. **What are the extreme cases?**

---

## Interview Strategy

### Communication Framework

**1. Clarify (30 seconds)**
- Restate the question
- Ask clarifying questions
- Define scope

**2. Structure (1 minute)**
- Outline your approach
- Identify key drivers
- State assumptions

**3. Estimate (5-7 minutes)**
- Break down the problem
- Make calculations
- Show your work

**4. Sanity Check (1 minute)**
- Verify order of magnitude
- Cross-check with known data
- Adjust if needed

**5. Communicate (throughout)**
- Think out loud
- Explain reasoning
- Be confident but not rigid

### What Interviewers Look For

✓ Structured thinking
✓ Reasonable assumptions
✓ Clear communication
✓ Numerical comfort
✓ Market intuition
✓ Ability to self-correct

✗ Memorized answers
✗ No structure
✗ Wrong order of magnitude
✗ Refusing to make assumptions
✗ Getting stuck

---

## Practice Problems

Try these on your own:

**Easy:**
1. Daily revenue of a McDonald's location
2. Number of iPhones sold per year
3. Total length of roads in the US

**Medium:**
4. AUM (assets under management) of Vanguard
5. Daily P&L of a volatility trader
6. Number of algorithmic trading firms

**Hard:**
7. Optimal number of market makers in SPY
8. Expected slippage for a $1B block trade
9. Total clearing fees paid by all US traders annually

**Remember:** It\'s about the process, not the exact answer!

---

## Summary

**Key Skills:**
1. Break down complex problems
2. Make reasonable assumptions
3. Estimate efficiently
4. Sanity check results
5. Communicate clearly

**Common Mistakes:**
- Analysis paralysis
- Unreasonable assumptions
- Poor communication
- No sanity checks
- Wrong order of magnitude

**Practice:** Do 2-3 Fermi problems daily. Time yourself. Think out loud.

Master Fermi estimation, and you'll demonstrate the quick, structured thinking that makes great traders!
`,
};
