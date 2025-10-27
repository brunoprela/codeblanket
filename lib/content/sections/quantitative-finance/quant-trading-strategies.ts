export const quantTradingStrategies = {
  id: 'quant-trading-strategies',
  title: 'Quantitative Trading Strategies',
  content: `
# Quantitative Trading Strategies

## Introduction

Quantitative trading strategies systematically exploit market inefficiencies using mathematical models, statistical analysis, and algorithmic execution. Unlike discretionary trading (which relies on human judgment), quantitative strategies are rules-based, backtested, and executed algorithmically.

**Core philosophy:** Markets exhibit exploitable patterns due to behavioral biases, structural constraints, and information frictions. Quantitative strategies identify these patterns, model them mathematically, and trade them systematically.

**Key characteristics:**
- **Rules-based**: Every decision codified (entry, exit, position sizing, risk management)
- **Backtested**: Historical performance validated before live trading
- **Systematic**: No discretionary overrides (discipline eliminates emotional bias)
- **Scalable**: Automated execution handles thousands of signals simultaneously

**Major strategy families:**1. **Momentum**: Trend continuation (time-series and cross-sectional)
2. **Mean reversion**: Price reversals (short-term and statistical arbitrage)
3. **Trend following**: Directional bets on sustained moves
4. **Volatility arbitrage**: Trade realized vs implied volatility
5. **Carry strategies**: Earn yield from interest rate differentials, contango/backwardation

This section provides comprehensive coverage of each strategy family, including mathematical foundations, Python implementations, backtesting methodologies, and real-world applications.

---

## Momentum Strategies

### Time-Series Momentum

**Definition:** Assets with positive (negative) past returns tend to exhibit positive (negative) future returns-**autocorrelation** in returns.

**Academic foundation:**
- **Jegadeesh and Titman (1993)**: Momentum premium of 1% per month (12% annually) for U.S. equities
- **Moskowitz, Ooi, Asness (2012)**: Time-series momentum works across 58 markets (equities, bonds, commodities, currencies)

**Key insight:** Momentum is NOT "buy high, sell low"-it's "buy winners (positive trend), sell losers (negative trend)" within the same asset.

**Implementation:**

**Lookback period:** 12 months (skip most recent month to avoid short-term reversal).

**Signal:**
\[
r_{t-12:t-1} = \\frac{P_{t-1}}{P_{t-12}} - 1
\]

**Position:**
\[
w_t = \\text{sign}(r_{t-12:t-1}) \\times \\text{vol\\_target} / \\sigma_t
\]

Where:
- \(w_t\): Weight at time \(t\) (+1 for long, -1 for short, scaled by vol)
- \(\\sigma_t\): Realized volatility (e.g., 20-day)
- \(\\text{vol\\_target}\): Target volatility (e.g., 10% annually)

**Why skip the most recent month?**
- Short-term reversal effect (Jegadeesh 1990): Stocks with extreme 1-month returns reverse in next month
- 12-1 momentum (12-month lookback, skip most recent month) earns 16% annually vs 12% for 12-0 momentum

**Example:**
- S&P 500 return (12 months ago to 1 month ago): +15%
- Signal: Positive → **Long S&P 500**
- Volatility scaling: If recent vol = 15%, target = 10% → position = 10%/15% = 0.67× notional

**Rebalancing:** Monthly (end of month).

**Performance:**
- **Sharpe ratio**: 0.6-0.8 (standalone)
- **Drawdown**: -30 to -50% (momentum crashes during reversals)
- **Correlation with equities**: 0.3-0.5 (diversification benefit)

### Cross-Sectional Momentum

**Definition:** Within an investment universe, buy top performers, sell bottom performers (relative ranking).

**Difference from time-series:** Time-series is absolute (long if positive return, short if negative), cross-sectional is relative (long top decile, short bottom decile regardless of absolute returns).

**Construction:**

**Step 1: Rank stocks by past return**
- Lookback: 6 or 12 months (skip most recent month)
- Rank all stocks in universe (e.g., S&P 500, Russell 3000)

**Step 2: Portfolio formation**
- Long top decile (10% best performers)
- Short bottom decile (10% worst performers)
- Equal weight or value weight within deciles

**Step 3: Rebalancing**
- Monthly rebalancing
- Transaction costs: 20-40 bps per rebalance (annual turnover 1200-2400%)

**Performance (Fama-French momentum factor, WML):**
- **Average return**: 6-8% annually (1927-2020)
- **Sharpe ratio**: 0.5-0.7
- **Maximum drawdown**: -73% (2009, momentum crash)

**Why does momentum work?**

**Behavioral explanations:**1. **Underreaction to news**: Investors slowly incorporate information → prices drift in direction of news
2. **Herding**: Institutional investors follow trends (career risk: "nobody gets fired for buying IBM")
3. **Disposition effect**: Investors hold losers too long, sell winners too early → delays price adjustment

**Risk-based explanations:**1. **Time-varying risk**: Momentum is compensation for crash risk (option-like payoff)
2. **Delayed information**: Small caps underreact more than large caps (information diffusion)

### Momentum Crashes

**2009 example:** Momentum strategy lost **-73%** in months following Lehman bankruptcy.

**Mechanism:**1. Pre-crisis: Momentum long high-beta stocks (rallied most), short low-beta stocks (fell least)
2. Crisis hits: High-beta stocks collapse → momentum longs crushed
3. Post-crisis rally: Losers (low-quality, high-beta) rally sharply (+100-200%) → momentum shorts squeezed

**Result:** Long book down 40%, short book down 60% (shorts rallied more than longs) → total loss 73%.

**Mitigation strategies:**

**1. Volatility scaling**

Reduce exposure when volatility spikes (VIX > 30).

\[
w_t = w_{base} \\times \\frac{\\sigma_{target}}{\\sigma_t}
\]

**Example:** If vol doubles (20% → 40%), cut position in half.

**2. Market state filter**

Exit momentum when market breaks 200-day moving average.

**Logic:** Momentum works in trending markets, fails in reversals. 200-day MA break signals potential reversal.

**3. Dynamic leverage**

Reduce leverage during drawdowns (cut position after -10% loss).

**4. Long-only momentum**

Avoid short side (momentum crashes driven by short squeezes).

**Performance:** Reduces max drawdown from -73% to -40%, but also reduces Sharpe (0.7 → 0.5).

---

## Mean Reversion Strategies

### Short-Term Reversal

**Definition:** Assets with extreme short-term returns (1 day to 1 week) reverse in the following period.

**Academic foundation:**
- **Jegadeesh (1990)**: Weekly reversals earn 50-100 bps per week
- **Lehmann (1990)**: Daily reversals (contrarian strategies) profitable for large-cap stocks

**Mechanism:**
- **Microstructure effects**: Bid-ask bounce, inventory effects
- **Liquidity provision**: Market makers profit from spread
- **Overreaction**: Behavioral overreaction to news (reverts within days)

**Implementation:**

**Signal:** Rank stocks by 1-week return.

**Portfolio:**
- Long bottom decile (worst performers, down 10-20% in week)
- Short top decile (best performers, up 10-20% in week)
- Hold 1-5 days

**Position sizing:**
\[
w_i = \\frac{1}{N} \\times \\frac{\\sigma_{target}}{\\sigma_i}
\]

Equal-weighted within deciles, scaled by volatility.

**Transaction costs:** Critical! Round-trip costs 10-20 bps × weekly turnover (5000%) = 500-1000 bps annually.

**Net performance:**
- **Gross return**: 50-100 bps per week (2600-5200% annually-obviously too good to be true)
- **Net return**: 10-20 bps per week after costs (520-1040% annually-still too high, likely data mining)

**Reality check:** Most academic reversal strategies fail out-of-sample due to:
- Transaction costs (underestimated in academic studies)
- Market impact (walking the book on large positions)
- Capacity limits (only profitable on $1-10M AUM)

**Modern implementation:** Intraday mean reversion (HFT-only, need sub-millisecond execution).

### Pairs Trading

**Covered extensively in Statistical Arbitrage section.**

**Key points:**
- Trade cointegrated pairs (spread is stationary)
- Enter at ±2σ, exit at 0.5σ, stop loss at 4σ
- Half-life 5-20 days (optimal reversion speed)
- Capacity $50-500M (multi-pair portfolio)

---

## Trend Following Strategies

### Moving Average Crossovers

**Golden Cross:** 50-day MA crosses above 200-day MA → **Buy signal**.

**Death Cross:** 50-day MA crosses below 200-day MA → **Sell signal**.

**Implementation:**

\`\`\`python
if MA_50 > MA_200 and position == 0:
    buy()
elif MA_50 < MA_200 and position == 1:
    sell()
\`\`\`

**Performance:**
- Works in **trending markets** (2003-2007, 2009-2020 bull runs)
- Fails in **choppy markets** (2008, 2015-2016, 2022-whipsaws)

**Example:** S&P 500, 2008.
- January: Golden cross → Buy at 1400
- March: Death cross → Sell at 1300 (loss -100 points)
- May: Golden cross → Buy at 1400
- June: Death cross → Sell at 1280 (loss -120 points)
- **Total**: 4 whipsaws, -15% loss (vs buy-and-hold -37%)

**Pros:**
- Simple, intuitive
- Captures major trends (avoids 2008 crash)

**Cons:**
- Whipsaws in ranging markets (15-25% of time)
- Lags true turning points (enter late, exit late)

### Donchian Channels (Breakout Strategy)

**Entry:** Buy when price breaks 20-day high, sell when breaks 20-day low.

**Exit:** Exit long when price breaks 10-day low, exit short when breaks 10-day high.

**Logic:** Breakouts signal trend continuation (momentum), faster exit on reversal.

**Example:** Crude oil.
- Price: $70, 20-day high: $75, 10-day low: $68
- Price breaks $75 → **Long** (breakout signal)
- Hold until price breaks $68 (10-day low) → Exit

**Performance:**
- **Sharpe**: 0.4-0.6 (lower than momentum due to whipsaws)
- **Win rate**: 35-45% (low, but winners 2-3× larger than losers)
- **Tail risk**: Profits in crisis (2008, 2020-captures big moves)

**Applications:** Futures (commodities, currencies, indices) where trends persist longer than equities.

### Dual Momentum (Antonacci)

**Combines time-series and cross-sectional momentum.**

**Step 1 (Time-series filter):** Is momentum positive?
- If S&P 500 > T-Bills (12-month return) → risk-on
- If S&P 500 < T-Bills → risk-off (hold T-Bills)

**Step 2 (Cross-sectional):** If risk-on, which asset?
- Compare S&P 500 vs International (EAFE)
- Hold whichever has higher 12-month return

**Performance (1974-2020):**
- **Return**: 17.5% annually (vs 10.5% S&P 500 buy-and-hold)
- **Sharpe**: 1.0 (vs 0.5 buy-and-hold)
- **Max drawdown**: -18% (vs -51% buy-and-hold in 2008)

**Key insight:** Time-series filter avoids crashes (switches to T-Bills), cross-sectional captures best opportunities.

---

## Volatility Strategies

### Volatility Arbitrage

**Concept:** Trade difference between implied volatility (IV) and realized volatility (RV).

**Long volatility:** Buy options when IV < expected RV.
- Example: VIX = 15 (IV), but expect RV = 20 → **Buy straddles**
- Profit if realized vol exceeds implied

**Short volatility:** Sell options when IV > expected RV.
- Example: VIX = 30 (IV), but expect RV = 20 → **Sell straddles**
- Profit if realized vol below implied (collect premium)

**Volatility risk premium:** On average, IV > RV by 2-5% (volatility sellers earn premium).

**Why?**
- **Hedging demand**: Investors pay premium to hedge downside (buy puts)
- **Crash aversion**: Deep out-of-money puts expensive (tail risk hedging)
- **Supply-demand**: More natural buyers (hedgers) than sellers (risk-takers)

**Example:** S&P 500, 1-month ATM implied vol = 18%, realized vol = 15%.
- Sell 1-month straddle, earn premium from 3% vol difference
- **Risk**: Realized vol spikes to 40% (2008, 2020) → massive losses

### VIX Term Structure Trading

**Contango (normal):** VIX futures > VIX spot.
- Example: VIX spot = 15, 1-month future = 17, 2-month future = 19
- **Trade**: Short VIX futures (XIV, SVXY ETFs pre-2018)
- **Profit**: Roll yield as futures converge to spot (earn 2 points per month)

**Backwardation (crisis):** VIX futures < VIX spot.
- Example: VIX spot = 40, 1-month future = 35, 2-month future = 30
- **Trade**: Long VIX futures (VXX ETF)
- **Profit**: Futures rise toward elevated spot

**Historical performance (contango trade):**
- **2009-2017**: +100-200% annually (massive roll yield)
- **Feb 2018 (Volmageddon)**: -95% in single day (VIX spiked 100%, short vol products liquidated)
- **Post-2018**: Strategy capacity severely reduced (product liquidations)

**Lesson:** Selling volatility is "picking up pennies in front of a steamroller"-high Sharpe until tail event wipes out years of gains.

### Dispersion Trading

**Trade:** Sell index volatility, buy constituent volatility.

**Logic:** Index volatility < weighted average of constituent volatilities due to diversification.

**Example:** S&P 500 implied vol = 15%, average single-stock vol = 25%.
- **Trade**: Short S&P 500 straddle, long straddles on 50 stocks (weighted)
- **Profit**: Capture 10% vol spread

**Risk:** Correlation spikes during crisis (diversification breaks down) → losses.

---

## Strategy Implementation

### Backtesting Best Practices

**1. Data quality**

**Survivorship bias:** Dead stocks excluded from databases → overstates returns.
- **Impact**: 2-4% annual return overstatement
- **Solution**: Use survivorship-bias-free databases (CRSP, Compustat)

**Point-in-time data:** Use only information available at time of decision.
- **Lookback bias**: Using future data (e.g., restated financials) → overstates returns
- **Solution**: Use "as-reported" databases

**Corporate actions:** Adjust for splits, dividends, spinoffs.

**2. Transaction costs**

**Components:**
- Commissions: $0.0001-0.0005 per share (retail), $0 (high-volume institutional)
- Bid-ask spread: 5-20 bps (liquid), 50-100 bps (illiquid)
- Market impact: √(participation rate) × volatility (Almgren-Chriss)

**Total round-trip:** 10-40 bps (liquid stocks), 50-200 bps (illiquid stocks).

**Modeling:**
\`\`\`python
slippage = spread/2 + eta * volatility * sqrt (order_size / daily_volume)
\`\`\`

**3. Out-of-sample testing**

**In-sample:** 1990-2010 (develop strategy, optimize parameters).
**Out-of-sample:** 2011-2023 (test on unseen data).

**Walk-forward analysis:**
- Optimize on rolling 5-year window
- Test on next 1-year out-of-sample
- Roll forward, repeat

**Performance degradation:** Expect 20-50% lower Sharpe out-of-sample vs in-sample.

**4. Parameter sensitivity**

**Robust strategy:** Performance stable across parameter ranges.

**Example:** Momentum lookback.
- Test 6, 9, 12, 15 months
- If only 12 months works → overfit (data mining)
- If all work similarly → robust

**5. Regime analysis**

Test across different market regimes:
- Bull markets (2009-2020)
- Bear markets (2000-2002, 2008)
- Sideways markets (2015-2016)

**Momentum example:**
- Bull: Sharpe 1.0
- Bear: Sharpe -0.3 (momentum crashes)
- Sideways: Sharpe 0.2 (whipsaws)

### Risk Management

**1. Position sizing (Kelly Criterion)**

**Optimal leverage:**
\[
f^* = \\frac{\\mu}{\\sigma^2}
\]

Where:
- \(f^*\): Fraction of capital to risk
- \(\\mu\): Expected return
- \(\\sigma\): Volatility

**Example:** Strategy with 12% return, 15% vol.
- \(f^* = 0.12 / 0.15^2 = 0.12 / 0.0225 = 5.33×\) leverage

**Half-Kelly:** Use 50% of Kelly (2.67× leverage) to reduce volatility and drawdown risk.

**2. Volatility targeting**

**Constant risk:** Scale positions to maintain constant volatility.

\[
w_t = w_{base} \\times \\frac{\\sigma_{target}}{\\sigma_t}
\]

**Example:** Target 10% vol.
- If realized vol = 5% → leverage 2× (10%/5%)
- If realized vol = 20% → leverage 0.5× (10%/20%)

**Benefit:** Sharpe ratio improves 20-30% (reduces drawdowns in high-vol regimes).

**3. Stop losses**

**Fixed stop:** Exit position after -5% loss.

**Trailing stop:** Exit if price falls 10% from peak.

**Time stop:** Exit if position held > 30 days without profit.

**Pros:** Limits max loss per position.
**Cons:** Can be stopped out before reversal (reduces win rate).

**4. Diversification**

**Combine uncorrelated strategies:**
- Momentum (Sharpe 0.7, correlation with equities 0.4)
- Mean reversion (Sharpe 0.5, correlation -0.2)
- Trend following (Sharpe 0.5, correlation 0.3)

**Portfolio:** 1/3 each.
- **Portfolio Sharpe**: 1.1 (higher than any individual strategy due to diversification)
- **Correlation**: Near zero (strategies offset each other's losses)

---

## Python Implementation

### Momentum Strategy

\`\`\`python
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

def time_series_momentum (ticker, lookback=252, skip=21, vol_target=0.10, start='2010-01-01', end='2023-12-31'):
    """
    Time-series momentum with volatility targeting.
    
    Parameters:
    - ticker: Stock/ETF ticker
    - lookback: Lookback period (252 = 12 months)
    - skip: Skip period (21 = 1 month)
    - vol_target: Target volatility (0.10 = 10%)
    - start, end: Backtest period
    
    Returns:
    - DataFrame with signals, positions, returns
    """
    # Download data
    data = yf.download (ticker, start=start, end=end, progress=False)
    data = data['Adj Close'].to_frame('price')
    
    # Calculate momentum signal
    data['momentum'] = data['price'].pct_change (lookback).shift (skip)
    
    # Calculate realized volatility (20-day)
    data['returns'] = data['price'].pct_change()
    data['volatility'] = data['returns'].rolling(20).std() * np.sqrt(252)
    
    # Generate position (sign of momentum, scaled by vol target)
    data['position'] = np.sign (data['momentum']) * (vol_target / data['volatility'])
    data['position'] = data['position'].clip(-2, 2)  # Limit leverage to 2×
    
    # Calculate strategy returns
    data['strategy_returns'] = data['position'].shift(1) * data['returns']
    
    # Cumulative returns
    data['cumulative'] = (1 + data['strategy_returns']).cumprod()
    data['buy_hold'] = (1 + data['returns']).cumprod()
    
    return data.dropna()

# Backtest on SPY (S&P 500 ETF)
print("="*60)
print("TIME-SERIES MOMENTUM BACKTEST: SPY")
print("="*60)

results = time_series_momentum('SPY', lookback=252, skip=21, vol_target=0.10)

# Performance metrics
total_return = results['cumulative'].iloc[-1] - 1
annual_return = (1 + total_return) ** (252 / len (results)) - 1
volatility = results['strategy_returns'].std() * np.sqrt(252)
sharpe = annual_return / volatility

max_dd = (results['cumulative'] / results['cumulative'].cummax() - 1).min()

buy_hold_return = results['buy_hold'].iloc[-1] - 1
buy_hold_annual = (1 + buy_hold_return) ** (252 / len (results)) - 1

print(f"\\nStrategy Performance:")
print(f"  Total Return: {total_return*100:.2f}%")
print(f"  Annual Return: {annual_return*100:.2f}%")
print(f"  Volatility: {volatility*100:.2f}%")
print(f"  Sharpe Ratio: {sharpe:.2f}")
print(f"  Max Drawdown: {max_dd*100:.2f}%")

print(f"\\nBuy & Hold (SPY):")
print(f"  Annual Return: {buy_hold_annual*100:.2f}%")

# Plot
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

# Cumulative returns
ax1.plot (results.index, results['cumulative'], label='Momentum Strategy', linewidth=2)
ax1.plot (results.index, results['buy_hold'], label='Buy & Hold', linewidth=2, alpha=0.7)
ax1.set_title('Cumulative Returns: Momentum vs Buy & Hold', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Position over time
ax2.plot (results.index, results['position'], linewidth=1, color='navy')
ax2.axhline(0, color='red', linestyle='--', alpha=0.5)
ax2.set_title('Position (Volatility-Scaled)', fontweight='bold')
ax2.set_ylabel('Leverage')
ax2.grid(True, alpha=0.3)

# Rolling Sharpe (252-day)
results['rolling_sharpe'] = (results['strategy_returns'].rolling(252).mean() / 
                              results['strategy_returns'].rolling(252).std() * np.sqrt(252))
ax3.plot (results.index, results['rolling_sharpe'], linewidth=2, color='green')
ax3.axhline(0, color='red', linestyle='--', alpha=0.5)
ax3.set_title('Rolling 1-Year Sharpe Ratio', fontweight='bold')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('momentum_backtest.png', dpi=300, bbox_inches='tight')
plt.show()
\`\`\`

### Multi-Strategy Portfolio

\`\`\`python
def multi_strategy_portfolio (tickers=['SPY', 'TLT', 'GLD'], weights=[0.33, 0.33, 0.34]):
    """
    Combine momentum strategies on multiple assets.
    """
    all_results = {}
    
    for ticker in tickers:
        results = time_series_momentum (ticker, lookback=252, skip=21, vol_target=0.10)
        all_results[ticker] = results['strategy_returns']
    
    # Combine strategies
    portfolio_df = pd.DataFrame (all_results).fillna(0)
    portfolio_df['portfolio'] = sum (portfolio_df[ticker] * weight 
                                    for ticker, weight in zip (tickers, weights))
    
    # Metrics
    portfolio_df['cumulative'] = (1 + portfolio_df['portfolio']).cumprod()
    
    total_return = portfolio_df['cumulative'].iloc[-1] - 1
    annual_return = (1 + total_return) ** (252 / len (portfolio_df)) - 1
    volatility = portfolio_df['portfolio'].std() * np.sqrt(252)
    sharpe = annual_return / volatility
    max_dd = (portfolio_df['cumulative'] / portfolio_df['cumulative'].cummax() - 1).min()
    
    print("\\n" + "="*60)
    print("MULTI-ASSET MOMENTUM PORTFOLIO")
    print("="*60)
    print(f"Assets: {', '.join (tickers)}")
    print(f"Weights: {weights}")
    print(f"\\nPortfolio Performance:")
    print(f"  Annual Return: {annual_return*100:.2f}%")
    print(f"  Volatility: {volatility*100:.2f}%")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  Max Drawdown: {max_dd*100:.2f}%")
    
    return portfolio_df

# Run multi-strategy portfolio
portfolio = multi_strategy_portfolio(['SPY', 'TLT', 'GLD'], [0.5, 0.3, 0.2])
\`\`\`

---

## Real-World Applications

### 1. **AQR Momentum Funds**

**Strategy:** Diversified momentum across equities, bonds, commodities, currencies.

**Performance (AQR Managed Futures Fund):**
- **1990-2020**: 8% annual return, Sharpe 0.7
- **2008**: +20% (crisis alpha-profits when equities crashed)
- **AUM**: $150B+ (world's largest quant hedge fund)

**Key features:**
- Volatility scaling (reduces leverage in high-vol regimes)
- Multi-asset diversification (58 markets)
- Transaction cost optimization (smart execution)

### 2. **Winton Capital (Trend Following)**

**Strategy:** Systematic trend following on futures (commodities, currencies, rates, equities).

**Implementation:**
- 100+ futures markets
- Multiple timeframes (short, medium, long-term trends)
- Dynamic position sizing (volatility-adjusted)

**Performance:**
- **1997-2020**: 10% annual return, Sharpe 0.6
- **2008**: +8% (captured downtrend in equities)
- **AUM**: $25B

### 3. **Renaissance Technologies (Mean Reversion)**

**Strategy:** Short-term mean reversion + statistical arbitrage (proprietary models).

**Medallion Fund performance:**
- **1988-2020**: 66% annual return (before fees), 39% after fees
- **Sharpe**: 3.5 (best in history)
- **Max drawdown**: -10% (vs -50% for equities)

**Secret sauce:**
- Ultra-short holding periods (minutes to hours)
- Massive data (tick-by-tick, order flow, alternative data)
- Transaction cost optimization (co-location, smart routing)

**Limitation:** Capacity ~$10B (closed to external capital since 2005).

---

## Key Takeaways

1. **Momentum works across 58 markets** (Moskowitz 2012) with 6-8% annual premium-robust across asset classes
2. **Time-series momentum** (12-1 month lookback) earns 12-16% annually, Sharpe 0.6-0.8
3. **Cross-sectional momentum** (long winners, short losers) earns 6-8% annually but suffers -73% crashes
4. **Volatility scaling** improves Sharpe by 20-30% and reduces drawdowns in crisis periods
5. **Mean reversion works short-term** (1-5 days) but requires ultra-low transaction costs (<5 bps)
6. **Trend following** captures tail events (2008: +20%) but suffers whipsaws in ranging markets
7. **Selling volatility** earns 15-25% annually (roll yield) but risks -95% losses in single day (Volmageddon)
8. **Backtesting best practices**: Survivorship-bias-free data, realistic transaction costs, out-of-sample testing
9. **Diversification is king**: Combining momentum + mean reversion + trend following yields Sharpe >1.0
10. **Capacity limits**: Strategies decay with AUM-momentum ~$100B, mean reversion ~$10B, HFT ~$1B

Quantitative strategies systematically harvest risk premia, but require rigorous research, careful implementation, and disciplined risk management to succeed in live trading.
`,
};
