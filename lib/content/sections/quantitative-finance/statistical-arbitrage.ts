export const statisticalArbitrage = {
  id: 'statistical-arbitrage',
  title: 'Statistical Arbitrage',
  content: `
# Statistical Arbitrage

## Introduction

Statistical arbitrage (stat arb) is a quantitative trading strategy that exploits short-term mean-reverting patterns in financial markets. Unlike pure arbitrage (which is risk-free by definition), stat arb involves statistical risk-profits are probabilistic, not guaranteed.

**Core premise:** Related securities should maintain stable long-run relationships. Temporary deviations from equilibrium create trading opportunities that revert to the mean.

**Key characteristics:**
- **Market-neutral**: Long-short positions eliminate systematic risk (beta ≈ 0)
- **Mean reversion**: Spreads/prices revert to historical averages (half-life: 5-20 days)
- **High turnover**: Positions held days to weeks (vs months/years for fundamental strategies)
- **Scalability limits**: Capacity constraints as AUM grows (returns decay with size)

**Applications:**
- **Pairs trading**: Long-short cointegrated stock pairs
- **Index arbitrage**: ETF vs basket of constituent stocks
- **Cross-asset stat arb**: Bonds vs equities, currencies vs commodities
- **Factor-neutral strategies**: Residual returns after factor hedging

This section covers cointegration testing, pairs trading implementation, mean reversion models, portfolio construction, and risk management for stat arb strategies.

---

## Pairs Trading

### Concept

**Pairs trading** exploits temporary mispricings between two cointegrated securities.

**Example:** Coca-Cola (KO) and PepsiCo (PEP)
- Both beverage companies, similar business models
- Share prices move together over time (correlation ≈ 0.8)
- Spread: \(S_t = P_{KO,t} - \\beta \\cdot P_{PEP,t}\) should be stationary

**Trade logic:**
1. Spread widens beyond 2σ (KO expensive, PEP cheap) → Short KO, long PEP
2. Spread reverts to mean → Close positions, profit
3. Stop loss if spread exceeds 3-4σ (regime change, cointegration broke)

### Cointegration vs Correlation

**Correlation** measures linear relationship (movement together).
- High correlation (0.8+) does NOT guarantee mean reversion
- Example: Two stocks can be correlated but drift apart permanently (non-stationary spread)

**Cointegration** means **spread is stationary** (reverts to mean despite non-stationary prices).
- Formal definition: \(P_A - \\beta P_B = \\epsilon_t\) where \(\\epsilon_t\) is stationary
- \(\\beta\) is the hedge ratio (cointegrating coefficient)

**Why cointegration matters:**
- Correlated but non-cointegrated: Spread drifts → losses accumulate
- Cointegrated: Spread mean-reverts → profitable over time

**Example of correlation without cointegration:**
- GM and TSLA correlation = 0.6 (both auto stocks)
- But GM declines 50% over 5 years, TSLA rises 300%
- Spread drifts continuously (non-stationary) → pairs trade would lose money

**Cointegration testing:**
- **Engle-Granger test**: Tests if \(P_A - \\beta P_B\) is stationary
- **Augmented Dickey-Fuller (ADF)**: Tests for unit root in spread
- **p-value < 0.05**: Reject unit root → spread is stationary (cointegrated)

### Pairs Selection

**Step 1: Identify candidate pairs**

Screen for:
- Same sector (retail, banks, tech)
- Similar market cap
- Similar business models
- High historical correlation (> 0.7)

**Step 2: Test cointegration**

For each pair, run Engle-Granger test:
1. Regress \(P_A\) on \(P_B\): \(P_{A,t} = \\alpha + \\beta P_{B,t} + \\epsilon_t\)
2. Extract residuals: \(\\epsilon_t = P_{A,t} - (\\hat{\\alpha} + \\hat{\\beta} P_{B,t})\)
3. Run ADF test on \(\\epsilon_t\)
4. If p-value < 0.05: Cointegrated ✓

**Step 3: Calculate half-life**

Half-life measures speed of mean reversion (time for spread to decay 50%).

**Ornstein-Uhlenbeck process:**
\[
d\\epsilon_t = -\\theta (\\epsilon_t - \\mu) dt + \\sigma dW_t
\]

Where:
- \(\\theta\): Mean reversion speed
- \(\\mu\): Long-run mean
- Half-life: \(t_{1/2} = \\ln(2) / \\theta\)

**Typical half-lives:**
- **5-10 days**: Fast mean reversion (ideal for pairs trading)
- **20-30 days**: Slower reversion (still tradable)
- **> 50 days**: Too slow (capital tied up too long, risk increases)

**Example:** KO-PEP pair with half-life 8 days.
- Spread deviates +2σ from mean
- Expected: Revert 50% within 8 days, 75% within 16 days, 90% within 24 days
- Position holding period: ~10-15 days on average

### Trading Rules

**Entry signal:**
\[
z_t = \\frac{\\epsilon_t - \\mu}{\\sigma}
\]

- If \(z_t > +2\): Spread too high → Short A, long B
- If \(z_t < -2\): Spread too low → Long A, short B

**Position sizing:**
\[
\\text{Notional}_A = \\$N, \\quad \\text{Notional}_B = \\$N \\cdot \\beta
\]

Where \(\\beta\) is the hedge ratio (from cointegration regression).

**Exit signal:**
- \(|z_t| < 0.5\): Spread reverted → Close positions (take profit)
- \(|z_t| > 4\): Spread diverged further → Stop loss (regime change)
- Holding period exceeds 30 days → Exit (mean reversion failed)

**Example trade:**
- KO = $60, PEP = $180, \(\\beta = 0.3\)
- Spread: \(60 - 0.3 \\times 180 = 60 - 54 = 6\)
- Mean: 4, Std Dev: 1 → \(z = (6-4)/1 = +2\) (entry signal!)
- Trade: Short $10,000 KO, long $10,000 × 0.3 = $3,000 PEP
- Wait 10 days: Spread reverts to 4 → Close, profit ≈ $2,000 (200 bps on $10k)

---

## Mean Reversion Models

### Ornstein-Uhlenbeck Process

**Continuous-time mean reversion model:**
\[
dX_t = \\theta(\\mu - X_t)dt + \\sigma dW_t
\]

**Interpretation:**
- When \(X_t > \\mu\): Drift is negative (pulls toward mean)
- When \(X_t < \\mu\): Drift is positive (pushes toward mean)
- \(\\theta\): Reversion speed (higher = faster reversion)

**Discrete approximation (AR(1) model):**
\[
X_t = \\phi X_{t-1} + (1-\\phi)\\mu + \\epsilon_t
\]

Where \(\\phi = e^{-\\theta \\Delta t}\) (relates to continuous-time \(\\theta\)).

**Estimation:**
1. Fit AR(1): \(X_t = a + b X_{t-1} + \\epsilon_t\)
2. Extract: \(\\phi = b\), \(\\mu = a/(1-b)\), \(\\theta = -\\ln(\\phi)/\\Delta t\)
3. Half-life: \(t_{1/2} = \\ln(2)/\\theta\)

### Z-Score Normalization

**Z-score** measures how many standard deviations spread is from mean:
\[
z_t = \\frac{X_t - \\bar{X}}{\\sigma_X}
\]

**Rolling window:**
- Use 60-day rolling mean and std dev
- Updates daily (adapts to changing mean)
- Too short (20 days): Noisy
- Too long (250 days): Stale (doesn't adapt to regime changes)

**Thresholds:**
- \(|z| > 2\): Enter trade (95% confidence deviation)
- \(|z| > 3\): Strong signal (99.7% confidence)
- \(|z| > 4\): Stop loss (regime likely changed)

---

## Portfolio Construction

### Multi-Pair Portfolio

**Goal:** Diversify across 20-50 pairs to reduce idiosyncratic risk.

**Benefits:**
- Lower volatility (correlation between pairs ≈ 0.2-0.3)
- Smoother equity curve
- Better Sharpe ratio (0.8-1.5 for multi-pair vs 0.5-0.8 for single pair)

**Position sizing:**
\[
w_i = \\frac{1}{n} \\times \\frac{\\sigma_{target}}{\\sigma_i}
\]

Where:
- \(n\): Number of pairs
- \(\\sigma_{target}\): Target portfolio volatility (e.g., 10% annually)
- \(\\sigma_i\): Volatility of pair \(i\)

**Example:** 20 pairs, target vol 10%.
- Pair 1 vol: 15% → weight = (1/20) × (10%/15%) = 0.033 (3.3% of capital)
- Pair 2 vol: 8% → weight = (1/20) × (10%/8%) = 0.063 (6.3% of capital)

### Risk Management

**1. Correlation risk**

Pairs can decorrelate during crises (2008, 2020).

**Example:** JPM-BAC (both banks, normally cointegrated).
- 2008 Lehman bankruptcy: JPM survived, BAC nearly failed
- Spread diverged 6σ (cointegration broke temporarily)
- Pairs traders lost 20-30% on this single pair

**Mitigation:**
- Sector limits (max 30% in financials)
- Correlation monitoring (if avg pair correlation > 0.5, reduce exposure)
- Stop losses (exit if spread > 4σ)

**2. Capacity constraints**

Stat arb strategies have limited capacity.

**Why:**
- Spreads narrow as more capital chases same opportunities
- Market impact (large positions move prices before profit realized)

**Typical capacity:**
- Single pair: $1-10M (depending on liquidity)
- Multi-pair portfolio (50 pairs): $50-500M
- Industry-wide stat arb capacity: ~$50B (beyond this, returns compress)

**Alpha decay:** For every 2× increase in AUM, Sharpe ratio drops ~25%.

**3. Transaction costs**

Round-trip costs: 10-20 bps (commissions + spread).

**Impact on strategy:**
- Annual turnover: 1000-2000% (10-20× per year)
- Transaction costs: 10-20 bps × 1500% = 15-30 bps annually
- Must earn >50 bps per trade to overcome costs

**Breakeven analysis:**
- Gross Sharpe: 1.5 (before costs)
- Transaction costs: 20 bps × 1500% = 30% of return
- Net Sharpe: 1.5 × (1 - 0.3) = 1.05 (still attractive)

---

## Python Implementation

### Cointegration Testing

\`\`\`python
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def test_cointegration(ticker_a, ticker_b, start='2020-01-01', end='2023-12-31'):
    """
    Test if two stocks are cointegrated.
    
    Returns:
    - p_value: Cointegration test p-value
    - hedge_ratio: Beta (hedge ratio)
    - half_life: Mean reversion half-life (days)
    """
    # Download data
    data = yf.download([ticker_a, ticker_b], start=start, end=end, progress=False)['Adj Close']
    data = data.dropna()
    
    # Engle-Granger cointegration test
    score, p_value, _ = coint(data[ticker_a], data[ticker_b])
    
    # Calculate hedge ratio
    model = OLS(data[ticker_a], data[ticker_b]).fit()
    hedge_ratio = model.params[0]
    
    # Calculate spread
    spread = data[ticker_a] - hedge_ratio * data[ticker_b]
    
    # Calculate half-life (Ornstein-Uhlenbeck)
    spread_lag = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()
    
    # Fit AR(1): spread_diff = theta * spread_lag + const
    model_ou = OLS(spread_diff, spread_lag).fit()
    theta = -model_ou.params[0]  # Mean reversion speed
    half_life = np.log(2) / theta if theta > 0 else np.inf
    
    return {
        'p_value': p_value,
        'cointegrated': p_value < 0.05,
        'hedge_ratio': hedge_ratio,
        'half_life': half_life,
        'spread': spread
    }

# Example: Test pairs
print("="*60)
print("PAIRS TRADING: COINTEGRATION ANALYSIS")
print("="*60)

# Test multiple pairs
pairs = [
    ('KO', 'PEP'),
    ('JPM', 'BAC'),
    ('XOM', 'CVX'),
    ('WMT', 'TGT'),
]

results = []
for ticker_a, ticker_b in pairs:
    print(f"\\nTesting {ticker_a} vs {ticker_b}...")
    try:
        result = test_cointegration(ticker_a, ticker_b)
        print(f"  P-value: {result['p_value']:.4f} {'✓ Cointegrated!' if result['cointegrated'] else '✗ Not cointegrated'}")
        print(f"  Hedge Ratio: {result['hedge_ratio']:.4f}")
        print(f"  Half-life: {result['half_life']:.1f} days")
        
        results.append({
            'Pair': f'{ticker_a}-{ticker_b}',
            'P-value': f"{result['p_value']:.4f}",
            'Cointegrated': '✓' if result['cointegrated'] else '✗',
            'Hedge Ratio': f"{result['hedge_ratio']:.4f}",
            'Half-life (days)': f"{result['half_life']:.1f}"
        })
    except Exception as e:
        print(f"  Error: {e}")

# Summary table
print("\\n" + "="*60)
print("SUMMARY")
print("="*60)
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))
\`\`\`

### Pairs Trading Strategy

\`\`\`python
class PairsTradingStrategy:
    """
    Pairs trading strategy with z-score entry/exit.
    """
    
    def __init__(self, ticker_a, ticker_b, hedge_ratio, window=60, 
                 entry_z=2.0, exit_z=0.5, stop_z=4.0):
        """
        Parameters:
        - ticker_a, ticker_b: Stock tickers
        - hedge_ratio: Beta from cointegration
        - window: Rolling window for z-score calculation
        - entry_z: Z-score threshold for entry
        - exit_z: Z-score threshold for exit
        - stop_z: Z-score threshold for stop loss
        """
        self.ticker_a = ticker_a
        self.ticker_b = ticker_b
        self.hedge_ratio = hedge_ratio
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
    
    def calculate_spread(self, prices_a, prices_b):
        """Calculate spread and z-score."""
        spread = prices_a - self.hedge_ratio * prices_b
        
        # Rolling mean and std
        spread_mean = spread.rolling(self.window).mean()
        spread_std = spread.rolling(self.window).std()
        
        # Z-score
        z_score = (spread - spread_mean) / spread_std
        
        return spread, z_score
    
    def generate_signals(self, prices_a, prices_b):
        """Generate trading signals."""
        spread, z_score = self.calculate_spread(prices_a, prices_b)
        
        signals = pd.DataFrame(index=z_score.index)
        signals['spread'] = spread
        signals['z_score'] = z_score
        signals['position'] = 0  # 0: no position, 1: long spread, -1: short spread
        
        # Generate signals
        position = 0
        for i in range(self.window, len(z_score)):
            z = z_score.iloc[i]
            
            if position == 0:  # No position
                if z > self.entry_z:  # Spread too high
                    position = -1  # Short spread (short A, long B)
                elif z < -self.entry_z:  # Spread too low
                    position = 1  # Long spread (long A, short B)
            
            elif position == 1:  # Long spread
                if abs(z) < self.exit_z:  # Mean reversion
                    position = 0  # Close position
                elif z < -self.stop_z:  # Stop loss
                    position = 0
            
            elif position == -1:  # Short spread
                if abs(z) < self.exit_z:  # Mean reversion
                    position = 0
                elif z > self.stop_z:  # Stop loss
                    position = 0
            
            signals['position'].iloc[i] = position
        
        return signals
    
    def backtest(self, start='2020-01-01', end='2023-12-31', capital=100000):
        """Backtest strategy."""
        # Download data
        data = yf.download([self.ticker_a, self.ticker_b], start=start, end=end, progress=False)['Adj Close']
        data = data.dropna()
        
        # Generate signals
        signals = self.generate_signals(data[self.ticker_a], data[self.ticker_b])
        
        # Calculate returns
        # Position 1: Long A, short B×beta
        # Position -1: Short A, long B×beta
        returns_a = data[self.ticker_a].pct_change()
        returns_b = data[self.ticker_b].pct_change()
        
        # Strategy returns (simplified: equal dollar allocation)
        signals['returns'] = signals['position'].shift(1) * (
            returns_a - self.hedge_ratio * returns_b
        )
        
        # Cumulative returns
        signals['cumulative'] = (1 + signals['returns']).cumprod()
        
        # Performance metrics
        total_return = signals['cumulative'].iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(signals)) - 1
        volatility = signals['returns'].std() * np.sqrt(252)
        sharpe = annual_return / volatility if volatility > 0 else 0
        
        max_dd = (signals['cumulative'] / signals['cumulative'].cummax() - 1).min()
        
        return {
            'signals': signals,
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe': sharpe,
            'max_drawdown': max_dd
        }

# Example: Backtest KO-PEP pairs trade
print("\\n" + "="*60)
print("PAIRS TRADING BACKTEST: KO vs PEP")
print("="*60)

# First get hedge ratio
result = test_cointegration('KO', 'PEP', start='2020-01-01', end='2023-12-31')
hedge_ratio = result['hedge_ratio']

# Run strategy
strategy = PairsTradingStrategy('KO', 'PEP', hedge_ratio, 
                                entry_z=2.0, exit_z=0.5, stop_z=4.0)
backtest_results = strategy.backtest(start='2020-01-01', end='2023-12-31')

print(f"\\nTotal Return: {backtest_results['total_return']*100:.2f}%")
print(f"Annual Return: {backtest_results['annual_return']*100:.2f}%")
print(f"Volatility: {backtest_results['volatility']*100:.2f}%")
print(f"Sharpe Ratio: {backtest_results['sharpe']:.2f}")
print(f"Max Drawdown: {backtest_results['max_drawdown']*100:.2f}%")

# Plot
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

signals = backtest_results['signals']

# Z-score and signals
ax1.plot(signals.index, signals['z_score'], linewidth=1, label='Z-score')
ax1.axhline(2, color='red', linestyle='--', alpha=0.5, label='Entry threshold')
ax1.axhline(-2, color='red', linestyle='--', alpha=0.5)
ax1.axhline(0, color='black', linestyle='-', alpha=0.3)
ax1.set_title('Spread Z-Score', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Positions
ax2.plot(signals.index, signals['position'], linewidth=2, color='navy')
ax2.set_title('Position (1=Long Spread, -1=Short Spread, 0=No Position)', fontweight='bold')
ax2.set_ylim(-1.5, 1.5)
ax2.grid(True, alpha=0.3)

# Cumulative returns
ax3.plot(signals.index, signals['cumulative'], linewidth=2, color='green')
ax3.set_title('Cumulative Returns', fontweight='bold')
ax3.set_ylabel('Portfolio Value (Base=1.0)')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pairs_trading_backtest.png', dpi=300, bbox_inches='tight')
plt.show()
\`\`\`

---

## Real-World Applications

### 1. **Index Arbitrage**

**Strategy:** Exploit mispricing between ETF and basket of underlying stocks.

**Example:** SPY (S&P 500 ETF) vs S&P 500 basket.
- SPY trades at $450.00 (NAV = $450.10)
- SPY trades at 0.02% discount → arbitrage!
- **Trade**: Buy SPY at $450.00, short S&P 500 basket at $450.10
- **Profit**: $0.10 per share (≈2.2 bps)

**Scalability:** Large capacity ($1B+) due to liquidity.

### 2. **Merger Arbitrage**

**Strategy:** Long target, short acquirer in announced mergers.

**Example:** Microsoft acquires Activision at $95/share.
- ATVI trades at $92 (3.26% spread)
- Trade: Long ATVI at $92, short MSFT (hedge ratio 0.06)
- If deal closes: ATVI → $95, profit $3/share (3.26%)
- If deal breaks: ATVI crashes to $75, lose $17/share (risk!)

**Risk:** Deal breaks (regulatory, financing) → large losses.

### 3. **Cross-Asset Stat Arb**

**Strategy:** Bonds vs equities spreads.

**Example:** High-yield bonds (HYG) vs equities (SPY).
- Normally correlated 0.7 (both risk assets)
- HYG underperforms SPY by 2σ → arbitrage
- Long HYG, short SPY (beta-hedged)

---

## Key Takeaways

1. **Cointegration (not correlation)** is required for pairs trading-spread must be stationary (mean-reverting)
2. **Half-life** measures reversion speed-ideal range is 5-20 days (too fast: noise, too slow: capital inefficiency)
3. **Z-score thresholds**: Enter at ±2σ (95% confidence), exit at 0.5σ (mean), stop loss at 4σ (regime change)
4. **Multi-pair portfolio** reduces risk-diversify across 20-50 pairs, target Sharpe 1.0-1.5
5. **Capacity limits** constrain stat arb-typical capacity $50-500M per strategy, $50B industry-wide
6. **Transaction costs** matter-10-20 bps round-trip × 1500% turnover = 15-30% of returns
7. **Correlation risk** in crises-pairs decorrelate during 2008/2020, requiring sector limits and stop losses
8. **Regime changes** kill stat arb-monitor cointegration p-values, exit if p > 0.10 (cointegration broke)

Statistical arbitrage requires rigorous quantitative analysis, disciplined risk management, and acceptance of statistical (not guaranteed) profits.
`,
};
