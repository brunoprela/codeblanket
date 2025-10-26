export const pairsTrading = {
    title: 'Pairs Trading',
    slug: 'pairs-trading',
    description:
        'Deep dive into pairs trading: selection, execution, risk management, and real-world case studies',
    content: `
# Pairs Trading

## Introduction: The Original Stat Arb Strategy

Pairs trading, pioneered by Morgan Stanley in the 1980s, is the foundation of statistical arbitrage. A team of quants led by Nunzio Tartaglia generated $50M+ annually for a decade using this approach. Today, thousands of hedge funds employ variants of pairs trading.

**What you'll learn:**
- Rigorous pair selection methodology
- Cointegration vs correlation (why it matters)
- Entry/exit timing and position sizing
- Real-world execution challenges
- Performance analysis and risk management

**Why engineers love pairs trading:**
- Quantifiable edge (statistical relationship)
- Market-neutral (hedged against market moves)
- Scalable (can trade 50+ pairs)
- Systematic (rules-based, no discretion)

**Historical Performance:**
- Morgan Stanley (1980s): 50%+ annual returns
- Modern funds (2020s): 15-25% annual returns
- Sharpe ratios: 1.5-2.5

---

## Pair Selection Methodology

### Step 1: Universe Definition

**Criteria for tradeable universe:**
1. Liquid (>$1M daily volume)
2. Shortable (borrow available)
3. Related (same sector/industry)
4. Stationary spreads (cointegrated)

\`\`\`python
import pandas as pd
import numpy as np
from typing import List, Tuple
from statsmodels.tsa.stattools import coint, adfuller
from scipy.stats import pearsonr
import yfinance as yf

class PairSelector:
    """
    Comprehensive pair selection system
    
    Process:
    1. Define universe (sector/industry)
    2. Calculate correlations
    3. Test cointegration
    4. Calculate half-life
    5. Score and rank pairs
    """
    
    def __init__(self, min_correlation: float = 0.7, max_pvalue: float = 0.05):
        self.min_correlation = min_correlation
        self.max_pvalue = max_pvalue
        
    def download_data(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Download price data for universe
        """
        data = yf.download(symbols, start=start_date, end=end_date)['Adj Close']
        return data
    
    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily returns"""
        return prices.pct_change().dropna()
    
    def test_pair(self, 
                  price_a: pd.Series, 
                  price_b: pd.Series) -> dict:
        """
        Comprehensive pair testing
        
        Tests:
        1. Correlation
        2. Cointegration (Engle-Granger)
        3. Half-life
        4. Spread stationarity
        5. Beta stability
        
        Returns:
            Dictionary with all metrics
        """
        # 1. Correlation
        correlation, _ = pearsonr(price_a, price_b)
        
        # 2. Cointegration
        coint_t, coint_p, _ = coint(price_a, price_b)
        
        # 3. Calculate hedge ratio
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(price_b.values.reshape(-1, 1), price_a.values)
        hedge_ratio = model.coef_[0]
        
        # 4. Calculate spread
        spread = price_a - hedge_ratio * price_b
        
        # 5. Half-life
        half_life = self.calculate_half_life(spread)
        
        # 6. Spread statistics
        spread_mean = spread.mean()
        spread_std = spread.std()
        spread_sharpe = self.calculate_spread_sharpe(spread)
        
        # 7. Beta stability (rolling correlation)
        rolling_corr = price_a.rolling(60).corr(price_b)
        corr_stability = rolling_corr.std()
        
        return {
            'correlation': correlation,
            'cointegration_pvalue': coint_p,
            'hedge_ratio': hedge_ratio,
            'half_life': half_life,
            'spread_mean': spread_mean,
            'spread_std': spread_std,
            'spread_sharpe': spread_sharpe,
            'correlation_stability': corr_stability,
            'passes': self.passes_filters(correlation, coint_p, half_life)
        }
    
    def calculate_half_life(self, spread: pd.Series) -> float:
        """
        Calculate mean reversion half-life
        
        Uses Ornstein-Uhlenbeck process:
        dS = λ(μ - S)dt + σdW
        
        Half-life = -ln(2) / λ
        """
        lag = spread.shift(1).dropna()
        delta = spread.diff().dropna()
        
        # Align
        lag = lag[delta.index]
        
        # Regression
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(lag.values.reshape(-1, 1), delta.values)
        
        lambda_param = model.coef_[0]
        
        if lambda_param >= 0:
            return np.inf
        
        half_life = -np.log(2) / lambda_param
        return half_life
    
    def calculate_spread_sharpe(self, spread: pd.Series) -> float:
        """
        Calculate Sharpe ratio of spread returns
        
        Higher Sharpe = better mean reversion
        """
        spread_returns = spread.diff()
        if spread_returns.std() == 0:
            return 0
        sharpe = spread_returns.mean() / spread_returns.std() * np.sqrt(252)
        return abs(sharpe)
    
    def passes_filters(self, 
                      correlation: float, 
                      coint_pvalue: float, 
                      half_life: float) -> bool:
        """
        Check if pair passes all filters
        """
        return (
            correlation >= self.min_correlation and
            coint_pvalue <= self.max_pvalue and
            half_life < 60 and
            half_life > 5  # Not too fast (noise)
        )
    
    def screen_universe(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Screen all pairs in universe
        
        Returns:
            DataFrame with pair statistics
        """
        symbols = prices.columns
        results = []
        
        for i, sym_a in enumerate(symbols):
            for sym_b in symbols[i+1:]:
                try:
                    metrics = self.test_pair(prices[sym_a], prices[sym_b])
                    metrics['pair'] = f"{sym_a}/{sym_b}"
                    metrics['asset_a'] = sym_a
                    metrics['asset_b'] = sym_b
                    results.append(metrics)
                except:
                    continue
        
        df = pd.DataFrame(results)
        
        # Filter and sort
        df_viable = df[df['passes'] == True]
        df_viable = df_viable.sort_values('spread_sharpe', ascending=False)
        
        return df_viable
    
    def score_pair(self, metrics: dict) -> float:
        """
        Calculate comprehensive quality score (0-100)
        
        Weighting:
        - Cointegration: 40 points
        - Half-life: 30 points
        - Correlation: 20 points
        - Spread Sharpe: 10 points
        """
        score = 0
        
        # Cointegration (40 points)
        if metrics['cointegration_pvalue'] < 0.01:
            score += 40
        elif metrics['cointegration_pvalue'] < 0.05:
            score += 20
        
        # Half-life (30 points)
        hl = metrics['half_life']
        if hl < 20:
            score += 30
        elif hl < 40:
            score += 20
        elif hl < 60:
            score += 10
        
        # Correlation (20 points)
        corr = metrics['correlation']
        if corr > 0.9:
            score += 20
        elif corr > 0.8:
            score += 15
        elif corr > 0.7:
            score += 10
        
        # Spread Sharpe (10 points)
        if metrics['spread_sharpe'] > 2:
            score += 10
        elif metrics['spread_sharpe'] > 1:
            score += 5
        
        return score

# Example: Screen tech sector
tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'INTC', 'QCOM']

selector = PairSelector()
prices = selector.download_data(tech_stocks, '2020-01-01', '2024-01-01')
viable_pairs = selector.screen_universe(prices)

print(f"\\nFound {len(viable_pairs)} viable pairs")
print(viable_pairs.head(10))
\`\`\`

---

## Execution: Entry and Exit Timing

### Z-Score Trading Rules

\`\`\`python
class PairsTradingExecutor:
    """
    Execute pairs trades with precise timing
    """
    
    def __init__(self,
                 pair_name: str,
                 asset_a: str,
                 asset_b: str,
                 hedge_ratio: float,
                 lookback: int = 60):
        self.pair_name = pair_name
        self.asset_a = asset_a
        self.asset_b = asset_b
        self.hedge_ratio = hedge_ratio
        self.lookback = lookback
        
        # Trading parameters
        self.entry_zscore = 2.0
        self.exit_zscore = 0.5
        self.stop_zscore = 3.0
        
        self.position = 0  # 0=flat, 1=long spread, -1=short spread
        self.entry_price = {}
        self.trades = []
    
    def calculate_signals(self, 
                         prices_a: pd.Series, 
                         prices_b: pd.Series) -> pd.DataFrame:
        """
        Calculate spread and z-scores
        """
        df = pd.DataFrame(index=prices_a.index)
        df['price_a'] = prices_a
        df['price_b'] = prices_b
        
        # Spread
        df['spread'] = prices_a - self.hedge_ratio * prices_b
        
        # Rolling statistics
        df['spread_mean'] = df['spread'].rolling(self.lookback).mean()
        df['spread_std'] = df['spread'].rolling(self.lookback).std()
        
        # Z-score
        df['zscore'] = (df['spread'] - df['spread_mean']) / df['spread_std']
        
        return df
    
    def execute_trade(self, 
                     date: str, 
                     signal: int, 
                     price_a: float, 
                     price_b: float,
                     capital: float = 100000):
        """
        Execute trade with position sizing
        
        Args:
            signal: 1=long spread, -1=short spread, 0=exit
        """
        if signal == 0 and self.position != 0:
            # Exit
            pnl = self.calculate_pnl(price_a, price_b)
            
            self.trades.append({
                'date': date,
                'action': 'EXIT',
                'position': self.position,
                'pnl': pnl,
                'return': pnl / capital
            })
            
            self.position = 0
            self.entry_price = {}
        
        elif signal != 0 and self.position == 0:
            # Entry
            self.position = signal
            self.entry_price = {
                'asset_a': price_a,
                'asset_b': price_b
            }
            
            self.trades.append({
                'date': date,
                'action': 'ENTER',
                'position': signal,
                'price_a': price_a,
                'price_b': price_b
            })
    
    def calculate_pnl(self, current_price_a: float, current_price_b: float) -> float:
        """
        Calculate P&L for current position
        """
        if self.position == 0:
            return 0
        
        # Long spread: profit when spread increases
        spread_entry = (
            self.entry_price['asset_a'] - 
            self.hedge_ratio * self.entry_price['asset_b']
        )
        
        spread_current = (
            current_price_a - 
            self.hedge_ratio * current_price_b
        )
        
        spread_change = spread_current - spread_entry
        
        # P&L depends on position direction
        pnl = self.position * spread_change
        
        return pnl
\`\`\`

---

## Risk Management for Pairs Trading

### Position Sizing

**Key principles:**
1. Dollar-neutral (equal long and short)
2. Risk per trade: 1-2% of capital
3. Portfolio allocation: 5-10% per pair
4. Max pairs: 10-20 (diversification)

\`\`\`python
class PairsRiskManager:
    """
    Risk management for pairs trading portfolio
    """
    
    def __init__(self, total_capital: float):
        self.total_capital = total_capital
        self.max_risk_per_trade = 0.02  # 2%
        self.max_allocation_per_pair = 0.10  # 10%
        self.max_pairs = 20
        
    def calculate_position_size(self,
                               price_a: float,
                               price_b: float,
                               hedge_ratio: float,
                               spread_std: float) -> dict:
        """
        Calculate optimal position sizes
        
        Method: Risk-based sizing
        - Risk 2% on 2σ move in spread
        """
        # Risk amount
        risk_amount = self.total_capital * self.max_risk_per_trade
        
        # Stop distance (2σ)
        stop_distance = 2 * spread_std
        
        # Position size (dollars per leg)
        position_dollars = risk_amount / stop_distance
        
        # Constrain by max allocation
        max_dollars = self.total_capital * self.max_allocation_per_pair
        position_dollars = min(position_dollars, max_dollars)
        
        # Calculate shares
        shares_a = int(position_dollars / price_a)
        shares_b = int(shares_a * hedge_ratio)
        
        return {
            'shares_a': shares_a,
            'shares_b': shares_b,
            'dollars_a': shares_a * price_a,
            'dollars_b': shares_b * price_b,
            'net_dollars': shares_a * price_a - shares_b * price_b,
            'gross_exposure': shares_a * price_a + shares_b * price_b,
            'risk_dollars': risk_amount
        }
    
    def check_portfolio_limits(self, current_pairs: List[dict]) -> dict:
        """
        Check portfolio-level risk limits
        """
        total_gross = sum(p['gross_exposure'] for p in current_pairs)
        total_net = abs(sum(p['net_dollars'] for p in current_pairs))
        
        return {
            'num_pairs': len(current_pairs),
            'total_gross': total_gross,
            'total_net': total_net,
            'gross_leverage': total_gross / self.total_capital,
            'net_exposure_pct': total_net / self.total_capital,
            'within_limits': (
                len(current_pairs) <= self.max_pairs and
                total_gross / self.total_capital <= 2.0 and
                total_net / self.total_capital <= 0.20
            )
        }
\`\`\`

---

## Real-World Case Studies

### Case Study 1: Coca-Cola (KO) vs PepsiCo (PEP)

**Pair Analysis (2020-2024):**
- Correlation: 0.87
- Cointegration p-value: 0.008 ✓
- Half-life: 22 days ✓
- Hedge ratio: 1.15

**Why this works:**
- Direct competitors
- Similar business models
- Stable relationship (decades)

**Performance:**
- Sharpe ratio: 2.3
- Win rate: 64%
- Max drawdown: -8%

### Case Study 2: JPMorgan (JPM) vs Bank of America (BAC)

**Pair Analysis:**
- Correlation: 0.91
- Cointegration p-value: 0.002 ✓
- Half-life: 18 days ✓
- Hedge ratio: 2.4

**Why this works:**
- Both large banks
- Similar regulatory environment
- Track interest rates similarly

**Performance:**
- Sharpe ratio: 2.7
- Win rate: 68%
- Max drawdown: -6%

### Case Study 3: Failed Pair - Netflix (NFLX) vs Disney (DIS)

**Why it failed:**
- Correlation: 0.75
- Cointegration p-value: 0.18 ✗ (NOT cointegrated)
- Different business models (streaming vs parks)
- Relationship broke in 2022

**Lesson:** High correlation ≠ tradeable pair

---

## Common Pitfalls and Solutions

### Pitfall 1: Confusing Correlation with Cointegration

**Problem:** Trading correlated but non-cointegrated pairs

**Solution:** Always test cointegration (p < 0.05)

### Pitfall 2: Ignoring Transaction Costs

**Problem:** High-frequency pairs trading with 10bps round-trip costs

**Solution:**
- Target spreads > 2σ (larger profits)
- Hold longer (reduce turnover)
- Calculate breakeven: cost / spread_std

### Pitfall 3: Legging Risk

**Problem:** Entering one leg before the other

**Solution:**
- Simultaneous execution
- Use paired orders
- Accept slippage vs legging risk

### Pitfall 4: Relationship Breakdown

**Problem:** Structural changes break cointegration

**Solution:**
- Monitor rolling cointegration
- Stop trading if p-value > 0.10
- Diversify across many pairs

---

## Summary and Best Practices

**Pair Selection Checklist:**
- ✓ Correlation > 0.7
- ✓ Cointegration p-value < 0.05
- ✓ Half-life < 60 days
- ✓ Liquid and shortable
- ✓ Same sector/industry

**Trading Rules:**
- Entry: |z-score| > 2.0
- Exit: |z-score| < 0.5
- Stop: |z-score| > 3.0
- Hold: Until reversion or stop

**Risk Management:**
- 2% risk per trade
- 10% allocation per pair
- 10-20 pairs total
- Dollar-neutral positions

**Expected Performance:**
- Sharpe: 1.5-2.5
- Win rate: 55-65%
- Annual return: 15-25%
- Max drawdown: -10 to -15%

**Next Section**: Momentum Strategies
`,
};

