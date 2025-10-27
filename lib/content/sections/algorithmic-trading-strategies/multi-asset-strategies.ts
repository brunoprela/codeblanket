export const multiAssetStrategies = {
  title: 'Multi-Asset Strategies',
  slug: 'multi-asset-strategies',
  description:
    'Cross-asset trading strategies across equities, bonds, currencies, commodities, and cryptocurrencies',
  content: `
# Multi-Asset Strategies

## Introduction: Diversification Across Asset Classes

Multi-asset strategies trade across different asset classes—equities, bonds, currencies, commodities, and cryptocurrencies—to capture opportunities wherever they emerge and reduce portfolio risk through diversification. Rather than specializing in stocks alone, multi-asset traders exploit correlations, relative value, and macro trends across the entire investment universe.

**What you'll learn:**
- Asset class characteristics and correlations
- Cross-asset arbitrage strategies
- Risk parity and balanced portfolios
- Tactical asset allocation (TAA)
- Global macro strategies
- Currency carry trade
- Commodity momentum
- Crypto-equity correlation trading

**Why this matters for engineers:**
- Different data sources and APIs (stocks, FX, futures, crypto)
- Correlation analysis and cointegration
- Portfolio optimization (mean-variance, risk parity)
- Multi-market execution and liquidity management
- Real-time cross-asset monitoring

**Performance Characteristics:**
- **Sharpe Ratio**: 0.8-1.5 (diversification benefit)
- **Max Drawdown**: 15-25% (lower than equity-only)
- **Correlation to Stocks**: 0.3-0.6 (diversification works)
- **Win Rate**: 60-70% of years positive
- **Crisis Performance**: Better than equity-only (bonds hedge)

---

## Asset Class Characteristics

### 1. Equities

**Characteristics:**
- High expected return (8-10% annually)
- High volatility (15-20% annualized)
- Long-term growth
- Cyclical (economic sensitivity)

**Trading Approaches:**
- Long-only (buy and hold)
- Market timing (tactical allocation)
- Factor strategies (value, momentum)
- Pairs trading (relative value)

**Data:**
- Prices, volumes, fundamentals
- Indices: S&P 500, Russell 2000, MSCI World
- ETFs: SPY, QQQ, IWM

### 2. Fixed Income (Bonds)

**Characteristics:**
- Lower expected return (3-5% annually)
- Lower volatility (5-10% annualized)
- Negative correlation to stocks (flight to safety)
- Interest rate sensitivity

**Trading Approaches:**
- Duration management (long/short rates)
- Yield curve strategies (steepeners, flatteners)
- Credit spreads (corporate vs. treasury)
- Inflation protection (TIPS)

**Data:**
- Yields, spreads, duration
- Indices: Bloomberg Agg, BarCap Treasury
- ETFs: TLT (20Y Treasury), AGG (Aggregate)

### 3. Currencies (FX)

**Characteristics:**
- Medium volatility (8-12% annualized)
- Mean-reverting (purchasing power parity)
- Carry opportunities (interest rate differentials)
- 24-hour market

**Trading Approaches:**
- Carry trade (borrow low-yield, lend high-yield)
- Momentum (trend following)
- Mean reversion (PPP)
- Macro (central bank policy)

**Data:**
- Spot rates, forwards, interest rates
- Pairs: EUR/USD, USD/JPY, GBP/USD
- Futures: 6E (Euro), 6J (Yen)

### 4. Commodities

**Characteristics:**
- High volatility (20-30% annualized)
- Inflation hedge
- Supply/demand driven
- Seasonality

**Trading Approaches:**
- Momentum (trends persist)
- Seasonality (harvest cycles, weather)
- Contango/backwardation (roll yield)
- Macro (China demand, OPEC supply)

**Data:**
- Spot prices, futures curves
- Categories: Energy (oil, gas), Metals (gold, copper), Agriculture (corn, wheat)
- ETFs: GLD (gold), USO (oil), DBA (agriculture)

### 5. Cryptocurrencies

**Characteristics:**
- Extreme volatility (50-100%+ annualized)
- 24/7 trading
- Decentralized, tech-driven
- Emerging correlations

**Trading Approaches:**
- Momentum (strong trends)
- Mean reversion (within trends)
- Cross-exchange arbitrage
- Crypto-equity rotation

**Data:**
- Spot prices, funding rates, on-chain metrics
- Coins: BTC, ETH, SOL
- Exchanges: Binance, Coinbase, Kraken

---

## Correlation Matrix and Diversification

### Historical Correlations (Long-Term)

\`\`\`
            Stocks  Bonds   Gold    USD     Oil
Stocks      1.00    -0.20   0.05    -0.15   0.35
Bonds       -0.20   1.00    0.10    0.25    -0.10
Gold        0.05    0.10    1.00    -0.40   0.20
USD         -0.15   0.25    -0.40   1.00    -0.25
Oil         0.35    -0.10   0.20    -0.25   1.00
\`\`\`

**Key Insights:**
- **Stocks-Bonds**: Negative correlation (flight to safety)
- **Stocks-Gold**: Near zero (gold is hedge, not perfect)
- **USD-Gold**: Negative (gold priced in USD)
- **Stocks-Oil**: Positive (economic growth)

**Crisis Behavior:**
- 2008: Stocks down 38%, Bonds up 5%, Gold up 5%
- 2020 COVID: Stocks down 34% then V-recovery, Bonds up 8%, Gold up 25%
- Diversification works!

\`\`\`python
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from enum import Enum
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy import stats
import logging

class AssetClass(Enum):
    """Asset classes"""
    EQUITY = "EQUITY"
    BOND = "BOND"
    COMMODITY = "COMMODITY"
    CURRENCY = "CURRENCY"
    CRYPTO = "CRYPTO"

@dataclass
class Asset:
    """
    Individual asset
    """
    symbol: str
    asset_class: AssetClass
    expected_return: float  # Annual
    volatility: float  # Annual
    current_price: float
    
    def sharpe_ratio(self, risk_free_rate: float = 0.03) -> float:
        """Calculate Sharpe ratio"""
        return (self.expected_return - risk_free_rate) / self.volatility

@dataclass
class Portfolio:
    """
    Multi-asset portfolio
    """
    assets: List[Asset]
    weights: List[float]
    rebalance_frequency: str = "QUARTERLY"  # MONTHLY, QUARTERLY, ANNUAL
    
    def __post_init__(self):
        assert len(self.assets) == len(self.weights)
        assert abs(sum(self.weights) - 1.0) < 0.01
    
    @property
    def expected_return(self) -> float:
        """Portfolio expected return"""
        return sum(a.expected_return * w for a, w in zip(self.assets, self.weights))
    
    def calculate_volatility(self, correlation_matrix: np.ndarray) -> float:
        """
        Calculate portfolio volatility considering correlations
        
        σ_p = sqrt(w' Σ w)
        
        Args:
            correlation_matrix: Asset correlation matrix
            
        Returns:
            Portfolio volatility (annualized)
        """
        # Covariance matrix
        vols = np.array([a.volatility for a in self.assets])
        cov_matrix = np.outer(vols, vols) * correlation_matrix
        
        # Portfolio variance
        weights = np.array(self.weights)
        variance = weights @ cov_matrix @ weights
        
        return np.sqrt(variance)
    
    def sharpe_ratio(self,
                    correlation_matrix: np.ndarray,
                    risk_free_rate: float = 0.03) -> float:
        """Calculate portfolio Sharpe ratio"""
        vol = self.calculate_volatility(correlation_matrix)
        return (self.expected_return - risk_free_rate) / vol

class CorrelationAnalyzer:
    """
    Analyze correlations between asset classes
    """
    
    def __init__(self, lookback_days: int = 252):
        """
        Initialize analyzer
        
        Args:
            lookback_days: Lookback period for correlation (default 1 year)
        """
        self.lookback_days = lookback_days
        self.logger = logging.getLogger(__name__)
    
    def calculate_correlation_matrix(self,
                                    returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation matrix from returns
        
        Args:
            returns: DataFrame with asset returns (columns = assets)
            
        Returns:
            Correlation matrix
        """
        return returns.corr()
    
    def rolling_correlation(self,
                          returns1: pd.Series,
                          returns2: pd.Series,
                          window: int = 60) -> pd.Series:
        """
        Calculate rolling correlation
        
        Args:
            returns1: First asset returns
            returns2: Second asset returns
            window: Rolling window (days)
            
        Returns:
            Rolling correlation series
        """
        return returns1.rolling(window).corr(returns2)
    
    def detect_regime_shift(self,
                          correlation: pd.Series,
                          threshold: float = 0.3) -> List[datetime]:
        """
        Detect correlation regime shifts
        
        Large changes in correlation = regime change
        
        Args:
            correlation: Time series of correlation
            threshold: Minimum change to detect
            
        Returns:
            List of regime shift dates
        """
        # Calculate changes
        changes = correlation.diff().abs()
        
        # Find large changes
        shifts = changes[changes > threshold]
        
        return list(shifts.index)

class RiskParityStrategy:
    """
    Risk Parity: Allocate based on risk contribution, not capital
    
    Each asset contributes equally to portfolio risk
    """
    
    def __init__(self, assets: List[Asset]):
        """
        Initialize risk parity strategy
        
        Args:
            assets: List of assets
        """
        self.assets = assets
        self.logger = logging.getLogger(__name__)
    
    def calculate_risk_parity_weights(self,
                                     correlation_matrix: np.ndarray) -> np.ndarray:
        """
        Calculate risk parity weights
        
        Optimization problem:
        Minimize: sum of squared differences in risk contributions
        Subject to: weights sum to 1, weights >= 0
        
        Args:
            correlation_matrix: Asset correlation matrix
            
        Returns:
            Optimal weights
        """
        n = len(self.assets)
        
        # Initial guess: equal weight
        initial_weights = np.array([1.0 / n] * n)
        
        # Objective: minimize variance in risk contributions
        def objective(weights):
            # Calculate risk contributions
            vols = np.array([a.volatility for a in self.assets])
            cov_matrix = np.outer(vols, vols) * correlation_matrix
            
            portfolio_var = weights @ cov_matrix @ weights
            marginal_risk = cov_matrix @ weights
            risk_contributions = weights * marginal_risk
            
            # Normalize
            risk_contributions = risk_contributions / risk_contributions.sum()
            
            # Target: equal risk contribution (1/n each)
            target = 1.0 / n
            
            # Sum of squared deviations
            return np.sum((risk_contributions - target) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Sum to 1
        ]
        
        # Bounds: 0 <= weight <= 1
        bounds = [(0, 1) for _ in range(n)]
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not result.success:
            self.logger.warning(f"Optimization failed: {result.message}")
            return initial_weights
        
        return result.x
    
    def construct_portfolio(self,
                           correlation_matrix: np.ndarray) -> Portfolio:
        """
        Construct risk parity portfolio
        
        Args:
            correlation_matrix: Asset correlation matrix
            
        Returns:
            Optimized portfolio
        """
        weights = self.calculate_risk_parity_weights(correlation_matrix)
        
        return Portfolio(
            assets=self.assets,
            weights=list(weights),
            rebalance_frequency="QUARTERLY"
        )

class TacticalAssetAllocation:
    """
    Tactical Asset Allocation (TAA): Dynamic allocation based on signals
    
    Overweight attractive assets, underweight unattractive
    """
    
    def __init__(self,
                 strategic_weights: Dict[AssetClass, float],
                 max_deviation: float = 0.20):
        """
        Initialize TAA strategy
        
        Args:
            strategic_weights: Long-term strategic allocation
            max_deviation: Max deviation from strategic (20% default)
        """
        self.strategic_weights = strategic_weights
        self.max_deviation = max_deviation
    
    def calculate_momentum_score(self,
                                returns: pd.Series,
                                lookbacks: List[int] = [21, 63, 252]) -> float:
        """
        Calculate momentum score using multiple lookbacks
        
        Args:
            returns: Asset return series
            lookbacks: Lookback periods to average
            
        Returns:
            Momentum score (-1 to +1)
        """
        scores = []
        
        for lookback in lookbacks:
            if len(returns) >= lookback:
                cumulative_return = returns.iloc[-lookback:].sum()
                scores.append(cumulative_return)
        
        if not scores:
            return 0.0
        
        # Average and normalize
        avg_score = np.mean(scores)
        
        # Normalize to -1 to +1 (simple approach)
        normalized = np.tanh(avg_score / 0.10)  # tanh(x/0.1) maps returns to -1,1
        
        return normalized
    
    def calculate_value_score(self,
                             current_price: float,
                             historical_avg: float,
                             historical_std: float) -> float:
        """
        Calculate value score (z-score)
        
        Args:
            current_price: Current asset price
            historical_avg: Historical average price
            historical_std: Historical standard deviation
            
        Returns:
            Value score (negative = cheap)
        """
        if historical_std == 0:
            return 0.0
        
        z_score = (current_price - historical_avg) / historical_std
        
        # Negative z-score = cheap = positive score
        return -z_score
    
    def generate_tactical_weights(self,
                                 assets: List[Asset],
                                 momentum_scores: Dict[str, float],
                                 value_scores: Dict[str, float]) -> Dict[AssetClass, float]:
        """
        Generate tactical weights based on signals
        
        Args:
            assets: List of assets
            momentum_scores: Momentum scores by asset
            value_scores: Value scores by asset
            
        Returns:
            Tactical weights by asset class
        """
        tactical_weights = {}
        
        for asset in assets:
            # Get strategic weight
            strategic = self.strategic_weights.get(asset.asset_class, 0.0)
            
            # Get signals
            momentum = momentum_scores.get(asset.symbol, 0.0)
            value = value_scores.get(asset.symbol, 0.0)
            
            # Combined signal (equal weight)
            signal = (momentum + value) / 2
            
            # Tactical tilt (signal * max_deviation)
            tilt = signal * self.max_deviation
            
            # Tactical weight
            tactical = strategic * (1 + tilt)
            
            # Clamp to reasonable bounds
            tactical = max(0, min(tactical, strategic * 2))
            
            tactical_weights[asset.asset_class] = tactical
        
        # Normalize to sum to 1
        total = sum(tactical_weights.values())
        tactical_weights = {k: v / total for k, v in tactical_weights.items()}
        
        return tactical_weights

class CurrencyCarryTrade:
    """
    Currency carry trade: Borrow low-yield currency, lend high-yield
    
    Profit = Interest rate differential - Currency depreciation
    """
    
    def __init__(self, transaction_cost: float = 0.0005):
        """
        Initialize carry trade strategy
        
        Args:
            transaction_cost: Transaction cost (5 bps default)
        """
        self.transaction_cost = transaction_cost
    
    def calculate_carry(self,
                       interest_rate_long: float,
                       interest_rate_short: float) -> float:
        """
        Calculate carry (interest rate differential)
        
        Args:
            interest_rate_long: Interest rate of long currency
            interest_rate_short: Interest rate of short currency
            
        Returns:
            Annual carry
        """
        return interest_rate_long - interest_rate_short
    
    def expected_return(self,
                       carry: float,
                       expected_depreciation: float = 0.0) -> float:
        """
        Calculate expected return from carry trade
        
        Args:
            carry: Interest rate differential
            expected_depreciation: Expected currency depreciation
            
        Returns:
            Expected annual return
        """
        # Return = carry - depreciation - costs
        return carry - expected_depreciation - self.transaction_cost
    
    def construct_carry_portfolio(self,
                                 currencies: List[Tuple[str, float]],
                                 n_long: int = 3,
                                 n_short: int = 3) -> Dict[str, float]:
        """
        Construct carry portfolio
        
        Long top N high-yield currencies
        Short bottom N low-yield currencies
        
        Args:
            currencies: List of (currency, interest_rate)
            n_long: Number of currencies to long
            n_short: Number of currencies to short
            
        Returns:
            Portfolio weights
        """
        # Sort by interest rate
        sorted_currencies = sorted(currencies, key=lambda x: x[1], reverse=True)
        
        # Select top N (long) and bottom N (short)
        long_currencies = sorted_currencies[:n_long]
        short_currencies = sorted_currencies[-n_short:]
        
        # Equal weight
        long_weight = 1.0 / n_long
        short_weight = -1.0 / n_short
        
        portfolio = {}
        
        for currency, _ in long_currencies:
            portfolio[currency] = long_weight
        
        for currency, _ in short_currencies:
            portfolio[currency] = short_weight
        
        return portfolio

class CommodityMomentumStrategy:
    """
    Commodity momentum: Trend following in commodities
    """
    
    def __init__(self, lookback_days: int = 252):
        """
        Initialize commodity momentum
        
        Args:
            lookback_days: Momentum lookback (1 year default)
        """
        self.lookback_days = lookback_days
    
    def calculate_momentum(self, prices: pd.Series) -> float:
        """
        Calculate momentum (12-month return)
        
        Args:
            prices: Price series
            
        Returns:
            12-month return
        """
        if len(prices) < self.lookback_days:
            return 0.0
        
        return (prices.iloc[-1] / prices.iloc[-self.lookback_days]) - 1
    
    def generate_signals(self,
                        commodities: Dict[str, pd.Series]) -> Dict[str, int]:
        """
        Generate momentum signals for commodities
        
        Args:
            commodities: Dict of commodity name to price series
            
        Returns:
            Dict of signals (1 = long, 0 = neutral, -1 = short)
        """
        signals = {}
        
        for commodity, prices in commodities.items():
            momentum = self.calculate_momentum(prices)
            
            # Signal: positive momentum = long, negative = short
            if momentum > 0.05:  # >5% positive
                signals[commodity] = 1
            elif momentum < -0.05:  # >5% negative
                signals[commodity] = -1
            else:
                signals[commodity] = 0
        
        return signals

# Example usage
if __name__ == "__main__":
    print("\\n=== Multi-Asset Strategy System ===\\n")
    
    # 1. Define asset universe
    print("1. Asset Universe")
    
    assets = [
        Asset(symbol="SPY", asset_class=AssetClass.EQUITY,
              expected_return=0.09, volatility=0.18, current_price=450),
        Asset(symbol="TLT", asset_class=AssetClass.BOND,
              expected_return=0.04, volatility=0.08, current_price=95),
        Asset(symbol="GLD", asset_class=AssetClass.COMMODITY,
              expected_return=0.05, volatility=0.15, current_price=180),
        Asset(symbol="UUP", asset_class=AssetClass.CURRENCY,
              expected_return=0.02, volatility=0.07, current_price=28),
    ]
    
    for asset in assets:
        print(f"   {asset.symbol} ({asset.asset_class.value}):")
        print(f"     Expected Return: {asset.expected_return:.1%}")
        print(f"     Volatility: {asset.volatility:.1%}")
        print(f"     Sharpe: {asset.sharpe_ratio():.2f}")
    
    # 2. Correlation matrix (example)
    print("\\n2. Correlation Analysis")
    
    correlation_matrix = np.array([
        [1.00, -0.20, 0.05, -0.15],  # SPY
        [-0.20, 1.00, 0.10, 0.25],   # TLT
        [0.05, 0.10, 1.00, -0.40],   # GLD
        [-0.15, 0.25, -0.40, 1.00],  # UUP
    ])
    
    print("   Correlation Matrix:")
    print("         SPY    TLT    GLD    UUP")
    for i, symbol in enumerate(['SPY', 'TLT', 'GLD', 'UUP']):
        row = "   " + symbol + "  " + "  ".join(f"{correlation_matrix[i][j]:+.2f}" for j in range(4))
        print(row)
    
    # 3. Risk Parity
    print("\\n3. Risk Parity Portfolio")
    
    rp = RiskParityStrategy(assets=assets)
    rp_portfolio = rp.construct_portfolio(correlation_matrix)
    
    print("   Weights:")
    for asset, weight in zip(rp_portfolio.assets, rp_portfolio.weights):
        print(f"     {asset.symbol}: {weight:.1%}")
    
    print(f"\\n   Expected Return: {rp_portfolio.expected_return:.1%}")
    print(f"   Volatility: {rp_portfolio.calculate_volatility(correlation_matrix):.1%}")
    print(f"   Sharpe Ratio: {rp_portfolio.sharpe_ratio(correlation_matrix):.2f}")
    
    # 4. Tactical Asset Allocation
    print("\\n4. Tactical Asset Allocation")
    
    strategic_weights = {
        AssetClass.EQUITY: 0.40,
        AssetClass.BOND: 0.30,
        AssetClass.COMMODITY: 0.20,
        AssetClass.CURRENCY: 0.10
    }
    
    taa = TacticalAssetAllocation(strategic_weights=strategic_weights)
    
    # Example signals
    momentum_scores = {
        'SPY': 0.5,   # Positive momentum
        'TLT': -0.2,  # Negative momentum
        'GLD': 0.3,   # Positive momentum
        'UUP': 0.0    # Neutral
    }
    
    value_scores = {
        'SPY': -0.3,  # Expensive
        'TLT': 0.4,   # Cheap
        'GLD': 0.1,   # Slightly cheap
        'UUP': 0.0    # Fair value
    }
    
    tactical_weights = taa.generate_tactical_weights(assets, momentum_scores, value_scores)
    
    print("   Strategic vs. Tactical Weights:")
    for asset_class in AssetClass:
        strategic = strategic_weights.get(asset_class, 0)
        tactical = tactical_weights.get(asset_class, 0)
        change = tactical - strategic
        print(f"     {asset_class.value}: {strategic:.0%} → {tactical:.0%} ({change:+.0%})")
    
    # 5. Currency Carry Trade
    print("\\n5. Currency Carry Trade")
    
    carry_trade = CurrencyCarryTrade()
    
    currencies = [
        ('USD', 0.05),  # 5% rate
        ('JPY', 0.00),  # 0% rate
        ('AUD', 0.04),  # 4% rate
        ('TRY', 0.15),  # 15% rate (high yield)
        ('CHF', -0.01), # -1% rate (negative)
    ]
    
    carry_portfolio = carry_trade.construct_carry_portfolio(currencies, n_long=2, n_short=2)
    
    print("   Carry Portfolio:")
    for currency, weight in carry_portfolio.items():
        direction = "LONG" if weight > 0 else "SHORT"
        print(f"     {currency}: {direction} {abs(weight):.1%}")
    
    # Calculate expected carry
    long_rate = np.mean([rate for curr, rate in currencies[:2]])
    short_rate = np.mean([rate for curr, rate in currencies[-2:]])
    carry = carry_trade.calculate_carry(long_rate, short_rate)
    
    print(f"\\n   Expected Carry: {carry:.1%}")
    print(f"   Expected Return: {carry_trade.expected_return(carry):.1%}")
\`\`\`

---

## Risk Parity Deep Dive

### The Risk Parity Principle

**Traditional 60/40 Portfolio:**
- 60% stocks, 40% bonds by capital
- But stocks have 3x the volatility
- Result: 90%+ of risk from stocks!

**Risk Parity Approach:**
- Allocate so each asset contributes equal risk
- Leverage low-vol assets (bonds) to match contribution
- Result: Balanced risk, smoother returns

### Implementation

**Step 1: Calculate Risk Contributions**
\`\`\`
Risk Contribution_i = Weight_i * (∂σ_p / ∂Weight_i)
                    = Weight_i * (Cov Matrix × Weights)_i / σ_p
\`\`\`

**Step 2: Optimize Weights**
Minimize variance in risk contributions

**Example Allocation:**
- Stocks: 25% (high vol → low weight)
- Bonds: 50% (low vol → high weight, often leveraged)
- Commodities: 15%
- Others: 10%

### Advantages and Disadvantages

**Pros:**
- More balanced risk
- Lower drawdowns
- Works in different regimes

**Cons:**
- Requires leverage (bonds)
- Rebalancing costs
- Model risk (correlation estimates)

---

## Global Macro Strategies

### Overview

**Definition**: Trade based on macroeconomic views across assets and geographies

**Approach:**1. Analyze macro indicators (GDP, inflation, rates)
2. Form views on direction
3. Express views across assets
4. Manage risk dynamically

**Famous Examples:**
- George Soros: "Breaking the Bank of England" (1992)
- Ray Dalio: Bridgewater Pure Alpha
- Paul Tudor Jones: Macro trader

### Macro Indicators

**Growth:**
- GDP growth
- PMI (Manufacturing index)
- Employment data
- Consumer confidence

**Inflation:**
- CPI, PPI
- Commodity prices
- Wage growth

**Policy:**
- Central bank rates
- Fiscal stimulus
- Regulatory changes

### Trade Examples

**Scenario 1: Rising Inflation**
- Short bonds (yields up, prices down)
- Long commodities (inflation hedge)
- Long value stocks (benefit from rates)
- Short duration (reduce rate sensitivity)

**Scenario 2: Recession Coming**
- Long bonds (flight to safety)
- Short cyclical stocks
- Long defensive sectors (utilities, healthcare)
- Long USD (safe haven)

---

## Crypto-Equity Correlation Trading

### Emerging Opportunity

**Historical Pattern:**
- 2017-2020: Low correlation (0.1-0.3)
- 2021-2023: Rising correlation (0.5-0.7)
- Crypto becoming more like "risk-on" asset

**Strategy:**
- When correlation breaks down → arbitrage
- Risk-on regime: Long both
- Risk-off regime: Short both or long bonds

### Implementation

\`\`\`python
def crypto_equity_signal(btc_return, spy_return, correlation):
    """
    Generate signal based on crypto-equity relationship
    
    If correlation high and both moving same direction: momentum
    If correlation breaks: mean reversion
    """
    if correlation > 0.7:
        # High correlation: momentum
        if btc_return > 0 and spy_return > 0:
            return "LONG_BOTH"
        elif btc_return < 0 and spy_return < 0:
            return "SHORT_BOTH"
    else:
        # Low correlation: divergence trade
        if btc_return > 0 and spy_return < 0:
            return "LONG_BTC_SHORT_SPY"
        elif btc_return < 0 and spy_return > 0:
            return "SHORT_BTC_LONG_SPY"
    
    return "NEUTRAL"
\`\`\`

---

## Summary and Key Takeaways

**Multi-Asset Strategies Work When:**
- Diversification provides real benefit (crisis periods)
- Correlations stable and predictable
- Execution costs manageable
- Leverage available (for risk parity)

**Multi-Asset Strategies Fail When:**
- Correlations converge (everything falls together)
- Poor diversification (pseudo-diversification)
- Over-optimization (too many parameters)
- Execution complexity

**Critical Success Factors:**1. **True Diversification**: Uncorrelated assets
2. **Risk Management**: Position sizing by risk, not capital
3. **Execution**: Multiple brokers, APIs, markets
4. **Rebalancing**: Systematic and disciplined
5. **Cost Control**: Transaction costs add up

**Best Practices:**
- Start with 3-4 core asset classes
- Use risk parity or strategic allocation as base
- Add tactical tilts cautiously
- Monitor correlations dynamically
- Backtest across market regimes

**Next Section:** Strategy Performance Attribution
`,
};
