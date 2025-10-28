import { Content } from '@/lib/types';

const transactionCostsAndSlippage: Content = {
  title: 'Transaction Costs and Slippage',
  description:
    'Master realistic transaction cost modeling, bid-ask spreads, market impact, commission structures, and slippage estimation for accurate backtesting',
  sections: [
    {
      title: 'Introduction to Transaction Costs',
      content: `
# Introduction to Transaction Costs

Transaction costs are often the difference between a profitable strategy on paper and one that loses money in live trading. Many backtests fail in production because they underestimate or completely ignore these costs.

## Why Transaction Costs Matter

**Real-World Impact**: A high-frequency trading firm discovered their "profitable" strategy was actually losing $50,000 per day once commissions and slippage were properly accounted for. The backtest showed a 15% annual return, but live trading resulted in a -8% return.

### Components of Transaction Costs

Transaction costs include multiple elements:

1. **Explicit Costs**:
   - Broker commissions and fees
   - Exchange fees
   - Regulatory fees (SEC fees, FINRA fees)
   - Data fees
   - Platform fees

2. **Implicit Costs**:
   - Bid-ask spread
   - Market impact (price movement from your order)
   - Slippage (execution at worse price than expected)
   - Opportunity cost (missed trades)

3. **Indirect Costs**:
   - Technology infrastructure
   - Data subscriptions
   - Compliance and regulatory costs
   - Risk management systems

## Cost Structure by Trading Frequency

\`\`\`python
from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np

class TradingFrequency(Enum):
    """Trading frequency categories"""
    HIGH_FREQUENCY = "high_frequency"  # < 1 second
    MEDIUM_FREQUENCY = "medium_frequency"  # Minutes to hours
    LOW_FREQUENCY = "low_frequency"  # Days to weeks
    POSITION_TRADING = "position_trading"  # Weeks to months

@dataclass
class CostStructure:
    """Cost structure for different trading styles"""
    commission_per_share: float  # USD per share
    commission_minimum: float  # Minimum commission per trade
    exchange_fee_rate: float  # As % of trade value
    sec_fee_rate: float  # SEC fees as % of trade value
    typical_spread_bps: float  # Bid-ask spread in basis points
    market_impact_factor: float  # Market impact coefficient
    
    def total_explicit_cost(
        self, 
        shares: int, 
        price: float
    ) -> float:
        """
        Calculate total explicit costs for a trade
        
        Args:
            shares: Number of shares traded
            price: Price per share
            
        Returns:
            Total explicit cost in USD
        """
        trade_value = shares * price
        
        # Commission
        commission = max(
            shares * self.commission_per_share,
            self.commission_minimum
        )
        
        # Exchange fees
        exchange_fee = trade_value * self.exchange_fee_rate
        
        # SEC fees (only on sells)
        sec_fee = trade_value * self.sec_fee_rate
        
        total_cost = commission + exchange_fee + sec_fee
        
        return total_cost
    
    def spread_cost(
        self, 
        shares: int, 
        price: float
    ) -> float:
        """
        Calculate bid-ask spread cost
        
        Args:
            shares: Number of shares traded
            price: Mid price
            
        Returns:
            Cost of crossing the spread in USD
        """
        spread_dollars = price * (self.typical_spread_bps / 10000)
        # Cross half the spread on each trade
        return shares * spread_dollars * 0.5
    
    def total_cost(
        self, 
        shares: int, 
        price: float,
        is_sell: bool = False
    ) -> float:
        """
        Calculate total transaction cost
        
        Args:
            shares: Number of shares
            price: Price per share
            is_sell: Whether this is a sell order
            
        Returns:
            Total cost in USD
        """
        explicit = self.total_explicit_cost(shares, price)
        spread = self.spread_cost(shares, price)
        
        # SEC fees only on sells
        if not is_sell:
            trade_value = shares * price
            explicit -= trade_value * self.sec_fee_rate
            
        return explicit + spread


def get_cost_structure(
    frequency: TradingFrequency,
    broker_tier: str = "retail"
) -> CostStructure:
    """
    Get typical cost structure for trading frequency
    
    Args:
        frequency: Trading frequency category
        broker_tier: Broker tier (retail, professional, institutional)
        
    Returns:
        CostStructure for the given parameters
    """
    # Commission structures
    commissions = {
        "retail": {
            "per_share": 0.0,  # Many retail brokers now commission-free
            "minimum": 0.0,
        },
        "professional": {
            "per_share": 0.005,
            "minimum": 1.0,
        },
        "institutional": {
            "per_share": 0.003,
            "minimum": 0.5,
        }
    }
    
    # Spread and market impact by frequency
    characteristics = {
        TradingFrequency.HIGH_FREQUENCY: {
            "spread_bps": 1.0,  # Trade very liquid stocks
            "market_impact_factor": 0.1,
        },
        TradingFrequency.MEDIUM_FREQUENCY: {
            "spread_bps": 3.0,
            "market_impact_factor": 0.3,
        },
        TradingFrequency.LOW_FREQUENCY: {
            "spread_bps": 5.0,
            "market_impact_factor": 0.5,
        },
        TradingFrequency.POSITION_TRADING: {
            "spread_bps": 8.0,
            "market_impact_factor": 0.7,
        }
    }
    
    comm = commissions[broker_tier]
    char = characteristics[frequency]
    
    return CostStructure(
        commission_per_share=comm["per_share"],
        commission_minimum=comm["minimum"],
        exchange_fee_rate=0.0000119,  # Typical exchange fee
        sec_fee_rate=0.0000051,  # SEC Section 31 fee
        typical_spread_bps=char["spread_bps"],
        market_impact_factor=char["market_impact_factor"]
    )


# Example usage
if __name__ == "__main__":
    # Medium frequency trading with professional broker
    costs = get_cost_structure(
        TradingFrequency.MEDIUM_FREQUENCY,
        broker_tier="professional"
    )
    
    # Calculate cost for trading 1000 shares at $50
    shares = 1000
    price = 50.0
    
    total_cost = costs.total_cost(shares, price, is_sell=False)
    cost_per_share = total_cost / shares
    cost_bps = (cost_per_share / price) * 10000
    
    print(f"Trading {shares} shares at \${price}")
    print(f"Total cost: \${total_cost:.2f}")
    print(f"Cost per share: \${cost_per_share:.4f}")
    print(f"Cost in basis points: {cost_bps:.2f} bps")
\`\`\`

## Cost Analysis Output

For the example above (1000 shares at $50 with professional broker):
- Commission: $5.00 (1000 × $0.005)
- Exchange fee: $0.60 (0.00119% of $50,000)
- Spread cost: $7.50 (3 bps × $50,000 / 2)
- **Total: $13.10 (2.62 basis points)**

This may seem small, but for a strategy that makes 100 round trips per day, the annual cost would be $657,000!

## Production Considerations

### 1. **Dynamic Cost Models**

Real-world costs vary by:
- Time of day (spreads wider at open/close)
- Market conditions (volatility increases spreads)
- Stock liquidity
- Order size relative to volume
- Market regime (bull vs bear markets)

### 2. **Cost Attribution**

Track costs by:
- Strategy
- Asset class
- Time period
- Broker/venue
- Order type

### 3. **Cost Optimization**

Strategies for minimizing costs:
- **Smart order routing**: Route to venues with best prices
- **Order type selection**: Limit vs market orders
- **Trade timing**: Avoid crossing during volatile periods
- **Order splitting**: Break large orders into smaller pieces
- **Maker/taker strategies**: Earn rebates by providing liquidity

### 4. **Benchmark Against Actual Costs**

Regularly compare:
- Backtest cost assumptions vs actual costs
- Broker reports vs internal calculations
- Pre-trade estimates vs post-trade analysis
- Different brokers and venues

## Common Pitfalls

1. **Zero-cost assumption**: Assuming trades execute at mid-price with no costs
2. **Fixed cost per trade**: Ignoring that costs scale with order size and market conditions
3. **Ignoring market impact**: Large orders move prices against you
4. **Stale cost assumptions**: Not updating for changing market structure
5. **Forgetting opportunity costs**: Unfilled limit orders have costs too
`,
    },
    {
      title: 'Bid-Ask Spread Modeling',
      content: `
# Bid-Ask Spread Modeling

The bid-ask spread is the difference between the highest price a buyer is willing to pay (bid) and the lowest price a seller will accept (ask). It's a fundamental transaction cost.

## Understanding the Spread

**Market Microstructure**: The spread compensates market makers for:
- Inventory risk (holding positions)
- Adverse selection (trading with informed traders)
- Order processing costs
- Opportunity costs

### Spread Components

The spread can be decomposed into:

1. **Order Processing Cost**: Infrastructure and labor
2. **Inventory Holding Cost**: Risk of price movement
3. **Adverse Selection Cost**: Risk of trading with informed traders

## Spread Estimation Models

\`\`\`python
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass

@dataclass
class SpreadEstimate:
    """Spread estimation results"""
    effective_spread_bps: float
    quoted_spread_bps: float
    realized_spread_bps: float
    price_impact_bps: float
    timestamp: pd.Timestamp

class SpreadEstimator:
    """
    Estimate bid-ask spreads using various methods
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with market data
        
        Args:
            data: DataFrame with columns ['timestamp', 'price', 'volume', 
                  'bid', 'ask'] if available
        """
        self.data = data.copy()
        self.data.sort_values('timestamp', inplace=True)
    
    def quoted_spread(self) -> pd.Series:
        """
        Calculate quoted spread from bid/ask data
        
        Returns:
            Series of quoted spreads in basis points
        """
        if 'bid' not in self.data.columns or 'ask' not in self.data.columns:
            raise ValueError("Bid/ask data required for quoted spread")
        
        mid_price = (self.data['bid'] + self.data['ask']) / 2
        spread = self.data['ask'] - self.data['bid']
        spread_bps = (spread / mid_price) * 10000
        
        return spread_bps
    
    def roll_spread(self, window: int = 20) -> pd.Series:
        """
        Estimate spread using Roll (1984) model
        
        The Roll model estimates spread from price autocovariance.
        Spread = 2 * sqrt(-Cov(ΔP_t, ΔP_{t-1}))
        
        Args:
            window: Rolling window size
            
        Returns:
            Series of estimated spreads in basis points
        """
        returns = self.data['price'].diff()
        
        # Calculate rolling autocovariance
        autocov = returns.rolling(window).apply(
            lambda x: np.cov(x[:-1], x[1:])[0, 1] if len(x) > 1 else np.nan,
            raw=True
        )
        
        # Roll spread formula
        # If autocov is positive, spread estimate is undefined (set to 0)
        spread_estimate = np.where(
            autocov < 0,
            2 * np.sqrt(-autocov),
            0.0
        )
        
        # Convert to basis points
        spread_bps = (spread_estimate / self.data['price']) * 10000
        
        return pd.Series(spread_bps, index=self.data.index)
    
    def effective_spread(
        self, 
        trade_direction: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Calculate effective spread from trade prices
        
        Effective spread = 2 * |trade_price - mid_price|
        
        Args:
            trade_direction: Series indicating buy (1) or sell (-1)
            
        Returns:
            Series of effective spreads in basis points
        """
        if 'bid' in self.data.columns and 'ask' in self.data.columns:
            mid_price = (self.data['bid'] + self.data['ask']) / 2
        else:
            # Estimate mid price as average of adjacent trades
            mid_price = self.data['price'].rolling(2).mean()
        
        # If trade direction not provided, estimate it
        if trade_direction is None:
            trade_direction = self._estimate_trade_direction()
        
        # Effective spread
        spread = 2 * np.abs(self.data['price'] - mid_price)
        spread_bps = (spread / mid_price) * 10000
        
        return spread_bps
    
    def _estimate_trade_direction(self) -> pd.Series:
        """
        Estimate trade direction using tick rule
        
        Returns:
            Series of trade directions (1 for buy, -1 for sell)
        """
        price_change = self.data['price'].diff()
        
        # Tick rule: compare to previous price
        direction = np.where(price_change > 0, 1, 
                           np.where(price_change < 0, -1, 0))
        
        # Forward fill zero values (trade at same price as previous)
        direction = pd.Series(direction, index=self.data.index)
        direction.replace(0, np.nan, inplace=True)
        direction.ffill(inplace=True)
        direction.fillna(1, inplace=True)  # Default to buy
        
        return direction
    
    def high_low_spread(self, window: int = 20) -> pd.Series:
        """
        Estimate spread using Corwin-Schultz (2012) high-low method
        
        Uses daily high and low prices to estimate spread.
        
        Args:
            window: Rolling window for estimation
            
        Returns:
            Series of estimated spreads in basis points
        """
        if 'high' not in self.data.columns or 'low' not in self.data.columns:
            # Calculate from price data
            self.data['high'] = self.data['price'].rolling(window).max()
            self.data['low'] = self.data['price'].rolling(window).min()
        
        # High-low ratio
        hl_ratio = self.data['high'] / self.data['low']
        beta = np.log(hl_ratio) ** 2
        
        # Rolling sum for two-day window
        beta_sum = beta.rolling(2).sum()
        
        # Spread estimate
        spread_estimate = (2 * (np.exp(beta_sum / 2) - 1)) / \
                         (1 + np.exp(beta_sum / 2))
        
        # Convert to basis points
        mid_price = (self.data['high'] + self.data['low']) / 2
        spread_bps = (spread_estimate * mid_price / mid_price) * 10000
        
        return spread_bps.clip(lower=0)  # Spread must be non-negative
    
    def realized_spread(
        self, 
        horizon: int = 5
    ) -> pd.Series:
        """
        Calculate realized spread (spread minus price impact)
        
        Realized spread = 2 * D * (trade_price - mid_price_{t+horizon})
        where D is trade direction
        
        Args:
            horizon: Periods ahead to measure price impact
            
        Returns:
            Series of realized spreads in basis points
        """
        if 'bid' not in self.data.columns or 'ask' not in self.data.columns:
            raise ValueError("Bid/ask data required for realized spread")
        
        mid_price = (self.data['bid'] + self.data['ask']) / 2
        future_mid = mid_price.shift(-horizon)
        
        trade_direction = self._estimate_trade_direction()
        
        realized = 2 * trade_direction * (self.data['price'] - future_mid)
        realized_bps = (realized / mid_price) * 10000
        
        return realized_bps
    
    def get_comprehensive_analysis(self) -> pd.DataFrame:
        """
        Calculate all spread measures
        
        Returns:
            DataFrame with all spread estimates
        """
        results = pd.DataFrame(index=self.data.index)
        results['timestamp'] = self.data['timestamp']
        results['price'] = self.data['price']
        
        # Calculate available spread measures
        if 'bid' in self.data.columns and 'ask' in self.data.columns:
            results['quoted_spread_bps'] = self.quoted_spread()
            results['realized_spread_bps'] = self.realized_spread()
        
        results['roll_spread_bps'] = self.roll_spread()
        results['effective_spread_bps'] = self.effective_spread()
        results['high_low_spread_bps'] = self.high_low_spread()
        
        return results


# Example: Analyze spread patterns
def analyze_spread_patterns(
    ticker: str, 
    data: pd.DataFrame
) -> Dict[str, float]:
    """
    Analyze spread patterns for a ticker
    
    Args:
        ticker: Stock ticker
        data: Market data
        
    Returns:
        Dictionary of spread statistics
    """
    estimator = SpreadEstimator(data)
    analysis = estimator.get_comprehensive_analysis()
    
    # Calculate statistics
    stats_dict = {}
    
    for col in analysis.columns:
        if col.endswith('_bps'):
            stats_dict[f"{col}_mean"] = analysis[col].mean()
            stats_dict[f"{col}_median"] = analysis[col].median()
            stats_dict[f"{col}_std"] = analysis[col].std()
            stats_dict[f"{col}_p95"] = analysis[col].quantile(0.95)
    
    # Time-of-day analysis
    analysis['hour'] = pd.to_datetime(analysis['timestamp']).dt.hour
    stats_dict['spread_at_open'] = analysis[
        analysis['hour'] == 9
    ]['effective_spread_bps'].mean()
    stats_dict['spread_midday'] = analysis[
        (analysis['hour'] >= 11) & (analysis['hour'] <= 14)
    ]['effective_spread_bps'].mean()
    stats_dict['spread_at_close'] = analysis[
        analysis['hour'] == 15
    ]['effective_spread_bps'].mean()
    
    return stats_dict


# Example usage
if __name__ == "__main__":
    # Simulate market data
    np.random.seed(42)
    n_points = 1000
    
    dates = pd.date_range('2024-01-01', periods=n_points, freq='1min')
    base_price = 100.0
    
    # Simulate prices with bid-ask spread
    mid_prices = base_price + np.cumsum(np.random.randn(n_points) * 0.1)
    spread_bps = 5.0  # 5 basis points
    spread_dollars = mid_prices * (spread_bps / 10000)
    
    data = pd.DataFrame({
        'timestamp': dates,
        'bid': mid_prices - spread_dollars / 2,
        'ask': mid_prices + spread_dollars / 2,
        'price': mid_prices + np.random.choice([-1, 1], n_points) * spread_dollars / 2,
        'volume': np.random.randint(100, 1000, n_points)
    })
    
    # Analyze spreads
    estimator = SpreadEstimator(data)
    analysis = estimator.get_comprehensive_analysis()
    
    print("\\nSpread Analysis:")
    print(f"Quoted spread (mean): {analysis['quoted_spread_bps'].mean():.2f} bps")
    print(f"Effective spread (mean): {analysis['effective_spread_bps'].mean():.2f} bps")
    print(f"Roll spread (mean): {analysis['roll_spread_bps'].mean():.2f} bps")
\`\`\`

## Spread Patterns

### Intraday Patterns

Spreads typically:
- **Widen at open**: Information asymmetry from overnight news
- **Narrow midday**: Peak liquidity
- **Widen at close**: Position squaring, inventory management

### Volatility Impact

During high volatility:
- Spreads can double or triple
- Market makers widen quotes to protect against adverse selection
- Liquidity provision becomes more expensive

## Production Implementation

### 1. **Real-Time Spread Monitoring**

\`\`\`python
class SpreadMonitor:
    """Monitor spreads in real-time"""
    
    def __init__(self, alert_threshold_bps: float = 20.0):
        self.alert_threshold = alert_threshold_bps
        self.spread_history: List[SpreadEstimate] = []
    
    def update(
        self, 
        bid: float, 
        ask: float, 
        timestamp: pd.Timestamp
    ) -> Optional[str]:
        """
        Update with new quote and check for alerts
        
        Returns:
            Alert message if spread exceeds threshold
        """
        mid = (bid + ask) / 2
        spread_bps = ((ask - bid) / mid) * 10000
        
        estimate = SpreadEstimate(
            effective_spread_bps=spread_bps,
            quoted_spread_bps=spread_bps,
            realized_spread_bps=0.0,  # Calculated later
            price_impact_bps=0.0,
            timestamp=timestamp
        )
        
        self.spread_history.append(estimate)
        
        # Alert if spread exceeds threshold
        if spread_bps > self.alert_threshold:
            return f"ALERT: Spread {spread_bps:.1f} bps exceeds threshold {self.alert_threshold} bps"
        
        return None
\`\`\`

### 2. **Dynamic Spread Adjustment**

Adjust backtested strategies for actual spread conditions:
- Use historical spread data for backtesting
- Apply percentile-based spreads (e.g., 75th percentile for conservative estimates)
- Stress test with widened spreads

### 3. **Spread Forecasting**

Predict future spreads based on:
- Time of day
- Volatility forecasts
- Volume patterns
- Market regime
`,
    },
    {
      title: 'Market Impact Models',
      content: `
# Market Impact Models

Market impact is the price movement caused by executing a trade. Large orders can significantly move prices against the trader, especially in less liquid stocks.

## Impact Components

Market impact consists of:

1. **Temporary Impact**: Short-term price movement that reverts
2. **Permanent Impact**: Lasting price change from information revelation
3. **Timing Impact**: Cost of waiting to execute (opportunity cost)

## Square Root Law

The most widely-used market impact model is the square root law:

**Impact ∝ √(Q/V) × σ**

Where:
- Q = Order size
- V = Average daily volume
- σ = Volatility

\`\`\`python
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

class ImpactModel(Enum):
    """Market impact model types"""
    SQUARE_ROOT = "square_root"
    LINEAR = "linear"
    ALMGREN_CHRISS = "almgren_chriss"
    KISSELL = "kissell"

@dataclass
class MarketImpactParams:
    """Parameters for market impact calculation"""
    temporary_impact_coef: float  # Coefficient for temporary impact
    permanent_impact_coef: float  # Coefficient for permanent impact
    volatility_multiplier: float  # How much volatility affects impact
    liquidity_exponent: float  # Exponent in participation rate (usually 0.5)
    
class MarketImpactCalculator:
    """
    Calculate market impact using various models
    """
    
    def __init__(
        self, 
        model: ImpactModel = ImpactModel.SQUARE_ROOT,
        params: Optional[MarketImpactParams] = None
    ):
        self.model = model
        
        if params is None:
            # Default parameters (calibrated to US equities)
            self.params = MarketImpactParams(
                temporary_impact_coef=0.314,  # Empirically calibrated
                permanent_impact_coef=0.142,
                volatility_multiplier=0.5,
                liquidity_exponent=0.5
            )
        else:
            self.params = params
    
    def calculate_impact(
        self,
        order_size: int,
        avg_daily_volume: float,
        current_price: float,
        volatility: float,
        execution_time_seconds: float = 300.0
    ) -> Dict[str, float]:
        """
        Calculate market impact
        
        Args:
            order_size: Number of shares to trade
            avg_daily_volume: Average daily volume in shares
            current_price: Current stock price
            volatility: Daily volatility (annualized)
            execution_time_seconds: Time to execute order
            
        Returns:
            Dictionary with impact breakdown
        """
        if self.model == ImpactModel.SQUARE_ROOT:
            return self._square_root_impact(
                order_size, avg_daily_volume, current_price, 
                volatility, execution_time_seconds
            )
        elif self.model == ImpactModel.LINEAR:
            return self._linear_impact(
                order_size, avg_daily_volume, current_price, volatility
            )
        elif self.model == ImpactModel.ALMGREN_CHRISS:
            return self._almgren_chriss_impact(
                order_size, avg_daily_volume, current_price, 
                volatility, execution_time_seconds
            )
        else:
            raise ValueError(f"Unknown model: {self.model}")
    
    def _square_root_impact(
        self,
        order_size: int,
        avg_daily_volume: float,
        current_price: float,
        volatility: float,
        execution_time_seconds: float
    ) -> Dict[str, float]:
        """
        Square root impact model
        
        Impact = η * σ * sqrt(Q/V) * price
        """
        # Participation rate (fraction of daily volume)
        participation_rate = order_size / avg_daily_volume
        
        # Daily volatility to per-second volatility
        # Assuming 6.5 trading hours (23400 seconds)
        vol_per_second = volatility / np.sqrt(252 * 23400)
        
        # Temporary impact
        temp_impact_bps = (
            self.params.temporary_impact_coef * 
            volatility *  # Daily vol
            np.sqrt(participation_rate) *
            10000  # Convert to bps
        )
        
        # Permanent impact (typically smaller)
        perm_impact_bps = (
            self.params.permanent_impact_coef *
            volatility *
            np.sqrt(participation_rate) *
            10000
        )
        
        # Total impact in dollars
        temp_impact_dollars = (temp_impact_bps / 10000) * current_price * order_size
        perm_impact_dollars = (perm_impact_bps / 10000) * current_price * order_size
        
        # Average execution price shift
        avg_price_shift = (temp_impact_bps + perm_impact_bps) / 2
        
        return {
            "temporary_impact_bps": temp_impact_bps,
            "permanent_impact_bps": perm_impact_bps,
            "total_impact_bps": temp_impact_bps + perm_impact_bps,
            "temporary_impact_dollars": temp_impact_dollars,
            "permanent_impact_dollars": perm_impact_dollars,
            "total_impact_dollars": temp_impact_dollars + perm_impact_dollars,
            "avg_execution_price_shift_bps": avg_price_shift,
            "participation_rate": participation_rate
        }
    
    def _linear_impact(
        self,
        order_size: int,
        avg_daily_volume: float,
        current_price: float,
        volatility: float
    ) -> Dict[str, float]:
        """
        Linear impact model (conservative for large orders)
        
        Impact = β * (Q/V) * σ * price
        """
        participation_rate = order_size / avg_daily_volume
        
        # Linear impact tends to be higher for same participation rate
        impact_bps = (
            self.params.temporary_impact_coef * 2.0 *  # Higher coefficient
            volatility *
            participation_rate *  # Linear, not square root
            10000
        )
        
        impact_dollars = (impact_bps / 10000) * current_price * order_size
        
        return {
            "temporary_impact_bps": impact_bps,
            "permanent_impact_bps": 0.0,
            "total_impact_bps": impact_bps,
            "temporary_impact_dollars": impact_dollars,
            "permanent_impact_dollars": 0.0,
            "total_impact_dollars": impact_dollars,
            "avg_execution_price_shift_bps": impact_bps,
            "participation_rate": participation_rate
        }
    
    def _almgren_chriss_impact(
        self,
        order_size: int,
        avg_daily_volume: float,
        current_price: float,
        volatility: float,
        execution_time_seconds: float
    ) -> Dict[str, float]:
        """
        Almgren-Chriss optimal execution model
        
        Balances market impact vs timing risk
        """
        # Model parameters
        gamma = self.params.temporary_impact_coef  # Temporary impact
        eta = self.params.permanent_impact_coef  # Permanent impact
        
        # Participation rate
        participation_rate = order_size / avg_daily_volume
        
        # Time factor (normalize to fraction of day)
        time_fraction = execution_time_seconds / 23400  # 6.5 trading hours
        
        # Temporary impact (decreases with execution time)
        temp_impact_bps = (
            gamma *
            volatility *
            np.sqrt(participation_rate / time_fraction) *
            10000
        )
        
        # Permanent impact (independent of execution time)
        perm_impact_bps = (
            eta *
            volatility *
            participation_rate *
            10000
        )
        
        temp_impact_dollars = (temp_impact_bps / 10000) * current_price * order_size
        perm_impact_dollars = (perm_impact_bps / 10000) * current_price * order_size
        
        return {
            "temporary_impact_bps": temp_impact_bps,
            "permanent_impact_bps": perm_impact_bps,
            "total_impact_bps": temp_impact_bps + perm_impact_bps,
            "temporary_impact_dollars": temp_impact_dollars,
            "permanent_impact_dollars": perm_impact_dollars,
            "total_impact_dollars": temp_impact_dollars + perm_impact_dollars,
            "avg_execution_price_shift_bps": (temp_impact_bps + perm_impact_bps) / 2,
            "participation_rate": participation_rate,
            "execution_time_minutes": execution_time_seconds / 60
        }
    
    def optimal_execution_schedule(
        self,
        total_shares: int,
        execution_horizon_seconds: float,
        avg_daily_volume: float,
        current_price: float,
        volatility: float,
        risk_aversion: float = 1.0
    ) -> pd.DataFrame:
        """
        Calculate optimal execution schedule (VWAP-style)
        
        Args:
            total_shares: Total shares to execute
            execution_horizon_seconds: Time to complete execution
            avg_daily_volume: Average daily volume
            current_price: Current price
            volatility: Volatility
            risk_aversion: Risk aversion parameter (higher = more conservative)
            
        Returns:
            DataFrame with execution schedule
        """
        # Divide execution into intervals
        num_intervals = max(10, int(execution_horizon_seconds / 60))  # At least 10, or 1 per minute
        interval_seconds = execution_horizon_seconds / num_intervals
        
        # Almgren-Chriss optimal trajectory
        # For simplicity, use arithmetic schedule (can be optimized further)
        kappa = np.sqrt(
            self.params.permanent_impact_coef * risk_aversion * volatility ** 2 /
            self.params.temporary_impact_coef
        )
        
        # Calculate trade list
        times = np.arange(0, num_intervals + 1) * interval_seconds
        
        # Arithmetic participation (simplified)
        shares_per_interval = total_shares / num_intervals
        
        schedule_data = []
        remaining_shares = total_shares
        
        for i in range(num_intervals):
            # Calculate impact for this slice
            impact = self.calculate_impact(
                int(shares_per_interval),
                avg_daily_volume,
                current_price,
                volatility,
                interval_seconds
            )
            
            schedule_data.append({
                'interval': i,
                'time_seconds': times[i],
                'shares_to_trade': int(shares_per_interval),
                'remaining_shares': remaining_shares,
                'impact_bps': impact['total_impact_bps'],
                'impact_dollars': impact['total_impact_dollars'],
                'cumulative_cost': sum(s['impact_dollars'] for s in schedule_data) if schedule_data else 0
            })
            
            remaining_shares -= shares_per_interval
        
        return pd.DataFrame(schedule_data)


# Example: Compare impact models
def compare_impact_models(
    order_size: int,
    avg_daily_volume: float,
    price: float,
    volatility: float
) -> pd.DataFrame:
    """Compare different impact models"""
    
    models = [
        ImpactModel.SQUARE_ROOT,
        ImpactModel.LINEAR,
        ImpactModel.ALMGREN_CHRISS
    ]
    
    results = []
    
    for model in models:
        calculator = MarketImpactCalculator(model=model)
        impact = calculator.calculate_impact(
            order_size,
            avg_daily_volume,
            price,
            volatility
        )
        
        results.append({
            'model': model.value,
            'total_impact_bps': impact['total_impact_bps'],
            'total_impact_dollars': impact['total_impact_dollars'],
            'participation_rate': impact['participation_rate']
        })
    
    return pd.DataFrame(results)


# Example usage
if __name__ == "__main__":
    # Trade parameters
    order_size = 50000  # shares
    avg_daily_volume = 2000000  # shares
    price = 100.0
    volatility = 0.25  # 25% annualized
    
    print("Market Impact Comparison\\n")
    comparison = compare_impact_models(
        order_size, avg_daily_volume, price, volatility
    )
    print(comparison.to_string(index=False))
    
    # Optimal execution schedule
    calculator = MarketImpactCalculator(model=ImpactModel.ALMGREN_CHRISS)
    schedule = calculator.optimal_execution_schedule(
        total_shares=order_size,
        execution_horizon_seconds=3600,  # 1 hour
        avg_daily_volume=avg_daily_volume,
        current_price=price,
        volatility=volatility,
        risk_aversion=1.0
    )
    
    print("\\n\\nOptimal Execution Schedule (first 5 intervals):")
    print(schedule.head().to_string(index=False))
    print(f"\\nTotal execution cost: \${schedule['impact_dollars'].sum():,.2f}")
\`\`\`

## Production Considerations

### 1. **Parameter Calibration**

Calibrate impact parameters using:
- Historical execution data
- Transaction Cost Analysis (TCA)
- Broker TCA reports
- Academic research (updated coefficients)

### 2. **Real-Time Adjustment**

Adjust impact estimates based on:
- Current market conditions
- Real-time volume
- Spread widening
- News events

### 3. **Order Slicing**

Implement algorithms to minimize impact:
- **VWAP**: Volume-Weighted Average Price
- **TWAP**: Time-Weighted Average Price
- **POV**: Percentage of Volume
- **Implementation Shortfall**: Balance speed vs cost

## Summary

Market impact is a critical cost that scales non-linearly with order size. Production systems must:
- Use calibrated impact models
- Optimize execution strategies
- Monitor actual vs predicted impact
- Adjust strategies based on realized costs
`,
    },
    {
      title: 'Slippage Estimation and Modeling',
      content: `
# Slippage Estimation and Modeling

Slippage is the difference between the expected execution price and the actual execution price. It combines spread costs, market impact, and timing effects.

## Sources of Slippage

1. **Quote Slippage**: Quotes change between signal generation and order arrival
2. **Execution Slippage**: Price moves during order execution
3. **Partial Fills**: Unable to complete order at desired price
4. **Order Type Slippage**: Market orders vs limit orders

## Slippage Models

\`\`\`python
from typing import Optional, List, Dict
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats

class OrderType(Enum):
    """Order types with different slippage characteristics"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class LiquidityRegime(Enum):
    """Market liquidity regimes"""
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    CRISIS = "crisis"

@dataclass
class SlippageParams:
    """Parameters for slippage calculation"""
    base_slippage_bps: float  # Base slippage in normal conditions
    volatility_multiplier: float  # How volatility affects slippage
    volume_impact_factor: float  # Impact of order size vs volume
    liquidity_regime_multiplier: Dict[LiquidityRegime, float]  # Regime adjustments
    order_type_multiplier: Dict[OrderType, float]  # Order type adjustments
    
class SlippageEstimator:
    """
    Estimate and simulate slippage for backtesting
    """
    
    def __init__(self, params: Optional[SlippageParams] = None):
        if params is None:
            # Default parameters calibrated to US equities
            self.params = SlippageParams(
                base_slippage_bps=2.0,
                volatility_multiplier=0.3,
                volume_impact_factor=0.5,
                liquidity_regime_multiplier={
                    LiquidityRegime.HIGH: 0.5,
                    LiquidityRegime.NORMAL: 1.0,
                    LiquidityRegime.LOW: 2.0,
                    LiquidityRegime.CRISIS: 5.0
                },
                order_type_multiplier={
                    OrderType.MARKET: 1.0,
                    OrderType.LIMIT: 0.5,  # Better prices, but fill risk
                    OrderType.STOP: 1.5,  # Worse prices
                    OrderType.STOP_LIMIT: 1.2
                }
            )
        else:
            self.params = params
    
    def estimate_slippage(
        self,
        order_size: int,
        avg_daily_volume: float,
        volatility: float,
        current_spread_bps: float,
        order_type: OrderType = OrderType.MARKET,
        liquidity_regime: LiquidityRegime = LiquidityRegime.NORMAL,
        add_random_component: bool = True
    ) -> float:
        """
        Estimate slippage for an order
        
        Args:
            order_size: Number of shares
            avg_daily_volume: Average daily volume
            volatility: Daily volatility (annualized)
            current_spread_bps: Current bid-ask spread in bps
            order_type: Type of order
            liquidity_regime: Current market liquidity
            add_random_component: Add random variation
            
        Returns:
            Estimated slippage in basis points
        """
        # Base slippage
        slippage = self.params.base_slippage_bps
        
        # Volatility component
        vol_component = self.params.volatility_multiplier * volatility * 100  # Scale volatility
        slippage += vol_component
        
        # Volume impact (square root law)
        participation_rate = order_size / avg_daily_volume
        volume_impact = (
            self.params.volume_impact_factor *
            np.sqrt(participation_rate) *
            current_spread_bps
        )
        slippage += volume_impact
        
        # Spread component (cross half spread minimum)
        slippage += current_spread_bps * 0.5
        
        # Liquidity regime adjustment
        regime_mult = self.params.liquidity_regime_multiplier[liquidity_regime]
        slippage *= regime_mult
        
        # Order type adjustment
        order_mult = self.params.order_type_multiplier[order_type]
        slippage *= order_mult
        
        # Add random component (slippage is stochastic)
        if add_random_component:
            # Log-normal random variation (heavy-tailed)
            random_mult = np.random.lognormal(0, 0.5)
            slippage *= random_mult
        
        return max(0, slippage)  # Slippage must be non-negative
    
    def simulate_realistic_slippage(
        self,
        signal_price: float,
        order_size: int,
        avg_daily_volume: float,
        volatility: float,
        spread_bps: float,
        direction: int,  # 1 for buy, -1 for sell
        order_type: OrderType = OrderType.MARKET,
        fill_probability: float = 1.0
    ) -> Dict[str, float]:
        """
        Simulate realistic execution with slippage
        
        Args:
            signal_price: Price when signal generated
            order_size: Shares to trade
            avg_daily_volume: Average daily volume
            volatility: Volatility
            spread_bps: Bid-ask spread
            direction: Buy (1) or sell (-1)
            order_type: Order type
            fill_probability: Probability of fill (for limit orders)
            
        Returns:
            Dictionary with execution results
        """
        # Determine liquidity regime (simplified)
        if spread_bps < 5:
            regime = LiquidityRegime.HIGH
        elif spread_bps < 15:
            regime = LiquidityRegime.NORMAL
        elif spread_bps < 50:
            regime = LiquidityRegime.LOW
        else:
            regime = LiquidityRegime.CRISIS
        
        # Calculate slippage
        slippage_bps = self.estimate_slippage(
            order_size,
            avg_daily_volume,
            volatility,
            spread_bps,
            order_type,
            regime,
            add_random_component=True
        )
        
        # Check if order fills (important for limit orders)
        filled = np.random.random() < fill_probability
        
        if not filled:
            return {
                'filled': False,
                'execution_price': None,
                'slippage_bps': None,
                'slippage_dollars': None,
                'fill_probability': fill_probability
            }
        
        # Calculate execution price
        # Direction matters: buys have positive slippage (pay more), sells negative (receive less)
        slippage_dollars = signal_price * (slippage_bps / 10000) * direction
        execution_price = signal_price + slippage_dollars
        
        # Total cost
        total_slippage_cost = slippage_dollars * order_size
        
        return {
            'filled': True,
            'execution_price': execution_price,
            'signal_price': signal_price,
            'slippage_bps': slippage_bps,
            'slippage_dollars_per_share': slippage_dollars,
            'total_slippage_dollars': total_slippage_cost,
            'order_size': order_size,
            'direction': 'BUY' if direction > 0 else 'SELL',
            'order_type': order_type.value,
            'liquidity_regime': regime.value
        }
    
    def backtest_with_slippage(
        self,
        trades: pd.DataFrame,
        market_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply slippage to backtest trades
        
        Args:
            trades: DataFrame with columns ['timestamp', 'ticker', 'signal_price', 
                    'order_size', 'direction', 'order_type']
            market_data: DataFrame with market data for slippage calculation
            
        Returns:
            DataFrame with execution results including slippage
        """
        results = []
        
        for idx, trade in trades.iterrows():
            # Get market data for this trade
            ticker_data = market_data[
                (market_data['ticker'] == trade['ticker']) &
                (market_data['timestamp'] == trade['timestamp'])
            ]
            
            if len(ticker_data) == 0:
                # No market data, skip
                continue
            
            row = ticker_data.iloc[0]
            
            # Simulate execution
            execution = self.simulate_realistic_slippage(
                signal_price=trade['signal_price'],
                order_size=trade['order_size'],
                avg_daily_volume=row['avg_volume'],
                volatility=row['volatility'],
                spread_bps=row['spread_bps'],
                direction=trade['direction'],
                order_type=OrderType(trade.get('order_type', 'market')),
                fill_probability=0.95 if trade.get('order_type') == 'limit' else 1.0
            )
            
            execution['ticker'] = trade['ticker']
            execution['timestamp'] = trade['timestamp']
            results.append(execution)
        
        return pd.DataFrame(results)


class SlippageAnalyzer:
    """Analyze historical slippage patterns"""
    
    def __init__(self, execution_data: pd.DataFrame):
        """
        Args:
            execution_data: Historical execution data with realized slippage
        """
        self.data = execution_data.copy()
    
    def calculate_realized_slippage(self) -> pd.DataFrame:
        """
        Calculate realized slippage from execution data
        
        Expects columns: ['signal_price', 'execution_price', 'direction']
        """
        self.data['slippage_per_share'] = (
            (self.data['execution_price'] - self.data['signal_price']) *
            self.data['direction']
        )
        
        self.data['slippage_bps'] = (
            self.data['slippage_per_share'] / self.data['signal_price'] * 10000
        )
        
        return self.data
    
    def slippage_by_category(
        self, 
        category: str
    ) -> pd.DataFrame:
        """
        Analyze slippage by category (ticker, order_type, time_of_day, etc.)
        
        Args:
            category: Column name to group by
            
        Returns:
            Summary statistics by category
        """
        if 'slippage_bps' not in self.data.columns:
            self.calculate_realized_slippage()
        
        summary = self.data.groupby(category).agg({
            'slippage_bps': ['mean', 'median', 'std', 'min', 'max'],
            'order_size': 'sum',
            'execution_price': 'count'
        }).round(2)
        
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary.rename(columns={'execution_price_count': 'num_trades'}, inplace=True)
        
        return summary
    
    def intraday_slippage_pattern(self) -> pd.DataFrame:
        """Analyze slippage patterns by time of day"""
        if 'slippage_bps' not in self.data.columns:
            self.calculate_realized_slippage()
        
        self.data['hour'] = pd.to_datetime(self.data['timestamp']).dt.hour
        self.data['minute'] = pd.to_datetime(self.data['timestamp']).dt.minute
        
        hourly = self.data.groupby('hour')['slippage_bps'].agg([
            'mean', 'median', 'std', 'count'
        ]).round(2)
        
        return hourly
    
    def cost_attribution(self) -> Dict[str, float]:
        """
        Break down total slippage costs
        
        Returns:
            Dictionary with cost attribution
        """
        if 'slippage_per_share' not in self.data.columns:
            self.calculate_realized_slippage()
        
        self.data['total_slippage_cost'] = (
            self.data['slippage_per_share'] * self.data['order_size']
        )
        
        total_cost = self.data['total_slippage_cost'].sum()
        
        return {
            'total_slippage_cost': total_cost,
            'average_slippage_bps': self.data['slippage_bps'].mean(),
            'median_slippage_bps': self.data['slippage_bps'].median(),
            'worst_slippage_bps': self.data['slippage_bps'].max(),
            'best_slippage_bps': self.data['slippage_bps'].min(),
            'total_trades': len(self.data),
            'cost_per_trade': total_cost / len(self.data),
        }


# Example: Backtest with realistic slippage
def example_backtest_with_slippage():
    """Example of backtesting with slippage modeling"""
    np.random.seed(42)
    
    # Generate synthetic trades
    n_trades = 100
    dates = pd.date_range('2024-01-01', periods=n_trades, freq='1H')
    
    trades = pd.DataFrame({
        'timestamp': dates,
        'ticker': np.random.choice(['AAPL', 'MSFT', 'GOOGL'], n_trades),
        'signal_price': 100 + np.random.randn(n_trades) * 10,
        'order_size': np.random.randint(100, 10000, n_trades),
        'direction': np.random.choice([1, -1], n_trades),
        'order_type': np.random.choice(['market', 'limit'], n_trades)
    })
    
    # Generate market data
    market_data = pd.DataFrame({
        'timestamp': dates,
        'ticker': trades['ticker'],
        'avg_volume': 5000000,
        'volatility': 0.25,
        'spread_bps': np.random.uniform(2, 10, n_trades)
    })
    
    # Apply slippage
    estimator = SlippageEstimator()
    results = estimator.backtest_with_slippage(trades, market_data)
    
    # Analyze results
    print("Slippage Analysis:")
    print(f"Total trades: {len(results)}")
    print(f"Filled trades: {results['filled'].sum()}")
    print(f"\\nAverage slippage: {results[results['filled']]['slippage_bps'].mean():.2f} bps")
    print(f"Median slippage: {results[results['filled']]['slippage_bps'].median():.2f} bps")
    print(f"Total slippage cost: \${results[results['filled']]['total_slippage_dollars'].sum():,.2f}")

return results


if __name__ == "__main__":
    results = example_backtest_with_slippage()
\`\`\`

## Production Implementation

### 1. **Slippage Monitoring**

Track realized vs expected slippage:

\`\`\`python
class SlippageMonitor:
    """Monitor slippage in production"""
    
    def __init__(self):
        self.trades: List[Dict] = []
    
    def record_trade(
        self,
        signal_price: float,
        execution_price: float,
        expected_slippage: float,
        order_size: int,
        direction: int
    ):
        """Record trade execution"""
        realized_slippage = (
            (execution_price - signal_price) / signal_price * 10000 * direction
        )
        
        self.trades.append({
            'realized_slippage_bps': realized_slippage,
            'expected_slippage_bps': expected_slippage,
            'error_bps': realized_slippage - expected_slippage,
            'order_size': order_size
        })
    
    def get_model_accuracy(self) -> Dict[str, float]:
        """Evaluate slippage model accuracy"""
        df = pd.DataFrame(self.trades)
        
        return {
            'mean_error_bps': df['error_bps'].mean(),
            'rmse_bps': np.sqrt((df['error_bps'] ** 2).mean()),
            'correlation': df['realized_slippage_bps'].corr(
                df['expected_slippage_bps']
            )
        }
\`\`\`

### 2. **Adaptive Slippage Models**

Update parameters based on realized slippage:
- Weekly recalibration
- Regime-dependent parameters
- Ticker-specific adjustments

### 3. **Risk Controls**

Implement safeguards:
- Maximum slippage thresholds
- Cancel orders if slippage exceeds limit
- Dynamic order type selection
- Size limits based on liquidity

## Summary

Accurate slippage modeling is essential for:
- Realistic backtest performance
- Strategy validation
- Risk management
- Cost optimization

Production systems must continuously calibrate slippage models against realized execution costs.
`,
    },
    {
      title: 'Production Implementation and Best Practices',
      content: `
# Production Implementation and Best Practices

Implementing transaction cost and slippage models in production requires careful engineering, monitoring, and continuous calibration.

## System Architecture

### Transaction Cost Analysis (TCA) Pipeline

\`\`\`python
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np
from enum import Enum
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExecutionVenue(Enum):
    """Execution venues"""
    NASDAQ = "nasdaq"
    NYSE = "nyse"
    BATS = "bats"
    IEX = "iex"
    DARK_POOL = "dark_pool"

@dataclass
class ExecutionReport:
    """Detailed execution report"""
    order_id: str
    timestamp: datetime
    ticker: str
    side: str  # 'BUY' or 'SELL'
    order_size: int
    signal_price: float
    execution_price: float
    venue: ExecutionVenue
    commission: float
    exchange_fee: float
    sec_fee: float
    spread_at_execution: float
    volume_at_execution: float
    volatility_at_execution: float
    execution_time_ms: float
    
    # Calculated fields
    slippage_bps: float = field(init=False)
    total_cost: float = field(init=False)
    market_impact_bps: float = field(init=False)
    
    def __post_init__(self):
        """Calculate derived metrics"""
        direction = 1 if self.side == 'BUY' else -1
        
        # Slippage
        self.slippage_bps = (
            (self.execution_price - self.signal_price) / 
            self.signal_price * 10000 * direction
        )
        
        # Total cost
        self.total_cost = (
            self.commission + 
            self.exchange_fee + 
            self.sec_fee +
            abs(self.execution_price - self.signal_price) * self.order_size
        )
        
        # Market impact (simplified)
        self.market_impact_bps = max(0, self.slippage_bps - self.spread_at_execution * 0.5)

class TransactionCostAnalyzer:
    """
    Production-grade Transaction Cost Analysis system
    """
    
    def __init__(
        self,
        cost_structure: CostStructure,
        spread_estimator: SpreadEstimator,
        impact_calculator: MarketImpactCalculator,
        slippage_estimator: SlippageEstimator
    ):
        self.cost_structure = cost_structure
        self.spread_estimator = spread_estimator
        self.impact_calculator = impact_calculator
        self.slippage_estimator = slippage_estimator
        
        # Storage for execution history
        self.execution_history: List[ExecutionReport] = []
        
        # Performance tracking
        self.metrics_cache: Dict[str, float] = {}
    
    def pre_trade_analysis(
        self,
        ticker: str,
        order_size: int,
        current_price: float,
        market_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Pre-trade cost estimation
        
        Args:
            ticker: Stock ticker
            order_size: Planned order size
            current_price: Current market price
            market_data: Recent market data for ticker
            
        Returns:
            Dictionary with cost estimates
        """
        logger.info(f"Pre-trade analysis for {ticker}: {order_size} shares")
        
        # Calculate market characteristics
        avg_volume = market_data['volume'].mean()
        volatility = market_data['returns'].std() * np.sqrt(252)
        
        # Estimate spread
        spread_analysis = self.spread_estimator.get_comprehensive_analysis()
        current_spread_bps = spread_analysis['effective_spread_bps'].iloc[-1]
        
        # Estimate market impact
        impact = self.impact_calculator.calculate_impact(
            order_size=order_size,
            avg_daily_volume=avg_volume,
            current_price=current_price,
            volatility=volatility
        )
        
        # Estimate slippage
        expected_slippage = self.slippage_estimator.estimate_slippage(
            order_size=order_size,
            avg_daily_volume=avg_volume,
            volatility=volatility,
            current_spread_bps=current_spread_bps
        )
        
        # Total cost estimate
        explicit_costs = self.cost_structure.total_cost(
            order_size, current_price
        )
        
        total_estimated_cost = (
            explicit_costs +
            impact['total_impact_dollars'] +
            (expected_slippage / 10000) * current_price * order_size
        )
        
        cost_bps = (total_estimated_cost / (order_size * current_price)) * 10000
        
        return {
            'ticker': ticker,
            'order_size': order_size,
            'estimated_explicit_cost': explicit_costs,
            'estimated_impact_cost': impact['total_impact_dollars'],
            'estimated_slippage_cost': (expected_slippage / 10000) * current_price * order_size,
            'total_estimated_cost': total_estimated_cost,
            'cost_bps': cost_bps,
            'participation_rate': order_size / avg_volume,
            'current_spread_bps': current_spread_bps,
            'volatility': volatility,
            'recommendation': self._generate_recommendation(cost_bps, order_size / avg_volume)
        }
    
    def _generate_recommendation(
        self, 
        cost_bps: float, 
        participation_rate: float
    ) -> str:
        """Generate execution recommendation"""
        if cost_bps < 5 and participation_rate < 0.01:
            return "LOW_COST: Execute immediately with market order"
        elif cost_bps < 15 and participation_rate < 0.05:
            return "MODERATE_COST: Consider limit order or VWAP"
        elif cost_bps < 30:
            return "HIGH_COST: Use TWAP or split into multiple orders"
        else:
            return "VERY_HIGH_COST: Reconsider trade size or use algorithmic execution over extended period"
    
    def post_trade_analysis(
        self,
        execution: ExecutionReport
    ) -> Dict[str, float]:
        """
        Post-trade cost analysis
        
        Args:
            execution: Execution report
            
        Returns:
            Analysis of execution quality
        """
        logger.info(f"Post-trade analysis for order {execution.order_id}")
        
        # Store execution
        self.execution_history.append(execution)
        
        # Calculate components
        explicit_cost = (
            execution.commission +
            execution.exchange_fee +
            execution.sec_fee
        )
        
        explicit_cost_bps = (
            explicit_cost / (execution.order_size * execution.signal_price) * 10000
        )
        
        # Decompose slippage into components
        spread_cost_bps = execution.spread_at_execution * 0.5
        impact_bps = execution.market_impact_bps
        unexplained_slippage_bps = execution.slippage_bps - spread_cost_bps - impact_bps
        
        return {
            'order_id': execution.order_id,
            'ticker': execution.ticker,
            'explicit_cost_bps': explicit_cost_bps,
            'spread_cost_bps': spread_cost_bps,
            'market_impact_bps': impact_bps,
            'unexplained_slippage_bps': unexplained_slippage_bps,
            'total_cost_bps': execution.slippage_bps + explicit_cost_bps,
            'execution_time_ms': execution.execution_time_ms,
            'venue': execution.venue.value
        }
    
    def benchmark_execution(
        self,
        execution: ExecutionReport,
        benchmark: str = "VWAP"
    ) -> Dict[str, float]:
        """
        Compare execution against benchmark
        
        Args:
            execution: Execution report
            benchmark: Benchmark to compare against (VWAP, TWAP, Arrival)
            
        Returns:
            Benchmark comparison metrics
        """
        # This would fetch actual benchmark prices from market data
        # For now, simplified implementation
        
        if benchmark == "VWAP":
            # Compare to volume-weighted average price
            # In production, calculate from actual market data
            vwap = execution.signal_price  # Simplified
            implementation_shortfall = (
                (execution.execution_price - vwap) /
                vwap * 10000 *
                (1 if execution.side == 'BUY' else -1)
            )
        elif benchmark == "ARRIVAL":
            # Compare to price at decision time
            implementation_shortfall = execution.slippage_bps
        else:
            implementation_shortfall = 0.0
        
        return {
            'benchmark': benchmark,
            'implementation_shortfall_bps': implementation_shortfall,
            'execution_quality': 'GOOD' if implementation_shortfall < 10 else 'POOR'
        }
    
    def generate_tca_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Generate comprehensive TCA report
        
        Args:
            start_date: Start of analysis period
            end_date: End of analysis period
            
        Returns:
            DataFrame with TCA metrics
        """
        # Filter executions in date range
        executions = [
            e for e in self.execution_history
            if start_date <= e.timestamp <= end_date
        ]
        
        if not executions:
            return pd.DataFrame()
        
        # Analyze each execution
        analyses = []
        for execution in executions:
            post_trade = self.post_trade_analysis(execution)
            benchmark = self.benchmark_execution(execution)
            
            analyses.append({
                **post_trade,
                **benchmark,
                'timestamp': execution.timestamp,
                'order_size': execution.order_size,
                'side': execution.side
            })
        
        df = pd.DataFrame(analyses)
        
        # Add summary statistics
        logger.info(f"\\nTCA Report ({start_date} to {end_date}):")
        logger.info(f"Total executions: {len(df)}")
        logger.info(f"Average total cost: {df['total_cost_bps'].mean():.2f} bps")
        logger.info(f"Median total cost: {df['total_cost_bps'].median():.2f} bps")
        logger.info(f"Best execution: {df['total_cost_bps'].min():.2f} bps")
        logger.info(f"Worst execution: {df['total_cost_bps'].max():.2f} bps")
        
        return df
    
    def calibrate_models(
        self,
        lookback_days: int = 30
    ):
        """
        Calibrate cost models based on recent execution history
        
        Args:
            lookback_days: Number of days to use for calibration
        """
        logger.info(f"Calibrating models using last {lookback_days} days of data")
        
        # Get recent executions
        cutoff = datetime.now() - pd.Timedelta(days=lookback_days)
        recent = [e for e in self.execution_history if e.timestamp >= cutoff]
        
        if len(recent) < 10:
            logger.warning("Insufficient data for calibration")
            return
        
        # Analyze realized vs predicted costs
        df = pd.DataFrame([{
            'realized_slippage': e.slippage_bps,
            'order_size': e.order_size,
            'volatility': e.volatility_at_execution,
            'spread': e.spread_at_execution
        } for e in recent])
        
        # Update slippage model parameters (simplified)
        realized_mean = df['realized_slippage'].mean()
        
        # Adjust base slippage
        adjustment_factor = realized_mean / self.slippage_estimator.params.base_slippage_bps
        
        logger.info(f"Adjustment factor: {adjustment_factor:.2f}")
        logger.info(f"Old base slippage: {self.slippage_estimator.params.base_slippage_bps:.2f} bps")
        
        # Apply 20% of adjustment (don't overreact to short-term data)
        self.slippage_estimator.params.base_slippage_bps *= (1 + (adjustment_factor - 1) * 0.2)
        
        logger.info(f"New base slippage: {self.slippage_estimator.params.base_slippage_bps:.2f} bps")


# Example: Production TCA system
if __name__ == "__main__":
    # Initialize components
    cost_structure = get_cost_structure(
        TradingFrequency.MEDIUM_FREQUENCY,
        broker_tier="professional"
    )
    
    # Create mock market data
    dates = pd.date_range('2024-01-01', periods=100, freq='1min')
    market_data = pd.DataFrame({
        'timestamp': dates,
        'price': 100 + np.cumsum(np.random.randn(100) * 0.1),
        'volume': np.random.randint(1000, 10000, 100),
        'bid': 99.95,
        'ask': 100.05
    })
    market_data['returns'] = market_data['price'].pct_change()
    
    spread_estimator = SpreadEstimator(market_data)
    impact_calculator = MarketImpactCalculator()
    slippage_estimator = SlippageEstimator()
    
    # Initialize TCA system
    tca = TransactionCostAnalyzer(
        cost_structure,
        spread_estimator,
        impact_calculator,
        slippage_estimator
    )
    
    # Pre-trade analysis
    pre_trade = tca.pre_trade_analysis(
        ticker='AAPL',
        order_size=5000,
        current_price=100.0,
        market_data=market_data
    )
    
    print("\\nPre-Trade Analysis:")
    for key, value in pre_trade.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # Simulate execution
    execution = ExecutionReport(
        order_id='ORD-001',
        timestamp=datetime.now(),
        ticker='AAPL',
        side='BUY',
        order_size=5000,
        signal_price=100.0,
        execution_price=100.15,
        venue=ExecutionVenue.NASDAQ,
        commission=25.0,
        exchange_fee=0.60,
        sec_fee=0.26,
        spread_at_execution=5.0,
        volume_at_execution=50000,
        volatility_at_execution=0.25,
        execution_time_ms=250.0
    )
    
    # Post-trade analysis
    post_trade = tca.post_trade_analysis(execution)
    
    print("\\nPost-Trade Analysis:")
    for key, value in post_trade.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
\`\`\`

## Key Takeaways

1. **Transaction costs are real and material** - they can turn profitable strategies into losers
2. **Model all components**: commissions, spreads, market impact, and slippage
3. **Use realistic assumptions**: base on historical data and broker reports
4. **Calibrate continuously**: update models based on realized costs
5. **Monitor in production**: track predicted vs actual costs
6. **Optimize execution**: use appropriate order types and algorithms

## Hands-On Exercise

Implement a backtest comparison:
1. Run a strategy with zero transaction costs
2. Run the same strategy with realistic costs
3. Compare performance metrics
4. Identify break-even trading frequency

## Summary

Accurate transaction cost modeling is essential for realistic backtesting and strategy evaluation. Production systems must integrate pre-trade analysis, post-trade measurement, and continuous model calibration to ensure strategies remain profitable in live trading.
`,
    },
  ],
};

export default transactionCostsAndSlippage;
