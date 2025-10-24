export const tradingStrategyDevelopment = {
  title: 'Trading Strategy Development',
  id: 'trading-strategy-development',
  content: `
# Trading Strategy Development

## Introduction

Trading strategies are the engine room of quantitative finance—they convert market insights, predictions, and theories into concrete buy/sell decisions that generate returns. A well-designed trading strategy has:

1. **Clear Logic**: Transparent entry/exit rules
2. **Risk Management**: Position sizing, stops, limits
3. **Robustness**: Works across market conditions
4. **Scalability**: Can handle different capital levels
5. **Measurability**: Trackable performance metrics

**The Strategy Development Pipeline**:
\`\`\`
Hypothesis → Signal Generation → Position Sizing → Risk Controls → Execution → Evaluation
\`\`\`

This section covers the full spectrum from basic momentum strategies to advanced multi-factor ML-based systems.

---

## Strategy Classification

### By Logic Type

**1. Trend Following / Momentum**
- Buy winners, sell losers
- Works in trending markets
- Example: Moving average crossover

**2. Mean Reversion**
- Buy oversold, sell overbought
- Works in range-bound markets
- Example: Bollinger Bands, RSI

**3. Arbitrage**
- Exploit mispricings between related assets
- Market-neutral
- Example: Statistical arbitrage, pairs trading

**4. Factor-Based**
- Trade based on fundamental/statistical factors
- Example: Value, quality, low volatility

**5. Machine Learning**
- Use ML predictions for signals
- Example: Random Forest, LSTM predictions

**6. Market Making**
- Provide liquidity, capture spread
- High frequency
- Example: Bid-ask spread capture

---

## Base Strategy Framework

\`\`\`python
"""
Comprehensive Trading Strategy Framework
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Trade:
    """Individual trade record"""
    timestamp: datetime
    ticker: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    commission: float = 0.0
    
    @property
    def value(self):
        return self.quantity * self.price
    
    @property
    def total_cost(self):
        return self.value + self.commission


@dataclass
class Position:
    """Current position in an asset"""
    ticker: str
    quantity: float
    avg_entry_price: float
    current_price: float
    
    @property
    def market_value(self):
        return self.quantity * self.current_price
    
    @property
    def unrealized_pnl(self):
        return (self.current_price - self.avg_entry_price) * self.quantity
    
    @property
    def return_pct(self):
        if self.avg_entry_price == 0:
            return 0
        return (self.current_price - self.avg_entry_price) / self.avg_entry_price


class TradingStrategy(ABC):
    """
    Base class for all trading strategies
    
    Implement this abstract class to create custom strategies
    """
    
    def __init__(self, name: str, initial_capital: float = 100000):
        self.name = name
        self.initial_capital = initial_capital
        self.capital = initial_capital
        
        # Track everything
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [initial_capital]
        self.timestamps: List[datetime] = []
        
        # Performance tracking
        self.metrics = {}
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals from market data
        
        Args:
            data: DataFrame with OHLCV + features
        
        Returns:
            Series with signals: 1 (buy), -1 (sell), 0 (hold)
        """
        pass
    
    @abstractmethod
    def size_position(self, signal: float, ticker: str, price: float) -> float:
        """
        Determine position size for a signal
        
        Args:
            signal: Trading signal strength
            ticker: Asset ticker
            price: Current price
        
        Returns:
            Quantity to trade (positive = buy, negative = sell)
        """
        pass
    
    def execute_trade(self, ticker: str, quantity: float, price: float, 
                     timestamp: datetime, commission_rate: float = 0.001):
        """
        Execute a trade and update portfolio
        
        Args:
            ticker: Asset ticker
            quantity: Shares to trade (positive = buy, negative = sell)
            price: Execution price
            timestamp: Trade timestamp
            commission_rate: Commission as % of trade value
        """
        side = 'BUY' if quantity > 0 else 'SELL'
        value = abs(quantity * price)
        commission = value * commission_rate
        
        # Create trade record
        trade = Trade(
            timestamp=timestamp,
            ticker=ticker,
            side=side,
            quantity=abs(quantity),
            price=price,
            commission=commission
        )
        self.trades.append(trade)
        
        # Update capital
        if quantity > 0:  # Buy
            total_cost = value + commission
            if total_cost > self.capital:
                raise ValueError(f"Insufficient capital: need {total_cost}, have {self.capital}")
            self.capital -= total_cost
        else:  # Sell
            self.capital += value - commission
        
        # Update position
        if ticker not in self.positions:
            self.positions[ticker] = Position(ticker, 0, 0, price)
        
        position = self.positions[ticker]
        
        if quantity > 0:  # Buy
            # Update average entry price
            total_cost = position.quantity * position.avg_entry_price + value
            new_quantity = position.quantity + quantity
            position.avg_entry_price = total_cost / new_quantity if new_quantity > 0 else 0
            position.quantity = new_quantity
        else:  # Sell
            position.quantity += quantity  # quantity is negative
        
        position.current_price = price
        
        # Remove zero positions
        if abs(position.quantity) < 1e-6:
            del self.positions[ticker]
    
    def update_positions(self, prices: Dict[str, float]):
        """Update position values with current prices"""
        for ticker, position in self.positions.items():
            if ticker in prices:
                position.current_price = prices[ticker]
    
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.capital + positions_value
    
    def run_backtest(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Run strategy backtest
        
        Args:
            data: Dict of ticker -> OHLCV DataFrame
        
        Returns:
            DataFrame with performance metrics
        """
        # Get common date range
        all_dates = set()
        for df in data.values():
            all_dates.update(df.index)
        dates = sorted(all_dates)
        
        for date in dates:
            # Get current prices
            prices = {}
            for ticker, df in data.items():
                if date in df.index:
                    prices[ticker] = df.loc[date, 'Close']
            
            # Update existing positions
            self.update_positions(prices)
            
            # Generate signals for each asset
            for ticker, df in data.items():
                if date not in df.index:
                    continue
                
                # Get data up to current date
                historical_data = df.loc[:date]
                
                # Generate signal
                signals = self.generate_signals(historical_data)
                if len(signals) == 0:
                    continue
                
                signal = signals.iloc[-1]
                price = prices[ticker]
                
                # Determine position size
                target_quantity = self.size_position(signal, ticker, price)
                
                # Calculate current quantity
                current_quantity = self.positions[ticker].quantity if ticker in self.positions else 0
                
                # Execute trade if position change needed
                quantity_to_trade = target_quantity - current_quantity
                
                if abs(quantity_to_trade) > 0:
                    try:
                        self.execute_trade(ticker, quantity_to_trade, price, date)
                    except ValueError:
                        # Insufficient capital - skip trade
                        pass
            
            # Record portfolio value
            portfolio_value = self.get_portfolio_value()
            self.equity_curve.append(portfolio_value)
            self.timestamps.append(date)
        
        # Calculate returns
        equity_series = pd.Series(self.equity_curve, index=[None] + self.timestamps)
        returns = equity_series.pct_change().dropna()
        
        return pd.DataFrame({
            'equity': equity_series.dropna(),
            'returns': returns
        })


# ============================================================================
# 1. MOMENTUM STRATEGY
# ============================================================================

class MomentumStrategy(TradingStrategy):
    """
    Classic momentum strategy: Buy recent winners, sell recent losers
    
    Theory: Assets with strong recent performance continue to perform well
    Works best: Trending markets
    """
    
    def __init__(self, lookback: int = 21, holding_period: int = 5, 
                 top_pct: float = 0.2, initial_capital: float = 100000):
        super().__init__("Momentum", initial_capital)
        self.lookback = lookback
        self.holding_period = holding_period
        self.top_pct = top_pct
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Signal = past N-day return
        
        Positive signal = upward momentum
        """
        if len(data) < self.lookback:
            return pd.Series(0, index=data.index)
        
        momentum = data['Close'].pct_change(self.lookback)
        return momentum
    
    def size_position(self, signal: float, ticker: str, price: float) -> float:
        """
        Equal weight top 20% momentum stocks
        """
        if signal > 0:
            # Allocate 10% of capital per position (max 10 positions)
            target_value = self.initial_capital * 0.1
            return target_value / price
        else:
            return 0


class CrossSectionalMomentumStrategy(TradingStrategy):
    """
    Cross-sectional momentum: Rank assets relative to each other
    
    Long top performers, short bottom performers
    Market neutral strategy
    """
    
    def __init__(self, lookback: int = 21, n_long: int = 5, n_short: int = 5,
                 initial_capital: float = 100000):
        super().__init__("Cross-Sectional Momentum", initial_capital)
        self.lookback = lookback
        self.n_long = n_long
        self.n_short = n_short
        self.tickers = []
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Generate signals for multiple assets"""
        momentums = {}
        for ticker, df in data.items():
            if len(df) >= self.lookback:
                mom = df['Close'].iloc[-1] / df['Close'].iloc[-self.lookback] - 1
                momentums[ticker] = mom
        
        # Rank
        sorted_tickers = sorted(momentums.items(), key=lambda x: x[1], reverse=True)
        
        signals = {}
        # Long top N
        for ticker, _ in sorted_tickers[:self.n_long]:
            signals[ticker] = 1
        
        # Short bottom N
        for ticker, _ in sorted_tickers[-self.n_short:]:
            signals[ticker] = -1
        
        return signals


# ============================================================================
# 2. MEAN REVERSION STRATEGY
# ============================================================================

class MeanReversionStrategy(TradingStrategy):
    """
    Mean reversion: Buy oversold, sell overbought
    
    Theory: Prices revert to their mean over time
    Works best: Range-bound, sideways markets
    """
    
    def __init__(self, window: int = 20, std_dev: float = 2.0,
                 initial_capital: float = 100000):
        super().__init__("Mean Reversion", initial_capital)
        self.window = window
        self.std_dev = std_dev
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Z-score signals:
        < -2: Oversold (buy)
        > +2: Overbought (sell)
        """
        if len(data) < self.window:
            return pd.Series(0, index=data.index)
        
        sma = data['Close'].rolling(self.window).mean()
        std = data['Close'].rolling(self.window).std()
        z_score = (data['Close'] - sma) / std
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        signals[z_score < -self.std_dev] = 1  # Buy oversold
        signals[z_score > self.std_dev] = -1  # Sell overbought
        
        return signals
    
    def size_position(self, signal: float, ticker: str, price: float) -> float:
        """Fixed position size"""
        if signal == 1:  # Buy
            target_value = self.initial_capital * 0.2  # 20% per position
            return target_value / price
        elif signal == -1:  # Sell
            return 0
        return 0


class BollingerBandStrategy(TradingStrategy):
    """
    Bollinger Bands mean reversion
    
    Buy when price touches lower band
    Sell when price touches upper band
    """
    
    def __init__(self, window: int = 20, num_std: float = 2.0,
                 initial_capital: float = 100000):
        super().__init__("Bollinger Bands", initial_capital)
        self.window = window
        self.num_std = num_std
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Bollinger Band signals"""
        if len(data) < self.window:
            return pd.Series(0, index=data.index)
        
        sma = data['Close'].rolling(self.window).mean()
        std = data['Close'].rolling(self.window).std()
        
        upper_band = sma + self.num_std * std
        lower_band = sma - self.num_std * std
        
        signals = pd.Series(0, index=data.index)
        signals[data['Close'] < lower_band] = 1  # Buy at lower band
        signals[data['Close'] > upper_band] = -1  # Sell at upper band
        
        return signals
    
    def size_position(self, signal: float, ticker: str, price: float) -> float:
        """Fixed position size"""
        if signal != 0:
            target_value = self.initial_capital * 0.25
            return target_value / price * signal
        return 0


# ============================================================================
# 3. PAIRS TRADING (STATISTICAL ARBITRAGE)
# ============================================================================

class PairsTradingStrategy:
    """
    Pairs trading: Trade spread between correlated assets
    
    Theory: Correlated assets revert to historical relationship
    Market neutral
    """
    
    def __init__(self, asset1: str, asset2: str, window: int = 20,
                 entry_threshold: float = 2.0, exit_threshold: float = 0.5,
                 initial_capital: float = 100000):
        self.asset1 = asset1
        self.asset2 = asset2
        self.window = window
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.initial_capital = initial_capital
        self.capital = initial_capital
        
        self.position = None  # 'long_spread' or 'short_spread'
    
    def calculate_spread(self, data1: pd.DataFrame, data2: pd.DataFrame) -> pd.Series:
        """Calculate price spread"""
        # Log prices for better stationarity
        log_p1 = np.log(data1['Close'])
        log_p2 = np.log(data2['Close'])
        
        spread = log_p1 - log_p2
        return spread
    
    def generate_signals(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Tuple[float, float]:
        """
        Generate signals for both assets
        
        Returns:
            (signal1, signal2): Signals for both assets
        """
        spread = self.calculate_spread(data1, data2)
        
        if len(spread) < self.window:
            return 0, 0
        
        # Z-score of spread
        mean = spread.rolling(self.window).mean()
        std = spread.rolling(self.window).std()
        z_score = (spread - mean) / std
        
        current_z = z_score.iloc[-1]
        
        # Entry signals
        if self.position is None:
            if current_z > self.entry_threshold:
                # Spread too high: short spread (short asset1, long asset2)
                self.position = 'short_spread'
                return -1, 1
            elif current_z < -self.entry_threshold:
                # Spread too low: long spread (long asset1, short asset2)
                self.position = 'long_spread'
                return 1, -1
        
        # Exit signals
        elif abs(current_z) < self.exit_threshold:
            # Close position
            if self.position == 'short_spread':
                self.position = None
                return 1, -1  # Close shorts
            elif self.position == 'long_spread':
                self.position = None
                return -1, 1  # Close longs
        
        return 0, 0


# ============================================================================
# SIGNAL PROCESSING & COMBINATION
# ============================================================================

class SignalProcessor:
    """
    Advanced signal processing and filtering
    """
    
    @staticmethod
    def smooth_signals(signals: pd.Series, window: int = 5) -> pd.Series:
        """
        Smooth noisy signals using moving average
        
        Reduces whipsaws and false signals
        """
        return signals.rolling(window, min_periods=1).mean()
    
    @staticmethod
    def filter_by_volatility(signals: pd.Series, data: pd.DataFrame,
                            vol_threshold: float = 0.02,
                            vol_window: int = 21) -> pd.Series:
        """
        Suppress signals during high volatility
        
        High vol = higher risk, less predictable
        """
        returns = data['Close'].pct_change()
        volatility = returns.rolling(vol_window).std()
        
        filtered = signals.copy()
        filtered[volatility > vol_threshold] = 0
        
        return filtered
    
    @staticmethod
    def filter_by_volume(signals: pd.Series, data: pd.DataFrame,
                        volume_percentile: float = 0.3,
                        window: int = 21) -> pd.Series:
        """
        Only trade when volume is above threshold
        
        High volume = more liquidity, better fills
        """
        volume_threshold = data['Volume'].rolling(window).quantile(volume_percentile)
        
        filtered = signals.copy()
        filtered[data['Volume'] < volume_threshold] = 0
        
        return filtered
    
    @staticmethod
    def filter_by_trend(signals: pd.Series, data: pd.DataFrame,
                       ma_period: int = 50) -> pd.Series:
        """
        Align signals with longer-term trend
        
        Only long when above MA, only short when below MA
        """
        ma = data['Close'].rolling(ma_period).mean()
        
        filtered = signals.copy()
        # Suppress long signals when price below MA
        filtered[(signals > 0) & (data['Close'] < ma)] = 0
        # Suppress short signals when price above MA
        filtered[(signals < 0) & (data['Close'] > ma)] = 0
        
        return filtered
    
    @staticmethod
    def combine_signals(signals_list: List[pd.Series],
                       weights: Optional[List[float]] = None,
                       method: str = 'weighted_average') -> pd.Series:
        """
        Ensemble multiple signals
        
        Methods:
        - weighted_average: Weight and average
        - majority_vote: Take sign of majority
        - unanimous: Only trade when all agree
        """
        if weights is None:
            weights = [1.0 / len(signals_list)] * len(signals_list)
        
        if method == 'weighted_average':
            combined = sum(w * s for w, s in zip(weights, signals_list))
            return combined
        
        elif method == 'majority_vote':
            signs = pd.DataFrame({i: np.sign(s) for i, s in enumerate(signals_list)})
            majority = signs.sum(axis=1)
            return np.sign(majority)
        
        elif method == 'unanimous':
            signs = pd.DataFrame({i: np.sign(s) for i, s in enumerate(signals_list)})
            all_long = (signs == 1).all(axis=1)
            all_short = (signs == -1).all(axis=1)
            
            result = pd.Series(0, index=signs.index)
            result[all_long] = 1
            result[all_short] = -1
            return result
        
        return combined


# ============================================================================
# EXAMPLE: COMPLETE STRATEGY WITH FILTERING
# ============================================================================

# Download data
import yfinance as yf

data = yf.download('AAPL', start='2020-01-01', end='2024-01-01')

# Generate raw signals (momentum + mean reversion)
momentum_strategy = MomentumStrategy(lookback=21)
momentum_signals = momentum_strategy.generate_signals(data)

mr_strategy = MeanReversionStrategy(window=20, std_dev=2)
mr_signals = mr_strategy.generate_signals(data)

# Combine signals
combined = SignalProcessor.combine_signals(
    [momentum_signals, mr_signals],
    weights=[0.6, 0.4]
)

# Apply filters
filtered = SignalProcessor.filter_by_volatility(combined, data, vol_threshold=0.025)
filtered = SignalProcessor.filter_by_volume(filtered, data, volume_percentile=0.3)
filtered = SignalProcessor.filter_by_trend(filtered, data, ma_period=50)

print(f"Raw signals: {(combined != 0).sum()}")
print(f"Filtered signals: {(filtered != 0).sum()}")
print(f"Filter reduction: {1 - (filtered != 0).sum() / (combined != 0).sum():.1%}")

---

## Key Takeaways

**Strategy Types**:
- **Momentum**: Trend-following, works in trending markets
- **Mean Reversion**: Fade extremes, works in range-bound markets
- **Arbitrage**: Market-neutral, exploit mispricings
- **Multi-Factor**: Combine multiple signals

**Signal Processing**:
- **Smooth**: Reduce noise
- **Filter**: Volatility, volume, trend alignment
- **Combine**: Ensemble multiple strategies

**Best Practices**:
1. Start simple, add complexity gradually
2. Always use stop losses
3. Validate on out-of-sample data
4. Monitor live vs backtest performance
5. Have clear entry/exit rules
6. Document everything

**Common Pitfalls**:
- Over-optimization (curve fitting)
- Ignoring transaction costs
- Not accounting for slippage
- Overly complex strategies (overfit)
- No risk management
`,
};
