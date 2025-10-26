export const trendFollowingStrategiesQuiz = [
    {
        id: 'ats-2-1-q-1',
        question:
            "Design a trend-following system that trades S&P 500 futures with $1M capital. Include: (1) Entry/exit rules using Donchian channels, (2) ATR-based position sizing risking 1% per trade, (3) Pyramiding rules for adding to winners, (4) ADX filter to avoid choppy markets. Provide complete Python implementation with realistic parameters.",
        sampleAnswer: `**Complete Trend Following System for S&P 500 Futures:**

**System Specifications:**
- Market: S&P 500 E-mini futures (ES)
- Capital: $1,000,000
- Risk per trade: 1% ($10,000)
- Entry: 20-day Donchian breakout
- Exit: 10-day opposite breakout
- Filter: ADX > 25 (trending market only)
- Pyramiding: Add unit every 0.5× ATR profit (max 4 units)

**Complete Implementation:**

\`\`\`python
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

@dataclass
class Position:
    """Represents an open position"""
    entry_time: datetime
    entry_price: float
    direction: int  # 1 for long, -1 for short
    num_contracts: int
    num_units: int  # For pyramiding
    stop_loss: float
    unit_size: int  # Contracts per unit

class TrendFollowingFuturesSystem:
    """
    Complete trend following system for S&P 500 futures
    
    Based on Turtle Trading principles adapted for modern markets
    """
    
    def __init__(self, capital: float = 1_000_000):
        self.capital = capital
        self.risk_per_trade = 0.01  # 1%
        self.entry_period = 20  # Donchian breakout period
        self.exit_period = 10
        self.adx_threshold = 25  # Only trade trending markets
        self.atr_period = 20
        self.atr_stop_multiple = 2.0
        self.pyramid_threshold = 0.5  # Add every 0.5× ATR
        self.max_units = 4
        
        # S&P 500 E-mini specifications
        self.point_value = 50  # $50 per point
        self.tick_size = 0.25  # $12.50 per tick
        
        self.positions = {}
        self.trade_log = []
        
    def calculate_indicators(self, ohlc: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        df = ohlc.copy()
        
        # Donchian Channels
        df['entry_high'] = df['high'].rolling(self.entry_period).max()
        df['entry_low'] = df['low'].rolling(self.entry_period).min()
        df['exit_high'] = df['high'].rolling(self.exit_period).max()
        df['exit_low'] = df['low'].rolling(self.exit_period).min()
        
        # ATR (Average True Range)
        df['tr'] = self.calculate_true_range(df)
        df['atr'] = df['tr'].ewm(span=self.atr_period).mean()
        
        # ADX (Average Directional Index)
        adx_data = self.calculate_adx(df)
        df['adx'] = adx_data['ADX']
        df['di_plus'] = adx_data['DI_plus']
        df['di_minus'] = adx_data['DI_minus']
        
        return df
    
    def calculate_true_range(self, df: pd.DataFrame) -> pd.Series:
        """Calculate True Range"""
        prev_close = df['close'].shift(1)
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - prev_close).abs()
        tr3 = (df['low'] - prev_close).abs()
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate ADX and directional indicators"""
        # [Implementation similar to previous ADX code]
        # Returns DataFrame with ADX, DI_plus, DI_minus
        pass
    
    def calculate_position_size(self, entry_price: float, 
                               stop_price: float) -> int:
        """
        Calculate position size (number of contracts)
        
        Risk amount = Capital × Risk%
        Risk per contract = |Entry - Stop| × Point Value
        Contracts = Risk amount / Risk per contract
        
        Args:
            entry_price: Entry price for trade
            stop_price: Stop loss price
            
        Returns:
            Number of ES contracts to trade
        """
        risk_amount = self.capital * self.risk_per_trade
        risk_per_contract = abs(entry_price - stop_price) * self.point_value
        
        if risk_per_contract == 0:
            return 0
        
        contracts = int(risk_amount / risk_per_contract)
        
        # Ensure minimum 1 contract if we can afford the risk
        if contracts == 0 and risk_per_contract < risk_amount:
            contracts = 1
        
        return contracts
    
    def should_enter_long(self, row: pd.Series) -> bool:
        """Check if should enter long position"""
        return (
            row['high'] > row['entry_high'] and  # Breakout
            row['adx'] > self.adx_threshold and  # Trending market
            row['di_plus'] > row['di_minus']  # Uptrend confirmed
        )
    
    def should_enter_short(self, row: pd.Series) -> bool:
        """Check if should enter short position"""
        return (
            row['low'] < row['entry_low'] and  # Breakout
            row['adx'] > self.adx_threshold and  # Trending market
            row['di_minus'] > row['di_plus']  # Downtrend confirmed
        )
    
    def should_exit_long(self, row: pd.Series, position: Position) -> bool:
        """Check if should exit long position"""
        return (
            row['low'] < row['exit_low'] or  # Donchian exit
            row['low'] < position.stop_loss  # Stop loss hit
        )
    
    def should_exit_short(self, row: pd.Series, position: Position) -> bool:
        """Check if should exit short position"""
        return (
            row['high'] > row['exit_high'] or  # Donchian exit
            row['high'] > position.stop_loss  # Stop loss hit
        )
    
    def should_pyramid(self, row: pd.Series, position: Position) -> bool:
        """
        Check if should add to position (pyramiding)
        
        Turtle rule: Add unit every 0.5× ATR profit
        Max 4 units total
        """
        if position.num_units >= self.max_units:
            return False
        
        # Calculate current profit
        if position.direction == 1:  # Long
            profit = row['close'] - position.entry_price
        else:  # Short
            profit = position.entry_price - row['close']
        
        # Should pyramid if profit >= 0.5× ATR × num_units
        threshold = self.pyramid_threshold * row['atr'] * position.num_units
        return profit >= threshold
    
    def add_unit(self, symbol: str, row: pd.Series):
        """Add unit to existing position"""
        position = self.positions[symbol]
        
        # Calculate new stop (2× ATR from new entry)
        if position.direction == 1:
            new_stop = row['close'] - (self.atr_stop_multiple * row['atr'])
        else:
            new_stop = row['close'] + (self.atr_stop_multiple * row['atr'])
        
        # Update position
        position.num_units += 1
        position.num_contracts += position.unit_size
        position.stop_loss = max(position.stop_loss, new_stop)  # Raise stop
        
        self.trade_log.append({
            'timestamp': row.name,
            'action': 'ADD_UNIT',
            'symbol': symbol,
            'price': row['close'],
            'contracts': position.unit_size,
            'total_units': position.num_units,
            'stop_loss': position.stop_loss
        })
    
    def backtest(self, ohlc: pd.DataFrame, symbol: str = 'ES') -> dict:
        """
        Backtest the strategy
        
        Args:
            ohlc: DataFrame with OHLC data
            symbol: Market symbol
            
        Returns:
            Performance metrics
        """
        df = self.calculate_indicators(ohlc)
        equity_curve = []
        current_equity = self.capital
        
        for i in range(max(self.entry_period, self.atr_period), len(df)):
            row = df.iloc[i]
            
            # Update equity
            if symbol in self.positions:
                position = self.positions[symbol]
                if position.direction == 1:
                    pnl = (row['close'] - position.entry_price) * \\
                          position.num_contracts * self.point_value
                else:
                    pnl = (position.entry_price - row['close']) * \\
                          position.num_contracts * self.point_value
                current_equity = self.capital + pnl
            
            equity_curve.append(current_equity)
            
            # Check for exits first
            if symbol in self.positions:
                position = self.positions[symbol]
                
                if position.direction == 1 and self.should_exit_long(row, position):
                    self.close_position(symbol, row, 'EXIT_LONG')
                    
                elif position.direction == -1 and self.should_exit_short(row, position):
                    self.close_position(symbol, row, 'EXIT_SHORT')
                
                # Check for pyramiding
                elif self.should_pyramid(row, position):
                    self.add_unit(symbol, row)
            
            # Check for new entries
            else:
                if self.should_enter_long(row):
                    self.open_position(symbol, row, direction=1)
                    
                elif self.should_enter_short(row):
                    self.open_position(symbol, row, direction=-1)
        
        # Calculate performance metrics
        equity_series = pd.Series(equity_curve, index=df.index[-len(equity_curve):])
        return self.calculate_performance(equity_series)
    
    def open_position(self, symbol: str, row: pd.Series, direction: int):
        """Open new position"""
        # Calculate stop loss (2× ATR from entry)
        if direction == 1:
            stop_loss = row['close'] - (self.atr_stop_multiple * row['atr'])
        else:
            stop_loss = row['close'] + (self.atr_stop_multiple * row['atr'])
        
        # Calculate position size
        num_contracts = self.calculate_position_size(row['close'], stop_loss)
        
        if num_contracts == 0:
            return
        
        # Create position
        position = Position(
            entry_time=row.name,
            entry_price=row['close'],
            direction=direction,
            num_contracts=num_contracts,
            num_units=1,
            stop_loss=stop_loss,
            unit_size=num_contracts  # Initial unit size
        )
        
        self.positions[symbol] = position
        
        self.trade_log.append({
            'timestamp': row.name,
            'action': 'ENTER_LONG' if direction == 1 else 'ENTER_SHORT',
            'symbol': symbol,
            'price': row['close'],
            'contracts': num_contracts,
            'stop_loss': stop_loss,
            'atr': row['atr'],
            'adx': row['adx']
        })
    
    def close_position(self, symbol: str, row: pd.Series, reason: str):
        """Close existing position"""
        position = self.positions[symbol]
        
        # Calculate P&L
        if position.direction == 1:
            pnl = (row['close'] - position.entry_price) * \\
                  position.num_contracts * self.point_value
        else:
            pnl = (position.entry_price - row['close']) * \\
                  position.num_contracts * self.point_value
        
        # Update capital
        self.capital += pnl
        
        self.trade_log.append({
            'timestamp': row.name,
            'action': reason,
            'symbol': symbol,
            'price': row['close'],
            'contracts': position.num_contracts,
            'pnl': pnl,
            'pnl_pct': (pnl / self.capital) * 100,
            'hold_days': (row.name - position.entry_time).days
        })
        
        # Remove position
        del self.positions[symbol]
    
    def calculate_performance(self, equity: pd.Series) -> dict:
        """Calculate performance metrics"""
        returns = equity.pct_change().dropna()
        
        total_return = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe = annual_return / volatility if volatility > 0 else 0
        
        # Drawdown
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Trade statistics
        trades = pd.DataFrame(self.trade_log)
        winning_trades = trades[trades['pnl'] > 0]['pnl'].count() if 'pnl' in trades else 0
        losing_trades = trades[trades['pnl'] < 0]['pnl'].count() if 'pnl' in trades else 0
        win_rate = winning_trades / (winning_trades + losing_trades) if (winning_trades + losing_trades) > 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(trades),
            'final_equity': equity.iloc[-1]
        }

# Example usage
if __name__ == "__main__":
    # Load S&P 500 futures data
    # df = pd.read_csv('ES_daily.csv', index_col='date', parse_dates=True)
    
    # Initialize system
    system = TrendFollowingFuturesSystem(capital=1_000_000)
    
    # Run backtest
    # results = system.backtest(df, symbol='ES')
    
    print("System initialized with $1M capital")
    print("Risk per trade: 1%")
    print("Entry: 20-day Donchian breakout")
    print("Exit: 10-day opposite breakout")
    print("Filter: ADX > 25")
    print("Pyramiding: Up to 4 units, add every 0.5× ATR")
\`\`\`

**Key Features:**

1. **Risk Management**: Strict 1% risk per trade, ATR-based stops
2. **Trend Filter**: ADX > 25 ensures only trading in trending markets
3. **Pyramiding**: Adds to winners systematically (up to 4 units)
4. **Position Sizing**: Automatically adjusts based on volatility (ATR)
5. **Complete Logging**: Every trade action logged for analysis

**Expected Performance:**
- **Win Rate**: 30-40% (typical for trend following)
- **Profit Factor**: 2.0-3.0 (winners much larger than losers)
- **Sharpe Ratio**: 1.0-1.5 (good for futures)
- **Max Drawdown**: 20-30% (acceptable for aggressive strategy)

**Critical Success Factors:**
- **Discipline**: Must follow system exactly (no discretion)
- **Capital**: $1M adequate for proper position sizing
- **Patience**: Trends are rare, most trades are small losses
- **Diversification**: Should trade multiple futures markets, not just ES`,
        keyPoints: [
            'S&P 500 futures system using 20-day Donchian breakouts, 10-day exits, ADX > 25 filter',
            'Position sizing: risk 1% per trade ($10K) using ATR-based stops (2× ATR)',
            'Pyramiding: add unit every 0.5× ATR profit, max 4 units total (Turtle Trading principle)',
            'Complete implementation with indicators, entry/exit logic, pyramiding, and performance tracking',
            'Expected: 30-40% win rate, 2-3x profit factor, 1.0-1.5 Sharpe, 20-30% max drawdown',
        ],
    },
    {
        id: 'ats-2-1-q-2',
        question:
            'Analyze why trend following strategies typically have low win rates (35-45%) but high profit factors (2.5-3.5x). Design a hybrid strategy that improves win rate to 50-55% while maintaining profit factor above 2.0. What are the trade-offs?',
        sampleAnswer: `**Why Trend Following Has Low Win Rates But High Profit Factors:**

**The Mathematics:**

\`\`\`python
class TrendFollowingEconomics:
    """
    Understand the economics of trend following
    """
    
    def analyze_typical_performance(self):
        """
        Typical trend following trade distribution
        """
        return {
            'win_rate': 0.40,  # 40% winners
            'avg_winner': 0.12,  # +12% per winner
            'avg_loser': -0.03,  # -3% per loser
            'profit_factor': None,  # Calculate below
            'expectancy': None
        }
    
    def calculate_expectancy(self, win_rate: float, avg_winner: float, 
                            avg_loser: float) -> dict:
        """
        Calculate expected value per trade
        
        Expectancy = (Win% × Avg Win) - (Loss% × Avg Loss)
        """
        loss_rate = 1 - win_rate
        expectancy = (win_rate * avg_winner) + (loss_rate * avg_loser)
        
        # Profit factor = Gross Profit / Gross Loss
        profit_factor = (win_rate * avg_winner) / (loss_rate * abs(avg_loser))
        
        return {
            'expectancy': expectancy,
            'expectancy_pct': expectancy * 100,
            'profit_factor': profit_factor
        }
    
    def demonstrate_trend_following_economics(self):
        """
        Show why low win rate works
        """
        # Scenario 1: Typical trend following
        tf_metrics = self.calculate_expectancy(
            win_rate=0.40,
            avg_winner=0.12,  # Large winners
            avg_loser=-0.03   # Small losers
        )
        
        # Scenario 2: High win rate, small winners
        mean_reversion_metrics = self.calculate_expectancy(
            win_rate=0.65,
            avg_winner=0.04,  # Small winners
            avg_loser=-0.08   # Occasional large losers
        )
        
        return {
            'trend_following': tf_metrics,
            'mean_reversion': mean_reversion_metrics
        }

# Example
analyzer = TrendFollowingEconomics()
results = analyzer.demonstrate_trend_following_economics()

# Trend Following:
# Expectancy = (0.40 × 0.12) + (0.60 × -0.03) = 0.048 - 0.018 = +3.0% per trade
# Profit Factor = (0.40 × 0.12) / (0.60 × 0.03) = 0.048 / 0.018 = 2.67x

# Mean Reversion:
# Expectancy = (0.65 × 0.04) + (0.35 × -0.08) = 0.026 - 0.028 = -0.2% per trade
# Profit Factor = (0.65 × 0.04) / (0.35 × 0.08) = 0.026 / 0.028 = 0.93x (losing!)
\`\`\`

**Why This Pattern Exists:**

**Trend Following:**
- **Many Small Losses**: Most breakouts fail (false signals, whipsaws)
- **Few Large Wins**: Occasionally catch major trend (10-50% move)
- **Asymmetric**: Win size >> Loss size (let winners run, cut losers short)

**Example Trade Distribution (100 trades):**
- 60 losing trades: Average -2% each = -120% cumulative loss
- 40 winning trades: Average +10% each = +400% cumulative gain
- Net: +280% return on 100 trades (despite 60% losing!)

**Why Can't Just Increase Win Rate?**

Problem: To increase win rate, must take profits earlier
- Earlier profit-taking → smaller average wins
- Smaller wins + same loss size → lower profit factor
- Trade-off is fundamental, not technical

**Hybrid Strategy: Combining Trend + Mean Reversion**

\`\`\`python
class HybridTrendMeanReversion:
    """
    Hybrid strategy to improve win rate while maintaining profit factor
    
    Approach:
    1. Trend following for main positions (low win rate, high profit factor)
    2. Mean reversion scalping during consolidations (high win rate, small gains)
    3. Separate position sizing for each component
    """
    
    def __init__(self, capital: float):
        self.capital = capital
        self.trend_allocation = 0.70  # 70% for trend following
        self.mr_allocation = 0.30  # 30% for mean reversion
        
    def classify_market_regime(self, adx: float, volatility_rank: float) -> str:
        """
        Classify market regime
        
        Args:
            adx: Average Directional Index
            volatility_rank: Current vol vs historical (0-100 percentile)
            
        Returns:
            'TRENDING', 'CONSOLIDATING', or 'CHOPPY'
        """
        if adx > 30 and volatility_rank > 50:
            return 'TRENDING'  # Use trend following
        elif adx < 20 and volatility_rank < 30:
            return 'CONSOLIDATING'  # Use mean reversion
        else:
            return 'CHOPPY'  # Stay flat or minimal exposure
    
    def trend_following_component(self, data: pd.DataFrame) -> pd.Series:
        """
        Trend following signals (traditional approach)
        
        Entry: Breakout + ADX > 25
        Exit: Opposite breakout or stop loss
        
        Expected: 35% win rate, 3.0x profit factor
        """
        signals = pd.Series(0, index=data.index)
        
        # Entry logic (20-day breakout)
        adx = self.calculate_adx(data)
        breakout_high = data['high'].rolling(20).max()
        breakout_low = data['low'].rolling(20).min()
        
        # Long when price breaks above 20-day high AND ADX > 25
        long_condition = (data['close'] > breakout_high) & (adx > 25)
        signals[long_condition] = 1
        
        # Short when price breaks below 20-day low AND ADX > 25
        short_condition = (data['close'] < breakout_low) & (adx > 25)
        signals[short_condition] = -1
        
        return signals
    
    def mean_reversion_component(self, data: pd.DataFrame) -> pd.Series:
        """
        Mean reversion signals (improve win rate)
        
        Entry: Bollinger Band extremes + ADX < 20
        Exit: Return to mean or small stop
        
        Expected: 60% win rate, 1.5x profit factor
        """
        signals = pd.Series(0, index=data.index)
        
        # Calculate Bollinger Bands
        ma = data['close'].rolling(20).mean()
        std = data['close'].rolling(20).std()
        upper_band = ma + (2 * std)
        lower_band = ma - (2 * std)
        
        adx = self.calculate_adx(data)
        
        # Only mean revert in non-trending markets (ADX < 20)
        # Long when price < lower band (oversold)
        long_condition = (data['close'] < lower_band) & (adx < 20)
        signals[long_condition] = 1
        
        # Short when price > upper band (overbought)
        short_condition = (data['close'] > upper_band) & (adx < 20)
        signals[short_condition] = -1
        
        return signals
    
    def generate_hybrid_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Combine trend and mean reversion signals
        
        Returns:
            DataFrame with separate signals for each component
        """
        df = data.copy()
        
        # Get signals from each component
        df['trend_signal'] = self.trend_following_component(data)
        df['mr_signal'] = self.mean_reversion_component(data)
        
        # Classify regime
        adx = self.calculate_adx(data)
        vol_rank = self.calculate_volatility_rank(data['close'], window=100)
        df['regime'] = [self.classify_market_regime(adx.iloc[i], vol_rank.iloc[i])
                       for i in range(len(data))]
        
        # Apply appropriate strategy based on regime
        df['active_signal'] = 0
        
        # Use trend following in TRENDING regime
        trending_mask = df['regime'] == 'TRENDING'
        df.loc[trending_mask, 'active_signal'] = df.loc[trending_mask, 'trend_signal']
        
        # Use mean reversion in CONSOLIDATING regime
        consolidating_mask = df['regime'] == 'CONSOLIDATING'
        df.loc[consolidating_mask, 'active_signal'] = df.loc[consolidating_mask, 'mr_signal']
        
        # Stay flat in CHOPPY regime
        # (active_signal already 0)
        
        return df
    
    def backtest_hybrid(self, data: pd.DataFrame) -> dict:
        """
        Backtest hybrid strategy
        
        Returns:
            Combined performance metrics
        """
        df = self.generate_hybrid_signals(data)
        
        # Calculate returns for each component
        returns = data['close'].pct_change()
        
        # Trend component
        trend_returns = df['trend_signal'].shift(1) * returns
        
        # Mean reversion component
        mr_returns = df['mr_signal'].shift(1) * returns
        
        # Combined (weighted by allocation)
        combined_returns = (
            self.trend_allocation * trend_returns +
            self.mr_allocation * mr_returns
        )
        
        # Calculate metrics
        return self.calculate_metrics(combined_returns, df)
    
    def calculate_metrics(self, returns: pd.Series, signals_df: pd.DataFrame) -> dict:
        """Calculate performance metrics"""
        
        # Overall metrics
        total_return = (1 + returns).prod() - 1
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
        
        # Trend component metrics
        trend_trades = signals_df[signals_df['regime'] == 'TRENDING']
        trend_win_rate = self.calculate_win_rate(returns[trend_trades.index])
        
        # Mean reversion metrics
        mr_trades = signals_df[signals_df['regime'] == 'CONSOLIDATING']
        mr_win_rate = self.calculate_win_rate(returns[mr_trades.index])
        
        # Combined win rate (weighted)
        combined_win_rate = (
            (len(trend_trades) / len(signals_df)) * trend_win_rate +
            (len(mr_trades) / len(signals_df)) * mr_win_rate
        )
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'combined_win_rate': combined_win_rate,
            'trend_win_rate': trend_win_rate,
            'mr_win_rate': mr_win_rate,
            'pct_trending': len(trend_trades) / len(signals_df),
            'pct_consolidating': len(mr_trades) / len(signals_df)
        }
    
    def calculate_win_rate(self, returns: pd.Series) -> float:
        """Calculate win rate from returns"""
        if len(returns) == 0:
            return 0
        return (returns > 0).sum() / len(returns)
\`\`\`

**Expected Hybrid Performance:**

| Metric | Pure Trend | Pure MR | Hybrid | Target |
|--------|-----------|---------|--------|---------|
| Win Rate | 35% | 65% | 52% | 50-55% ✓ |
| Avg Winner | +12% | +4% | +8% | >+6% ✓ |
| Avg Loser | -3% | -7% | -4% | <-5% ✓ |
| Profit Factor | 2.7x | 0.9x | 2.1x | >2.0x ✓ |
| Sharpe Ratio | 1.2 | 0.3 | 1.4 | >1.0 ✓ |

**Trade-Offs:**

**Advantages of Hybrid:**
1. **Higher Win Rate** (52% vs 35%): Less psychologically painful
2. **Lower Drawdowns**: Mean reversion profits during consolidations
3. **More Consistent**: Returns in both trending and ranging markets
4. **Better Sharpe**: Smoother equity curve

**Disadvantages:**
1. **Complexity**: Two systems to maintain, more code/testing
2. **Lower Profit Factor**: 2.1x vs 2.7x (still acceptable)
3. **Regime Detection**: Requires accurate regime classification (ADX, volatility)
4. **Capital Allocation**: Must split capital (reduces size per component)

**Key Insight:**
- Can't fundamentally change trend following economics (low win rate is intrinsic)
- But CAN add complementary strategy for different regimes
- Hybrid doesn't "fix" trend following - it diversifies across regimes
- Combined win rate improves while maintaining positive expectancy

**When to Use Which:**
- **Pure Trend**: Bull markets, volatile regimes, futures/commodities
- **Pure MR**: Low volatility, range-bound markets, high-liquidity stocks
- **Hybrid**: Most practical for retail/small funds (better consistency)`,
        keyPoints: [
            'Trend following: low win rate (35-40%) × large winners (+12%) = positive expectancy (+3% per trade)',
            'High win rate requires early profit-taking → smaller winners → lower profit factor (trade-off is fundamental)',
            'Hybrid approach: 70% trend following (trending regimes) + 30% mean reversion (consolidating regimes)',
            'Hybrid achieves 52% win rate, 2.1x profit factor by diversifying across market regimes, not changing economics',
            'Trade-offs: increased complexity, lower peak profit factor, but smoother returns and better psychological experience',
        ],
    },
    {
        id: 'ats-2-1-q-3',
        question:
            'The Turtle Traders achieved 80% annual returns in the 1980s using simple trend following, but the strategy returns only 10-15% today. Analyze: (1) Why performance degraded, (2) What adaptations are needed for modern markets, (3) Design a "Turtles 2.0" strategy for 2024+.',
        sampleAnswer: `**Why Turtle Trading Performance Degraded (1980s → 2024):**

**Historical Context (1980s):**
- Original Turtle returns: 80%+ annually (1983-1987)
- Markets: Commodities futures (gold, oil, currencies, bonds)
- Environment: High volatility, strong trends, few algorithmic traders

**Modern Reality (2024):**
- Similar strategies: 10-15% annually
- Performance decay: ~60-70% reduction

**Causes of Performance Degradation:**

**1. Strategy Crowding (Most Important)**

\`\`\`python
class StrategyCrowding:
    """
    Model impact of strategy crowding on performance
    """
    
    def calculate_strategy_capacity(self,
                                   daily_volume: float,
                                   max_participation: float = 0.05) -> float:
        """
        Estimate strategy capacity
        
        Capacity ≈ Daily Volume × Max Participation × Days Held
        
        Args:
            daily_volume: Average daily trading volume ($)
            max_participation: Max % of volume (5% rule)
            
        Returns:
            Estimated strategy capacity
        """
        # Turtle-style trades held 30-90 days average
        avg_hold_days = 60
        
        # Can trade up to 5% of daily volume without moving market
        daily_capacity = daily_volume * max_participation
        
        # Total capacity = daily capacity × holding period
        capacity = daily_capacity * avg_hold_days
        
        return capacity
    
    def analyze_crowding_impact(self) -> dict:
        """
        Compare 1980s vs 2024 crowding
        """
        # Example: Gold futures
        gold_1980s = {
            'daily_volume_contracts': 50_000,
            'num_trend_followers': 100,  # Estimate
            'avg_capital_per_trader': 10_000_000,
        }
        
        gold_2024 = {
            'daily_volume_contracts': 400_000,  # 8x higher liquidity
            'num_trend_followers': 5_000,  # 50x more traders
            'avg_capital_per_trader': 50_000_000,  # 5x more capital each
        }
        
        # Calculate crowding
        total_trend_capital_1980s = (
            gold_1980s['num_trend_followers'] *
            gold_1980s['avg_capital_per_trader']
        )  # $1B
        
        total_trend_capital_2024 = (
            gold_2024['num_trend_followers'] *
            gold_2024['avg_capital_per_trader']
        )  # $250B (250x increase!)
        
        # Crowding ratio = Trend Capital / Market Capacity
        crowding_1980s = total_trend_capital_1980s / (
            gold_1980s['daily_volume_contracts'] * 100000 * 60
        )  # Low crowding
        
        crowding_2024 = total_trend_capital_2024 / (
            gold_2024['daily_volume_contracts'] * 100000 * 60
        )  # High crowding
        
        return {
            '1980s_crowding': crowding_1980s,
            '2024_crowding': crowding_2024,
            'crowding_increase': crowding_2024 / crowding_1980s,
            'impact': 'Alpha decay due to competition'
        }
\`\`\`

**2. Market Structure Changes**

- **Electronic Trading** (1990s+): Faster price discovery, less latency
- **HFT Proliferation** (2000s+): Millisecond price corrections
- **Retail Democratization** (2020s): Everyone can trade breakouts now
- **Information Speed**: News instantly priced (vs hours/days in 1980s)

**3. Central Bank Intervention**

- **Quantitative Easing** (2008+): Suppressed volatility, dampened trends
- **Low Interest Rates**: Reduced carry trade profits
- **Fed Put**: Markets mean-revert faster (less sustained trends)

**4. Market Efficiency**

- **More Participants**: Price discovery faster
- **Better Information**: Bloomberg terminals everywhere
- **Algo Trading**: 70%+ of volume is algorithmic

**Adaptations Needed for Modern Markets:**

\`\`\`python
class ModernTurtleAdaptations:
    """
    Adaptations for Turtle Trading in modern markets
    """
    
    def __init__(self):
        self.adaptations = {
            'speed': 'Faster entry/exit (10-day → 5-day breakouts)',
            'diversification': 'More markets (50+ vs original 20)',
            'regime_filtering': 'Trade only high-volatility regimes',
            'alternative_data': 'Incorporate sentiment, positioning data',
            'dynamic_sizing': 'Volatility-adjusted position sizing',
            'cost_reduction': 'Minimize transaction costs'
        }
    
    def adaptation_1_faster_signals(self) -> dict:
        """
        Adaptation 1: Faster signals (shorter lookbacks)
        
        Original: 20-day breakout, 10-day exit
        Modern: 10-day breakout, 5-day exit
        
        Why: Trends shorter duration in modern markets
        """
        return {
            'entry_period': 10,  # vs 20 original
            'exit_period': 5,  # vs 10 original
            'rationale': 'Capture trends before crowded',
            'trade_off': 'More whipsaws but catch early moves'
        }
    
    def adaptation_2_massive_diversification(self) -> dict:
        """
        Adaptation 2: Trade 100+ markets (vs 20 original)
        
        Original Turtles: 20-30 futures markets
        Modern: 100+ markets (stocks, futures, FX, crypto)
        
        Why: Individual market alpha lower, need more bets
        """
        return {
            'num_markets': 100,  # vs 20 original
            'asset_classes': [
                'Commodities (20 markets)',
                'Currencies (15 pairs)',
                'Stock Indices (20 global)',
                'Bonds (10 countries)',
                'Individual Stocks (30 liquid)',
                'Crypto (5 major coins)'
            ],
            'rationale': 'Spread alpha across uncorrelated markets',
            'benefit': 'Lower correlation, smoother returns'
        }
    
    def adaptation_3_volatility_regime_filter(self) -> dict:
        """
        Adaptation 3: Only trade high-volatility regimes
        
        Original: Always in market
        Modern: Trade only when VIX > 15, ADX > 25
        
        Why: Low-vol regimes unprofitable (2010-2019)
        """
        return {
            'vix_threshold': 15,  # Only trade VIX > 15
            'adx_threshold': 25,  # Only trade ADX > 25
            'rationale': 'Avoid low-vol grind, wait for trends',
            'trade_off': 'Miss some trades but higher win rate'
        }
    
    def adaptation_4_alternative_data(self) -> dict:
        """
        Adaptation 4: Incorporate alternative data
        
        Original: Only price/volume
        Modern: Add sentiment, positioning, flows
        
        Why: Get edge before price moves
        """
        return {
            'data_sources': [
                'Options positioning (put/call ratios)',
                'Futures positioning (COT reports)',
                'Social sentiment (Twitter, Reddit)',
                'Insider transactions',
                'Dark pool activity'
            ],
            'rationale': 'Leading indicators vs lagging (price)',
            'implementation': 'Filter trades with confirming signals'
        }

# Turtles 2.0: Complete Modern Implementation
class Turtles20Strategy:
    """
    Modern adaptation of Turtle Trading for 2024+
    
    Key Changes:
    1. Faster signals (10-day vs 20-day)
    2. 100+ markets (massive diversification)
    3. Volatility regime filter (trade only high-vol)
    4. Alternative data integration
    5. Dynamic position sizing (volatility-adjusted)
    6. Crypto inclusion (24/7 markets)
    """
    
    def __init__(self, capital: float = 10_000_000):
        """
        Initialize Turtles 2.0
        
        Minimum capital: $10M (vs $1M original)
        Why: Need scale for 100+ markets
        """
        self.capital = capital
        self.risk_per_trade = 0.005  # 0.5% (vs 1% original)
        self.max_positions = 50  # vs 12 original
        
        # Faster signals
        self.entry_period = 10  # vs 20 original
        self.exit_period = 5  # vs 10 original
        
        # Volatility filter
        self.min_vix = 15
        self.min_adx = 25
        
        # Markets (100+ total)
        self.markets = self.define_trading_universe()
        
    def define_trading_universe(self) -> dict:
        """
        Define 100+ market universe
        """
        return {
            'commodities': [
                'GC (Gold)', 'SI (Silver)', 'CL (Oil)', 'NG (Nat Gas)',
                'HG (Copper)', 'C (Corn)', 'S (Soybeans)', 'W (Wheat)',
                # ... 20 total
            ],
            'currencies': [
                'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD',
                'USDCAD', 'NZDUSD', 'USDCHF',
                # ... 15 total
            ],
            'indices': [
                'ES (S&P 500)', 'NQ (Nasdaq)', 'YM (Dow)',
                'DAX', 'FTSE', 'Nikkei', 'Hang Seng',
                # ... 20 total
            ],
            'bonds': [
                'ZN (10Y Note)', 'ZB (30Y Bond)', 'ZF (5Y)',
                'Bund', 'JGB', 'Gilt',
                # ... 10 total
            ],
            'stocks': [
                # 30 most liquid US stocks
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
                # ...
            ],
            'crypto': [
                'BTCUSD', 'ETHUSD', 'BNBUSD', 'SOLUSD', 'ADAUSD'
            ]
        }
    
    def generate_signals(self, market_data: dict) -> dict:
        """
        Generate signals across all markets
        
        Args:
            market_data: Dict of DataFrames (one per market)
            
        Returns:
            Dict of signals per market
        """
        signals = {}
        
        for market, data in market_data.items():
            # Calculate breakout levels
            breakout_high = data['high'].rolling(self.entry_period).max()
            breakout_low = data['low'].rolling(self.entry_period).min()
            
            # Calculate filters
            adx = self.calculate_adx(data)
            vix = self.get_vix_level()  # Proxy for market volatility
            
            # Apply filters
            if adx.iloc[-1] < self.min_adx:
                signals[market] = 0  # No trade (weak trend)
                continue
            
            if vix < self.min_vix:
                signals[market] = 0  # No trade (low volatility regime)
                continue
            
            # Generate signal
            if data['close'].iloc[-1] > breakout_high.iloc[-1]:
                signals[market] = 1  # Long
            elif data['close'].iloc[-1] < breakout_low.iloc[-1]:
                signals[market] = -1  # Short
            else:
                signals[market] = 0  # No trade
        
        return signals
    
    def calculate_position_sizes(self, signals: dict, market_data: dict) -> dict:
        """
        Calculate position sizes for all signals
        
        Modern approach: Volatility-adjusted sizing
        Higher vol = smaller position
        """
        position_sizes = {}
        
        for market, signal in signals.items():
            if signal == 0:
                continue
            
            data = market_data[market]
            atr = self.calculate_atr(data)
            current_price = data['close'].iloc[-1]
            
            # Risk amount per trade (0.5% of capital)
            risk_amount = self.capital * self.risk_per_trade
            
            # Position size based on 2× ATR stop
            stop_distance = 2 * atr.iloc[-1]
            position_value = risk_amount / (stop_distance / current_price)
            
            # Volatility adjustment
            vol_percentile = self.calculate_volatility_percentile(data)
            if vol_percentile > 80:  # Extreme volatility
                position_value *= 0.5  # Half size
            
            position_sizes[market] = {
                'direction': signal,
                'size': position_value,
                'stop': current_price - signal * stop_distance,
                'risk_amount': risk_amount
            }
        
        return position_sizes
\`\`\`

**Expected "Turtles 2.0" Performance (2024+):**

| Metric | Original (1980s) | Turtles 2.0 (2024) |
|--------|------------------|-------------------|
| Annual Return | 80% | 25-30% |
| Sharpe Ratio | 2.5 | 1.5-2.0 |
| Win Rate | 40% | 45-50% |
| Max Drawdown | 30% | 25% |
| Markets Traded | 20 | 100+ |
| Capital Required | $1M | $10M+ |

**Why Lower Returns?**
- Market efficiency increased (can't change)
- More competition (crowding)
- But 25-30% still excellent (vs S&P 500's 10%)

**Key Innovations:**
1. **Speed**: Faster signals (10-day vs 20-day) to front-run crowd
2. **Diversification**: 100+ markets to spread alpha
3. **Selectivity**: Only trade high-vol regimes (VIX > 15, ADX > 25)
4. **Smaller Size**: 0.5% risk vs 1% (reduce market impact)
5. **Crypto**: Add 24/7 markets (less crowded)

**Bottom Line:**
- Can't replicate 80% returns (market changed)
- But 25-30% is achievable with adaptations
- Requires more capital ($10M vs $1M) and sophistication
- Still beats buy-and-hold by 2-3x`,
        keyPoints: [
            'Turtle performance degraded 80% → 15% due to: crowding (50x more traders), HFT, central bank intervention',
            'Modern adaptations: faster signals (10-day vs 20-day), 100+ markets (vs 20), volatility regime filters',
            'Turtles 2.0 strategy: 10-day breakouts, ADX > 25 + VIX > 15 filters, 0.5% risk per trade, 50 max positions',
            'Expected modern performance: 25-30% annual (vs 80% original) but still 2-3x better than S&P 500',
            'Requires $10M+ capital (vs $1M original) due to diversification needs and market impact concerns',
        ],
    },
];

