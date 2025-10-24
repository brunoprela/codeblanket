export const backtestingSimulation = {
  title: 'Backtesting & Simulation',
  id: 'backtesting-simulation',
  content: `
# Backtesting & Simulation

## Introduction

Backtesting is the process of testing a trading strategy on historical data to estimate its future performance. It's the **most critical step** in algorithmic trading—a strategy that looks good in backtest might still fail live, but one that fails in backtest will **definitely** fail live.

**The Challenge**: 80% of strategies that pass backtesting fail in live trading. Why?
- **Overfitting**: Curve-fitting to past data
- **Biases**: Lookahead, survivorship, selection bias
- **Reality Gap**: Missing costs, slippage, liquidity constraints
- **Psychology**: Paper trading has no emotional component

By the end of this section, you'll master:
- Building professional backtesting frameworks
- Avoiding common pitfalls and biases
- Monte Carlo simulation for robustness testing
- Walk-forward analysis for realistic validation
- Transaction cost modeling
- Multi-strategy and portfolio backtesting

**Key Principle**: "In backtesting, you're guilty until proven innocent." Assume every good result is a bug until thoroughly validated.

---

## Common Backtesting Pitfalls

### The 7 Deadly Sins of Backtesting

\`\`\`python
"""
Common Backtesting Mistakes and How to Avoid Them
"""

# SIN 1: Lookahead Bias
# ❌ WRONG: Using future information
def calculate_signal_WRONG(prices):
    # This uses t+1 information at time t!
    future_return = prices.pct_change().shift(-1)
    return (future_return > 0).astype(int)

# ✅ CORRECT: Only use past information
def calculate_signal_CORRECT(prices):
    # Past return available at time t
    past_return = prices.pct_change(5)  # 5-day return
    return (past_return > 0.05).astype(int).shift(1)  # Shift by 1!


# SIN 2: Survivorship Bias
# ❌ WRONG: Using current index constituents
current_sp500 = ['AAPL', 'MSFT', 'GOOGL', ...]  # Only survivors
backtest_data = yf.download(current_sp500, start='2000-01-01')  # Missing failures!

# ✅ CORRECT: Use point-in-time universe
def get_point_in_time_universe(date):
    """Return stocks actually in S&P 500 on this date"""
    # Include delisted/bankrupt stocks that were in index
    # Use services like Sharadar, Quandl, or exchange archives
    return historical_constituents[date]


# SIN 3: Selection Bias
# ❌ WRONG: Testing many strategies, reporting best
for strategy in range(1000):
    result = backtest(strategy, data)
    if result.sharpe > 2.0:
        print(f"Found amazing strategy {strategy}!")  # Pure luck!

# ✅ CORRECT: Bonferroni correction for multiple testing
alpha = 0.05
num_strategies = 1000
corrected_alpha = alpha / num_strategies  # 0.00005
# Only significant if p-value < corrected_alpha


# SIN 4: Ignoring Transaction Costs
# ❌ WRONG: Assume free trading
signals = generate_signals(data)
returns = signals.shift(1) * data['Close'].pct_change()  # No costs!

# ✅ CORRECT: Include all costs
def apply_transaction_costs(returns, signals, commission=0.001, slippage=0.0005):
    """Apply realistic costs"""
    # Commission on each trade
    trades = signals.diff().abs()
    commission_cost = trades * commission
    
    # Slippage (wider on volatile days)
    volatility = returns.rolling(20).std()
    slippage_cost = trades * (slippage + volatility * 0.5)
    
    net_returns = returns - commission_cost - slippage_cost
    return net_returns


# SIN 5: Overfitting
# ❌ WRONG: Optimizing hundreds of parameters
best_sharpe = 0
best_params = None
for param1 in range(1, 100):
    for param2 in range(1, 100):
        for param3 in range(1, 100):  # 1 million combinations!
            sharpe = backtest(param1, param2, param3)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = (param1, param2, param3)

# ✅ CORRECT: Limit parameters, use walk-forward
# Maximum 3-5 parameters
# Test each parameter combination on multiple periods
# Use walk-forward: train on period 1, test on period 2


# SIN 6: Not Testing in Different Market Regimes
# ❌ WRONG: Only backtest 2020-2024 (bull market)
backtest_period = data['2020':'2024']  # Bull market only

# ✅ CORRECT: Test across regimes
bull_market = data['2016':'2020']  # Bull
bear_market = data['2007':'2009']  # Financial crisis
sideways = data['2014':'2016']  # Range-bound
volatile = data['2020-03':'2020-06']  # COVID crash

# Strategy should work in all regimes (or adapt)


# SIN 7: Data Snooping
# ❌ WRONG: Repeatedly testing and adjusting
while sharpe < 1.5:
    # Keep tweaking until good result
    strategy = adjust_strategy()
    sharpe = backtest(strategy, data)

# ✅ CORRECT: Hold out final test set
# Train set: 60% (develop strategy)
# Validation set: 20% (tune parameters)  
# Test set: 20% (final evaluation, touch ONCE only)
\`\`\`

---

## Professional Backtesting Framework

### Complete Implementation

\`\`\`python
"""
Production-Grade Backtesting Engine
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
import logging

@dataclass
class Trade:
    """Single trade record with full details"""
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    symbol: str
    direction: int  # 1 = long, -1 = short
    entry_price: float
    exit_price: float
    shares: int
    entry_value: float
    exit_value: float
    pnl: float
    return_pct: float
    commission: float
    slippage: float
    holding_period: int  # Days
    mae: float = 0.0  # Maximum Adverse Excursion
    mfe: float = 0.0  # Maximum Favorable Excursion


@dataclass
class BacktestConfig:
    """Backtest configuration"""
    initial_capital: float = 100000
    commission: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005  # 0.05% slippage
    position_size: float = 0.2  # 20% of capital per position
    max_positions: int = 5  # Maximum concurrent positions
    enable_shorts: bool = True
    # Risk controls
    max_position_risk: float = 0.02  # 2% risk per trade
    max_portfolio_risk: float = 0.20  # 20% total portfolio risk
    stop_loss_pct: float = 0.05  # 5% stop loss
    take_profit_pct: float = 0.10  # 10% take profit
    # Data quality
    min_volume: int = 100000  # Minimum daily volume
    min_price: float = 5.0  # Minimum price (avoid penny stocks)


class AdvancedBacktester:
    """
    Professional backtesting engine with full features
    """
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.positions: Dict[str, Dict] = {}  # Current positions
        self.cash = self.config.initial_capital
        self.daily_returns: List[float] = []
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def run_backtest(self, data: pd.DataFrame, signals: pd.DataFrame):
        """
        Run comprehensive backtest
        
        Args:
            data: DataFrame with columns [symbol, date, open, high, low, close, volume]
            signals: DataFrame with columns [symbol, date, signal]
                     signal: 1 (long), -1 (short), 0 (flat)
        
        Returns:
            Dictionary with results
        """
        self.logger.info(f"Starting backtest: {data.index[0]} to {data.index[-1]}")
        
        dates = sorted(data.index.unique())
        
        for date in dates:
            # Get today's data and signals
            today_data = data.loc[date]
            today_signals = signals.loc[date] if date in signals.index else pd.DataFrame()
            
            # Update existing positions (check stops, etc.)
            self._update_positions(today_data)
            
            # Exit positions based on signals or risk management
            self._exit_positions(today_data, today_signals)
            
            # Enter new positions
            self._enter_positions(today_data, today_signals)
            
            # Record equity
            portfolio_value = self._calculate_portfolio_value(today_data)
            self.equity_curve.append(portfolio_value)
            
            # Daily return
            if len(self.equity_curve) > 1:
                daily_return = (portfolio_value - self.equity_curve[-2]) / self.equity_curve[-2]
                self.daily_returns.append(daily_return)
        
        # Calculate final metrics
        return self._calculate_metrics()
    
    def _update_positions(self, data: pd.DataFrame):
        """Update position prices, check MAE/MFE"""
        for symbol, position in list(self.positions.items()):
            if symbol not in data.index:
                continue
            
            current_price = data.loc[symbol, 'close']
            entry_price = position['entry_price']
            direction = position['direction']
            
            # Calculate unrealized P&L
            if direction == 1:  # Long
                pnl_pct = (current_price - entry_price) / entry_price
            else:  # Short
                pnl_pct = (entry_price - current_price) / entry_price
            
            # Update MAE/MFE
            position['mae'] = min(position.get('mae', 0), pnl_pct)
            position['mfe'] = max(position.get('mfe', 0), pnl_pct)
            
            # Check stop loss
            if pnl_pct < -self.config.stop_loss_pct:
                self.logger.info(f"Stop loss hit for {symbol}: {pnl_pct:.2%}")
                self._close_position(symbol, data, reason='stop_loss')
            
            # Check take profit
            if pnl_pct > self.config.take_profit_pct:
                self.logger.info(f"Take profit hit for {symbol}: {pnl_pct:.2%}")
                self._close_position(symbol, data, reason='take_profit')
    
    def _enter_positions(self, data: pd.DataFrame, signals: pd.DataFrame):
        """Enter new positions based on signals"""
        # Portfolio heat check
        current_risk = len(self.positions) * self.config.max_position_risk
        if current_risk >= self.config.max_portfolio_risk:
            return  # Portfolio risk limit reached
        
        # Max positions check
        if len(self.positions) >= self.config.max_positions:
            return
        
        for symbol in signals.index:
            if symbol in self.positions:
                continue  # Already have position
            
            signal = signals.loc[symbol, 'signal']
            if signal == 0:
                continue
            
            if symbol not in data.index:
                continue
            
            # Data quality checks
            price = data.loc[symbol, 'close']
            volume = data.loc[symbol, 'volume']
            
            if price < self.config.min_price or volume < self.config.min_volume:
                continue
            
            # Short selling check
            if signal == -1 and not self.config.enable_shorts:
                continue
            
            # Calculate position size
            portfolio_value = self.cash + self._calculate_positions_value(data)
            position_value = portfolio_value * self.config.position_size
            
            # Apply slippage
            if signal == 1:
                entry_price = price * (1 + self.config.slippage)
            else:
                entry_price = price * (1 - self.config.slippage)
            
            # Calculate shares
            shares = int(position_value / entry_price)
            if shares == 0:
                continue
            
            actual_value = shares * entry_price
            commission = actual_value * self.config.commission
            
            # Check sufficient cash
            if actual_value + commission > self.cash:
                continue
            
            # Execute trade
            self.cash -= (actual_value + commission)
            
            self.positions[symbol] = {
                'entry_date': data.index[0] if isinstance(data.index, pd.DatetimeIndex) else pd.Timestamp.now(),
                'entry_price': entry_price,
                'shares': shares,
                'direction': signal,
                'entry_value': actual_value,
                'commission_paid': commission,
                'mae': 0.0,
                'mfe': 0.0
            }
            
            self.logger.info(f"Entered {symbol}: {signal} {shares} @ \${entry_price:.2f}")
    
    def _close_position(self, symbol: str, data: pd.DataFrame, reason: str = 'signal'):
        """Close a position"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        current_price = data.loc[symbol, 'close']
        
        # Apply slippage
        direction = position['direction']
        if direction == 1:
            exit_price = current_price * (1 - self.config.slippage)
        else:
            exit_price = current_price * (1 + self.config.slippage)
        
        shares = position['shares']
        exit_value = shares * exit_price
        commission = exit_value * self.config.commission
        
        # Calculate P&L
        if direction == 1:  # Long
            pnl = exit_value - position['entry_value'] - position['commission_paid'] - commission
        else:  # Short
            pnl = position['entry_value'] - exit_value - position['commission_paid'] - commission
        
        return_pct = pnl / position['entry_value']
        
        # Record trade
        trade = Trade(
            entry_date=position['entry_date'],
            exit_date=data.index[0] if isinstance(data.index, pd.DatetimeIndex) else pd.Timestamp.now(),
            symbol=symbol,
            direction=direction,
            entry_price=position['entry_price'],
            exit_price=exit_price,
            shares=shares,
            entry_value=position['entry_value'],
            exit_value=exit_value,
            pnl=pnl,
            return_pct=return_pct,
            commission=position['commission_paid'] + commission,
            slippage=abs(exit_price - current_price),
            holding_period=(data.index[0] - position['entry_date']).days if isinstance(data.index, pd.DatetimeIndex) else 0,
            mae=position['mae'],
            mfe=position['mfe']
        )
        
        self.trades.append(trade)
        
        # Update cash
        self.cash += exit_value - commission
        
        # Remove position
        del self.positions[symbol]
        
        self.logger.info(f"Closed {symbol}: PnL \${pnl:.2f} ({return_pct:.2%}), Reason: {reason}")
    
    def _exit_positions(self, data: pd.DataFrame, signals: pd.DataFrame):
        """Exit positions based on signals"""
        for symbol in list(self.positions.keys()):
            # Check if signal changed
            if symbol in signals.index:
                current_signal = signals.loc[symbol, 'signal']
                position_direction = self.positions[symbol]['direction']
                
                if current_signal != position_direction:
                    self._close_position(symbol, data, reason='signal_change')
    
    def _calculate_positions_value(self, data: pd.DataFrame) -> float:
        """Calculate total value of open positions"""
        total = 0
        for symbol, position in self.positions.items():
            if symbol in data.index:
                current_price = data.loc[symbol, 'close']
                direction = position['direction']
                shares = position['shares']
                
                if direction == 1:
                    total += shares * current_price
                else:
                    # Short position value
                    total += position['entry_value'] - (shares * current_price - position['entry_value'])
        
        return total
    
    def _calculate_portfolio_value(self, data: pd.DataFrame) -> float:
        """Calculate total portfolio value"""
        return self.cash + self._calculate_positions_value(data)
    
    def _calculate_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return {'error': 'No trades executed'}
        
        # Convert trades to DataFrame
        trades_df = pd.DataFrame([vars(t) for t in self.trades])
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = (trades_df['pnl'] > 0).sum()
        losing_trades = (trades_df['pnl'] < 0).sum()
        win_rate = winning_trades / total_trades
        
        # P&L metrics
        total_pnl = trades_df['pnl'].sum()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        profit_factor = -avg_win / avg_loss * win_rate / (1 - win_rate) if avg_loss != 0 else 0
        
        # Returns
        equity = pd.Series(self.equity_curve)
        returns = pd.Series(self.daily_returns)
        
        total_return = (equity.iloc[-1] - self.config.initial_capital) / self.config.initial_capital
        annual_return = (1 + total_return) ** (252 / len(equity)) - 1
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return - 0.02) / volatility if volatility > 0 else 0
        
        # Drawdown
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        max_drawdown = drawdown.min()
        
        # Trade analysis
        avg_holding_period = trades_df['holding_period'].mean()
        max_consecutive_wins = self._max_consecutive(trades_df['pnl'] > 0)
        max_consecutive_losses = self._max_consecutive(trades_df['pnl'] < 0)
        
        # MAE/MFE analysis
        avg_mae = trades_df['mae'].mean()
        avg_mfe = trades_df['mfe'].mean()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_holding_period': avg_holding_period,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'avg_mae': avg_mae,
            'avg_mfe': avg_mfe,
            'final_capital': equity.iloc[-1],
            'total_pnl': total_pnl
        }
    
    @staticmethod
    def _max_consecutive(series):
        """Calculate maximum consecutive True values"""
        max_count = 0
        current_count = 0
        for val in series:
            if val:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        return max_count


# Example Usage
if __name__ == "__main__":
    # Configure backtest
    config = BacktestConfig(
        initial_capital=100000,
        commission=0.001,
        slippage=0.0005,
        position_size=0.15,  # 15% per position
        max_positions=6,
        stop_loss_pct=0.03,  # 3% stop
        take_profit_pct=0.08  # 8% target
    )
    
    # Create backtest engine
    backtester = AdvancedBacktester(config)
    
    # Run backtest (assuming data and signals prepared)
    # results = backtester.run_backtest(data, signals)
    
    # Display results
    # for metric, value in results.items():
    #     if isinstance(value, float):
    #         if 'return' in metric or 'rate' in metric:
    #             print(f"{metric}: {value:.2%}")
    #         else:
    #             print(f"{metric}: {value:.2f}")
    #     else:
    #         print(f"{metric}: {value}")
\`\`\`

---

## Monte Carlo Simulation

Test strategy robustness by randomly resampling trades or returns.

\`\`\`python
"""
Monte Carlo Analysis for Strategy Validation
"""

class MonteCarloSimulator:
    """
    Monte Carlo simulation for trading strategies
    """
    
    def __init__(self, trades: List[Trade]):
        self.trades = trades
    
    def resample_trades(self, n_simulations=1000, block_size=1):
        """
        Randomly resample trades to test strategy robustness
        
        Args:
            n_simulations: Number of Monte Carlo runs
            block_size: Size of blocks to preserve serial correlation
        
        Returns:
            DataFrame with simulation results
        """
        results = []
        
        for sim in range(n_simulations):
            # Resample trades with replacement
            if block_size == 1:
                sampled_trades = np.random.choice(self.trades, size=len(self.trades), replace=True)
            else:
                # Block bootstrap to preserve correlation
                sampled_trades = self._block_bootstrap(self.trades, block_size)
            
            # Calculate metrics
            total_return = sum(t.return_pct for t in sampled_trades)
            sharpe = self._calculate_sharpe(sampled_trades)
            max_dd = self._calculate_max_drawdown([t.return_pct for t in sampled_trades])
            
            results.append({
                'total_return': total_return,
                'sharpe': sharpe,
                'max_drawdown': max_dd
            })
        
        return pd.DataFrame(results)
    
    def _block_bootstrap(self, trades, block_size):
        """Block bootstrap resampling"""
        n_blocks = len(trades) // block_size
        sampled_trades = []
        
        for _ in range(n_blocks):
            start_idx = np.random.randint(0, len(trades) - block_size + 1)
            block = trades[start_idx:start_idx + block_size]
            sampled_trades.extend(block)
        
        return sampled_trades[:len(trades)]
    
    @staticmethod
    def _calculate_sharpe(trades):
        """Calculate Sharpe ratio from trades"""
        returns = [t.return_pct for t in trades]
        if len(returns) == 0:
            return 0
        mean_return = np.mean(returns) * 252  # Annualize
        std_return = np.std(returns) * np.sqrt(252)
        return mean_return / std_return if std_return > 0 else 0
    
    @staticmethod
    def _calculate_max_drawdown(returns):
        """Calculate maximum drawdown"""
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max)
        return drawdown.min()
    
    def analyze_results(self, results_df):
        """Analyze Monte Carlo results"""
        analysis = {
            'mean_return': results_df['total_return'].mean(),
            'median_return': results_df['total_return'].median(),
            'std_return': results_df['total_return'].std(),
            'percentile_5': results_df['total_return'].quantile(0.05),
            'percentile_95': results_df['total_return'].quantile(0.95),
            'prob_positive': (results_df['total_return'] > 0).mean(),
            'prob_sharpe_above_1': (results_df['sharpe'] > 1.0).mean(),
            'worst_drawdown': results_df['max_drawdown'].min(),
            'best_return': results_df['total_return'].max(),
            'worst_return': results_df['total_return'].min()
        }
        
        return analysis


# Example: Run Monte Carlo
mc_simulator = MonteCarloSimulator(backtester.trades)
mc_results = mc_simulator.resample_trades(n_simulations=1000)
mc_analysis = mc_simulator.analyze_results(mc_results)

print("\\n=== Monte Carlo Analysis (1000 simulations) ===")
for metric, value in mc_analysis.items():
    if isinstance(value, float):
        if 'return' in metric or 'drawdown' in metric or 'prob' in metric:
            print(f"{metric}: {value:.2%}")
        else:
            print(f"{metric}: {value:.2f}")

# Visualize distribution
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.hist(mc_results['total_return'], bins=50, edgecolor='black', alpha=0.7)
plt.axvline(mc_analysis['mean_return'], color='red', linestyle='--', label='Mean')
plt.axvline(mc_analysis['percentile_5'], color='orange', linestyle='--', label='5th %ile')
plt.xlabel('Total Return')
plt.ylabel('Frequency')
plt.title('Return Distribution')
plt.legend()

plt.subplot(132)
plt.hist(mc_results['sharpe'], bins=50, edgecolor='black', alpha=0.7)
plt.axvline(1.0, color='red', linestyle='--', label='Sharpe=1.0')
plt.xlabel('Sharpe Ratio')
plt.title('Sharpe Distribution')
plt.legend()

plt.subplot(133)
plt.hist(mc_results['max_drawdown'], bins=50, edgecolor='black', alpha=0.7)
plt.axvline(-0.20, color='red', linestyle='--', label='-20%')
plt.xlabel('Max Drawdown')
plt.title('Drawdown Distribution')
plt.legend()

plt.tight_layout()
plt.show()
\`\`\`

---

## Walk-Forward Analysis

The gold standard for time series validation.

\`\`\`python
"""
Walk-Forward Optimization and Testing
"""

class WalkForwardAnalyzer:
    """
    Walk-forward analysis with rolling optimization
    """
    
    def __init__(self, data, strategy_class):
        self.data = data
        self.strategy_class = strategy_class
    
    def run_walk_forward(self, train_window=252, test_window=21,
                         reoptimize_freq=21, param_grid=None):
        """
        Perform walk-forward analysis
        
        Args:
            train_window: Training period (days)
            test_window: Testing period (days)
            reoptimize_freq: How often to reoptimize (days)
            param_grid: Parameter grid for optimization
        
        Returns:
            Walk-forward results
        """
        results = {
            'test_returns': [],
            'train_returns': [],
            'optimal_params': [],
            'dates': []
        }
        
        for i in range(train_window, len(self.data) - test_window, reoptimize_freq):
            # Training period
            train_data = self.data.iloc[i-train_window:i]
            
            # Optimize on training data
            if param_grid:
                best_params = self._optimize_parameters(train_data, param_grid)
            else:
                best_params = {}
            
            results['optimal_params'].append(best_params)
            
            # Test period (out-of-sample)
            test_data = self.data.iloc[i:i+test_window]
            
            # Apply strategy with optimal parameters
            strategy = self.strategy_class(**best_params)
            test_return = self._backtest_strategy(strategy, test_data)
            
            # Also calculate in-sample performance for comparison
            train_return = self._backtest_strategy(strategy, train_data)
            
            results['test_returns'].append(test_return)
            results['train_returns'].append(train_return)
            results['dates'].append(test_data.index[0])
        
        return pd.DataFrame(results)
    
    def _optimize_parameters(self, train_data, param_grid):
        """Optimize parameters on training data"""
        best_sharpe = -np.inf
        best_params = {}
        
        # Grid search
        from itertools import product
        param_combinations = list(product(*param_grid.values()))
        param_names = list(param_grid.keys())
        
        for combo in param_combinations:
            params = dict(zip(param_names, combo))
            strategy = self.strategy_class(**params)
            
            sharpe = self._backtest_strategy(strategy, train_data, return_sharpe=True)
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = params
        
        return best_params
    
    def _backtest_strategy(self, strategy, data, return_sharpe=False):
        """Backtest strategy on data"""
        signals = strategy.generate_signals(data)
        returns = signals.shift(1) * data['Close'].pct_change()
        
        if return_sharpe:
            sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            return sharpe
        else:
            return returns.sum()
    
    def analyze_walk_forward(self, results):
        """Analyze walk-forward results"""
        analysis = {
            'avg_test_return': results['test_returns'].mean(),
            'avg_train_return': results['train_returns'].mean(),
            'overfitting_ratio': results['train_returns'].mean() / results['test_returns'].mean(),
            'test_sharpe': results['test_returns'].mean() / results['test_returns'].std() * np.sqrt(12),  # Assuming monthly
            'stability': results['test_returns'].std() / abs(results['test_returns'].mean()),
            'win_rate': (results['test_returns'] > 0).mean(),
            'worst_period': results['test_returns'].min(),
            'best_period': results['test_returns'].max()
        }
        
        return analysis


# Example: Walk-forward analysis
# wf_analyzer = WalkForwardAnalyzer(data, MomentumStrategy)
# param_grid = {
#     'lookback': [10, 21, 63],
#     'threshold': [0.02, 0.05, 0.10]
# }
# 
# wf_results = wf_analyzer.run_walk_forward(
#     train_window=252,
#     test_window=21,
#     reoptimize_freq=21,
#     param_grid=param_grid
# )
# 
# wf_analysis = wf_analyzer.analyze_walk_forward(wf_results)
# 
# print("\\n=== Walk-Forward Analysis ===")
# for metric, value in wf_analysis.items():
#     if isinstance(value, float):
#         print(f"{metric}: {value:.3f}")
\`\`\`

---

## Key Takeaways

**Critical Principles**:
1. **Lookahead Bias**: Always shift features by 1 period
2. **Survivorship Bias**: Use point-in-time universe, include delisted stocks
3. **Transaction Costs**: Include commission (0.1%) + slippage (0.05%) + market impact
4. **Walk-Forward**: Train on period N, test on N+1. Never touch test set until final evaluation.
5. **Overfitting**: Limit to 3-5 parameters, test across regimes
6. **Monte Carlo**: Resample trades 1000 times, check 5th percentile > -10%
7. **Reality Check**: If Sharpe > 2.0 in backtest → probably overfit

**Expected Degradation**:
- In-sample Sharpe: 1.5
- Out-of-sample Sharpe: 1.0 (33% degradation normal)
- Live trading Sharpe: 0.7-0.8 (additional 20-30% degradation)

**Red Flags** (likely overfitting):
- Win rate > 70%
- Sharpe ratio > 3.0
- Perfect equity curve (no drawdowns)
- Works only in one market regime
- Too many parameters (>10)
- Train-test performance gap > 50%

**Backtesting Checklist**:
- ✅ Walk-forward validation
- ✅ Transaction costs included
- ✅ Tested across bull/bear/sideways markets
- ✅ Monte Carlo simulation passed
- ✅ No lookahead bias (verified with correlation test)
- ✅ Point-in-time universe (or survivorship-free data)
- ✅ < 5 parameters
- ✅ Independent test set (touched once)
- ✅ Sharpe < 2.0, Win rate < 65%
- ✅ Documented assumptions

Only after all checks passed → Consider paper trading.
`,
};
