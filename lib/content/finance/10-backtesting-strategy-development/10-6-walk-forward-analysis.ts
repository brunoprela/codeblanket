import { Content } from '@/lib/types';

const walkForwardAnalysis: Content = {
  title: 'Walk-Forward Analysis',
  description:
    'Master walk-forward optimization, rolling window backtesting, strategy degradation detection, and adaptive parameter selection for robust trading systems',
  sections: [
    {
      title: 'Introduction to Walk-Forward Analysis',
      content: `
# Introduction to Walk-Forward Analysis

Walk-forward analysis (WFA) is a robust backtesting methodology that simulates how a trading strategy would perform in real-world conditions where parameters must be optimized on past data and then applied to unseen future data.

## The Problem with Traditional Backtesting

**Case Study**: A quantitative trading firm developed a mean-reversion strategy that showed exceptional performance over 10 years of historical data with a Sharpe ratio of 2.5. They optimized parameters on the entire dataset and went live. Within 3 months, the strategy was losing money. What went wrong?

**The issue**: They optimized on all available data, including what would have been "future" data at any point in time. This is a classic case of **look-ahead bias** and **in-sample overfitting**.

### Traditional Optimization Problems

1. **Parameter Overfitting**: Finding parameters that work perfectly on historical data but fail in live trading
2. **Look-Ahead Bias**: Using information that wouldn't have been available at the decision point
3. **Static Parameters**: Assuming parameters that worked in the past will continue to work
4. **No Out-of-Sample Validation**: No way to assess future performance

## Walk-Forward Analysis Solution

Walk-forward analysis addresses these issues by:

1. **Dividing data into sequential windows**: Train on historical data, test on immediate future data
2. **Rolling optimization**: Periodically re-optimize parameters as new data becomes available
3. **Out-of-sample testing**: Each test period uses parameters from previous training, simulating real conditions
4. **Adaptive parameters**: Allows strategy to evolve with changing market conditions

### Walk-Forward Process

\`\`\`
Timeline: [==========================================]
           2020        2021        2022        2023

Window 1:  [Train1][Test1]
Window 2:           [Train2][Test2]
Window 3:                   [Train3][Test3]
Window 4:                           [Train4][Test4]
\`\`\`

## Python Implementation: Walk-Forward Framework

\`\`\`python
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)

@dataclass
class WalkForwardWindow:
    """Represents a single walk-forward window"""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    optimal_params: Dict[str, float] = field(default_factory=dict)
    train_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return (
            f"WalkForwardWindow("
            f"train={self.train_start.date()} to {self.train_end.date()}, "
            f"test={self.test_start.date()} to {self.test_end.date()})"
        )

class WalkForwardAnalyzer:
    """
    Walk-forward analysis framework for strategy optimization and validation
    """
    
    def __init__(
        self,
        strategy_func: Callable,
        optimization_metric: str = 'sharpe_ratio',
        train_period_days: int = 252,  # 1 year
        test_period_days: int = 63,    # 3 months
        step_days: int = 63,           # Step forward 3 months each iteration
        anchored: bool = False         # Anchored (expanding) vs rolling window
    ):
        """
        Initialize walk-forward analyzer
        
        Args:
            strategy_func: Function that takes (data, params) and returns trades
            optimization_metric: Metric to optimize ('sharpe_ratio', 'sortino', 'calmar')
            train_period_days: Training window size in days
            test_period_days: Testing window size in days
            step_days: Step size between windows
            anchored: If True, use anchored (expanding) windows; if False, use rolling
        """
        self.strategy_func = strategy_func
        self.optimization_metric = optimization_metric
        self.train_period_days = train_period_days
        self.test_period_days = test_period_days
        self.step_days = step_days
        self.anchored = anchored
        
        self.windows: List[WalkForwardWindow] = []
        self.all_test_results: List[Dict] = []
    
    def generate_windows(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[WalkForwardWindow]:
        """
        Generate walk-forward windows
        
        Args:
            start_date: Start date of analysis
            end_date: End date of analysis
            
        Returns:
            List of WalkForwardWindow objects
        """
        windows = []
        current_date = start_date
        anchor_date = start_date if self.anchored else None
        
        while current_date + timedelta(days=self.train_period_days + self.test_period_days) <= end_date:
            # Training period
            if self.anchored and anchor_date is not None:
                train_start = anchor_date
            else:
                train_start = current_date
            
            train_end = current_date + timedelta(days=self.train_period_days)
            
            # Testing period (immediately after training)
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=self.test_period_days)
            
            window = WalkForwardWindow(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end
            )
            
            windows.append(window)
            
            # Step forward
            current_date += timedelta(days=self.step_days)
        
        logger.info(f"Generated {len(windows)} walk-forward windows")
        return windows
    
    def optimize_parameters(
        self,
        train_data: pd.DataFrame,
        param_bounds: Dict[str, Tuple[float, float]],
        method: str = 'nelder-mead'
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Optimize strategy parameters on training data
        
        Args:
            train_data: Training data
            param_bounds: Dictionary of parameter bounds {param_name: (min, max)}
            method: Optimization method
            
        Returns:
            Tuple of (optimal_params, train_metrics)
        """
        # Initial guess (midpoint of bounds)
        initial_params = {
            param: (bounds[0] + bounds[1]) / 2
            for param, bounds in param_bounds.items()
        }
        
        def objective_function(param_values: np.ndarray) -> float:
            """Objective function to minimize (negative metric to maximize)"""
            # Convert array to parameter dictionary
            params = {
                name: value 
                for name, value in zip(param_bounds.keys(), param_values)
            }
            
            try:
                # Run strategy with these parameters
                trades = self.strategy_func(train_data, params)
                
                if len(trades) == 0:
                    return 1e6  # Penalize no trades
                
                # Calculate performance metrics
                metrics = self._calculate_metrics(trades, train_data)
                
                # Return negative of optimization metric (to minimize)
                return -metrics.get(self.optimization_metric, -1e6)
            
            except Exception as e:
                logger.warning(f"Error in objective function: {e}")
                return 1e6  # Large penalty for errors
        
        # Bounds for optimization
        bounds = [param_bounds[param] for param in param_bounds.keys()]
        
        # Run optimization
        result = minimize(
            objective_function,
            x0=list(initial_params.values()),
            method=method,
            bounds=bounds,
            options={'maxiter': 100, 'disp': False}
        )
        
        # Extract optimal parameters
        optimal_params = {
            name: value
            for name, value in zip(param_bounds.keys(), result.x)
        }
        
        # Calculate final metrics with optimal parameters
        optimal_trades = self.strategy_func(train_data, optimal_params)
        train_metrics = self._calculate_metrics(optimal_trades, train_data)
        
        logger.info(
            f"Optimized parameters: {optimal_params}, "
            f"{self.optimization_metric}={train_metrics[self.optimization_metric]:.3f}"
        )
        
        return optimal_params, train_metrics
    
    def test_parameters(
        self,
        test_data: pd.DataFrame,
        params: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Test strategy with given parameters on test data
        
        Args:
            test_data: Testing data
            params: Strategy parameters
            
        Returns:
            Dictionary of test metrics
        """
        try:
            trades = self.strategy_func(test_data, params)
            
            if len(trades) == 0:
                logger.warning("No trades generated in test period")
                return {metric: 0.0 for metric in ['sharpe_ratio', 'total_return', 'max_drawdown']}
            
            metrics = self._calculate_metrics(trades, test_data)
            return metrics
        
        except Exception as e:
            logger.error(f"Error testing parameters: {e}")
            return {metric: 0.0 for metric in ['sharpe_ratio', 'total_return', 'max_drawdown']}
    
    def run_walk_forward(
        self,
        data: pd.DataFrame,
        param_bounds: Dict[str, Tuple[float, float]],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Run complete walk-forward analysis
        
        Args:
            data: Complete dataset with 'timestamp' index
            param_bounds: Parameter bounds for optimization
            start_date: Analysis start date (default: data start)
            end_date: Analysis end date (default: data end)
            
        Returns:
            DataFrame with walk-forward results
        """
        if start_date is None:
            start_date = data.index.min()
        if end_date is None:
            end_date = data.index.max()
        
        # Generate windows
        self.windows = self.generate_windows(start_date, end_date)
        
        # Process each window
        for i, window in enumerate(self.windows):
            logger.info(f"\\nProcessing window {i+1}/{len(self.windows)}: {window}")
            
            # Extract train and test data
            train_data = data[
                (data.index >= window.train_start) &
                (data.index <= window.train_end)
            ]
            
            test_data = data[
                (data.index >= window.test_start) &
                (data.index <= window.test_end)
            ]
            
            if len(train_data) == 0 or len(test_data) == 0:
                logger.warning(f"Insufficient data for window {i+1}, skipping")
                continue
            
            # Optimize on training data
            optimal_params, train_metrics = self.optimize_parameters(
                train_data,
                param_bounds
            )
            
            # Test on out-of-sample data
            test_metrics = self.test_parameters(test_data, optimal_params)
            
            # Store results
            window.optimal_params = optimal_params
            window.train_metrics = train_metrics
            window.test_metrics = test_metrics
            
            self.all_test_results.append({
                'window': i + 1,
                'test_start': window.test_start,
                'test_end': window.test_end,
                **{f'param_{k}': v for k, v in optimal_params.items()},
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'test_{k}': v for k, v in test_metrics.items()}
            })
        
        # Compile results
        results_df = pd.DataFrame(self.all_test_results)
        
        return results_df
    
    def _calculate_metrics(
        self,
        trades: pd.DataFrame,
        price_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate performance metrics
        
        Args:
            trades: DataFrame with trade signals
            price_data: Price data
            
        Returns:
            Dictionary of metrics
        """
        if len(trades) == 0:
            return {
                'sharpe_ratio': 0.0,
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'num_trades': 0
            }
        
        # Calculate returns
        returns = trades['pnl'].values if 'pnl' in trades.columns else np.zeros(len(trades))
        
        # Sharpe ratio
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Total return
        total_return = np.sum(returns)
        
        # Maximum drawdown
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'num_trades': len(trades),
            'avg_return': returns.mean() if len(returns) > 0 else 0.0
        }
    
    def analyze_results(self) -> Dict[str, Any]:
        """
        Analyze walk-forward results
        
        Returns:
            Dictionary with analysis results
        """
        if not self.all_test_results:
            return {}
        
        df = pd.DataFrame(self.all_test_results)
        
        # In-sample vs out-of-sample comparison
        is_sharpe = df['train_sharpe_ratio'].mean()
        oos_sharpe = df['test_sharpe_ratio'].mean()
        
        # Degradation
        degradation = ((is_sharpe - oos_sharpe) / is_sharpe * 100) if is_sharpe != 0 else 0
        
        # Consistency (% of windows with positive returns)
        win_rate = (df['test_total_return'] > 0).mean() * 100
        
        # Parameter stability (coefficient of variation)
        param_cols = [col for col in df.columns if col.startswith('param_')]
        param_stability = {}
        for col in param_cols:
            param_name = col.replace('param_', '')
            cv = df[col].std() / df[col].mean() if df[col].mean() != 0 else np.inf
            param_stability[param_name] = cv
        
        return {
            'num_windows': len(self.windows),
            'in_sample_sharpe': is_sharpe,
            'out_of_sample_sharpe': oos_sharpe,
            'performance_degradation_pct': degradation,
            'win_rate_pct': win_rate,
            'avg_test_return': df['test_total_return'].mean(),
            'test_return_std': df['test_total_return'].std(),
            'parameter_stability': param_stability,
            'overfitting_warning': degradation > 30  # Flag if >30% degradation
        }


# Example: Simple Moving Average Crossover Strategy
def sma_crossover_strategy(
    data: pd.DataFrame,
    params: Dict[str, float]
) -> pd.DataFrame:
    """
    Simple moving average crossover strategy
    
    Args:
        data: DataFrame with 'close' prices
        params: Dictionary with 'fast_period' and 'slow_period'
        
    Returns:
        DataFrame with trades
    """
    fast_period = int(params['fast_period'])
    slow_period = int(params['slow_period'])
    
    # Calculate moving averages
    fast_ma = data['close'].rolling(window=fast_period).mean()
    slow_ma = data['close'].rolling(window=slow_period).mean()
    
    # Generate signals
    signals = pd.DataFrame(index=data.index)
    signals['position'] = 0
    signals['position'][fast_ma > slow_ma] = 1
    signals['position'][fast_ma <= slow_ma] = -1
    
    # Generate trades (when position changes)
    signals['trade'] = signals['position'].diff()
    trades = signals[signals['trade'] != 0].copy()
    
    # Calculate PnL (simplified)
    if len(trades) > 1:
        trades['pnl'] = data.loc[trades.index, 'close'].pct_change() * trades['position'].shift()
        trades = trades.dropna()
    else:
        trades['pnl'] = 0
    
    return trades


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    # Simulate price data with trend and noise
    trend = np.linspace(100, 150, len(dates))
    noise = np.random.randn(len(dates)) * 2
    prices = trend + np.cumsum(noise)
    
    data = pd.DataFrame({
        'close': prices
    }, index=dates)
    
    # Define parameter bounds
    param_bounds = {
        'fast_period': (5, 50),
        'slow_period': (20, 200)
    }
    
    # Run walk-forward analysis
    wfa = WalkForwardAnalyzer(
        strategy_func=sma_crossover_strategy,
        optimization_metric='sharpe_ratio',
        train_period_days=252,  # 1 year training
        test_period_days=63,    # 3 months testing
        step_days=63,           # Step forward 3 months
        anchored=False          # Rolling window
    )
    
    results = wfa.run_walk_forward(
        data=data,
        param_bounds=param_bounds
    )
    
    # Analyze results
    analysis = wfa.analyze_results()
    
    print("\\n" + "="*80)
    print("WALK-FORWARD ANALYSIS RESULTS")
    print("="*80)
    print(f"Number of windows: {analysis['num_windows']}")
    print(f"In-sample Sharpe: {analysis['in_sample_sharpe']:.3f}")
    print(f"Out-of-sample Sharpe: {analysis['out_of_sample_sharpe']:.3f}")
    print(f"Performance degradation: {analysis['performance_degradation_pct']:.1f}%")
    print(f"Win rate: {analysis['win_rate_pct']:.1f}%")
    print(f"\\nParameter Stability (CV):")
    for param, cv in analysis['parameter_stability'].items():
        print(f"  {param}: {cv:.3f}")
    
    if analysis['overfitting_warning']:
        print("\\n⚠️  WARNING: Significant performance degradation detected!")
        print("   Strategy may be overfitted to historical data.")
    
    print("\\n" + "="*80)
\`\`\`

## Key Concepts

### 1. **Rolling vs Anchored Windows**

- **Rolling**: Fixed-size window that moves forward (most recent data only)
- **Anchored**: Expanding window that grows over time (all historical data)

**Trade-offs**:
- Rolling: Adapts faster to regime changes, but less data for optimization
- Anchored: More stable estimates, but slower to adapt

### 2. **Window Sizing**

- **Training period**: Long enough for statistical significance (typically 1-3 years)
- **Testing period**: Short enough to detect degradation quickly (typically 1-3 months)
- **Step size**: Balance between overlapping data and independent tests

### 3. **Optimization Metric Selection**

Choose metrics that align with strategy goals:
- **Sharpe Ratio**: Risk-adjusted returns
- **Sortino Ratio**: Downside risk focus
- **Calmar Ratio**: Drawdown-sensitive
- **Total Return**: Pure performance (risky!)

## Production Considerations

### Computational Efficiency

Walk-forward analysis is computationally expensive:
- Multiple optimization runs
- Each optimization involves many strategy evaluations
- Long historical periods

**Solutions**:
- Parallel processing of windows
- Efficient optimization algorithms
- Caching intermediate results
- Grid search for initial parameters, then refinement
`,
    },
    {
      title: 'Advanced Walk-Forward Techniques',
      content: `
# Advanced Walk-Forward Techniques

## Adaptive Window Sizing

Fixed window sizes may not be optimal across all market regimes. Adaptive sizing adjusts based on market conditions.

\`\`\`python
from typing import Dict, List
import pandas as pd
import numpy as np
from enum import Enum

class MarketRegime(Enum):
    """Market regime classifications"""
    HIGH_VOLATILITY = "high_vol"
    LOW_VOLATILITY = "low_vol"
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"

class AdaptiveWalkForward:
    """
    Walk-forward analysis with adaptive window sizing
    """
    
    def __init__(
        self,
        strategy_func: Callable,
        base_train_days: int = 252,
        base_test_days: int = 63
    ):
        self.strategy_func = strategy_func
        self.base_train_days = base_train_days
        self.base_test_days = base_test_days
    
    def detect_regime(
        self,
        data: pd.DataFrame,
        lookback_days: int = 60
    ) -> MarketRegime:
        """
        Detect current market regime
        
        Args:
            data: Price data
            lookback_days: Period for regime detection
            
        Returns:
            Detected market regime
        """
        recent_data = data.tail(lookback_days)
        
        # Calculate volatility
        returns = recent_data['close'].pct_change()
        volatility = returns.std() * np.sqrt(252)
        
        # Calculate trend strength (R-squared of linear regression)
        x = np.arange(len(recent_data))
        y = recent_data['close'].values
        correlation = np.corrcoef(x, y)[0, 1]
        trend_strength = correlation ** 2
        
        # Classify regime
        if volatility > 0.30:  # High volatility
            return MarketRegime.HIGH_VOLATILITY
        elif volatility < 0.15:  # Low volatility
            return MarketRegime.LOW_VOLATILITY
        elif trend_strength > 0.6:  # Strong trend
            return MarketRegime.TRENDING
        else:  # Mean-reverting
            return MarketRegime.MEAN_REVERTING
    
    def get_adaptive_window_sizes(
        self,
        regime: MarketRegime
    ) -> Tuple[int, int]:
        """
        Get adaptive window sizes based on regime
        
        Args:
            regime: Current market regime
            
        Returns:
            Tuple of (train_days, test_days)
        """
        adjustments = {
            MarketRegime.HIGH_VOLATILITY: (0.7, 0.5),  # Shorter windows
            MarketRegime.LOW_VOLATILITY: (1.3, 1.5),   # Longer windows
            MarketRegime.TRENDING: (1.0, 1.0),         # Standard windows
            MarketRegime.MEAN_REVERTING: (1.2, 0.8)    # Longer train, shorter test
        }
        
        train_mult, test_mult = adjustments[regime]
        
        train_days = int(self.base_train_days * train_mult)
        test_days = int(self.base_test_days * test_mult)
        
        return train_days, test_days
    
    def run_adaptive_walk_forward(
        self,
        data: pd.DataFrame,
        param_bounds: Dict[str, Tuple[float, float]]
    ) -> pd.DataFrame:
        """
        Run walk-forward with adaptive window sizing
        
        Args:
            data: Complete dataset
            param_bounds: Parameter bounds
            
        Returns:
            Results DataFrame
        """
        results = []
        current_idx = self.base_train_days
        
        while current_idx + self.base_test_days < len(data):
            # Detect regime
            regime_data = data.iloc[:current_idx]
            regime = self.detect_regime(regime_data)
            
            # Get adaptive window sizes
            train_days, test_days = self.get_adaptive_window_sizes(regime)
            
            # Ensure we don't go back further than data allows
            train_start_idx = max(0, current_idx - train_days)
            
            # Extract data
            train_data = data.iloc[train_start_idx:current_idx]
            test_data = data.iloc[current_idx:current_idx + test_days]
            
            logger.info(
                f"Regime: {regime.value}, "
                f"Train: {len(train_data)} days, "
                f"Test: {len(test_data)} days"
            )
            
            # Optimize and test (implementation similar to WalkForwardAnalyzer)
            # ... optimization code here ...
            
            results.append({
                'regime': regime.value,
                'train_days': train_days,
                'test_days': test_days,
                # ... metrics ...
            })
            
            # Step forward
            current_idx += test_days
        
        return pd.DataFrame(results)


## Multi-Objective Optimization

Optimize for multiple objectives simultaneously (Sharpe ratio AND drawdown).

class MultiObjectiveWalkForward:
    """
    Walk-forward with multi-objective optimization
    """
    
    def __init__(
        self,
        strategy_func: Callable,
        objectives: List[str] = ['sharpe_ratio', 'max_drawdown'],
        weights: Optional[List[float]] = None
    ):
        self.strategy_func = strategy_func
        self.objectives = objectives
        
        # Default equal weights
        if weights is None:
            self.weights = [1.0 / len(objectives)] * len(objectives)
        else:
            self.weights = weights
    
    def calculate_composite_score(
        self,
        metrics: Dict[str, float]
    ) -> float:
        """
        Calculate weighted composite score
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            Composite score (higher is better)
        """
        score = 0.0
        
        for objective, weight in zip(self.objectives, self.weights):
            value = metrics.get(objective, 0.0)
            
            # Normalize metrics (convert drawdown to positive)
            if 'drawdown' in objective.lower():
                value = -value  # Less negative drawdown is better
            
            # Min-max normalization would go here in production
            score += weight * value
        
        return score
    
    def optimize_multi_objective(
        self,
        train_data: pd.DataFrame,
        param_bounds: Dict[str, Tuple[float, float]]
    ) -> Dict[str, float]:
        """
        Optimize for multiple objectives
        
        Args:
            train_data: Training data
            param_bounds: Parameter bounds
            
        Returns:
            Optimal parameters
        """
        def objective_function(param_values: np.ndarray) -> float:
            params = {
                name: value
                for name, value in zip(param_bounds.keys(), param_values)
            }
            
            trades = self.strategy_func(train_data, params)
            metrics = self._calculate_metrics(trades, train_data)
            
            # Return negative composite score (for minimization)
            return -self.calculate_composite_score(metrics)
        
        # Run optimization
        bounds = [param_bounds[param] for param in param_bounds.keys()]
        initial = [np.mean(b) for b in bounds]
        
        result = minimize(
            objective_function,
            x0=initial,
            method='L-BFGS-B',
            bounds=bounds
        )
        
        return {
            name: value
            for name, value in zip(param_bounds.keys(), result.x)
        }


## Combinatorial Walk-Forward

Test multiple parameter sets and combine predictions.

class CombinatorialWalkForward:
    """
    Combinatorial walk-forward with ensemble predictions
    """
    
    def __init__(
        self,
        strategy_func: Callable,
        num_parameter_sets: int = 5
    ):
        self.strategy_func = strategy_func
        self.num_parameter_sets = num_parameter_sets
    
    def generate_parameter_ensemble(
        self,
        train_data: pd.DataFrame,
        param_bounds: Dict[str, Tuple[float, float]]
    ) -> List[Dict[str, float]]:
        """
        Generate ensemble of parameter sets
        
        Uses multiple optimization runs with different initializations
        
        Args:
            train_data: Training data
            param_bounds: Parameter bounds
            
        Returns:
            List of parameter dictionaries
        """
        parameter_sets = []
        
        for i in range(self.num_parameter_sets):
            # Random initialization
            np.random.seed(i)
            initial_params = {
                name: np.random.uniform(bounds[0], bounds[1])
                for name, bounds in param_bounds.items()
            }
            
            # Optimize from this initialization
            # ... optimization code ...
            
            parameter_sets.append(initial_params)  # Placeholder
        
        return parameter_sets
    
    def ensemble_prediction(
        self,
        test_data: pd.DataFrame,
        parameter_ensemble: List[Dict[str, float]],
        aggregation: str = 'mean'
    ) -> pd.DataFrame:
        """
        Generate ensemble prediction
        
        Args:
            test_data: Test data
            parameter_ensemble: List of parameter sets
            aggregation: How to combine predictions ('mean', 'median', 'vote')
            
        Returns:
            Combined predictions
        """
        all_signals = []
        
        # Generate signals from each parameter set
        for params in parameter_ensemble:
            trades = self.strategy_func(test_data, params)
            signals = trades['position'] if 'position' in trades.columns else pd.Series()
            all_signals.append(signals)
        
        # Combine signals
        if aggregation == 'mean':
            combined = pd.concat(all_signals, axis=1).mean(axis=1)
        elif aggregation == 'median':
            combined = pd.concat(all_signals, axis=1).median(axis=1)
        elif aggregation == 'vote':
            # Majority vote
            combined = pd.concat(all_signals, axis=1).apply(
                lambda x: 1 if x.sum() > 0 else -1,
                axis=1
            )
        
        return combined
\`\`\`

## Robustness Testing

### Parameter Sensitivity Analysis

\`\`\`python
class ParameterSensitivityAnalyzer:
    """
    Analyze parameter sensitivity in walk-forward context
    """
    
    def __init__(self, walk_forward_results: pd.DataFrame):
        self.results = walk_forward_results
    
    def analyze_parameter_stability(
        self,
        parameter_name: str
    ) -> Dict[str, float]:
        """
        Analyze stability of a parameter across windows
        
        Args:
            parameter_name: Name of parameter to analyze
            
        Returns:
            Dictionary with stability metrics
        """
        param_col = f'param_{parameter_name}'
        
        if param_col not in self.results.columns:
            raise ValueError(f"Parameter {parameter_name} not found in results")
        
        param_values = self.results[param_col]
        
        # Calculate statistics
        mean_value = param_values.mean()
        std_value = param_values.std()
        cv = std_value / mean_value if mean_value != 0 else np.inf
        
        # Calculate transitions (how often does optimal value change significantly)
        param_diff = param_values.diff().abs()
        significant_changes = (param_diff > std_value).sum()
        change_rate = significant_changes / len(param_values)
        
        return {
            'mean': mean_value,
            'std': std_value,
            'coefficient_of_variation': cv,
            'change_rate': change_rate,
            'min': param_values.min(),
            'max': param_values.max(),
            'range': param_values.max() - param_values.min()
        }
    
    def test_parameter_perturbation(
        self,
        test_data: pd.DataFrame,
        base_params: Dict[str, float],
        perturbation_pct: float = 0.10
    ) -> Dict[str, Dict[str, float]]:
        """
        Test how strategy performs with perturbed parameters
        
        Args:
            test_data: Test data
            base_params: Base parameter set
            perturbation_pct: Perturbation percentage (e.g., 0.10 for ±10%)
            
        Returns:
            Dictionary with perturbation results
        """
        results = {}
        
        for param_name, base_value in base_params.items():
            # Test perturbations
            perturbations = {
                'base': base_value,
                f'minus_{int(perturbation_pct*100)}pct': base_value * (1 - perturbation_pct),
                f'plus_{int(perturbation_pct*100)}pct': base_value * (1 + perturbation_pct)
            }
            
            param_results = {}
            
            for pert_name, pert_value in perturbations.items():
                # Create perturbed parameter set
                perturbed_params = base_params.copy()
                perturbed_params[param_name] = pert_value
                
                # Test strategy
                # ... testing code ...
                
                # Store results
                param_results[pert_name] = {
                    'value': pert_value,
                    'sharpe': 0.0,  # Placeholder
                    'return': 0.0   # Placeholder
                }
            
            results[param_name] = param_results
        
        return results
    
    def generate_sensitivity_report(self) -> str:
        """Generate comprehensive sensitivity report"""
        report = []
        report.append("="*80)
        report.append("PARAMETER SENSITIVITY ANALYSIS")
        report.append("="*80)
        report.append("")
        
        # Analyze each parameter
        param_cols = [col for col in self.results.columns if col.startswith('param_')]
        
        for param_col in param_cols:
            param_name = param_col.replace('param_', '')
            stats = self.analyze_parameter_stability(param_name)
            
            report.append(f"Parameter: {param_name}")
            report.append(f"  Mean: {stats['mean']:.2f}")
            report.append(f"  Std Dev: {stats['std']:.2f}")
            report.append(f"  Coefficient of Variation: {stats['coefficient_of_variation']:.3f}")
            report.append(f"  Change Rate: {stats['change_rate']:.1%}")
            report.append(f"  Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
            
            # Stability assessment
            if stats['coefficient_of_variation'] < 0.2:
                report.append("  ✓ Stability: GOOD (low variation)")
            elif stats['coefficient_of_variation'] < 0.5:
                report.append("  ⚠ Stability: MODERATE (some variation)")
            else:
                report.append("  ✗ Stability: POOR (high variation)")
            
            report.append("")
        
        return "\\n".join(report)
\`\`\`

## Production Implementation

### Walk-Forward in Production

\`\`\`python
class ProductionWalkForward:
    """
    Production-ready walk-forward system
    """
    
    def __init__(
        self,
        strategy_func: Callable,
        reoptimization_frequency_days: int = 63  # Re-optimize quarterly
    ):
        self.strategy_func = strategy_func
        self.reoptimization_frequency = reoptimization_frequency_days
        
        # State management
        self.current_params: Optional[Dict[str, float]] = None
        self.last_optimization_date: Optional[datetime] = None
        self.optimization_history: List[Dict] = []
    
    def should_reoptimize(self, current_date: datetime) -> bool:
        """
        Check if reoptimization is needed
        
        Args:
            current_date: Current date
            
        Returns:
            True if reoptimization needed
        """
        if self.last_optimization_date is None:
            return True
        
        days_since = (current_date - self.last_optimization_date).days
        return days_since >= self.reoptimization_frequency
    
    def get_current_parameters(
        self,
        current_date: datetime,
        historical_data: pd.DataFrame,
        param_bounds: Dict[str, Tuple[float, float]]
    ) -> Dict[str, float]:
        """
        Get current parameters (reoptimize if needed)
        
        Args:
            current_date: Current date
            historical_data: All historical data up to current date
            param_bounds: Parameter bounds
            
        Returns:
            Current optimal parameters
        """
        if self.should_reoptimize(current_date):
            logger.info(f"Reoptimizing parameters on {current_date}")
            
            # Use recent data for optimization (e.g., last 252 days)
            train_data = historical_data.tail(252)
            
            # Optimize
            # ... optimization code ...
            
            # Update state
            self.current_params = {}  # Placeholder for optimized params
            self.last_optimization_date = current_date
            
            # Store in history
            self.optimization_history.append({
                'date': current_date,
                'params': self.current_params.copy()
            })
        
        return self.current_params
    
    def monitor_performance_degradation(
        self,
        recent_returns: pd.Series,
        lookback_days: int = 30
    ) -> Dict[str, Any]:
        """
        Monitor for performance degradation
        
        Args:
            recent_returns: Recent strategy returns
            lookback_days: Lookback period
            
        Returns:
            Degradation metrics and alerts
        """
        if len(recent_returns) < lookback_days:
            return {'alert': False, 'reason': 'Insufficient data'}
        
        recent = recent_returns.tail(lookback_days)
        
        # Calculate metrics
        sharpe = (recent.mean() / recent.std()) * np.sqrt(252)
        cumulative_return = (1 + recent).prod() - 1
        
        # Check for degradation
        alerts = []
        
        if sharpe < 0.5:
            alerts.append("Low Sharpe ratio detected")
        
        if cumulative_return < -0.05:  # -5% over lookback period
            alerts.append("Negative cumulative return")
        
        # Consecutive losing days
        losing_days = (recent < 0).sum()
        if losing_days / len(recent) > 0.7:
            alerts.append("High proportion of losing days")
        
        return {
            'alert': len(alerts) > 0,
            'alerts': alerts,
            'sharpe': sharpe,
            'cumulative_return': cumulative_return,
            'recommendation': 'Trigger immediate reoptimization' if len(alerts) >= 2 else 'Continue monitoring'
        }
\`\`\`

## Summary

Walk-forward analysis is essential for realistic strategy validation. Key points:

1. **Prevents overfitting**: Out-of-sample testing in every window
2. **Simulates reality**: Parameters optimized on past, applied to future
3. **Detects degradation**: Compare in-sample vs out-of-sample performance
4. **Adaptive**: Can adjust to changing market conditions

**Best Practices**:
- Use rolling windows for adaptive strategies
- Monitor parameter stability
- Test robustness with perturbations
- Implement automatic reoptimization in production
- Set performance degradation alerts
`,
    },
  ],
};

export default walkForwardAnalysis;
