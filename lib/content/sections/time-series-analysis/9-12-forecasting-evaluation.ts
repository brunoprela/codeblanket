export const forecastingEvaluation = {
  title: 'Forecasting and Evaluation',
  slug: 'forecasting-evaluation',
  description: 'Rigorous forecast evaluation and model comparison',
  content: `
# Forecasting and Evaluation

## Introduction: Beyond In-Sample Fit

**Proper forecast evaluation** is the cornerstone of reliable time series modeling. A model that fits historical data perfectly may fail miserably out-of-sample.

**Why evaluation matters:**
- In-sample fit ≠ forecast accuracy
- Overfitting is the norm, not the exception
- Statistical significance ≠ economic significance
- Model comparison requires proper testing
- Production models need continuous monitoring

**What you'll learn:**
- Out-of-sample testing methodologies
- Forecast error metrics (RMSE, MAE, MAPE, direction accuracy)
- Statistical tests for forecast comparison
- Rolling vs expanding windows
- Economic evaluation of forecasts
- Production monitoring frameworks

**Key insight:** The true test of a model is its out-of-sample performance, not its ability to fit historical data!

---

## Out-of-Sample Testing

### Train-Validation-Test Split

**Critical rule:** NEVER test on data used for training!

\`\`\`python
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
import matplotlib.pyplot as plt

class OutOfSampleEvaluator:
    """
    Comprehensive out-of-sample evaluation framework.
    
    Features:
    - Proper temporal splitting
    - Walk-forward validation
    - Rolling vs expanding windows
    - Multiple metrics
    """
    
    def __init__(self, data: pd.Series):
        self.data = data
        self.n = len(data)
        
    def simple_split(self, 
                    train_ratio: float = 0.7,
                    val_ratio: float = 0.15) -> dict:
        """
        Simple chronological split.
        
        Args:
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            
        Returns:
            Dictionary with train/val/test splits
        """
        train_end = int(self.n * train_ratio)
        val_end = int(self.n * (train_ratio + val_ratio))
        
        return {
            'train': self.data.iloc[:train_end],
            'validation': self.data.iloc[train_end:val_end],
            'test': self.data.iloc[val_end:],
            'train_size': train_end,
            'val_size': val_end - train_end,
            'test_size': self.n - val_end
        }
    
    def walk_forward_validation(self,
                               model_fn,
                               initial_train_size: int = 252,
                               refit_freq: int = 21,
                               forecast_horizon: int = 1) -> dict:
        """
        Walk-forward (rolling) validation.
        
        Most realistic for time series: refit periodically on expanding window.
        
        Args:
            model_fn: Function that takes training data and returns forecast
            initial_train_size: Initial training window
            refit_freq: How often to refit (in periods)
            forecast_horizon: Steps ahead to forecast
            
        Returns:
            Out-of-sample forecasts and actuals
        """
        forecasts = []
        actuals = []
        timestamps = []
        
        for t in range(initial_train_size, self.n - forecast_horizon, refit_freq):
            # Training data (expanding window)
            train = self.data.iloc[:t]
            
            # Forecast
            try:
                forecast = model_fn(train, steps=forecast_horizon)
                actual = self.data.iloc[t:t+forecast_horizon].values
                
                forecasts.append(forecast)
                actuals.append(actual)
                timestamps.append(self.data.index[t:t+forecast_horizon])
            except Exception as e:
                print(f"Error at t={t}: {e}")
                continue
        
        # Flatten for 1-step forecasts
        if forecast_horizon == 1:
            forecasts = np.array([f[0] if hasattr(f, '__len__') else f for f in forecasts])
            actuals = np.array([a[0] for a in actuals])
        
        return {
            'forecasts': forecasts,
            'actuals': actuals,
            'timestamps': timestamps,
            'n_forecasts': len(forecasts)
        }
    
    def rolling_window_cv(self,
                         model_fn,
                         window_size: int = 252,
                         step_size: int = 21) -> dict:
        """
        Rolling window cross-validation.
        
        Uses fixed-size window (adapts to regime changes faster).
        
        Args:
            model_fn: Model training and forecast function
            window_size: Size of rolling window
            step_size: How often to refit
            
        Returns:
            CV results
        """
        forecasts = []
        actuals = []
        
        for t in range(window_size, self.n - 1, step_size):
            # Fixed-size rolling window
            train = self.data.iloc[t-window_size:t]
            
            try:
                forecast = model_fn(train, steps=1)
                actual = self.data.iloc[t]
                
                forecasts.append(forecast)
                actuals.append(actual)
            except:
                continue
        
        return {
            'forecasts': np.array(forecasts),
            'actuals': np.array(actuals)
        }


# Example: Walk-forward validation
print("=== Walk-Forward Validation Example ===\\n")

# Generate AR(1) data
np.random.seed(42)
n = 1000
ar_coef = 0.7
data = np.zeros(n)
data[0] = np.random.randn()

for t in range(1, n):
    data[t] = ar_coef * data[t-1] + np.random.randn()

data_series = pd.Series(data)

# Simple AR(1) forecast function
def ar1_forecast(train_data, steps=1):
    from statsmodels.tsa.ar_model import AutoReg
    model = AutoReg(train_data, lags=1).fit()
    return model.forecast(steps=steps)

# Evaluate
evaluator = OutOfSampleEvaluator(data_series)
results = evaluator.walk_forward_validation(
    ar1_forecast,
    initial_train_size=100,
    refit_freq=20,
    forecast_horizon=1
)

print(f"Number of out-of-sample forecasts: {results['n_forecasts']}")
print(f"Forecast period: {len(results['forecasts'])} observations")
\`\`\`

---

## Forecast Error Metrics

### Point Forecast Accuracy

\`\`\`python
class ForecastMetrics:
    """
    Comprehensive forecast accuracy metrics.
    """
    
    def __init__(self, actuals: np.ndarray, forecasts: np.ndarray):
        self.actuals = actuals
        self.forecasts = forecasts
        self.errors = forecasts - actuals
        
    def rmse(self) -> float:
        """Root Mean Squared Error."""
        return np.sqrt(mean_squared_error(self.actuals, self.forecasts))
    
    def mae(self) -> float:
        """Mean Absolute Error."""
        return mean_absolute_error(self.actuals, self.forecasts)
    
    def mape(self) -> float:
        """Mean Absolute Percentage Error."""
        # Avoid division by zero
        mask = self.actuals != 0
        if mask.sum() == 0:
            return np.nan
        return np.mean(np.abs(self.errors[mask] / self.actuals[mask])) * 100
    
    def bias(self) -> float:
        """Mean forecast error (bias)."""
        return np.mean(self.errors)
    
    def direction_accuracy(self) -> float:
        """
        Fraction of times forecast direction was correct.
        
        Critical for trading!
        """
        if len(self.actuals) < 2:
            return np.nan
        
        actual_direction = np.sign(np.diff(self.actuals))
        forecast_direction = np.sign(np.diff(self.forecasts))
        
        return np.mean(actual_direction == forecast_direction)
    
    def hit_rate(self, confidence: float = 0.95, std: float = None) -> float:
        """
        Fraction of actuals within forecast confidence interval.
        
        For well-calibrated forecasts: hit_rate ≈ confidence
        """
        if std is None:
            std = np.std(self.errors)
        
        z_crit = stats.norm.ppf((1 + confidence) / 2)
        
        within_ci = np.abs(self.errors) <= z_crit * std
        
        return np.mean(within_ci)
    
    def theil_u(self) -> float:
        """
        Theil's U statistic.
        
        Compares to naive (no-change) forecast.
        U < 1: Better than naive
        U = 1: Same as naive
        U > 1: Worse than naive
        """
        naive_forecast = np.roll(self.actuals, 1)[1:]
        naive_error = self.actuals[1:] - naive_forecast
        
        rmse_model = np.sqrt(np.mean(self.errors[1:]**2))
        rmse_naive = np.sqrt(np.mean(naive_error**2))
        
        return rmse_model / rmse_naive if rmse_naive != 0 else np.inf
    
    def all_metrics(self) -> dict:
        """Calculate all metrics."""
        return {
            'rmse': self.rmse(),
            'mae': self.mae(),
            'mape': self.mape(),
            'bias': self.bias(),
            'direction_accuracy': self.direction_accuracy(),
            'hit_rate_95': self.hit_rate(0.95),
            'theil_u': self.theil_u()
        }
    
    def print_summary(self):
        """Print formatted summary."""
        metrics = self.all_metrics()
        
        print("\\nForecast Evaluation Metrics:")
        print("-" * 50)
        print(f"  RMSE:                {metrics['rmse']:.4f}")
        print(f"  MAE:                 {metrics['mae']:.4f}")
        print(f"  MAPE:                {metrics['mape']:.2f}%")
        print(f"  Bias:                {metrics['bias']:.4f}")
        print(f"  Direction Accuracy:  {metrics['direction_accuracy']*100:.1f}%")
        print(f"  95% Hit Rate:        {metrics['hit_rate_95']*100:.1f}%")
        print(f"  Theil U:             {metrics['theil_u']:.3f}")
        
        # Interpretation
        print("\\nInterpretation:")
        if abs(metrics['bias']) > 0.1 * metrics['rmse']:
            print("  ⚠ Significant bias detected")
        if metrics['direction_accuracy'] < 0.55:
            print("  ⚠ Poor direction accuracy (little better than random)")
        if metrics['hit_rate_95'] < 0.90 or metrics['hit_rate_95'] > 0.98:
            print("  ⚠ Miscalibrated confidence intervals")
        if metrics['theil_u'] > 1:
            print("  ⚠ Forecast worse than naive (no-change) benchmark!")


# Example
metrics = ForecastMetrics(results['actuals'], results['forecasts'])
metrics.print_summary()
\`\`\`

---

## Model Comparison Tests

### Diebold-Mariano Test

Tests if two forecast methods have equal predictive accuracy.

**Null hypothesis:** $E[L(e_{1t})] = E[L(e_{2t})]$ (equal expected loss)

\`\`\`python
def diebold_mariano_test(errors1: np.ndarray,
                        errors2: np.ndarray,
                        loss_fn: str = 'squared',
                        h: int = 1) -> dict:
    """
    Diebold-Mariano test for forecast comparison.
    
    Args:
        errors1: Forecast errors from model 1
        errors2: Forecast errors from model 2
        loss_fn: Loss function ('squared' or 'absolute')
        h: Forecast horizon (for HAC standard errors)
        
    Returns:
        Test results
    """
    # Calculate losses
    if loss_fn == 'squared':
        loss1 = errors1 ** 2
        loss2 = errors2 ** 2
    elif loss_fn == 'absolute':
        loss1 = np.abs(errors1)
        loss2 = np.abs(errors2)
    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")
    
    # Loss differential
    d = loss1 - loss2
    
    # Mean loss differential
    d_mean = np.mean(d)
    
    # Standard error (with HAC correction for h > 1)
    if h == 1:
        d_std = np.std(d, ddof=1) / np.sqrt(len(d))
    else:
        # Newey-West HAC standard error
        d_std = newey_west_se(d, h)
    
    # DM statistic
    dm_stat = d_mean / d_std
    
    # P-value (two-tailed)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    
    return {
        'dm_statistic': dm_stat,
        'p_value': p_value,
        'mean_loss_diff': d_mean,
        'reject_equal_accuracy': p_value < 0.05,
        'interpretation': (
            f"Model 1 {'significantly better' if dm_stat < -1.96 else 'significantly worse' if dm_stat > 1.96 else 'not significantly different'} "
            f"than Model 2 (p={p_value:.4f})"
        )
    }

def newey_west_se(data: np.ndarray, lags: int) -> float:
    """
    Newey-West HAC standard error.
    
    Args:
        data: Data series
        lags: Number of lags for autocorrelation
        
    Returns:
        HAC standard error
    """
    n = len(data)
    data_demean = data - np.mean(data)
    
    # Variance
    var = np.sum(data_demean ** 2) / n
    
    # Autocovariances
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)  # Bartlett kernel
        autocov = np.sum(data_demean[lag:] * data_demean[:-lag]) / n
        var += 2 * weight * autocov
    
    return np.sqrt(var / n)


# Example: Compare AR(1) vs AR(2)
print("\\n=== Model Comparison: Diebold-Mariano Test ===\\n")

# Fit AR(2)
def ar2_forecast(train_data, steps=1):
    from statsmodels.tsa.ar_model import AutoReg
    model = AutoReg(train_data, lags=2).fit()
    return model.forecast(steps=steps)

results_ar2 = evaluator.walk_forward_validation(
    ar2_forecast,
    initial_train_size=100,
    refit_freq=20
)

# Compare
errors_ar1 = results['forecasts'] - results['actuals']
errors_ar2 = results_ar2['forecasts'] - results_ar2['actuals']

dm_test = diebold_mariano_test(errors_ar1, errors_ar2)
print(dm_test['interpretation'])
\`\`\`

---

## Economic Evaluation

### Utility-Based Evaluation

Statistical accuracy ≠ economic value!

\`\`\`python
class EconomicEvaluation:
    """
    Evaluate forecasts using economic criteria.
    
    Focus: Trading profitability, not just statistical fit.
    """
    
    def __init__(self,
                 forecasts: np.ndarray,
                 actuals: np.ndarray,
                 returns: np.ndarray = None):
        self.forecasts = forecasts
        self.actuals = actuals
        self.returns = returns if returns is not None else np.diff(actuals) / actuals[:-1]
        
    def trading_strategy_returns(self,
                                 threshold: float = 0.0,
                                 transaction_cost: float = 0.001) -> dict:
        """
        Evaluate forecast as trading signal.
        
        Args:
            threshold: Minimum forecast magnitude to trade
            transaction_cost: Transaction cost (fraction)
            
        Returns:
            Strategy performance
        """
        # Trading signals
        signals = np.sign(self.forecasts[:-1])  # Align with returns
        signals[np.abs(self.forecasts[:-1]) < threshold] = 0  # Filter weak signals
        
        # Strategy returns (before costs)
        gross_returns = signals * self.returns
        
        # Transaction costs
        turnover = np.abs(np.diff(np.concatenate([[0], signals])))
        costs = turnover * transaction_cost
        
        # Net returns
        net_returns = gross_returns - costs[1:]
        
        # Performance metrics
        total_return = np.prod(1 + net_returns) - 1
        sharpe = np.mean(net_returns) / np.std(net_returns) * np.sqrt(252) if np.std(net_returns) > 0 else 0
        
        # Maximum drawdown
        cum_returns = np.cumprod(1 + net_returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Win rate
        wins = net_returns[net_returns > 0]
        losses = net_returns[net_returns < 0]
        win_rate = len(wins) / len(net_returns) if len(net_returns) > 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': (1 + total_return) ** (252 / len(net_returns)) - 1,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_win': np.mean(wins) if len(wins) > 0 else 0,
            'avg_loss': np.mean(losses) if len(losses) > 0 else 0,
            'profit_factor': -np.sum(wins) / np.sum(losses) if len(losses) > 0 else np.inf
        }
    
    def certainty_equivalent_return(self,
                                    risk_aversion: float = 2.0) -> float:
        """
        Certainty equivalent return (CEQ).
        
        Accounts for risk aversion:
        CEQ = μ - (γ/2) * σ²
        
        Args:
            risk_aversion: Risk aversion parameter (γ)
            
        Returns:
            Certainty equivalent return
        """
        strategy_perf = self.trading_strategy_returns()
        
        mu = strategy_perf['annual_return']
        sigma_sq = (strategy_perf['sharpe_ratio'] / np.sqrt(252)) ** 2
        
        ceq = mu - (risk_aversion / 2) * sigma_sq
        
        return ceq


# Example: Economic evaluation
print("\\n=== Economic Evaluation ===\\n")

econ_eval = EconomicEvaluation(
    results['forecasts'],
    results['actuals']
)

strategy_perf = econ_eval.trading_strategy_returns(
    threshold=0.0,
    transaction_cost=0.001
)

print("Trading Strategy Performance:")
print(f"  Total Return: {strategy_perf['total_return']*100:.2f}%")
print(f"  Annual Return: {strategy_perf['annual_return']*100:.2f}%")
print(f"  Sharpe Ratio: {strategy_perf['sharpe_ratio']:.2f}")
print(f"  Max Drawdown: {strategy_perf['max_drawdown']*100:.2f}%")
print(f"  Win Rate: {strategy_perf['win_rate']*100:.1f}%")
print(f"  Profit Factor: {strategy_perf['profit_factor']:.2f}")

ceq = econ_eval.certainty_equivalent_return(risk_aversion=2.0)
print(f"\\nCertainty Equivalent Return: {ceq*100:.2f}%/year")
\`\`\`

---

## Production Monitoring

\`\`\`python
class ForecastMonitoring:
    """
    Real-time forecast monitoring system.
    
    Tracks model performance and alerts on degradation.
    """
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.history = []
        
    def update(self, forecast: float, actual: float):
        """Add new forecast-actual pair."""
        error = forecast - actual
        self.history.append({
            'forecast': forecast,
            'actual': actual,
            'error': error,
            'squared_error': error ** 2,
            'abs_error': abs(error)
        })
        
        # Keep only recent window
        if len(self.history) > self.window_size * 2:
            self.history = self.history[-self.window_size * 2:]
    
    def check_degradation(self) -> dict:
        """
        Check if performance degrading.
        
        Returns:
            Alert status and metrics
        """
        if len(self.history) < self.window_size * 2:
            return {'alert': False, 'reason': 'Insufficient data'}
        
        # Split into old and recent
        old = self.history[:self.window_size]
        recent = self.history[-self.window_size:]
        
        # Compare RMSE
        rmse_old = np.sqrt(np.mean([h['squared_error'] for h in old]))
        rmse_recent = np.sqrt(np.mean([h['squared_error'] for h in recent]))
        
        # Alert if recent RMSE > 1.5x old
        degradation_ratio = rmse_recent / rmse_old if rmse_old > 0 else 1.0
        
        alert = degradation_ratio > 1.5
        
        return {
            'alert': alert,
            'rmse_old': rmse_old,
            'rmse_recent': rmse_recent,
            'degradation_ratio': degradation_ratio,
            'recommendation': 'Retrain model' if alert else 'Continue monitoring'
        }


# Example: Monitoring
print("\\n=== Forecast Monitoring ===\\n")

monitor = ForecastMonitoring(window_size=30)

# Simulate deteriorating performance
for t in range(100):
    # Model gets worse over time
    noise_level = 0.1 + 0.01 * t
    forecast = data_series.iloc[t] + np.random.randn() * noise_level
    actual = data_series.iloc[t]
    
    monitor.update(forecast, actual)

# Check degradation
status = monitor.check_degradation()
print(f"Alert: {status['alert']}")
print(f"Degradation ratio: {status['degradation_ratio']:.2f}x")
print(f"Recommendation: {status['recommendation']}")
\`\`\`

---

## Summary

**Key Takeaways:**

1. **Out-of-sample testing**: Critical for model validation
2. **Walk-forward**: Most realistic evaluation methodology
3. **Multiple metrics**: RMSE, MAE, direction accuracy, economic value
4. **Model comparison**: Use Diebold-Mariano test for statistical significance
5. **Economic evaluation**: Trading performance > statistical fit
6. **Monitoring**: Continuous performance tracking in production

**Best Practices:**
- Always use proper temporal splits (no leakage!)
- Test on multiple time periods (including crises)
- Compare to simple benchmarks (naive, historical mean)
- Evaluate economically (Sharpe, profit factor)
- Monitor continuously in production

**Next:** Apply everything in the final project!
`,
};
