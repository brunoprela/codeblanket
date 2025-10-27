export const strategyPerformanceAttribution = {
  title: 'Strategy Performance Attribution',
  slug: 'strategy-performance-attribution',
  description:
    'Analyze and decompose strategy returns: factor attribution, alpha/beta, risk decomposition',
  content: `
# Strategy Performance Attribution

## Introduction: Understanding the "Why" Behind Returns

Performance attribution answers the critical question: "Why did my strategy make or lose money?" It decomposes returns into components—market exposure, factor tilts, stock selection, timing—to understand what's working and what isn't. For algorithmic traders, attribution is essential for diagnosing strategies, allocating capital, and demonstrating skill to investors.

**What you'll learn:**
- Return decomposition (alpha vs. beta)
- Factor attribution (Fama-French, Carhart)
- Brinson attribution (allocation vs. selection)
- Risk-adjusted performance metrics (Sharpe, Information Ratio, Sortino)
- Transaction cost attribution
- Drawdown analysis
- Style analysis

**Why this matters for engineers:**
- Quantifiable diagnostics (which part of strategy works?)
- A/B testing for strategies (compare versions)
- Capital allocation (allocate more to high-alpha strategies)
- Debugging (find sources of underperformance)
- Investor reporting (explain returns)

**Key Metrics:**
- **Alpha**: Return above benchmark (skill)
- **Beta**: Market exposure (systematic risk)
- **Information Ratio**: Alpha / tracking error (risk-adjusted alpha)
- **Sharpe Ratio**: (Return - RFR) / volatility
- **Sortino Ratio**: (Return - RFR) / downside deviation

---

## Return Decomposition: Alpha vs. Beta

### The Capital Asset Pricing Model (CAPM)

**Formula:**
\`\`\`
R_portfolio = R_f + β(R_market - R_f) + α
\`\`\`

**Components:**
- **R_f**: Risk-free rate (T-bills)
- **β**: Market sensitivity (1.0 = moves with market)
- **R_market - R_f**: Market excess return
- **α**: Alpha (skill-based return)

**Example:**
\`\`\`
Portfolio return = 15%
Market return = 10%
Risk-free rate = 2%
Beta = 1.2

Expected return = 2% + 1.2 × (10% - 2%) = 11.6%
Alpha = 15% - 11.6% = 3.4% ✓ (positive alpha!)
\`\`\`

**Interpretation:**
- **α > 0**: Outperformance (skill)
- **α < 0**: Underperformance
- **β > 1**: More volatile than market (aggressive)
- **β < 1**: Less volatile (defensive)

\`\`\`python
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats
import logging

@dataclass
class PerformanceMetrics:
    """
    Comprehensive performance metrics
    """
    total_return: float
    annualized_return: float
    volatility: float  # Annualized
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    
    # Risk-adjusted
    alpha: float
    beta: float
    information_ratio: float
    
    # Factor exposures
    factor_exposures: Dict[str, float]
    
    def __str__(self) -> str:
        return f"""
Performance Metrics:
  Total Return: {self.total_return:.2%}
  Annualized Return: {self.annualized_return:.2%}
  Volatility: {self.volatility:.2%}
  Sharpe Ratio: {self.sharpe_ratio:.2f}
  Sortino Ratio: {self.sortino_ratio:.2f}
  Max Drawdown: {self.max_drawdown:.2%}
  Calmar Ratio: {self.calmar_ratio:.2f}
  Win Rate: {self.win_rate:.2%}
  
Risk-Adjusted:
  Alpha: {self.alpha:.2%}
  Beta: {self.beta:.2f}
  Information Ratio: {self.information_ratio:.2f}
        """

class PerformanceAnalyzer:
    """
    Comprehensive performance analysis and attribution
    """
    
    def __init__(self, risk_free_rate: float = 0.03):
        """
        Initialize analyzer
        
        Args:
            risk_free_rate: Annual risk-free rate (default 3%)
        """
        self.risk_free_rate = risk_free_rate
        self.logger = logging.getLogger(__name__)
    
    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """
        Calculate returns from prices
        
        Args:
            prices: Price series
            
        Returns:
            Return series
        """
        return prices.pct_change().dropna()
    
    def calculate_alpha_beta(self,
                            strategy_returns: pd.Series,
                            benchmark_returns: pd.Series) -> Tuple[float, float]:
        """
        Calculate alpha and beta via linear regression
        
        Strategy = α + β × Benchmark + ε
        
        Args:
            strategy_returns: Strategy returns
            benchmark_returns: Benchmark returns
            
        Returns:
            (alpha, beta)
        """
        # Align series
        aligned = pd.DataFrame({
            'strategy': strategy_returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(aligned) < 2:
            return 0.0, 1.0
        
        # Regression
        X = aligned['benchmark'].values
        y = aligned['strategy'].values
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
        
        beta = slope
        alpha = intercept
        
        # Annualize alpha (assumes daily returns)
        alpha_annual = alpha * 252
        
        return alpha_annual, beta
    
    def calculate_sharpe_ratio(self,
                              returns: pd.Series,
                              periods_per_year: int = 252) -> float:
        """
        Calculate Sharpe ratio
        
        Sharpe = (Return - RFR) / Volatility
        
        Args:
            returns: Return series
            periods_per_year: Trading periods per year (252 for daily)
            
        Returns:
            Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0
        
        # Annualize
        mean_return = returns.mean() * periods_per_year
        volatility = returns.std() * np.sqrt(periods_per_year)
        
        if volatility == 0:
            return 0.0
        
        sharpe = (mean_return - self.risk_free_rate) / volatility
        
        return sharpe
    
    def calculate_sortino_ratio(self,
                               returns: pd.Series,
                               periods_per_year: int = 252) -> float:
        """
        Calculate Sortino ratio (penalizes downside volatility only)
        
        Sortino = (Return - RFR) / Downside Deviation
        
        Args:
            returns: Return series
            periods_per_year: Trading periods per year
            
        Returns:
            Sortino ratio
        """
        if len(returns) < 2:
            return 0.0
        
        # Annualize return
        mean_return = returns.mean() * periods_per_year
        
        # Downside deviation (only negative returns)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf
        
        downside_deviation = downside_returns.std() * np.sqrt(periods_per_year)
        
        if downside_deviation == 0:
            return np.inf
        
        sortino = (mean_return - self.risk_free_rate) / downside_deviation
        
        return sortino
    
    def calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """
        Calculate maximum drawdown
        
        Args:
            equity_curve: Cumulative equity
            
        Returns:
            Max drawdown (negative)
        """
        # Calculate running maximum
        running_max = equity_curve.expanding().max()
        
        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max
        
        # Maximum drawdown
        max_dd = drawdown.min()
        
        return max_dd
    
    def calculate_calmar_ratio(self,
                              returns: pd.Series,
                              equity_curve: pd.Series,
                              periods_per_year: int = 252) -> float:
        """
        Calculate Calmar ratio
        
        Calmar = Annualized Return / |Max Drawdown|
        
        Args:
            returns: Return series
            equity_curve: Cumulative equity
            periods_per_year: Trading periods per year
            
        Returns:
            Calmar ratio
        """
        annualized_return = returns.mean() * periods_per_year
        max_dd = abs(self.calculate_max_drawdown(equity_curve))
        
        if max_dd == 0:
            return np.inf
        
        return annualized_return / max_dd
    
    def calculate_information_ratio(self,
                                   strategy_returns: pd.Series,
                                   benchmark_returns: pd.Series,
                                   periods_per_year: int = 252) -> float:
        """
        Calculate Information Ratio
        
        IR = (Return - Benchmark) / Tracking Error
        
        Measures risk-adjusted alpha
        
        Args:
            strategy_returns: Strategy returns
            benchmark_returns: Benchmark returns
            periods_per_year: Trading periods per year
            
        Returns:
            Information ratio
        """
        # Excess returns
        excess_returns = strategy_returns - benchmark_returns
        
        # Annualize
        mean_excess = excess_returns.mean() * periods_per_year
        tracking_error = excess_returns.std() * np.sqrt(periods_per_year)
        
        if tracking_error == 0:
            return 0.0
        
        ir = mean_excess / tracking_error
        
        return ir
    
    def calculate_win_rate(self, returns: pd.Series) -> float:
        """
        Calculate win rate (% of positive periods)
        
        Args:
            returns: Return series
            
        Returns:
            Win rate (0-1)
        """
        if len(returns) == 0:
            return 0.0
        
        wins = (returns > 0).sum()
        total = len(returns)
        
        return wins / total
    
    def analyze_performance(self,
                          strategy_returns: pd.Series,
                          benchmark_returns: pd.Series) -> PerformanceMetrics:
        """
        Comprehensive performance analysis
        
        Args:
            strategy_returns: Strategy returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Performance metrics
        """
        # Calculate alpha/beta
        alpha, beta = self.calculate_alpha_beta(strategy_returns, benchmark_returns)
        
        # Calculate equity curve
        equity_curve = (1 + strategy_returns).cumprod()
        
        # Calculate metrics
        total_return = equity_curve.iloc[-1] - 1
        annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
        volatility = strategy_returns.std() * np.sqrt(252)
        
        sharpe = self.calculate_sharpe_ratio(strategy_returns)
        sortino = self.calculate_sortino_ratio(strategy_returns)
        max_dd = self.calculate_max_drawdown(equity_curve)
        calmar = self.calculate_calmar_ratio(strategy_returns, equity_curve)
        win_rate = self.calculate_win_rate(strategy_returns)
        ir = self.calculate_information_ratio(strategy_returns, benchmark_returns)
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            win_rate=win_rate,
            alpha=alpha,
            beta=beta,
            information_ratio=ir,
            factor_exposures={}  # Would calculate separately
        )

class FactorAttributionAnalyzer:
    """
    Factor-based performance attribution (Fama-French, Carhart)
    """
    
    def __init__(self):
        """Initialize factor attribution analyzer"""
        self.logger = logging.getLogger(__name__)
    
    def calculate_factor_exposures(self,
                                  strategy_returns: pd.Series,
                                  factor_returns: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate factor exposures via multiple regression
        
        R_strategy = α + β_market × R_market + β_SMB × SMB + β_HML × HML + β_MOM × MOM + ε
        
        Args:
            strategy_returns: Strategy returns
            factor_returns: DataFrame with factor returns (columns: market, SMB, HML, MOM, etc.)
            
        Returns:
            Dict of factor exposures (betas)
        """
        # Align data
        data = pd.DataFrame({'strategy': strategy_returns})
        data = data.join(factor_returns, how='inner')
        data = data.dropna()
        
        if len(data) < len(factor_returns.columns) + 1:
            self.logger.warning("Insufficient data for factor regression")
            return {}
        
        # Regression
        y = data['strategy'].values
        X = data[factor_returns.columns].values
        
        # Add constant (alpha)
        X_with_const = np.column_stack([np.ones(len(X)), X])
        
        # OLS: β = (X'X)^(-1) X'y
        try:
            betas = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            self.logger.error("Singular matrix in factor regression")
            return {}
        
        # Extract alpha and factor betas
        alpha = betas[0] * 252  # Annualize
        factor_betas = betas[1:]
        
        # Create result dict
        exposures = {'alpha': alpha}
        for i, factor_name in enumerate(factor_returns.columns):
            exposures[factor_name] = factor_betas[i]
        
        return exposures
    
    def decompose_returns(self,
                         strategy_returns: pd.Series,
                         factor_returns: pd.DataFrame,
                         factor_exposures: Dict[str, float]) -> Dict[str, float]:
        """
        Decompose returns into factor contributions
        
        Args:
            strategy_returns: Strategy returns
            factor_returns: Factor returns
            factor_exposures: Factor betas
            
        Returns:
            Dict of factor contributions to return
        """
        contributions = {}
        
        # Alpha contribution
        alpha = factor_exposures.get('alpha', 0)
        contributions['alpha'] = alpha
        
        # Factor contributions
        for factor_name in factor_returns.columns:
            if factor_name in factor_exposures:
                beta = factor_exposures[factor_name]
                factor_return = factor_returns[factor_name].mean() * 252  # Annualize
                contribution = beta * factor_return
                contributions[factor_name] = contribution
        
        return contributions

class BrinsonAttribution:
    """
    Brinson attribution: Decompose returns into allocation and selection
    
    Used for portfolio managers with sector/asset allocation decisions
    """
    
    def __init__(self):
        """Initialize Brinson attribution"""
        pass
    
    def calculate_attribution(self,
                            portfolio_weights: Dict[str, float],
                            portfolio_returns: Dict[str, float],
                            benchmark_weights: Dict[str, float],
                            benchmark_returns: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate Brinson attribution
        
        Components:
        1. Allocation Effect: (w_p - w_b) × (R_b - R_benchmark)
        2. Selection Effect: w_b × (R_p - R_b)
        3. Interaction Effect: (w_p - w_b) × (R_p - R_b)
        
        Args:
            portfolio_weights: Portfolio sector weights
            portfolio_returns: Portfolio sector returns
            benchmark_weights: Benchmark sector weights
            benchmark_returns: Benchmark sector returns
            
        Returns:
            Attribution components
        """
        # Calculate benchmark total return
        benchmark_total = sum(
            benchmark_weights[sector] * benchmark_returns[sector]
            for sector in benchmark_weights
        )
        
        allocation_effect = 0.0
        selection_effect = 0.0
        interaction_effect = 0.0
        
        # Calculate effects for each sector
        sectors = set(portfolio_weights.keys()) | set(benchmark_weights.keys())
        
        for sector in sectors:
            w_p = portfolio_weights.get(sector, 0.0)
            w_b = benchmark_weights.get(sector, 0.0)
            r_p = portfolio_returns.get(sector, 0.0)
            r_b = benchmark_returns.get(sector, 0.0)
            
            # Allocation effect
            allocation_effect += (w_p - w_b) * (r_b - benchmark_total)
            
            # Selection effect
            selection_effect += w_b * (r_p - r_b)
            
            # Interaction effect
            interaction_effect += (w_p - w_b) * (r_p - r_b)
        
        return {
            'allocation': allocation_effect,
            'selection': selection_effect,
            'interaction': interaction_effect,
            'total': allocation_effect + selection_effect + interaction_effect
        }

class TransactionCostAttribution:
    """
    Attribute performance impact of transaction costs
    """
    
    def __init__(self):
        """Initialize transaction cost attribution"""
        pass
    
    def calculate_cost_impact(self,
                            gross_returns: pd.Series,
                            net_returns: pd.Series) -> Dict[str, float]:
        """
        Calculate transaction cost impact
        
        Args:
            gross_returns: Returns before costs
            net_returns: Returns after costs
            
        Returns:
            Cost attribution
        """
        # Total cost
        total_cost = (gross_returns - net_returns).sum()
        
        # Annualized
        periods_per_year = 252
        annualized_cost = total_cost * periods_per_year / len(gross_returns)
        
        # Impact on Sharpe ratio
        gross_sharpe = (gross_returns.mean() * 252) / (gross_returns.std() * np.sqrt(252))
        net_sharpe = (net_returns.mean() * 252) / (net_returns.std() * np.sqrt(252))
        sharpe_impact = gross_sharpe - net_sharpe
        
        return {
            'total_cost': total_cost,
            'annualized_cost': annualized_cost,
            'sharpe_impact': sharpe_impact,
            'cost_as_pct_of_return': annualized_cost / (gross_returns.mean() * 252) if gross_returns.mean() != 0 else 0
        }

class DrawdownAnalyzer:
    """
    Detailed drawdown analysis
    """
    
    def __init__(self):
        """Initialize drawdown analyzer"""
        pass
    
    def calculate_drawdown_series(self, equity_curve: pd.Series) -> pd.Series:
        """
        Calculate drawdown series
        
        Args:
            equity_curve: Cumulative equity
            
        Returns:
            Drawdown series
        """
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        return drawdown
    
    def find_drawdown_periods(self,
                             equity_curve: pd.Series,
                             threshold: float = -0.05) -> List[Dict]:
        """
        Identify distinct drawdown periods
        
        Args:
            equity_curve: Cumulative equity
            threshold: Minimum drawdown to consider (default -5%)
            
        Returns:
            List of drawdown period info
        """
        drawdown = self.calculate_drawdown_series(equity_curve)
        
        periods = []
        in_drawdown = False
        start_idx = None
        
        for i, dd in enumerate(drawdown):
            if dd < threshold and not in_drawdown:
                # Start of drawdown
                in_drawdown = True
                start_idx = i
            elif dd >= 0 and in_drawdown:
                # End of drawdown
                in_drawdown = False
                
                # Record period
                dd_period = drawdown.iloc[start_idx:i]
                periods.append({
                    'start': equity_curve.index[start_idx],
                    'end': equity_curve.index[i],
                    'duration': i - start_idx,
                    'max_drawdown': dd_period.min(),
                    'recovery': equity_curve.iloc[i] / equity_curve.iloc[start_idx] - 1
                })
        
        # Sort by magnitude
        periods.sort(key=lambda x: x['max_drawdown'])
        
        return periods

# Example usage
if __name__ == "__main__":
    print("\\n=== Performance Attribution System ===\\n")
    
    # 1. Generate sample returns
    print("1. Sample Strategy Performance")
    
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    # Strategy returns (outperforms with higher vol)
    strategy_returns = pd.Series(
        np.random.normal(0.0008, 0.015, len(dates)),
        index=dates
    )
    
    # Benchmark returns
    benchmark_returns = pd.Series(
        np.random.normal(0.0005, 0.012, len(dates)),
        index=dates
    )
    
    print(f"   Period: {dates[0].date()} to {dates[-1].date()}")
    print(f"   Trading Days: {len(dates)}")
    
    # 2. Calculate performance metrics
    print("\\n2. Performance Metrics")
    
    analyzer = PerformanceAnalyzer(risk_free_rate=0.03)
    metrics = analyzer.analyze_performance(strategy_returns, benchmark_returns)
    
    print(metrics)
    
    # 3. Factor attribution
    print("\\n3. Factor Attribution (Fama-French)")
    
    # Generate sample factor returns
    factor_returns = pd.DataFrame({
        'market': benchmark_returns,
        'SMB': pd.Series(np.random.normal(0.0002, 0.008, len(dates)), index=dates),
        'HML': pd.Series(np.random.normal(0.0001, 0.007, len(dates)), index=dates),
        'MOM': pd.Series(np.random.normal(0.0003, 0.010, len(dates)), index=dates)
    })
    
    factor_analyzer = FactorAttributionAnalyzer()
    exposures = factor_analyzer.calculate_factor_exposures(strategy_returns, factor_returns)
    
    print("   Factor Exposures:")
    for factor, beta in exposures.items():
        if factor == 'alpha':
            print(f"     Alpha: {beta:.2%} (annualized)")
        else:
            print(f"     {factor.upper()}: {beta:.2f}")
    
    # 4. Return decomposition
    contributions = factor_analyzer.decompose_returns(strategy_returns, factor_returns, exposures)
    
    print("\\n   Return Contributions:")
    total_contribution = 0.0
    for source, contribution in contributions.items():
        print(f"     {source}: {contribution:.2%}")
        total_contribution += contribution
    print(f"     Total: {total_contribution:.2%}")
    
    # 5. Drawdown analysis
    print("\\n4. Drawdown Analysis")
    
    equity_curve = (1 + strategy_returns).cumprod()
    dd_analyzer = DrawdownAnalyzer()
    
    max_dd = analyzer.calculate_max_drawdown(equity_curve)
    print(f"   Maximum Drawdown: {max_dd:.2%}")
    
    dd_periods = dd_analyzer.find_drawdown_periods(equity_curve, threshold=-0.10)
    
    print(f"\\n   Significant Drawdowns (>10%):")
    for i, period in enumerate(dd_periods[:3], 1):
        print(f"     {i}. {period['start'].date()} to {period['end'].date()}")
        print(f"        Max DD: {period['max_drawdown']:.2%}, Duration: {period['duration']} days")
    
    # 6. Transaction cost impact
    print("\\n5. Transaction Cost Impact")
    
    # Simulate costs (0.05% per trade, assume 20% turnover)
    turnover = 0.20
    avg_cost_per_period = 0.0005 * turnover / 252
    gross_returns = strategy_returns.copy()
    net_returns = strategy_returns - avg_cost_per_period
    
    cost_analyzer = TransactionCostAttribution()
    cost_impact = cost_analyzer.calculate_cost_impact(gross_returns, net_returns)
    
    print(f"   Annualized Cost: {cost_impact['annualized_cost']:.2%}")
    print(f"   Cost as % of Return: {cost_impact['cost_as_pct_of_return']:.1%}")
    print(f"   Sharpe Impact: {cost_impact['sharpe_impact']:.2f}")
\`\`\`

---

## Style Analysis

### Returns-Based Style Analysis (RBSA)

**Objective**: Determine asset class exposures from returns alone

**Method**: Constrained regression
\`\`\`
R_portfolio = Σ w_i × R_i + ε

Constraints:
- Σ w_i = 1 (fully invested)
- w_i ≥ 0 (long-only, optional)
\`\`\`

**Use Cases:**
- Reverse-engineer hedge fund strategies
- Detect style drift
- Compare stated vs. actual strategy

**Example:**
- Fund claims "equity long-short"
- RBSA shows: 70% equity long, 20% bonds, 10% cash
- Actual strategy more balanced than claimed

---

## Attribution in Practice

### Monthly Attribution Report

**Components:**1. **Return Summary**: Strategy vs. benchmark
2. **Alpha/Beta**: Skill vs. market exposure
3. **Factor Attribution**: Which factors contributed
4. **Sector/Stock Attribution**: Which holdings contributed
5. **Transaction Costs**: Impact of trading
6. **Risk Metrics**: Volatility, drawdown, Sharpe

### Investor Presentation

**Key Messages:**
- "We generated 12% return (8% from alpha, 4% from market)"
- "Our momentum factor added 3%, value added 2%"
- "Transaction costs were 50 bps, well below target"
- "Sharpe ratio 1.8, top quartile for strategy type"

---

## Real-World Examples

### Renaissance Technologies

**Reported Performance:**
- Medallion Fund: 66% annual (1988-2018)
- After fees: 39% to investors

**Attribution (speculated):**
- Alpha: ~40-50% annually (pure skill)
- Beta: Near zero (market neutral)
- Factors: Momentum, mean reversion, statistical arb
- Transaction costs: High but managed via speed

### Bridgewater Pure Alpha

**Strategy**: Global macro, multi-asset
**Attribution:**
- Asset allocation: ~40% of return
- Security selection: ~30%
- Currency: ~20%
- Timing: ~10%

**Performance:**
- 12% annual return
- Sharpe ratio: 0.7-1.0
- Low correlation to stocks/bonds

---

## Summary and Key Takeaways

**Performance Attribution Is Essential For:**
- Understanding strategy mechanics
- Diagnosing problems
- Allocating capital
- Demonstrating skill to investors
- Regulatory reporting

**Key Metrics to Track:**1. **Alpha**: Skill-based return
2. **Beta**: Market exposure
3. **Sharpe/Sortino**: Risk-adjusted return
4. **Information Ratio**: Risk-adjusted alpha
5. **Max Drawdown**: Worst loss
6. **Win Rate**: Consistency

**Best Practices:**
- Calculate attribution daily/weekly
- Compare to benchmark and peers
- Track factor exposures over time
- Attribute costs separately
- Investigate outliers (big wins/losses)
- Report transparently to investors

**Common Pitfalls:**
- Cherry-picking metrics
- Ignoring transaction costs
- Overfitting factor models
- Survivorship bias in benchmarks
- Not updating attributions as strategy evolves

**Next Section:** Project: Multi-Strategy Trading System
`,
};
