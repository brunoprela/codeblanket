import { Content } from '@/lib/types';

const benchmarkComparisonDiscussion: Content = {
  title: 'Benchmark Comparison - Discussion Questions',
  description:
    'Deep-dive discussion questions on benchmark selection, attribution analysis, and interpreting relative performance',
  sections: [
    {
      title: 'Discussion Questions',
      content: `
# Discussion Questions: Benchmark Comparison

## Question 1: Multi-Benchmark Portfolio Evaluation

**Scenario**: You're evaluating a multi-strategy hedge fund that operates three distinct strategies:
1. **Long/Short Equity** (50% of capital): Beta ~0.3, targets market-neutral but allows some net exposure
2. **Trend Following** (30% of capital): Trades futures across commodities, FX, indices
3. **Statistical Arbitrage** (20% of capital): High-frequency pairs trading, beta ~0.0

The fund reports:
- Overall return: 18% annually
- Overall Sharpe: 1.4
- Correlation to S&P 500: 0.45

**Challenge**: What benchmark(s) should you use? A single benchmark won't capture the multi-strategy nature. How do you fairly evaluate performance and decompose returns?

### Comprehensive Answer

Multi-strategy funds require composite benchmark approaches and sophisticated attribution analysis.

**Approach 1: Weighted Composite Benchmark**

\`\`\`python
import pandas as pd
import numpy as np
from typing import Dict, List

class MultiStrategyBenchmark:
    """
    Composite benchmark for multi-strategy portfolios
    """
    
    def __init__(
        self,
        strategy_allocations: Dict[str, float],
        strategy_benchmarks: Dict[str, pd.Series]
    ):
        """
        Initialize composite benchmark
        
        Args:
            strategy_allocations: Strategy name -> allocation %
            strategy_benchmarks: Strategy name -> benchmark returns
        """
        self.allocations = strategy_allocations
        self.benchmarks = strategy_benchmarks
        
        # Validate
        assert np.isclose(sum(strategy_allocations.values()), 1.0), \
            "Allocations must sum to 1.0"
    
    def construct_composite(self) -> pd.Series:
        """
        Construct composite benchmark returns
        
        Returns:
            Composite benchmark return series
        """
        # Align all benchmarks to common dates
        common_dates = None
        for benchmark in self.benchmarks.values():
            if common_dates is None:
                common_dates = set(benchmark.index)
            else:
                common_dates = common_dates.intersection(set(benchmark.index))
        
        common_dates = sorted(list(common_dates))
        
        # Calculate weighted returns
        composite_returns = pd.Series(0.0, index=common_dates)
        
        for strategy_name, allocation in self.allocations.items():
            benchmark = self.benchmarks[strategy_name]
            aligned_benchmark = benchmark.loc[common_dates]
            composite_returns += allocation * aligned_benchmark
        
        return composite_returns
    
    def decompose_performance(
        self,
        fund_returns: pd.Series
    ) -> Dict:
        """
        Decompose fund performance vs composite benchmark
        
        Returns:
            Attribution of returns to each strategy
        """
        composite = self.construct_composite()
        
        # Align fund returns
        common_dates = fund_returns.index.intersection(composite.index)
        fund_aligned = fund_returns.loc[common_dates]
        composite_aligned = composite.loc[common_dates]
        
        # Calculate overall excess return
        total_excess = (fund_aligned - composite_aligned).sum()
        
        # Attribute excess to each strategy sub-portfolio
        # This requires fund's individual strategy returns
        # Simplified version: show contribution of each benchmark
        
        attribution = {
            'total_fund_return': fund_aligned.sum(),
            'composite_benchmark_return': composite_aligned.sum(),
            'total_excess': total_excess,
            'strategy_contributions': {}
        }
        
        for strategy_name, allocation in self.allocations.items():
            benchmark = self.benchmarks[strategy_name].loc[common_dates]
            contribution = (allocation * benchmark).sum()
            attribution['strategy_contributions'][strategy_name] = contribution
        
        return attribution


# Example: Multi-strategy fund
def evaluate_multistrategy_fund():
    """Evaluate multi-strategy hedge fund"""
    
    # Generate sample data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    # Strategy allocations
    allocations = {
        'long_short_equity': 0.50,
        'trend_following': 0.30,
        'stat_arb': 0.20
    }
    
    # Benchmark returns for each strategy
    benchmarks = {
        'long_short_equity': pd.Series(
            np.random.randn(len(dates)) * 0.008 + 0.0003,  # ~8% vol, ~8% return
            index=dates
        ),
        'trend_following': pd.Series(
            np.random.randn(len(dates)) * 0.015 + 0.0002,  # ~15% vol, ~5% return
            index=dates
        ),
        'stat_arb': pd.Series(
            np.random.randn(len(dates)) * 0.005 + 0.0004,  # ~5% vol, ~10% return
            index=dates
        )
    }
    
    # Fund returns (slightly better than benchmark)
    fund_returns = pd.Series(
        np.random.randn(len(dates)) * 0.01 + 0.0007,  # ~10% vol, ~18% return
        index=dates
    )
    
    # Create composite benchmark
    multi_bench = MultiStrategyBenchmark(
        strategy_allocations=allocations,
        strategy_benchmarks=benchmarks
    )
    
    # Get composite
    composite = multi_bench.construct_composite()
    
    # Decompose performance
    attribution = multi_bench.decompose_performance(fund_returns)
    
    print("\\nMULTI-STRATEGY FUND EVALUATION")
    print("="*80)
    print(f"\\nFund Return: {attribution['total_fund_return']:.4f}")
    print(f"Composite Benchmark Return: {attribution['composite_benchmark_return']:.4f}")
    print(f"Excess Return: {attribution['total_excess']:.4f}")
    print("\\nBenchmark Contributions by Strategy:")
    for strategy, contribution in attribution['strategy_contributions'].items():
        print(f"  {strategy}: {contribution:.4f}")
    
    # Calculate metrics vs composite
    fund_cumulative = (1 + fund_returns).cumprod()
    composite_cumulative = (1 + composite).cumprod()
    
    fund_sharpe = (fund_returns.mean() / fund_returns.std()) * np.sqrt(252)
    composite_sharpe = (composite.mean() / composite.std()) * np.sqrt(252)
    
    print(f"\\nFund Sharpe: {fund_sharpe:.3f}")
    print(f"Composite Benchmark Sharpe: {composite_sharpe:.3f}")
    
    # Information ratio vs composite
    excess_returns = fund_returns - composite
    ir = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
    print(f"Information Ratio: {ir:.3f}")


if __name__ == "__main__":
    evaluate_multistrategy_fund()
\`\`\`

**Key Insights:**1. **Composite Benchmark**: Weight strategy-specific benchmarks by allocation
   - Long/Short Equity → HFRI Equity Hedge Index (50%)
   - Trend Following → SG CTA Index (30%)
   - Stat Arb → HFRI Market Neutral Index (20%)

2. **Two-Level Attribution**:
   - **Allocation Effect**: Did manager allocate capital to right strategies?
   - **Selection Effect**: Did each sub-strategy outperform its benchmark?

3. **Problems with Single Benchmark**:
   - S&P 500: Misses 80% of strategies (trend + stat arb)
   - Market Neutral Index: Ignores equity beta in long/short book
   - Any single benchmark: Cannot capture multi-strategy dynamics

**Recommendation**: Use composite benchmark for overall evaluation, then drill down into each strategy component for detailed attribution.

---

## Question 2: Benchmark Gaming and Manipulation

**Scenario**: You're the CIO reviewing performance reports. You notice concerning patterns:

**Fund A (Equity Long/Short)**: 
- Reports benchmark: Russell 2000 (small-cap)
- Actual holdings: 80% mega-cap tech (FAANG)
- During period: Russell 2000 -5%, Mega-cap tech +35%
- Fund claims "40% outperformance vs benchmark"

**Fund B (Market Neutral)**:
- Reports benchmark: 3-month T-bills (2% return)
- Takes significant factor exposures (value, momentum, size)
- Returned 8% vs 2% benchmark
- Markets "300% outperformance"

**Fund C (Global Macro)**:
- No disclosed benchmark
- Claims "absolute return strategy"
- Refuses benchmark comparison: "We're uncorrelated to traditional assets"

**Questions:**1. What benchmark manipulation tactics are being used?
2. How would you establish fair benchmarks for each fund?
3. What policies prevent benchmark gaming?

### Comprehensive Answer

Benchmark gaming is common in asset management—managers cherry-pick benchmarks to make performance look better.

**Analysis of Each Fund:**

**Fund A: Benchmark Mismatch**

*Tactic*: Selecting Russell 2000 benchmark while holding mega-caps
- During mega-cap outperformance: Claim "beat benchmark"
- During small-cap outperformance: Switch holdings to small-cap, claim "we followed benchmark"

*Fair Benchmark*: Style-based benchmark matching actual holdings:
\`\`\`python
def construct_holdings_based_benchmark(
    holdings: pd.DataFrame,  # Holdings by market cap, style
    period_returns: Dict[str, float]  # Returns for each segment
) -> float:
    """Construct benchmark matching actual portfolio style"""
    
    # Analyze holdings
    mega_cap_pct = holdings['mega_cap_weight'].mean()
    large_cap_pct = holdings['large_cap_weight'].mean()
    small_cap_pct = holdings['small_cap_weight'].mean()
    
    # Construct benchmark
    benchmark_return = (
        mega_cap_pct * period_returns['mega_cap'] +
        large_cap_pct * period_returns['large_cap'] +
        small_cap_pct * period_returns['small_cap']
    )
    
    return benchmark_return

# For Fund A:
# Actual holdings: 80% mega-cap
# Fair benchmark: 80% S&P 100 + 20% Russell 2000
# This would show Fund A *underperformed* its true style benchmark
\`\`\`

**Fund B: Benchmark Too Low**

*Tactic*: Using risk-free rate for a strategy taking substantial factor risk
- Any positive return beats T-bills
- Ignores that factor exposures are essentially beta (systematic risk)

*Fair Benchmark*: Factor-based benchmark:
\`\`\`python
def construct_factor_benchmark(
    factor_exposures: Dict[str, float],  # Fund's factor loadings
    factor_returns: Dict[str, float]  # Factor returns during period
) -> float:
    """Construct benchmark based on factor exposures"""
    
    # Fund's factor exposures (from regression)
    # value_beta = 0.4, momentum_beta = 0.3, size_beta = 0.2
    
    benchmark_return = (
        factor_exposures['value'] * factor_returns['HML'] +
        factor_exposures['momentum'] * factor_returns['UMD'] +
        factor_exposures['size'] * factor_returns['SMB'] +
        factor_returns['risk_free']
    )
    
    return benchmark_return

# For Fund B:
# If factor exposure would predict 7% return
# Fund returned 8%
# True alpha: Only 1% (not 6%)
\`\`\`

**Fund C: No Benchmark**

*Tactic*: Claiming "absolute return" to avoid comparison
- If returns are good: Highlight them
- If returns are bad: "We're uncorrelated, can't be compared"

*Fair Treatment*:
1. Even "absolute return" strategies need benchmarks
2. Minimum: Compare to risk-free rate + volatility scaling
3. Better: Construct strategy-specific benchmark

\`\`\`python
def evaluate_absolute_return_strategy(
    strategy_returns: pd.Series,
    risk_free_rate: float = 0.02
) -> Dict:
    """
    Evaluate absolute return strategy
    
    Even without natural benchmark, we can evaluate:
    - Return vs risk-free rate (minimum hurdle)
    - Sharpe ratio (return per unit risk)
    - Maximum drawdown (downside protection)
    """
    
    sharpe = (
        (strategy_returns.mean() * 252 - risk_free_rate) /
        (strategy_returns.std() * np.sqrt(252))
    )
    
    # Construct "synthetic benchmark":
    # Risk-free rate + (strategy vol × market Sharpe)
    market_sharpe = 0.4  # Typical equity market Sharpe
    target_return = risk_free_rate + (
        strategy_returns.std() * np.sqrt(252) * market_sharpe
    )
    
    evaluation = {
        'strategy_return': strategy_returns.mean() * 252,
        'risk_free_rate': risk_free_rate,
        'excess_vs_rf': strategy_returns.mean() * 252 - risk_free_rate,
        'sharpe': sharpe,
        'synthetic_benchmark': target_return,
        'excess_vs_synthetic': strategy_returns.mean() * 252 - target_return
    }
    
    return evaluation
\`\`\`

**Policies to Prevent Benchmark Gaming:**1. **Pre-specified Benchmarks**: Benchmark chosen at fund inception, not in hindsight
2. **Style-Consistent Benchmarks**: Benchmark must match actual portfolio characteristics
3. **Multiple Benchmarks**: Report against several benchmarks to show robustness
4. **Peer Group Comparison**: Compare to similar strategies (e.g., Morningstar categories)
5. **Full Disclosure**: Disclose factor exposures, not just headline benchmark
6. **Independent Review**: Third-party performance attribution analysis

**Industry Example**: After the 2008 crisis, SEC increased scrutiny of benchmark claims. Many "absolute return" funds were revealed to have significant equity beta—they were just long-only equity funds with misleading names.

---

## Question 3: Benchmark Selection for Novel Strategies

**Scenario**: You've developed a novel AI-driven trading strategy that:
- Trades options (not underlying stocks)
- Uses NLP sentiment analysis of news/social media
- Targets volatility arbitrage opportunities
- Has zero correlation to equity markets (0.02 correlation to S&P 500)
- Average holding period: 2-3 days

**Challenge**: There's no obvious benchmark. The VIX index trades volatility but in a different manner. Market-neutral indices don't capture options strategies. Cryptocurrency quant strategies are thematically similar but different asset class.

**How do you establish a fair benchmark for performance evaluation and investor communication?**

### Comprehensive Answer

Novel strategies require creative but rigorous benchmark construction.

**Option 1: Synthetic Benchmark**

Construct a "synthetic" benchmark that represents passive exposure to similar characteristics:

\`\`\`python
def construct_synthetic_options_benchmark(
    vix_returns: pd.Series,
    short_vol_etf: pd.Series,  # SVXY or similar
    market_neutral_fund: pd.Series
) -> pd.Series:
    """
    Construct synthetic benchmark for options volatility strategy
    
    Combines:
    - VIX exposure (volatility sensitivity)
    - Short volatility ETF (similar mechanics)
    - Market neutral returns (directional neutrality)
    """
    
    # Optimize weights to match strategy characteristics
    # Target: Zero equity beta, similar volatility profile
    
    weights = {
        'vix': 0.3,
        'short_vol': 0.5,
        'market_neutral': 0.2
    }
    
    synthetic = (
        weights['vix'] * vix_returns +
        weights['short_vol'] * short_vol_etf +
        weights['market_neutral'] * market_neutral_fund
    )
    
    return synthetic
\`\`\`

**Option 2: Peer Group Benchmark**

Create benchmark from similar strategies:

\`\`\`python
class PeerGroupBenchmark:
    """Benchmark based on peer strategies"""
    
    def __init__(
        self,
        peer_strategies: Dict[str, pd.Series],
        selection_criteria: Dict
    ):
        """
        Args:
            peer_strategies: Dictionary of strategy_name -> returns
            selection_criteria: Criteria for peer selection
                - max_equity_correlation: 0.15
                - min_sharpe: 0.8
                - similar_vol_range: (0.08, 0.15)
        """
        self.peers = peer_strategies
        self.criteria = selection_criteria
        
        # Filter peers
        self.selected_peers = self._select_peers()
    
    def _select_peers(self) -> Dict[str, pd.Series]:
        """Select peers matching criteria"""
        selected = {}
        
        for name, returns in self.peers.items():
            # Check correlation to equities
            # Check Sharpe ratio
            # Check volatility
            # (Simplified - would use actual filtering)
            
            selected[name] = returns
        
        return selected
    
    def construct_peer_benchmark(self) -> pd.Series:
        """Construct equal-weight benchmark from peers"""
        if not self.selected_peers:
            raise ValueError("No peers match criteria")
        
        # Equal-weight peer portfolio
        n_peers = len(self.selected_peers)
        benchmark = sum(self.selected_peers.values()) / n_peers
        
        return benchmark
\`\`\`

**Option 3: Factor-Based Benchmark**

Decompose returns into factor exposures:

\`\`\`python
def factor_based_benchmark_for_novel_strategy(
    strategy_returns: pd.Series,
    factor_returns: pd.DataFrame  # VIX, term structure, momentum, etc.
) -> Dict:
    """
    Use factor regression to construct benchmark
    
    For options volatility strategy, relevant factors:
    - VIX level
    - VIX term structure slope
    - Equity put/call ratio
    - Realized vs implied vol spread
    """
    
    from sklearn.linear_model import LinearRegression
    
    # Align data
    common_idx = strategy_returns.index.intersection(factor_returns.index)
    y = strategy_returns.loc[common_idx].values.reshape(-1, 1)
    X = factor_returns.loc[common_idx].values
    
    # Fit regression
    model = LinearRegression()
    model.fit(X, y)
    
    # Predicted returns = benchmark
    benchmark_returns = model.predict(X).flatten()
    
    # Alpha = actual - benchmark
    alpha = y.flatten() - benchmark_returns
    
    return {
        'factor_loadings': dict(zip(factor_returns.columns, model.coef_[0])),
        'benchmark_returns': pd.Series(benchmark_returns, index=common_idx),
        'alpha': pd.Series(alpha, index=common_idx),
        'r_squared': model.score(X, y)
    }
\`\`\`

**Recommended Approach:**1. **Primary Benchmark**: Synthetic benchmark from related instruments/strategies
2. **Secondary Benchmark**: Risk-free rate + volatility-adjusted hurdle
3. **Tertiary Comparison**: Peer group (other options/volatility strategies)
4. **Full Disclosure**: 
   - Factor exposures
   - Correlation to major indices
   - Performance across different market regimes

**Communication to Investors:**

"Given the novel nature of our strategy, we use a multi-benchmark approach:
- **Synthetic Options Benchmark** (primary): 30% VIX, 50% short-vol ETF, 20% market-neutral
- **Hurdle Rate** (minimum): Risk-free + 300bps (commensurate with strategy volatility)
- **Peer Group**: Top quartile of volatility arbitrage strategies

We report against all three to provide comprehensive performance context."

This transparent, multi-faceted approach builds credibility even without a perfect benchmark.
`,
    },
  ],
};

export default benchmarkComparisonDiscussion;
