export const factorInvestingStrategies = {
    title: 'Factor Investing Strategies',
    slug: 'factor-investing-strategies',
    description: 'Systematic factor-based strategies: value, momentum, quality, size, and low volatility',
    content: `
# Factor Investing Strategies

## Introduction: The Science of Systematic Investing

Factor investing is the systematic approach to capturing return premiums by targeting specific characteristics (factors) that explain differences in asset returns. Instead of picking individual stocks, factor strategies invest in portfolios tilted toward proven sources of outperformance: value stocks, momentum winners, high-quality companies, small caps, and low-volatility securities.

**What you'll learn:**
- Core factors: Value, Momentum, Quality, Size, Low Volatility
- Factor construction and portfolio implementation
- Multi-factor models (Fama-French, Carhart)
- Factor timing and allocation
- Smart beta and factor ETFs
- Factor risk and exposure management

**Why this matters for engineers:**
- Systematic rules-based approach (perfect for algorithms)
- Large-scale data analysis (thousands of stocks)
- Quantifiable signals (P/E, ROE, volatility)
- Backtestable strategies (decades of data)
- Production-ready frameworks

**Performance Characteristics:**
- **Historical Alpha**: 2-6% annually (varies by factor)
- **Sharpe Ratio**: 0.3-0.8 per factor, 0.8-1.5 combined
- **Win Rate**: 55-65% of years positive
- **Holding Period**: Months to years
- **Correlation**: Factors work in different regimes

---

## The Five Core Factors

### 1. Value Factor

**Definition**: Cheap stocks outperform expensive stocks

**Metrics:**
- **P/E Ratio**: Price-to-Earnings (low is cheap)
- **P/B Ratio**: Price-to-Book (low is value)
- **P/S Ratio**: Price-to-Sales
- **Dividend Yield**: High yield = value
- **EV/EBITDA**: Enterprise Value to EBITDA

**Why It Works:**
- Mean reversion (cheap gets expensive)
- Behavioral bias (overreaction to bad news)
- Risk premium (value stocks riskier)

**Performance:**
- Long-term alpha: ~3-4% annually
- Works better in value cycles
- Struggled 2010-2020 (growth outperformed)

**Implementation:**
\`\`\`python
def calculate_value_score(stock):
    """
    Calculate composite value score
    
    Lower score = more valuable
    """
    pe_ratio = stock.price / stock.earnings_per_share
    pb_ratio = stock.price / stock.book_value_per_share
    ps_ratio = stock.market_cap / stock.revenue
    
    # Percentile rank (0-100)
    pe_percentile = percentile_rank(pe_ratio, universe)
    pb_percentile = percentile_rank(pb_ratio, universe)
    ps_percentile = percentile_rank(ps_ratio, universe)
    
    # Average (equal weight)
    value_score = (pe_percentile + pb_percentile + ps_percentile) / 3
    
    return value_score  # 0-100, low = cheap
\`\`\`

### 2. Momentum Factor

**Definition**: Winners keep winning, losers keep losing

**Metrics:**
- **12-month return** (most common)
- **6-month return**
- **52-week high proximity**
- **Relative strength**

**Why It Works:**
- Behavioral: Under-reaction to news
- Anchoring and herding
- Trend persistence
- Information diffusion takes time

**Performance:**
- Long-term alpha: ~5-8% annually
- Works in trending markets
- Crash risk (sharp reversals)

**Implementation:**
\`\`\`python
def calculate_momentum_score(stock, lookback_days=252):
    """
    Calculate momentum score (12-month return)
    
    Higher = stronger momentum
    """
    # Get returns for last 12 months, skipping last month
    # (skip last month to avoid short-term reversal)
    start_date = today - timedelta(days=lookback_days)
    end_date = today - timedelta(days=21)
    
    price_start = stock.get_price(start_date)
    price_end = stock.get_price(end_date)
    
    momentum_return = (price_end - price_start) / price_start
    
    # Percentile rank
    momentum_score = percentile_rank(momentum_return, universe)
    
    return momentum_score  # 0-100, high = strong momentum
\`\`\`

### 3. Quality Factor

**Definition**: High-quality companies outperform

**Metrics:**
- **ROE**: Return on Equity (profitability)
- **ROA**: Return on Assets
- **Debt/Equity**: Lower is better
- **Earnings Quality**: Accruals, stability
- **Gross Margin**: Higher is better

**Why It Works:**
- Quality companies compound
- Downside protection in recessions
- Sustainable competitive advantages
- Better risk-adjusted returns

**Performance:**
- Long-term alpha: ~3-4% annually
- Defensive (works in downturns)
- Consistent performance

**Implementation:**
\`\`\`python
def calculate_quality_score(stock):
    """
    Calculate composite quality score
    
    Higher = higher quality
    """
    roe = stock.net_income / stock.shareholders_equity
    roa = stock.net_income / stock.total_assets
    debt_to_equity = stock.total_debt / stock.shareholders_equity
    gross_margin = stock.gross_profit / stock.revenue
    
    # Percentile ranks (high is good except debt)
    roe_pct = percentile_rank(roe, universe)
    roa_pct = percentile_rank(roa, universe)
    debt_pct = 100 - percentile_rank(debt_to_equity, universe)  # Invert
    margin_pct = percentile_rank(gross_margin, universe)
    
    # Average
    quality_score = (roe_pct + roa_pct + debt_pct + margin_pct) / 4
    
    return quality_score  # 0-100, high = quality
\`\`\`

### 4. Size Factor

**Definition**: Small-cap stocks outperform large-cap

**Metrics:**
- **Market Capitalization**
- **Total Assets**
- **Revenue**

**Why It Works:**
- Higher growth potential
- Less analyst coverage (inefficiency)
- Liquidity premium
- Risk premium

**Performance:**
- Long-term alpha: ~2-3% annually
- Volatile (small caps riskier)
- Cyclical (works in bull markets)

**Implementation:**
\`\`\`python
def calculate_size_score(stock):
    """
    Calculate size score
    
    Lower = smaller (more attractive for size factor)
    """
    market_cap = stock.shares_outstanding * stock.price
    
    # Percentile rank (invert: small = high score)
    size_percentile = percentile_rank(market_cap, universe)
    size_score = 100 - size_percentile
    
    return size_score  # 0-100, high = small
\`\`\`

### 5. Low Volatility Factor

**Definition**: Low-volatility stocks outperform high-volatility

**Metrics:**
- **Historical Volatility** (standard deviation)
- **Beta** (market sensitivity)
- **Downside Deviation**

**Why It Works:**
- Behavioral: Investors overpay for lottery tickets
- Leverage constraints
- Risk-adjusted returns better
- Defensive in downturns

**Performance:**
- Long-term alpha: ~2-3% annually
- Lower drawdowns
- Works in bear markets

**Implementation:**
\`\`\`python
def calculate_low_vol_score(stock, lookback_days=252):
    """
    Calculate low volatility score
    
    Higher = lower volatility (more attractive)
    """
    # Get daily returns for last year
    returns = stock.get_returns(lookback_days)
    
    # Calculate volatility (annualized)
    volatility = returns.std() * np.sqrt(252)
    
    # Percentile rank (invert: low vol = high score)
    vol_percentile = percentile_rank(volatility, universe)
    low_vol_score = 100 - vol_percentile
    
    return low_vol_score  # 0-100, high = low volatility
\`\`\`

---

## Multi-Factor Models

### Fama-French 3-Factor Model

**Formula:**
\`\`\`
R_i - R_f = α + β_market(R_m - R_f) + β_size(SMB) + β_value(HML) + ε
\`\`\`

**Factors:**
- **Market**: Excess return of market
- **SMB**: Small Minus Big (size factor)
- **HML**: High Minus Low (value factor)

**Why It Matters:**
- Explains 90%+ of portfolio returns
- Better than CAPM alone
- Industry standard

### Carhart 4-Factor Model

**Adds Momentum:**
\`\`\`
R_i - R_f = α + β_market(R_m - R_f) + β_size(SMB) + β_value(HML) + β_momentum(UMD) + ε
\`\`\`

**UMD**: Up Minus Down (momentum factor)

### 5-Factor Model (Fama-French)

**Adds Quality:**
- **RMW**: Robust Minus Weak (profitability)
- **CMA**: Conservative Minus Aggressive (investment)

\`\`\`python
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
import logging

@dataclass
class Stock:
    """
    Stock with fundamental data
    """
    symbol: str
    price: float
    market_cap: float
    
    # Value metrics
    earnings_per_share: float
    book_value_per_share: float
    revenue: float
    
    # Quality metrics
    net_income: float
    shareholders_equity: float
    total_assets: float
    total_debt: float
    gross_profit: float
    
    # Other
    shares_outstanding: float
    
    @property
    def pe_ratio(self) -> float:
        """Price-to-Earnings ratio"""
        if self.earnings_per_share <= 0:
            return np.inf
        return self.price / self.earnings_per_share
    
    @property
    def pb_ratio(self) -> float:
        """Price-to-Book ratio"""
        if self.book_value_per_share <= 0:
            return np.inf
        return self.price / self.book_value_per_share
    
    @property
    def roe(self) -> float:
        """Return on Equity"""
        if self.shareholders_equity <= 0:
            return 0
        return self.net_income / self.shareholders_equity
    
    @property
    def debt_to_equity(self) -> float:
        """Debt-to-Equity ratio"""
        if self.shareholders_equity <= 0:
            return np.inf
        return self.total_debt / self.shareholders_equity

class FactorCalculator:
    """
    Calculate factor scores for stocks
    """
    
    def __init__(self, universe: List[Stock]):
        """
        Initialize with stock universe
        
        Args:
            universe: List of stocks to analyze
        """
        self.universe = universe
        self.logger = logging.getLogger(__name__)
    
    def percentile_rank(self, value: float, values: List[float]) -> float:
        """
        Calculate percentile rank (0-100)
        
        Args:
            value: Value to rank
            values: All values in universe
            
        Returns:
            Percentile (0-100)
        """
        values_clean = [v for v in values if not np.isinf(v) and not np.isnan(v)]
        
        if not values_clean:
            return 50.0
        
        percentile = stats.percentileofscore(values_clean, value)
        return percentile
    
    def calculate_value_score(self, stock: Stock) -> float:
        """
        Calculate composite value score
        
        Components:
        - P/E ratio (low is good)
        - P/B ratio (low is good)
        - P/S ratio (low is good)
        
        Args:
            stock: Stock to analyze
            
        Returns:
            Value score (0-100, low = value)
        """
        # Get all ratios
        pe_ratios = [s.pe_ratio for s in self.universe]
        pb_ratios = [s.pb_ratio for s in self.universe]
        
        # Percentile ranks (low ratio = low percentile = value)
        pe_pct = self.percentile_rank(stock.pe_ratio, pe_ratios)
        pb_pct = self.percentile_rank(stock.pb_ratio, pb_ratios)
        
        # Average (equal weight)
        value_score = (pe_pct + pb_pct) / 2
        
        # Invert so low score = value
        value_score = 100 - value_score
        
        return value_score
    
    def calculate_momentum_score(self,
                                 stock: Stock,
                                 returns: pd.Series,
                                 lookback_days: int = 252) -> float:
        """
        Calculate momentum score (12-month return)
        
        Args:
            stock: Stock to analyze
            returns: Historical returns
            lookback_days: Lookback period (default 252 = 1 year)
            
        Returns:
            Momentum score (0-100, high = strong momentum)
        """
        # Calculate cumulative return over lookback (skip last month)
        momentum_return = returns.iloc[-lookback_days:-21].add(1).prod() - 1
        
        # Get all momentum returns
        all_momentum = []  # Would calculate for all stocks
        
        # Percentile rank
        # momentum_score = self.percentile_rank(momentum_return, all_momentum)
        
        # Placeholder
        momentum_score = 50.0
        
        return momentum_score
    
    def calculate_quality_score(self, stock: Stock) -> float:
        """
        Calculate composite quality score
        
        Components:
        - ROE (high is good)
        - Debt/Equity (low is good)
        - Gross Margin (high is good)
        
        Args:
            stock: Stock to analyze
            
        Returns:
            Quality score (0-100, high = quality)
        """
        # Get all metrics
        roes = [s.roe for s in self.universe]
        debt_ratios = [s.debt_to_equity for s in self.universe]
        
        margins = []
        for s in self.universe:
            if s.revenue > 0:
                margins.append(s.gross_profit / s.revenue)
            else:
                margins.append(0)
        
        # Percentile ranks
        roe_pct = self.percentile_rank(stock.roe, roes)
        debt_pct = 100 - self.percentile_rank(stock.debt_to_equity, debt_ratios)  # Invert
        
        stock_margin = stock.gross_profit / stock.revenue if stock.revenue > 0 else 0
        margin_pct = self.percentile_rank(stock_margin, margins)
        
        # Average
        quality_score = (roe_pct + debt_pct + margin_pct) / 3
        
        return quality_score
    
    def calculate_size_score(self, stock: Stock) -> float:
        """
        Calculate size score
        
        Args:
            stock: Stock to analyze
            
        Returns:
            Size score (0-100, high = small)
        """
        market_caps = [s.market_cap for s in self.universe]
        
        # Percentile rank
        size_pct = self.percentile_rank(stock.market_cap, market_caps)
        
        # Invert so small = high score
        size_score = 100 - size_pct
        
        return size_score
    
    def calculate_all_factors(self, stock: Stock) -> Dict[str, float]:
        """
        Calculate all factor scores
        
        Args:
            stock: Stock to analyze
            
        Returns:
            Dict of factor scores
        """
        return {
            'value': self.calculate_value_score(stock),
            'quality': self.calculate_quality_score(stock),
            'size': self.calculate_size_score(stock),
            # 'momentum': Would need returns data
            # 'low_vol': Would need returns data
        }

class MultiFactorStrategy:
    """
    Multi-factor portfolio construction
    
    Combines multiple factors into single strategy
    """
    
    def __init__(self,
                 factors: List[str] = ['value', 'momentum', 'quality'],
                 weights: Optional[List[float]] = None,
                 n_stocks: int = 50):
        """
        Initialize multi-factor strategy
        
        Args:
            factors: Factors to include
            weights: Factor weights (default equal)
            n_stocks: Number of stocks in portfolio
        """
        self.factors = factors
        self.weights = weights or [1.0 / len(factors)] * len(factors)
        self.n_stocks = n_stocks
        
        assert len(self.factors) == len(self.weights)
        assert abs(sum(self.weights) - 1.0) < 0.01
    
    def calculate_composite_score(self,
                                  factor_scores: Dict[str, float]) -> float:
        """
        Calculate composite score from individual factors
        
        Args:
            factor_scores: Dict mapping factor to score
            
        Returns:
            Composite score (0-100)
        """
        composite = 0.0
        
        for factor, weight in zip(self.factors, self.weights):
            if factor in factor_scores:
                composite += factor_scores[factor] * weight
        
        return composite
    
    def construct_portfolio(self,
                           universe: List[Stock],
                           factor_calculator: FactorCalculator) -> List[Tuple[Stock, float]]:
        """
        Construct portfolio using multi-factor ranking
        
        Args:
            universe: Stock universe
            factor_calculator: Factor calculator
            
        Returns:
            List of (stock, weight) tuples
        """
        # Calculate composite scores for all stocks
        stock_scores = []
        
        for stock in universe:
            try:
                # Get factor scores
                factor_scores = factor_calculator.calculate_all_factors(stock)
                
                # Calculate composite
                composite = self.calculate_composite_score(factor_scores)
                
                stock_scores.append((stock, composite))
                
            except Exception as e:
                self.logger.warning(f"Error calculating scores for {stock.symbol}: {e}")
                continue
        
        # Sort by composite score (descending)
        stock_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take top N stocks
        top_stocks = stock_scores[:self.n_stocks]
        
        # Equal weight
        weight = 1.0 / len(top_stocks)
        
        portfolio = [(stock, weight) for stock, score in top_stocks]
        
        return portfolio
    
    def calculate_factor_exposures(self,
                                   portfolio: List[Tuple[Stock, float]],
                                   factor_calculator: FactorCalculator) -> Dict[str, float]:
        """
        Calculate portfolio's factor exposures
        
        Args:
            portfolio: List of (stock, weight)
            factor_calculator: Factor calculator
            
        Returns:
            Dict of factor exposures
        """
        exposures = {factor: 0.0 for factor in self.factors}
        
        for stock, weight in portfolio:
            factor_scores = factor_calculator.calculate_all_factors(stock)
            
            for factor in self.factors:
                if factor in factor_scores:
                    # Weight-adjusted exposure
                    # Normalize to -1 to +1 (50 = neutral)
                    normalized = (factor_scores[factor] - 50) / 50
                    exposures[factor] += normalized * weight
        
        return exposures

class FactorTimingStrategy:
    """
    Dynamic factor allocation based on macro regime
    
    Different factors work in different environments:
    - Value: Works in value cycles, rising rates
    - Momentum: Works in trending markets
    - Quality: Works in recessions, bear markets
    - Size: Works in bull markets, low rates
    - Low Vol: Works in bear markets, high uncertainty
    """
    
    def __init__(self):
        self.regime_weights = {
            'bull_market': {
                'value': 0.15,
                'momentum': 0.35,
                'quality': 0.20,
                'size': 0.30,
                'low_vol': 0.00
            },
            'bear_market': {
                'value': 0.20,
                'momentum': 0.10,
                'quality': 0.40,
                'size': 0.00,
                'low_vol': 0.30
            },
            'high_inflation': {
                'value': 0.30,
                'momentum': 0.20,
                'quality': 0.30,
                'size': 0.10,
                'low_vol': 0.10
            },
            'rising_rates': {
                'value': 0.35,
                'momentum': 0.15,
                'quality': 0.30,
                'size': 0.10,
                'low_vol': 0.10
            }
        }
    
    def detect_regime(self,
                     market_return_1y: float,
                     vix: float,
                     inflation_rate: float,
                     fed_rate_change: float) -> str:
        """
        Detect current market regime
        
        Args:
            market_return_1y: Market return last 12 months
            vix: VIX level
            inflation_rate: CPI year-over-year
            fed_rate_change: Fed rate change last 6 months
            
        Returns:
            Regime name
        """
        # Bull/Bear
        if market_return_1y > 0.10 and vix < 20:
            return 'bull_market'
        elif market_return_1y < -0.05 or vix > 30:
            return 'bear_market'
        
        # Inflation
        if inflation_rate > 0.04:
            return 'high_inflation'
        
        # Rising rates
        if fed_rate_change > 0.5:
            return 'rising_rates'
        
        # Default: balanced
        return 'bull_market'
    
    def get_factor_weights(self, regime: str) -> Dict[str, float]:
        """
        Get factor weights for regime
        
        Args:
            regime: Market regime
            
        Returns:
            Factor weights
        """
        return self.regime_weights.get(regime, {
            'value': 0.20,
            'momentum': 0.20,
            'quality': 0.20,
            'size': 0.20,
            'low_vol': 0.20
        })

# Example usage
if __name__ == "__main__":
    print("\\n=== Factor Investing System ===\\n")
    
    # 1. Create sample universe
    print("1. Sample Universe")
    
    stocks = [
        Stock(
            symbol="AAPL",
            price=150.0,
            market_cap=2_500_000_000_000,
            earnings_per_share=6.0,
            book_value_per_share=4.0,
            revenue=400_000_000_000,
            net_income=100_000_000_000,
            shareholders_equity=60_000_000_000,
            total_assets=350_000_000_000,
            total_debt=120_000_000_000,
            gross_profit=170_000_000_000,
            shares_outstanding=16_000_000_000
        ),
        Stock(
            symbol="F",  # Value stock
            price=12.0,
            market_cap=48_000_000_000,
            earnings_per_share=1.5,
            book_value_per_share=10.0,
            revenue=150_000_000_000,
            net_income=6_000_000_000,
            shareholders_equity=40_000_000_000,
            total_assets=260_000_000_000,
            total_debt=150_000_000_000,
            gross_profit=20_000_000_000,
            shares_outstanding=4_000_000_000
        )
    ]
    
    print(f"   Universe Size: {len(stocks)} stocks")
    print(f"   Symbols: {[s.symbol for s in stocks]}")
    
    # 2. Calculate factor scores
    print("\\n2. Factor Scores")
    
    calculator = FactorCalculator(universe=stocks)
    
    for stock in stocks:
        print(f"\\n   {stock.symbol}:")
        print(f"     P/E: {stock.pe_ratio:.1f}")
        print(f"     P/B: {stock.pb_ratio:.1f}")
        print(f"     ROE: {stock.roe:.1%}")
        print(f"     Market Cap: ${stock.market_cap / 1e9: .1f
}B")

scores = calculator.calculate_all_factors(stock)
print(f"     Value Score: {scores['value']:.1f}/100")
print(f"     Quality Score: {scores['quality']:.1f}/100")
print(f"     Size Score: {scores['size']:.1f}/100")
    
    # 3. Multi - factor strategy
print("\\n3. Multi-Factor Portfolio Construction")

strategy = MultiFactorStrategy(
    factors = ['value', 'quality'],
    weights = [0.6, 0.4],
    n_stocks = 2
)

portfolio = strategy.construct_portfolio(stocks, calculator)

print(f"   Strategy: 60% Value + 40% Quality")
print(f"   Portfolio Size: {len(portfolio)} stocks")
print("\\n   Holdings:")
for stock, weight in portfolio:
    print(f"     {stock.symbol}: {weight:.1%}")
    
    # 4. Factor exposures
exposures = strategy.calculate_factor_exposures(portfolio, calculator)

print("\\n   Factor Exposures:")
for factor, exposure in exposures.items():
    print(f"     {factor.capitalize()}: {exposure:+.2f}")
    
    # 5. Factor timing
print("\\n4. Factor Timing (Dynamic Allocation)")

timing = FactorTimingStrategy()

regimes = ['bull_market', 'bear_market', 'high_inflation']

for regime in regimes:
    weights = timing.get_factor_weights(regime)
print(f"\\n   {regime.replace('_', ' ').title()}:")
for factor, weight in weights.items():
    if weight > 0:
        print(f"     {factor.capitalize()}: {weight:.0%}")
\`\`\`

---

## Smart Beta and Factor ETFs

### Smart Beta ETFs

**Definition**: Rules-based ETFs targeting specific factors

**Examples:**
- **iShares Edge MSCI USA Value** (VLUE): Value factor
- **iShares Edge MSCI USA Momentum** (MTUM): Momentum factor
- **iShares Edge MSCI USA Quality** (QUAL): Quality factor
- **iShares Edge MSCI Min Vol** (USMV): Low volatility
- **Invesco S&P 500 Pure Value** (RPV): Pure value play

**Advantages:**
- Low cost (0.15-0.25% expense ratio)
- Liquid (tight spreads)
- Transparent rules
- Tax efficient

**Disadvantages:**
- Factor timing risk (wrong factor for regime)
- Crowding (everyone doing same thing)
- Limited customization

### Multi-Factor ETFs

**Examples:**
- **Goldman Sachs ActiveBeta** (GSLC): 4 factors
- **JPMorgan Diversified Return** (JPUS): 5 factors
- **MSCI USA Factor Mix** (QUS): Multi-factor

**Performance:**
- Typically outperform market cap weighted
- Lower volatility than single factors
- Consistent 1-2% alpha over long term

---

## Factor Investing in Practice

### Portfolio Construction Steps

1. **Define Universe**: S&P 500, Russell 3000, All-World
2. **Calculate Factor Scores**: Use quantitative metrics
3. **Rank Stocks**: Composite score or individual factors
4. **Select Top Stocks**: Top 20-100 stocks
5. **Weight Stocks**: Equal weight or factor-weighted
6. **Rebalance**: Monthly, quarterly, or annually
7. **Risk Management**: Sector limits, concentration limits

### Backtesting Considerations

- **Survivorship Bias**: Include delisted stocks
- **Look-Ahead Bias**: Only use data available at time
- **Transaction Costs**: Factor strategies trade frequently
- **Implementation Lag**: Data delay, execution time
- **Factor Definitions**: Consistent methodology

### Risk Management

**Diversification:**
- Multiple factors (reduce factor risk)
- Multiple geographies
- Sector constraints

**Rebalancing:**
- Too frequent → high costs
- Too infrequent → factor drift
- Optimal: Quarterly for most factors

**Factor Exposure Monitoring:**
- Track factor loadings over time
- Ensure intended exposures maintained
- Avoid unintended factor bets

---

## Real-World Examples

### AQR Capital Management

**Strategy**: Multi-factor quantitative
**AUM**: $100B+
**Approach**:
- Value, momentum, quality, carry
- Global multi-asset
- Academic research-driven

### Dimensional Fund Advisors (DFA)

**Strategy**: Factor-based indexing
**AUM**: $600B+
**Approach**:
- Fama-French research
- Small-cap and value tilt
- Long-term holding periods

### Research Affiliates

**Strategy**: Fundamental indexing
**Innovation**: Weight by fundamentals (revenue, book value)
**Performance**: Outperformed cap-weighted by ~2% annually

---

## Summary and Key Takeaways

**Factor Investing Works When:**
- Long-term horizon (5-10+ years)
- Disciplined implementation
- Diversified across factors
- Low costs

**Factor Investing Fails When:**
- Short-term focus (factors have dry spells)
- Poor execution (high costs)
- Single factor concentration
- Chasing recent winners

**Critical Success Factors:**
1. **Multiple Factors**: Don't rely on one
2. **Long Horizon**: Factors work over years, not months
3. **Low Costs**: Turnover erodes returns
4. **Discipline**: Stay the course in drawdowns
5. **Risk Management**: Diversify and monitor

**Best Practices:**
- Start with 3-5 core factors
- Rebalance quarterly
- Monitor factor exposures
- Consider factor timing (advanced)
- Use ETFs for simplicity or build custom for control

**Next Section:** Multi-Asset Strategies
`,
};
