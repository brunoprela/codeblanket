export const factorModels = {
    title: 'Factor Models (Fama-French)',
    id: 'factor-models',
    content: `
# Factor Models (Fama-French)

## Introduction

Factor models revolutionized finance by explaining why some portfolios outperform others. Instead of viewing returns as random, factor models decompose returns into systematic components (factors) plus idiosyncratic noise.

**The Question Factor Models Answer**:

"Why did my portfolio return 15% while the S&P 500 returned 12%?"

- **Old answer**: "I picked good stocks" (vague)
- **Factor model answer**: "I had 1.2x market exposure (+3%), 0.5x small-cap tilt (+1%), and 0.3x value tilt (+0.5%), totaling 3% alpha after risk adjustment."

**Evolution of Factor Models**:

1. **CAPM** (1964): Single factor (market beta)
2. **Fama-French 3-Factor** (1993): + Size + Value
3. **Fama-French 5-Factor** (2015): + Profitability + Investment
4. **Carhart 4-Factor** (1997): 3-Factor + Momentum
5. **Modern**: 100+ factors proposed (quality, low vol, etc.)

**Why Factors Matter**:

- **Performance Attribution**: Understand where returns come from
- **Risk Management**: Quantify factor exposures
- **Portfolio Construction**: Target specific factor tilts
- **Smart Beta**: Build factor-based ETFs
- **Academic Foundation**: Most tested and validated framework

**Real-World Applications**:

- **Dimensional Fund Advisors (DFA)**: $650B AUM, built on Fama-French
- **AQR**: Quantitative hedge fund using multi-factor models
- **Factor ETFs**: MSCI, iShares, Vanguard all offer factor ETFs
- **Performance Evaluation**: Standard tool for evaluating managers

**What You'll Learn**:

1. CAPM limitations and the need for multi-factor models
2. Fama-French 3-Factor model (market, size, value)
3. Fama-French 5-Factor model (+ profitability, investment)
4. Other important factors (momentum, quality, low vol)
5. Factor regression and attribution
6. Building factor portfolios
7. Implementation in Python with real data

---

## CAPM: The Single-Factor Model

### Capital Asset Pricing Model

**CAPM Equation**:

\\[
E(R_i) = R_f + \\beta_i [E(R_m) - R_f]
\\]

Where:
- \\( E(R_i) \\) = Expected return of asset i
- \\( R_f \\) = Risk-free rate
- \\( \\beta_i \\) = Asset's market beta
- \\( E(R_m) - R_f \\) = Market risk premium

**Beta Calculation**:

\\[
\\beta_i = \\frac{Cov(R_i, R_m)}{Var(R_m)}
\\]

**Interpretation**:
- \\( \\beta = 1 \\): Moves with market
- \\( \\beta > 1 \\): More volatile than market (amplifies moves)
- \\( \\beta < 1 \\): Less volatile than market (dampens moves)
- \\( \\beta = 0 \\): Uncorrelated with market

### CAPM's Limitations

**Problem 1: Size Effect**

Small-cap stocks outperform large-cap stocks historically, even after adjusting for beta.

**Evidence**: Banz (1981) showed smallest quintile of stocks outperformed largest by 5%+ annually.

**CAPM prediction**: If small-caps have same beta as large-caps, should have same expected return. **Reality**: Small-caps outperform.

**Problem 2: Value Effect**

Value stocks (low P/B, high B/M) outperform growth stocks, even after adjusting for beta.

**Evidence**: Value stocks returned 5% more annually than growth stocks (1926-2023).

**CAPM prediction**: Value and growth with same beta should have same return. **Reality**: Value outperforms.

**Problem 3: Low Beta Anomaly**

Low-beta stocks have higher risk-adjusted returns than high-beta stocks.

**CAPM prediction**: Higher beta → higher return (linear relationship).  
**Reality**: Low-beta stocks have higher Sharpe ratios.

**Problem 4: Poor Empirical Fit**

CAPM explains only ~70% of return variation. 30% unexplained!

**Solution**: Need additional factors beyond market beta.

---

## Fama-French 3-Factor Model

### The Breakthrough (1993)

Eugene Fama and Kenneth French showed two additional factors explain much of what CAPM misses:

1. **SMB (Small Minus Big)**: Size factor
2. **HML (High Minus Low)**: Value factor

**3-Factor Model**:

\\[
R_i - R_f = \\alpha_i + \\beta_{i,M}(R_M - R_f) + \\beta_{i,SMB} SMB + \\beta_{i,HML} HML + \\epsilon_i
\\]

Where:
- \\( R_i - R_f \\) = Excess return of asset i
- \\( R_M - R_f \\) = Market excess return (market factor)
- \\( SMB \\) = Small Minus Big (size factor)
- \\( HML \\) = High Minus Low book-to-market (value factor)
- \\( \\alpha_i \\) = Intercept (alpha - manager skill after risk adjustment)
- \\( \\epsilon_i \\) = Idiosyncratic error

**Key Insight**: Alpha (\\( \\alpha_i \\)) is true skill. Everything else is factor exposure (systematic risk).

### Market Factor (\\( R_M - R_f \\))

**Construction**: Return of market portfolio minus risk-free rate.

**Proxy**: S&P 500 or CRSP value-weighted index.

**Interpretation**: Captures broad market movements. Beta on this factor = traditional CAPM beta.

### Size Factor (SMB)

**SMB = Small Minus Big**

**Construction**:
1. Rank all stocks by market cap
2. Form portfolios:
   - Small: Bottom 30% by market cap
   - Big: Top 30% by market cap
3. SMB = Return(Small) - Return(Big)

**Historical Performance**:
- SMB premium: ~3% annually (1926-2023)
- Volatile: Large periods where big caps outperform

**Interpretation**: 
- Positive SMB exposure: Portfolio tilted toward small caps
- Negative SMB exposure: Portfolio tilted toward large caps

**Why Small Cap Premium Exists**:
1. **Liquidity**: Small caps less liquid → higher required return
2. **Information**: Less analyst coverage → inefficiencies
3. **Risk**: Higher bankruptcy risk
4. **Behavioral**: Institutional bias toward large caps

### Value Factor (HML)

**HML = High Minus Low (Book-to-Market)**

**Construction**:
1. Calculate book-to-market ratio for all stocks (B/M = Book Value / Market Value)
2. Form portfolios:
   - Value: Top 30% by B/M (high B/M = low P/B = "cheap")
   - Growth: Bottom 30% by B/M (low B/M = high P/B = "expensive")
3. HML = Return(Value) - Return(Growth)

**Historical Performance**:
- HML premium: ~5% annually (1926-2023)
- Strongest in recessions

**Interpretation**:
- Positive HML exposure: Value tilt (cheap stocks)
- Negative HML exposure: Growth tilt (expensive stocks)

**Why Value Premium Exists**:
1. **Risk**: Value stocks riskier (financial distress)
2. **Behavioral**: Investors overextrapolate growth → overpay for growth stocks
3. **Time-varying risk**: Value suffers more in bad times → higher required return

### Using the 3-Factor Model

**Performance Attribution**:

\\[
R_i - R_f = \\alpha + 0.95 \\times (R_M - R_f) + 0.30 \\times SMB + 0.40 \\times HML
\\]

**Interpretation**:
- \\( \\alpha \\): True skill (after adjusting for all factor exposures)
- \\( \\beta_M = 0.95 \\): Slightly less volatile than market
- \\( \\beta_{SMB} = 0.30 \\): Moderate small-cap tilt
- \\( \\beta_{HML} = 0.40 \\): Strong value tilt

**Conclusion**: Most of return explained by value tilt. Low alpha suggests little manager skill beyond factor exposures.

---

## Fama-French 5-Factor Model

### Adding Two More Factors (2015)

Fama and French extended the model with two profitability and investment factors:

4. **RMW (Robust Minus Weak)**: Profitability factor
5. **CMA (Conservative Minus Aggressive)**: Investment factor

**5-Factor Model**:

\\[
R_i - R_f = \\alpha + \\beta_M (R_M - R_f) + \\beta_{SMB} SMB + \\beta_{HML} HML + \\beta_{RMW} RMW + \\beta_{CMA} CMA + \\epsilon
\\]

### Profitability Factor (RMW)

**RMW = Robust Minus Weak**

**Construction**:
1. Calculate operating profitability = (Revenue - COGS - SG&A) / Book Equity
2. Form portfolios:
   - Robust: Top 30% by profitability
   - Weak: Bottom 30% by profitability
3. RMW = Return(Robust) - Return(Weak)

**Historical Performance**: ~3% annual premium

**Interpretation**:
- Positive RMW: Tilt toward profitable companies
- Negative RMW: Tilt toward unprofitable companies

**Why Profitability Premium**:
1. **Quality**: Profitable companies less risky
2. **Sustainability**: Profits indicate competitive advantage
3. **Mispricing**: Market undervalues profitability

### Investment Factor (CMA)

**CMA = Conservative Minus Aggressive**

**Construction**:
1. Calculate asset growth = (Total Assets_t - Total Assets_t-1) / Total Assets_t-1
2. Form portfolios:
   - Conservative: Bottom 30% by asset growth (slow growers)
   - Aggressive: Top 30% by asset growth (fast growers)
3. CMA = Return(Conservative) - Return(Aggressive)

**Historical Performance**: ~3% annual premium

**Interpretation**:
- Positive CMA: Tilt toward companies that invest conservatively
- Negative CMA: Tilt toward companies that invest aggressively

**Why Investment Premium**:
1. **q-Theory**: Aggressive investors have lower future returns (diminishing returns to capital)
2. **Agency**: Managers overinvest (empire building)
3. **Mispricing**: Market overvalues growth through investment

### 5-Factor vs 3-Factor

**Improved Explanatory Power**:
- 3-Factor R²: ~85-90%
- 5-Factor R²: ~90-93%

**Key Finding**: HML (value factor) becomes less significant when RMW and CMA are included.

**Implication**: Value premium may be driven by profitability and investment patterns, not just B/M ratio.

---

## Other Important Factors

### Momentum Factor (UMD)

**UMD = Up Minus Down** (Carhart 1997)

**Construction**:
1. Calculate 12-month momentum (returns over past year, skipping most recent month)
2. Winners: Top 30% by momentum
3. Losers: Bottom 30% by momentum
4. UMD = Return(Winners) - Return(Losers)

**Historical Performance**: ~8% annual premium (strongest of all factors!)

**Carhart 4-Factor Model**: FF3 + Momentum

\\[
R_i - R_f = \\alpha + \\beta_M (R_M - R_f) + \\beta_{SMB} SMB + \\beta_{HML} HML + \\beta_{UMD} UMD + \\epsilon
\\]

**Why Momentum Works**:
1. **Behavioral**: Under-reaction to news (slow information diffusion)
2. **Behavioral**: Herding (trend-following)
3. **Risk**: Time-varying risk premiums

### Quality Factor

**Multiple definitions**:
- High profitability, low leverage, stable earnings
- High ROE, low accruals
- Strong balance sheet, consistent growth

**Construction** (simplified):
1. Score stocks on multiple quality metrics
2. Quality = Top minus Bottom quintile

**Historical Performance**: ~3-4% annual premium

**Why Quality Premium**:
1. **Risk**: High-quality companies less risky
2. **Persistence**: Quality is persistent (hard to disrupt great companies)
3. **Mispricing**: Market undervalues quality

### Low Volatility Factor

**Anomaly**: Low-volatility stocks outperform high-volatility stocks on risk-adjusted basis.

**Construction**:
1. Calculate historical volatility
2. Low Vol = Bottom quintile by volatility
3. High Vol = Top quintile by volatility
4. Low Vol premium = Return(Low Vol) - Return(High Vol) *adjusted for beta*

**Historical Performance**: ~2% annual premium

**Why Low Vol Works**:
1. **Behavioral**: Investors prefer lottery-like stocks (high vol)
2. **Leverage constraints**: Institutions can't use leverage, so buy high-beta stocks instead
3. **Benchmarking**: Managers judged relative to benchmark, incentivized to take on more risk

---

## Factor Regression and Attribution

### Running Factor Regressions

**Objective**: Decompose portfolio returns into factor exposures and alpha.

**Steps**:
1. Obtain factor returns (from Kenneth French's data library)
2. Calculate portfolio excess returns
3. Regress portfolio returns on factor returns
4. Interpret coefficients (factor loadings) and alpha

### Python Implementation

\`\`\`python
"""
Fama-French Factor Models Implementation
"""

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from datetime import datetime, timedelta
import pandas_datareader as pdr
from typing import Dict, Tuple

class FamaFrenchAnalysis:
    """
    Perform Fama-French factor analysis.
    """
    
    def __init__(self, start_date: str, end_date: str):
        """
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        """
        self.start_date = start_date
        self.end_date = end_date
        self.factors = None
        
    def load_ff_factors(self, model: str = '3factor') -> pd.DataFrame:
        """
        Load Fama-French factors from Ken French's data library.
        
        Args:
            model: '3factor', '5factor', or '4factor' (with momentum)
        
        Returns:
            DataFrame of factor returns
        """
        # Factor codes in Kenneth French data library
        factor_codes = {
            '3factor': 'F-F_Research_Data_Factors',
            '5factor': 'F-F_Research_Data_5_Factors_2x3',
            '4factor': 'F-F_Research_Data_Factors'  # Will add momentum separately
        }
        
        try:
            # Load factors from Ken French data library
            factors = pdr.DataReader(
                factor_codes[model],
                'famafrench',
                start=self.start_date,
                end=self.end_date
            )[0]  # [0] gets monthly data
            
            # Convert to decimal (data is in percentage)
            factors = factors / 100
            
            # If 4-factor, need to add momentum
            if model == '4factor':
                momentum = pdr.DataReader(
                    'F-F_Momentum_Factor',
                    'famafrench',
                    start=self.start_date,
                    end=self.end_date
                )[0] / 100
                factors = factors.join(momentum, how='inner')
            
            self.factors = factors
            return factors
            
        except Exception as e:
            print(f"Error loading Fama-French factors: {e}")
            print("Using simulated factors for demonstration...")
            return self._simulate_factors(model)
    
    def _simulate_factors(self, model: str) -> pd.DataFrame:
        """Simulate factor returns for demonstration (if data library unavailable)."""
        dates = pd.date_range(self.start_date, self.end_date, freq='M')
        
        np.random.seed(42)
        
        # Simulate with realistic properties
        data = {
            'Mkt-RF': np.random.normal(0.007, 0.04, len(dates)),  # Market premium
            'SMB': np.random.normal(0.002, 0.03, len(dates)),     # Size
            'HML': np.random.normal(0.003, 0.03, len(dates)),     # Value
            'RF': np.full(len(dates), 0.0003)                     # Risk-free ~4% annual
        }
        
        if model in ['5factor']:
            data['RMW'] = np.random.normal(0.002, 0.02, len(dates))  # Profitability
            data['CMA'] = np.random.normal(0.002, 0.02, len(dates))  # Investment
        
        if model == '4factor':
            data['Mom'] = np.random.normal(0.005, 0.04, len(dates))  # Momentum
        
        return pd.DataFrame(data, index=dates)
    
    def run_regression(self, 
                      portfolio_returns: pd.Series,
                      model: str = '3factor') -> Dict:
        """
        Run factor regression.
        
        Args:
            portfolio_returns: Portfolio returns (monthly)
            model: '3factor', '5factor', or '4factor'
        
        Returns:
            Dict with regression results
        """
        # Load factors if not already loaded
        if self.factors is None:
            self.load_ff_factors(model)
        
        # Align data
        data = pd.DataFrame({
            'Portfolio': portfolio_returns,
            **self.factors
        }).dropna()
        
        # Calculate excess returns
        excess_returns = data['Portfolio'] - data['RF']
        
        # Prepare factor data for regression
        if model == '3factor':
            X = data[['Mkt-RF', 'SMB', 'HML']]
        elif model == '5factor':
            X = data[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
        elif model == '4factor':
            X = data[['Mkt-RF', 'SMB', 'HML', 'Mom']]
        
        # Add constant for alpha
        X = sm.add_constant(X)
        
        # Run regression
        model_fit = sm.OLS(excess_returns, X).fit()
        
        # Extract results
        results = {
            'alpha': model_fit.params['const'],
            'alpha_tstat': model_fit.tvalues['const'],
            'alpha_pvalue': model_fit.pvalues['const'],
            'betas': {col: model_fit.params[col] for col in X.columns if col != 'const'},
            'tstats': {col: model_fit.tvalues[col] for col in X.columns if col != 'const'},
            'pvalues': {col: model_fit.pvalues[col] for col in X.columns if col != 'const'},
            'rsquared': model_fit.rsquared,
            'adj_rsquared': model_fit.rsquared_adj,
            'model_summary': model_fit.summary()
        }
        
        return results
    
    def performance_attribution(self, results: Dict, avg_factor_returns: pd.Series) -> pd.DataFrame:
        """
        Decompose portfolio return into factor contributions.
        
        Args:
            results: Regression results
            avg_factor_returns: Average factor returns over period
        
        Returns:
            DataFrame with attribution
        """
        attribution = []
        
        # Alpha contribution
        attribution.append({
            'Component': 'Alpha (Skill)',
            'Beta': 1.0,
            'Factor Return': results['alpha'],
            'Contribution': results['alpha']
        })
        
        # Factor contributions
        for factor, beta in results['betas'].items():
            factor_return = avg_factor_returns.get(factor, 0)
            contribution = beta * factor_return
            
            attribution.append({
                'Component': factor,
                'Beta': beta,
                'Factor Return': factor_return,
                'Contribution': contribution
            })
        
        df = pd.DataFrame(attribution)
        df['Contribution %'] = df['Contribution'] / df['Contribution'].sum() * 100
        
        return df

# Example Usage
print("=== Fama-French Factor Analysis Demo ===\\n")

# Date range
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

# Create analyzer
ff = FamaFrenchAnalysis(
    start_date.strftime('%Y-%m-%d'),
    end_date.strftime('%Y-%m-%d')
)

# Load Fama-French 3-Factor data
print("Loading Fama-French factors...")
factors = ff.load_ff_factors(model='3factor')
print(f"✓ Loaded {len(factors)} months of factor data\\n")

print("Factor Summary Statistics:")
print(factors.describe())

# Get portfolio returns (example: small-cap value portfolio)
# Simulating with tilts toward SMB and HML
np.random.seed(42)
dates = factors.index

# Portfolio = 1.0*Market + 0.5*SMB + 0.6*HML + alpha + noise
portfolio_returns = (
    1.0 * factors['Mkt-RF'] +
    0.5 * factors['SMB'] +
    0.6 * factors['HML'] +
    0.001 +  # 0.1% monthly alpha
    np.random.normal(0, 0.02, len(factors))  # Idiosyncratic risk
)
portfolio_returns = pd.Series(portfolio_returns, index=dates)

# Run 3-factor regression
print("\\n=== 3-Factor Regression Results ===\\n")
results = ff.run_regression(portfolio_returns, model='3factor')

print(f"Alpha: {results['alpha']:.4f} ({results['alpha']*12:.2%} annualized)")
print(f"  t-stat: {results['alpha_tstat']:.2f}")
print(f"  p-value: {results['alpha_pvalue']:.4f}")
print(f"  {'Significant' if results['alpha_pvalue'] < 0.05 else 'Not significant'} at 5% level\\n")

print("Factor Loadings (Betas):")
for factor, beta in results['betas'].items():
    tstat = results['tstats'][factor]
    pval = results['pvalues'][factor]
    sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
    print(f"  {factor:10s}: {beta:6.3f} (t={tstat:5.2f}) {sig}")

print(f"\\nR-squared: {results['rsquared']:.4f}")
print(f"Adjusted R-squared: {results['adj_rsquared']:.4f}")

# Performance attribution
print("\\n=== Performance Attribution ===\\n")
avg_factor_returns = factors.mean()
attribution = ff.performance_attribution(results, avg_factor_returns)
print(attribution.to_string(index=False))

print(f"\\nTotal Monthly Return: {attribution['Contribution'].sum():.4f} ({attribution['Contribution'].sum()*12:.2%} annualized)")
\`\`\`

---

## Building Factor Portfolios

### Factor Tilting

**Objective**: Overweight factors with premiums (value, small-cap, profitability, momentum).

**Example Portfolio Construction**:

**Traditional 60/40**:
- 60% S&P 500 (market cap weighted)
- 40% Bonds

**Factor-Tilted 60/40**:
- 20% Large-cap blend
- 20% Small-cap value (size + value tilts)
- 10% Momentum stocks
- 10% Quality stocks
- 40% Bonds

**Expected Impact**: Higher expected return from factor premiums, possibly higher volatility.

### Factor ETFs

**Implementation**: Use factor ETFs to gain exposures.

**Examples**:
- **Value**: iShares MSCI USA Value (VLUE), Vanguard Value ETF (VTV)
- **Small-Cap**: iShares Russell 2000 (IWM), Vanguard Small-Cap (VB)
- **Momentum**: iShares MSCI USA Momentum (MTUM)
- **Quality**: iShares MSCI USA Quality (QUAL)
- **Multi-Factor**: Vanguard Multi-Factor ETF (VFMF), iShares Edge MSCI Multifactor (LRGF)

**Advantages**:
- Systematic exposure to factors
- Low cost (0.15-0.30% expense ratios)
- Diversified (100-1000+ stocks per ETF)
- Liquid (easy to trade)

### Factor Timing

**Question**: Should you rotate between factors based on market conditions?

**Evidence**: Mixed. Some factors perform better in certain regimes:
- **Value**: Outperforms in recoveries, underperforms in growth phases
- **Momentum**: Works in trending markets, fails in choppy markets
- **Quality**: Defensive, performs better in downturns
- **Size**: Small-cap does well in expansions

**Approach**:
1. **Strategic**: Constant factor allocation (DFA approach)
2. **Tactical**: Rotate based on valuation or macro signals
3. **Dynamic**: Adjust based on factor momentum/valuations

**Recommendation**: Factor timing is hard. Most evidence supports strategic constant allocation.

---

## Real-World Applications

### Dimensional Fund Advisors (DFA)

**Strategy**: Systematic factor-based investing (Fama-French disciples).

**Approach**:
1. Tilt toward small-cap and value (SMB, HML factors)
2. Avoid overpriced growth stocks
3. Trade patiently (minimize transaction costs)
4. Tax-manage (harvest losses)

**Results**: DFA funds have outperformed comparable benchmarks over long periods.

**AUM**: $650B+ (institutional and advisor-sold)

### AQR Capital Management

**Strategy**: Multi-factor quantitative strategies.

**Factors Used**:
- Value (across all asset classes)
- Momentum (time-series and cross-sectional)
- Carry (interest rate differential, commodity contango)
- Defensive (quality, low beta)

**Approach**: Combine factors across stocks, bonds, commodities, currencies.

**Results**: Strong long-term performance, though challenged in recent years (value struggled 2017-2020).

**AUM**: $100B+ hedge fund and mutual fund assets

### Factor ETF Performance

**2010-2020 Performance** (annualized):
- S&P 500: 13.9%
- MSCI USA Value: 11.2% (value struggled post-2008)
- MSCI USA Momentum: 15.8% (strong)
- MSCI USA Quality: 14.5% (solid)
- Russell 2000 (Small-Cap): 11.5% (underperformed large-cap)

**Key Lesson**: Factor premiums are not guaranteed every period. Long-term view required.

---

## Practical Considerations

### Factor Diversification

**Don't just tilt one factor**. Factors have their own cycles.

**Diversified Factor Portfolio**:
- 25% Value
- 25% Momentum
- 25% Quality
- 25% Size

**Rationale**: When value struggles (2017-2020), momentum may thrive. Quality defensive in downturns.

### Implementation Costs

**Factor strategies require active management**:
- Higher turnover than market cap weighting
- Momentum: 50-100% annual turnover
- Value: 20-40% annual turnover

**Costs**:
- ETF expense ratios: 0.15-0.30%
- Trading costs: 0.10-0.30% annually
- **Total**: 0.25-0.60% vs 0.03% for S&P 500 index fund

**Breakeven**: Need factor premium > costs. Historically yes, but not guaranteed.

### Factor Crowding

**Problem**: As factor investing becomes popular, premiums may shrink.

**Evidence**: 
- Value premium weaker post-publication (post-1993)
- Momentum still works but lower Sharpe ratio

**Future**: Factor premiums may persist (risk-based) or diminish (mispricing corrected).

---

## Practical Exercises

### Exercise 1: Factor Regression

Run 3-factor and 5-factor regressions on:
1. Your portfolio (or mutual fund)
2. Warren Buffett (BRK.B)
3. ARK Innovation ETF (ARKK)

Compare factor loadings. What drives their returns?

### Exercise 2: Build Factor Portfolio

Construct portfolio with:
- 25% each: Value, Momentum, Quality, Low Vol factors

Backtest over 10 years. Compare to S&P 500.

### Exercise 3: Factor Performance Analysis

Download factor returns 1926-2024. Analyze:
- Which factor had highest Sharpe ratio?
- Correlations between factors
- Worst drawdowns for each factor
- Factor performance in recessions vs expansions

### Exercise 4: Factor Timing

Test simple factor timing rules:
- Momentum: Overweight when 12-month return > 0
- Value: Overweight when CAPE ratio > 25

Does timing add value vs constant allocation?

### Exercise 5: Multi-Factor Portfolio Optimizer

Build optimizer that:
1. Takes factor exposures as constraints
2. Minimizes tracking error to custom factor benchmark
3. Implements with individual stocks or ETFs

---

## Key Takeaways

1. **Factor Models Decompose Returns**: Into systematic factor exposures + alpha (skill).

2. **CAPM Limitations**: Single factor (market beta) explains only ~70% of returns. Missing size and value effects.

3. **Fama-French 3-Factor**: + Size (SMB) + Value (HML). Explains ~85-90% of returns.

4. **Fama-French 5-Factor**: + Profitability (RMW) + Investment (CMA). Explains ~90-93%.

5. **Other Key Factors**:
   - Momentum (UMD): ~8% premium (strongest)
   - Quality: ~3-4% premium
   - Low Volatility: ~2% premium

6. **Factor Regression**:
\\[
R - R_f = \\alpha + \\beta_M (R_M - R_f) + \\beta_{SMB} SMB + \\beta_{HML} HML + \\epsilon
\\]
Alpha = skill after adjusting for factors.

7. **Factor Premiums** (Historical, 1926-2023):
   - Market: 8% over T-bills
   - Size (SMB): 3%
   - Value (HML): 5%
   - Momentum: 8%
   - Profitability: 3%

8. **Real-World Usage**:
   - DFA: $650B using Fama-French factors
   - AQR: Multi-factor quant strategies
   - Factor ETFs: Growing category

9. **Implementation**: Factor ETFs or systematic stock selection. Costs 0.25-0.60% vs 0.03% for index.

10. **Caveat**: Factor premiums not guaranteed every period. Require long-term commitment (10+ years) and diversification across factors.

In the next section, we'll explore **Risk Budgeting**: allocating portfolio risk (not just capital) across assets and factors.
`,
};

