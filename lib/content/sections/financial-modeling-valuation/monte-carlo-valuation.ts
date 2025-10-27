export const monteCarloValuation = {
    title: 'Monte Carlo Simulation for Valuation',
    id: 'monte-carlo-valuation',
    content: `
# Monte Carlo Simulation for Valuation

## Introduction

**Monte Carlo simulation** is a probabilistic modeling technique that runs thousands of scenarios to generate a distribution of possible outcomes, rather than a single point estimate.

**Named After**: Monte Carlo Casino in Monaco (random outcomes, like dice rolls)

**Core Principle**: Instead of asking "What is the company worth?", ask "What is the probability distribution of company value?"

**Why Monte Carlo Over Deterministic Models:**

Traditional approach:
- Single DCF: "Company is worth $10B"
- Sensitivity: "Worth $8B-$12B depending on assumptions"
- 3 scenarios: "Bear $8B, Base $10B, Bull $12B"

Monte Carlo approach:
- 10,000 scenarios: "70% probability company is worth $9B-$11B"
- "15% chance worth >$12B (upside)"
- "10% chance worth <$8B (downside)"
- "Expected value: $10.2B ± $2.1B (1 std dev)"

**When to Use Monte Carlo:**
- **High uncertainty**: Multiple variables uncertain simultaneously
- **Risk quantification**: Need to quantify downside/upside probabilities
- **Option pricing**: Valuing flexibility, abandonment options, growth options
- **Regulatory**: Required for certain valuations (insurance reserves, etc.)
- **Portfolio analysis**: Modeling portfolio of investments with correlations

**By the end of this section, you'll be able to:**
- Build Monte Carlo simulations for DCF valuation
- Define probability distributions for key variables
- Interpret simulation results (percentiles, confidence intervals)
- Model correlations between variables
- Use Monte Carlo for risk management and decision-making
- Implement production-grade Monte Carlo in Python

---

## Monte Carlo Framework

### Step-by-Step Process

1. **Identify Uncertain Variables**
   - Revenue growth, EBITDA margins, WACC, terminal growth
   
2. **Define Probability Distributions**
   - Normal, lognormal, triangular, uniform
   - Parameters: mean, standard deviation, min/max
   
3. **Model Correlations**
   - Revenue growth and margins often correlated
   - WACC and terminal growth inversely correlated (rising rates)
   
4. **Run Simulations**
   - 10,000+ iterations
   - Each iteration: Draw random values, calculate valuation
   
5. **Analyze Results**
   - Distribution of outcomes (histogram)
   - Percentiles (P10, P50, P90)
   - Risk metrics (VaR, CVaR)

### Types of Probability Distributions

**1. Normal Distribution**
- Bell curve, symmetric
- Defined by mean (μ) and standard deviation (σ)
- Use for: Growth rates, margins (when no skew expected)

**2. Lognormal Distribution**
- Right-skewed (long tail of high values)
- Ensures positive values (can't have negative stock price)
- Use for: Stock prices, valuations, revenues

**3. Triangular Distribution**
- Defined by min, most likely, max
- Simple, intuitive
- Use for: When limited data (expert judgment on min/max)

**4. Uniform Distribution**
- All values equally likely
- Use for: When no information (pure uncertainty)

---

## Building Monte Carlo DCF

### Basic Implementation

\`\`\`python
"""
Monte Carlo Simulation for DCF Valuation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class DistributionParams:
    """Parameters for probability distribution"""
    dist_type: str  # 'normal', 'lognormal', 'triangular', 'uniform'
    params: Tuple  # Distribution-specific parameters

class MonteCarloValuation:
    """
    Monte Carlo simulation for DCF valuation.
    
    Models multiple uncertain variables with probability distributions,
    runs thousands of simulations, outputs distribution of valuations.
    """
    
    def __init__(
        self,
        base_revenue: float,
        projection_years: int = 10,
        n_simulations: int = 10000,
        random_seed: int = 42
    ):
        self.base_revenue = base_revenue
        self.projection_years = projection_years
        self.n_simulations = n_simulations
        np.random.seed(random_seed)
        
        self.results = None
    
    def define_distributions(self) -> Dict[str, DistributionParams]:
        """
        Define probability distributions for uncertain variables.
        
        Returns:
            Dictionary of variable distributions
        """
        
        distributions = {
            'revenue_growth': DistributionParams(
                dist_type='normal',
                params=(0.12, 0.05)  # mean 12%, std 5%
            ),
            'ebitda_margin': DistributionParams(
                dist_type='normal',
                params=(0.25, 0.03)  # mean 25%, std 3%
            ),
            'wacc': DistributionParams(
                dist_type='normal',
                params=(0.095, 0.015)  # mean 9.5%, std 1.5%
            ),
            'terminal_growth': DistributionParams(
                dist_type='normal',
                params=(0.025, 0.005)  # mean 2.5%, std 0.5%
            ),
            'tax_rate': DistributionParams(
                dist_type='uniform',
                params=(0.19, 0.23)  # range 19-23% (policy uncertainty)
            ),
            'capex_pct_revenue': DistributionParams(
                dist_type='triangular',
                params=(0.03, 0.05, 0.08)  # min 3%, most likely 5%, max 8%
            ),
            'nwc_pct_revenue': DistributionParams(
                dist_type='normal',
                params=(0.12, 0.02)  # mean 12%, std 2%
            )
        }
        
        return distributions
    
    def draw_sample(self, dist_params: DistributionParams) -> float:
        """Draw random sample from distribution"""
        
        if dist_params.dist_type == 'normal':
            mean, std = dist_params.params
            return np.random.normal(mean, std)
        
        elif dist_params.dist_type == 'lognormal':
            mean, std = dist_params.params
            return np.random.lognormal(mean, std)
        
        elif dist_params.dist_type == 'triangular':
            low, mode, high = dist_params.params
            return np.random.triangular(low, mode, high)
        
        elif dist_params.dist_type == 'uniform':
            low, high = dist_params.params
            return np.random.uniform(low, high)
        
        else:
            raise ValueError(f"Unknown distribution type: {dist_params.dist_type}")
    
    def project_cash_flows(
        self,
        revenue_growth: float,
        ebitda_margin: float,
        tax_rate: float,
        capex_pct: float,
        nwc_pct: float
    ) -> Tuple[List[float], float]:
        """
        Project free cash flows for one simulation run.
        
        Returns:
            Tuple of (list of annual FCF, final year EBITDA)
        """
        
        fcf_stream = []
        revenue = self.base_revenue
        nwc = self.base_revenue * nwc_pct
        
        for year in range(self.projection_years):
            # Grow revenue
            revenue = revenue * (1 + revenue_growth)
            
            # Calculate EBITDA
            ebitda = revenue * ebitda_margin
            
            # Simplified: D&A = 4% of revenue
            da = revenue * 0.04
            ebit = ebitda - da
            
            # Taxes
            nopat = ebit * (1 - tax_rate)
            
            # CapEx
            capex = revenue * capex_pct
            
            # Working capital
            new_nwc = revenue * nwc_pct
            change_nwc = new_nwc - nwc
            nwc = new_nwc
            
            # FCF
            fcf = nopat + da - capex - change_nwc
            fcf_stream.append(fcf)
        
        return fcf_stream, ebitda
    
    def calculate_enterprise_value(
        self,
        fcf_stream: List[float],
        final_ebitda: float,
        wacc: float,
        terminal_growth: float
    ) -> float:
        """
        Calculate enterprise value from cash flows.
        
        Args:
            fcf_stream: List of projected FCF
            final_ebitda: EBITDA in final projection year
            wacc: Weighted average cost of capital
            terminal_growth: Perpetual growth rate
        
        Returns:
            Enterprise value
        """
        
        # PV of projected FCF (mid-year convention)
        pv_fcf = sum(
            fcf / (1 + wacc) ** (year + 0.5)
            for year, fcf in enumerate(fcf_stream, start=1)
        )
        
        # Terminal value
        fcf_final = fcf_stream[-1]
        
        # Ensure WACC > terminal growth (avoid infinite value)
        if wacc <= terminal_growth:
            terminal_growth = wacc - 0.02  # Cap at WACC - 2%
        
        terminal_value = (fcf_final * (1 + terminal_growth)) / (wacc - terminal_growth)
        
        # PV of terminal value
        pv_terminal = terminal_value / (1 + wacc) ** self.projection_years
        
        # Enterprise value
        ev = pv_fcf + pv_terminal
        
        return ev
    
    def run_simulation(self) -> pd.DataFrame:
        """
        Run Monte Carlo simulation.
        
        Returns:
            DataFrame with simulation results
        """
        
        distributions = self.define_distributions()
        
        results = []
        
        for i in range(self.n_simulations):
            # Draw random samples for all variables
            revenue_growth = self.draw_sample(distributions['revenue_growth'])
            ebitda_margin = self.draw_sample(distributions['ebitda_margin'])
            wacc = self.draw_sample(distributions['wacc'])
            terminal_growth = self.draw_sample(distributions['terminal_growth'])
            tax_rate = self.draw_sample(distributions['tax_rate'])
            capex_pct = self.draw_sample(distributions['capex_pct_revenue'])
            nwc_pct = self.draw_sample(distributions['nwc_pct_revenue'])
            
            # Apply constraints
            ebitda_margin = np.clip(ebitda_margin, 0.10, 0.40)  # 10-40% range
            terminal_growth = np.clip(terminal_growth, 0.01, 0.04)  # 1-4% range
            
            # Project cash flows
            fcf_stream, final_ebitda = self.project_cash_flows(
                revenue_growth=revenue_growth,
                ebitda_margin=ebitda_margin,
                tax_rate=tax_rate,
                capex_pct=capex_pct,
                nwc_pct=nwc_pct
            )
            
            # Calculate EV
            ev = self.calculate_enterprise_value(
                fcf_stream=fcf_stream,
                final_ebitda=final_ebitda,
                wacc=wacc,
                terminal_growth=terminal_growth
            )
            
            results.append({
                'simulation': i + 1,
                'enterprise_value': ev,
                'revenue_growth': revenue_growth,
                'ebitda_margin': ebitda_margin,
                'wacc': wacc,
                'terminal_growth': terminal_growth,
                'final_revenue': self.base_revenue * (1 + revenue_growth) ** self.projection_years,
                'final_ebitda': final_ebitda
            })
        
        self.results = pd.DataFrame(results)
        return self.results
    
    def analyze_results(self) -> Dict:
        """
        Analyze simulation results.
        
        Returns:
            Dictionary with statistics
        """
        
        if self.results is None:
            raise ValueError("Must run simulation first")
        
        ev = self.results['enterprise_value']
        
        analysis = {
            'mean': ev.mean(),
            'median': ev.median(),
            'std': ev.std(),
            'coefficient_of_variation': ev.std() / ev.mean(),
            
            # Percentiles
            'p05': ev.quantile(0.05),
            'p10': ev.quantile(0.10),
            'p25': ev.quantile(0.25),
            'p75': ev.quantile(0.75),
            'p90': ev.quantile(0.90),
            'p95': ev.quantile(0.95),
            
            # Range
            'min': ev.min(),
            'max': ev.max(),
            'range': ev.max() - ev.min(),
            'iqr': ev.quantile(0.75) - ev.quantile(0.25),
            
            # Risk metrics
            'probability_below_base': (ev < ev.median()).mean(),
            'expected_shortfall_10pct': ev[ev <= ev.quantile(0.10)].mean(),
            
            # Confidence intervals
            'ci_95_lower': ev.quantile(0.025),
            'ci_95_upper': ev.quantile(0.975),
            'ci_90_lower': ev.quantile(0.05),
            'ci_90_upper': ev.quantile(0.95)
        }
        
        return analysis
    
    def plot_distribution(self, save_path: str = None):
        """Plot distribution of valuations"""
        
        if self.results is None:
            raise ValueError("Must run simulation first")
        
        ev = self.results['enterprise_value'] / 1_000_000_000  # Convert to billions
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Histogram
        ax = axes[0, 0]
        ax.hist(ev, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(ev.median(), color='red', linestyle='--', linewidth=2, label=f'Median: \${ev.median():.2f}B')
        ax.axvline(ev.quantile(0.10), color='orange', linestyle='--', label=f'P10: \${ev.quantile(0.10):.2f}B')
        ax.axvline(ev.quantile(0.90), color='green', linestyle='--', label=f'P90: \${ev.quantile(0.90):.2f}B')
ax.set_xlabel('Enterprise Value ($ Billions)')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Enterprise Value')
ax.legend()
ax.grid(alpha = 0.3)
        
        # Cumulative distribution
ax = axes[0, 1]
sorted_ev = np.sort(ev)
cumulative_prob = np.arange(1, len(sorted_ev) + 1) / len(sorted_ev)
ax.plot(sorted_ev, cumulative_prob, linewidth = 2)
ax.axhline(0.50, color = 'red', linestyle = '--', alpha = 0.5, label = 'Median')
ax.axhline(0.10, color = 'orange', linestyle = '--', alpha = 0.5, label = 'P10')
ax.axhline(0.90, color = 'green', linestyle = '--', alpha = 0.5, label = 'P90')
ax.set_xlabel('Enterprise Value ($ Billions)')
ax.set_ylabel('Cumulative Probability')
ax.set_title('Cumulative Distribution Function')
ax.legend()
ax.grid(alpha = 0.3)
        
        # Box plot
ax = axes[1, 0]
ax.boxplot(ev, vert = False)
ax.set_xlabel('Enterprise Value ($ Billions)')
ax.set_title('Box Plot of Valuation Distribution')
ax.grid(alpha = 0.3)
        
        # Sensitivity scatter
ax = axes[1, 1]
ax.scatter(
    self.results['wacc'] * 100,
    ev,
    alpha = 0.3,
    s = 10
)
ax.set_xlabel('WACC (%)')
ax.set_ylabel('Enterprise Value ($ Billions)')
ax.set_title('EV vs WACC (scatter)')
ax.grid(alpha = 0.3)

plt.tight_layout()

if save_path:
    plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
print(f"Plot saved to {save_path}")
        else:
plt.show()

# Example: Run Monte Carlo Valuation
mc_model = MonteCarloValuation(
    base_revenue = 1_000_000_000,  # $1B
    projection_years = 10,
    n_simulations = 10000
)

print("Running Monte Carlo simulation (10,000 iterations)...")
results_df = mc_model.run_simulation()

print("\\nAnalyzing results...")
analysis = mc_model.analyze_results()

print("\\n" + "=" * 70)
print("MONTE CARLO DCF ANALYSIS")
print("=" * 70)

print("\\nCENTRAL TENDENCY:")
print(f"  Mean:                      ${analysis['mean'] / 1e9:> 10,.2f}B")
print(f"  Median (P50):              ${analysis['median']/1e9:>10,.2f}B")
print(f"  Standard Deviation:        ${analysis['std']/1e9:>10,.2f}B")
print(f"  Coefficient of Variation:  {analysis['coefficient_of_variation']:>10.1%}")

print("\\nPERCENTILES:")
print(f"  P05 (Very Pessimistic):    ${analysis['p05']/1e9:>10,.2f}B")
print(f"  P10 (Pessimistic):         ${analysis['p10']/1e9:>10,.2f}B")
print(f"  P25 (Below Average):       ${analysis['p25']/1e9:>10,.2f}B")
print(f"  P50 (Median):              ${analysis['median']/1e9:>10,.2f}B")
print(f"  P75 (Above Average):       ${analysis['p75']/1e9:>10,.2f}B")
print(f"  P90 (Optimistic):          ${analysis['p90']/1e9:>10,.2f}B")
print(f"  P95 (Very Optimistic):     ${analysis['p95']/1e9:>10,.2f}B")

print("\\nRANGE:")
print(f"  Minimum:                   ${analysis['min']/1e9:>10,.2f}B")
print(f"  Maximum:                   ${analysis['max']/1e9:>10,.2f}B")
print(f"  Total Range:               ${analysis['range']/1e9:>10,.2f}B")
print(f"  Interquartile Range (IQR): ${analysis['iqr']/1e9:>10,.2f}B")

print("\\nCONFIDENCE INTERVALS:")
print(f"  90% CI:                    ${analysis['ci_90_lower']/1e9:>10,.2f}B - ${analysis['ci_90_upper']/1e9:>10,.2f}B")
print(f"  95% CI:                    ${analysis['ci_95_lower']/1e9:>10,.2f}B - ${analysis['ci_95_upper']/1e9:>10,.2f}B")

print("\\nRISK METRICS:")
print(f"  Expected Shortfall (P10):  ${analysis['expected_shortfall_10pct']/1e9:>10,.2f}B")
print(f"  Probability of Loss:       {analysis['probability_below_base']:>10.1%}")

print("\\nINTERPRETATION:")
print(f"  There is a 90% probability the company is worth between")
print(f"  ${analysis['p05']/1e9:.2f}B and ${analysis['p95']/1e9:.2f}B.")
print(f"  ")
print(f"  There is a 50% probability the company is worth between")
print(f"  ${analysis['p25']/1e9:.2f}B and ${analysis['p75']/1e9:.2f}B (IQR).")
print(f"  ")
print(f"  The expected value (mean) is ${analysis['mean']/1e9:.2f}B")
print(f"  with a standard deviation of ${analysis['std']/1e9:.2f}B.")

# Plot distribution
mc_model.plot_distribution(save_path = 'monte_carlo_valuation_distribution.png')
\`\`\`

---

## Modeling Correlations

**Problem**: Variables aren't independent. Revenue growth and margins often correlate positively (growing companies have operating leverage).

**Solution**: Use correlated random variables.

\`\`\`python
"""
Monte Carlo with Correlated Variables
"""

def generate_correlated_samples(
    n_samples: int,
    mean_vector: np.ndarray,
    correlation_matrix: np.ndarray
) -> np.ndarray:
    """
    Generate correlated random samples.
    
    Args:
        n_samples: Number of samples
        mean_vector: Mean of each variable
        correlation_matrix: Correlation matrix (symmetric, positive definite)
    
    Returns:
        Array of shape (n_samples, n_variables)
    """
    
    # Convert correlation to covariance (assume std=1 for simplicity, scale later)
    cov_matrix = correlation_matrix
    
    # Generate correlated normal samples
    samples = np.random.multivariate_normal(
        mean=mean_vector,
        cov=cov_matrix,
        size=n_samples
    )
    
    return samples

# Example: Revenue growth and EBITDA margin correlated
n_sims = 10000
mean_vector = np.array([0.12, 0.25])  # 12% growth, 25% margin

# Correlation matrix
# [[1.0, 0.6],   # Revenue growth correlated 0.6 with margin
#  [0.6, 1.0]]   # (positive correlation—high growth = expanding margins)
correlation_matrix = np.array([
    [0.05**2, 0.6 * 0.05 * 0.03],  # Covariance matrix (from corr and std devs)
    [0.6 * 0.05 * 0.03, 0.03**2]
])

correlated_samples = generate_correlated_samples(
    n_samples=n_sims,
    mean_vector=mean_vector,
    correlation_matrix=correlation_matrix
)

growth_samples = correlated_samples[:, 0]
margin_samples = correlated_samples[:, 1]

# Verify correlation
actual_correlation = np.corrcoef(growth_samples, margin_samples)[0, 1]
print(f"\\nCorrelation between revenue growth and EBITDA margin: {actual_correlation:.2f}")
print(f"(Target was 0.60)")
\`\`\`

---

## Key Takeaways

### When to Use Monte Carlo

✅ **Multiple uncertain variables**: Can't capture with 3 scenarios
✅ **Risk quantification**: Need downside probability (VaR, expected shortfall)
✅ **Regulatory requirements**: Some valuations require probabilistic approach
✅ **Portfolio decisions**: Comparing risk-adjusted returns across investments
✅ **Option pricing**: Valuing flexibility, abandonment, growth options

### When NOT to Use Monte Carlo

❌ **Deterministic assumptions**: If growth is 10% with certainty, don't simulate
❌ **Presentation to non-technical**: Hard to explain to boards/clients unfamiliar with stats
❌ **Garbage in, garbage out**: If distributions wrong, output is meaningless
❌ **Overfitting**: Don't model 20 variables with 10 data points each

### Advantages

- Captures full distribution of outcomes
- Quantifies probability of extreme scenarios
- Models complex interactions between variables
- Provides confidence intervals

### Disadvantages

- Requires assumptions about distributions (often unknown)
- Computationally intensive
- Can give false sense of precision ("We ran 10,000 simulations!")
- Difficult to communicate to non-technical audiences

### Best Practices

1. **Start simple**: 3-5 key uncertain variables
2. **Validate distributions**: Use historical data where possible
3. **Model correlations**: Don't assume independence
4. **Run enough simulations**: 10,000+ for stable results
5. **Sensitivity check**: Which variables drive uncertainty?
6. **Present clearly**: Show P10-P90 range, not just mean
7. **Supplement deterministic**: Use Monte Carlo alongside base/bear/bull scenarios

---

## Next Steps

With Monte Carlo mastered, you're ready for:

- **Real Options Valuation** (Section 10): Valuing flexibility and strategic options
- **Dividend Discount Model** (Section 11): Equity valuation for mature companies
- **Automated Model Generation** (Section 13): Building valuation platforms

**Practice**: Take your DCF model, identify 5 uncertain variables, define distributions, run 10,000 simulations. Present P10/P50/P90 valuation range with confidence intervals.

Monte Carlo transforms valuation from "the company is worth $10B" to "there's a 70% probability the company is worth $9-11B, with 10% downside risk of <$7B".

---

**Next Section**: [Real Options Valuation](./real-options-valuation) →
`,
};
