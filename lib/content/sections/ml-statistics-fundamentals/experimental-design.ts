/**
 * Experimental Design Section
 */

export const experimentaldesignSection = {
  id: 'experimental-design',
  title: 'Experimental Design',
  content: `# Experimental Design

## Introduction

Experimental design is how we plan studies to draw valid causal conclusions. Critical for:
- A/B testing ML models
- Feature experiments
- Treatment effect estimation
- Causal inference

**Key principles**: Randomization, Control, Replication

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# A/B Test Example
n_per_group = 1000

# Control group (Model A)
control = np.random.binomial(1, 0.75, n_per_group)  # 75% accuracy

# Treatment group (Model B)  
treatment = np.random.binomial(1, 0.78, n_per_group)  # 78% accuracy

print("=== A/B Test ===")
print(f"Control (A): {control.mean():.3f}")
print(f"Treatment (B): {treatment.mean():.3f}")
print(f"Difference: {(treatment.mean() - control.mean()):.3f}")

# Statistical test
from statsmodels.stats.proportion import proportions_ztest
z_stat, p_value = proportions_ztest([treatment.sum(), control.sum()], 
                                     [n_per_group, n_per_group], 
                                     alternative='larger')

print(f"\\nZ-statistic: {z_stat:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Result: {'Significant' if p_value < 0.05 else 'Not significant'}")
\`\`\`

## Sample Size Determination

\`\`\`python
from statsmodels.stats.power import zt_ind_solve_power

# Calculate required sample size
effect_size = 0.03  # 3% improvement
alpha = 0.05
power = 0.80
baseline = 0.75

# Convert to Cohen\'s h for proportions
from statsmodels.stats.proportion import proportion_effectsize
h = proportion_effectsize (baseline, baseline + effect_size)

n = zt_ind_solve_power (h, alpha=alpha, power=power, alternative='larger')

print(f"\\n=== Sample Size Calculation ===")
print(f"To detect {effect_size:.1%} improvement:")
print(f"Need n ≈ {n:.0f} per group")
print(f"Total: {2*n:.0f} observations")
\`\`\`

## Randomization

\`\`\`python
# Proper randomization
users = np.arange(1000)
np.random.shuffle (users)

group_a = users[:500]
group_b = users[500:]

print("\\n=== Randomization ===")
print("✓ Ensures groups are comparable")
print("✓ Controls for confounders")
print("✓ Enables causal inference")
\`\`\`

## Key Takeaways

1. **Randomization eliminates confounders**2. **Calculate sample size before experiment**3. **Power analysis prevents wasted effort**4. **A/B testing = proper experimental design**5. **Causal inference requires experiments**

Proper experimental design is essential for reliable ML experiments!
`,
};
