/**
 * Bayesian Statistics Section
 */

export const bayesianstatisticsSection = {
  id: 'bayesian-statistics',
  title: 'Bayesian Statistics',
  content: `# Bayesian Statistics

## Introduction

Bayesian statistics treats parameters as random variables with probability distributions, incorporating:
- **Prior knowledge** before seeing data
- **Data** (likelihood)  
- **Posterior** probability after seeing data

**Bayes' Theorem**:
\\[ P(\\theta|data) = \\frac{P(data|\\theta) \\cdot P(\\theta)}{P(data)} \\]

Posterior = (Likelihood × Prior) / Evidence

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# Example: Estimating coin bias
# Prior: Beta(2, 2) - slightly fair coin
# Data: 7 heads out of 10 flips
# Posterior: Beta(2+7, 2+3) = Beta(9, 5)

theta = np.linspace(0, 1, 100)
prior = stats.beta(2, 2).pdf (theta)
likelihood = stats.binom(10, theta).pmf(7)
posterior = stats.beta(9, 5).pdf (theta)

plt.figure (figsize=(10, 6))
plt.plot (theta, prior, label='Prior: Beta(2,2)', linewidth=2)
plt.plot (theta, likelihood/likelihood.max()*prior.max(), label='Likelihood (scaled)', linewidth=2)
plt.plot (theta, posterior, label='Posterior: Beta(9,5)', linewidth=2)
plt.axvline(7/10, color='r', linestyle='--', label='MLE: 0.7')
plt.axvline(9/14, color='g', linestyle='--', label='Posterior mean: 0.64')
plt.xlabel('θ (coin bias)')
plt.ylabel('Density')
plt.title('Bayesian Updating')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('bayesian_updating.png', dpi=300, bbox_inches='tight')
plt.show()

print("=== Bayesian vs Frequentist ===")
print(f"MLE estimate: {7/10:.3f}")
print(f"Bayesian (posterior mean): {9/14:.3f}")
print("Bayesian incorporates prior belief!")
\`\`\`

## Credible Intervals vs Confidence Intervals

\`\`\`python
# 95% Credible Interval (Bayesian)
credible = stats.beta(9, 5).interval(0.95)
print(f"\\n95% Credible Interval: [{credible[0]:.3f}, {credible[1]:.3f}]")
print("Interpretation: 95% probability θ is in this interval")

# Compare to frequentist CI
# (requires different calculation, not shown)
print("\\nFrequentist CI: Procedure captures true θ 95% of times")
print("Bayesian: Direct probability statement about θ!")
\`\`\`

## Bayesian Linear Regression

\`\`\`python
# Simple Bayesian regression
X = np.random.randn(50, 1)
y = 2 + 3*X[:, 0] + np.random.normal(0, 1, 50)

# Prior: β ~ N(0, 10²) - weak prior
# Likelihood: y|X,β ~ N(Xβ, σ²)
# Posterior: β|X,y ~ N(posterior_mean, posterior_cov)

# Simplified calculation (assuming known σ=1)
prior_mean = 0
prior_var = 100
sigma = 1

X_aug = np.column_stack([np.ones (len(X)), X])
posterior_cov = np.linalg.inv(X_aug.T @ X_aug / sigma**2 + np.eye(2) / prior_var)
posterior_mean = posterior_cov @ (X_aug.T @ y / sigma**2)

print("\\n=== Bayesian Regression ===")
print(f"Posterior mean: {posterior_mean}")
print(f"True values: [2, 3]")
print("With more data, posterior → MLE")
\`\`\`

## Key Takeaways

1. **Bayes: P(θ|data) ∝ P(data|θ) × P(θ)**2. **Prior → Data → Posterior**3. **Credible intervals**: Direct probability statements
4. **Incorporates prior knowledge**5. **Foundation for Bayesian ML**

Bayesian methods are powerful for incorporating domain knowledge and quantifying uncertainty!
`,
};
