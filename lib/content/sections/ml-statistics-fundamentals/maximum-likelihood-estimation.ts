/**
 * Maximum Likelihood Estimation Section
 */

export const maximumlikelihoodestimationSection = {
  id: 'maximum-likelihood-estimation',
  title: 'Maximum Likelihood Estimation',
  content: `# Maximum Likelihood Estimation (MLE)

## Introduction

MLE is a fundamental method for estimating parameters that **maximize the likelihood of observing the data**. It's the foundation of:
- Linear regression (equivalent to OLS under normality)
- Logistic regression
- Neural networks (cross-entropy loss)
- Many statistical models

**Key idea**: Find parameters that make the observed data most probable.

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize

np.random.seed(42)

# Simple example: Estimating mean of normal distribution
data = np.random.normal(5, 2, 100)

# Likelihood function
def likelihood(mu, data, sigma=2):
    return np.prod(stats.norm.pdf(data, mu, sigma))

# Log-likelihood (more stable)
def log_likelihood(mu, data, sigma=2):
    return np.sum(stats.norm.logpdf(data, mu, sigma))

# Test different mu values
mus = np.linspace(3, 7, 100)
lls = [log_likelihood(mu, data) for mu in mus]

plt.figure(figsize=(10, 5))
plt.plot(mus, lls, linewidth=2)
plt.axvline(data.mean(), color='r', linestyle='--', label=f'MLE: {data.mean():.3f}')
plt.xlabel('μ (parameter)')
plt.ylabel('Log-Likelihood')
plt.title('Log-Likelihood Function')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('mle_likelihood.png', dpi=300, bbox_inches='tight')
plt.show()

print("=== MLE Example ===")
print(f"Sample mean (MLE): {data.mean():.4f}")
print(f"True mean: 5.0000")
\`\`\`

## MLE for Regression

\`\`\`python
# Linear regression as MLE
X = np.random.randn(100, 2)
beta_true = np.array([1, 2, -1])  # intercept + 2 slopes
y = beta_true[0] + X @ beta_true[1:] + np.random.normal(0, 1, 100)

def negative_log_likelihood(params, X, y):
    beta = params[:-1]
    sigma = params[-1]
    y_pred = beta[0] + X @ beta[1:]
    residuals = y - y_pred
    return -np.sum(stats.norm.logpdf(residuals, 0, sigma))

# Optimize
X_with_const = np.column_stack([np.ones(len(X)), X])
initial = np.zeros(4)  # 3 betas + 1 sigma
result = minimize(negative_log_likelihood, initial, args=(X_with_const, y))

print("\\n=== Regression via MLE ===")
print(f"MLE estimates: {result.x[:-1]}")
print(f"True values: {beta_true}")
print(f"Estimated σ: {result.x[-1]:.4f}")
\`\`\`

## Connection to Neural Networks

Cross-entropy loss = Negative log-likelihood!

\`\`\`python
# Binary classification
def cross_entropy_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# This is exactly negative log-likelihood for Bernoulli distribution!
print("\\n=== Neural Network Connection ===")
print("Cross-entropy = Negative log-likelihood")
print("→ Training neural networks = Maximum likelihood estimation!")
\`\`\`

## Key Takeaways

1. **MLE finds parameters that maximize P(data|parameters)**
2. **Log-likelihood is more stable than likelihood**
3. **OLS = MLE under normality assumption**
4. **Cross-entropy = Negative log-likelihood**
5. **Foundation of most statistical models**

MLE connects classical statistics to modern machine learning!
`,
};
