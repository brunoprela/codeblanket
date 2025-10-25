/**
 * Robust Statistics Section
 */

export const robuststatisticsSection = {
  id: 'robust-statistics',
  title: 'Robust Statistics',
  content: `# Robust Statistics

## Introduction

Robust statistics provides methods that perform well even with:
- Outliers
- Violations of assumptions
- Heavy-tailed distributions
- Contaminated data

Critical for **real-world ML** where data is messy!

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import HuberRegressor, RANSACRegressor
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest

np.random.seed(42)

# Data with outliers
n = 50
X = np.random.randn (n, 1) * 2
y = 2 + 3*X[:, 0] + np.random.randn (n) * 0.5

# Add outliers
outlier_idx = [5, 15, 25]
y[outlier_idx] += np.random.choice([-10, 10], len (outlier_idx))

# Compare estimators
from sklearn.linear_model import LinearRegression

ols = LinearRegression().fit(X, y)
huber = HuberRegressor().fit(X, y)
ransac = RANSACRegressor().fit(X, y)

plt.figure (figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X, y, alpha=0.6)
plt.scatter(X[outlier_idx], y[outlier_idx], c='r', s=100, label='Outliers')
x_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
plt.plot (x_plot, ols.predict (x_plot), 'b-', linewidth=2, label='OLS')
plt.xlabel('X')
plt.ylabel('y')
plt.title('OLS (Affected by Outliers)')
plt.legend()

plt.subplot(1, 3, 2)
plt.scatter(X, y, alpha=0.6)
plt.scatter(X[outlier_idx], y[outlier_idx], c='r', s=100, label='Outliers')
plt.plot (x_plot, huber.predict (x_plot), 'g-', linewidth=2, label='Huber')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Huber Regression (Robust)')
plt.legend()

plt.subplot(1, 3, 3)
plt.scatter(X, y, alpha=0.6)
plt.scatter(X[outlier_idx], y[outlier_idx], c='r', s=100, label='Outliers')
plt.plot (x_plot, ransac.predict (x_plot), 'purple', linewidth=2, label='RANSAC')
plt.xlabel('X')
plt.ylabel('y')
plt.title('RANSAC (Very Robust)')
plt.legend()

plt.tight_layout()
plt.savefig('robust_regression.png', dpi=300, bbox_inches='tight')
plt.show()

print("=== Regression Coefficients ===")
print(f"True: β₀=2, β₁=3")
print(f"OLS: β₀={ols.intercept_:.2f}, β₁={ols.coef_[0]:.2f} (biased!)")
print(f"Huber: β₀={huber.intercept_:.2f}, β₁={huber.coef_[0]:.2f}")
print(f"RANSAC: β₀={ransac.estimator_.intercept_:.2f}, β₁={ransac.estimator_.coef_[0]:.2f}")
\`\`\`

## Robust Central Tendency

\`\`\`python
# Mean vs Median
data = np.array([1, 2, 3, 4, 5, 100])  # One outlier

print("\\n=== Central Tendency with Outlier ===")
print(f"Mean: {np.mean (data):.2f} (sensitive)")
print(f"Median: {np.median (data):.2f} (robust)")
print(f"Trimmed mean (10%): {stats.trim_mean (data, 0.1):.2f}")

# Winsorization
from scipy.stats.mstats import winsorize
winsorized = winsorize (data, limits=[0.1, 0.1])
print(f"Winsorized mean: {np.mean (winsorized):.2f}")
\`\`\`

## Outlier Detection

\`\`\`python
# Multiple methods
X = np.random.randn(100, 2)
X = np.vstack([X, [[5, 5], [-5, -5]]])  # Add outliers

# Method 1: Z-score
z_scores = np.abs (stats.zscore(X, axis=0))
outliers_zscore = np.any (z_scores > 3, axis=1)

# Method 2: IQR
Q1 = np.percentile(X, 25, axis=0)
Q3 = np.percentile(X, 75, axis=0)
IQR = Q3 - Q1
outliers_iqr = np.any((X < Q1 - 1.5*IQR) | (X > Q3 + 1.5*IQR), axis=1)

# Method 3: Isolation Forest
iso_forest = IsolationForest (contamination=0.1, random_state=42)
outliers_iso = iso_forest.fit_predict(X) == -1

# Method 4: Elliptic Envelope (assumes Gaussian)
elliptic = EllipticEnvelope (contamination=0.1, random_state=42)
outliers_elliptic = elliptic.fit_predict(X) == -1

print("\\n=== Outlier Detection Methods ===")
print(f"Z-score (>3σ): {outliers_zscore.sum()} outliers")
print(f"IQR method: {outliers_iqr.sum()} outliers")
print(f"Isolation Forest: {outliers_iso.sum()} outliers")
print(f"Elliptic Envelope: {outliers_elliptic.sum()} outliers")

# Visualize
plt.figure (figsize=(15, 4))

plt.subplot(1, 4, 1)
plt.scatter(X[~outliers_zscore, 0], X[~outliers_zscore, 1], alpha=0.6)
plt.scatter(X[outliers_zscore, 0], X[outliers_zscore, 1], c='r', s=100)
plt.title('Z-score')

plt.subplot(1, 4, 2)
plt.scatter(X[~outliers_iqr, 0], X[~outliers_iqr, 1], alpha=0.6)
plt.scatter(X[outliers_iqr, 0], X[outliers_iqr, 1], c='r', s=100)
plt.title('IQR')

plt.subplot(1, 4, 3)
plt.scatter(X[~outliers_iso, 0], X[~outliers_iso, 1], alpha=0.6)
plt.scatter(X[outliers_iso, 0], X[outliers_iso, 1], c='r', s=100)
plt.title('Isolation Forest')

plt.subplot(1, 4, 4)
plt.scatter(X[~outliers_elliptic, 0], X[~outliers_elliptic, 1], alpha=0.6)
plt.scatter(X[outliers_elliptic, 0], X[outliers_elliptic, 1], c='r', s=100)
plt.title('Elliptic Envelope')

plt.tight_layout()
plt.savefig('outlier_detection.png', dpi=300, bbox_inches='tight')
plt.show()
\`\`\`

## Financial Data Application

\`\`\`python
# Financial returns often have heavy tails and outliers
returns = np.random.standard_t(5, 1000) * 0.02  # Heavy-tailed

print("\\n=== Financial Returns Statistics ===")
print(f"Mean: {np.mean (returns):.6f}")
print(f"Median: {np.median (returns):.6f}")
print(f"Std: {np.std (returns):.6f}")
print(f"MAD (robust): {stats.median_abs_deviation (returns):.6f}")
print(f"Skewness: {stats.skew (returns):.4f}")
print(f"Kurtosis: {stats.kurtosis (returns):.4f} (excess, heavy tails!)")

# Robust covariance
from sklearn.covariance import EmpiricalCovariance, MinCovDet

X_returns = np.random.standard_t(5, (200, 3)) * 0.02

emp_cov = EmpiricalCovariance().fit(X_returns)
robust_cov = MinCovDet().fit(X_returns)

print("\\n=== Covariance Estimation ===")
print("Standard covariance:")
print(emp_cov.covariance_)
print("\\nRobust covariance (MCD):")
print(robust_cov.covariance_)
\`\`\`

## Key Takeaways

1. **Robust methods handle outliers gracefully**
2. **Median > Mean for heavy-tailed data**
3. **Huber/RANSAC regression for outlier resistance**
4. **Multiple outlier detection methods exist**
5. **Critical for financial and real-world data**

Robust statistics is essential for production ML systems!
`,
};
