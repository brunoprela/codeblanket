/**
 * Linear Regression Section
 */

export const linearregressionSection = {
  id: 'linear-regression',
  title: 'Linear Regression',
  content: `# Linear Regression

## Introduction

Linear regression is one of the most fundamental and widely-used machine learning algorithms. Despite its simplicity, it's incredibly powerful and forms the foundation for understanding more complex models. Linear regression models the relationship between input features and a continuous target variable using a linear function.

**Applications**:
- Predicting house prices based on size, location, and features
- Forecasting sales based on advertising spend
- Estimating crop yields from weather and soil data
- Predicting stock returns from market factors
- Modeling insurance claims from customer characteristics

**Why Start with Linear Regression?**:
- Mathematically well-understood
- Fast to train, even on large datasets
- Interpretable results
- Serves as baseline for more complex models
- Foundation for understanding regularization and optimization

## Simple Linear Regression

### The Model

Simple linear regression models the relationship between one input feature (x) and one output (y) using a straight line:

\\[ y = \\beta_0 + \\beta_1 x + \\epsilon \\]

Where:
- \\( y \\): Target variable (what we want to predict)
- \\( x \\): Input feature (predictor)
- \\( \\beta_0 \\): Intercept (value of y when x = 0)
- \\( \\beta_1 \\): Slope (change in y for unit change in x)
- \\( \\epsilon \\): Error term (noise)

**Goal**: Find the best values of \\( \\beta_0 \\) and \\( \\beta_1 \\) that minimize prediction errors.

### Geometric Interpretation

The regression line represents the "best fit" through the data points. For each data point, the vertical distance from the point to the line is the residual (error). We want to minimize these residuals.

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.linspace(0, 10, 50)
y_true = 2 + 3 * X  # True relationship: y = 2 + 3x
y = y_true + np.random.randn(50) * 2  # Add noise

# Fit linear regression manually
X_mean = np.mean(X)
y_mean = np.mean (y)

# Calculate slope (beta_1) and intercept (beta_0)
numerator = np.sum((X - X_mean) * (y - y_mean))
denominator = np.sum((X - X_mean)**2)
beta_1 = numerator / denominator
beta_0 = y_mean - beta_1 * X_mean

print(f"Intercept (β₀): {beta_0:.3f}")
print(f"Slope (β₁): {beta_1:.3f}")
print(f"\\nTrue values: β₀ = 2, β₁ = 3")

# Make predictions
y_pred = beta_0 + beta_1 * X

# Visualize
plt.figure (figsize=(10, 6))
plt.scatter(X, y, alpha=0.5, label='Data points')
plt.plot(X, y_pred, 'r-', linewidth=2, label=f'Fitted line: y = {beta_0:.2f} + {beta_1:.2f}x')
plt.plot(X, y_true, 'g--', linewidth=2, label='True relationship', alpha=0.7)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Simple Linear Regression')
plt.grid(True, alpha=0.3)
plt.show()

# Calculate residuals
residuals = y - y_pred
print(f"\\nMean residual: {np.mean (residuals):.6f} (should be ~0)")
print(f"Standard deviation of residuals: {np.std (residuals):.3f}")
\`\`\`

## Ordinary Least Squares (OLS)

### The Cost Function

We find the best parameters by minimizing the **Mean Squared Error (MSE)**:

\\[ MSE = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2 = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - (\\beta_0 + \\beta_1 x_i))^2 \\]

Where:
- \\( n \\): Number of samples
- \\( y_i \\): Actual value
- \\( \\hat{y}_i \\): Predicted value

**Why squared errors?**:
- Penalizes large errors more than small errors
- Mathematically convenient (differentiable)
- Has statistical justification (maximum likelihood under Gaussian noise)

### Analytical Solution

For simple linear regression, we can derive the optimal parameters analytically:

\\[ \\beta_1 = \\frac{\\sum_{i=1}^{n}(x_i - \\bar{x})(y_i - \\bar{y})}{\\sum_{i=1}^{n}(x_i - \\bar{x})^2} = \\frac{Cov (x, y)}{Var (x)} \\]

\\[ \\beta_0 = \\bar{y} - \\beta_1 \\bar{x} \\]

\`\`\`python
def calculate_mse (y_true, y_pred):
    """Calculate Mean Squared Error"""
    return np.mean((y_true - y_pred)**2)

def fit_simple_linear_regression(X, y):
    """Fit simple linear regression using OLS formulas"""
    X_mean = np.mean(X)
    y_mean = np.mean (y)
    
    # Calculate covariance and variance
    cov_xy = np.mean((X - X_mean) * (y - y_mean))
    var_x = np.mean((X - X_mean)**2)
    
    # Calculate parameters
    beta_1 = cov_xy / var_x
    beta_0 = y_mean - beta_1 * X_mean
    
    return beta_0, beta_1

def predict(X, beta_0, beta_1):
    """Make predictions"""
    return beta_0 + beta_1 * X

# Example usage
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

beta_0, beta_1 = fit_simple_linear_regression(X, y)
y_pred = predict(X, beta_0, beta_1)
mse = calculate_mse (y, y_pred)

print(f"β₀ = {beta_0:.3f}, β₁ = {beta_1:.3f}")
print(f"MSE = {mse:.3f}")
print(f"\\nPredictions: {y_pred}")
print(f"Actual values: {y}")
\`\`\`

## Multiple Linear Regression

### The Model

Multiple linear regression extends simple regression to multiple input features:

\\[ y = \\beta_0 + \\beta_1 x_1 + \\beta_2 x_2 + ... + \\beta_p x_p + \\epsilon \\]

Or in matrix notation:

\\[ \\mathbf{y} = \\mathbf{X}\\boldsymbol{\\beta} + \\boldsymbol{\\epsilon} \\]

Where:
- \\( \\mathbf{y} \\): Vector of target values (n × 1)
- \\( \\mathbf{X} \\): Design matrix (n × (p+1)), includes column of 1s for intercept
- \\( \\boldsymbol{\\beta} \\): Parameter vector (p+1 × 1)
- \\( \\boldsymbol{\\epsilon} \\): Error vector (n × 1)

### Matrix Solution

The OLS solution in matrix form:

\\[ \\hat{\\boldsymbol{\\beta}} = (\\mathbf{X}^T \\mathbf{X})^{-1} \\mathbf{X}^T \\mathbf{y} \\]

This is called the **Normal Equation**.

**Computational Note**: Direct matrix inversion is expensive for large datasets. In practice, we use:
- QR decomposition
- Singular Value Decomposition (SVD)  
- Gradient descent for very large datasets

\`\`\`python
def fit_multiple_linear_regression(X, y):
    """
    Fit multiple linear regression using normal equation
    
    Parameters:
    X: array of shape (n_samples, n_features)
    y: array of shape (n_samples,)
    
    Returns:
    beta: array of coefficients including intercept
    """
    # Add column of ones for intercept
    X_with_intercept = np.column_stack([np.ones (len(X)), X])
    
    # Normal equation: β = (X^T X)^(-1) X^T y
    XtX = X_with_intercept.T @ X_with_intercept
    Xty = X_with_intercept.T @ y
    beta = np.linalg.solve(XtX, Xty)  # More stable than direct inverse
    
    return beta

# Example: House price prediction
np.random.seed(42)
n_samples = 100

# Generate features: size, bedrooms, age
size = np.random.uniform(1000, 3000, n_samples)
bedrooms = np.random.randint(1, 6, n_samples)
age = np.random.uniform(0, 50, n_samples)

# Generate target: price (with some relationship to features)
price = 50000 + 100 * size + 20000 * bedrooms - 500 * age + np.random.randn (n_samples) * 10000

# Prepare data
X = np.column_stack([size, bedrooms, age])
y = price

# Fit model
beta = fit_multiple_linear_regression(X, y)

print("Fitted coefficients:")
print(f"Intercept: \${beta[0]:,.0f}")
print(f"Size coefficient: \${beta[1]:.2f} per sq ft")
print(f"Bedrooms coefficient: \${beta[2]:,.0f} per bedroom")
print(f"Age coefficient: \${beta[3]:.2f} per year")

# Make predictions
X_with_intercept = np.column_stack([np.ones (len(X)), X])
y_pred = X_with_intercept @beta

# Evaluate
mse = np.mean((y - y_pred) ** 2)
rmse = np.sqrt (mse)
print(f"\\nRMSE: \${rmse:,.0f}")
\`\`\`

## Using Scikit-Learn

In practice, we use scikit-learn's optimized implementation:

\`\`\`python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd

# Create DataFrame for better visualization
data = pd.DataFrame({
    'size': size,
    'bedrooms': bedrooms,
    'age': age,
    'price': price
})

# Split features and target
X = data[['size', 'bedrooms', 'age']]
y = data['price']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Model parameters
print("Model Parameters:")
print(f"Intercept: \${model.intercept_:,.0f}")
print("\\nCoefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature}: {coef:,.2f}")

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluate
print("\\nModel Performance:")
print(f"Training R²: {r2_score (y_train, y_train_pred):.4f}")
print(f"Test R²: {r2_score (y_test, y_test_pred):.4f}")
print(f"\\nTraining RMSE: \${np.sqrt (mean_squared_error (y_train, y_train_pred)):,.0f}")
print(f"Test RMSE: \${np.sqrt (mean_squared_error (y_test, y_test_pred)):,.0f}")
print(f"\\nTest MAE: \${mean_absolute_error (y_test, y_test_pred):,.0f}")
\`\`\`

## Assumptions of Linear Regression

For valid inference and optimal performance, linear regression assumes:

### 1. Linearity

The relationship between features and target is linear. Check using:
- Scatter plots of each feature vs. target
- Residual plots (should show random pattern)

\`\`\`python
# Check linearity
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, feature in enumerate(['size', 'bedrooms', 'age']):
    axes[idx].scatter (data[feature], data['price'], alpha=0.5)
    axes[idx].set_xlabel (feature)
    axes[idx].set_ylabel('price')
    axes[idx].set_title (f'{feature} vs price')
    
plt.tight_layout()
plt.show()
\`\`\`

### 2. Independence

Observations are independent. Violated when:
- Time series data (autocorrelation)
- Clustered/hierarchical data
- Repeated measurements

### 3. Homoscedasticity

Constant variance of errors across all levels of features.

Check using residual plot: residuals should have constant spread.

\`\`\`python
# Check homoscedasticity
residuals = y_test - y_test_pred

plt.figure (figsize=(10, 6))
plt.scatter (y_test_pred, residuals, alpha=0.5)
plt.axhline (y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# Should show random scatter around zero with constant spread
\`\`\`

### 4. Normality of Errors

Errors follow a normal distribution. Check using:
- Histogram of residuals
- Q-Q plot

\`\`\`python
from scipy import stats

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Histogram
axes[0].hist (residuals, bins=30, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Residuals')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Residuals')

# Q-Q plot
stats.probplot (residuals, dist="norm", plot=axes[1])
axes[1].set_title('Q-Q Plot')

plt.tight_layout()
plt.show()
\`\`\`

### 5. No Multicollinearity

Features should not be highly correlated with each other.

\`\`\`python
# Check multicollinearity using correlation matrix
correlation_matrix = data[['size', 'bedrooms', 'age']].corr()

print("Correlation Matrix:")
print(correlation_matrix)

# Variance Inflation Factor (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range (len(X.columns))]

print("\\nVariance Inflation Factors:")
print(vif_data)
# VIF > 10 indicates problematic multicollinearity
\`\`\`

## Evaluation Metrics

### R² Score (Coefficient of Determination)

\\[ R^2 = 1 - \\frac{\\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2}{\\sum_{i=1}^{n}(y_i - \\bar{y})^2} = 1 - \\frac{SS_{res}}{SS_{tot}} \\]

**Interpretation**:
- \\( R^2 = 1 \\): Perfect predictions
- \\( R^2 = 0 \\): Model no better than predicting mean
- \\( R^2 < 0 \\): Model worse than predicting mean

**Note**: R² always increases with more features, even if they're random!

### Adjusted R²

Penalizes for number of features:

\\[ R^2_{adj} = 1 - \\frac{(1-R^2)(n-1)}{n-p-1} \\]

Where p is the number of features.

### Mean Absolute Error (MAE)

\\[ MAE = \\frac{1}{n}\\sum_{i=1}^{n}|y_i - \\hat{y}_i| \\]

- Easy to interpret (average absolute error)
- Less sensitive to outliers than MSE

### Root Mean Squared Error (RMSE)

\\[ RMSE = \\sqrt{\\frac{1}{n}\\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2} \\]

- Same units as target variable
- Penalizes large errors more than MAE

\`\`\`python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_regression (y_true, y_pred, model_name="Model"):
    """Comprehensive regression evaluation"""
    mae = mean_absolute_error (y_true, y_pred)
    mse = mean_squared_error (y_true, y_pred)
    rmse = np.sqrt (mse)
    r2 = r2_score (y_true, y_pred)
    
    print(f"\\n{model_name} Performance:")
    print(f"  MAE:  {mae:,.2f}")
    print(f"  MSE:  {mse:,.2f}")
    print(f"  RMSE: {rmse:,.2f}")
    print(f"  R²:   {r2:.4f}")
    
    return {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2}

# Evaluate on train and test sets
train_metrics = evaluate_regression (y_train, y_train_pred, "Training Set")
test_metrics = evaluate_regression (y_test, y_test_pred, "Test Set")
\`\`\`

## Real-World Example: Stock Return Prediction

\`\`\`python
# Simulating a financial example: predicting stock returns
np.random.seed(42)
n_days = 252  # Trading days in a year

# Generate features
market_return = np.random.randn (n_days) * 0.01  # Daily market return
sector_return = np.random.randn (n_days) * 0.015  # Sector return
volume_ratio = np.random.uniform(0.5, 1.5, n_days)  # Volume relative to average
volatility = np.random.uniform(0.01, 0.03, n_days)  # Daily volatility

# Generate stock return (with realistic relationship)
stock_return = (
    0.0001 +  # Alpha (excess return)
    1.2 * market_return +  # Beta > 1 (more volatile than market)
    0.3 * sector_return +  # Sector exposure
    0.005 * volume_ratio +  # Volume effect
    np.random.randn (n_days) * 0.015  # Idiosyncratic noise
)

# Create DataFrame
stock_data = pd.DataFrame({
    'market_return': market_return,
    'sector_return': sector_return,
    'volume_ratio': volume_ratio,
    'volatility': volatility,
    'stock_return': stock_return
})

# Split data (time-series: use first 200 days for train, last 52 for test)
train_data = stock_data.iloc[:200]
test_data = stock_data.iloc[200:]

X_train = train_data[['market_return', 'sector_return', 'volume_ratio', 'volatility']]
y_train = train_data['stock_return']
X_test = test_data[['market_return', 'sector_return', 'volume_ratio', 'volatility']]
y_test = test_data['stock_return']

# Fit model
model = LinearRegression()
model.fit(X_train, y_train)

# Interpret coefficients
print("Factor Model Coefficients:")
print(f"Alpha (intercept): {model.intercept_:.6f}")
print(f"Beta (market sensitivity): {model.coef_[0]:.3f}")
print(f"Sector beta: {model.coef_[1]:.3f}")
print(f"Volume coefficient: {model.coef_[2]:.6f}")
print(f"Volatility coefficient: {model.coef_[3]:.6f}")

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluate
print(f"\\nIn-sample R²: {r2_score (y_train, y_pred_train):.4f}")
print(f"Out-of-sample R²: {r2_score (y_test, y_pred_test):.4f}")

# Trading strategy simulation
print("\\n--- Trading Strategy Performance ---")
# Simple strategy: go long when predicted return > 0, short otherwise
predicted_direction = np.sign (y_pred_test)
actual_direction = np.sign (y_test)
accuracy = np.mean (predicted_direction == actual_direction)
print(f"Direction prediction accuracy: {accuracy:.2%}")

# Calculate strategy returns
strategy_returns = predicted_direction * y_test.values
cumulative_return = np.cumprod(1 + strategy_returns) - 1
sharpe_ratio = np.mean (strategy_returns) / np.std (strategy_returns) * np.sqrt(252)

print(f"Cumulative return: {cumulative_return[-1]:.2%}")
print(f"Sharpe ratio: {sharpe_ratio:.2f}")
\`\`\`

## Gradient Descent Alternative

For very large datasets, matrix inversion becomes expensive. Gradient descent is an iterative alternative:

\`\`\`python
def gradient_descent_linear_regression(X, y, learning_rate=0.01, iterations=1000):
    """
    Fit linear regression using gradient descent
    """
    # Add intercept term
    X_with_intercept = np.column_stack([np.ones (len(X)), X])
    n_samples, n_features = X_with_intercept.shape
    
    # Initialize parameters
    beta = np.zeros (n_features)
    
    # Store history
    cost_history = []
    
    for i in range (iterations):
        # Predictions
        y_pred = X_with_intercept @ beta
        
        # Calculate gradients
        errors = y_pred - y
        gradients = (1/n_samples) * X_with_intercept.T @ errors
        
        # Update parameters
        beta = beta - learning_rate * gradients
        
        # Calculate cost
        cost = (1/(2*n_samples)) * np.sum (errors**2)
        cost_history.append (cost)
        
        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost:.4f}")
    
    return beta, cost_history

# Example
X_simple = np.random.randn(1000, 3)
y_simple = 2 + 3*X_simple[:, 0] + 1*X_simple[:, 1] - 2*X_simple[:, 2] + np.random.randn(1000)*0.5

beta_gd, cost_history = gradient_descent_linear_regression(X_simple, y_simple, learning_rate=0.1, iterations=1000)

print(f"\\nFinal parameters: {beta_gd}")
print(f"True parameters: [2, 3, 1, -2]")

# Plot cost convergence
plt.figure (figsize=(10, 6))
plt.plot (cost_history)
plt.xlabel('Iteration')
plt.ylabel('Cost (MSE)')
plt.title('Gradient Descent Convergence')
plt.grid(True, alpha=0.3)
plt.show()
\`\`\`

## Common Pitfalls

1. **Not scaling features**: Features with larger scales dominate
2. **Using linear regression for non-linear relationships**: Transform features or use non-linear models
3. **Ignoring outliers**: Can severely affect the fit
4. **Multicollinearity**: Makes coefficient interpretation unreliable
5. **Extrapolation**: Predictions outside training data range are unreliable
6. **Assuming causation from correlation**: Correlation doesn't imply causation

## Summary

Linear regression is a fundamental algorithm that:
- Models linear relationships between features and targets
- Uses ordinary least squares to minimize prediction error
- Has closed-form solution (normal equation) and iterative solution (gradient descent)
- Requires checking assumptions (linearity, independence, homoscedasticity, normality)
- Serves as baseline for more complex models
- Is interpretable and fast to train

**Key Takeaways**:
- Start with linear regression as a baseline
- Check assumptions using diagnostic plots
- Use appropriate metrics (R², RMSE, MAE)
- Be careful with multicollinearity and outliers
- Consider feature engineering for non-linear relationships

Next, we'll explore polynomial regression and how to handle non-linear relationships!
`,
  codeExample: `# Complete Linear Regression Implementation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Generate realistic dataset: House prices
np.random.seed(42)
n_samples = 500

# Features
size = np.random.uniform(800, 3500, n_samples)  # Square feet
bedrooms = np.random.randint(1, 6, n_samples)
bathrooms = bedrooms + np.random.choice([-1, 0, 1], n_samples, p=[0.2, 0.5, 0.3])
bathrooms = np.clip (bathrooms, 1, 5)
age = np.random.uniform(0, 50, n_samples)
lot_size = np.random.uniform(2000, 10000, n_samples)
distance_city = np.random.uniform(1, 30, n_samples)  # Miles from city center

# Generate price with realistic relationships
base_price = 100000
price = (
    base_price +
    150 * size +                    # Size is most important
    25000 * bedrooms +              # Bedrooms add value
    15000 * bathrooms +             # Bathrooms add value
    -1000 * age +                   # Older houses worth less
    10 * lot_size +                 # Lot size adds value
    -2000 * distance_city +         # Closer to city is better
    np.random.randn (n_samples) * 50000  # Noise
)

# Create DataFrame
df = pd.DataFrame({
    'size': size,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'age': age,
    'lot_size': lot_size,
    'distance_city': distance_city,
    'price': price
})

print("="*60)
print("LINEAR REGRESSION: HOUSE PRICE PREDICTION")
print("="*60)

print(f"\\nDataset shape: {df.shape}")
print(f"\\nFeatures: {list (df.columns[:-1])}")
print(f"Target: {df.columns[-1]}")

print(f"\\nPrice statistics:")
print(f"  Mean: \${df['price'].mean():,.0f}")
print(f"  Median: \${df['price'].median():,.0f}")
print(f"  Std: \${df['price'].std():,.0f}")
print(f"  Min: \${df['price'].min():,.0f}")
print(f"  Max: \${df['price'].max():,.0f}")

# Split features and target
X = df.drop('price', axis = 1)
y = df['price']

# Split into train and test sets(80 - 20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42
)

print(f"\\nTrain set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Feature scaling (important for gradient descent and regularization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
print("\\n" + "=" * 60)
print("TRAINING MODEL")
print("=" * 60)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("\\nModel trained successfully!")

# Display coefficients
print("\\nModel Parameters:")
print(f"Intercept: \${model.intercept_:,.2f}")
print("\\nCoefficients:")
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_,
    'Abs_Coefficient': np.abs (model.coef_)
}).sort_values('Abs_Coefficient', ascending = False)

for _, row in coef_df.iterrows():
    print(f"  {row['Feature']:15s}: {row['Coefficient']:12,.2f}")

# Make predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Evaluate model
print("\\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

def print_metrics (y_true, y_pred, dataset_name):
mae = mean_absolute_error (y_true, y_pred)
mse = mean_squared_error (y_true, y_pred)
rmse = np.sqrt (mse)
r2 = r2_score (y_true, y_pred)
mape = np.mean (np.abs((y_true - y_pred) / y_true)) * 100

print(f"\\n{dataset_name}:")
print(f"  MAE:  \${mae:,.2f}")
print(f"  RMSE: \${rmse:,.2f}")
print(f"  R²:   {r2:.4f}")
print(f"  MAPE: {mape:.2f}%")

print_metrics (y_train, y_train_pred, "Training Set")
print_metrics (y_test, y_test_pred, "Test Set")

# Check for overfitting / underfitting
train_r2 = r2_score (y_train, y_train_pred)
test_r2 = r2_score (y_test, y_test_pred)

print("\\n" + "=" * 60)
print("MODEL DIAGNOSIS")
print("=" * 60)

if train_r2 - test_r2 > 0.1:
    print("⚠️  Possible overfitting: Large gap between train and test R²")
elif train_r2 < 0.5 and test_r2 < 0.5:
print("⚠️  Possible underfitting: Both train and test R² are low")
else:
print("✓  Model shows good generalization")

# Example predictions
print("\\n" + "=" * 60)
print("EXAMPLE PREDICTIONS")
print("=" * 60)

# Select a few test samples
sample_indices = np.random.choice (len(X_test), 5, replace = False)

print("\\nComparison of Actual vs Predicted Prices:")
print(f"{'Actual':>15s} {'Predicted':>15s} {'Difference':>15s} {'% Error':>10s}")
print("-" * 60)

for idx in sample_indices:
    actual = y_test.iloc[idx]
predicted = y_test_pred[idx]
diff = predicted - actual
pct_error = (diff / actual) * 100

print(f"\${actual:>14,.0f} \${predicted:>14,.0f} \${diff:>14,.0f} {pct_error:>9.1f}%")

# Feature importance visualization
print("\\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

print("\\nFeatures ranked by absolute coefficient value:")
print("(Note: Features are scaled, so coefficients are directly comparable)")
print(coef_df[['Feature', 'Coefficient']].to_string (index = False))

# Residual analysis
residuals = y_test - y_test_pred

print("\\n" + "=" * 60)
print("RESIDUAL ANALYSIS")
print("=" * 60)

print(f"\\nResidual statistics:")
print(f"  Mean: \${np.mean (residuals):,.2f} (should be close to 0)")
print(f"  Std:  \${np.std (residuals):,.2f}")
print(f"  Min:  \${np.min (residuals):,.2f}")
print(f"  Max:  \${np.max (residuals):,.2f}")

# Summary
print("\\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"""
The linear regression model explains { test_r2: .1 %} of the variance in house prices.
On average, predictions are off by \${ mean_absolute_error (y_test, y_test_pred):,.0f }(MAE).

Key drivers of house price (by coefficient magnitude):
    1. { coef_df.iloc[0]['Feature'] }: \${ coef_df.iloc[0]['Coefficient']:,.2f }
2. { coef_df.iloc[1]['Feature'] }: \${ coef_df.iloc[1]['Coefficient']:,.2f }
3. { coef_df.iloc[2]['Feature'] }: \${ coef_df.iloc[2]['Coefficient']:,.2f }

Model performance is suitable for deployment as a baseline predictor.
Consider polynomial features or interaction terms for potential improvement.
""")
        `,
};
