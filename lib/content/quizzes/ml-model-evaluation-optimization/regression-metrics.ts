import { QuizQuestion } from '../../../types';

export const regressionMetricsQuiz: QuizQuestion[] = [
  {
    id: 'regression-metrics-dq-1',
    question:
      'Compare and contrast MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error). Discuss their mathematical differences, how they respond to outliers, and provide specific scenarios where you would choose one over the other.',
    sampleAnswer: `MAE and RMSE are the two most common regression metrics, and understanding their differences is crucial for correctly evaluating models.

**Mathematical Definitions:**

MAE (Mean Absolute Error):
$$\\text{MAE} = \\frac{1}{n} \\sum_{i=1}^{n} |y_i - \\hat{y}_i|$$

RMSE (Root Mean Squared Error):
$$\\text{RMSE} = \\sqrt{\\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2}$$

**Key Mathematical Difference:**

The fundamental difference is that RMSE squares the errors before averaging, then takes the square root. This seemingly small change has profound implications:

\`\`\`python
# Example: Same dataset, different error patterns
errors1 = [1, 1, 1, 1]  # Uniform errors
errors2 = [0, 0, 0, 4]  # One large error

# MAE treats them similarly
MAE1 = mean(|errors1|) = 1.0
MAE2 = mean(|errors2|) = 1.0  # Same!

# RMSE penalizes the large error heavily
RMSE1 = sqrt(mean(errors1²)) = sqrt(1) = 1.0
RMSE2 = sqrt(mean(errors2²)) = sqrt(4) = 2.0  # 2x larger!
\`\`\`

**Response to Outliers:**

**MAE is Robust to Outliers:**
- Uses absolute values (linear relationship)
- Each error contributes proportionally
- Large errors don't dominate the metric

**RMSE is Sensitive to Outliers:**
- Uses squared errors (quadratic relationship)
- Large errors contribute disproportionately
- A single huge error can dominate the metric

**Concrete Example:**

\`\`\`python
# House price predictions (in $1000s)
true_prices = [200, 300, 400, 500, 600]
pred_prices_good = [205, 295, 405, 495, 605]  # Consistently off by ~5
pred_prices_outlier = [200, 300, 400, 500, 800]  # One 200 error

# Good predictions:
MAE_good = 5.0
RMSE_good = 5.0
# Both metrics agree: typical error is $5,000

# Prediction with outlier:
MAE_outlier = 40.0  # (0+0+0+0+200)/5
RMSE_outlier = 89.4  # sqrt((0²+0²+0²+0²+200²)/5)

# Impact:
# MAE increased by 8x (40/5)
# RMSE increased by 17.9x (89.4/5) - much more sensitive!
\`\`\`

**RMSE/MAE Ratio as Diagnostic:**

The ratio RMSE/MAE reveals error distribution:
- Ratio ≈ 1.0: Errors are uniform
- Ratio >> 1.0: Presence of outliers or high variability
- Mathematically: RMSE ≥ MAE always (equality only when all errors are identical)

**When to Use MAE:**

1. **When outliers should not dominate**
   - Example: Delivery time prediction. Being 60 minutes late once shouldn't dominate metric more than being 5 minutes late twelve times.

2. **When all errors are equally important**
   - Example: Predicting daily temperature. Being off by 5°F is exactly twice as bad as being off by 2.5°F.

3. **When you have noisy data with outliers**
   - Example: Sensor readings with occasional faults. You want to evaluate typical performance, not be dominated by rare sensor failures.

4. **When interpretability is paramount**
   - MAE directly answers: "On average, how far off are my predictions?"
   - No squaring/square root conceptual overhead.

**When to Use RMSE:**

1. **When large errors are disproportionately bad**
   - Example: Medical dosage prediction. Being off by 2x the dosage is more than twice as bad—potentially fatal. The quadratic penalty makes sense.

2. **When training models with gradient descent**
   - MSE (and thus RMSE) is differentiable everywhere
   - MAE has a non-differentiable point at zero, causing optimization issues
   - Most deep learning uses MSE loss for this reason

3. **Default/standard comparisons**
   - RMSE is the conventional metric for Kaggle competitions and research papers
   - Allows easier comparison with published results

4. **When you want to penalize variance**
   - RMSE penalizes models with inconsistent errors
   - Two models with same MAE: RMSE is lower for consistent errors, higher for variable errors

**Practical Decision Framework:**

\`\`\`
Has outliers you want to ignore? → MAE
Training a neural network? → MSE/RMSE (differentiable)
Large errors catastrophic? → RMSE
Comparing to literature? → RMSE (standard)
Maximum interpretability? → MAE
Default choice? → RMSE
\`\`\`

**Real-World Example: Demand Forecasting**

\`\`\`python
# Retail demand forecasting: predict daily sales

# Scenario A: Use MAE
# - Occasional stockouts cause zero sales (outliers)
# - Want metric focused on typical prediction accuracy
# - Extreme stockouts are data quality issues, not prediction failures

# Scenario B: Use RMSE
# - No outliers in data
# - Large prediction errors mean either massive overstock (waste) or stockouts (lost revenue)
# - Cost is non-linear: 2x error → more than 2x cost
# - RMSE's quadratic penalty aligns with business cost
\`\`\`

**Key Insight:**

MAE and RMSE answer different questions:
- **MAE**: "What is my typical error?"
- **RMSE**: "What is my error, considering that large errors are worse?"

Choose based on:
1. Whether large errors are disproportionately bad (business context)
2. Presence of outliers (data quality)
3. Optimization requirements (training vs. evaluation)

**Best Practice**: Report both! MAE shows typical error, RMSE shows sensitivity to large errors. The ratio RMSE/MAE reveals error distribution quality.`,
    keyPoints: [
      'MAE uses absolute values (linear), RMSE squares errors (quadratic)',
      'RMSE penalizes large errors much more heavily than MAE',
      'MAE is robust to outliers, RMSE is sensitive to outliers',
      'RMSE/MAE ratio reveals error distribution: ~1.0 = uniform, >>1.0 = outliers present',
      'Use MAE when outliers should not dominate or all errors equally important',
      'Use RMSE when large errors are disproportionately bad or for standard comparisons',
      'Best practice: report both metrics for complete picture',
    ],
  },
  {
    id: 'regression-metrics-dq-2',
    question:
      'Explain what R-squared (R²) represents, how it differs from RMSE, and discuss its limitations. Why can a model have high R² but still make poor predictions? Provide an example scenario.',
    sampleAnswer: `R-squared (R²) is one of the most commonly used yet frequently misunderstood regression metrics. Understanding what it actually measures—and doesn't measure—is critical for proper model evaluation.

**What R² Represents:**

R² measures the **proportion of variance in the target variable that is explained by the model**. Mathematically:

$$R^2 = 1 - \\frac{\\text{SS}_{\\text{res}}}{\\text{SS}_{\\text{tot}}} = 1 - \\frac{\\sum(y_i - \\hat{y}_i)^2}{\\sum(y_i - \\bar{y})^2}$$

Where:
- SS_res = Sum of squared residuals (model errors)
- SS_tot = Total sum of squares (variance from mean)

**Intuitive Interpretation:**

R² compares your model to a naive baseline that always predicts the mean:

\`\`\`python
# Baseline model: always predict mean
baseline_predictions = [mean(y)] * n
baseline_MSE = variance(y)

# Your model
your_model_MSE = MSE(y_true, y_pred)

# R² tells you the improvement
R² = 1 - (your_model_MSE / baseline_MSE)

# Examples:
# R² = 0.0: Your model is no better than predicting mean
# R² = 0.5: Your model is 50% better than baseline
# R² = 0.9: Your model is 90% better than baseline
# R² = 1.0: Perfect predictions
# R² < 0: Your model is WORSE than predicting mean!
\`\`\`

**How R² Differs from RMSE:**

**R² (Scale-Independent):**
- Bounded: typically [0, 1] (can be negative)
- Relative measure: compares to baseline
- No units: pure proportion
- Same R² can mean different prediction quality depending on data variance

**RMSE (Scale-Dependent):**
- Unbounded: [0, ∞)
- Absolute measure: actual prediction error
- Same units as target variable
- Directly interpretable: "predictions are off by X units"

**Example Demonstrating the Difference:**

\`\`\`python
# Dataset 1: House prices in dollars
y1_true = [100000, 200000, 300000, 400000]
y1_pred = [110000, 210000, 290000, 390000]
RMSE1 = 10,000
R²1 = 0.99

# Dataset 2: House prices in different city (higher variance)
y2_true = [100000, 500000, 900000, 1300000]
y2_pred = [110000, 510000, 890000, 1290000]
RMSE2 = 10,000  # Same RMSE!
R²2 = 0.999  # Higher R²!

# Same prediction error (RMSE = $10k), but different R² because
# Dataset 2 has higher variance - model explains more of it
\`\`\`

**Major Limitations of R²:**

**1. High R² Doesn't Guarantee Good Predictions**

\`\`\`python
# Scenario: Stock price prediction

# Dataset with high variance
prices = [10, 50, 100, 150, 200]  # High variance
predictions = [15, 45, 95, 145, 195]  # Consistently off by 5
R² = 0.998  # Excellent!
RMSE = $5

# But $5 error might be huge for your trading strategy!
# If you're trading on 1% moves, 5% errors are catastrophic

# High R² just means you captured the variance (10-200 range)
# Doesn't mean predictions are good enough for your use case
\`\`\`

**2. R² Increases with More Features (Even Random Ones)**

\`\`\`python
# Original model: 5 features
R² = 0.75

# Add 20 random noise features
R² = 0.82  # Increased!

# Problem: R² always increases when adding features, even useless ones
# Model is actually worse (overfitted), but R² suggests improvement

# Solution: Use Adjusted R² which penalizes extra features
R²_adj = 1 - (1-R²) * (n-1)/(n-p-1)
\`\`\`

**3. Doesn't Detect Systematic Bias**

\`\`\`python
# Model that systematically overpredicts by 10%
y_true = [100, 200, 300, 400]
y_pred = [110, 220, 330, 440]  # All 10% too high

# Still can have high R² because variance is captured
# But model has systematic bias that R² doesn't reveal

# Need to check:
# - Residual plot (should be random around zero)
# - Mean error (should be ~0)
\`\`\`

**4. Context-Dependent Interpretation**

What's a "good" R² depends on the field:
- Physics: R² < 0.9 might be poor (high signal-to-noise)
- Social sciences: R² > 0.3 might be excellent (high noise)
- Stock prediction: R² = 0.01 might be valuable (extremely noisy)

**Example: When High R² Means Poor Predictions**

**Scenario: Predicting Individual Medical Costs**

\`\`\`python
# Medical cost data
costs = [1000, 2000, 3000, 4000, ..., 50000]  # Wide range
mean_cost = $10,000
variance = very high

# Model predictions
predictions = costs + random_noise(std=500)

# Results:
R² = 0.95  # Excellent! Explains 95% of variance
RMSE = $500

# BUT: For individual patient billing:
# - $500 error is huge (unacceptable for invoicing)
# - High R² just reflects capturing the wide range (1k-50k)
# - Doesn't mean predictions are accurate enough for the business need

# The high R² comes from capturing the gross pattern
# (expensive patients vs cheap patients)
# Not from accurate individual predictions

# For insurance pricing, might need RMSE < $50, which would still
# give R² ≈ 0.95 but be 10x better for the business
\`\`\`

**Why This Happens:**

When data has high variance (wide spread), even mediocre predictions can achieve high R² by simply capturing the gross pattern. R² measures "did you capture the variance?" not "are your predictions good enough?"

**Real-World Trading Example:**

\`\`\`python
# Stock return prediction
# True returns: -5%, -2%, 0%, +2%, +5%  (variance = 0.001)
# Predictions: -4.5%, -1.5%, +0.5%, +2.5%, +5.5%

R² = 0.97  # Excellent!

# But for a trading strategy:
# - Sign errors (predicted +0.5% but actually 0%)
# - Could lose money despite high R²
# - What matters: directional accuracy, not variance explained

# Better metrics for trading:
# - Directional accuracy: % of times you got the sign right
# - Sharpe ratio: risk-adjusted returns
# - Maximum drawdown: worst loss
\`\`\`

**Best Practices:**

1. **Never use R² alone** - Always pair with RMSE or MAE
2. **Check residual plots** - Verify no systematic patterns
3. **Use Adjusted R²** - When comparing models with different features
4. **Consider context** - "Good" R² depends on field and business needs
5. **Validate business utility** - High R² ≠ model is useful for your application

**Key Insight:**

R² answers: "Did my model capture the variance?"
RMSE answers: "How far off are my predictions?"

You need both:
- R² = 0.9, RMSE = $1 → Great model
- R² = 0.9, RMSE = $50,000 → Captured variance but huge errors
- R² = 0.3, RMSE = $1 → Low variance explained but excellent predictions

The "best" metric depends on your specific problem and business requirements, not just statistical elegance.`,
    keyPoints: [
      'R² measures proportion of variance explained, not prediction accuracy',
      'R² is scale-independent (unitless), RMSE is scale-dependent (same units as target)',
      'High R² can coexist with poor predictions when variance is high',
      'R² increases with more features even if useless (use Adjusted R² instead)',
      "R² doesn't detect systematic bias or directional errors",
      'Context matters: good R² varies by field (0.9 physics, 0.3 social science)',
      'Always report R² alongside RMSE/MAE for complete evaluation',
    ],
  },
  {
    id: 'regression-metrics-dq-3',
    question:
      'Discuss MAPE (Mean Absolute Percentage Error), including its advantages for interpretability and scale-independence, its major limitations and failure modes, and provide guidelines for when MAPE should and should not be used.',
    sampleAnswer: `MAPE (Mean Absolute Percentage Error) is popular in business contexts for its interpretability, but it has serious limitations that can make it inappropriate or even catastrophic for certain problems.

**Definition and Calculation:**

$$\\text{MAPE} = \\frac{100\\%}{n} \\sum_{i=1}^{n} \\left|\\frac{y_i - \\hat{y}_i}{y_i}\\right|$$

\`\`\`python
# Example calculation
y_true = [100, 200, 300]
y_pred = [110, 190, 330]

# Percentage errors
pe_1 = |100 - 110| / 100 = 0.10 (10%)
pe_2 = |200 - 190| / 200 = 0.05 (5%)
pe_3 = |300 - 330| / 300 = 0.10 (10%)

MAPE = (10% + 5% + 10%) / 3 = 8.33%
\`\`\`

**Advantages of MAPE:**

**1. Intuitive Interpretability**

MAPE provides a percentage that business stakeholders immediately understand:
- "Predictions are off by 15% on average" is crystal clear
- No need to know units or scales
- Universally understood metric

**2. Scale-Independence**

MAPE allows comparison across different scales and problems:

\`\`\`python
# Problem 1: Predicting apartment rent ($1000-$3000)
true1 = [1000, 2000, 3000]
pred1 = [1100, 2100, 3100]
MAE1 = 100
MAPE1 = 6.7%

# Problem 2: Predicting house prices ($200k-$600k)  
true2 = [200000, 400000, 600000]
pred2 = [220000, 420000, 620000]
MAE2 = 20000  # 200x larger!
MAPE2 = 6.7%  # Same! Comparable!

# MAPE reveals both models have similar relative accuracy
# despite vastly different absolute scales
\`\`\`

**3. Business-Friendly**

- Aligns with business KPIs (often expressed in percentages)
- Easy to set targets: "We need MAPE < 10%"
- Facilitates cross-departmental communication

**Major Limitations and Failure Modes:**

**1. FATAL: Division by Zero**

When true value is zero or near-zero, MAPE fails catastrophically:

\`\`\`python
# Product demand prediction
y_true = [0, 10, 20, 30]  # New product, zero initial sales
y_pred = [5, 12, 18, 32]

# MAPE calculation:
# Error 1: |0 - 5| / 0 = UNDEFINED (division by zero)
# → MAPE cannot be calculated!

# Even near-zero causes problems:
y_true = [0.1, 10, 20, 30]  # Tiny first value
y_pred = [1, 12, 18, 32]
# Error 1: |0.1 - 1| / 0.1 = 900%  (!!!)
# → MAPE = 233% (dominated by first term)
\`\`\`

**Solution**: Use sMAPE (symmetric MAPE) or avoid MAPE entirely for data with zeros.

**2. Asymmetric Penalty (Over vs Under-Prediction)**

MAPE penalizes under-predictions more severely than over-predictions:

\`\`\`python
# Predict sales of 100 units
y_true = 100

# Case A: Over-predict by 50 units
y_pred_over = 150
MAPE_over = |100 - 150| / 100 = 50%

# Case B: Under-predict by 50 units
y_pred_under = 50
MAPE_under = |100 - 50| / 100 = 50%  # Wait, same?

# But look at percentage terms relative to predictions:
# Over: predicted 150, off by 50 → 33% relative to prediction
# Under: predicted 50, off by 50 → 100% relative to prediction

# MAPE is symmetric in absolute terms but asymmetric in practice
# because division by true value creates bias

# Extreme example:
y_true = 100
over_pred = 200  # 100 units over
under_pred = 0   # 100 units under

MAPE_over = |100-200|/100 = 100%
MAPE_under = |100-0|/100 = 100%  # Same MAPE

# But under-prediction hit zero! Much worse in practice!
\`\`\`

**3. Bias Toward Under-Prediction**

MAPE incentivizes models to under-predict because errors on small values contribute more:

\`\`\`python
# Two products to forecast
product_A_true = 10 units
product_B_true = 100 units

# Model 1: Under-predicts both by 5 units
A1_pred = 5
B1_pred = 95
MAPE1 = (|10-5|/10 + |100-95|/100) / 2 = (50% + 5%) / 2 = 27.5%

# Model 2: Over-predicts both by 5 units
A2_pred = 15
B2_pred = 105
MAPE2 = (|10-15|/10 + |100-105|/100) / 2 = (50% + 5%) / 2 = 27.5%

# Same MAPE, but notice:
# Both models penalize small-value errors (product A) equally
# Under-prediction is penalized more for large values

# In practice, models learn to under-predict to avoid
# large penalties on small-value items
\`\`\`

**4. Disproportionate Penalty for Small Values**

Same absolute error → much higher MAPE for small values:

\`\`\`python
# Warehouse inventory prediction
# Same 10-unit error on different base levels

prediction_1 = 20, true_1 = 10  # Small item
MAPE_1 = |10-20|/10 = 100%

prediction_2 = 110, true_2 = 100  # Medium item
MAPE_2 = |100-110|/100 = 10%

prediction_3 = 1010, true_3 = 1000  # Large item
MAPE_3 = |1000-1010|/1000 = 1%

# Same 10-unit error, vastly different MAPE
# Model will focus on large-value items
# Small-value items will be poorly predicted
\`\`\`

**5. Cannot Handle Negative Values**

MAPE is undefined for negative values:

\`\`\`python
# Financial returns (can be negative)
returns_true = [-5%, 2%, 8%, -3%]
returns_pred = [-4%, 3%, 7%, -2%]

# MAPE fails:
# |-5 - (-4)| / |-5| = 1/5 = 20%  # OK
# |2 - 3| / |2| = 1/2 = 50%  # OK
# |8 - 7| / |8| = 1/8 = 12.5%  # OK
# |-3 - (-2)| / |-3| = 1/3 = 33%  # Signed value lost!

# Absolute value in denominator destroys sign information
# Cannot distinguish positive from negative returns properly
\`\`\`

**When to Use MAPE:**

✅ **Use MAPE when:**

1. **All values are strictly positive and away from zero**
   - Example: Revenue forecasting (always positive, typically > $1000)
   - Example: Population prediction (always positive, rarely near zero)

2. **Relative errors matter more than absolute**
   - Example: Retail demand (10% error on 100 units vs 1000 units should be treated similarly)
   - Example: Website traffic (10% error matters more than absolute visitor count)

3. **Communicating to business stakeholders**
   - Percentages are universally understood
   - Easier to set targets and track progress

4. **Comparing across different scales**
   - Evaluating the same model on products with different price points
   - Multi-region forecasting with different scales

**❌ Do NOT use MAPE when:**

1. **Data contains zeros or near-zeros**
   - Use MAE or RMSE instead
   - Or use symmetric MAPE (sMAPE) variant

2. **Data contains negative values**
   - Example: Profit/loss forecasting, temperature prediction
   - Use RMSE or MAE

3. **Small values are important**
   - MAPE will under-represent errors on small values
   - Use weighted MAPE or MAE

4. **Asymmetric costs for over vs under-prediction**
   - Inventory: stockouts (under) vs overstock (over) have different costs
   - Use asymmetric loss function instead

**Better Alternatives:**

1. **sMAPE (Symmetric MAPE)**:
   $$\\text{sMAPE} = \\frac{100\\%}{n} \\sum_{i=1}^{n} \\frac{|y_i - \\hat{y}_i|}{(|y_i| + |\\hat{y}_i|)/2}$$
   - Handles near-zero better
   - More symmetric for over/under-prediction

2. **WAPE (Weighted Absolute Percentage Error)**:
   $$\\text{WAPE} = \\frac{\\sum|y_i - \\hat{y}_i|}{\\sum|y_i|}$$
   - More robust to outliers
   - Better for sparse or intermittent demand

3. **MAE for absolute errors, report alongside percentages**:
   - "MAE = $500 (avg 15% error)"
   - Best of both worlds

**Best Practice Decision Tree:**

\`\`\`
Does data have zeros or negatives?
 YES → Don't use MAPE (use MAE/RMSE)
 NO ↓

Are all values >> 0 (e.g., all > 10)?
 NO → Don't use MAPE (small values will dominate)
 YES ↓

Do you need scale-independent comparison?
 YES → Consider MAPE (or sMAPE for safety)
 NO → Use RMSE/MAE (more robust)

Are you presenting to business stakeholders?
 YES → Report MAPE alongside RMSE (interpretability + rigor)
 NO → RMSE/MAE sufficient
\`\`\`

**Key Takeaway:**

MAPE is excellent for interpretability and business communication, but it fails catastrophically with zeros, penalizes small values disproportionately, and creates asymmetric incentives. Use it only when you've verified:
1. All values are positive and away from zero
2. Relative errors matter more than absolute
3. You understand and accept its biases

For critical applications, always pair MAPE with RMSE or MAE to get both interpretability and robustness.`,
    keyPoints: [
      'MAPE is interpretable (percentage error) and scale-independent',
      'Fatal flaw: division by zero when true value is zero or near-zero',
      'Asymmetric: penalizes under-predictions more than over-predictions',
      'Bias toward small values: same absolute error → higher MAPE for small values',
      'Cannot handle negative values (sign information lost)',
      'Use only when: values strictly positive, away from zero, relative errors matter',
      "Don't use when: zeros present, negative values, small values important",
      'Best practice: report MAPE alongside RMSE/MAE for interpretability + robustness',
    ],
  },
];
