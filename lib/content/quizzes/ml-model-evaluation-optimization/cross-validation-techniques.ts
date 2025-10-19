import { QuizQuestion } from '../../../types';

export const crossValidationTechniquesQuiz: QuizQuestion[] = [
  {
    id: 'cross-validation-techniques-dq-1',
    question:
      'Explain why a single train-test split might give misleading performance estimates, and how cross-validation addresses this problem. Include a discussion of the bias-variance tradeoff in choosing the number of folds K.',
    sampleAnswer: `A single train-test split is subject to random chance—your particular split might be "lucky" (easy test samples) or "unlucky" (hard test samples), leading to unreliable performance estimates. Cross-validation solves this by averaging across multiple splits, providing a more robust estimate.

**Problems with Single Split:**

1. **High Variance**: Different random splits can give very different performance estimates. For example, with 100 samples, one split might achieve 95% accuracy while another achieves 88% on the exact same model.

2. **Sample Selection Bias**: Your test set might accidentally contain only easy (or hard) examples, making your model appear better (or worse) than it actually is.

3. **Limited Data Usage**: Only a portion of data is used for training (e.g., 80%), potentially under-representing the true data distribution.

4. **No Confidence Interval**: A single number (e.g., "accuracy = 92%") provides no information about uncertainty or stability.

**How Cross-Validation Addresses These Issues:**

Cross-validation creates K different train-test splits, trains K models, and averages their performance:

\`\`\`python
# Single split: one performance number
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)  # Single number: 0.89
\`\`\`

\`\`\`python
# 5-fold CV: five performance numbers + average
scores = cross_val_score(model, X, y, cv=5)
# scores: [0.87, 0.91, 0.88, 0.92, 0.89]
# mean: 0.894, std: 0.019
\`\`\`

**Benefits:**
- **Lower Variance**: Averaging across K folds reduces variance in the estimate
- **Confidence Intervals**: Standard deviation across folds gives uncertainty estimate
- **Better Data Usage**: Every sample is used for training (K-1)/K times
- **Detect Instability**: High standard deviation reveals model/data issues

**Bias-Variance Tradeoff in Choosing K:**

The choice of K affects the bias and variance of the CV estimate:

**Small K (e.g., K=2 or K=3):**
- **Higher Bias**: Each training set uses only 50-67% of data, underrepresenting the full dataset. Performance estimate is pessimistic (biased downward) because models are trained on less data than they would be in production.
- **Lower Variance**: Fewer folds means less variability between folds.
- **Faster**: Fewer models to train.

**Large K (e.g., K=10 or K=N):**
- **Lower Bias**: Each training set uses 90-99% of data, closely representing production scenario where you'd train on all data.
- **Higher Variance**: More folds means more opportunity for variation between folds.
- **Slower**: More models to train (K times slower than single split).

**Mathematical Intuition:**
- Bias decreases as K increases because training sets become more representative
- Variance increases as K increases because test sets become smaller and more variable
- LOOCV (K=N) has minimum bias but high variance and high computational cost

**Practical Recommendations:**

| K Value | Use Case | Bias | Variance | Speed |
|---------|----------|------|----------|-------|
| K=3 | Very expensive models | High | Low | Fast |
| K=5 | Default choice | Medium | Medium | Medium |
| K=10 | Standard for large datasets | Low | Medium | Slow |
| K=N | Small datasets (<100 samples) | Minimum | High | Very Slow |

**Empirical Rule**: K=5 or K=10 provide good bias-variance balance for most problems. Research shows that K=10 often provides the best tradeoff.

**Example Demonstrating the Tradeoff:**

\`\`\`python
# Dataset: 1000 samples
# True model performance on infinite data: 0.85

# K=2 (50% training data per fold)
# Estimated performance: 0.82 (pessimistic due to less training data)
# Standard deviation: 0.01 (stable)

# K=10 (90% training data per fold)
# Estimated performance: 0.847 (close to true 0.85)
# Standard deviation: 0.025 (more variable)

# K=1000 (LOOCV, 99.9% training data)
# Estimated performance: 0.849 (very close to 0.85)
# Standard deviation: 0.045 (high variance, training sets are too similar)
\`\`\`

**Key Insight**: K=5 or K=10 hit the "sweet spot"—sufficient training data to get accurate estimate, enough folds to reduce variance, and reasonable computational cost. Don't obsess over the exact value; anything from K=5 to K=10 works well for most problems.`,
    keyPoints: [
      'Single split has high variance - different random splits give very different performance estimates',
      'Cross-validation averages across K splits, reducing variance and providing confidence intervals',
      'Small K (2-3): high bias (pessimistic), low variance, fast',
      'Large K (10-N): low bias (accurate), higher variance, slow',
      'K=5 or K=10 provide good bias-variance tradeoff for most problems',
      'LOOCV (K=N) has minimum bias but high variance and computational cost',
    ],
  },
  {
    id: 'cross-validation-techniques-dq-2',
    question:
      'For a time series forecasting problem (e.g., stock price prediction), explain why regular K-fold cross-validation is inappropriate and describe the correct approach using time series cross-validation. What is the purpose of adding a "gap" between training and test sets?',
    sampleAnswer: `Regular K-fold cross-validation is fundamentally broken for time series because it violates the temporal structure of the data, creating data leakage and unrealistic evaluation scenarios.

**Why Regular K-Fold Fails for Time Series:**

1. **Temporal Data Leakage (Look-Ahead Bias)**

Regular K-fold randomly shuffles data into folds, mixing past and future:

\`\`\`python
# Stock prices: 2020, 2021, 2022, 2023
# Regular K-fold might create:
#   Train: [2020, 2022, 2023]  # Uses future to predict past!
#   Test:  [2021]               # Tested on middle period

# Problem: Model can learn from 2023 data to "predict" 2021 prices
# This is impossible in production where you only have past data
\`\`\`

The model sees future patterns and relationships that wouldn't be available at prediction time, leading to grossly inflated performance estimates.

2. **Violates Temporal Dependencies**

Time series data has autocorrelation—today's value depends on yesterday's, last week's, etc. Shuffling destroys these dependencies:

\`\`\`python
# Sequential relationship:
#   Price(t) depends on Price(t-1), Price(t-2), ...
# Random shuffling breaks this:
#   Training might see: t=1, t=50, t=3, t=100
#   Missing the sequential pattern!
\`\`\`

3. **Unrealistic Production Scenario**

In deployment, you always predict future from past, never the reverse:
- Production: Use data up to today (Dec 15) to predict tomorrow (Dec 16)
- K-fold: Might use Dec 20 data to "predict" Dec 10 (impossible!)

**Correct Approach: Time Series Cross-Validation**

Time series CV uses **sequential splits** that preserve temporal order:

\`\`\`python
from sklearn.model_selection import TimeSeriesSplit

# Data: Jan-Dec 2023 (12 months)
# 5-fold time series split:

# Fold 1:
#   Train: Jan-Apr (4 months)
#   Test:  May      (1 month)

# Fold 2:
#   Train: Jan-Jun  (6 months)  # Expands
#   Test:  Jul      (1 month)

# Fold 3:
#   Train: Jan-Aug  (8 months)  # Expands
#   Test:  Sep      (1 month)

# Fold 4:
#   Train: Jan-Oct  (10 months) # Expands
#   Test:  Nov      (1 month)

# Fold 5:
#   Train: Jan-Nov  (11 months) # Expands
#   Test:  Dec      (1 month)
\`\`\`

**Key Properties:**
- **Training always precedes testing temporally**
- **Expanding window**: Training set grows over time (more realistic)
- **No look-ahead bias**: Future never influences past
- **Simulates production**: Each fold mimics deploying model at that time point

**Why Add a Gap Between Train and Test?**

In real-world production, there's often a delay between when you train a model and when you can deploy it:

\`\`\`python
# Without gap:
#   Train: Jan 1 - Mar 31
#   Test:  Apr 1 - Apr 30
#   Problem: In reality, training takes time!

# With 7-day gap:
#   Train: Jan 1 - Mar 31
#   Gap:   Apr 1 - Apr 7    # Model training + deployment time
#   Test:  Apr 8 - May 7
#   Realistic: Model deployed on Apr 8 using data up to Mar 31
\`\`\`

**Reasons for Gaps:**

1. **Training Time**: Model training takes hours or days
2. **Data Collection Lag**: Features might not be available immediately
3. **Deployment Overhead**: Testing, approval, rollout takes time
4. **Feature Engineering**: Calculating features requires recent history

**Example: Trading System**

\`\`\`python
# Real trading scenario:
#   - Collect data: End of day, 4 PM
#   - Calculate features: 4:30 PM (30 min lag)
#   - Train model: 4:30 PM - 6 PM (1.5 hours)
#   - Deploy: Next morning, 9:30 AM
#   - Total gap: ~17.5 hours

# CV should reflect this:
def time_series_cv_with_gap(data, gap_days=1):
    for train_end in range(60, len(data), 20):
        train_idx = range(0, train_end)
        test_start = train_end + gap_days
        test_idx = range(test_start, test_start + 20)
        yield train_idx, test_idx

# This prevents the model from using information that wouldn't be
# available at prediction time in production
\`\`\`

**Impact of Gap on Performance:**

\`\`\`python
# No gap: Predicting Apr 1 using data up to Mar 31
# Performance: R² = 0.85 (optimistic)

# 1-day gap: Predicting Apr 2 using data up to Mar 31  
# Performance: R² = 0.82 (more realistic)

# 7-day gap: Predicting Apr 8 using data up to Mar 31
# Performance: R² = 0.76 (realistic for weekly retraining)
\`\`\`

Gap reduces performance because:
- More time between training and prediction
- Market conditions may change during gap
- Recent trends not captured
- More realistic expectation for production

**Implementation Example:**

\`\`\`python
from sklearn.model_selection import TimeSeriesSplit

# Standard time series CV
tscv = TimeSeriesSplit(n_splits=5)
scores = []
for train_idx, test_idx in tscv.split(stock_data):
    X_train, X_test = stock_data[train_idx], stock_data[test_idx]
    y_train, y_test = returns[train_idx], returns[test_idx]
    
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    scores.append(score)

# With custom gap
def ts_split_with_gap(data, n_splits=5, gap=5):
    # Implementation that skips 'gap' days between train and test
    ...

# Compare
print(f"No gap: {np.mean(scores_no_gap):.3f}")
print(f"5-day gap: {np.mean(scores_gap):.3f}")
# No gap: 0.850 (optimistic)
# 5-day gap: 0.801 (realistic)
\`\`\`

**Best Practices for Time Series CV:**

1. **Always use TimeSeriesSplit** or custom temporal CV
2. **Add gap if production has deployment lag**
3. **Use expanding window** (default) rather than rolling window
4. **Match CV setup to production scenario** as closely as possible
5. **Test on most recent data** (last fold is most representative)
6. **Consider multiple seasonal cycles** in train/test sizes

**Key Principle**: Your cross-validation strategy should mimic your production deployment as closely as possible. For time series, this means respecting temporal order and accounting for real-world deployment constraints like gaps and retraining frequency.`,
    keyPoints: [
      'Regular K-fold creates look-ahead bias by using future data to predict past',
      'Time series CV uses sequential splits: training always precedes testing',
      'Expanding window approach grows training set over time',
      'Gap between train/test simulates real deployment delay (training time, data lag)',
      'Without gap: optimistic performance; with gap: realistic performance',
      'Gap accounts for model training time, data collection lag, and deployment overhead',
      'Always match CV strategy to production scenario for accurate performance estimates',
    ],
  },
  {
    id: 'cross-validation-techniques-dq-3',
    question:
      'Explain what nested cross-validation is, why it provides an unbiased estimate of model performance when doing hyperparameter tuning, and when you should use it vs. a simpler train-validation-test split.',
    sampleAnswer: `Nested cross-validation is a technique that provides truly unbiased performance estimates when selecting models or tuning hyperparameters. It uses two layers of cross-validation: an outer loop for performance estimation and an inner loop for model selection.

**The Problem: Regular CV Can Be Biased**

When you use cross-validation to select hyperparameters, your performance estimate becomes optimistically biased:

\`\`\`python
# ❌ Biased approach:
param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(Ridge(), param_grid, cv=5)
grid_search.fit(X, y)
best_score = grid_search.best_score_  # 0.92

# Problem: This score is optimistic! You tried 5 different alphas
# and reported the best one. You've implicitly "overfitted" to the
# CV folds by selecting the hyperparameter that works best on them.
\`\`\`

**Why This Creates Bias:**

1. **Multiple Hypothesis Testing**: Each hyperparameter combination is a hypothesis test. Trying many combinations increases the probability of finding one that works well by chance on your particular CV folds.

2. **Optimization for CV Folds**: By selecting the best-performing hyperparameter, you've optimized specifically for your CV folds. This hyperparameter might not work as well on truly new data.

3. **No Independent Test Set**: The CV folds that selected the hyperparameter are the same folds that estimated performance—there's no independent evaluation.

**Solution: Nested Cross-Validation**

Nested CV adds an outer loop that provides independent performance estimation:

\`\`\`python
# ✅ Unbiased approach: Nested CV

# Outer loop: performance estimation (5 folds)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
nested_scores = []

for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
    # Split into outer train/test
    X_train_outer = X[train_idx]
    X_test_outer = X[test_idx]
    y_train_outer = y[train_idx]
    y_test_outer = y[test_idx]
    
    # Inner loop: hyperparameter tuning (3 folds)
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(Ridge(), param_grid, cv=inner_cv)
    grid_search.fit(X_train_outer, y_train_outer)
    
    # Get best model from inner CV
    best_model = grid_search.best_estimator_
    
    # Evaluate on outer test set (INDEPENDENT!)
    score = best_model.score(X_test_outer, y_test_outer)
    nested_scores.append(score)

print(f"Nested CV score: {np.mean(nested_scores):.4f} +/- {np.std(nested_scores):.4f}")
# This is an unbiased estimate!
\`\`\`

**Structure of Nested CV:**

\`\`\`
Outer Fold 1:
  Inner Fold 1: train on 40%, test on 20% (of outer train)
  Inner Fold 2: train on 40%, test on 20% (of outer train)
  Inner Fold 3: train on 40%, test on 20% (of outer train)
  → Select best hyperparameter: alpha=1.0
  → Test on outer test set (20% of total) → Score: 0.89

Outer Fold 2:
  Inner Fold 1: train on 40%, test on 20% (of outer train)
  Inner Fold 2: train on 40%, test on 20% (of outer train)
  Inner Fold 3: train on 40%, test on 20% (of outer train)
  → Select best hyperparameter: alpha=10.0 (might differ!)
  → Test on outer test set (20% of total) → Score: 0.91

... (3 more outer folds)

Final nested CV score: mean of 5 outer fold scores
\`\`\`

**Why Nested CV Is Unbiased:**

1. **Independent Test Sets**: Outer test sets never participate in hyperparameter selection—they're only used for final evaluation.

2. **Realistic Scenario**: Each outer fold simulates a complete model development cycle (tune hyperparameters → evaluate), just as you would in production.

3. **No Information Leakage**: Hyperparameter selection (inner CV) and performance estimation (outer CV) are completely separate.

**When to Use Nested CV vs. Train-Val-Test Split:**

**Use Train-Val-Test Split When:**
- **Large dataset** (> 10,000 samples): Enough data to afford three separate sets
- **Fixed hyperparameters**: Not doing extensive hyperparameter search
- **Faster iteration**: Need quick feedback during development
- **Final model selection**: Once development is complete, want one final performance number
- **Simpler to understand**: Easier to explain to stakeholders

\`\`\`python
# Train-Val-Test split:
# 1. Split once: 60% train, 20% val, 20% test
# 2. Tune hyperparameters on train+val
# 3. Evaluate once on test
# Fast, simple, requires more data
\`\`\`

**Use Nested CV When:**
- **Small dataset** (< 1,000 samples): Need to maximize data usage
- **Extensive hyperparameter tuning**: Trying many hyperparameters and want unbiased estimate
- **Comparing multiple models**: Want fair comparison between different algorithms
- **Research/publication**: Need rigorous, unbiased performance estimates
- **High variance**: Want confidence intervals from multiple CV folds
- **Model stability**: Want to see how hyperparameters vary across folds

\`\`\`python
# Nested CV:
# 1. Outer 5-fold split for performance estimation
# 2. Inner 3-fold split for hyperparameter tuning (per outer fold)
# 3. Get 5 independent performance estimates
# More robust, requires less data, computationally expensive
\`\`\`

**Practical Comparison:**

\`\`\`python
# Scenario: 800 samples, tune 20 hyperparameters

# Train-Val-Test split:
#   Train: 480 samples (60%)
#   Val:   160 samples (20%)
#   Test:  160 samples (20%)
# Computational cost: Train 20 models (try 20 hyperparameters)
# Result: One performance number
# Problem: With only 800 samples, holding out 40% is wasteful

# Nested CV (5x3):
#   Outer loop: 5 folds of 640 train / 160 test
#   Inner loop: 3 folds of 426 train / 214 val (per outer fold)
# Computational cost: Train 5*20*3 = 300 models
# Result: 5 performance numbers + mean + std
# Advantage: Uses more data (80% for training in outer loop)
\`\`\`

**Computational Cost:**

Nested CV is expensive:
- Outer folds: K_outer
- Inner folds per outer fold: K_inner
- Hyperparameter combinations: H
- Total models trained: K_outer × K_inner × H

For K_outer=5, K_inner=3, H=20:
- Total models: 5 × 3 × 20 = 300 models

**Decision Matrix:**

| Criterion | Train-Val-Test | Nested CV |
|-----------|----------------|-----------|
| Dataset size | > 10,000 | < 1,000 |
| Computational budget | Limited | Sufficient |
| Hyperparameter search | Small | Extensive |
| Bias in estimate | Slightly biased | Unbiased |
| Confidence interval | No | Yes (from folds) |
| Typical use case | Industry/production | Research/small data |

**Hybrid Approach:**

Many practitioners use a combination:
1. **During development**: Use simple train-val split for fast iteration
2. **Before deployment**: Use nested CV to get unbiased performance estimate
3. **Final training**: Train on all data with best hyperparameters from nested CV

\`\`\`python
# Development phase: fast iteration
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# ... quick experiments ...

# Validation phase: rigorous evaluation
nested_scores = nested_cross_val(X_train, y_train, ...)  # Unbiased estimate

# Production phase: use all data
final_model = BestModel(best_hyperparameters)
final_model.fit(X, y)  # Train on ALL data
\`\`\`

**Key Principle**: Nested CV provides the gold standard for unbiased performance estimation when tuning hyperparameters, at the cost of computational expense. Use it when you need rigorous evaluation with limited data, or when publishing research. For large datasets or quick iterations, train-val-test split is more practical.`,
    keyPoints: [
      'Nested CV uses two layers: outer loop for performance estimation, inner loop for hyperparameter tuning',
      'Regular CV with hyperparameter selection is biased - you optimize for specific CV folds',
      'Nested CV provides unbiased estimates by keeping outer test sets independent',
      'Use nested CV for small datasets (<1000), extensive tuning, or research',
      'Use train-val-test split for large datasets (>10,000), faster iteration, or production',
      'Computational cost: K_outer × K_inner × H models trained',
      'Nested CV gives confidence intervals; train-val-test gives single estimate',
    ],
  },
];
