import { QuizQuestion } from '../../../types';

export const trainTestSplitValidationQuiz: QuizQuestion[] = [
  {
    id: 'train-test-split-validation-dq-1',
    question:
      'Explain what data leakage is, why it leads to overly optimistic model performance, and provide three specific examples of how data leakage can occur in a machine learning pipeline.',
    sampleAnswer: `Data leakage occurs when information from outside the training dataset influences the model during training, leading to unrealistic performance estimates that don't generalize to new data.

**Why It Causes Overly Optimistic Performance:**

When a model is exposed to information about the test set during training (even indirectly), it learns patterns that won't be available during real-world deployment. The model appears to perform well because it's partially "memorized" information about the test data, but this performance collapses when faced with truly unseen data in production.

**Three Specific Examples:**

1. **Scaling/Normalization Before Splitting:**
\`\`\`python
# WRONG: Leaks test set statistics into training
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Uses mean/std from ALL data
X_train, X_test = train_test_split(X_scaled, y)

# CORRECT: Fit only on training data
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Uses only training mean/std
X_test = scaler.transform(X_test)  # Apply training statistics
\`\`\`

The mean and standard deviation calculated from the entire dataset include information about the test set. The model indirectly learns about test set distributions, making it perform artificially well.

2. **Feature Selection on Entire Dataset:**
\`\`\`python
# WRONG: Selects features based on entire dataset
selector = SelectKBest(k=10)
X_selected = selector.fit_transform(X, y)  # Uses correlations from ALL data
X_train, X_test = train_test_split(X_selected, y)

# CORRECT: Select features only on training data
X_train, X_test, y_train, y_test = train_test_split(X, y)
selector = SelectKBest(k=10)
X_train = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)
\`\`\`

Features are selected based on their correlation with the target in the ENTIRE dataset, including the test set. This means features are chosen because they work well on test data, artificially inflating performance.

3. **Using Future Information in Time Series:**
\`\`\`python
# WRONG: Creates features using future information
df['rolling_mean_7d'] = df['value'].rolling(7, center=True).mean()  # center=True uses future values

# CORRECT: Only use past information
df['rolling_mean_7d'] = df['value'].rolling(7).mean()  # Only looks backward
\`\`\`

In time series, using \`center=True\` or accidentally including future data in features means the model learns from information that wouldn't be available at prediction time in production.

**Detection and Prevention:**

- **Use pipelines**: sklearn's Pipeline ensures all transformations are fitted only on training data
- **Split first**: Always split data before ANY preprocessing
- **Document data flow**: Track what information flows where
- **Test temporal consistency**: For time series, verify features only use past data
- **Validation checks**: Test that removing test set doesn't change preprocessing parameters

**Real-World Impact:**

Data leakage is one of the most common reasons models fail in production. A model might achieve 95% accuracy in testing but only 60% in production because the leakage gave it an unfair advantage. This wastes engineering time, computational resources, and damages trust in ML systems.`,
    keyPoints: [
      'Data leakage occurs when test set information influences training, even indirectly',
      'Scaling/normalizing before splitting leaks test set statistics into training',
      'Feature selection on entire dataset chooses features that work well on test data',
      'Time series leakage uses future information not available at prediction time',
      'Always split data BEFORE any preprocessing or feature engineering',
      'Use sklearn Pipeline to automatically prevent leakage',
    ],
  },
  {
    id: 'train-test-split-validation-dq-2',
    question:
      'Compare and contrast random splitting, stratified splitting, and sequential splitting for train-test splits. For each method, describe when it should be used and what problems it solves.',
    sampleAnswer: `Different data splitting strategies are designed for different data types and problem structures. Choosing the wrong strategy can lead to biased evaluation or unrealistic performance estimates.

**1. Random Splitting**

**How It Works:**
Randomly shuffles data and splits it into train/test sets, with each sample having equal probability of being in either set.

\`\`\`python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)
\`\`\`

**When to Use:**
- Regression problems with continuous targets
- Classification with balanced classes
- IID (independent and identically distributed) data
- Most general machine learning problems

**Advantages:**
- Simple and straightforward
- Works well when data has no special structure
- Ensures training and test sets are representative samples

**Problems It Solves:**
- Prevents temporal bias (for non-time-series data)
- Ensures independence between train and test sets

**Limitations:**
- May not preserve class proportions in classification
- Destroys temporal ordering (bad for time series)
- Can create unrepresentative splits by chance

**2. Stratified Splitting**

**How It Works:**
Ensures each split has the same proportion of each class as the original dataset.

\`\`\`python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
\`\`\`

**When to Use:**
- Classification problems (binary or multi-class)
- Imbalanced datasets (e.g., fraud detection: 99% non-fraud, 1% fraud)
- When maintaining class distribution is critical
- Small datasets where random chance could create bad splits

**Advantages:**
- Guarantees representative class distribution in both sets
- Prevents "unlucky" splits where minority class is mostly in one set
- More reliable performance estimates for imbalanced data

**Problems It Solves:**
- **Imbalanced classes**: Imagine 100 samples with 90 class 0 and 10 class 1. Random split might put 8 class 1 in train and only 2 in test, making test set unreliable.
- **Small datasets**: With only 50 samples, random splitting could accidentally put all rare class examples in training.

**Example of Problem Stratification Solves:**
\`\`\`python
# Dataset: 1000 samples, 95% class 0, 5% class 1
# Random split might give:
#   Train: 96% class 0, 4% class 1
#   Test: 92% class 0, 8% class 1  # Unrepresentative!
# Stratified split guarantees:
#   Train: 95% class 0, 5% class 1
#   Test: 95% class 0, 5% class 1  # Representative!
\`\`\`

**Limitations:**
- Only works for classification (need discrete classes to stratify)
- Doesn't handle time series

**3. Sequential Splitting**

**How It Works:**
Splits data in time order without shuffling, using earlier data for training and later data for testing.

\`\`\`python
# For time series, split chronologically
train_size = int(0.8 * len(df))
train_data = df[:train_size]
test_data = df[train_size:]
\`\`\`

**When to Use:**
- Time series forecasting (stock prices, weather, sales)
- Any data with temporal dependencies
- When future data should not influence past predictions
- When deployment scenario involves predicting future from past

**Advantages:**
- Preserves temporal structure
- Realistic evaluation (predicting future from past)
- Prevents "look-ahead bias"

**Problems It Solves:**
- **Temporal leakage**: Random splitting would mix past and future, letting model learn from future to predict past
- **Non-stationary data**: Data distribution changes over time; sequential split tests model's ability to adapt
- **Production realism**: In deployment, you always predict future from past, never the reverse

**Example of Problem Sequential Splitting Solves:**
\`\`\`python
# Stock prices: 2020-2023 data
# WRONG: Random split
#   Train: Mix of 2020, 2021, 2022, 2023
#   Test: Mix of 2020, 2021, 2022, 2023
#   Model can use 2023 data to "predict" 2020! Unrealistic.

# CORRECT: Sequential split
#   Train: 2020-2022
#   Test: 2023
#   Model predicts 2023 using only past data. Realistic.
\`\`\`

**Limitations:**
- Training set doesn't see recent patterns
- Test set may encounter distribution shift
- Smaller effective training set

**Decision Matrix:**

| Data Type | Best Strategy | Reason |
|-----------|---------------|---------|
| Regression, balanced | Random | No special structure to preserve |
| Classification, imbalanced | Stratified | Maintain class proportions |
| Time series | Sequential | Preserve temporal order |
| Classification + Time series | Sequential + Stratified | Use time-based CV with stratification |

**Key Principle:** Match your splitting strategy to your data structure and deployment scenario. The goal is to make your test set as representative as possible of the data your model will face in production.`,
    keyPoints: [
      'Random splitting: Default for IID data, simple but may not preserve class balance',
      'Stratified splitting: Maintains class proportions, essential for imbalanced classification',
      'Sequential splitting: Preserves time order, critical for time series to avoid look-ahead bias',
      'Stratified prevents unlucky splits where minority classes are poorly represented',
      'Sequential ensures training only uses past data to predict future',
      'Choice depends on data structure: random for general cases, stratified for classification, sequential for time series',
    ],
  },
  {
    id: 'train-test-split-validation-dq-3',
    question:
      'In practice, machine learning projects use a three-way split: training, validation, and test sets. Explain the role of each set, when it should be used, and why using the test set multiple times invalidates its purpose.',
    sampleAnswer: `The three-way split is the gold standard for machine learning projects, with each set serving a distinct purpose in the model development lifecycle.

**1. Training Set (60-80% of data)**

**Purpose:** Fit the model parameters/weights

**When to Use:** 
- During model training
- Every time you train or retrain a model
- As frequently as needed

**What It Does:**
- Learns patterns in data
- Adjusts model weights via gradient descent
- Fits coefficients in linear models
- Builds decision boundaries in classifiers

**Example:**
\`\`\`python
# Can use training set as many times as needed
model1 = RandomForestRegressor()
model1.fit(X_train, y_train)  # Use training set

model2 = XGBoost()
model2.fit(X_train, y_train)  # Use training set again

model3 = NeuralNetwork()
model3.fit(X_train, y_train)  # Use training set again
\`\`\`

**2. Validation Set (10-20% of data)**

**Purpose:** Tune hyperparameters and select models

**When to Use:**
- After training each model configuration
- During hyperparameter optimization
- For comparing different model architectures
- For early stopping in deep learning

**What It Does:**
- Evaluates different hyperparameter settings
- Helps choose between different models
- Prevents overfitting to training data
- Guides model selection process

**Example:**
\`\`\`python
# Try different hyperparameters, evaluate on validation set
results = []
for depth in [5, 10, 15, 20]:
    model = RandomForestRegressor(max_depth=depth)
    model.fit(X_train, y_train)
    val_score = model.score(X_val, y_val)  # Evaluate on validation
    results.append((depth, val_score))

# Choose best based on validation performance
best_depth = max(results, key=lambda x: x[1])[0]
print(f"Best depth from validation: {best_depth}")
\`\`\`

**Why Validation Set Doesn't Invalidate Estimates:**
You can use the validation set many times because you're explicitly using it for model selection. The key is to acknowledge that your final performance estimate will be slightly optimistic because you optimized for this validation set. This is why we need a test set.

**3. Test Set (10-20% of data)**

**Purpose:** Provide unbiased estimate of model performance on unseen data

**When to Use:**
- **ONCE** at the very end
- After all model selection and hyperparameter tuning
- When you're ready to report final results
- Before deploying to production

**What It Does:**
- Simulates model performance on real-world data
- Provides unbiased performance estimate
- Validates that model will generalize

**Example:**
\`\`\`python
# After selecting best model using validation set
final_model = RandomForestRegressor(max_depth=best_depth)
final_model.fit(X_train, y_train)  # Train on training set

# Use test set ONCE at the end
test_score = final_model.score(X_test, y_test)
print(f"Final test score (use only once!): {test_score}")
\`\`\`

**Why Multiple Test Set Uses Invalidate Its Purpose:**

**The Problem:**
Every time you evaluate on the test set and make a decision based on those results, you're implicitly "training" on the test set. You're using test set performance to guide your decisions, which means you're optimizing for the test set.

**Concrete Example:**
\`\`\`python
# WRONG: Using test set multiple times
model1 = RandomForestRegressor(max_depth=10)
model1.fit(X_train, y_train)
test_score1 = model1.score(X_test, y_test)  # Look at test set

# "Hmm, test score is only 0.80, let me try different depth"
model2 = RandomForestRegressor(max_depth=20)
model2.fit(X_train, y_train)
test_score2 = model2.score(X_test, y_test)  # Look at test set again

# "Better! Let me try more"
model3 = RandomForestRegressor(max_depth=15)
model3.fit(X_train, y_train)
test_score3 = model3.score(X_test, y_test)  # Look at test set AGAIN

# Problem: You've now implicitly optimized for the test set!
# Your final test_score3 is no longer an unbiased estimate
\`\`\`

**Why This Invalidates the Test Set:**

1. **Indirect Overfitting**: By trying different models and choosing based on test set performance, you've selected the model that happens to work best on that specific test set. This model might not work as well on truly new data.

2. **Multiple Hypothesis Testing**: Each time you evaluate on the test set, you're performing a hypothesis test. Multiple tests increase the probability of finding a good result by chance (p-hacking in statistics).

3. **Loss of Unbiased Estimate**: The test set was meant to simulate unseen data. Once you've used it to make decisions, it's no longer "unseen"—you've incorporated information about it into your model selection process.

4. **Optimistic Bias**: The more times you look at the test set, the more likely you are to find a configuration that works well on it by chance, leading to inflated performance estimates.

**Proper Workflow:**

\`\`\`python
# Step 1: Split data
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2)

# Step 2: Experiment with training and validation sets
for depth in [5, 10, 15, 20, 25]:
    model = RandomForestRegressor(max_depth=depth)
    model.fit(X_train, y_train)
    val_score = model.score(X_val, y_val)  # Use validation repeatedly
    print(f"Depth {depth}: Val score = {val_score}")

# Step 3: Select best model based on validation
best_depth = 15  # Based on validation results

# Step 4: Retrain on combined train+validation (optional but common)
X_combined = np.vstack([X_train, X_val])
y_combined = np.concatenate([y_train, y_val])
final_model = RandomForestRegressor(max_depth=best_depth)
final_model.fit(X_combined, y_combined)

# Step 5: Evaluate on test set ONCE
test_score = final_model.score(X_test, y_test)
print(f"Final test score: {test_score}")  # Use only once!

# DON'T go back and try different models based on test_score!
\`\`\`

**Alternative When You Need Multiple Evaluations:**

If you truly need to evaluate multiple times, you should:
1. Use cross-validation instead of a single validation set
2. Hold out a second test set (nested holdout)
3. Use techniques like bootstrapping for uncertainty estimates

**Key Principle:**
The test set represents the real world. Once you've deployed your model, you can't go back and retrain based on production results (at least not immediately). Your test set evaluation should mirror this constraint—look at it once, report results, and don't go back to tune based on those results.

**Real-World Analogy:**
Think of the test set like a final exam:
- Training set = homework (practice as much as you want)
- Validation set = practice exams (take many to see what you need to study)
- Test set = final exam (take it once, get your grade, can't retake)

If students could take the final exam multiple times and use those results to study more, the final exam score would no longer accurately reflect their knowledge—it would be inflated by optimization for that specific exam.`,
    keyPoints: [
      'Training set (60-80%): Fit model parameters, can use repeatedly',
      'Validation set (10-20%): Tune hyperparameters and select models, can use many times',
      'Test set (10-20%): Unbiased performance estimate, use ONCE at the end',
      'Using test set multiple times creates indirect overfitting through model selection',
      'Each test set evaluation is a hypothesis test; multiple tests lead to p-hacking',
      'Test set should simulate unseen production data; peeking invalidates this simulation',
      'Proper workflow: experiment with train/validation, then evaluate once on test',
    ],
  },
];
