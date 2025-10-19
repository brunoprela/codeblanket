import { QuizQuestion } from '../../../types';

export const modelDebuggingQuiz: QuizQuestion[] = [
  {
    id: 'model-debugging-dq-1',
    question:
      'Your model achieves 90% accuracy, which seems good, but it performs worse than a baseline that always predicts the majority class (92% accuracy). What does this indicate, and what steps would you take to debug this?',
    sampleAnswer: `A model performing worse than majority-class baseline indicates a serious problem—the model is not learning meaningful patterns, or is even learning harmful patterns.

**What This Indicates:**

1. **Severe Underfitting:**
   - Model is too simple to capture even basic patterns
   - Essentially performing random guessing or worse

2. **Data Quality Issues:**
   - Features have no predictive power
   - Labels might be incorrect or noisy
   - Features and target are disconnected

3. **Implementation Bug:**
   - Incorrect data preprocessing
   - Target leakage in reverse (using target incorrectly)
   - Training/test split issue

4. **Extreme Class Imbalance:**
   - If dataset is 92% class 0, 8% class 1
   - Model might be optimizing for wrong metric
   - Always predicting class 0 gives 92% accuracy
   - Your model at 90% is actually performing worse

**Debugging Steps:**

**Step 1: Verify the Baseline**
\`\`\`python
from sklearn.dummy import DummyClassifier

# Confirm baseline
baseline = DummyClassifier(strategy='most_frequent')
baseline.fit(X_train, y_train)
baseline_acc = baseline.score(X_test, y_test)  # 92%

# Check class distribution
print(np.bincount(y_test))  # e.g., [184, 16] = 92% class 0
\`\`\`

**Step 2: Check Class Distribution**
- If 92% is class 0, accuracy is misleading
- Use precision, recall, F1, AUC instead
- Model might predict class 0 slightly less often than it should

**Step 3: Examine Confusion Matrix**
\`\`\`python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Example output:
# [[170  14]  ← 170 correct class 0, 14 wrong
#  [ 6   10]]  ← 6 missed class 1, 10 correct
\`\`\`
- Model is catching some class 1 (good)
- But making many false positives on class 0 (bad)
- Net effect: worse than always predicting class 0

**Step 4: Check Training Performance**
\`\`\`python
train_acc = model.score(X_train, y_train)
print(f"Train: {train_acc:.3f}, Test: {test_acc:.3f}")
\`\`\`
- If train_acc is also low (~90%), model has high bias
- If train_acc is high (>95%), model has high variance

**Step 5: Verify Data Pipeline**
\`\`\`python
# Check features aren't constant
print(X_train.std(axis=0))  # Should not be 0

# Check for data leakage
print(X_train.columns)  # Look for suspicious features

# Verify label distribution
print(pd.Series(y_train).value_counts())
\`\`\`

**Step 6: Try Simpler Model**
\`\`\`python
from sklearn.linear_model import LogisticRegression

# If complex model fails, try simple model
simple_model = LogisticRegression()
simple_model.fit(X_train, y_train)
simple_acc = simple_model.score(X_test, y_test)
\`\`\`
- If simple model also fails → data problem
- If simple model works → complexity/hyperparameter issue

**Step 7: Use Better Metrics**
\`\`\`python
from sklearn.metrics import classification_report, roc_auc_score

print(classification_report(y_test, y_pred))
auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC: {auc:.3f}")
\`\`\`
- If AUC > 0.5, model has some predictive power
- If AUC ≈ 0.5, model is random guessing
- If AUC < 0.5, model is learning inverted relationship!

**Solutions Based on Root Cause:**

1. **If High Bias:**
   - Increase model complexity
   - Add more features
   - Reduce regularization

2. **If Data Quality Issue:**
   - Verify labels are correct
   - Check feature engineering
   - Remove noisy features
   - Get more/better data

3. **If Class Imbalance:**
   - Use class weights
   - Change threshold
   - Use F1 or AUC instead of accuracy
   - Resample data (SMOTE, undersampling)

4. **If Implementation Bug:**
   - Verify preprocessing (scaling, encoding)
   - Check train/test split
   - Ensure no data leakage
   - Verify model is training correctly

**Key Insight:** Never trust a single metric. Always compare to baselines and use multiple metrics, especially with imbalanced data.`,
    keyPoints: [
      'Model worse than baseline indicates serious problem',
      'Check class distribution first—imbalance makes accuracy misleading',
      'Examine confusion matrix to understand error types',
      'Compare training vs test accuracy to diagnose bias/variance',
      'Verify data pipeline and check for implementation bugs',
      'Use multiple metrics (precision, recall, F1, AUC)',
      'Try simpler model to isolate whether issue is data or model',
    ],
  },
  {
    id: 'model-debugging-dq-2',
    question:
      'Your model performs well in offline evaluation (AUC=0.90) but poorly in production (AUC=0.70). What could cause this gap, and how would you debug it?',
    sampleAnswer: `A large offline-online performance gap indicates distribution shift, data leakage, or improper evaluation. This is one of the most common and serious ML production issues.

**Possible Causes:**

**1. Data Leakage**
- Training data included information not available at prediction time
- Example: Used future data to create features
- Result: Overly optimistic offline performance

**2. Train-Test Distribution Shift**
- Training data doesn't represent production data
- Example: Trained on last year's data, deploying this year
- Example: Different user demographics, seasonality, market conditions

**3. Temporal Issues**
- Trained on old data, deployed on new data
- Model hasn't seen recent trends/patterns
- Data drift over time

**4. Improper Evaluation**
- Train/test split didn't preserve temporal order
- Random split leaked future information
- Test set not representative of production

**5. Feature Computation Differences**
- Offline: Computed features with full dataset statistics
- Online: Features computed differently or with limited data
- Example: Mean encoding using full dataset vs real-time

**6. Label Definition Mismatch**
- Offline labels don't match production reality
- Example: Predicting 30-day churn but labels are 7-day churn

**Debugging Steps:**

**Step 1: Check for Data Leakage**
\`\`\`python
# Review feature engineering pipeline
# Look for features that use future information
suspicious_features = [
    'future_purchase',  # ❌ Uses future data
    'total_lifetime_value',  # ❌ Includes future
    'next_month_activity',  # ❌ Future data
]

# Check feature importance
# Are there surprisingly predictive features?
# High importance for dates, IDs, or auxiliary data?
\`\`\`

**Step 2: Temporal Validation**
\`\`\`python
# Split data by time (not randomly)
split_date = '2024-01-01'
train = df[df['date'] < split_date]
test = df[df['date'] >= split_date]

# Train and evaluate
model.fit(X_train, y_train)
temporal_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"Temporal validation AUC: {temporal_auc:.3f}")
\`\`\`
- If temporal AUC matches production → validation issue fixed
- If temporal AUC still high → other issues exist

**Step 3: Compare Feature Distributions**
\`\`\`python
import pandas as pd

# Get production data sample
prod_sample = get_production_data(n=1000)

# Compare distributions
for feature in X_train.columns:
    train_mean = X_train[feature].mean()
    prod_mean = prod_sample[feature].mean()
    diff = abs(train_mean - prod_mean) / train_mean
    
    if diff > 0.2:  # >20% change
        print(f"⚠️  {feature}: train={train_mean:.2f}, prod={prod_mean:.2f} ({diff*100:.0f}% diff)")
\`\`\`

**Step 4: Check Feature Availability**
\`\`\`python
# Verify all features are available at prediction time
# Check for missing features in production
prod_features = set(prod_sample.columns)
train_features = set(X_train.columns)

missing = train_features - prod_features
if missing:
    print(f"⚠️  Missing in production: {missing}")
    
extra = prod_features - train_features
if extra:
    print(f"ℹ️  Extra in production: {extra}")
\`\`\`

**Step 5: Monitor Predictions**
\`\`\`python
# Check prediction distribution
train_pred_mean = model.predict_proba(X_train)[:, 1].mean()
prod_pred_mean = model.predict_proba(prod_sample)[:, 1].mean()

print(f"Train predictions: {train_pred_mean:.3f}")
print(f"Prod predictions: {prod_pred_mean:.3f}")

# Large difference indicates distribution shift
\`\`\`

**Step 6: Evaluate on Recent Data**
\`\`\`python
# If model was trained on old data
# Evaluate on most recent available data
recent_data = df[df['date'] >= '2024-10-01']
recent_auc = roc_auc_score(recent_data['target'], 
                            model.predict_proba(recent_data[features])[:, 1])
print(f"Recent data AUC: {recent_auc:.3f}")
\`\`\`

**Step 7: A/B Test with Simple Baseline**
\`\`\`python
# Deploy simple baseline alongside complex model
# If baseline performs similarly → complex model isn't helping
# If baseline performs better → complex model is overfitting
\`\`\`

**Solutions:**

**For Data Leakage:**
- Remove features with future information
- Careful feature engineering with temporal awareness
- Use temporal cross-validation

**For Distribution Shift:**
- Retrain on more recent data
- Implement continuous retraining
- Use online learning or model updating
- Add monitoring for drift

**For Feature Issues:**
- Ensure offline and online feature computation match
- Document feature computation pipeline
- Test feature pipeline end-to-end

**For Temporal Issues:**
- Use time-based train/test split
- Implement rolling window validation
- Retrain regularly (daily/weekly/monthly)

**Prevention:**
- Always use temporal validation for time-series problems
- Monitor feature and prediction distributions in production
- Implement drift detection
- A/B test new models before full deployment
- Document all features and their computation
- Regularly retrain models

**Key Insight:** Offline-online gaps usually indicate that offline evaluation didn't match production conditions. Always validate as close to production as possible.`,
    keyPoints: [
      'Offline-online gap indicates leakage, drift, or evaluation issues',
      'Check for data leakage first (future information in features)',
      'Use temporal validation (time-based split, not random)',
      'Compare feature distributions between train and production',
      'Verify feature computation matches between offline and online',
      'Implement monitoring for distribution drift',
      'Retrain regularly to adapt to changing patterns',
    ],
  },
  {
    id: 'model-debugging-dq-3',
    question:
      'Describe a systematic approach to debugging a model with high training accuracy (98%) but low test accuracy (70%). Include specific diagnostic steps and visualizations you would create.',
    sampleAnswer: `High training accuracy with low test accuracy is a classic overfitting problem. Here's a systematic debugging approach:

**Diagnosis: High Variance (Overfitting)**
- Training: 98% → Model memorized training data
- Test: 70% → Doesn't generalize to new data
- Gap: 28% → Severe overfitting

**Systematic Debugging Approach:**

**Step 1: Visualize Learning Curves**
\`\`\`python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X_train, y_train, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy'
)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores.mean(axis=1), label='Training', marker='o')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation', marker='o')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curves')
plt.legend()
plt.grid(True)
\`\`\`

**What to Look For:**
- Large gap between curves → overfitting confirmed
- Both curves plateau → unlikely to improve with more data
- Validation curve still rising → more data would help

**Step 2: Analyze Model Complexity**
\`\`\`python
# For tree models: check depth
if hasattr(model, 'max_depth'):
    print(f"Max depth: {model.max_depth}")
    print(f"Actual depth: {max([tree.max_depth for tree in model.estimators_])}")

# For neural networks: check architecture
if hasattr(model, 'n_layers_'):
    print(f"Layers: {model.n_layers_}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

# Compare to data size
print(f"Training samples: {len(X_train)}")
print(f"Features: {X_train.shape[1]}")
print(f"Samples per parameter: {len(X_train) / num_parameters}")

# Rule of thumb: want >10 samples per parameter
\`\`\`

**Step 3: Check Feature Importance for Overfitting Signs**
\`\`\`python
importances = model.feature_importances_
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print(feature_importance.head(10))

# Red flags:
# - ID columns have high importance
# - Single feature dominates (>80% importance)
# - Many zero-importance features
\`\`\`

**Step 4: Visualize Decision Boundaries (if possible)**
\`\`\`python
# For 2D visualization, select top 2 features
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)

# Plot decision boundary
h = 0.02
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, alpha=0.8)
plt.title('Decision Boundary')

# Look for: overly complex, jagged boundaries = overfitting
\`\`\`

**Step 5: Cross-Validation Analysis**
\`\`\`python
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')

print(f"CV mean: {cv_scores.mean():.3f}")
print(f"CV std: {cv_scores.std():.3f}")
print(f"CV scores: {cv_scores}")

# High variance in CV scores → model is unstable
if cv_scores.std() > 0.05:
    print("⚠️  High variance in CV scores - model is unstable")
\`\`\`

**Step 6: Regularization Experiment**
\`\`\`python
# Try different regularization strengths
regularization_values = [0.001, 0.01, 0.1, 1.0, 10.0]
results = []

for C in regularization_values:
    model_reg = LogisticRegression(C=C, max_iter=1000)
    model_reg.fit(X_train, y_train)
    
    train_acc = model_reg.score(X_train, y_train)
    test_acc = model_reg.score(X_test, y_test)
    gap = train_acc - test_acc
    
    results.append({'C': C, 'train': train_acc, 'test': test_acc, 'gap': gap})

results_df = pd.DataFrame(results)
print(results_df)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(results_df['C'], results_df['train'], label='Train', marker='o')
plt.plot(results_df['C'], results_df['test'], label='Test', marker='o')
plt.xscale('log')
plt.xlabel('Regularization (C)')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Effect of Regularization')
\`\`\`

**Step 7: Feature Selection Experiment**
\`\`\`python
from sklearn.feature_selection import SelectKBest, f_classif

# Try reducing features
n_features_list = [5, 10, 20, 50, X_train.shape[1]]
results = []

for n_features in n_features_list:
    selector = SelectKBest(f_classif, k=min(n_features, X_train.shape[1]))
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    model_temp = RandomForestClassifier(random_state=42)
    model_temp.fit(X_train_selected, y_train)
    
    train_acc = model_temp.score(X_train_selected, y_train)
    test_acc = model_temp.score(X_test_selected, y_test)
    
    results.append({'n_features': n_features, 'train': train_acc, 'test': test_acc})

# Lower n_features might reduce overfitting
\`\`\`

**Solutions (in order of priority):**

1. **Add Regularization:**
   - L1/L2 for linear models
   - Dropout for neural networks
   - Max depth, min_samples_split for trees

2. **Get More Training Data:**
   - If possible, collect more samples
   - Data augmentation
   - Synthetic data generation

3. **Reduce Model Complexity:**
   - Fewer layers, smaller hidden size (NN)
   - Shallower trees, fewer estimators
   - Feature selection

4. **Early Stopping:**
   - Stop training when validation performance plateaus
   - Prevents overfitting to training data

5. **Ensemble Methods:**
   - Bagging reduces variance
   - Use cross-validation predictions

6. **Feature Engineering:**
   - Remove noisy features
   - Add domain knowledge features
   - Reduce feature interactions

**Key Visualizations to Create:**
1. Learning curves (train vs validation accuracy over training set size)
2. Training history (train vs validation loss/accuracy over epochs)
3. Feature importance bar chart
4. Confusion matrix on test set
5. ROC curve comparison (train vs test)
6. Regularization sweep plot

**Expected Outcome:**
After applying regularization and reducing complexity:
- Training: 85% (down from 98% - good!)
- Test: 82% (up from 70% - good!)
- Gap: 3% (down from 28% - good!)`,
    keyPoints: [
      'Large train-test gap indicates overfitting (high variance)',
      'Learning curves show if more data would help',
      'Analyze model complexity relative to dataset size',
      'Check feature importance for data leakage signs',
      'Cross-validation shows model stability',
      'Experiment with regularization to reduce overfitting',
      'Solutions: regularization, more data, reduce complexity, early stopping',
    ],
  },
];
