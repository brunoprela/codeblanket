/**
 * Machine Learning Fundamentals Section
 */

export const mlfundamentalsSection = {
  id: 'ml-fundamentals',
  title: 'Machine Learning Fundamentals',
  content: `# Machine Learning Fundamentals

## Introduction

Machine Learning (ML) is the field of study that gives computers the ability to learn from data without being explicitly programmed. Instead of writing rules manually, ML algorithms discover patterns and make predictions based on examples. This fundamental shift in programming paradigm has revolutionized fields from computer vision to natural language processing to quantitative trading.

**Why Machine Learning Matters:**
- **Automation**: Handle tasks too complex to code manually
- **Adaptation**: Systems that improve with more data
- **Pattern Discovery**: Find hidden insights in large datasets
- **Prediction**: Forecast future outcomes based on historical patterns
- **Scalability**: Process massive amounts of data efficiently

## What is Machine Learning?

Machine Learning is a subset of Artificial Intelligence that focuses on developing algorithms that can:

1. **Learn from data**: Extract patterns and relationships
2. **Generalize**: Make accurate predictions on unseen data
3. **Improve with experience**: Performance increases with more data

**Traditional Programming vs. Machine Learning:**

\`\`\`
Traditional Programming:
Data + Rules → Computer → Output

Machine Learning:
Data + Output → Computer → Rules (Model)
\`\`\`

### Example: Email Spam Detection

**Traditional approach**: Write rules like "if email contains 'free money' or 'click here', mark as spam"
- Hard to maintain
- Misses new spam patterns
- Many false positives

**ML approach**: Show examples of spam and non-spam emails, let the algorithm learn patterns
- Adapts to new spam types
- Learns complex patterns
- Improves with more data

\`\`\`python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Simple spam detection example
emails = [
    "Free money! Click here now!",
    "Meeting at 3pm tomorrow",
    "You won a lottery! Claim now!",
    "Can you review the report?",
    "Get rich quick scheme",
    "Project deadline next week"
]

labels = [1, 0, 1, 0, 1, 0]  # 1 = spam, 0 = not spam

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform (emails)

# Train model
model = MultinomialNB()
model.fit(X, labels)

# Test on new email
new_email = ["Free cash waiting for you!"]
X_new = vectorizer.transform (new_email)
prediction = model.predict(X_new)

print(f"Prediction: {'Spam' if prediction[0] == 1 else 'Not Spam'}")
# Output: Prediction: Spam
\`\`\`

## Types of Machine Learning

### 1. Supervised Learning

**Definition**: Learning from labeled data - we know the correct answers.

**Input**: Features (X) and Labels (y)
**Goal**: Learn a function f(X) → y that maps inputs to outputs

**Examples**:
- **Classification**: Predict discrete categories (spam/not spam, cat/dog)
- **Regression**: Predict continuous values (house prices, stock prices)

**Real-world applications**:
- Email spam detection
- Credit card fraud detection
- House price prediction
- Medical diagnosis
- Stock price forecasting

\`\`\`python
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

# Generate sample data
X, y = make_regression (n_samples=100, n_features=1, noise=10, random_state=42)

# Train supervised learning model
model = LinearRegression()
model.fit(X, y)

# Make predictions
predictions = model.predict(X[:5])
print("Actual values:", y[:5])
print("Predictions:", predictions)
\`\`\`

### 2. Unsupervised Learning

**Definition**: Learning from unlabeled data - no correct answers provided.

**Input**: Only features (X), no labels
**Goal**: Discover hidden patterns and structures

**Examples**:
- **Clustering**: Group similar data points (customer segmentation)
- **Dimensionality Reduction**: Reduce number of features (PCA)
- **Anomaly Detection**: Find outliers (fraud detection)

**Real-world applications**:
- Customer segmentation for marketing
- Gene expression analysis
- Recommendation systems
- Anomaly detection in network traffic

\`\`\`python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.randn(300, 2)
X[:100] += [2, 2]
X[100:200] += [-2, -2]

# Clustering (unsupervised)
kmeans = KMeans (n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

print(f"Found {len (set (clusters))} clusters")
# Output: Found 3 clusters
\`\`\`

### 3. Reinforcement Learning

**Definition**: Learning through trial and error with rewards and penalties.

**Components**:
- **Agent**: The learner/decision maker
- **Environment**: What the agent interacts with
- **Actions**: What the agent can do
- **Rewards**: Feedback from the environment

**Examples**:
- Game playing (AlphaGo, Chess)
- Robotics
- Autonomous vehicles
- Algorithmic trading

## The Machine Learning Workflow

### Step 1: Problem Definition

**Key Questions**:
- What are we trying to predict?
- What data do we have?
- How will success be measured?
- What\'s the baseline performance?

**Example**: Predict house prices
- Target: Price (regression problem)
- Features: Size, bedrooms, location, age
- Metric: Mean Absolute Error
- Baseline: Average price

### Step 2: Data Collection

**Data Sources**:
- Databases
- APIs
- Web scraping
- Sensors
- Manual labeling

**Data Quality Considerations**:
- Sufficient quantity
- Relevant features
- Correct labels
- Representative samples

\`\`\`python
import pandas as pd

# Example: Loading data
# In practice, data comes from various sources
data = pd.DataFrame({
    'size_sqft': [1500, 2000, 1200, 2400, 1800],
    'bedrooms': [3, 4, 2, 4, 3],
    'age_years': [10, 5, 20, 2, 8],
    'price': [300000, 400000, 250000, 500000, 350000]
})

print(data.head())
print(f"\\nDataset shape: {data.shape}")
print(f"Features: {list (data.columns[:-1])}")
print(f"Target: {data.columns[-1]}")
\`\`\`

### Step 3: Data Preparation

**Common Tasks**:
- Handle missing values
- Remove duplicates
- Feature engineering
- Data splitting (train/validation/test)
- Feature scaling

\`\`\`python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Prepare data
X = data[['size_sqft', 'bedrooms', 'age_years']]
y = data['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
\`\`\`

### Step 4: Model Selection

**Considerations**:
- Problem type (classification/regression)
- Data size
- Feature types
- Interpretability requirements
- Computational resources

**Common algorithms**:
- Linear models: Fast, interpretable
- Tree-based: Handle non-linear relationships
- Neural networks: Complex patterns, lots of data

### Step 5: Training

**Process**:
1. Initialize model with parameters
2. Feed training data to model
3. Model learns patterns
4. Adjust parameters to minimize error

\`\`\`python
from sklearn.ensemble import RandomForestRegressor

# Train model
model = RandomForestRegressor (n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Model is now trained and ready for predictions
print("Model training complete!")
\`\`\`

### Step 6: Evaluation

**Metrics depend on problem type**:

**Regression**:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score

**Classification**:
- Accuracy
- Precision, Recall, F1-Score
- ROC-AUC

\`\`\`python
from sklearn.metrics import mean_absolute_error, r2_score

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate
mae = mean_absolute_error (y_test, y_pred)
r2 = r2_score (y_test, y_pred)

print(f"Mean Absolute Error: \${mae:,.0f}")
print(f"R² Score: {r2:.3f}")
\`\`\`

### Step 7: Deployment

**Considerations**:
- Model serving (API, batch processing)
- Monitoring performance
- Retraining schedule
- Version control

## Training, Validation, and Test Sets

### Why Split Data?

**Problem**: We need to evaluate how well our model generalizes to unseen data.

**Solution**: Hold out some data that the model never sees during training.

### Three-Way Split

**Training Set (60-80%)**:
- Used to train the model
- Model learns patterns from this data

**Validation Set (10-20%)**:
- Used to tune hyperparameters
- Select best model architecture
- Prevent overfitting to training set

**Test Set (10-20%)**:
- Final evaluation only
- Never used during training or tuning
- Represents real-world performance

\`\`\`python
from sklearn.model_selection import train_test_split

# Generate sample data
np.random.seed(42)
X = np.random.randn(1000, 5)
y = X[:, 0] + 2 * X[:, 1] + np.random.randn(1000) * 0.1

# First split: separate test set
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Second split: separate validation from training
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2
)

print(f"Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.0f}%)")
print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.0f}%)")
print(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.0f}%)")

# Output:
# Training set: 600 samples (60%)
# Validation set: 200 samples (20%)
# Test set: 200 samples (20%)
\`\`\`

### Common Mistake: Data Leakage

**Data leakage** occurs when information from the test set influences the training process.

**Examples of leakage**:
- Scaling on full dataset before splitting
- Feature selection using full dataset
- Looking at test set to tune hyperparameters

**Correct approach**:
\`\`\`python
# WRONG: Scaling before split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Uses test set info!
X_train, X_test = train_test_split(X_scaled, y)

# CORRECT: Scale after split
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Learn from train only
X_test_scaled = scaler.transform(X_test)  # Apply to test
\`\`\`

## Overfitting and Underfitting

### Underfitting (High Bias)

**Definition**: Model is too simple to capture the underlying patterns.

**Symptoms**:
- Poor performance on training data
- Poor performance on test data
- Model predictions are inaccurate

**Example**: Using a straight line to fit circular data

**Solutions**:
- Use more complex model
- Add more features
- Reduce regularization
- Train longer

\`\`\`python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Generate non-linear data
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 0.5 * X.ravel()**2 + np.random.randn(100) * 5

# Underfitting: Linear model on non-linear data
model_simple = LinearRegression()
model_simple.fit(X, y)

# Training error
train_pred = model_simple.predict(X)
train_error = np.mean((y - train_pred)**2)

print(f"Underfitting model training MSE: {train_error:.2f}")
# Output: High error (model too simple)
\`\`\`

### Overfitting (High Variance)

**Definition**: Model is too complex and learns noise in the training data.

**Symptoms**:
- Excellent performance on training data
- Poor performance on test data
- Model memorizes rather than generalizes

**Example**: Using a very high-degree polynomial

**Solutions**:
- Use simpler model
- Get more training data
- Use regularization
- Feature selection
- Early stopping
- Data augmentation

\`\`\`python
# Overfitting: Polynomial degree too high
poly_features = PolynomialFeatures (degree=15)
X_poly = poly_features.fit_transform(X)

model_complex = LinearRegression()
model_complex.fit(X_poly, y)

# Training error (very low - model memorizes)
train_pred_complex = model_complex.predict(X_poly)
train_error_complex = np.mean((y - train_pred_complex)**2)

print(f"Overfitting model training MSE: {train_error_complex:.2f}")
# Output: Very low error on training, but poor on new data
\`\`\`

### The Sweet Spot: Good Fit

**Goal**: Balance between bias and variance

**Characteristics**:
- Good performance on training data
- Good performance on test data
- Generalizes well to new data

**Visual Representation**:
\`\`\`
Model Complexity →

Training Error:    \\_____
                      ↓ decreases

Validation Error:  \\    /
                    ↓  ↑
                   Sweet spot

Underfitting ← | → Overfitting
\`\`\`

## Bias-Variance Tradeoff

### Understanding the Tradeoff

**Total Error = Bias² + Variance + Irreducible Error**

**Bias**:
- Error from wrong assumptions
- Underfitting
- High bias = model too simple

**Variance**:
- Error from sensitivity to training data fluctuations
- Overfitting
- High variance = model too complex

**Irreducible Error**:
- Noise in the data
- Cannot be reduced

\`\`\`python
# Demonstrating bias-variance tradeoff
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Generate data
np.random.seed(42)
X_train = np.linspace(0, 10, 50).reshape(-1, 1)
y_train = np.sin(X_train).ravel() + np.random.randn(50) * 0.1

X_test = np.linspace(0, 10, 20).reshape(-1, 1)
y_test = np.sin(X_test).ravel() + np.random.randn(20) * 0.1

# Models with different complexity
depths = [1, 3, 5, 10, 20]
results = []

for depth in depths:
    model = DecisionTreeRegressor (max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    
    train_error = mean_squared_error (y_train, model.predict(X_train))
    test_error = mean_squared_error (y_test, model.predict(X_test))
    
    results.append({
        'depth': depth,
        'train_error': train_error,
        'test_error': test_error
    })
    
for r in results:
    print(f"Depth {r['depth']}: Train MSE={r['train_error']:.4f}, Test MSE={r['test_error']:.4f}")

# Output shows bias-variance tradeoff:
# Low depth: High train error (bias), moderate test error
# Medium depth: Low train error, low test error (sweet spot)
# High depth: Very low train error, high test error (variance)
\`\`\`

## Generalization

**Generalization** is the ability of a model to perform well on new, unseen data.

### Key Principles

1. **Train on representative data**: Sample should reflect real-world distribution
2. **Prevent overfitting**: Use validation set, regularization
3. **Test rigorously**: Use held-out test set
4. **Monitor in production**: Performance may degrade over time

### Factors Affecting Generalization

**Data Quality**:
- More data generally helps
- Data must be representative
- Clean, accurate labels

**Model Complexity**:
- Match complexity to problem
- Simpler models generalize better with limited data

**Regularization**:
- Penalize model complexity
- Encourage simpler solutions

\`\`\`python
# Example: Evaluating generalization
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

# Generate data
X = np.random.randn(500, 10)
y = X[:, 0] + 2 * X[:, 1] + np.random.randn(500) * 0.5

# Evaluate generalization using cross-validation
model = RandomForestRegressor (n_estimators=100, random_state=42)

# 5-fold cross-validation
scores = cross_val_score (model, X, y, cv=5, scoring='r2')

print(f"Cross-validation R² scores: {scores}")
print(f"Mean R²: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

# Consistent scores across folds indicate good generalization
\`\`\`

## Real-World ML Example: Customer Churn Prediction

Let\'s put it all together with a complete example:

\`\`\`python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Create sample customer data
np.random.seed(42)
n_customers = 1000

data = pd.DataFrame({
    'monthly_charges': np.random.uniform(20, 100, n_customers),
    'tenure_months': np.random.randint(1, 72, n_customers),
    'num_support_calls': np.random.poisson(2, n_customers),
    'contract_type': np.random.choice([0, 1, 2], n_customers),  # 0: monthly, 1: yearly, 2: 2-year
})

# Generate target: customers with high charges, short tenure, many calls more likely to churn
churn_probability = (
    0.3 * (data['monthly_charges'] / 100) +
    0.3 * (1 - data['tenure_months'] / 72) +
    0.2 * (data['num_support_calls'] / 5) +
    0.2 * (data['contract_type'] == 0)
)
data['churned'] = (np.random.random (n_customers) < churn_probability).astype (int)

print("="*50)
print("CUSTOMER CHURN PREDICTION")
print("="*50)

# Step 1: Understand the data
print(f"\\nDataset shape: {data.shape}")
print(f"Churn rate: {data['churned'].mean():.1%}")
print(f"\\nFeatures: {list (data.columns[:-1])}")

# Step 2: Split data
X = data.drop('churned', axis=1)
y = data['churned']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Step 3: Preprocess
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Train model
model = RandomForestClassifier (n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train_scaled, y_train)

# Step 5: Evaluate
train_pred = model.predict(X_train_scaled)
test_pred = model.predict(X_test_scaled)

print(f"\\nTraining Accuracy: {accuracy_score (y_train, train_pred):.3f}")
print(f"Test Accuracy: {accuracy_score (y_test, test_pred):.3f}")

# Good generalization: similar train and test accuracy

# Step 6: Analyze results
print("\\nClassification Report:")
print(classification_report (y_test, test_pred, target_names=['Retained', 'Churned']))

# Step 7: Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\\nTop Features:")
print(feature_importance)
\`\`\`

## Summary

Machine Learning is a powerful paradigm that enables computers to learn from data. Key concepts:

1. **Supervised learning**: Learn from labeled examples
2. **Unsupervised learning**: Discover patterns without labels  
3. **Reinforcement learning**: Learn through trial and error
4. **Data splitting**: Train/validation/test for proper evaluation
5. **Overfitting vs underfitting**: Balance model complexity
6. **Bias-variance tradeoff**: Balance simplicity and flexibility
7. **Generalization**: The ultimate goal - perform well on new data

Understanding these fundamentals is crucial for success in machine learning. Every algorithm and technique builds on these concepts.

## Next Steps

In the following sections, we'll dive deep into specific supervised learning algorithms:
- Linear models for regression and classification
- Tree-based methods
- Support Vector Machines
- Ensemble methods
- And many more!

Each algorithm has different strengths, and choosing the right one depends on your data, problem, and constraints.
`,
  codeExample: `# Complete Machine Learning Pipeline Example

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# 1. Generate synthetic dataset
np.random.seed(42)
n_samples = 1000

# Features
X = pd.DataFrame({
    'feature1': np.random.randn (n_samples),
    'feature2': np.random.randn (n_samples),
    'feature3': np.random.randn (n_samples),
    'feature4': np.random.randn (n_samples),
})

# Target (classification)
y = (X['feature1'] + 2*X['feature2'] - X['feature3'] + np.random.randn (n_samples)*0.5 > 0).astype (int)

print("Dataset Overview:")
print(f"Samples: {n_samples}")
print(f"Features: {X.shape[1]}")
print(f"Classes: {len (np.unique (y))}")
print(f"Class distribution: {np.bincount (y)}")

# 2. Split data (60% train, 20% validation, 20% test)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

print(f"\\nData Split:")
print(f"Training: {len(X_train)} ({len(X_train)/n_samples*100:.0f}%)")
print(f"Validation: {len(X_val)} ({len(X_val)/n_samples*100:.0f}%)")
print(f"Test: {len(X_test)} ({len(X_test)/n_samples*100:.0f}%)")

# 3. Preprocess (scale features)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 4. Train model
model = RandomForestClassifier (n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train_scaled, y_train)

# 5. Evaluate on all sets
train_acc = accuracy_score (y_train, model.predict(X_train_scaled))
val_acc = accuracy_score (y_val, model.predict(X_val_scaled))
test_acc = accuracy_score (y_test, model.predict(X_test_scaled))

print(f"\\nModel Performance:")
print(f"Training Accuracy: {train_acc:.3f}")
print(f"Validation Accuracy: {val_acc:.3f}")
print(f"Test Accuracy: {test_acc:.3f}")

# Check for overfitting
if train_acc - test_acc > 0.1:
    print("⚠️ Warning: Possible overfitting (train >> test)")
elif test_acc - train_acc > 0.05:
    print("⚠️ Warning: Unusual pattern (test > train)")
else:
    print("✓ Good generalization")

# 6. Cross-validation for robust estimate
cv_scores = cross_val_score (model, X_train_scaled, y_train, cv=5)
print(f"\\nCross-Validation Scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")

# 7. Detailed test set evaluation
y_pred = model.predict(X_test_scaled)
print("\\nTest Set Classification Report:")
print(classification_report (y_test, y_pred, target_names=['Class 0', 'Class 1']))

# 8. Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\\nFeature Importance:")
print(feature_importance)

# 9. Confusion matrix
cm = confusion_matrix (y_test, y_pred)
print("\\nConfusion Matrix:")
print(cm)
print(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")
`,
};
