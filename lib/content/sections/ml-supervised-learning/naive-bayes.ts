/**
 * Naive Bayes Section
 */

export const naivebayesSection = {
  id: 'naive-bayes',
  title: 'Naive Bayes',
  content: `# Naive Bayes

## Introduction

Naive Bayes is a family of probabilistic classifiers based on Bayes' theorem with the "naive" assumption of feature independence. Despite this strong simplifying assumption, Naive Bayes performs surprisingly well in many real-world applications, especially text classification.

**Key Characteristics:**
- Based on Bayes' theorem
- Assumes feature independence (naive assumption)
- Probabilistic predictions
- Fast training and prediction
- Works well with high-dimensional data

**Applications:**
- Text classification (spam detection, sentiment analysis)
- Document categorization
- Medical diagnosis
- Real-time prediction
- Recommender systems

## Bayes' Theorem Review

\\[ P(y|\\mathbf{x}) = \\frac{P(\\mathbf{x}|y) \\cdot P(y)}{P(\\mathbf{x})} \\]

Where:
- \\( P(y|\\mathbf{x}) \\): Posterior probability (what we want)
- \\( P(\\mathbf{x}|y) \\): Likelihood
- \\( P(y) \\): Prior probability
- \\( P(\\mathbf{x}) \\): Evidence (constant for all classes)

**For classification**, we choose the class with highest posterior:
\\[ \\hat{y} = \\arg\\max_y P(y|\\mathbf{x}) = \\arg\\max_y P(\\mathbf{x}|y) \\cdot P(y) \\]

## The Naive Assumption

**Naive Bayes assumes features are conditionally independent given the class:**

\\[ P(\\mathbf{x}|y) = P(x_1, x_2, ..., x_n|y) = \\prod_{i=1}^{n} P(x_i|y) \\]

This dramatically simplifies computation but is rarely true in practice!

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Generate data
np.random.seed(42)
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                           n_informative=2, n_clusters_per_class=1, 
                           random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)

# Predictions
y_pred = nb.predict(X_test)
y_prob = nb.predict_proba(X_test)

print("Naive Bayes Classifier")
print(f"Training Accuracy: {nb.score(X_train, y_train):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"\\nPrediction probabilities (first 5):")
print(y_prob[:5])
\`\`\`

## Types of Naive Bayes

### 1. Gaussian Naive Bayes

Assumes features follow a normal distribution:

\\[ P(x_i|y) = \\frac{1}{\\sqrt{2\\pi\\sigma_y^2}} \\exp\\left(-\\frac{(x_i - \\mu_y)^2}{2\\sigma_y^2}\\right) \\]

**Use case**: Continuous features

\`\`\`python
from sklearn.naive_bayes import GaussianNB

# Parameters learned by Gaussian NB
print("\\nClass priors:")
print(f"P(y=0) = {np.exp(nb.class_log_prior_[0]):.4f}")
print(f"P(y=1) = {np.exp(nb.class_log_prior_[1]):.4f}")

print("\\nFeature means for each class:")
print("Class 0:", nb.theta_[0])
print("Class 1:", nb.theta_[1])

print("\\nFeature variances for each class:")
print("Class 0:", nb.var_[0])
print("Class 1:", nb.var_[1])
\`\`\`

### 2. Multinomial Naive Bayes

Assumes features follow a multinomial distribution (counts/frequencies):

\\[ P(x_i|y) = \\frac{N_{yi} + \\alpha}{N_y + \\alpha n} \\]

**Use case**: Text classification (word counts)

\`\`\`python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Text classification example
texts = [
    "free money click here",
    "meeting at 3pm tomorrow",
    "claim your prize now",
    "project deadline next week",
    "you won a lottery",
    "can you review the document",
]
labels = [1, 0, 1, 0, 1, 0]  # 1=spam, 0=not spam

# Convert text to word counts
vectorizer = CountVectorizer()
X_counts = vectorizer.fit_transform(texts)

# Train Multinomial Naive Bayes
mnb = MultinomialNB()
mnb.fit(X_counts, labels)

# Test
test_texts = ["free prize", "meeting document"]
X_test_counts = vectorizer.transform(test_texts)
predictions = mnb.predict(X_test_counts)
probabilities = mnb.predict_proba(X_test_counts)

print("\\nText Classification Results:")
for text, pred, prob in zip(test_texts, predictions, probabilities):
    label = "Spam" if pred == 1 else "Not Spam"
    print(f"'{text}' -> {label} (P={prob[pred]:.3f})")

print("\\nVocabulary:", vectorizer.get_feature_names_out())
print("\\nFeature log probabilities for Spam class:")
for word, log_prob in zip(vectorizer.get_feature_names_out(), mnb.feature_log_prob_[1]):
    print(f"  {word}: {log_prob:.3f}")
\`\`\`

### 3. Bernoulli Naive Bayes

Assumes binary features (present/absent):

\\[ P(x_i|y) = P(i|y)x_i + (1 - P(i|y))(1 - x_i) \\]

**Use case**: Binary text features (word present or not)

\`\`\`python
from sklearn.naive_bayes import BernoulliNB

# Binary features (word present=1, absent=0)
X_binary = (X_counts > 0).astype(int)

bnb = BernoulliNB()
bnb.fit(X_binary.toarray(), labels)

print("\\nBernoulli Naive Bayes (binary features):")
X_test_binary = (X_test_counts > 0).astype(int)
predictions_bnb = bnb.predict(X_test_binary.toarray())
print(f"Predictions: {predictions_bnb}")
\`\`\`

## Laplace Smoothing

Problem: If a feature value never appears with a class in training, \\( P(x_i|y) = 0 \\), making the entire posterior 0!

**Solution: Laplace smoothing (additive smoothing)**

\\[ P(x_i|y) = \\frac{\\text{count}(x_i, y) + \\alpha}{\\text{count}(y) + \\alpha n} \\]

Where \\( \\alpha \\) is the smoothing parameter (default=1).

\`\`\`python
# Demonstrate smoothing importance
print("\\nLaplace Smoothing:")

# Without smoothing (alpha=0) - can cause issues
mnb_no_smooth = MultinomialNB(alpha=0.0)
mnb_no_smooth.fit(X_counts, labels)

# With smoothing (alpha=1)
mnb_smooth = MultinomialNB(alpha=1.0)
mnb_smooth.fit(X_counts, labels)

# Test with unseen word
test_unseen = vectorizer.transform(["completely new words here"])
# This might fail or give extreme probabilities without smoothing
\`\`\`

## Real-World Example: Spam Detection

\`\`\`python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# Load email-like dataset
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, 
                                  shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test', categories=categories,
                                shuffle=True, random_state=42)

print("="*60)
print("EMAIL CLASSIFICATION WITH NAIVE BAYES")
print("="*60)
print(f"\\nTraining samples: {len(twenty_train.data)}")
print(f"Test samples: {len(twenty_test.data)}")
print(f"Categories: {twenty_train.target_names}")

# Create pipeline
text_clf = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
    ('clf', MultinomialNB(alpha=0.1)),
])

# Train
text_clf.fit(twenty_train.data, twenty_train.target)

# Predict
predicted = text_clf.predict(twenty_test.data)

# Evaluate
accuracy = np.mean(predicted == twenty_test.target)
print(f"\\nTest Accuracy: {accuracy:.4f}")

print("\\nClassification Report:")
print(classification_report(twenty_test.target, predicted, 
                           target_names=twenty_train.target_names))

# Example predictions
docs_new = [
    'God is love',
    'OpenGL on the GPU is fast',
    'Heart disease treatment',
]
predicted_new = text_clf.predict(docs_new)
for doc, category in zip(docs_new, predicted_new):
    print(f"'{doc}' => {twenty_train.target_names[category]}")
\`\`\`

## Why Naive Bayes Works Despite Naive Assumption

The independence assumption is almost always violated, yet Naive Bayes often performs well:

1. **Classification doesn't require accurate probabilities** - just correct ranking
2. **Features often somewhat correlated with class** even if correlated with each other
3. **Errors can cancel out** across features
4. **Small datasets** where simple models generalize better
5. **High-dimensional data** where curse of dimensionality affects other methods more

\`\`\`python
# Demonstrate with correlated features
from sklearn.datasets import make_classification

# Generate data with correlated features
X_corr, y_corr = make_classification(n_samples=1000, n_features=10,
                                     n_informative=10, n_redundant=0,
                                     n_repeated=0, random_state=42)

# Add correlation between features
X_corr[:, 1] = X_corr[:, 0] + np.random.randn(1000) * 0.5  # Feature 1 correlated with 0

# Check correlation
correlation = np.corrcoef(X_corr[:, 0], X_corr[:, 1])[0, 1]
print(f"\\nCorrelation between features 0 and 1: {correlation:.3f}")

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_corr, y_corr, test_size=0.3, random_state=42
)

# Naive Bayes still works!
nb_corr = GaussianNB()
nb_corr.fit(X_train_c, y_train_c)

print(f"\\nNaive Bayes accuracy (despite correlated features): {nb_corr.score(X_test_c, y_test_c):.4f}")
print("It works because classification depends on ranking, not exact probabilities!")
\`\`\`

## Advantages and Limitations

**Advantages:**
- Very fast training and prediction
- Works well with small training sets
- Naturally handles multi-class
- Performs well in high dimensions
- Probabilistic predictions
- Simple and interpretable
- Works well for text classification

**Limitations:**
- Assumes feature independence (rarely true)
- Probability estimates can be poor
- Sensitive to irrelevant features
- Can't learn feature interactions
- Zero-frequency problem (needs smoothing)
- Not suitable when independence assumption badly violated

## When to Use Naive Bayes

**Good scenarios:**
- Text classification (spam, sentiment, categorization)
- Real-time predictions (fast!)
- High-dimensional data
- Small training sets
- Baseline model
- Need probabilistic outputs

**Avoid when:**
- Features are highly dependent
- Need accurate probability estimates
- Feature interactions are important
- Have lots of training data (complex models can do better)

## Comparison with Other Classifiers

\`\`\`python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import time

# Compare classifiers
X_comp, y_comp = make_classification(n_samples=5000, n_features=20, random_state=42)
X_train_comp, X_test_comp, y_train_comp, y_test_comp = train_test_split(
    X_comp, y_comp, test_size=0.3, random_state=42
)

classifiers = {
    'Naive Bayes': GaussianNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(max_depth=10),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10)
}

print("\\n" + "="*60)
print("CLASSIFIER COMPARISON")
print("="*60)

results = []
for name, clf in classifiers.items():
    # Training time
    start = time.time()
    clf.fit(X_train_comp, y_train_comp)
    train_time = time.time() - start
    
    # Prediction time
    start = time.time()
    y_pred = clf.predict(X_test_comp)
    pred_time = time.time() - start
    
    # Accuracy
    accuracy = accuracy_score(y_test_comp, y_pred)
    
    results.append({
        'Classifier': name,
        'Train Time': train_time,
        'Pred Time': pred_time,
        'Accuracy': accuracy
    })
    
import pandas as pd
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))

print("\\nNaive Bayes is typically fastest for both training and prediction!")
\`\`\`

## Summary

Naive Bayes is a simple yet effective probabilistic classifier:
- Based on Bayes' theorem with independence assumption
- Three variants: Gaussian, Multinomial, Bernoulli
- Fast training and prediction
- Works well for text classification
- Surprisingly effective despite naive assumption
- Needs Laplace smoothing for unseen features

**Key Insight:** The "naive" assumption is strong but often doesn't hurt classification performance!

Next: Support Vector Machines for maximum margin classification!
`,
  codeExample: `# Complete Naive Bayes Pipeline

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# Load data
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

print("="*70)
print("NAIVE BAYES: BREAST CANCER CLASSIFICATION")
print("="*70)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)
y_prob = nb.predict_proba(X_test)[:, 1]

print(f"\\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=cancer.target_names))
`,
};
