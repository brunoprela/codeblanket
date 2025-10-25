/**
 * Common Discrete Distributions Section
 */

export const commondiscretedistributionsSection = {
  id: 'common-discrete-distributions',
  title: 'Common Discrete Distributions',
  content: `# Common Discrete Distributions

## Introduction

Discrete probability distributions are the foundation for modeling countable outcomes in machine learning. Understanding these distributions is essential for:
- Classification (Bernoulli, Categorical)
- Count data (Poisson)
- Sequential trials (Binomial, Geometric)
- Rare events (Poisson)
- Reinforcement learning (Geometric, Poisson)

## Bernoulli Distribution

Models a **single binary outcome** (success/failure, yes/no, 1/0).

**Parameters**: p (probability of success)

**PMF**:
\\[ P(X = k) = \\begin{cases} p & \\text{if } k=1 \\\\ 1-p & \\text{if } k=0 \\end{cases} \\]

**Properties**:
- E[X] = p
- Var(X) = p(1-p)

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def bernoulli_demo():
    """Demonstrate Bernoulli distribution"""
    
    p = 0.7  # 70% success rate
    
    # Create distribution
    bernoulli = stats.bernoulli (p)
    
    # Sample
    np.random.seed(42)
    samples = bernoulli.rvs (size=1000)
    
    print("=== Bernoulli Distribution ===")
    print(f"Parameter p = {p}")
    print(f"P(X=1) = {p}, P(X=0) = {1-p}")
    print(f"\\nTheoretical: E[X] = {p}, Var(X) = {p*(1-p):.4f}")
    print(f"Empirical:   E[X] = {samples.mean():.4f}, Var(X) = {samples.var():.4f}")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # PMF
    x = [0, 1]
    pmf = [1-p, p]
    ax1.bar (x, pmf, color=['red', 'green'], alpha=0.7)
    ax1.set_xlabel('Outcome')
    ax1.set_ylabel('Probability')
    ax1.set_title('Bernoulli PMF (p=0.7)')
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Failure (0)', 'Success (1)'])
    
    # Samples
    unique, counts = np.unique (samples, return_counts=True)
    ax2.bar (unique, counts/len (samples), color=['red', 'green'], alpha=0.7)
    ax2.set_xlabel('Outcome')
    ax2.set_ylabel('Empirical Probability')
    ax2.set_title('Bernoulli Samples (n=1000)')
    ax2.set_xticks([0, 1])
    
    plt.tight_layout()
    
    print("\\nML Applications:")
    print("- Binary classification (spam/not spam)")
    print("- Click prediction (click/no click)")
    print("- A/B testing outcomes")

bernoulli_demo()
\`\`\`

## Binomial Distribution

Models **number of successes** in n **independent** Bernoulli trials.

**Parameters**: n (number of trials), p (success probability)

**PMF**:
\\[ P(X = k) = \\binom{n}{k} p^k (1-p)^{n-k} \\]

**Properties**:
- E[X] = np
- Var(X) = np(1-p)

\`\`\`python
def binomial_demo():
    """Demonstrate Binomial distribution"""
    
    n, p = 10, 0.7  # 10 trials, 70% success
    
    binomial = stats.binom (n, p)
    
    print("=== Binomial Distribution ===")
    print(f"Parameters: n={n} trials, p={p}")
    print(f"Question: Out of {n} trials, how many successes?")
    print()
    
    # PMF for all possible values
    x = np.arange(0, n+1)
    pmf = binomial.pmf (x)
    
    print("PMF:")
    for k in x:
        print(f"P(X={k:2d}) = {pmf[k]:.4f}")
    
    print(f"\\nTheoretical: E[X] = {n*p}, Var(X) = {n*p*(1-p):.4f}")
    
    # Samples
    np.random.seed(42)
    samples = binomial.rvs (size=10000)
    print(f"Empirical:   E[X] = {samples.mean():.4f}, Var(X) = {samples.var():.4f}")
    
    # Plot
    plt.figure (figsize=(10, 6))
    plt.bar (x, pmf, alpha=0.7, edgecolor='black')
    plt.xlabel('Number of Successes (k)')
    plt.ylabel('Probability')
    plt.title (f'Binomial PMF (n={n}, p={p})')
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks (x)
    
    # Most likely value
    mode = np.argmax (pmf)
    plt.axvline (mode, color='r', linestyle='--', label=f'Mode = {mode}')
    plt.legend()
    
    print("\\nML Applications:")
    print("- A/B testing: conversions out of visitors")
    print("- Model evaluation: correct predictions out of n samples")
    print("- Feature engineering: count of binary events")

binomial_demo()
\`\`\`

## Poisson Distribution

Models **number of events** occurring in a fixed interval (time/space) when events occur independently at a constant average rate.

**Parameters**: λ (lambda) = average rate

**PMF**:
\\[ P(X = k) = \\frac{\\lambda^k e^{-\\lambda}}{k!} \\]

**Properties**:
- E[X] = λ
- Var(X) = λ
- Mean equals variance!

\`\`\`python
def poisson_demo():
    """Demonstrate Poisson distribution"""
    
    lambda_param = 3.5  # Average 3.5 events per interval
    
    poisson = stats.poisson (lambda_param)
    
    print("=== Poisson Distribution ===")
    print(f"Parameter λ = {lambda_param}")
    print(f"Question: How many events in an interval?")
    print(f"E[X] = Var(X) = λ = {lambda_param}")
    print()
    
    # PMF for reasonable range
    x = np.arange(0, 15)
    pmf = poisson.pmf (x)
    
    print("PMF (selected values):")
    for k in range(10):
        print(f"P(X={k}) = {pmf[k]:.4f}")
    
    # Samples
    np.random.seed(42)
    samples = poisson.rvs (size=10000)
    print(f"\\nEmpirical: E[X] = {samples.mean():.4f}, Var(X) = {samples.var():.4f}")
    
    # Plot
    plt.figure (figsize=(10, 6))
    plt.bar (x, pmf, alpha=0.7, edgecolor='black')
    plt.xlabel('Number of Events (k)')
    plt.ylabel('Probability')
    plt.title (f'Poisson PMF (λ={lambda_param})')
    plt.grid(True, alpha=0.3, axis='y')
    plt.axvline (lambda_param, color='r', linestyle='--', linewidth=2, label=f'Mean = λ = {lambda_param}')
    plt.legend()
    
    print("\\nML Applications:")
    print("- Count data: number of bugs in code, emails per hour")
    print("- Rare events modeling")
    print("- Queueing theory: server requests per second")
    print("- Text analysis: word counts in documents")

poisson_demo()
\`\`\`

## Geometric Distribution

Models the number of trials needed to get the **first success**.

**Parameters**: p (success probability per trial)

**PMF**:
\\[ P(X = k) = (1-p)^{k-1} p \\]

**Properties**:
- E[X] = 1/p
- Var(X) = (1-p)/p²
- Memoryless property!

\`\`\`python
def geometric_demo():
    """Demonstrate Geometric distribution"""
    
    p = 0.3  # 30% success rate
    
    geometric = stats.geom (p)
    
    print("=== Geometric Distribution ===")
    print(f"Parameter p = {p}")
    print(f"Question: How many trials until first success?")
    print(f"E[X] = 1/p = {1/p:.4f}")
    print()
    
    # PMF
    x = np.arange(1, 21)
    pmf = geometric.pmf (x)
    
    print("PMF (first 10 values):")
    for k in range(1, 11):
        print(f"P(X={k:2d}) = {pmf[k-1]:.4f}")
    
    # Samples
    np.random.seed(42)
    samples = geometric.rvs (size=10000)
    print(f"\\nEmpirical E[X] = {samples.mean():.4f}")
    
    # Plot
    plt.figure (figsize=(10, 6))
    plt.bar (x, pmf, alpha=0.7, edgecolor='black')
    plt.xlabel('Number of Trials to First Success')
    plt.ylabel('Probability')
    plt.title (f'Geometric PMF (p={p})')
    plt.grid(True, alpha=0.3, axis='y')
    plt.axvline(1/p, color='r', linestyle='--', linewidth=2, label=f'Mean = 1/p = {1/p:.2f}')
    plt.legend()
    
    print("\\nMemoryless Property:")
    print("If you've failed 5 times, expected trials remaining = 1/p")
    print("Same as if starting fresh! Past doesn't matter.")
    print("\\nML Applications:")
    print("- Reinforcement learning: episodes until success")
    print("- Random search: iterations until acceptable solution")
    print("- Survival analysis")

geometric_demo()
\`\`\`

## Categorical/Multinomial Distribution

Generalization of Bernoulli/Binomial to **multiple categories**.

**Categorical**: Single trial with k possible outcomes
**Multinomial**: n trials with k possible outcomes

\`\`\`python
def categorical_demo():
    """Demonstrate Categorical distribution"""
    
    # Probabilities for 3 classes
    probs = [0.5, 0.3, 0.2]  # Must sum to 1
    
    print("=== Categorical Distribution ===")
    print(f"Probabilities: {probs}")
    print(f"Classes: [0, 1, 2]")
    print()
    
    # Sample
    np.random.seed(42)
    samples = np.random.choice([0, 1, 2], size=10000, p=probs)
    
    unique, counts = np.unique (samples, return_counts=True)
    empirical_probs = counts / len (samples)
    
    print("Theoretical vs Empirical:")
    for i, (theo, emp) in enumerate (zip (probs, empirical_probs)):
        print(f"Class {i}: P={theo:.3f} (theory), {emp:.3f} (empirical)")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Theoretical
    ax1.bar (range (len (probs)), probs, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Probability')
    ax1.set_title('Categorical PMF (Theoretical)')
    ax1.set_xticks (range (len (probs)))
    
    # Empirical
    ax2.bar (unique, empirical_probs, alpha=0.7, edgecolor='black', color='orange')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Empirical Probability')
    ax2.set_title('Categorical Samples (n=10000)')
    ax2.set_xticks (range (len (probs)))
    
    plt.tight_layout()
    
    print("\\nML Applications:")
    print("- Multi-class classification output layer")
    print("- Softmax activation represents categorical distribution")
    print("- Cross-entropy loss assumes categorical distribution")

categorical_demo()
\`\`\`

## Distribution Comparison

\`\`\`python
def compare_distributions():
    """Compare different discrete distributions"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Bernoulli
    ax = axes[0, 0]
    x = [0, 1]
    pmf = [0.3, 0.7]
    ax.bar (x, pmf, alpha=0.7, edgecolor='black')
    ax.set_title('Bernoulli (p=0.7)')
    ax.set_xlabel('Outcome')
    ax.set_ylabel('Probability')
    ax.set_xticks([0, 1])
    
    # Binomial
    ax = axes[0, 1]
    x = np.arange(0, 11)
    pmf = stats.binom(10, 0.7).pmf (x)
    ax.bar (x, pmf, alpha=0.7, edgecolor='black')
    ax.set_title('Binomial (n=10, p=0.7)')
    ax.set_xlabel('Number of Successes')
    ax.set_ylabel('Probability')
    
    # Poisson
    ax = axes[1, 0]
    x = np.arange(0, 15)
    pmf = stats.poisson(5).pmf (x)
    ax.bar (x, pmf, alpha=0.7, edgecolor='black')
    ax.set_title('Poisson (λ=5)')
    ax.set_xlabel('Number of Events')
    ax.set_ylabel('Probability')
    
    # Geometric
    ax = axes[1, 1]
    x = np.arange(1, 16)
    pmf = stats.geom(0.3).pmf (x)
    ax.bar (x, pmf, alpha=0.7, edgecolor='black')
    ax.set_title('Geometric (p=0.3)')
    ax.set_xlabel('Trials to First Success')
    ax.set_ylabel('Probability')
    
    plt.tight_layout()
    
    print("=== Distribution Summary ===")
    print("Bernoulli: Single binary trial")
    print("Binomial: Multiple binary trials → count successes")
    print("Poisson: Count events in interval (rare events)")
    print("Geometric: Trials until first success (memoryless)")
    print("Categorical: Single multi-class trial")

compare_distributions()
\`\`\`

## ML Application: Classification

\`\`\`python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def classification_distributions():
    """Show how distributions relate to classification"""
    
    # Generate data
    X, y = make_classification (n_samples=1000, n_features=5, n_informative=3, 
                                n_redundant=0, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Predictions are Bernoulli random variables!
    proba = model.predict_proba(X_test)
    
    print("=== Classification as Bernoulli Distribution ===")
    print("Each prediction is a Bernoulli random variable")
    print("\\nFirst 5 predictions:")
    print("P(y=0)  P(y=1)  Predicted Class")
    print("-" * 40)
    for i in range(5):
        pred_class = 1 if proba[i, 1] > 0.5 else 0
        print(f"{proba[i, 0]:.3f}   {proba[i, 1]:.3f}   {pred_class}")
    
    print("\\nModel outputs probability parameter p for Bernoulli distribution")
    print("Each test sample ~ Bernoulli (p) where p = P(y=1|x)")

classification_distributions()
\`\`\`

## Key Takeaways

1. **Bernoulli**: Single binary outcome, parameter p
2. **Binomial**: n Bernoulli trials, count successes
3. **Poisson**: Count events in interval, mean = variance = λ
4. **Geometric**: Trials until first success, memoryless
5. **Categorical**: Multi-class generalization of Bernoulli
6. **ML connection**: Classification models output distribution parameters
7. **Choose distribution**: Based on problem structure (binary/count/categorical)

Understanding these distributions is essential for probabilistic modeling in ML!
`,
};
