/**
 * Information Theory Basics Section
 */

export const informationtheorybasicsSection = {
  id: 'information-theory-basics',
  title: 'Information Theory Basics',
  content: `# Information Theory Basics

## Introduction

**Information theory**, founded by Claude Shannon, quantifies information and uncertainty. It's fundamental to machine learning, especially:
- Classification loss functions (cross-entropy)
- Model complexity (mutual information)
- Feature selection (information gain)
- Compression and encoding

## Entropy

**Entropy** H(X) measures the average uncertainty or "surprise" in a random variable.

\\[ H(X) = -\\sum_{x} P(X=x) \\log_2 P(X=x) \\text{ bits} \\]

Or with natural log: \\( H(X) = -\\sum_{x} P(X=x) \\ln P(X=x) \\) nats

**Interpretation**: Average number of bits needed to encode outcomes.

**Properties**:
- H(X) ≥ 0 (non-negative)
- H(X) = 0 iff deterministic (one outcome has P=1)
- Maximum when uniform distribution

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt

def entropy_demo():
    """Demonstrate entropy"""
    
    def entropy(probs):
        """Calculate entropy"""
        probs = np.array(probs)
        probs = probs[probs > 0]  # Remove zeros (0 log 0 = 0)
        return -np.sum(probs * np.log2(probs))
    
    print("=== Entropy ===")
    print("H(X) = -Σ P(x) log₂ P(x)")
    print()
    
    # Different distributions
    examples = [
        ("Deterministic", [1.0, 0.0, 0.0, 0.0]),
        ("Mostly certain", [0.9, 0.05, 0.03, 0.02]),
        ("Somewhat uncertain", [0.5, 0.3, 0.15, 0.05]),
        ("Very uncertain", [0.25, 0.25, 0.25, 0.25]),
    ]
    
    print("Distribution  ->  Entropy (bits)")
    print("-" * 40)
    
    for name, probs in examples:
        h = entropy(probs)
        print(f"{name:20s}  {h:.3f}")
    
    print()
    print("Key insight: Uniform distribution has maximum entropy")
    print(f"For 4 outcomes: max H = log₂(4) = {np.log2(4):.3f} bits")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (name, probs) in enumerate(examples):
        ax = axes[idx]
        ax.bar(range(len(probs)), probs, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Outcome')
        ax.set_ylabel('Probability')
        ax.set_title(f'{name}\\nH = {entropy(probs):.3f} bits')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()

entropy_demo()
\`\`\`

## Cross-Entropy

**Cross-entropy** H(P, Q) measures how many bits we need if we use wrong distribution Q to encode data from true distribution P.

\\[ H(P, Q) = -\\sum_{x} P(x) \\log Q(x) \\]

**Interpretation**: Average surprise if we assume Q but reality is P.

\`\`\`python
def cross_entropy_demo():
    """Demonstrate cross-entropy"""
    
    def cross_entropy(p_true, q_model):
        """Calculate cross-entropy"""
        p_true = np.array(p_true)
        q_model = np.array(q_model)
        return -np.sum(p_true * np.log2(q_model + 1e-10))
    
    # True distribution
    p_true = [0.7, 0.2, 0.1]
    
    # Different model distributions
    models = [
        ("Perfect model", [0.7, 0.2, 0.1]),
        ("Good model", [0.6, 0.25, 0.15]),
        ("Bad model", [0.3, 0.3, 0.4]),
        ("Uniform model", [0.333, 0.333, 0.333]),
    ]
    
    print("=== Cross-Entropy ===")
    print(f"True distribution P: {p_true}")
    print()
    print("Model Distribution Q  ->  H(P, Q)")
    print("-" * 50)
    
    true_entropy = cross_entropy(p_true, p_true)
    print(f"Entropy H(P): {true_entropy:.3f} bits")
    print()
    
    for name, q_model in models:
        ce = cross_entropy(p_true, q_model)
        print(f"{name:20s}  {ce:.3f} bits")
    
    print()
    print("Note: Cross-entropy is minimized when Q = P")
    print("This is why we minimize cross-entropy loss in ML!")

cross_entropy_demo()
\`\`\`

## KL Divergence

**Kullback-Leibler (KL) divergence** measures how different Q is from P:

\\[ D_{KL}(P||Q) = \\sum_{x} P(x) \\log \\frac{P(x)}{Q(x)} = H(P, Q) - H(P) \\]

**Properties**:
- D_KL(P||Q) ≥ 0 (non-negative)
- D_KL(P||Q) = 0 iff P = Q
- NOT symmetric: D_KL(P||Q) ≠ D_KL(Q||P)

\`\`\`python
def kl_divergence_demo():
    """Demonstrate KL divergence"""
    
    def kl_divergence(p, q):
        """Calculate KL divergence"""
        p = np.array(p)
        q = np.array(q)
        return np.sum(p * np.log(p / (q + 1e-10) + 1e-10))
    
    # True distribution
    p_true = [0.6, 0.3, 0.1]
    
    models = [
        [0.6, 0.3, 0.1],  # Perfect
        [0.5, 0.35, 0.15],  # Close
        [0.4, 0.4, 0.2],  # Farther
        [0.2, 0.5, 0.3],  # Far
    ]
    
    print("=== KL Divergence ===")
    print(f"True P: {p_true}")
    print()
    print("Model Q           D_KL(P||Q) (nats)")
    print("-" * 45)
    
    for q in models:
        kl = kl_divergence(p_true, q)
        print(f"{q}  {kl:.4f}")
    
    print()
    print("Minimizing KL divergence ≡ Minimizing cross-entropy")
    print("(since H(P) is constant w.r.t. model parameters)")

kl_divergence_demo()
\`\`\`

## ML Applications

### Cross-Entropy Loss

\`\`\`python
def cross_entropy_loss_demo():
    """Cross-entropy as ML loss function"""
    
    print("=== Cross-Entropy Loss in ML ===")
    print()
    print("Binary Classification:")
    print("  Loss = -[y log(ŷ) + (1-y) log(1-ŷ)]")
    print("  Minimizing this = maximizing likelihood")
    print()
    print("Multi-class Classification:")
    print("  Loss = -Σ y_i log(ŷ_i)")
    print("  where y is one-hot encoded")
    print()
    
    # Example
    print("Example (3 classes):")
    y_true = [0, 1, 0]  # True class = 1
    y_pred_good = [0.1, 0.8, 0.1]
    y_pred_bad = [0.4, 0.3, 0.3]
    
    loss_good = -np.sum(np.array(y_true) * np.log(np.array(y_pred_good) + 1e-10))
    loss_bad = -np.sum(np.array(y_true) * np.log(np.array(y_pred_bad) + 1e-10))
    
    print(f"Good prediction {y_pred_good}: Loss = {loss_good:.3f}")
    print(f"Bad prediction {y_pred_bad}: Loss = {loss_bad:.3f}")
    print()
    print("Lower cross-entropy = better predictions")

cross_entropy_loss_demo()
\`\`\`

### Mutual Information

\\[ I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) \\]

Measures how much knowing Y reduces uncertainty about X.

\`\`\`python
def mutual_information_demo():
    """Demonstrate mutual information for feature selection"""
    
    print("=== Mutual Information ===")
    print("I(X;Y) = information shared between X and Y")
    print()
    print("Applications:")
    print("1. Feature selection: Choose features with high I(X; Y)")
    print("2. Redundancy: If I(X₁; X₂) high, features are redundant")
    print("3. Independence: I(X;Y) = 0 iff X and Y independent")
    print()
    print("Example: Feature importance for classification")
    
    from sklearn.datasets import load_iris
    from sklearn.feature_selection import mutual_info_classif
    
    data = load_iris()
    X, y = data.data, data.target
    
    mi = mutual_info_classif(X, y)
    
    for i, (name, score) in enumerate(zip(data.feature_names, mi)):
        print(f"  {name:20s}: MI = {score:.3f}")
    
    print()
    print("Higher MI = more informative feature for classification")

mutual_information_demo()
\`\`\`

## Key Takeaways

1. **Entropy H(X)**: Measures uncertainty, maximum for uniform distribution
2. **Cross-entropy H(P,Q)**: Cost of using wrong model Q for true distribution P
3. **KL divergence**: D_KL(P||Q) = H(P,Q) - H(P), measures distribution difference
4. **ML loss**: Cross-entropy loss = minimizing KL divergence
5. **Mutual information**: Measures dependence between variables
6. **Applications**: Classification loss, feature selection, model comparison

Information theory provides the mathematical foundation for many ML algorithms!
`,
};
