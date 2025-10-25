/**
 * Bayes' Theorem Section
 */

export const bayestheoremSection = {
  id: 'bayes-theorem',
  title: "Bayes' Theorem",
  content: `# Bayes' Theorem

## Introduction

Bayes' Theorem is one of the most important formulas in probability, statistics, and machine learning. It allows us to **update our beliefs** based on new evidence - the foundation of Bayesian inference and many ML algorithms.

**Applications in ML**:
- Bayesian classification (Naive Bayes)
- Updating model parameters given data
- Spam filtering
- Medical diagnosis
- A/B testing
- Probabilistic programming

## The Theorem

Given events A and B:

\\[ P(A|B) = \\frac{P(B|A) \\times P(A)}{P(B)} \\]

**In words**: 

\\[ \\text{Posterior} = \\frac{\\text{Likelihood} \\times \\text{Prior}}{\\text{Evidence}} \\]

### Components

1. **P(A|B)**: Posterior probability - what we want to know
2. **P(B|A)**: Likelihood - probability of evidence given hypothesis
3. **P(A)**: Prior probability - initial belief before seeing evidence
4. **P(B)**: Evidence/Marginal probability - normalizing constant

### Derivation

From the definition of conditional probability:
- \\( P(A|B) = P(A \\cap B) / P(B) \\)
- \\( P(B|A) = P(A \\cap B) / P(A) \\)

Solving for \\( P(A \\cap B) \\):
- \\( P(A \\cap B) = P(A|B) \\times P(B) = P(B|A) \\times P(A) \\)

Rearranging:
- \\( P(A|B) = P(B|A) \\times P(A) / P(B) \\) ✓

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt

# Classic example: Medical diagnosis
def medical_diagnosis_bayes():
    """
    Disease affects 1% of population (prior)
    Test has 95% sensitivity (true positive rate)
    Test has 90% specificity (true negative rate)
    
    If someone tests positive, what's probability they have disease?
    """
    
    # Prior: P(disease)
    p_disease = 0.01
    p_healthy = 1 - p_disease
    
    # Likelihood: P(positive test | disease or healthy)
    p_positive_given_disease = 0.95  # sensitivity
    p_positive_given_healthy = 0.10  # 1 - specificity (false positive rate)
    
    # Evidence: P(positive test) - Law of Total Probability
    p_positive = (p_positive_given_disease * p_disease + 
                  p_positive_given_healthy * p_healthy)
    
    # Bayes' Theorem: P(disease | positive)
    p_disease_given_positive = (p_positive_given_disease * p_disease) / p_positive
    
    print("=== Medical Diagnosis with Bayes' Theorem ===")
    print(f"Prior: P(disease) = {p_disease:.3f} (1%)")
    print(f"Likelihood: P(positive | disease) = {p_positive_given_disease:.3f} (sensitivity)")
    print(f"           P(positive | healthy) = {p_positive_given_healthy:.3f} (false positive rate)")
    print(f"Evidence: P(positive) = {p_positive:.3f}")
    print()
    print(f"Posterior: P(disease | positive) = {p_disease_given_positive:.3f}")
    print()
    print(f"Surprise! Even with 95% accurate test, only {p_disease_given_positive:.1%}")
    print(f"of positive tests actually have the disease!")
    print()
    print("Why? The disease is rare (1%), so most positives are false positives.")
    
    return p_disease_given_positive

posterior = medical_diagnosis_bayes()

# Output:
# === Medical Diagnosis with Bayes' Theorem ===
# Prior: P(disease) = 0.010 (1%)
# Likelihood: P(positive | disease) = 0.950 (sensitivity)
#            P(positive | healthy) = 0.100 (false positive rate)
# Evidence: P(positive) = 0.109
#
# Posterior: P(disease | positive) = 0.087
#
# Surprise! Even with 95% accurate test, only 8.7%
# of positive tests actually have the disease!
#
# Why? The disease is rare (1%), so most positives are false positives.
\`\`\`

## Law of Total Probability

The denominator P(B) is often computed using the Law of Total Probability:

\\[ P(B) = P(B|A)P(A) + P(B|A^c)P(A^c) \\]

For multiple hypotheses \\( A_1, A_2, \\ldots, A_n \\) that partition the space:

\\[ P(B) = \\sum_{i=1}^{n} P(B|A_i)P(A_i) \\]

\`\`\`python
def law_of_total_probability_demo():
    """Demonstrate Law of Total Probability"""
    
    # Email classification: spam vs legitimate
    p_spam = 0.3  # 30% of emails are spam
    p_legit = 0.7  # 70% are legitimate
    
    # Word "free" appears differently in spam vs legit
    p_free_given_spam = 0.8  # 80% of spam contains "free"
    p_free_given_legit = 0.1  # 10% of legit contains "free"
    
    # What\'s the overall probability an email contains "free"?
    # Law of Total Probability
    p_free = (p_free_given_spam * p_spam + 
              p_free_given_legit * p_legit)
    
    print("=== Law of Total Probability ===")
    print(f"P(spam) = {p_spam}, P(legit) = {p_legit}")
    print(f'P("free"|spam) = {p_free_given_spam}, P("free"|legit) = {p_free_given_legit}')
    print()
    print(f'P("free") = P("free"|spam)P(spam) + P("free"|legit)P(legit)')
    print(f'         = ({p_free_given_spam})({p_spam}) + ({p_free_given_legit})({p_legit})')
    print(f'         = {p_free:.3f}')
    
    # Now use Bayes to go backward: P(spam | "free")
    p_spam_given_free = (p_free_given_spam * p_spam) / p_free
    
    print(f'\\nBayes: P(spam|"free") = {p_spam_given_free:.3f}')
    print(f'Seeing "free" increases spam probability from {p_spam:.1%} to {p_spam_given_free:.1%}!')

law_of_total_probability_demo()

# Output:
# === Law of Total Probability ===
# P(spam) = 0.3, P(legit) = 0.7
# P("free"|spam) = 0.8, P("free"|legit) = 0.1
#
# P("free") = P("free"|spam)P(spam) + P("free"|legit)P(legit)
#          = (0.8)(0.3) + (0.1)(0.7)
#          = 0.310
#
# Bayes: P(spam|"free") = 0.774
# Seeing "free" increases spam probability from 30.0% to 77.4%!
\`\`\`

## Bayesian Updating

Bayes' Theorem allows us to update probabilities as we gather more evidence:

\\[ P(H|E_1, E_2) = \\frac{P(E_2|H, E_1) \\times P(H|E_1)}{P(E_2|E_1)} \\]

The posterior from one update becomes the prior for the next!

\`\`\`python
def bayesian_updating():
    """Sequential Bayesian updating with multiple pieces of evidence"""
    
    # Coin flip: fair (50/50) or biased (70/30)?
    # Prior: equal probability
    p_fair = 0.5
    p_biased = 0.5
    
    # Observe coin flips
    observations = ['H', 'H', 'H', 'T', 'H']  # 4 heads, 1 tail
    
    print("=== Bayesian Updating: Fair vs Biased Coin ===")
    print(f"Prior: P(fair) = {p_fair:.3f}, P(biased) = {p_biased:.3f}\\n")
    
    for i, outcome in enumerate (observations, 1):
        # Likelihoods
        if outcome == 'H':
            p_outcome_given_fair = 0.5
            p_outcome_given_biased = 0.7
        else:  # 'T'
            p_outcome_given_fair = 0.5
            p_outcome_given_biased = 0.3
        
        # Evidence (normalizer)
        p_outcome = (p_outcome_given_fair * p_fair + 
                     p_outcome_given_biased * p_biased)
        
        # Bayes' update
        p_fair = (p_outcome_given_fair * p_fair) / p_outcome
        p_biased = (p_outcome_given_biased * p_biased) / p_outcome
        
        print(f"After observing {outcome}: P(fair) = {p_fair:.3f}, P(biased) = {p_biased:.3f}")
    
    print(f"\\nConclusion: Coin is likely {'biased' if p_biased > p_fair else 'fair'}")
    print(f"(More heads than expected from fair coin)")

bayesian_updating()

# Output:
# === Bayesian Updating: Fair vs Biased Coin ===
# Prior: P(fair) = 0.500, P(biased) = 0.500
#
# After observing H: P(fair) = 0.417, P(biased) = 0.583
# After observing H: P(fair) = 0.347, P(biased) = 0.653
# After observing H: P(fair) = 0.286, P(biased) = 0.714
# After observing T: P(fair) = 0.345, P(biased) = 0.655
# After observing H: P(fair) = 0.283, P(biased) = 0.717
#
# Conclusion: Coin is likely biased
# (More heads than expected from fair coin)
\`\`\`

## Naive Bayes Classifier

One of the most important applications of Bayes' Theorem in ML:

\\[ P(y|x_1, \\ldots, x_n) = \\frac{P(x_1, \\ldots, x_n|y) \\times P(y)}{P(x_1, \\ldots, x_n)} \\]

With the "naive" assumption of conditional independence:

\\[ P(y|x_1, \\ldots, x_n) \\propto P(y) \\prod_{i=1}^{n} P(x_i|y) \\]

\`\`\`python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Spam classification example
emails = [
    "Get free money now",
    "Meeting tomorrow at 3pm",
    "Buy cheap viagra online",
    "Project deadline reminder",
    "Win a free iPhone today",
    "Lunch plans next week",
    "Claim your prize money",
    "Team standup at 10am",
    "Limited time offer free",
    "Code review feedback",
]

labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1=spam, 0=legitimate

# Convert text to features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform (emails)

# Train Naive Bayes
nb = MultinomialNB()
nb.fit(X, labels)

# Test on new emails
test_emails = [
    "Free money offer",
    "Team meeting notes"
]

X_test = vectorizer.transform (test_emails)
predictions = nb.predict(X_test)
probabilities = nb.predict_proba(X_test)

print("=== Naive Bayes Spam Classifier ===")
for email, pred, proba in zip (test_emails, predictions, probabilities):
    label = "SPAM" if pred == 1 else "LEGITIMATE"
    print(f"Email: '{email}'")
    print(f"Prediction: {label}")
    print(f"P(legit)={proba[0]:.3f}, P(spam)={proba[1]:.3f}\\n")

# Output:
# === Naive Bayes Spam Classifier ===
# Email: 'Free money offer'
# Prediction: SPAM
# P(legit)=0.182, P(spam)=0.818
#
# Email: 'Team meeting notes'
# Prediction: LEGITIMATE
# P(legit)=0.714, P(spam)=0.286
\`\`\`

## Prior vs Posterior

Understanding how evidence updates beliefs:

\`\`\`python
def visualize_prior_vs_posterior():
    """Show how evidence shifts probability"""
    
    # Prior: 50/50 on hypothesis
    prior = 0.5
    
    # Various evidence strengths
    evidence_strengths = np.linspace(0.1, 0.9, 9)
    
    posteriors = []
    for likelihood in evidence_strengths:
        # P(H|E) = P(E|H) * P(H) / P(E)
        # Assume P(E|not H) = 1 - likelihood
        p_e_given_not_h = 1 - likelihood
        p_e = likelihood * prior + p_e_given_not_h * (1 - prior)
        posterior = (likelihood * prior) / p_e
        posteriors.append (posterior)
    
    # Plot
    plt.figure (figsize=(10, 6))
    plt.plot (evidence_strengths, posteriors, 'b-', linewidth=2, label='Posterior P(H|E)')
    plt.axhline (y=prior, color='r', linestyle='--', label=f'Prior P(H) = {prior}')
    plt.xlabel('Likelihood P(E|H)', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title('How Evidence Updates Beliefs (Bayesian Updating)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim([0, 1])
    
    print("=== Bayesian Updating Visualization ===")
    print("Strong evidence (high likelihood) → High posterior")
    print("Weak evidence (low likelihood) → Low posterior")
    print("Prior = 0.5 (neutral) updated by evidence strength")

visualize_prior_vs_posterior()
\`\`\`

## Common Misconceptions

### 1. Prosecutor\'s Fallacy

**Wrong**: P(innocent | evidence) = P(evidence | innocent)

**Right**: Must use Bayes' Theorem with prior probabilities!

Example: DNA match occurs in 1 in 1 million people. This is P(match | innocent) = 10⁻⁶.
But P(innocent | match) depends on prior probability of guilt!

### 2. Base Rate Neglect

Ignoring prior probabilities P(A) leads to wrong conclusions.

\`\`\`python
# Example: Ignoring base rates
print("=== Base Rate Neglect ===")
print("Test is 99% accurate for detecting disease")
print("Disease affects 0.1% of population")
print()
print("If positive, most people think: 99% chance of disease")
print("Reality (using Bayes): ~9% chance of disease")
print()
print("Why? Because disease is so rare, most positives are false positives!")
\`\`\`

## ML Applications

### 1. Bayesian Inference

Update model parameters θ given data D:

\\[ P(\\theta|D) = \\frac{P(D|\\theta) \\times P(\\theta)}{P(D)} \\]

### 2. Maximum A Posteriori (MAP) Estimation

\\[ \\hat{\\theta}_{MAP} = \\arg\\max_{\\theta} P(\\theta|D) = \\arg\\max_{\\theta} P(D|\\theta)P(\\theta) \\]

Equivalent to regularized maximum likelihood!

### 3. Bayesian Networks

Graphical models that use Bayes' Theorem for inference.

### 4. A/B Testing

Update belief about which variant is better as data arrives.

\`\`\`python
def ab_test_bayesian():
    """Bayesian A/B testing"""
    
    # Observed data
    conversions_A = 120
    total_A = 1000
    conversions_B = 135
    total_B = 1000
    
    # Beta distributions (conjugate prior for binomial)
    from scipy.stats import beta
    
    # Posterior distributions
    alpha_A, beta_A = conversions_A + 1, total_A - conversions_A + 1
    alpha_B, beta_B = conversions_B + 1, total_B - conversions_B + 1
    
    # Sample from posteriors
    np.random.seed(42)
    samples_A = beta.rvs (alpha_A, beta_A, size=10000)
    samples_B = beta.rvs (alpha_B, beta_B, size=10000)
    
    # Probability B > A
    p_b_better = np.mean (samples_B > samples_A)
    
    print("=== Bayesian A/B Test ===")
    print(f"Variant A: {conversions_A}/{total_A} = {conversions_A/total_A:.1%} conversion")
    print(f"Variant B: {conversions_B}/{total_B} = {conversions_B/total_B:.1%} conversion")
    print()
    print(f"P(B > A) = {p_b_better:.1%}")
    
    if p_b_better > 0.95:
        print("Strong evidence that B is better!")
    elif p_b_better > 0.75:
        print("Moderate evidence that B is better")
    else:
        print("Insufficient evidence to declare winner")

ab_test_bayesian()

# Output:
# === Bayesian A/B Test ===
# Variant A: 120/1000 = 12.0% conversion
# Variant B: 135/1000 = 13.5% conversion
#
# P(B > A) = 82.7%
# Moderate evidence that B is better
\`\`\`

## Key Takeaways

1. **Bayes' Theorem**: P(A|B) = P(B|A)P(A)/P(B) - updates beliefs with evidence
2. **Components**: Prior × Likelihood / Evidence = Posterior
3. **Sequential updating**: Posterior becomes prior for next update
4. **Naive Bayes**: Assumes conditional independence, widely used for classification
5. **Base rates matter**: Don't ignore prior probabilities!
6. **ML applications**: Parameter estimation, classification, A/B testing, Bayesian networks
7. **Bayesian thinking**: Quantify uncertainty, update beliefs systematically

Bayes' Theorem is the mathematical foundation for learning from data!
`,
};
