/**
 * Quiz questions for Convex Optimization section
 */

export const convexoptimizationQuiz = [
  {
    id: 'convex-disc-1',
    question:
      'Explain why linear regression and logistic regression are convex optimization problems, but training deep neural networks is not. What are the practical implications?',
    hint: 'Consider the structure of the loss functions and the composition of operations.',
    sampleAnswer: `**Convexity in Machine Learning:**

**1. Linear Regression is Convex:**

**Loss function:**
\`\`\`
L(w) = (1/2n)||Xw - y||²
     = (1/2n)Σᵢ(wᵀxᵢ - yᵢ)²
\`\`\`

**Why convex:**

**Second-order condition:**
\`\`\`
∇L(w) = (1/n)X^T(Xw - y)
∇²L(w) = (1/n)X^T X  (Hessian)
\`\`\`

The Hessian H = (1/n)X^TX is positive semidefinite:
- For any vector v: v^T H v = v^T X^T X v = ||Xv||² ≥ 0
- Therefore, L is convex

**Implications:**
- Gradient descent converges to global minimum
- No local minima traps
- Solution unique (if X full rank)

**2. Logistic Regression is Convex:**

**Loss function (binary cross-entropy):**
\`\`\`
L(w) = -(1/n)Σᵢ[yᵢ log σ(w^Txᵢ) + (1-yᵢ) log(1-σ(w^Txᵢ))]
\`\`\`

where σ(z) = 1/(1+e^(-z)) is sigmoid.

**Why convex:**

**Hessian:**
\`\`\`
∇²L(w) = (1/n)X^T S X
\`\`\`

where S = diag(σ(w^Txᵢ)(1-σ(w^Txᵢ)))

**Key observation:**
- σ(z)(1-σ(z)) ∈ [0, 0.25] for all z
- S is positive semidefinite diagonal matrix
- Therefore X^T S X is positive semidefinite
- L is convex!

**Proof:**
For any v:
\`\`\`
v^T (X^T S X) v = (Xv)^T S (Xv) = Σᵢ sᵢ(Xv)ᵢ² ≥ 0
\`\`\`

**3. Deep Neural Networks are NOT Convex:**

**Loss function (e.g., for network with ReLU):**
\`\`\`
L(W₁, W₂, ..., Wₗ) = (1/n)Σᵢ ||fₗ(...f₂(f₁(xᵢ; W₁); W₂)...; Wₗ) - yᵢ||²
\`\`\`

where fₗ(z; W) = ReLU(Wz) or similar.

**Why NOT convex:**

**Problem 1: Composition of non-linearities**

Even simple 2-layer network:
\`\`\`
f (x; W₁, W₂) = W₂·ReLU(W₁x)
\`\`\`

Is NOT convex in (W₁, W₂):
- ReLU is convex in its input
- But composition W₂·ReLU(W₁x) is NOT convex in W₁
- Linear function composed with convex ≠ convex

**Example:**
\`\`\`python
# f (w₁, w₂) = w₂·ReLU(w₁) is not convex
w1 = np.linspace(-2, 2, 100)
w2 = 1.0
f = w2 * np.maximum(0, w1)

# Check second derivative
# f'(w₁) = w₂ if w₁ > 0, else 0
# f'(w₁) = 0 everywhere (except at 0, undefined)
# But f has a "kink" at 0 → not convex!
\`\`\`

**Problem 2: Permutation symmetry**

For network with hidden layer of size h:
- Any permutation of hidden units gives same function
- h! equivalent solutions
- Multiple global minima → non-convex structure

**Problem 3: Scaling symmetry**

Can scale weights in layer i by α and layer i+1 by 1/α:
- Same function output
- Continuum of equivalent solutions
- Non-convex landscape

**Problem 4: Activation functions**

Non-convex activations (tanh, sigmoid as function of weights):
\`\`\`
L(w) = ||σ(Wx) - y||²
\`\`\`

σ(Wx) is NOT convex in w because:
- σ is non-linear
- Composition with linear doesn't preserve convexity

**4. Practical Implications:**

**For Convex Problems (Linear/Logistic Regression):**

**Advantages:**
1. **Global optimum guaranteed**: Any optimization method finds global minimum
2. **Convergence guarantees**: Gradient descent converges with appropriate step size
3. **No initialization sensitivity**: Start anywhere, reach same solution
4. **Theoretical analysis**: Strong convergence rates, sample complexity bounds
5. **Optimization is easy**: Can use simple first-order methods

**Example:**
\`\`\`python
# Linear regression always converges
for any w_init:
    w = gradient_descent (w_init, X, y)
    # w converges to global optimum
\`\`\`

**For Non-Convex Problems (Deep Learning):**

**Challenges:**
1. **Local minima**: May get stuck in suboptimal solutions
2. **Saddle points**: Prevalent in high dimensions (though escapable)
3. **Initialization matters**: Different starts → different solutions
4. **No convergence guarantees**: May not reach global optimum
5. **Hyperparameter tuning**: Learning rate, batch size, architecture crucial

**Example:**
\`\`\`python
# Deep network: initialization matters
w_init_1 = random_init (seed=42)
w_init_2 = random_init (seed=123)

model_1 = train (w_init_1)  # May converge to different solution
model_2 = train (w_init_2)  # Than this one
\`\`\`

**Why Deep Learning Still Works:**

Despite non-convexity, deep learning succeeds because:

1. **Overparameterization**: More parameters than data points
   - Many paths to good solutions
   - Local minima often good enough

2. **Implicit regularization**: SGD noise acts as regularizer
   - Finds flat minima that generalize

3. **Good architecture design**: Skip connections, batch norm
   - Improve optimization landscape

4. **Careful initialization**: He, Xavier, etc.
   - Start in good region

5. **Modern optimizers**: Adam, RMSProp
   - Adaptive learning rates help

**5. Comparison Table:**

| Aspect | Convex (LR, Logistic) | Non-Convex (Deep Learning) |
|--------|----------------------|----------------------------|
| **Global optimum** | Always found | Not guaranteed |
| **Initialization** | Doesn't matter | Critical |
| **Convergence** | Guaranteed | Empirical |
| **Theory** | Strong | Limited |
| **Optimization** | Easy | Hard |
| **Local minima** | = global | May be poor |
| **Hyperparameters** | Few | Many |

**6. Practical Recommendations:**

**For Convex Problems:**
- Use simple methods (gradient descent, Newton's)
- Don't worry about initialization
- Focus on model selection, regularization

**For Non-Convex Problems:**
- Careful initialization (Xavier, He)
- Use modern optimizers (Adam)
- Multiple random restarts
- Architecture search
- Ensemble methods
- Don't expect global optimum, aim for "good enough"

**Conclusion:**

Convexity is a powerful property that makes optimization tractable. Linear and logistic regression enjoy this property due to their simple structure. Deep networks sacrifice convexity for expressiveness, accepting optimization challenges in exchange for representational power. Understanding this trade-off is fundamental to machine learning practice.`,
    keyPoints: [
      'Linear regression: Convex (Hessian = X^TX ⪰ 0)',
      'Logistic regression: Convex (cross-entropy with sigmoid)',
      'Deep networks: Non-convex (composition, symmetries)',
      'Convex: Global optimum guaranteed, easy optimization',
      'Non-convex: Local minima, initialization matters',
      'Deep learning works despite non-convexity (overparameterization, SGD)',
    ],
  },
  {
    id: 'convex-disc-2',
    question:
      'Describe the KKT conditions and their role in constrained optimization. How are they used in SVMs?',
    hint: 'Consider the primal and dual formulations of SVM, and what the KKT conditions tell us about the support vectors.',
    sampleAnswer: `**KKT Conditions and Support Vector Machines:**

**1. KKT Conditions Overview:**

For constrained optimization:
\`\`\`
minimize f (x)
subject to gᵢ(x) ≤ 0, i = 1,...,m
           hⱼ(x) = 0, j = 1,...,p
\`\`\`

**KKT Conditions** (necessary for optimality, sufficient if convex):

1. **Stationarity:**
   \`\`\`
   ∇f (x*) + Σᵢ λᵢ∇gᵢ(x*) + Σⱼ νⱼ∇hⱼ(x*) = 0
   \`\`\`

2. **Primal feasibility:**
   \`\`\`
   gᵢ(x*) ≤ 0, hⱼ(x*) = 0
   \`\`\`

3. **Dual feasibility:**
   \`\`\`
   λᵢ ≥ 0
   \`\`\`

4. **Complementary slackness:**
   \`\`\`
   λᵢ·gᵢ(x*) = 0 for all i
   \`\`\`

**Intuition:**
- **Stationarity**: Gradient of objective balanced by constraint gradients
- **Primal feasibility**: Solution satisfies constraints
- **Dual feasibility**: Lagrange multipliers non-negative (for inequality)
- **Complementary slackness**: Either constraint active (λᵢ > 0, gᵢ = 0) or inactive (λᵢ = 0, gᵢ < 0)

**2. Support Vector Machine Formulation:**

**Primal Problem:**
\`\`\`
minimize (1/2)||w||² + C·Σᵢ ξᵢ
subject to yᵢ(w^T xᵢ + b) ≥ 1 - ξᵢ, ∀i
           ξᵢ ≥ 0, ∀i
\`\`\`

**Interpretation:**
- Maximize margin (minimize ||w||²)
- Allow slack (ξᵢ) for misclassified/margin-violating points
- C: trade-off parameter

**Lagrangian:**
\`\`\`
L(w, b, ξ, α, μ) = (1/2)||w||² + C·Σᵢξᵢ 
                   - Σᵢαᵢ[yᵢ(w^Txᵢ + b) - 1 + ξᵢ]
                   - Σᵢμᵢξᵢ
\`\`\`

where αᵢ ≥ 0, μᵢ ≥ 0 are Lagrange multipliers.

**3. KKT Conditions for SVM:**

**Stationarity:**
\`\`\`
∂L/∂w = w - Σᵢαᵢyᵢxᵢ = 0  →  w* = Σᵢαᵢyᵢxᵢ
∂L/∂b = -Σᵢαᵢyᵢ = 0
∂L/∂ξᵢ = C - αᵢ - μᵢ = 0  →  αᵢ + μᵢ = C
\`\`\`

**Primal feasibility:**
\`\`\`
yᵢ(w^Txᵢ + b) ≥ 1 - ξᵢ
ξᵢ ≥ 0
\`\`\`

**Dual feasibility:**
\`\`\`
αᵢ ≥ 0, μᵢ ≥ 0
\`\`\`

**Complementary slackness:**
\`\`\`
αᵢ[yᵢ(w^Txᵢ + b) - 1 + ξᵢ] = 0
μᵢξᵢ = 0
\`\`\`

**4. Support Vectors Identification:**

From complementary slackness, for each point xᵢ:

**Case 1: αᵢ = 0**
- Point not a support vector
- Correctly classified, outside margin
- yᵢ(w^Txᵢ + b) > 1

**Case 2: 0 < αᵢ < C**
- Point is a support vector
- On the margin boundary
- ξᵢ = 0 (from μᵢξᵢ = 0 and μᵢ = C - αᵢ > 0)
- yᵢ(w^Txᵢ + b) = 1

**Case 3: αᵢ = C**
- Point is a support vector
- Inside margin or misclassified
- ξᵢ > 0
- yᵢ(w^Txᵢ + b) = 1 - ξᵢ < 1

**Visual Summary:**
\`\`\`
αᵢ = 0:        Outside margin (not support vector)
0 < αᵢ < C:    On margin (support vector)
αᵢ = C:        Inside margin/misclassified (support vector)
\`\`\`

**5. Dual Formulation:**

Using stationarity conditions, eliminate w, b, ξ:

**Dual Problem:**
\`\`\`
maximize Σᵢαᵢ - (1/2)ΣᵢΣⱼαᵢαⱼyᵢyⱼ(xᵢ^Txⱼ)
subject to 0 ≤ αᵢ ≤ C, ∀i
           Σᵢαᵢyᵢ = 0
\`\`\`

**Advantages of dual:**
1. **Kernel trick**: Can replace xᵢ^Txⱼ with K(xᵢ,xⱼ)
2. **Sparsity**: Many αᵢ = 0 (only support vectors matter)
3. **Convex quadratic program**: Efficiently solvable

**6. Practical Example:**

\`\`\`python
import numpy as np
from sklearn.svm import SVC

# Generate linearly separable data
np.random.seed(42)
X_pos = np.random.randn(50, 2) + [2, 2]
X_neg = np.random.randn(50, 2) + [-2, -2]
X = np.vstack([X_pos, X_neg])
y = np.array([1]*50 + [-1]*50)

# Train SVM
svm = SVC(kernel='linear', C=1.0)
svm.fit(X, y)

# Extract support vectors
support_vectors = svm.support_vectors_
support_indices = svm.support_
alphas = np.abs (svm.dual_coef_[0])

print("SVM with KKT Conditions:")
print(f"Total points: {len(X)}")
print(f"Support vectors: {len (support_vectors)}")
print(f"\\nSupport vector analysis:")

for i, (idx, alpha) in enumerate (zip (support_indices, alphas)):
    point = X[idx]
    label = y[idx]
    # Decision function value
    decision = svm.decision_function([point])[0]
    margin = label * decision
    
    print(f"\\nSV {i+1}:")
    print(f"  α = {alpha:.4f}")
    print(f"  Margin = {margin:.4f}")
    
    if alpha < 0.99 * svm.C:  # 0 < α < C
        print(f"  Status: On margin (ξ = 0)")
    else:  # α = C
        print(f"  Status: Inside margin (ξ > 0)")

print(f"\\nWeights w = Σαᵢyᵢxᵢ: {svm.coef_[0]}")
print(f"Bias b: {svm.intercept_[0]:.4f}")
\`\`\`

**7. Why KKT Matters for SVM:**

**Theoretical:**
1. **Optimality**: KKT conditions prove solution is optimal
2. **Uniqueness**: For strictly convex problem, unique solution
3. **Duality gap**: Zero for convex problems (strong duality)

**Practical:**
1. **Sparsity**: Only support vectors (αᵢ > 0) matter
   - Efficient prediction: O(# support vectors) not O(n)
   - Memory: Store only support vectors

2. **Kernel trick**: Decision function
   \`\`\`
   f (x) = Σᵢαᵢyᵢ K(xᵢ, x) + b
   \`\`\`
   Only need kernel evaluations with support vectors

3. **Optimization**: Can use specialized quadratic programming solvers
   - SMO (Sequential Minimal Optimization)
   - Coordinate ascent on dual

4. **Interpretability**: Support vectors are "important" points
   - On or inside margin
   - Define decision boundary

**8. Connection to Other ML Methods:**

**Boosting:**
KKT conditions show why boosting focuses on misclassified points:
- Points with α = C are misclassified/margin-violating
- Similar to boosting's reweighting

**Active Learning:**
Support vectors are informative points:
- Query points near decision boundary
- Similar to uncertainty sampling

**9. Summary:**

**KKT Conditions provide:**
- **Necessary + sufficient** conditions for convex optimization
- **Identify support vectors**: Points with αᵢ > 0
- **Enable dual formulation**: Kernel trick possible
- **Guarantee optimality**: Solution satisfies all conditions

**For SVM specifically:**
- Complementary slackness identifies 3 types of points
- Dual formulation leads to sparse solution
- Only support vectors needed for prediction
- Kernel trick enables non-linear decision boundaries

**Key Insight:**
KKT conditions transform constrained optimization into system of equations. For SVM, this reveals geometric interpretation: support vectors are the "critical" points that define the decision boundary.`,
    keyPoints: [
      'KKT: 4 conditions (stationarity, primal/dual feasibility, complementary slackness)',
      'Necessary + sufficient for convex problems',
      'SVM: Complementary slackness identifies support vectors',
      'αᵢ = 0: not SV; 0 < αᵢ < C: on margin; αᵢ = C: inside margin',
      'Dual formulation enables kernel trick',
      'Sparsity: only support vectors matter for prediction',
    ],
  },
  {
    id: 'convex-disc-3',
    question:
      'Why does gradient descent work well for non-convex deep learning despite the lack of convexity guarantees? Discuss overparameterization and the optimization landscape.',
    hint: 'Consider the number of parameters vs data points, properties of local minima in overparameterized networks, and implicit regularization.',
    sampleAnswer: `**Why Gradient Descent Works for Non-Convex Deep Learning:**

Despite non-convexity, SGD succeeds in deep learning. This is one of the most surprising and important phenomena in modern ML.

**1. The Paradox:**

**Classical optimization theory:**
- Non-convex → many local minima
- Gradient descent → trapped in bad local minima
- No guarantees of finding global optimum

**Deep learning reality:**
- Highly non-convex loss landscapes
- Simple SGD works remarkably well
- Often reaches solutions with excellent generalization

**Why the disconnect?**

**2. Overparameterization:**

**Definition:** Network has more parameters than training examples.

**Modern networks:**
\`\`\`
ResNet-50: ~25M parameters
Training data: ~1M images
Ratio: 25:1 overparameterized
\`\`\`

**Extreme cases:**
\`\`\`
GPT-3: 175B parameters
Training data: ~500B tokens
Still highly overparameterized at parameter level
\`\`\`

**3. Loss Landscape Properties in Overparameterized Networks:**

**Observation 1: Local Minima are Good**

**Theory (Choromanska et al., 2015):**
For sufficiently wide networks, most local minima have similar loss values close to global minimum.

**Why?**
- High dimensionality creates many descent directions
- Poor local minima become rare
- Most critical points are saddle points (escapable)

**Empirical evidence:**
\`\`\`python
# Train same architecture from different initializations
losses = []
for seed in range(10):
    model = Network (seed=seed)
    final_loss = train (model)
    losses.append (final_loss)

print(f"Final losses: {losses}")
# Output: [0.23, 0.24, 0.23, 0.24, 0.23, ...]
# Very similar despite different local minima!
\`\`\`

**Observation 2: No Bad Local Minima in Overparameterized Regime**

**Theorem (simplified):**
For 2-layer ReLU networks with sufficiently many hidden units (width >> data size), gradient descent finds global minimum.

**Intuition:**
- Each neuron can specialize to few data points
- Enough neurons → can fit all data
- No conflict between objectives

**4. Implicit Regularization of SGD:**

**Key insight:** SGD doesn't just minimize training loss - it implicitly regularizes.

**Mechanism 1: Noise as Regularization**

SGD gradient estimate:
\`\`\`
∇L_batch ≠ ∇L_full
\`\`\`

Noise helps:
- Escape sharp minima
- Find flat minima (better generalization)

**Flat vs Sharp Minima:**
\`\`\`
Flat minimum: Loss changes slowly around w*
  → Robust to perturbations
  → Better generalization

Sharp minimum: Loss changes rapidly
  → Overfits to training data
  → Poor generalization
\`\`\`

**Evidence:**
\`\`\`python
# Sharpness measurement
def sharpness (model, data):
    # Measure eigenvalues of Hessian
    loss = compute_loss (model, data)
    hessian = compute_hessian (loss)
    max_eigenvalue = max_eig (hessian)
    return max_eigenvalue

# SGD finds flatter minima than full-batch GD
sharp_sgd = sharpness (model_sgd, data)
sharp_gd = sharpness (model_gd, data)
print(f"SGD sharpness: {sharp_sgd}")  # Lower
print(f"GD sharpness: {sharp_gd}")    # Higher
\`\`\`

**Mechanism 2: Implicit Bias Toward Simple Solutions**

SGD favors solutions with lower "complexity":
- For linear models: Small-norm solutions
- For neural networks: Solutions with lower effective dimension

**Example:**
Two networks fit training data perfectly:
- Network A: Uses all parameters roughly equally
- Network B: Most weights near zero, few active
- SGD prefers Network B (implicit L2 regularization)

**5. Overparameterization + SGD = Generalization:**

**Double Descent Phenomenon:**

\`\`\`
Test Error
    |
    |     Classical U-curve
    |    /              \\
    |   /                \\___
    |  /                      \\_____
    | /                              \\
    |/____________________\\_____________\\___
     Under    Interpolation    Over
    -param      threshold    -param
              (# params = # data)
\`\`\`

**Surprising observation:**
- Classical: Overparameterization → overfitting
- Modern DL: Further overparameterization → better generalization!

**Explanation:**
- At interpolation threshold: Many solutions fit data, some bad
- In overparameterized regime: More solutions, SGD finds good one

**6. Neural Tangent Kernel (NTK) Theory:**

**Key idea:** In infinite-width limit, neural networks behave like kernel methods.

**Implication:**
- Optimization becomes convex (approximately)
- Explains why gradient descent works

**However:**
- Real networks finite width
- NTK approximation not perfect
- But provides theoretical insight

**7. Mode Connectivity:**

**Observation (Garipov et al., 2018):**
Different solutions found by SGD can be connected by paths of low loss.

**Implication:**
Loss landscape has connected "valleys" of good solutions.

\`\`\`python
# Connect two solutions
w1 = train (model, seed=1)
w2 = train (model, seed=2)

# Linear interpolation
alphas = np.linspace(0, 1, 100)
losses = []
for alpha in alphas:
    w_interp = alpha * w1 + (1-alpha) * w2
    loss = evaluate (w_interp)
    losses.append (loss)

# Often see: losses remain low throughout path!
plt.plot (alphas, losses)
\`\`\`

**8. Practical Factors:**

**Architecture Design:**
- Skip connections (ResNets): Improve gradient flow
- Batch normalization: Smooths loss landscape
- Attention mechanisms: Better optimization

**Initialization:**
- Xavier/He initialization: Start in good region
- Prevents gradient vanishing/exploding

**Hyperparameters:**
- Learning rate schedule: Annealing helps convergence
- Batch size: Affects generalization

**Data Augmentation:**
- Implicitly regularizes
- Prevents memorization

**9. Comparison: Convex vs Non-Convex:**

| Aspect | Convex (LR) | Non-Convex (DL) |
|--------|-------------|-----------------|
| **Theory** | Strong guarantees | Weak/empirical |
| **Optimization** | Guaranteed global | No guarantees |
| **Local minima** | All global | Many, but similar |
| **Initialization** | Doesn't matter | Critical |
| **Why it works** | Convexity | Overparameterization + SGD |
| **Generalization** | Classical theory | Modern phenomena |

**10. Summary:**

**Why SGD works for non-convex deep learning:**

1. **Overparameterization:**
   - More parameters than constraints
   - Many paths to good solutions
   - Local minima have similar quality

2. **Implicit regularization:**
   - SGD noise finds flat minima
   - Flat minima generalize better
   - Implicit bias toward simple solutions

3. **Loss landscape structure:**
   - Bad local minima rare in high dimensions
   - Saddle points escapable
   - Connected valleys of good solutions

4. **Architecture + engineering:**
   - Skip connections, batch norm
   - Good initialization
   - Careful hyperparameter tuning

**Key Insight:**

Deep learning doesn't succeed *despite* non-convexity - it succeeds because:
- Overparameterization creates benign loss landscapes
- SGD implicitly regularizes toward generalizing solutions
- The "right" inductive biases (architecture, optimization) guide search

**Practical Takeaway:**

You don't need convexity for successful optimization. With:
- Sufficient overparameterization
- Stochastic gradient descent
- Good architecture
- Proper initialization

You can reliably train deep networks to excellent solutions, even though the problem is highly non-convex.

This is why deep learning works - not despite theory, but because we've discovered that overparameterized non-convex optimization has surprisingly good properties!`,
    keyPoints: [
      'Overparameterization: more parameters than data',
      'Local minima in overparameterized networks have similar quality',
      'SGD implicitly regularizes toward flat minima',
      'Flat minima generalize better than sharp minima',
      'Bad local minima rare in high dimensions',
      'Architecture design + initialization + SGD → reliable training',
    ],
  },
];
