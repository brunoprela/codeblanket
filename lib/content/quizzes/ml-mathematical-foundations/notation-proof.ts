/**
 * Quiz questions for Mathematical Notation & Proof section
 */

export const notationproofQuiz = [
  {
    id: 'dq1-summation-notation-ml',
    question:
      'Explain how summation notation (Σ) is used in machine learning to express loss functions and gradient computation. Provide specific examples of Mean Squared Error (MSE) and cross-entropy loss written in summation notation, then show how to translate these into vectorized NumPy code. Discuss why understanding summation notation is crucial for reading ML research papers.',
    sampleAnswer: `Summation notation is the backbone of expressing mathematical operations over datasets in machine learning. Understanding it is essential for both implementing algorithms and reading research papers.

**Mean Squared Error (MSE)**:

**Mathematical notation**:
\\[
\\text{MSE} = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2
\\]

This reads as: "Sum the squared differences between true values \\(y_i\\) and predictions \\(\\hat{y}_i\\) for all n samples, then divide by n."

**Components**:
- \\(\\sum_{i=1}^{n}\\): Sum from sample 1 to sample n
- \\(y_i\\): True label for sample i
- \\(\\hat{y}_i\\): Predicted value for sample i
- \\((y_i - \\hat{y}_i)^2\\): Squared error for sample i

\`\`\`python
import numpy as np

# Example data
y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0, 2.0, 8.0])
n = len(y_true)

# Summation notation approach (explicit loop)
mse_loop = 0
for i in range(n):
    mse_loop += (y_true[i] - y_pred[i])**2
mse_loop = mse_loop / n

# Vectorized NumPy (translates Σ to array operations)
mse_vectorized = np.mean((y_true - y_pred)**2)

print(f"MSE (loop): {mse_loop:.4f}")
print(f"MSE (vectorized): {mse_vectorized:.4f}")
print(f"Match: {np.isclose(mse_loop, mse_vectorized)}")

# Breaking down the vectorized operation:
print("\\nStep-by-step vectorization:")
print(f"1. Differences (y - ŷ): {y_true - y_pred}")
print(f"2. Squared: {(y_true - y_pred)**2}")
print(f"3. Sum (Σ): {np.sum((y_true - y_pred)**2)}")
print(f"4. Divide by n: {np.sum((y_true - y_pred)**2) / n}")
\`\`\`

**Cross-Entropy Loss (Binary Classification)**:

**Mathematical notation**:
\\[
\\text{Loss} = -\\frac{1}{n} \\sum_{i=1}^{n} [y_i \\log(\\hat{y}_i) + (1-y_i) \\log(1-\\hat{y}_i)]
\\]

This reads as: "For each sample, compute the log loss based on whether the true label is 1 or 0, sum all losses, then divide by n."

\`\`\`python
# Binary classification
y_true = np.array([1, 0, 1, 1, 0])
y_pred = np.array([0.9, 0.1, 0.8, 0.7, 0.2])  # Predicted probabilities
n = len(y_true)

# Summation notation approach (explicit loop)
loss_loop = 0
for i in range(n):
    loss_loop += y_true[i] * np.log(y_pred[i]) + (1 - y_true[i]) * np.log(1 - y_pred[i])
loss_loop = -loss_loop / n

# Vectorized NumPy
loss_vectorized = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

print(f"\\nCross-Entropy Loss (loop): {loss_loop:.4f}")
print(f"Cross-Entropy Loss (vectorized): {loss_vectorized:.4f}")
print(f"Match: {np.isclose(loss_loop, loss_vectorized)}")
\`\`\`

**Gradient Computation with Summation**:

**Mathematical notation for MSE gradient**:
\\[
\\frac{\\partial \\text{MSE}}{\\partial w_j} = \\frac{2}{n} \\sum_{i=1}^{n} (\\hat{y}_i - y_i) x_{ij}
\\]

Where \\(x_{ij}\\) is feature j for sample i.

\`\`\`python
# Linear regression: ŷ = Xw
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])  # (n_samples=4, n_features=2)
y = np.array([5, 11, 17, 23])
w = np.array([1.5, 2.0])

# Predictions
y_pred = X @ w

# Gradient: ∂MSE/∂w_j = (2/n) Σ(ŷ_i - y_i) x_ij
n = len(y)

# Loop version (following summation notation exactly)
gradient_loop = np.zeros(2)
for i in range(n):
    for j in range(2):  # For each feature
        gradient_loop[j] += (y_pred[i] - y[i]) * X[i, j]
gradient_loop = (2 / n) * gradient_loop

# Vectorized version (matrix notation)
gradient_vectorized = (2 / n) * X.T @ (y_pred - y)

print("\\nGradient computation:")
print(f"Loop version: {gradient_loop}")
print(f"Vectorized version: {gradient_vectorized}")
print(f"Match: {np.allclose(gradient_loop, gradient_vectorized)}")
\`\`\`

**Double Summation (Nested Sums)**:

Used for operations over matrices, like computing pairwise distances.

**Mathematical notation**:
\\[
\\sum_{i=1}^{n} \\sum_{j=1}^{m} A_{ij}
\\]

This sums all elements in matrix A.

\`\`\`python
# Sum all elements in a matrix
A = np.array([[1, 2, 3],
              [4, 5, 6]])

# Double loop (following notation exactly)
total_loop = 0
for i in range(A.shape[0]):  # n rows
    for j in range(A.shape[1]):  # m columns
        total_loop += A[i, j]

# Vectorized
total_vectorized = np.sum(A)

print(f"\\nDouble summation:")
print(f"Loop: {total_loop}")
print(f"Vectorized: {total_vectorized}")
print(f"Match: {total_loop == total_vectorized}")
\`\`\`

**Why This Matters for Research Papers**:

Research papers use summation notation extensively. Being able to translate it to code is critical:

1. **Compact representation**: \\(\\sum_{i=1}^{n}\\) is clearer than describing loops in words
2. **Mathematical rigor**: Summation notation is precise and unambiguous
3. **Implementation**: Directly maps to vectorized NumPy operations
4. **Debugging**: Understanding the notation helps verify your implementation

**Example from a paper**:

"The attention mechanism is computed as:
\\[
\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V
\\]
where the softmax is applied row-wise."

Without understanding summation notation (implicit in softmax), you can't implement this correctly.`,
    keyPoints: [
      'Summation notation Σ expresses operations over all samples in a dataset',
      'MSE: (1/n)Σ(y_i - ŷ_i)² sums squared errors across all samples',
      'Cross-entropy: -(1/n)Σ[y_i log(ŷ_i) + (1-y_i)log(1-ŷ_i)] for binary classification',
      'Summation translates directly to vectorized NumPy: Σ → np.sum() or np.mean()',
      'Understanding summation notation essential for reading ML papers and implementing algorithms',
    ],
  },
  {
    id: 'dq2-proof-convergence',
    question:
      'In optimization theory, a key result states: "For convex functions with L-Lipschitz continuous gradients, gradient descent with learning rate α ≤ 1/L converges to the global minimum." Explain what this theorem means in practical terms, provide a proof sketch or intuition for why it works, and demonstrate with code how violating the learning rate condition (α > 1/L) causes divergence. Discuss the implications for choosing learning rates in deep learning.',
    sampleAnswer: `This theorem provides a mathematical guarantee for gradient descent convergence under specific conditions. Understanding it helps explain why learning rates matter and how to choose them systematically.

**Theorem Statement Breakdown**:

**Convex function**: f is convex if \\(f(\\lambda x + (1-\\lambda)y) \\leq \\lambda f(x) + (1-\\lambda)f(y)\\) for all \\(\\lambda \\in [0,1]\\).

**Practical meaning**: No local minima (only one global minimum), like a bowl shape.

**L-Lipschitz continuous gradient**: \\(\\|\\nabla f(x) - \\nabla f(y)\\| \\leq L\\|x - y\\|\\)

**Practical meaning**: The gradient doesn't change too rapidly. L is the "smoothness constant" - how much the slope can vary.

**Learning rate condition**: \\(\\alpha \\leq 1/L\\)

**Practical meaning**: Step size must be small enough relative to function's curvature.

**Convergence**: The sequence of parameters converges to the optimal value: \\(\\lim_{t \\to \\infty} \\theta_t = \\theta^*\\)

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt

# Simple convex function: f(x) = x²
def f(x):
    """Convex quadratic function"""
    return x**2

def grad_f(x):
    """Gradient: f'(x) = 2x"""
    return 2*x

# For f(x) = x², the Lipschitz constant L = 2
# (gradient is 2x, so |grad_f(x) - grad_f(y)| = |2x - 2y| = 2|x - y|)
L = 2

print("Function: f(x) = x²")
print(f"Gradient: f'(x) = 2x")
print(f"Lipschitz constant L = {L}")
print(f"Theorem says: Use α ≤ 1/L = {1/L}")
print(f"\\nOptimal point: x* = 0")
\`\`\`

**Proof Intuition**:

The key idea is that for a convex function with L-Lipschitz gradient, we can bound how much the function can increase in one gradient step.

**Mathematical insight**:

For convex f with L-Lipschitz gradient:
\\[
f(x - \\alpha \\nabla f(x)) \\leq f(x) - \\alpha \\|\\nabla f(x)\\|^2 + \\frac{\\alpha^2 L}{2} \\|\\nabla f(x)\\|^2
\\]

This says: stepping in the negative gradient direction decreases the function value (middle term) but there's a penalty for taking too large a step (last term).

For descent to be guaranteed:
\\[
-\\alpha + \\frac{\\alpha^2 L}{2} < 0
\\]
\\[
\\alpha(\\frac{\\alpha L}{2} - 1) < 0
\\]
\\[
\\alpha < \\frac{2}{L}
\\]

For guaranteed convergence (stronger result), we need \\(\\alpha \\leq \\frac{1}{L}\\).

\`\`\`python
# Visualize the descent condition
alphas = np.linspace(0, 2/L, 1000)
descent_guaranteed = -alphas + (alphas**2 * L / 2)

plt.figure(figsize=(10, 6))
plt.plot(alphas, descent_guaranteed, linewidth=2)
plt.axhline(y=0, color='r', linestyle='--', label='Descent boundary')
plt.axvline(x=1/L, color='g', linestyle='--', label=f'α = 1/L = {1/L}')
plt.axvline(x=2/L, color='orange', linestyle='--', label=f'α = 2/L = {2/L}')
plt.xlabel('Learning Rate α')
plt.ylabel('Change in function value')
plt.title('Gradient Descent: Function Decrease vs Learning Rate')
plt.legend()
plt.grid(True, alpha=0.3)
plt.fill_between(alphas, descent_guaranteed, 0, where=(descent_guaranteed<0), alpha=0.3, color='green', label='Guaranteed descent')
plt.show()
\`\`\`

**Demonstration: Convergence vs Divergence**:

\`\`\`python
def gradient_descent(f, grad_f, x0, alpha, n_iterations):
    """Run gradient descent"""
    trajectory = [x0]
    x = x0
    
    for _ in range(n_iterations):
        x = x - alpha * grad_f(x)
        trajectory.append(x)
        
        # Stop if diverging
        if abs(x) > 1e10:
            break
    
    return np.array(trajectory)

# Test different learning rates
x0 = 10.0
n_iterations = 50

learning_rates = [
    0.3,    # < 1/L = 0.5 → should converge
    0.5,    # = 1/L → should converge
    0.6,    # > 1/L but < 2/L → might converge slowly or oscillate
    0.8,    # > 1/L → will oscillate or diverge
    1.2,    # >> 1/L → will diverge
]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, alpha in enumerate(learning_rates):
    trajectory = gradient_descent(f, grad_f, x0, alpha, n_iterations)
    
    ax = axes[idx]
    iterations = range(len(trajectory))
    ax.plot(iterations, trajectory, 'bo-', markersize=3, linewidth=1)
    ax.axhline(y=0, color='r', linestyle='--', label='Optimal x*=0')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('x value')
    ax.set_title(f'α = {alpha} ({"✓ converges" if alpha <= 1/L else "✗ may diverge"})')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Print convergence status
    final_x = trajectory[-1]
    converged = abs(final_x) < 1e-6
    print(f"α = {alpha}: Final x = {final_x:.6f}, Converged: {converged}")

plt.tight_layout()
plt.show()
\`\`\`

**Detailed Analysis**:

\`\`\`python
# Analyze convergence rate
def analyze_convergence(alpha, x0=10, n_iter=100):
    """Analyze convergence behavior"""
    x = x0
    errors = []
    
    for t in range(n_iter):
        errors.append(abs(x))  # Distance from optimal x*=0
        x = x - alpha * grad_f(x)
        
        if abs(x) > 1e10:  # Diverged
            return errors, False
    
    return errors, abs(x) < 1e-6

print("\\nConvergence Analysis:\\n")
print("Learning Rate | Converged | Final Error | Iterations to 1e-6")
print("-" * 65)

test_alphas = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
for alpha in test_alphas:
    errors, converged = analyze_convergence(alpha, n_iter=200)
    final_error = errors[-1] if len(errors) > 0 else float('inf')
    
    # Find iteration where error < 1e-6
    iters_to_converge = next((i for i, e in enumerate(errors) if e < 1e-6), None)
    
    status = "✓ Yes" if converged else "✗ No"
    iter_str = str(iters_to_converge) if iters_to_converge else "N/A"
    
    print(f"    {alpha:.1f}       |   {status}    | {final_error:.2e}  |      {iter_str}")
\`\`\`

**Implications for Deep Learning**:

1. **Learning rate scheduling**: Start with a safe learning rate, then adapt
2. **Adam optimizer**: Adapts learning rates per parameter based on gradient history
3. **Learning rate warmup**: Start small, gradually increase to target value
4. **Non-convex functions**: Deep networks are non-convex, so this theorem doesn't directly apply, but insights still useful

\`\`\`python
# Deep learning example (conceptual)
print("\\nDeep Learning Learning Rates:")
print("- Small networks (few layers): α ∈ [0.01, 0.1]")
print("- Large networks (ResNet, Transformers): α ∈ [0.001, 0.01]")
print("- With momentum/Adam: Can use larger α (optimizer stabilizes)")
print("- Rule of thumb: Start with 0.001 for Adam, 0.1 for SGD")
print("- Always use learning rate schedule (decay or cosine annealing)")
\`\`\`

**Key Takeaways**:

- Mathematical proofs provide guarantees and insights
- Lipschitz constant L determines maximum safe learning rate
- Violating the condition (α > 1/L) can cause oscillation or divergence
- In practice, use adaptive optimizers (Adam) that handle this automatically
- For research: understanding proofs helps design better algorithms`,
    keyPoints: [
      'Theorem: For convex f with L-Lipschitz gradient, α ≤ 1/L guarantees convergence',
      'Lipschitz constant L measures how fast gradient changes',
      'Proof intuition: Step size must balance descent vs overshoot',
      'Violating α > 1/L causes oscillation or divergence',
      'Deep learning: Use adaptive optimizers (Adam) or learning rate schedules',
    ],
  },
  {
    id: 'dq3-notation-backpropagation',
    question:
      'The backpropagation algorithm is often expressed using chain rule notation. For a 2-layer neural network with weights W1, W2 and ReLU activation, write out the complete forward pass and backward pass using proper mathematical notation (including subscripts and superscripts). Then translate this notation into NumPy code. Explain how understanding the notation helps implement custom neural network layers and debug gradient computation errors.',
    sampleAnswer: `Backpropagation is the chain rule applied systematically to compute gradients in neural networks. Understanding the notation is essential for implementing custom architectures and debugging gradient errors.

**2-Layer Neural Network Architecture**:

**Network structure**:
- Input: \\(x \\in \\mathbb{R}^{d}\\) (d features)
- Hidden layer: \\(h \\in \\mathbb{R}^{m}\\) (m neurons)
- Output: \\(\\hat{y} \\in \\mathbb{R}\\) (scalar for regression)
- Loss: Mean Squared Error

**Mathematical Notation**:

**Forward Pass**:

1. First layer (linear transformation):
   \\[z^{(1)} = W^{(1)}x + b^{(1)}\\]
   where \\(W^{(1)} \\in \\mathbb{R}^{m \\times d}\\), \\(b^{(1)} \\in \\mathbb{R}^{m}\\)

2. ReLU activation:
   \\[h = \\text{ReLU}(z^{(1)}) = \\max(0, z^{(1)})\\]

3. Second layer (linear):
   \\[z^{(2)} = W^{(2)}h + b^{(2)}\\]
   where \\(W^{(2)} \\in \\mathbb{R}^{1 \\times m}\\), \\(b^{(2)} \\in \\mathbb{R}\\)

4. Output (no activation for regression):
   \\[\\hat{y} = z^{(2)}\\]

5. Loss:
   \\[L = (y - \\hat{y})^2\\]

**Backward Pass (Chain Rule)**:

We need: \\(\\frac{\\partial L}{\\partial W^{(1)}}\\), \\(\\frac{\\partial L}{\\partial b^{(1)}}\\), \\(\\frac{\\partial L}{\\partial W^{(2)}}\\), \\(\\frac{\\partial L}{\\partial b^{(2)}}\\)

**Layer 2 gradients**:

1. \\(\\frac{\\partial L}{\\partial \\hat{y}} = 2(\\hat{y} - y)\\)

2. \\(\\frac{\\partial L}{\\partial W^{(2)}} = \\frac{\\partial L}{\\partial \\hat{y}} \\cdot \\frac{\\partial \\hat{y}}{\\partial W^{(2)}} = 2(\\hat{y} - y) \\cdot h^T\\)

3. \\(\\frac{\\partial L}{\\partial b^{(2)}} = \\frac{\\partial L}{\\partial \\hat{y}} = 2(\\hat{y} - y)\\)

4. \\(\\frac{\\partial L}{\\partial h} = \\frac{\\partial L}{\\partial \\hat{y}} \\cdot \\frac{\\partial \\hat{y}}{\\partial h} = 2(\\hat{y} - y) \\cdot W^{(2)T}\\)

**Layer 1 gradients (chain rule through ReLU)**:

5. \\(\\frac{\\partial L}{\\partial z^{(1)}} = \\frac{\\partial L}{\\partial h} \\odot \\frac{\\partial h}{\\partial z^{(1)}}\\)

   where \\(\\frac{\\partial h}{\\partial z^{(1)}} = \\mathbb{1}_{z^{(1)} > 0}\\) (ReLU derivative)

6. \\(\\frac{\\partial L}{\\partial W^{(1)}} = \\frac{\\partial L}{\\partial z^{(1)}} \\cdot x^T\\)

7. \\(\\frac{\\partial L}{\\partial b^{(1)}} = \\frac{\\partial L}{\\partial z^{(1)}}\\)

\`\`\`python
import numpy as np

class TwoLayerNet:
    """2-layer neural network with ReLU activation"""
    
    def __init__(self, input_dim, hidden_dim):
        """
        Initialize weights and biases
        
        Notation:
        - W1: W^(1) ∈ R^(hidden_dim × input_dim)
        - b1: b^(1) ∈ R^(hidden_dim)
        - W2: W^(2) ∈ R^(1 × hidden_dim)
        - b2: b^(2) ∈ R^(1)
        """
        # Initialize with small random values
        self.W1 = np.random.randn(hidden_dim, input_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(1, hidden_dim) * 0.01
        self.b2 = np.zeros(1)
        
        # Cache for backward pass
        self.cache = {}
    
    def relu(self, z):
        """ReLU activation: h = max(0, z)"""
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        """ReLU derivative: 1 if z > 0, else 0"""
        return (z > 0).astype(float)
    
    def forward(self, x):
        """
        Forward pass
        
        Notation:
        - z1 = W^(1)x + b^(1)
        - h = ReLU(z1)
        - z2 = W^(2)h + b^(2)
        - ŷ = z2
        """
        # Layer 1
        z1 = self.W1 @ x + self.b1  # z^(1)
        h = self.relu(z1)            # h = ReLU(z^(1))
        
        # Layer 2
        z2 = self.W2 @ h + self.b2   # z^(2)
        y_pred = z2                  # ŷ
        
        # Cache for backward pass
        self.cache = {
            'x': x,
            'z1': z1,
            'h': h,
            'z2': z2,
            'y_pred': y_pred
        }
        
        return y_pred
    
    def backward(self, y_true):
        """
        Backward pass (backpropagation)
        
        Compute all gradients using chain rule:
        - ∂L/∂W^(2), ∂L/∂b^(2)
        - ∂L/∂W^(1), ∂L/∂b^(1)
        """
        # Retrieve cached values
        x = self.cache['x',]
        z1 = self.cache['z1',]
        h = self.cache['h',]
        y_pred = self.cache['y_pred',]
        
        # Gradient of loss w.r.t. output
        # ∂L/∂ŷ = 2(ŷ - y)
        dL_dy_pred = 2 * (y_pred - y_true)
        
        # Layer 2 gradients
        # ∂L/∂W^(2) = ∂L/∂ŷ · ∂ŷ/∂W^(2) = dL_dy_pred · h^T
        dL_dW2 = dL_dy_pred * h.reshape(1, -1)  # Outer product
        
        # ∂L/∂b^(2) = ∂L/∂ŷ
        dL_db2 = dL_dy_pred
        
        # Gradient w.r.t. hidden layer
        # ∂L/∂h = ∂L/∂ŷ · ∂ŷ/∂h = dL_dy_pred · W^(2)^T
        dL_dh = dL_dy_pred * self.W2.T  # Shape: (hidden_dim,)
        dL_dh = dL_dh.flatten()
        
        # Gradient w.r.t. z1 (before ReLU)
        # ∂L/∂z^(1) = ∂L/∂h ⊙ ∂h/∂z^(1)
        # where ∂h/∂z^(1) = 1 if z^(1) > 0, else 0
        dL_dz1 = dL_dh * self.relu_derivative(z1)
        
        # Layer 1 gradients
        # ∂L/∂W^(1) = ∂L/∂z^(1) · x^T
        dL_dW1 = dL_dz1.reshape(-1, 1) @ x.reshape(1, -1)
        
        # ∂L/∂b^(1) = ∂L/∂z^(1)
        dL_db1 = dL_dz1
        
        gradients = {
            'dW1': dL_dW1,
            'db1': dL_db1,
            'dW2': dL_dW2,
            'db2': dL_db2
        }
        
        return gradients
    
    def loss(self, y_true, y_pred):
        """MSE loss: L = (y - ŷ)²"""
        return (y_true - y_pred)**2

# Example usage
print("2-Layer Neural Network with Backpropagation\\n")
print("=" * 60)

# Initialize network
input_dim = 3
hidden_dim = 4
net = TwoLayerNet(input_dim, hidden_dim)

# Sample data
x = np.array([1.0, 2.0, 3.0])
y = np.array([10.0])

# Forward pass
y_pred = net.forward(x)
loss = net.loss(y, y_pred)

print(f"Input x: {x}")
print(f"True y: {y[0]}")
print(f"Predicted ŷ: {y_pred[0]:.4f}")
print(f"Loss: {loss[0]:.4f}\\n")

# Backward pass
gradients = net.backward(y)

print("Gradients:")
print(f"  ∂L/∂W^(2) shape: {gradients['dW2',].shape}")
print(f"  ∂L/∂b^(2) shape: {gradients['db2',].shape}")
print(f"  ∂L/∂W^(1) shape: {gradients['dW1',].shape}")
print(f"  ∂L/∂b^(1) shape: {gradients['db1',].shape}")
\`\`\`

**Gradient Checking (Verify Implementation)**:

Numerical gradient approximation: \\(\\frac{\\partial L}{\\partial w} \\approx \\frac{L(w + \\epsilon) - L(w - \\epsilon)}{2\\epsilon}\\)

\`\`\`python
def numerical_gradient(net, x, y, param_name, epsilon=1e-5):
    """
    Compute numerical gradient using finite differences
    
    ∂L/∂w ≈ [L(w+ε) - L(w-ε)] / 2ε
    """
    # Get parameter
    if param_name == 'W1':
        param = net.W1
    elif param_name == 'b1':
        param = net.b1
    elif param_name == 'W2':
        param = net.W2
    elif param_name == 'b2':
        param = net.b2
    
    numerical_grad = np.zeros_like(param)
    it = np.nditer(param, flags=['multi_index',], op_flags=['readwrite',])
    
    while not it.finished:
        idx = it.multi_index
        old_value = param[idx]
        
        # f(w + ε)
        param[idx] = old_value + epsilon
        y_pred_plus = net.forward(x)
        loss_plus = net.loss(y, y_pred_plus)[0]
        
        # f(w - ε)
        param[idx] = old_value - epsilon
        y_pred_minus = net.forward(x)
        loss_minus = net.loss(y, y_pred_minus)[0]
        
        # Numerical gradient
        numerical_grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)
        
        # Restore
        param[idx] = old_value
        it.iternext()
    
    return numerical_grad

# Verify gradients
print("\\nGradient Checking:\\n")

# Compute analytical gradients
y_pred = net.forward(x)
gradients = net.backward(y)

# Check each parameter
for param_name in ['W1', 'b1', 'W2', 'b2',]:
    analytical_grad = gradients[f'd{param_name}',]
    numerical_grad = numerical_gradient(net, x, y, param_name)
    
    # Compute relative error
    diff = np.linalg.norm(analytical_grad - numerical_grad)
    sum_norm = np.linalg.norm(analytical_grad) + np.linalg.norm(numerical_grad)
    relative_error = diff / (sum_norm + 1e-8)
    
    status = "✓ PASS" if relative_error < 1e-7 else "✗ FAIL"
    print(f"  {param_name}: relative error = {relative_error:.2e} {status}")
\`\`\`

**Key Insights**:

1. **Notation clarity**: Superscripts (layer index), subscripts (element index) prevent confusion
2. **Chain rule**: Gradients flow backward through layers: \\(\\frac{\\partial L}{\\partial W^{(1)}} = \\frac{\\partial L}{\\partial z^{(2)}} \\cdot \\frac{\\partial z^{(2)}}{ \\partial h} \\cdot \\frac{\\partial h}{\\partial z^{(1)}} \\cdot \\frac{\\partial z^{(1)}}{\\partial W^{(1)}}\\)
3. **Gradient checking**: Always verify custom layers with numerical gradients
4. **Debugging**: Understanding notation helps identify where gradients vanish or explode

This foundation extends to any architecture: CNNs, RNNs, Transformers all use the same backpropagation principle.`,
    keyPoints: [
      'Forward: z^(1)=W^(1)x+b^(1), h=ReLU(z^(1)), z^(2)=W^(2)h+b^(2), ŷ=z^(2)',
      'Backward: Chain rule computes ∂L/∂W^(2), ∂L/∂W^(1) by propagating gradients backward',
      'ReLU gradient: ∂h/∂z = 1 if z>0, else 0 (creates "dead neurons" if z always ≤0)',
      'Gradient checking: Verify analytical gradients match numerical approximation',
      'Understanding notation essential for implementing custom layers and debugging',
    ],
  },
];
