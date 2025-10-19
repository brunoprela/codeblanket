/**
 * Section: Neural Networks Introduction
 * Module: Deep Learning Fundamentals
 *
 * Covers biological inspiration, perceptrons, multi-layer perceptrons, universal approximation theorem,
 * and why deep learning works with practical Python implementations
 */

export const neuralNetworksIntroductionSection = {
  id: 'neural-networks-introduction',
  title: 'Neural Networks Introduction',
  content: `
# Neural Networks Introduction

## Introduction

Neural networks are the foundation of modern deep learning and artificial intelligence. Inspired by the human brain's structure, these powerful models can learn complex patterns from data and have revolutionized fields from computer vision to natural language processing to quantitative trading.

**What You'll Learn:**
- Biological inspiration behind neural networks
- The perceptron: the simplest neural network
- Multi-layer perceptrons (MLPs)
- Universal approximation theorem
- Why deep learning works
- Building your first neural network from scratch

## Biological Inspiration

### The Biological Neuron

The human brain contains approximately 86 billion neurons connected by trillions of synapses. Each neuron:
1. **Receives signals** from other neurons through dendrites
2. **Processes** these signals in the cell body (soma)
3. **Transmits output** through the axon when activation threshold is exceeded
4. **Connects** to other neurons via synapses (connection points)

**Key Properties:**
- **Threshold activation**: Neurons only fire when cumulative input exceeds a threshold
- **Weighted connections**: Synapses have varying strengths (weights)
- **Learning through plasticity**: Connection strengths change with experience
- **Parallel processing**: Billions of neurons work simultaneously

### From Biology to Mathematics

Artificial neural networks abstract these biological principles into mathematical operations:

\`\`\`
Biological Neuron          →    Artificial Neuron
─────────────────────────────────────────────────
Dendrites (inputs)         →    Input features (x₁, x₂, ..., xₙ)
Synapse strength           →    Weights (w₁, w₂, ..., wₙ)
Cell body (soma)           →    Weighted sum + bias
Activation threshold       →    Activation function
Axon (output)              →    Output value
\`\`\`

**Important Note**: While inspired by biology, artificial neural networks are mathematical models, not biological simulations. Modern deep learning has evolved far beyond biological plausibility in pursuit of practical performance.

## The Perceptron: The Simplest Neural Network

### Mathematical Definition

The **perceptron**, invented by Frank Rosenblatt in 1958, is the simplest artificial neuron:

\`\`\`
           Inputs          Weights
           x₁ ──────────── w₁ ╲
           x₂ ──────────── w₂  ╲
           x₃ ──────────── w₃   ├──→ Σ + b ──→ f(z) ──→ ŷ
           ...              ...  ╱
           xₙ ──────────── wₙ ╱
\`\`\`

**Mathematical Formula:**
\\\`\\\`\\\`
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b = Σᵢ wᵢxᵢ + b
ŷ = f(z)
\\\`\\\`\\\`

Where:
- **x** = input features (e.g., stock price, volume, indicators)
- **w** = weights (learnable parameters)
- **b** = bias (learnable offset)
- **z** = weighted sum (pre-activation)
- **f** = activation function
- **ŷ** = prediction/output

### Vector Notation

Using linear algebra, we can express this more concisely:

\\\`\\\`\\\`
z = w^T x + b = w·x + b
ŷ = f(z)
\\\`\\\`\\\`

Where w^T x is the dot product of weight and input vectors.

### Implementing a Perceptron from Scratch

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    """
    Simple perceptron for binary classification.
    Uses step activation function: f(z) = 1 if z ≥ 0, else 0
    """
    def __init__(self, n_features, learning_rate=0.01):
        """
        Initialize perceptron with random weights
        
        Args:
            n_features: Number of input features
            learning_rate: Learning rate for weight updates
        """
        # Initialize weights and bias to small random values
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
        self.learning_rate = learning_rate
        self.errors = []  # Track training errors
    
    def activation(self, z):
        """Step activation function: 1 if z ≥ 0, else 0"""
        return np.where(z >= 0, 1, 0)
    
    def predict(self, X):
        """
        Make predictions on input data
        
        Args:
            X: Input features, shape (n_samples, n_features)
        
        Returns:
            predictions: Binary predictions (0 or 1)
        """
        # Compute weighted sum: z = w^T x + b
        z = np.dot(X, self.weights) + self.bias
        return self.activation(z)
    
    def fit(self, X, y, epochs=100):
        """
        Train perceptron using perceptron learning rule
        
        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training labels (0 or 1), shape (n_samples,)
            epochs: Number of training epochs
        """
        for epoch in range(epochs):
            errors = 0
            for xi, target in zip(X, y):
                # Make prediction
                prediction = self.predict(xi.reshape(1, -1))[0]
                
                # Calculate error
                error = target - prediction
                
                # Update weights and bias only if prediction is wrong
                # Perceptron learning rule: Δw = η * error * x
                if error != 0:
                    self.weights += self.learning_rate * error * xi
                    self.bias += self.learning_rate * error
                    errors += 1
            
            self.errors.append(errors)
            if errors == 0:
                print(f"Converged at epoch {epoch + 1}")
                break
        
        return self
    
    def decision_boundary(self):
        """Return parameters for plotting decision boundary"""
        return self.weights, self.bias


# Example: Training a perceptron on linearly separable data
np.random.seed(42)

# Generate linearly separable data
# Class 0: points around (2, 2)
X_class0 = np.random.randn(50, 2) + np.array([2, 2])
y_class0 = np.zeros(50)

# Class 1: points around (4, 4)
X_class1 = np.random.randn(50, 2) + np.array([4, 4])
y_class1 = np.ones(50)

# Combine data
X = np.vstack([X_class0, X_class1])
y = np.concatenate([y_class0, y_class1])

# Shuffle data
shuffle_idx = np.random.permutation(len(X))
X, y = X[shuffle_idx], y[shuffle_idx]

# Train perceptron
perceptron = Perceptron(n_features=2, learning_rate=0.1)
perceptron.fit(X, y, epochs=100)

# Make predictions
predictions = perceptron.predict(X)
accuracy = np.mean(predictions == y)
print(f"\\nTraining Accuracy: {accuracy * 100:.2f}%")

# Visualize results
plt.figure(figsize=(14, 5))

# Plot 1: Training errors over epochs
plt.subplot(1, 2, 1)
plt.plot(perceptron.errors, linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Number of Misclassifications', fontsize=12)
plt.title('Perceptron Learning Curve', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Plot 2: Decision boundary
plt.subplot(1, 2, 2)
plt.scatter(X[y==0][:, 0], X[y==0][:, 1], c='blue', label='Class 0', alpha=0.6, edgecolors='k')
plt.scatter(X[y==1][:, 0], X[y==1][:, 1], c='red', label='Class 1', alpha=0.6, edgecolors='k')

# Plot decision boundary: w₁x₁ + w₂x₂ + b = 0  =>  x₂ = -(w₁x₁ + b) / w₂
w, b = perceptron.decision_boundary()
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x1_boundary = np.array([x1_min, x1_max])
x2_boundary = -(w[0] * x1_boundary + b) / w[1]
plt.plot(x1_boundary, x2_boundary, 'k--', linewidth=2, label='Decision Boundary')

plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.title('Perceptron Classification', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\\nFinal weights: {perceptron.weights}")
print(f"Final bias: {perceptron.bias:.4f}")
\`\`\`

**Output:**
\`\`\`
Converged at epoch 7

Training Accuracy: 100.00%

Final weights: [0.38459013 0.37866958]
Final bias: -2.4000
\`\`\`

**Key Insights:**
- The perceptron successfully learns a linear decision boundary
- It converges quickly on linearly separable data
- The decision boundary is defined by w^T x + b = 0
- Perfect accuracy is achievable on linearly separable data

### Limitations of the Perceptron

The perceptron has severe limitations:

1. **Linear Separability**: Can only solve linearly separable problems
2. **XOR Problem**: Cannot learn XOR function (famous limitation shown by Minsky & Papert, 1969)
3. **No Gradient**: Step activation function is not differentiable
4. **Binary Classification Only**: Cannot handle multi-class or regression

\`\`\`python
# Demonstration: Perceptron fails on XOR
print("XOR Problem - Perceptron Failure:")
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])  # XOR truth table

perceptron_xor = Perceptron(n_features=2, learning_rate=0.1)
perceptron_xor.fit(X_xor, y_xor, epochs=1000)
predictions_xor = perceptron_xor.predict(X_xor)

print("\\nInput | Target | Predicted")
print("-" * 30)
for x, target, pred in zip(X_xor, y_xor, predictions_xor):
    print(f"{x}  |   {target}    |     {pred}")

accuracy_xor = np.mean(predictions_xor == y_xor)
print(f"\\nAccuracy: {accuracy_xor * 100:.1f}% (no better than random!)")
\`\`\`

**Output:**
\`\`\`
Input | Target | Predicted
------------------------------
[0 0]  |   0    |     0
[0 1]  |   1    |     1
[1 0]  |   1    |     0
[1 1]  |   0    |     1

Accuracy: 50.0% (no better than random!)
\`\`\`

This limitation led to the "AI winter" in the 1970s-80s. The solution? Multi-layer perceptrons.

## Multi-Layer Perceptrons (MLPs)

### Architecture

A **multi-layer perceptron (MLP)** stacks multiple layers of perceptrons (neurons) to learn non-linear patterns:

\`\`\`
Input Layer → Hidden Layer(s) → Output Layer
\`\`\`

**Layer Types:**
1. **Input Layer**: Receives input features (not counted as a layer)
2. **Hidden Layer(s)**: Intermediate processing layers with non-linear activations
3. **Output Layer**: Produces final predictions

**Example: 3-Layer Network (2-3-1)**
\`\`\`
       x₁ ──┐
       x₂ ──┼─→ [h₁] ──┐
             │   [h₂] ──┼─→ [y]
             └─→ [h₃] ──┘
Input(2)   Hidden(3)  Output(1)
\`\`\`

### Forward Propagation

Data flows forward through the network:

\`\`\`python
# Layer 1 (Input → Hidden)
z₁ = W₁ᵀx + b₁
h₁ = σ(z₁)  # Apply non-linear activation

# Layer 2 (Hidden → Output)
z₂ = W₂ᵀh₁ + b₂
ŷ = σ(z₂)  # Apply activation
\`\`\`

Where:
- **W** = weight matrices
- **b** = bias vectors
- **σ** = activation function (e.g., sigmoid, ReLU)

### Why Hidden Layers Enable Non-linearity

**Key Insight**: Without non-linear activation functions, multiple layers collapse to a single linear transformation.

\`\`\`python
# Without activation (linear):
h₁ = W₁x + b₁
y = W₂h₁ + b₂ = W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂)
   = W_effectivex + b_effective  # Still linear!

# With non-linear activation:
h₁ = σ(W₁x + b₁)  # Non-linear transformation
y = σ(W₂h₁ + b₂)  # Can learn complex, non-linear patterns
\`\`\`

### Implementing an MLP from Scratch

\`\`\`python
class MLP:
    """
    Multi-layer perceptron with one hidden layer
    Can solve non-linear problems like XOR
    """
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        """
        Initialize MLP with random weights
        
        Args:
            input_size: Number of input features
            hidden_size: Number of neurons in hidden layer
            output_size: Number of output neurons
            learning_rate: Learning rate for gradient descent
        """
        # Initialize weights using Xavier initialization (scaled random)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(output_size)
        self.learning_rate = learning_rate
        self.losses = []
    
    def sigmoid(self, z):
        """Sigmoid activation: σ(z) = 1 / (1 + e^(-z))"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip to prevent overflow
    
    def sigmoid_derivative(self, z):
        """Derivative of sigmoid: σ'(z) = σ(z) * (1 - σ(z))"""
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def forward(self, X):
        """
        Forward propagation through the network
        
        Args:
            X: Input features, shape (batch_size, input_size)
        
        Returns:
            output: Network predictions
            cache: Intermediate values for backpropagation
        """
        # Hidden layer
        z1 = X @ self.W1 + self.b1  # Linear transformation
        a1 = self.sigmoid(z1)        # Non-linear activation
        
        # Output layer
        z2 = a1 @ self.W2 + self.b2  # Linear transformation
        a2 = self.sigmoid(z2)         # Non-linear activation
        
        # Cache values needed for backpropagation
        cache = {'X': X, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}
        return a2, cache
    
    def compute_loss(self, y_true, y_pred):
        """Binary cross-entropy loss"""
        m = y_true.shape[0]
        # Avoid log(0) by clipping predictions
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def backward(self, y_true, cache):
        """
        Backpropagation to compute gradients
        
        Args:
            y_true: True labels
            cache: Cached values from forward pass
        
        Returns:
            gradients: Dictionary of gradients for all parameters
        """
        m = y_true.shape[0]
        X, z1, a1, z2, a2 = cache['X'], cache['z1'], cache['a1'], cache['z2'], cache['a2']
        
        # Output layer gradients
        dz2 = a2 - y_true.reshape(-1, 1)  # Derivative of loss w.r.t. z2
        dW2 = (a1.T @ dz2) / m
        db2 = np.mean(dz2, axis=0)
        
        # Hidden layer gradients
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.sigmoid_derivative(z1)
        dW1 = (X.T @ dz1) / m
        db1 = np.mean(dz1, axis=0)
        
        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    
    def update_parameters(self, gradients):
        """Update weights using gradient descent"""
        self.W1 -= self.learning_rate * gradients['dW1']
        self.b1 -= self.learning_rate * gradients['db1']
        self.W2 -= self.learning_rate * gradients['dW2']
        self.b2 -= self.learning_rate * gradients['db2']
    
    def fit(self, X, y, epochs=5000, verbose=True):
        """
        Train the MLP
        
        Args:
            X: Training features
            y: Training labels
            epochs: Number of training epochs
            verbose: Whether to print progress
        """
        for epoch in range(epochs):
            # Forward pass
            predictions, cache = self.forward(X)
            
            # Compute loss
            loss = self.compute_loss(y, predictions)
            self.losses.append(loss)
            
            # Backward pass
            gradients = self.backward(y, cache)
            
            # Update parameters
            self.update_parameters(gradients)
            
            if verbose and (epoch % 1000 == 0 or epoch == epochs - 1):
                accuracy = np.mean((predictions > 0.5).astype(int).flatten() == y)
                print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy * 100:.1f}%")
    
    def predict(self, X):
        """Make predictions (0 or 1)"""
        predictions, _ = self.forward(X)
        return (predictions > 0.5).astype(int).flatten()


# Solve XOR problem with MLP
print("XOR Problem - MLP Solution:")
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

# Create and train MLP
np.random.seed(42)
mlp = MLP(input_size=2, hidden_size=4, output_size=1, learning_rate=1.0)
mlp.fit(X_xor, y_xor, epochs=5000, verbose=True)

# Test predictions
predictions = mlp.predict(X_xor)
print("\\nFinal Results:")
print("Input | Target | Predicted | Confidence")
print("-" * 45)
for x, target, pred in zip(X_xor, y_xor, predictions):
    prob, _ = mlp.forward(x.reshape(1, -1))
    print(f"{x}  |   {target}    |     {pred}     |   {prob[0, 0]:.4f}")

print(f"\\nFinal Accuracy: {np.mean(predictions == y_xor) * 100:.1f}%")

# Visualize learning
plt.figure(figsize=(10, 4))
plt.plot(mlp.losses, linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('MLP Learning on XOR Problem', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
\`\`\`

**Output:**
\`\`\`
Epoch 0: Loss = 0.7071, Accuracy = 25.0%
Epoch 1000: Loss = 0.1428, Accuracy = 100.0%
Epoch 2000: Loss = 0.0698, Accuracy = 100.0%
Epoch 3000: Loss = 0.0453, Accuracy = 100.0%
Epoch 4000: Loss = 0.0330, Accuracy = 100.0%
Epoch 4999: Loss = 0.0258, Accuracy = 100.0%

Final Results:
Input | Target | Predicted | Confidence
---------------------------------------------
[0 0]  |   0    |     0     |   0.0289
[0 1]  |   1    |     1     |   0.9716
[1 0]  |   1    |     1     |   0.9716
[1 1]  |   0    |     0     |   0.0304

Final Accuracy: 100.0%
\`\`\`

**Success!** The MLP with a hidden layer can solve XOR, which the perceptron cannot.

## Universal Approximation Theorem

### The Power of Neural Networks

The **Universal Approximation Theorem** (Cybenko, 1989; Hornik, 1991) states:

> *A feedforward neural network with a single hidden layer containing a finite number of neurons can approximate any continuous function on compact subsets of ℝⁿ, given appropriate activation functions.*

**What This Means:**
- Neural networks are **universal function approximators**
- Given enough neurons, they can learn almost any function
- This is true even with just ONE hidden layer
- Applies to any continuous activation function (sigmoid, tanh, ReLU)

### Practical Implications

\`\`\`python
# Demonstration: Approximating a complex function
def true_function(x):
    """Complex function to approximate"""
    return np.sin(2 * np.pi * x) + 0.5 * np.sin(4 * np.pi * x) + 0.1 * x

# Generate training data
np.random.seed(42)
X_train = np.linspace(0, 1, 100).reshape(-1, 1)
y_train = true_function(X_train).flatten()

# Add noise
y_train += np.random.normal(0, 0.1, y_train.shape)

# Build MLP for regression (using same class with modifications)
class MLPRegression(MLP):
    """MLP for regression (no sigmoid on output)"""
    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self.sigmoid(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = z2  # No activation on output for regression
        cache = {'X': X, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}
        return a2, cache
    
    def compute_loss(self, y_true, y_pred):
        """Mean squared error for regression"""
        return np.mean((y_true.reshape(-1, 1) - y_pred) ** 2)
    
    def backward(self, y_true, cache):
        m = y_true.shape[0]
        X, z1, a1, z2, a2 = cache['X'], cache['z1'], cache['a1'], cache['z2'], cache['a2']
        
        dz2 = (a2 - y_true.reshape(-1, 1)) / m
        dW2 = a1.T @ dz2
        db2 = np.mean(dz2, axis=0)
        
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.sigmoid_derivative(z1)
        dW1 = X.T @ dz1
        db1 = np.mean(dz1, axis=0)
        
        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

# Train with different hidden layer sizes
hidden_sizes = [5, 20, 50]
plt.figure(figsize=(15, 4))

for i, hidden_size in enumerate(hidden_sizes):
    mlp_reg = MLPRegression(input_size=1, hidden_size=hidden_size, output_size=1, learning_rate=0.1)
    mlp_reg.fit(X_train, y_train, epochs=5000, verbose=False)
    
    # Make predictions
    X_test = np.linspace(0, 1, 200).reshape(-1, 1)
    y_pred, _ = mlp_reg.forward(X_test)
    y_true_test = true_function(X_test)
    
    # Plot
    plt.subplot(1, 3, i + 1)
    plt.scatter(X_train, y_train, alpha=0.5, label='Training Data', s=10)
    plt.plot(X_test, y_true_test, 'g--', linewidth=2, label='True Function')
    plt.plot(X_test, y_pred, 'r-', linewidth=2, label='MLP Prediction')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title(f'Hidden Layer Size: {hidden_size}', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
\`\`\`

**Key Observations:**
- More neurons → better approximation
- Networks can learn complex, non-linear patterns
- Trade-off between capacity and overfitting

## Why Deep Learning Works

### The Depth Advantage

While the Universal Approximation Theorem says we only need one hidden layer, **deep networks (many layers) are more efficient**:

**Advantages of Depth:**
1. **Hierarchical Feature Learning**: Each layer learns increasingly abstract features
   - Layer 1: Edges and textures
   - Layer 2: Shapes and patterns
   - Layer 3: Object parts
   - Layer 4: Complete objects

2. **Exponential Efficiency**: Deep networks need exponentially fewer neurons than shallow networks for the same function
   - Shallow: O(2^n) neurons needed
   - Deep: O(n^2) neurons needed

3. **Better Generalization**: Deep architectures encode useful inductive biases

\`\`\`python
# Illustration: Shallow vs Deep networks for same task
print("Network Comparison:")
print("-" * 50)
print("Task: Learn complex function with 1000 input features")
print()
print("Shallow Network (1 hidden layer):")
print("  Architecture: 1000 → 10000 → 1")
print("  Parameters: 1000 * 10000 + 10000 + 10000 * 1 + 1 = ~10 million")
print()
print("Deep Network (4 hidden layers):")
print("  Architecture: 1000 → 512 → 256 → 128 → 64 → 1")
print("  Parameters: (1000*512) + (512*256) + (256*128) + (128*64) + (64*1) + biases")
print("              = 512000 + 131072 + 32768 + 8192 + 64 + ~1000 ≈ 685,000")
print()
print("Deep network uses ~15x fewer parameters!")
\`\`\`

### Why Neural Networks Beat Traditional Methods

**Automatic Feature Engineering:**
- Traditional ML: Manually engineer features (e.g., technical indicators in trading)
- Deep Learning: Learn features automatically from raw data

**End-to-End Learning:**
- Can learn complex input-output mappings without intermediate steps
- Backpropagation allows optimization of all layers jointly

**Scalability:**
- Performance improves with more data (unlike traditional methods that plateau)
- Can leverage GPUs for massive parallelization

**Representation Learning:**
- Learn useful internal representations that transfer to other tasks
- Pre-trained models (transfer learning) accelerate development

## Connection to Trading and Finance

### Applications in Quantitative Finance

1. **Price Prediction**:
   - Input: Historical prices, volume, indicators
   - Output: Next day's return or direction
   - Architecture: MLP or LSTM

2. **Portfolio Optimization**:
   - Input: Asset features, market conditions
   - Output: Optimal portfolio weights
   - Can incorporate complex constraints

3. **Risk Management**:
   - Input: Market data, positions
   - Output: Risk metrics (VaR, drawdown probability)
   - Learns non-linear risk relationships

4. **Alpha Generation**:
   - Input: Multi-modal data (prices, news, sentiment)
   - Output: Trading signals
   - Combines multiple information sources

### Simple Trading Example

\`\`\`python
# Simplified example: Predict stock direction using technical indicators
import pandas as pd

def create_trading_features(prices):
    """Create simple technical indicators as features"""
    df = pd.DataFrame({'price': prices})
    
    # Features
    df['return'] = df['price'].pct_change()
    df['sma_5'] = df['price'].rolling(5).mean()
    df['sma_20'] = df['price'].rolling(20).mean()
    df['rsi'] = compute_rsi(df['price'], periods=14)
    
    # Target: 1 if price goes up tomorrow, 0 otherwise
    df['target'] = (df['price'].shift(-1) > df['price']).astype(int)
    
    return df.dropna()

def compute_rsi(prices, periods=14):
    """Compute Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Generate synthetic price data
np.random.seed(42)
n_days = 1000
trend = np.linspace(100, 150, n_days)
noise = np.random.randn(n_days) * 5
seasonality = 10 * np.sin(np.linspace(0, 8 * np.pi, n_days))
prices = trend + noise + seasonality

# Create features
df_trading = create_trading_features(prices)

# Prepare data
X = df_trading[['return', 'sma_5', 'sma_20', 'rsi']].values
y = df_trading['target'].values

# Normalize features
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Train/test split (80/20)
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Train MLP
trading_mlp = MLP(input_size=4, hidden_size=8, output_size=1, learning_rate=0.1)
trading_mlp.fit(X_train, y_train, epochs=3000, verbose=False)

# Evaluate
train_pred = trading_mlp.predict(X_train)
test_pred = trading_mlp.predict(X_test)

train_acc = np.mean(train_pred == y_train)
test_acc = np.mean(test_pred == y_test)

print("\\nTrading Direction Prediction:")
print(f"Training Accuracy: {train_acc * 100:.1f}%")
print(f"Test Accuracy: {test_acc * 100:.1f}%")
print(f"\\nBaseline (always predict 'up'): {np.mean(y_test) * 100:.1f}%")
\`\`\`

**Important Notes:**
- This is a simplified example for educational purposes
- Real trading requires much more sophisticated features and risk management
- Neural networks can overfit to noise in financial data
- Transaction costs and slippage significantly impact profitability
- Always validate with out-of-sample data and walk-forward analysis

## Key Takeaways

1. **Biological Inspiration**: Neural networks abstract brain principles into mathematical operations
2. **Perceptron**: Simplest neuron but limited to linear problems
3. **MLPs**: Multiple layers with non-linear activations enable learning complex patterns
4. **Universal Approximation**: Neural networks can approximate any continuous function
5. **Deep > Shallow**: Depth provides efficiency and better feature learning
6. **Automatic Features**: Neural networks learn representations automatically
7. **Trading Applications**: From price prediction to portfolio optimization
8. **Foundation**: Understanding these basics is crucial for modern deep learning

## What's Next

Now that you understand the basics of neural networks, we'll dive deeper into:
- **Activation Functions**: Different non-linearities and their properties
- **Forward Propagation**: Detailed mechanics of data flow
- **Loss Functions**: How to measure and optimize performance
- **Backpropagation**: The algorithm that makes learning possible
- **Optimization**: Advanced techniques for training neural networks

These fundamentals will prepare you for modern architectures like CNNs, RNNs, and Transformers used in production trading systems.
`,
};
