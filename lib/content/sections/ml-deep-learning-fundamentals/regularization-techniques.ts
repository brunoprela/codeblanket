/**
 * Section: Regularization Techniques
 * Module: Deep Learning Fundamentals
 *
 * Covers L1/L2 regularization, Dropout, Batch Normalization, Layer Normalization,
 * Early Stopping, and modern best practices for preventing overfitting
 */

export const regularizationTechniquesSection = {
  id: 'regularization-techniques',
  title: 'Regularization Techniques',
  content: `
# Regularization Techniques

## Introduction

**Overfitting** is the enemy of generalization. Neural networks with millions of parameters can easily memorize training data. **Regularization** prevents this by constraining the model's capacity.

**What You'll Learn:**
- L1 and L2 weight regularization
- Dropout: randomly dropping neurons
- Batch Normalization: normalizing activations
- Layer Normalization: alternative to Batch Norm
- Early Stopping: when to stop training
- Combining multiple regularization techniques

## L1 and L2 Regularization

### L2 Regularization (Weight Decay)

**Idea**: Penalize large weights by adding squared weight magnitude to loss.

**Modified Loss**:
\\\`\\\`\\\`
L_total = L_data + λ/2 * Σ(w²)
\\\`\\\`\\\`

\`\`\`python
def l2_loss(y_true, y_pred, weights, lambda_reg=0.01):
    """Loss with L2 regularization"""
    data_loss = np.mean((y_true - y_pred)**2)
    l2_penalty = lambda_reg / 2 * sum(np.sum(w**2) for w in weights)
    return data_loss + l2_penalty

# Gradient includes regularization term
def l2_gradient(w, grad_data, lambda_reg=0.01):
    """Gradient with L2 regularization"""
    return grad_data + lambda_reg * w
\`\`\`

**Effect**: Drives weights toward zero, preferring small weights. Reduces model sensitivity to individual features.

### L1 Regularization (Lasso)

**Modified Loss**:
\\\`\\\`\\\`
L_total = L_data + λ * Σ|w|
\\\`\\\`\\\`

\`\`\`python
def l1_loss(y_true, y_pred, weights, lambda_reg=0.01):
    """Loss with L1 regularization"""
    data_loss = np.mean((y_true - y_pred)**2)
    l1_penalty = lambda_reg * sum(np.sum(np.abs(w)) for w in weights)
    return data_loss + l1_penalty

# Gradient includes sign of weights
def l1_gradient(w, grad_data, lambda_reg=0.01):
    """Gradient with L1 regularization"""
    return grad_data + lambda_reg * np.sign(w)
\`\`\`

**Effect**: Drives weights exactly to zero, producing sparse models. Good for feature selection.

**Comparison**:
- L2: smooth, all weights shrunk proportionally
- L1: sparse, drives unimportant weights to exactly zero

## Dropout

### Random Neuron Deactivation

**Idea**: During training, randomly set neuron outputs to zero with probability \\(p\\) (typically 0.5).

**Benefits**:
- Prevents co-adaptation: neurons can't rely on specific other neurons
- Ensemble effect: training multiple "sub-networks"
- Strong regularizer without explicit constraint

\`\`\`python
def dropout_forward(x, p=0.5, training=True):
    """
    Apply dropout
    
    x: input activations
    p: dropout probability
    training: if False (inference), dropout is disabled
    """
    if not training:
        return x
    
    # Create dropout mask
    mask = (np.random.rand(*x.shape) > p) / (1 - p)  # Scale to maintain expectation
    return x * mask

# Example usage
X = np.random.randn(32, 128)  # batch_size=32, features=128

# Training
X_train = dropout_forward(X, p=0.5, training=True)
print(f"Training: {(X_train == 0).mean():.2f} zeros")  # ~50%

# Inference
X_test = dropout_forward(X, p=0.5, training=False)
print(f"Inference: {(X_test == 0).mean():.2f} zeros")  # 0%
\`\`\`

**Important**: 
- Only apply during training
- Scale remaining activations by \\(1/(1-p)\\) to maintain expected values
- Typical \\(p\\): 0.2 for input layer, 0.5 for hidden layers

### Inverted Dropout (Modern Approach)

\`\`\`python
class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.mask = None
    
    def forward(self, x, training=True):
        if not training:
            return x
        
        # Inverted dropout: scale during training
        self.mask = (np.random.rand(*x.shape) > self.p)
        return x * self.mask / (1 - self.p)
    
    def backward(self, grad_output):
        # Only backprop through active neurons
        return grad_output * self.mask / (1 - self.p)

# Usage
dropout = Dropout(p=0.5)
output = dropout.forward(X, training=True)
grad = dropout.backward(grad_output)
\`\`\`

## Batch Normalization

### Normalizing Layer Activations

**Problem**: Internal covariate shift - distribution of layer inputs changes during training.

**Solution**: Normalize activations to zero mean and unit variance within each mini-batch.

**Algorithm**:
\\\`\\\`\\\`
μ_batch = mean(x, axis=0)           # per-feature mean
σ²_batch = var(x, axis=0)           # per-feature variance
x_norm = (x - μ_batch) / √(σ²_batch + ε)
y = γ * x_norm + β                  # learnable scale and shift
\\\`\\\`\\\`

\`\`\`python
class BatchNorm:
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        
        # Running statistics for inference
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
    
    def forward(self, x, training=True):
        if training:
            # Compute batch statistics
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            
            # Normalize
            x_norm = (x - batch_mean) / np.sqrt(batch_var + self.eps)
            
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
        else:
            # Use running statistics
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
        
        # Scale and shift
        return self.gamma * x_norm + self.beta

# Example
bn = BatchNorm(num_features=128)
X = np.random.randn(32, 128)  # batch_size=32

X_bn = bn.forward(X, training=True)
print(f"Before BN: mean={X.mean():.4f}, std={X.std():.4f}")
print(f"After BN: mean={X_bn.mean():.4f}, std={X_bn.std():.4f}")
\`\`\`

**Benefits**:
- Stabilizes training (can use higher learning rates)
- Reduces sensitivity to initialization
- Acts as regularizer (adds noise via batch statistics)
- Enables deeper networks

**Drawbacks**:
- Couples training examples within batch (can be issue for very small batches)
- Different behavior in training vs inference
- Not ideal for RNNs (sequence lengths vary)

## Layer Normalization

### Alternative to Batch Norm

**Key difference**: Normalize across features (not batch).

\`\`\`python
def layer_norm(x, gamma, beta, eps=1e-5):
    """
    Layer normalization
    
    x: (batch_size, features)
    gamma, beta: learnable parameters of shape (features,)
    """
    mean = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True)
    
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta

# Example
X = np.random.randn(32, 128)
gamma = np.ones(128)
beta = np.zeros(128)

X_ln = layer_norm(X, gamma, beta)
print(f"Layer Norm: mean per sample = {np.mean(X_ln, axis=1)[:5]}")  # ≈ 0 for each sample
\`\`\`

**When to use**:
- RNNs, Transformers (Batch Norm problematic)
- Small batch sizes
- Online learning
- When samples should be independent

## Early Stopping

### Stop Training Before Overfitting

**Idea**: Monitor validation loss, stop when it stops improving.

\`\`\`python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
    
    def step(self, val_loss, model_weights):
        """
        Returns True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            # Improvement
            self.best_loss = val_loss
            self.best_weights = model_weights.copy()
            self.counter = 0
            return False
        else:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Stop training
            return False

# Usage
early_stop = EarlyStopping(patience=10)

for epoch in range(1000):
    train_loss = train_one_epoch()
    val_loss = validate()
    
    if early_stop.step(val_loss, model.get_weights()):
        print(f"Early stopping at epoch {epoch}")
        model.set_weights(early_stop.best_weights)
        break
\`\`\`

## Combining Regularization Techniques

**Best Practice**: Use multiple regularization techniques together.

\`\`\`python
class RegularizedModel:
    def __init__(self, input_dim, hidden_dims, output_dim, 
                 dropout_p=0.5, l2_lambda=0.01):
        self.layers = []
        self.dropout_layers = []
        self.bn_layers = []
        
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            # Linear layer with L2 regularization
            self.layers.append(Linear(dims[i], dims[i+1], l2_lambda=l2_lambda))
            
            if i < len(dims) - 2:  # Not output layer
                # Batch normalization
                self.bn_layers.append(BatchNorm(dims[i+1]))
                
                # Dropout
                self.dropout_layers.append(Dropout(dropout_p))
    
    def forward(self, x, training=True):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.bn_layers[i].forward(x, training)
            x = relu(x)
            x = self.dropout_layers[i].forward(x, training)
        
        # Output layer (no BN/dropout)
        x = self.layers[-1](x)
        return x

# Create model with multiple regularization techniques
model = RegularizedModel(
    input_dim=784,
    hidden_dims=[512, 256, 128],
    output_dim=10,
    dropout_p=0.5,
    l2_lambda=0.01
)
\`\`\`

## Regularization Guidelines

**General Rules**:

1. **Always use**:
   - Batch/Layer Normalization (makes training much easier)
   - Early Stopping (free performance gain)

2. **Start with**:
   - L2 regularization (λ ≈ 0.001 to 0.01)
   - Dropout 0.5 in hidden layers

3. **Adjust based on overfitting**:
   - Overfitting → increase dropout, increase L2
   - Underfitting → decrease regularization

4. **Architecture matters**:
   - CNNs: Batch Norm + Dropout (lighter)
   - RNNs: Layer Norm + Dropout (heavier)
   - Transformers: Layer Norm + Dropout + Warmup

5. **Modern trend**: 
   - Batch Norm often sufficient regularization
   - Heavy data augmentation > aggressive regularization
   - Larger models + more regularization > smaller models

## Key Takeaways

1. **Regularization prevents overfitting** - essential for deep learning
2. **L2 (weight decay)** - penalizes large weights, most common
3. **Dropout** - randomly deactivates neurons, strong regularizer
4. **Batch Norm** - normalizes activations, stabilizes training
5. **Layer Norm** - alternative to Batch Norm for RNNs/Transformers
6. **Early Stopping** - simple and effective
7. **Combine techniques** - multiple regularizers work synergistically

## What's Next

Regularization keeps training stable. Next: **Training Neural Networks** - putting it all together with batch sizes, learning rate scheduling, and monitoring.
`,
};
