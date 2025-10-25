/**
 * Section: Deep Learning Best Practices
 * Module: Deep Learning Fundamentals
 *
 * Covers data preprocessing, learning rate finder, gradient checking, debugging,
 * hyperparameter tuning, and production-ready model development
 */

export const deepLearningBestPracticesSection = {
  id: 'deep-learning-best-practices',
  title: 'Deep Learning Best Practices',
  content: `
# Deep Learning Best Practices

## Introduction

Building effective deep learning models requires more than understanding algorithms. This section covers practical techniques for debugging, tuning, and deploying production-ready models.

**What You'll Learn:**
- Data preprocessing and normalization
- Learning rate finder
- Gradient checking for correctness
- Debugging training issues
- Hyperparameter tuning strategies
- Model interpretability
- Production best practices

## Data Preprocessing

### Normalization and Standardization

\`\`\`python
import numpy as np

# Standardization (zero mean, unit variance)
def standardize(X_train, X_val, X_test):
    """Standardize features to mean=0, std=1"""
    mean = X_train.mean (axis=0)
    std = X_train.std (axis=0) + 1e-8  # Avoid division by zero
    
    X_train_std = (X_train - mean) / std
    X_val_std = (X_val - mean) / std
    X_test_std = (X_test - mean) / std
    
    return X_train_std, X_val_std, X_test_std

# Min-Max Normalization (scale to [0, 1])
def normalize(X_train, X_val, X_test):
    """Normalize features to [0, 1]"""
    min_val = X_train.min (axis=0)
    max_val = X_train.max (axis=0)
    range_val = max_val - min_val + 1e-8
    
    X_train_norm = (X_train - min_val) / range_val
    X_val_norm = (X_val - min_val) / range_val
    X_test_norm = (X_test - min_val) / range_val
    
    return X_train_norm, X_val_norm, X_test_norm

# Example
X_train = np.random.randn(1000, 10) * 10 + 50  # mean ≈ 50, std ≈ 10
X_val = np.random.randn(200, 10) * 10 + 50
X_test = np.random.randn(200, 10) * 10 + 50

print(f"Before: mean={X_train.mean():.2f}, std={X_train.std():.2f}")

X_train_std, X_val_std, X_test_std = standardize(X_train, X_val, X_test)
print(f"After: mean={X_train_std.mean():.2f}, std={X_train_std.std():.2f}")
\`\`\`

**Key Rule**: Compute statistics (mean, std, min, max) on training set only, then apply to val/test. Never let test data leak into preprocessing!

### Input Validation

\`\`\`python
def validate_input(X, y, num_classes):
    """Validate input data for common issues"""
    issues = []
    
    # Check shapes
    if X.shape[0] != y.shape[0]:
        issues.append (f"Shape mismatch: X has {X.shape[0]} samples, y has {y.shape[0]}")
    
    # Check for NaN/Inf
    if np.isnan(X).any():
        issues.append (f"X contains {np.isnan(X).sum()} NaN values")
    if np.isinf(X).any():
        issues.append (f"X contains {np.isinf(X).sum()} Inf values")
    
    # Check label range
    if y.min() < 0 or y.max() >= num_classes:
        issues.append (f"Labels out of range: [{y.min()}, {y.max()}], expected [0, {num_classes-1}]")
    
    # Check class balance
    class_counts = np.bincount (y)
    imbalance_ratio = class_counts.max() / (class_counts.min() + 1e-8)
    if imbalance_ratio > 10:
        issues.append (f"Severe class imbalance: ratio = {imbalance_ratio:.1f}:1")
    
    if issues:
        print("Data issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Data validation passed!")
    
    return len (issues) == 0

# Example
X = np.random.randn(1000, 784)
y = np.random.randint(0, 10, 1000)
validate_input(X, y, num_classes=10)
\`\`\`

## Learning Rate Finder

**Technique**: Train with exponentially increasing LR, plot loss vs LR, pick LR where loss decreases fastest.

\`\`\`python
import matplotlib.pyplot as plt

def find_lr (model, train_loader, optimizer, init_lr=1e-7, final_lr=10, num_iter=100):
    """
    Find optimal learning rate using LR range test
    
    Returns: lrs (list), losses (list)
    """
    model.train()
    lrs = []
    losses = []
    
    # Exponential growth factor
    lr_mult = (final_lr / init_lr) ** (1 / num_iter)
    lr = init_lr
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    avg_loss = 0.0
    best_loss = float('inf')
    batch_num = 0
    
    for data, target in train_loader:
        batch_num += 1
        
        # Forward pass
        optimizer.zero_grad()
        output = model (data)
        loss = criterion (output, target)
        
        # Track smoothed loss
        avg_loss = 0.9 * avg_loss + 0.1 * loss.item() if avg_loss > 0 else loss.item()
        
        # Stop if loss explodes
        if avg_loss > 4 * best_loss or batch_num >= num_iter:
            break
        
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        # Record
        lrs.append (lr)
        losses.append (avg_loss)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Increase learning rate
        lr *= lr_mult
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    return lrs, losses

# Usage
lrs, losses = find_lr (model, train_loader, optimizer)

# Plot
plt.figure (figsize=(10, 6))
plt.plot (lrs, losses)
plt.xscale('log')
plt.xlabel('Learning Rate (log scale)')
plt.ylabel('Loss')
plt.title('Learning Rate Finder')
plt.grid(True, alpha=0.3)
plt.show()

# Find steepest point
gradients = np.gradient (losses)
best_idx = np.argmin (gradients)
best_lr = lrs[best_idx]
print(f"Suggested learning rate: {best_lr:.2e}")
\`\`\`

**Interpretation**:
- Loss decreases → good LR range
- Loss flat → LR too low
- Loss increases → LR too high
- Choose LR at steepest descent point (or slightly lower)

## Gradient Checking

**Verify backpropagation implementation** by comparing analytical gradients with numerical gradients.

\`\`\`python
def numerical_gradient (f, x, eps=1e-5):
    """
    Compute numerical gradient using finite differences
    
    f: function that takes x and returns scalar loss
    x: parameter (numpy array)
    """
    grad = np.zeros_like (x)
    it = np.nditer (x, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        old_value = x[idx]
        
        # f (x + eps)
        x[idx] = old_value + eps
        f_plus = f (x)
        
        # f (x - eps)
        x[idx] = old_value - eps
        f_minus = f (x)
        
        # Gradient: (f (x+eps) - f (x-eps)) / (2*eps)
        grad[idx] = (f_plus - f_minus) / (2 * eps)
        
        x[idx] = old_value  # Restore
        it.iternext()
    
    return grad

def gradient_check (model, x, y, eps=1e-5, threshold=1e-4):
    """
    Check if analytical gradients match numerical gradients
    """
    # Compute analytical gradient
    model.zero_grad()
    output = model (x)
    loss = criterion (output, y)
    loss.backward()
    
    params = [p for p in model.parameters() if p.requires_grad]
    analytical_grads = [p.grad.clone() for p in params]
    
    # Compute numerical gradient for each parameter
    print("Gradient checking...")
    for i, param in enumerate (params):
        def f (p):
            """Loss as function of single parameter"""
            param.data.copy_(torch.from_numpy (p))
            output = model (x)
            return criterion (output, y).item()
        
        numerical_grad = numerical_gradient (f, param.data.cpu().numpy(), eps)
        analytical_grad = analytical_grads[i].cpu().numpy()
        
        # Compute relative difference
        diff = np.linalg.norm (numerical_grad - analytical_grad) / \\
               (np.linalg.norm (numerical_grad) + np.linalg.norm (analytical_grad) + 1e-8)
        
        status = "✓ PASS" if diff < threshold else "✗ FAIL"
        print(f"Parameter {i}: diff = {diff:.2e} {status}")
    
    print("Gradient check complete!")

# Example usage
x_sample = torch.randn(1, 784)
y_sample = torch.randint(0, 10, (1,))
gradient_check (model, x_sample, y_sample)
\`\`\`

## Debugging Training Issues

### Diagnostic Checklist

\`\`\`python
def diagnose_training (model, train_loader, val_loader):
    """Diagnose common training issues"""
    print("Running diagnostics...\\n")
    
    # 1. Overfit single batch
    print("1. Testing model capacity (overfitting single batch)")
    single_batch = next (iter (train_loader))
    x, y = single_batch
    
    for epoch in range(100):
        optimizer.zero_grad()
        output = model (x)
        loss = criterion (output, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"   Epoch {epoch+1}: loss = {loss.item():.4f}")
    
    if loss.item() < 0.01:
        print("   ✓ Model can overfit single batch (good capacity)")
    else:
        print("   ✗ Model cannot overfit single batch (insufficient capacity or bug)")
    
    # 2. Check gradient flow
    print("\\n2. Checking gradient flow")
    model.zero_grad()
    output = model (x)
    loss = criterion (output, y)
    loss.backward()
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            grad_norm = param.grad.norm().item()
            if grad_norm < 1e-7:
                print(f"   ⚠ {name}: very small gradient ({grad_norm:.2e})")
            elif grad_norm > 100:
                print(f"   ⚠ {name}: very large gradient ({grad_norm:.2e})")
    
    # 3. Check weight statistics
    print("\\n3. Checking weight statistics")
    for name, param in model.named_parameters():
        mean = param.data.mean().item()
        std = param.data.std().item()
        if abs (mean) > 1 or std > 10:
            print(f"   ⚠ {name}: unusual distribution (mean={mean:.2f}, std={std:.2f})")
    
    print("\\nDiagnostics complete!")
\`\`\`

### Common Issues and Solutions

| Issue | Symptoms | Solutions |
|-------|----------|-----------|
| **Vanishing Gradients** | Training stalls, deep layers don't learn | Use ReLU, He init, residual connections, gradient clipping |
| **Exploding Gradients** | Loss becomes NaN, weights explode | Reduce LR, gradient clipping, check normalization |
| **Overfitting** | Train acc high, val acc low | More data, regularization, smaller model |
| **Underfitting** | Both train and val acc low | Larger model, train longer, better features |
| **Slow Convergence** | Loss decreases very slowly | Higher LR, better optimizer (Adam), better initialization |
| **Class Imbalance** | High acc but poor minority class performance | Weighted loss, oversampling, focal loss |

## Hyperparameter Tuning

### Manual Search (Systematic)

\`\`\`python
# Grid search (exhaustive but expensive)
learning_rates = [1e-4, 1e-3, 1e-2]
batch_sizes = [32, 64, 128]
dropout_rates = [0.3, 0.5, 0.7]

best_val_acc = 0
best_params = None

for lr in learning_rates:
    for batch_size in batch_sizes:
        for dropout in dropout_rates:
            print(f"\\nTrying lr={lr}, batch={batch_size}, dropout={dropout}")
            
            model = create_model (dropout=dropout)
            optimizer = Adam (model.parameters(), lr=lr)
            
            val_acc = train_and_evaluate (model, optimizer, batch_size)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = {'lr': lr, 'batch_size': batch_size, 'dropout': dropout}

print(f"\\nBest params: {best_params}, Val Acc: {best_val_acc:.4f}")
\`\`\`

### Random Search (More Efficient)

\`\`\`python
import random

def random_search (num_trials=20):
    """Random hyperparameter search"""
    best_val_acc = 0
    best_params = None
    
    for trial in range (num_trials):
        # Sample hyperparameters
        lr = 10 ** random.uniform(-5, -2)  # Log-uniform
        batch_size = random.choice([32, 64, 128, 256])
        dropout = random.uniform(0.2, 0.7)
        hidden_size = random.choice([128, 256, 512, 1024])
        
        params = {
            'lr': lr,
            'batch_size': batch_size,
            'dropout': dropout,
            'hidden_size': hidden_size
        }
        
        print(f"\\nTrial {trial+1}/{num_trials}: {params}")
        
        model = create_model (hidden_size=hidden_size, dropout=dropout)
        optimizer = Adam (model.parameters(), lr=lr)
        val_acc = train_and_evaluate (model, optimizer, batch_size, epochs=10)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = params
    
    return best_params, best_val_acc

best_params, best_acc = random_search (num_trials=20)
print(f"\\nBest: {best_params}, Acc: {best_acc:.4f}")
\`\`\`

**Random search often better than grid search** because it explores more values for important hyperparameters.

### Bayesian Optimization (Advanced)

\`\`\`python
# Using Optuna library
import optuna

def objective (trial):
    """Objective function for Optuna"""
    # Suggest hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    dropout = trial.suggest_uniform('dropout', 0.2, 0.7)
    hidden_size = trial.suggest_categorical('hidden_size', [128, 256, 512])
    
    # Train model
    model = create_model (hidden_size=hidden_size, dropout=dropout)
    optimizer = Adam (model.parameters(), lr=lr)
    val_acc = train_and_evaluate (model, optimizer, batch_size, epochs=10)
    
    return val_acc

# Run optimization
study = optuna.create_study (direction='maximize')
study.optimize (objective, n_trials=50)

print(f"Best params: {study.best_params}")
print(f"Best value: {study.best_value:.4f}")
\`\`\`

## Production Best Practices

### Model Versioning

\`\`\`python
import json
from datetime import datetime

def save_model_with_metadata (model, metadata, path):
    """Save model with version metadata"""
    # Save model
    torch.save (model.state_dict(), f"{path}/model.pth")
    
    # Add timestamp and version
    metadata['saved_at'] = datetime.now().isoformat()
    metadata['model_architecture'] = str (model)
    
    # Save metadata
    with open (f"{path}/metadata.json", 'w') as f:
        json.dump (metadata, f, indent=2)

# Usage
metadata = {
    'version': '1.0.0',
    'training_data': 'dataset_v2',
    'hyperparameters': {
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs': 50
    },
    'metrics': {
        'val_accuracy': 0.95,
        'val_loss': 0.15
    }
}

save_model_with_metadata (model, metadata, 'models/v1.0.0')
\`\`\`

### Input Validation (Production)

\`\`\`python
def validate_and_preprocess (x, scaler, expected_shape):
    """Validate and preprocess input for production inference"""
    # Check shape
    if x.shape[-1] != expected_shape:
        raise ValueError (f"Expected {expected_shape} features, got {x.shape[-1]}")
    
    # Check for invalid values
    if np.isnan (x).any() or np.isinf (x).any():
        raise ValueError("Input contains NaN or Inf values")
    
    # Apply same preprocessing as training
    x = scaler.transform (x)
    
    return x
\`\`\`

## Key Takeaways

1. **Always normalize/standardize** - compute stats on train only
2. **Use LR finder** - find optimal learning rate systematically
3. **Gradient checking** - verify backprop implementation
4. **Debug systematically** - overfit single batch first
5. **Random search > grid search** - more efficient exploration
6. **Monitor everything** - logs, metrics, visualizations
7. **Version models** - track hyperparameters, data, metrics

## What's Next

Final section: **Efficient Training Techniques** - how to train bigger models faster with mixed precision, gradient accumulation, and parallelism!
`,
};
