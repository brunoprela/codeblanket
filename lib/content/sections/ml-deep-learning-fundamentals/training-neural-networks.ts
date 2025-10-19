/**
 * Section: Training Neural Networks
 * Module: Deep Learning Fundamentals
 *
 * Covers practical aspects of training: batch size selection, learning rate scheduling,
 * gradient clipping, monitoring training, and validation strategies
 */

export const trainingNeuralNetworksSection = {
  id: 'training-neural-networks',
  title: 'Training Neural Networks',
  content: `
# Training Neural Networks

## Introduction

You've learned the components: forward prop, backprop, optimizers, regularization. Now let's put it all together and train effectively.

**What You'll Learn:**
- Batch size selection and trade-offs
- Learning rate scheduling strategies
- Gradient clipping for stability
- Monitoring and debugging training
- Validation strategies
- Complete training loop implementation

## Batch Size Selection

### What is Batch Size?

**Batch Size**: Number of training examples processed before updating weights.

**Three approaches**:
1. **Batch Gradient Descent**: Use all training data (batch_size = N)
2. **Stochastic Gradient Descent (SGD)**: One example at a time (batch_size = 1)
3. **Mini-batch Gradient Descent**: Between 1 and N (typical: 32-512)

### Trade-offs

\`\`\`python
# Small batch (e.g., 32)
# + More frequent updates → faster convergence (steps)
# + Better generalization (noise helps escape local minima)
# + Lower memory requirement
# - Slower per step (less GPU utilization)
# - Noisier gradients

# Large batch (e.g., 512)
# + Faster per step (better GPU utilization)
# + More stable gradients
# + Better for distributed training
# - Slower convergence (steps)
# - May generalize worse
# - Higher memory requirement
\`\`\`

**Modern Practice**:
- Start with 32 or 64
- Increase until GPU memory is full or convergence slows
- Use learning rate scaling: \\(lr_{new} = lr_{base} * (batch_{new} / batch_{base})\\)

## Learning Rate Scheduling

### Why Schedule Learning Rate?

**Problem**: Fixed LR either:
- Too high → never converges
- Too low → converges very slowly

**Solution**: Start high (fast initial progress), decay gradually (fine-tune).

### Common Schedules

#### 1. Step Decay

\`\`\`python
def step_decay_lr(epoch, initial_lr=0.1, drop_rate=0.5, epochs_drop=10):
    """Reduce LR by drop_rate every epochs_drop epochs"""
    lr = initial_lr * (drop_rate ** (epoch // epochs_drop))
    return lr

# Example: 0.1 → 0.05 → 0.025 → ...
lrs = [step_decay_lr(e) for e in range(30)]
plt.plot(lrs)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Step Decay')
plt.show()
\`\`\`

#### 2. Exponential Decay

\`\`\`python
def exp_decay_lr(epoch, initial_lr=0.1, decay_rate=0.95):
    """Smooth exponential decay"""
    lr = initial_lr * (decay_rate ** epoch)
    return lr
\`\`\`

#### 3. Cosine Annealing

\`\`\`python
import math

def cosine_annealing_lr(epoch, initial_lr=0.1, min_lr=1e-6, T_max=100):
    """Cosine decay from initial_lr to min_lr over T_max epochs"""
    lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * epoch / T_max))
    return lr

# Smooth decay following cosine curve
lrs = [cosine_annealing_lr(e, T_max=50) for e in range(50)]
plt.plot(lrs)
plt.title('Cosine Annealing')
plt.show()
\`\`\`

#### 4. Learning Rate Warmup

\`\`\`python
def warmup_cosine_lr(epoch, initial_lr=0.1, warmup_epochs=5, total_epochs=100):
    """Linear warmup followed by cosine annealing"""
    if epoch < warmup_epochs:
        # Linear warmup
        lr = initial_lr * (epoch + 1) / warmup_epochs
    else:
        # Cosine annealing
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        lr = initial_lr * 0.5 * (1 + math.cos(math.pi * progress))
    return lr

lrs = [warmup_cosine_lr(e) for e in range(100)]
plt.plot(lrs)
plt.title('Warmup + Cosine Annealing')
plt.show()
\`\`\`

**Why warmup?** Large batches + high initial LR can destabilize training. Warmup gradually increases LR to prevent this.

### Reduce on Plateau

\`\`\`python
class ReduceLROnPlateau:
    def __init__(self, initial_lr=0.1, factor=0.5, patience=10, min_lr=1e-6):
        self.lr = initial_lr
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best_loss = float('inf')
        self.counter = 0
    
    def step(self, val_loss):
        """Update LR based on validation loss"""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.lr = max(self.lr * self.factor, self.min_lr)
                self.counter = 0
                print(f"Reducing LR to {self.lr}")
        return self.lr

# Usage
scheduler = ReduceLROnPlateau(initial_lr=0.1, patience=10)
for epoch in range(100):
    val_loss = validate()
    new_lr = scheduler.step(val_loss)
    optimizer.lr = new_lr
\`\`\`

## Gradient Clipping

### Preventing Exploding Gradients

**Problem**: In deep/recurrent networks, gradients can explode (grow exponentially).

**Solution**: Clip gradient norm to maximum value.

\`\`\`python
def clip_gradients_norm(gradients, max_norm=5.0):
    """
    Clip gradients by global norm
    
    If ||grad|| > max_norm, scale down: grad = grad * max_norm / ||grad||
    """
    # Compute global norm
    total_norm = 0
    for grad in gradients:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)
    
    # Clip if necessary
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for grad in gradients:
            grad *= clip_coef
    
    return gradients

# Usage in training loop
gradients = compute_gradients(loss)
gradients = clip_gradients_norm(gradients, max_norm=5.0)
apply_gradients(gradients)
\`\`\`

**Alternatives**:
- **Clip by value**: \\(grad = clip(grad, -threshold, threshold)\\)
- **Clip by norm**: \\(grad = grad * min(1, threshold / ||grad||)\\) (preferred)

## Monitoring Training

### What to Track

\`\`\`python
import matplotlib.pyplot as plt

class TrainingMonitor:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.lrs = []
        self.grad_norms = []
    
    def log(self, epoch, train_loss, val_loss, train_acc, val_acc, lr, grad_norm):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        self.lrs.append(lr)
        self.grad_norms.append(grad_norm)
        
        print(f"Epoch {epoch}: "
              f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, "
              f"lr={lr:.6f}")
    
    def plot(self):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss
        axes[0, 0].plot(self.train_losses, label='Train')
        axes[0, 0].plot(self.val_losses, label='Val')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training vs Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(self.train_accs, label='Train')
        axes[0, 1].plot(self.val_accs, label='Val')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Training vs Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 0].plot(self.lrs)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Gradient Norm
        axes[1, 1].plot(self.grad_norms)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Gradient Norm')
        axes[1, 1].set_title('Gradient Norm (check for exploding/vanishing)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Usage
monitor = TrainingMonitor()
for epoch in range(100):
    train_loss, train_acc = train_one_epoch()
    val_loss, val_acc = validate()
    grad_norm = compute_grad_norm()
    monitor.log(epoch, train_loss, val_loss, train_acc, val_acc, lr, grad_norm)
\`\`\`

### Interpreting Training Curves

**Healthy Training**:
- Train and val loss both decreasing
- Gap between train/val reasonable (< 10%)
- Smooth curves (not too noisy)

**Overfitting**:
- Train loss decreasing, val loss increasing
- Large gap between train and val accuracy
- → Add regularization, get more data

**Underfitting**:
- Both train and val loss high
- Both still decreasing at end
- → Increase model capacity, train longer

**Unstable Training**:
- Loss spikes, NaN, or diverges
- → Reduce learning rate, add gradient clipping, check data

## Complete Training Loop

\`\`\`python
import numpy as np
from tqdm import tqdm

def train_model(model, train_loader, val_loader, epochs=100, 
                initial_lr=0.001, device='cpu'):
    """Complete training loop with all best practices"""
    
    # Setup
    optimizer = Adam(model.parameters(), lr=initial_lr)
    scheduler = ReduceLROnPlateau(initial_lr, patience=10)
    early_stop = EarlyStopping(patience=20)
    monitor = TrainingMonitor()
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}')):
            data, target = data.to(device), target.to(device)
            
            # Forward
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            grad_norm = clip_gradients_norm(model.parameters(), max_norm=5.0)
            
            # Update
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += (pred == target).sum().item()
            train_total += target.size(0)
        
        # Calculate epoch metrics
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                pred = output.argmax(dim=1)
                val_correct += (pred == target).sum().item()
                val_total += target.size(0)
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        # Learning rate scheduling
        current_lr = scheduler.step(val_loss)
        
        # Log metrics
        monitor.log(epoch, train_loss, val_loss, train_acc, val_acc, current_lr, grad_norm)
        
        # Early stopping
        if early_stop.step(val_loss, model.state_dict()):
            print(f"Early stopping at epoch {epoch+1}")
            model.load_state_dict(early_stop.best_weights)
            break
    
    # Final visualization
    monitor.plot()
    
    return model, monitor

# Usage
model, history = train_model(
    model=my_model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    initial_lr=0.001,
    device='cuda'
)
\`\`\`

## Validation Strategies

### Hold-out Validation

\`\`\`python
from sklearn.model_selection import train_test_split

# Split data: 60% train, 20% val, 20% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
\`\`\`

### K-Fold Cross-Validation

\`\`\`python
from sklearn.model_selection import KFold

def cross_validate_model(model_fn, X, y, k=5):
    """K-fold cross-validation"""
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"\\nFold {fold+1}/{k}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = model_fn()
        model.fit(X_train, y_train)
        
        val_score = model.evaluate(X_val, y_val)
        scores.append(val_score)
        print(f"Val accuracy: {val_score:.4f}")
    
    print(f"\\nAverage: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    return scores
\`\`\`

## Key Takeaways

1. **Batch size**: Start with 32-64, increase until GPU full
2. **Learning rate**: Use scheduling (warmup + cosine annealing)
3. **Gradient clipping**: Prevents exploding gradients (max_norm=5)
4. **Monitor everything**: Loss, accuracy, LR, gradient norms
5. **Early stopping**: Prevent overfitting automatically
6. **Validation**: Hold-out for large datasets, K-fold for small
7. **Complete loop**: Combine all techniques for robust training

## What's Next

You now understand training from scratch. Next: **PyTorch Fundamentals** - the industry-standard deep learning framework that automates most of this!
`,
};
