/**
 * Section: PyTorch Fundamentals
 * Module: Deep Learning Fundamentals
 *
 * Covers PyTorch tensors, autograd, nn.Module, building models, training loops,
 * GPU acceleration, and saving/loading models
 */

export const pytorchFundamentalsSection = {
  id: 'pytorch-fundamentals',
  title: 'PyTorch Fundamentals',
  content: `
# PyTorch Fundamentals

## Introduction

**PyTorch** is the most popular deep learning framework in research and increasingly in production. It provides:
- Tensor operations (like NumPy on GPU)
- Automatic differentiation (autograd)
- Neural network building blocks
- GPU acceleration
- Production deployment tools

**What You'll Learn:**
- PyTorch tensors and operations
- Autograd: automatic differentiation
- Building models with nn.Module
- Complete training pipeline
- GPU acceleration
- Saving and loading models

## Installing PyTorch

\`\`\`bash
# CPU version
pip install torch torchvision

# GPU version (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Check installation
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
\`\`\`

## PyTorch Tensors

### Creating Tensors

\`\`\`python
import torch
import numpy as np

# From Python list
x = torch.tensor([1, 2, 3, 4, 5])
print(x)  # tensor([1, 2, 3, 4, 5])

# From NumPy array
arr = np.array([1, 2, 3])
x = torch.from_numpy(arr)
print(x)  # tensor([1, 2, 3])

# Common creation functions
zeros = torch.zeros(3, 4)           # 3x4 tensor of zeros
ones = torch.ones(2, 3)             # 2x3 tensor of ones
rand = torch.rand(2, 3)             # Random uniform [0, 1)
randn = torch.randn(2, 3)           # Random normal N(0, 1)
arange = torch.arange(0, 10, 2)     # [0, 2, 4, 6, 8]
linspace = torch.linspace(0, 1, 5)  # [0.0, 0.25, 0.5, 0.75, 1.0]

# Specify dtype
x = torch.tensor([1, 2, 3], dtype=torch.float32)
print(x.dtype)  # torch.float32
\`\`\`

### Tensor Operations

\`\`\`python
# Basic operations
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

print(x + y)      # Element-wise addition
print(x * y)      # Element-wise multiplication
print(x @ y)      # Dot product (1*4 + 2*5 + 3*6 = 32)

# Matrix operations
A = torch.randn(3, 4)
B = torch.randn(4, 5)

C = torch.matmul(A, B)  # Matrix multiplication (3x5)
# Or use @ operator
C = A @ B

# Reshaping
x = torch.arange(12)
x_reshaped = x.view(3, 4)      # Reshape to 3x4
x_reshaped = x.reshape(3, 4)   # Alternative (safer)

# Indexing and slicing
x = torch.arange(12).reshape(3, 4)
print(x[0, :])        # First row
print(x[:, 1])        # Second column
print(x[0:2, 1:3])    # Submatrix

# Broadcasting
x = torch.ones(3, 4)
y = torch.ones(4)
z = x + y  # y is broadcast to (3, 4)
\`\`\`

## Autograd: Automatic Differentiation

### Computing Gradients

\`\`\`python
# Create tensor with gradient tracking
x = torch.tensor([2.0], requires_grad=True)

# Compute y = x^2
y = x ** 2

# Compute gradient dy/dx
y.backward()

# Access gradient
print(x.grad)  # tensor([4.]) since dy/dx = 2x = 2*2 = 4
\`\`\`

### Multi-variable Example

\`\`\`python
# f(x, y) = x^2 + y^3
x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)

f = x**2 + y**3

f.backward()

print(x.grad)  # df/dx = 2x = 4
print(y.grad)  # df/dy = 3y^2 = 27
\`\`\`

### Gradient Accumulation

\`\`\`python
x = torch.tensor([1.0], requires_grad=True)

# Gradients accumulate by default!
y = x ** 2
y.backward()
print(x.grad)  # tensor([2.])

y = x ** 2
y.backward()
print(x.grad)  # tensor([4.]) - accumulated!

# Zero gradients before next backward pass
x.grad.zero_()
y = x ** 2
y.backward()
print(x.grad)  # tensor([2.])
\`\`\`

### Computational Graph Example

\`\`\`python
# Build computation graph
x = torch.tensor([2.0], requires_grad=True)
w = torch.tensor([3.0], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)

# y = w * x + b
y = w * x + b  # y = 3*2 + 1 = 7

# Loss = (y - target)^2
target = torch.tensor([10.0])
loss = (y - target) ** 2  # (7 - 10)^2 = 9

# Compute gradients
loss.backward()

print(f"dL/dx = {x.grad}")  # Gradient of loss w.r.t. x
print(f"dL/dw = {w.grad}")  # Gradient of loss w.r.t. w
print(f"dL/db = {b.grad}")  # Gradient of loss w.r.t. b
\`\`\`

## Building Neural Networks

### Using nn.Module

\`\`\`python
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Define forward pass
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create model
model = SimpleNet(input_size=784, hidden_size=128, output_size=10)

# Forward pass
x = torch.randn(32, 784)  # Batch of 32 samples
output = model(x)
print(output.shape)  # torch.Size([32, 10])

# Access parameters
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
\`\`\`

### More Complex Model

\`\`\`python
class DeepNet(nn.Module):
    def __init__(self):
        super(DeepNet, self).__init__()
        
        self.fc1 = nn.Linear(784, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x

model = DeepNet()
print(model)
\`\`\`

### Sequential API (Simpler)

\`\`\`python
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.5),
    
    nn.Linear(512, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.5),
    
    nn.Linear(256, 10)
)
\`\`\`

## Complete Training Pipeline

\`\`\`python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 1. Prepare data
X_train = torch.randn(1000, 784)
y_train = torch.randint(0, 10, (1000,))
X_val = torch.randn(200, 784)
y_val = torch.randint(0, 10, (200,))

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 2. Define model
model = DeepNet()

# 3. Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Training loop
num_epochs = 10

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient computation
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    val_loss /= len(val_loader)
    val_acc = correct / total
    
    print(f"Epoch {epoch+1}/{num_epochs}: "
          f"Train Loss: {train_loss:.4f}, "
          f"Val Loss: {val_loss:.4f}, "
          f"Val Acc: {val_acc:.4f}")
\`\`\`

## GPU Acceleration

\`\`\`python
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move model to GPU
model = model.to(device)

# Training loop with GPU
for batch_X, batch_y in train_loader:
    # Move data to GPU
    batch_X = batch_X.to(device)
    batch_y = batch_y.to(device)
    
    # Forward pass (happens on GPU)
    outputs = model(batch_X)
    loss = criterion(outputs, batch_y)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Move tensor between devices
x_cpu = torch.randn(10)
x_gpu = x_cpu.to("cuda")
x_back_to_cpu = x_gpu.to("cpu")
# Or
x_back_to_cpu = x_gpu.cpu()
\`\`\`

## Saving and Loading Models

### Save/Load State Dict (Recommended)

\`\`\`python
# Save model
torch.save(model.state_dict(), 'model.pth')

# Load model
model = DeepNet()  # Must create model with same architecture
model.load_state_dict(torch.load('model.pth'))
model.eval()  # Set to evaluation mode
\`\`\`

### Save/Load Entire Model

\`\`\`python
# Save entire model
torch.save(model, 'model_complete.pth')

# Load entire model
model = torch.load('model_complete.pth')
model.eval()
\`\`\`

### Save Checkpoint (Training State)

\`\`\`python
# Save checkpoint
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pth')

# Load checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
\`\`\`

## Key Takeaways

1. **PyTorch tensors** - like NumPy but with GPU support
2. **Autograd** - automatic differentiation for any computation
3. **nn.Module** - base class for all models
4. **Forward method** - defines computation
5. **Training loop** - train(), eval(), zero_grad(), backward(), step()
6. **GPU acceleration** - .to(device) for models and tensors
7. **Save/load** - state_dict() for portability

## What's Next

PyTorch is the research standard. **TensorFlow/Keras** is another major framework with different design philosophy and strong production tools. We'll explore it next!
`,
};
