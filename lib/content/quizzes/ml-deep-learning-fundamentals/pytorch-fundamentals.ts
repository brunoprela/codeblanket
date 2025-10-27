import { QuizQuestion } from '../../../types';

export const pytorchFundamentalsQuiz: QuizQuestion[] = [
  {
    id: 'pytorch-q1',
    question:
      "Explain how PyTorch\'s autograd system works. What is a computational graph, how does it track operations, and how does the backward() method compute gradients? Why is it important to call optimizer.zero_grad() before each training step?",
    sampleAnswer: `PyTorch's autograd is the engine that makes deep learning possible by automatically computing gradients for any operation. Understanding it is crucial for effective PyTorch usage:

**Computational Graph:**

A directed acyclic graph (DAG) that records all operations performed on tensors:
- Nodes: tensors and operations
- Edges: data flow between operations
- Built dynamically during forward pass
- Used for backward pass to compute gradients

**How Autograd Tracks Operations:**1. **Requires Grad Flag:**
\`\`\`python
x = torch.tensor([2.0], requires_grad=True)  # Track this tensor
\`\`\`

2. **Operation Recording:**
Each operation creates a Function object that knows:
- How to compute forward pass
- How to compute backward pass (gradient)
- Links to input tensors

3. **Graph Construction:**
\`\`\`python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2      # Creates PowBackward function
z = y + 3       # Creates AddBackward function
loss = z.mean() # Creates MeanBackward function
\`\`\`

Graph: x → Pow → y → Add → z → Mean → loss

**The backward() Method:**

Computes gradients using chain rule, traversing graph in reverse:

\`\`\`python
loss.backward()  # Computes gradients for all tensors
print(x.grad)    # Access gradient
\`\`\`

Process:
1. Start at loss: dL/dL = 1
2. Traverse backward: dL/dz, dL/dy, dL/dx
3. Apply chain rule at each node
4. Store gradients in .grad attribute

**Why zero_grad() Is Essential:**

PyTorch accumulates gradients by default:

\`\`\`python
# Iteration 1
loss1.backward()  # x.grad = grad1

# Iteration 2 (without zero_grad)
loss2.backward()  # x.grad = grad1 + grad2  # WRONG!

# Correct:
optimizer.zero_grad()  # x.grad = 0
loss2.backward()       # x.grad = grad2  # CORRECT
\`\`\`

**Why accumulation exists:**
- Useful for gradient accumulation (simulating large batches)
- Useful for multiple loss terms
- But problematic for normal training

**Best Practice:**
Always call optimizer.zero_grad() or model.zero_grad() before backward()`,
    keyPoints: [
      'Autograd builds computational graph tracking all operations on tensors with requires_grad=True',
      'Graph nodes store forward computation and gradient functions',
      'backward() traverses graph in reverse, applying chain rule',
      'Gradients accumulate by default in .grad attribute',
      'Must call zero_grad() before each backward pass',
      'Without zero_grad(), gradients from multiple iterations add up incorrectly',
      'Dynamic graphs rebuilt each forward pass (flexible control flow)',
      'Autograd enables automatic differentiation for any PyTorch operation',
    ],
  },
  {
    id: 'pytorch-q2',
    question:
      'Compare creating models using nn.Module subclasses vs nn.Sequential. What are the advantages and limitations of each approach? When would you choose one over the other?',
    sampleAnswer: `PyTorch offers two main ways to define models, each with distinct use cases:

**nn.Sequential - Simple Linear Architectures:**

\`\`\`python
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 10)
)
\`\`\`

**Advantages:**
- Concise, minimal boilerplate
- Clear, readable for simple models
- Good for rapid prototyping
- Layers execute in order automatically

**Limitations:**
- No branching or skip connections
- No conditional execution
- Single input, single output only
- Cannot share parameters between layers
- No access to intermediate activations
- Cannot implement complex architectures

**nn.Module - Full Flexibility:**

\`\`\`python
class CustomModel (nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 512)
        self.layer2 = nn.Linear(512, 10)
    
    def forward (self, x):
        x = F.relu (self.layer1(x))
        if self.training:  # Conditional logic
            x = F.dropout (x, 0.5)
        return self.layer2(x)
\`\`\`

**Advantages:**
- Complete control over forward pass
- Can implement any architecture
- Support for skip connections (ResNet)
- Multiple inputs/outputs
- Conditional execution (if/else, loops)
- Access intermediate values
- Custom operations

**Limitations:**
- More boilerplate code
- Must define forward() explicitly
- Slightly more complex for beginners

**When to Use Each:**

**Use Sequential for:**
- Simple feedforward networks
- Quick prototyping
- Linear layer stacks
- No branching needed
- Example: Basic MLP for MNIST

**Use nn.Module for:**
- ResNets (skip connections)
- Multi-task models (multiple outputs)
- Attention mechanisms
- Custom training logic
- Any non-trivial architecture
- Example: Transformer, U-Net, GAN

**Hybrid Approach:**

Best of both worlds - use Sequential within nn.Module:

\`\`\`python
class HybridModel (nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.decoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    
    def forward (self, x):
        features = self.encoder (x)
        # Custom logic here
        output = self.decoder (features)
        return output
\`\`\``,
    keyPoints: [
      'Sequential: concise, linear layer stacks, no branching',
      'nn.Module: full flexibility, any architecture, custom logic',
      'Sequential limitations: no skip connections, single input/output',
      'nn.Module enables: ResNets, multi-task, attention, conditionals',
      'Both inherit from nn.Module (Sequential is special case)',
      'Use Sequential for simple feedforward networks',
      'Use nn.Module for complex modern architectures',
      'Can combine: Sequential blocks within nn.Module',
    ],
  },
  {
    id: 'pytorch-q3',
    question:
      'Describe the complete training loop in PyTorch. What is the purpose of model.train() vs model.eval()? Why use torch.no_grad() during validation? How does moving tensors and models to GPU affect the training pipeline?',
    sampleAnswer: `A proper PyTorch training loop involves several critical steps, each serving a specific purpose:

**Complete Training Loop Structure:**

\`\`\`python
model.train()  # Set to training mode

for epoch in range (num_epochs):
    for batch_x, batch_y in train_loader:
        # Move to GPU
        batch_x = batch_x.to (device)
        batch_y = batch_y.to (device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model (batch_x)
        loss = criterion (outputs, batch_y)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        for val_x, val_y in val_loader:
            val_x = val_x.to (device)
            val_y = val_y.to (device)
            outputs = model (val_x)
            val_loss = criterion (outputs, val_y)
    
    model.train()  # Back to training mode
\`\`\`

**model.train() vs model.eval():**

**train() mode:**
- Enables dropout (randomly drops neurons)
- Uses batch statistics for BatchNorm
- Layers behave as intended for training

**eval() mode:**
- Disables dropout (uses all neurons)
- Uses running statistics for BatchNorm
- Deterministic behavior for inference

**Critical:** Forgetting to call eval() during validation gives incorrect results!

**torch.no_grad() During Validation:**

\`\`\`python
with torch.no_grad():
    outputs = model (inputs)
\`\`\`

**Why it's essential:**
- Disables gradient tracking
- Reduces memory usage by ~50%
- Faster computation
- We don't need gradients (no backprop during validation)

**Without no_grad():**
- Wastes memory storing gradients
- Slower computation
- Same results, but inefficient

**GPU Training:**

**Setup:**
\`\`\`python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to (device)  # Move model once
\`\`\`

**In training loop:**
\`\`\`python
data = data.to (device)    # Move each batch
target = target.to (device)
\`\`\`

**Key points:**
- All tensors in operation must be on same device
- Model moved once at start
- Data moved for each batch
- GPU dramatically speeds up training (10-100x)
- Larger batches benefit more from GPU

**Common Mistakes:**1. Forgetting eval() during validation
2. Not using no_grad() for inference
3. Mixing CPU and GPU tensors
4. Not zeroing gradients
5. Forgetting to move data to GPU

**Complete Best Practice:**
\`\`\`python
def train_epoch (model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for data, target in loader:
        data, target = data.to (device), target.to (device)
        optimizer.zero_grad()
        output = model (data)
        loss = criterion (output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len (loader)

def validate (model, loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to (device), target.to (device)
            output = model (data)
            loss = criterion (output, target)
            total_loss += loss.item()
    
    return total_loss / len (loader)
\`\`\``,
    keyPoints: [
      'Training loop: zero_grad() → forward() → loss.backward() → optimizer.step()',
      'model.train() enables dropout and uses batch statistics',
      'model.eval() disables dropout and uses running statistics',
      'torch.no_grad() disables gradient tracking for efficiency',
      'Without no_grad(), validation wastes memory storing unused gradients',
      'GPU usage: model.to (device) once, data.to (device) per batch',
      'All tensors must be on same device (CPU or GPU)',
      'Always call eval() before validation/inference',
    ],
  },
];
