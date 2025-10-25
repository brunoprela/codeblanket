/**
 * Convolutional Neural Networks (CNNs) Section
 */

export const convolutionalNeuralNetworksSection = {
  id: 'convolutional-neural-networks',
  title: 'Convolutional Neural Networks (CNNs)',
  content: `# Convolutional Neural Networks (CNNs)

## Introduction

Convolutional Neural Networks (CNNs) are specialized neural networks designed to process grid-like data such as images. They've revolutionized computer vision, enabling breakthroughs in image classification, object detection, facial recognition, and medical imaging.

**Why CNNs Matter**:
- **Spatial Pattern Recognition**: Automatically learn hierarchical visual features
- **Parameter Efficiency**: Far fewer parameters than fully connected networks
- **Translation Invariance**: Detect features regardless of position in image
- **State-of-the-Art**: Superhuman performance on many vision tasks

**Key Insight**: Rather than treating images as flat vectors (losing spatial structure), CNNs preserve spatial relationships through specialized operations that exploit local connectivity.

## The Problem with Fully Connected Networks for Images

### Why Not Use Regular Neural Networks?

Consider a small 224×224 RGB image:
- Input size: 224 × 224 × 3 = 150,528 pixels
- First hidden layer with 1000 neurons: 150,528 × 1000 = **150 million parameters**!
- Computationally infeasible and prone to overfitting
- Doesn't exploit spatial structure or local patterns

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from PIL import Image

# Problem: Fully connected network for images
class FullyConnectedNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear (input_size, hidden_size)
        self.fc2 = nn.Linear (hidden_size, num_classes)
    
    def forward (self, x):
        # Flatten image
        x = x.view (x.size(0), -1)  # Loses spatial structure!
        x = F.relu (self.fc1(x))
        x = self.fc2(x)
        return x

# Example with MNIST (28x28 grayscale images)
input_size = 28 * 28  # 784 pixels
hidden_size = 256
num_classes = 10

fc_model = FullyConnectedNN(input_size, hidden_size, num_classes)

# Count parameters
def count_parameters (model):
    return sum (p.numel() for p in model.parameters() if p.requires_grad)

print(f"Fully Connected Model Parameters: {count_parameters (fc_model):,}")
# Output: ~200,000 parameters for tiny MNIST images!

# Problems:
print("\\nProblems with Fully Connected Networks:")
print("1. Too many parameters (overfitting risk)")
print("2. Loses spatial structure (treats pixels independently)")
print("3. Not translation invariant (same pattern at different locations = different neurons)")
print("4. Computationally expensive")
\`\`\`

## The Convolution Operation

### What is Convolution?

Convolution is a mathematical operation that slides a small filter (kernel) across an image, computing dot products at each position. This produces a **feature map** highlighting where the filter pattern appears in the image.

**Intuition**: 
- **Filter (Kernel)**: Small matrix of weights (e.g., 3×3) that detects a specific pattern
- **Sliding Window**: Move filter across image, one position at a time
- **Dot Product**: Multiply filter weights with image region, sum results
- **Feature Map**: Output showing filter response at each position

\`\`\`python
# Understanding Convolution from Scratch

def convolve2d (image, kernel, stride=1, padding=0):
    """
    Perform 2D convolution.
    
    Args:
        image: Input image (H, W) or (H, W, C)
        kernel: Convolution kernel/filter (k_h, k_w) or (k_h, k_w, C_in, C_out)
        stride: Step size for sliding window
        padding: Zero-padding around image
    
    Returns:
        Feature map after convolution
    """
    # Handle 2D grayscale
    if len (image.shape) == 2:
        image = image[:, :, np.newaxis]
    if len (kernel.shape) == 2:
        kernel = kernel[:, :, np.newaxis, np.newaxis]
    
    H, W, C_in = image.shape
    k_h, k_w, C_in, C_out = kernel.shape
    
    # Add padding
    if padding > 0:
        image = np.pad (image, ((padding, padding), (padding, padding), (0, 0)), mode='constant')
        H, W = image.shape[:2]
    
    # Calculate output dimensions
    out_h = (H - k_h) // stride + 1
    out_w = (W - k_w) // stride + 1
    
    # Initialize output
    output = np.zeros((out_h, out_w, C_out))
    
    # Perform convolution
    for c_out in range(C_out):
        for i in range (out_h):
            for j in range (out_w):
                h_start = i * stride
                w_start = j * stride
                h_end = h_start + k_h
                w_end = w_start + k_w
                
                # Extract region
                region = image[h_start:h_end, w_start:w_end, :]
                
                # Element-wise multiplication and sum
                output[i, j, c_out] = np.sum (region * kernel[:, :, :, c_out])
    
    return output

# Example: Edge detection with Sobel filter
# Create sample image
image = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0],
], dtype=np.float32) * 255

# Vertical edge detector (Sobel-X)
sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32).reshape(3, 3, 1, 1)

# Horizontal edge detector (Sobel-Y)
sobel_y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float32).reshape(3, 3, 1, 1)

# Apply convolutions
edges_x = convolve2d (image, sobel_x, stride=1, padding=0)
edges_y = convolve2d (image, sobel_y, stride=1, padding=0)

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow (image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow (edges_x.squeeze(), cmap='gray')
axes[1].set_title('Vertical Edges (Sobel-X)')
axes[1].axis('off')

axes[2].imshow (edges_y.squeeze(), cmap='gray')
axes[2].set_title('Horizontal Edges (Sobel-Y)')
axes[2].axis('off')

plt.tight_layout()
plt.show()

print("Convolution Output:")
print(f"Input shape: {image.shape}")
print(f"Kernel shape: {sobel_x.squeeze().shape}")
print(f"Output shape: {edges_x.shape}")
print(f"\\nNotice: Output is smaller due to kernel size (valid convolution)")
\`\`\`

### Stride and Padding

**Stride**: Controls how many pixels the filter moves at each step
- Stride = 1: Move one pixel at a time (dense sampling)
- Stride = 2: Move two pixels (skip positions, downsample)
- Larger stride = smaller output, less computation

**Padding**: Add zeros around image borders
- **Valid (No Padding)**: Output shrinks with each layer
- **Same Padding**: Output same size as input (common choice)
- **Full Padding**: Output larger than input (rare)

\`\`\`python
# Visualize Stride and Padding Effects

def calculate_output_size (input_size, kernel_size, stride, padding):
    """Calculate output dimensions after convolution."""
    return (input_size - kernel_size + 2 * padding) // stride + 1

# Example configurations
configs = [
    {'name': 'Valid (no padding)', 'padding': 0, 'stride': 1},
    {'name': 'Same padding', 'padding': 1, 'stride': 1},
    {'name': 'Stride 2 (downsample)', 'padding': 0, 'stride': 2},
    {'name': 'Stride 2 + padding', 'padding': 1, 'stride': 2},
]

input_size = 7
kernel_size = 3

print("Convolution Output Sizes")
print("=" * 60)
print(f"Input size: {input_size}×{input_size}")
print(f"Kernel size: {kernel_size}×{kernel_size}\\n")

for config in configs:
    output_size = calculate_output_size(
        input_size, kernel_size, config['stride'], config['padding']
    )
    print(f"{config['name']:25s} → Output: {output_size}×{output_size}")

# PyTorch implementation
print("\\n" + "=" * 60)
print("PyTorch Conv2d Example\\n")

# Create sample input: (batch_size, channels, height, width)
x = torch.randn(1, 1, 7, 7)  # 1 image, 1 channel, 7×7

for config in configs:
    conv = nn.Conv2d(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        stride=config['stride'],
        padding=config['padding']
    )
    output = conv (x)
    print(f"{config['name']:25s} → Output shape: {tuple (output.shape)}")
\`\`\`

## Filters and Feature Maps

### What Do Filters Learn?

In CNNs, filters are **learned** (not hand-crafted). Through training:
- **Early layers**: Learn low-level features (edges, corners, textures)
- **Middle layers**: Learn mid-level features (shapes, patterns, object parts)
- **Deep layers**: Learn high-level features (faces, objects, complex concepts)

\`\`\`python
# Visualize learned filters

# Common filter patterns
filters_examples = {
    'Vertical Edge': np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ]),
    'Horizontal Edge': np.array([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1]
    ]),
    'Diagonal Edge': np.array([
        [ 0, -1, -1],
        [ 1,  0, -1],
        [ 1,  1,  0]
    ]),
    'Sharpening': np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ]),
    'Blur': np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]) / 9.0,
}

# Load sample image
image = np.random.rand(28, 28) * 255

# Apply filters
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

axes[0].imshow (image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

for idx, (name, kernel) in enumerate (filters_examples.items(), 1):
    # Apply convolution
    kernel_3d = kernel.reshape(3, 3, 1, 1)
    filtered = convolve2d (image, kernel_3d)
    
    axes[idx].imshow (filtered.squeeze(), cmap='gray')
    axes[idx].set_title (f'{name} Filter')
    axes[idx].axis('off')

plt.tight_layout()
plt.show()

print("Filter Visualization:")
print("- Each filter detects different patterns")
print("- CNNs learn optimal filters during training")
print("- Multiple filters = multiple feature maps")
\`\`\`

### Multiple Feature Maps

Each convolutional layer has multiple filters, producing multiple feature maps:
- **Input**: (H, W, C_in) where C_in = number of input channels
- **Filters**: (k_h, k_w, C_in, C_out) where C_out = number of filters
- **Output**: (H', W', C_out) where C_out feature maps

\`\`\`python
# Multiple filters in PyTorch

# Create convolutional layer
# in_channels=3 (RGB), out_channels=32 (32 different filters)
conv_layer = nn.Conv2d (in_channels=3, out_channels=32, kernel_size=3, padding=1)

# Sample RGB image: (batch=1, channels=3, height=64, width=64)
rgb_image = torch.randn(1, 3, 64, 64)

# Forward pass
feature_maps = conv_layer (rgb_image)

print("Multiple Feature Maps:")
print(f"Input shape: {rgb_image.shape}")
print(f"  (batch, channels, height, width)")
print(f"\\nOutput shape: {feature_maps.shape}")
print(f"  32 feature maps, each 64×64")
print(f"\\nNumber of parameters: {conv_layer.weight.numel() + conv_layer.bias.numel()}")
print(f"  Filters: 3×3×3×32 = {3*3*3*32}")
print(f"  Biases: 32")
print(f"  Total: {3*3*3*32 + 32}")

# Compare to fully connected
fc_params = (3 * 64 * 64) * 32  # Input size × output size
print(f"\\nFully connected would need: {fc_params:,} parameters")
print(f"Convolution uses: {3*3*3*32 + 32} parameters")
print(f"Reduction: {fc_params / (3*3*3*32 + 32):.1f}× fewer parameters!")
\`\`\`

## Pooling Layers

### Why Pooling?

Pooling (downsampling) reduces spatial dimensions while retaining important features:
- **Reduce Computation**: Fewer pixels in subsequent layers
- **Translation Invariance**: Small shifts don't change output
- **Increase Receptive Field**: See larger image regions
- **Regularization**: Slight data compression prevents overfitting

### Types of Pooling

**Max Pooling** (most common):
- Take maximum value in each window
- Preserves strongest activations
- More common in practice

**Average Pooling**:
- Take average value in each window
- Smoother downsampling
- Used in some architectures (e.g., GoogLeNet)

\`\`\`python
# Pooling Operations

def max_pool2d (x, pool_size=2, stride=2):
    """Max pooling operation."""
    if len (x.shape) == 2:
        x = x[:, :, np.newaxis]
    
    H, W, C = x.shape
    out_h = (H - pool_size) // stride + 1
    out_w = (W - pool_size) // stride + 1
    
    output = np.zeros((out_h, out_w, C))
    
    for c in range(C):
        for i in range (out_h):
            for j in range (out_w):
                h_start = i * stride
                w_start = j * stride
                h_end = h_start + pool_size
                w_end = w_start + pool_size
                
                # Take maximum in window
                window = x[h_start:h_end, w_start:w_end, c]
                output[i, j, c] = np.max (window)
    
    return output

def avg_pool2d (x, pool_size=2, stride=2):
    """Average pooling operation."""
    if len (x.shape) == 2:
        x = x[:, :, np.newaxis]
    
    H, W, C = x.shape
    out_h = (H - pool_size) // stride + 1
    out_w = (W - pool_size) // stride + 1
    
    output = np.zeros((out_h, out_w, C))
    
    for c in range(C):
        for i in range (out_h):
            for j in range (out_w):
                h_start = i * stride
                w_start = j * stride
                h_end = h_start + pool_size
                w_end = w_start + pool_size
                
                # Take average in window
                window = x[h_start:h_end, w_start:w_end, c]
                output[i, j, c] = np.mean (window)
    
    return output

# Example
feature_map = np.array([
    [1, 3, 2, 4],
    [5, 6, 1, 3],
    [7, 2, 8, 1],
    [4, 5, 3, 9]
], dtype=np.float32)

max_pooled = max_pool2d (feature_map, pool_size=2, stride=2)
avg_pooled = avg_pool2d (feature_map, pool_size=2, stride=2)

print("Pooling Example:")
print(f"Input (4×4):\\n{feature_map}\\n")
print(f"Max Pooling (2×2):\\n{max_pooled.squeeze()}\\n")
print(f"Average Pooling (2×2):\\n{avg_pooled.squeeze()}")

# PyTorch pooling
print("\\n" + "=" * 50)
print("PyTorch Pooling\\n")

x_torch = torch.tensor (feature_map).unsqueeze(0).unsqueeze(0)  # (1, 1, 4, 4)

max_pool = nn.MaxPool2d (kernel_size=2, stride=2)
avg_pool = nn.AvgPool2d (kernel_size=2, stride=2)

print(f"Input shape: {x_torch.shape}")
print(f"Max pooled shape: {max_pool (x_torch).shape}")
print(f"Avg pooled shape: {avg_pool (x_torch).shape}")
\`\`\`

## Building a Complete CNN

### CNN Architecture Pattern

Typical CNN structure:
\`\`\`
INPUT → [CONV → RELU → POOL] × N → [FC → RELU] × M → OUTPUT
\`\`\`

- **Convolutional layers**: Feature extraction
- **Activation (ReLU)**: Non-linearity
- **Pooling**: Downsampling
- **Fully connected**: Classification

\`\`\`python
# Simple CNN for MNIST Classification

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d (in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d (in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d (in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d (kernel_size=2, stride=2)
        
        # Fully connected layers
        # After 3 pooling layers: 28×28 → 14×14 → 7×7 → 3×3 (with rounding)
        # Actually: 28×28 → 14×14 → 7×7 → 3×3 = 3×3×128 = 1152
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
    
    def forward (self, x):
        # Conv block 1: 28×28×1 → 14×14×32
        x = self.pool(F.relu (self.conv1(x)))
        
        # Conv block 2: 14×14×32 → 7×7×64
        x = self.pool(F.relu (self.conv2(x)))
        
        # Conv block 3: 7×7×64 → 3×3×128
        x = self.pool(F.relu (self.conv3(x)))
        
        # Flatten: 3×3×128 → 1152
        x = x.view(-1, 128 * 3 * 3)
        
        # Fully connected layers
        x = F.relu (self.fc1(x))
        x = self.dropout (x)
        x = self.fc2(x)
        
        return x

# Create model
model = SimpleCNN(num_classes=10)
print(model)
print(f"\\nTotal parameters: {count_parameters (model):,}")

# Test forward pass
sample_input = torch.randn(1, 1, 28, 28)  # Single MNIST image
output = model (sample_input)
print(f"\\nInput shape: {sample_input.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output.shape} (10 class scores)")

# Trace dimensions through network
print("\\n" + "=" * 60)
print("Dimension Flow Through Network:")
print("=" * 60)

def trace_dimensions (model, input_shape):
    """Trace tensor dimensions through network."""
    x = torch.randn (input_shape)
    
    print(f"Input: {tuple (x.shape)}")
    
    # Conv1 + Pool
    x = model.pool(F.relu (model.conv1(x)))
    print(f"After Conv1 + Pool: {tuple (x.shape)}")
    
    # Conv2 + Pool
    x = model.pool(F.relu (model.conv2(x)))
    print(f"After Conv2 + Pool: {tuple (x.shape)}")
    
    # Conv3 + Pool
    x = model.pool(F.relu (model.conv3(x)))
    print(f"After Conv3 + Pool: {tuple (x.shape)}")
    
    # Flatten
    x = x.view(-1, 128 * 3 * 3)
    print(f"After Flatten: {tuple (x.shape)}")
    
    # FC1
    x = F.relu (model.fc1(x))
    print(f"After FC1: {tuple (x.shape)}")
    
    # FC2
    x = model.fc2(x)
    print(f"After FC2 (Output): {tuple (x.shape)}")

trace_dimensions (model, (1, 1, 28, 28))
\`\`\`

## Why CNNs Work for Images

### Key Advantages

**1. Parameter Sharing**:
- Same filter used across entire image
- Dramatically reduces parameters
- Example: 3×3 filter = 9 parameters (vs thousands in FC layer)

**2. Sparse Connectivity**:
- Each output pixel depends on small local region
- Exploits spatial locality (nearby pixels are related)
- Reduces computation

**3. Translation Invariance**:
- Same filter detects feature anywhere in image
- Cat in top-left = same features as cat in bottom-right

**4. Hierarchical Feature Learning**:
- Low-level → Mid-level → High-level features
- Mimics visual cortex

\`\`\`python
# Comparison: CNN vs Fully Connected

def compare_architectures (input_shape=(1, 28, 28), num_classes=10):
    """Compare CNN vs FC network."""
    
    # CNN
    cnn = SimpleCNN(num_classes)
    cnn_params = count_parameters (cnn)
    
    # Fully Connected
    fc = FullyConnectedNN(np.prod (input_shape), 256, num_classes)
    fc_params = count_parameters (fc)
    
    print("Architecture Comparison")
    print("=" * 60)
    print(f"Input shape: {input_shape}")
    print(f"\\nCNN Parameters: {cnn_params:,}")
    print(f"FC Parameters: {fc_params:,}")
    print(f"\\nParameter reduction: {fc_params / cnn_params:.2f}×")
    print(f"\\nCNN Advantages:")
    print("✓ Fewer parameters (less overfitting)")
    print("✓ Preserves spatial structure")
    print("✓ Translation invariant")
    print("✓ Hierarchical features")
    print("✓ Faster training and inference")

compare_architectures()
\`\`\`

## Receptive Field

The **receptive field** is the region of the input image that affects a particular neuron's output.

- **Early layers**: Small receptive fields (local features)
- **Deep layers**: Large receptive fields (global context)

\`\`\`python
# Calculate receptive field

def calculate_receptive_field (layers):
    """
    Calculate receptive field size.
    
    Args:
        layers: List of (kernel_size, stride) tuples
    """
    rf = 1  # Starting receptive field
    
    print("Receptive Field Calculation:")
    print("=" * 50)
    print(f"Layer 0 (Input): RF = {rf}\\n")
    
    for i, (kernel_size, stride) in enumerate (layers, 1):
        # Receptive field grows with each layer
        rf = rf + (kernel_size - 1) * np.prod([s for _, s in layers[:i-1]], initial=1)
        print(f"Layer {i} (k={kernel_size}, s={stride}): RF = {rf}")
    
    return rf

# Example: Our SimpleCNN
layers = [
    (3, 1),  # Conv1: 3×3, stride 1
    (2, 2),  # Pool1: 2×2, stride 2
    (3, 1),  # Conv2: 3×3, stride 1
    (2, 2),  # Pool2: 2×2, stride 2
    (3, 1),  # Conv3: 3×3, stride 1
    (2, 2),  # Pool3: 2×2, stride 2
]

final_rf = calculate_receptive_field (layers)
print(f"\\nFinal receptive field: {final_rf}×{final_rf}")
print("→ Neurons in final layer see large image regions!")
\`\`\`

## Practical Training Example

Let\'s train our CNN on MNIST:

\`\`\`python
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Prepare data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean/std
])

train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root='./data', train=False, transform=transform
)

train_loader = DataLoader (train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader (test_dataset, batch_size=1000, shuffle=False)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN(num_classes=10).to (device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam (model.parameters(), lr=0.001)

# Training function
def train_epoch (model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    
    for batch_idx, (data, target) in enumerate (loader):
        data, target = data.to (device), target.to (device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model (data)
        loss = criterion (output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        pred = output.argmax (dim=1)
        correct += pred.eq (target).sum().item()
    
    avg_loss = total_loss / len (loader)
    accuracy = 100. * correct / len (loader.dataset)
    return avg_loss, accuracy

# Evaluation function
def evaluate (model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to (device), target.to (device)
            output = model (data)
            loss = criterion (output, target)
            
            total_loss += loss.item()
            pred = output.argmax (dim=1)
            correct += pred.eq (target).sum().item()
    
    avg_loss = total_loss / len (loader)
    accuracy = 100. * correct / len (loader.dataset)
    return avg_loss, accuracy

# Training loop
print("Training CNN on MNIST...")
print("=" * 60)

num_epochs = 5
for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train_epoch (model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = evaluate (model, test_loader, criterion, device)
    
    print(f"Epoch {epoch}/{num_epochs}:")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

print("\\n✓ Training complete!")
print(f"Final test accuracy: {test_acc:.2f}%")
\`\`\`

## Key Takeaways

1. **CNNs are specialized for spatial data** (images, video, etc.)

2. **Convolution operation**: Slides filters across input to detect patterns

3. **Key components**:
   - Convolutional layers: Feature extraction
   - Activation functions: Non-linearity
   - Pooling layers: Downsampling
   - Fully connected layers: Classification

4. **Advantages over fully connected networks**:
   - Parameter efficiency (sharing)
   - Preserves spatial structure
   - Translation invariance
   - Hierarchical feature learning

5. **Receptive field grows with depth**: Deep layers see large image regions

6. **Typical architecture**: \`[CONV → RELU → POOL] × N → [FC] × M → OUTPUT\`

## Coming Next

In the next section, we'll explore **CNN Architectures** - the evolution from LeNet to modern networks like ResNet, covering architectural innovations that enabled deeper, more powerful networks!
`,
};
