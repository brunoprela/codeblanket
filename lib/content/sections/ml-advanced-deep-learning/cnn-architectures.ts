/**
 * CNN Architectures Section
 */

export const cnnArchitecturesSection = {
  id: 'cnn-architectures',
  title: 'CNN Architectures',
  content: `# CNN Architectures

## Introduction

The evolution of CNN architectures over the past decade has been remarkable. From LeNet-5 (1998) to modern transformers, each architectural innovation has pushed the boundaries of what's possible in computer vision. Understanding these architectures teaches us fundamental design principles applicable to any deep learning problem.

**Why Study CNN Architectures**:
- **Learn design patterns** that generalize across domains
- **Understand trade-offs** between accuracy, speed, and memory
- **Historical perspective** shows what works and why
- **Transfer learning** - these architectures are the foundation
- **Inspiration** for your own architecture designs

**Key Metrics**:
- **Accuracy**: Performance on ImageNet or target task
- **Parameters**: Model size (storage, memory)
- **FLOPs**: Computational cost (speed, energy)
- **Depth**: Number of layers (representational power)

## LeNet-5 (1998) - The Pioneer

### Architecture

LeNet-5, designed by Yann LeCun, was the first successful CNN for handwritten digit recognition (used in ATMs, check reading).

**Structure**:
\`\`\`
INPUT(32×32) → CONV(6@28×28) → POOL(6@14×14) → 
CONV(16@10×10) → POOL(16@5×5) → 
FC(120) → FC(84) → OUTPUT(10)
\`\`\`

**Key Features**:
- Only ~60K parameters
- Used tanh activation (sigmoid was common then)
- Average pooling instead of max pooling
- Trained on MNIST dataset

\`\`\`python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    """
    LeNet-5 Architecture (1998)
    Original paper: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
    """
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        
        # Pooling
        self.pool = nn.AvgPool2d (kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    
    def forward (self, x):
        # Conv block 1: 32×32×1 → 28×28×6 → 14×14×6
        x = self.pool (torch.tanh (self.conv1(x)))
        
        # Conv block 2: 14×14×6 → 10×10×16 → 5×5×16
        x = self.pool (torch.tanh (self.conv2(x)))
        
        # Flatten: 5×5×16 → 400
        x = x.view(-1, 16 * 5 * 5)
        
        # Fully connected layers
        x = torch.tanh (self.fc1(x))
        x = torch.tanh (self.fc2(x))
        x = self.fc3(x)
        
        return x

# Create and analyze
lenet = LeNet5()
print("LeNet-5 Architecture:")
print(lenet)

# Count parameters
def count_params (model):
    return sum (p.numel() for p in model.parameters() if p.requires_grad)

print(f"\\nParameters: {count_params (lenet):,}")

# Test forward pass
x = torch.randn(1, 1, 32, 32)
output = lenet (x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")

print("\\nLeNet-5 Significance:")
print("✓ Proved CNNs could work on real-world tasks")
print("✓ Established conv→pool pattern")
print("✓ Inspired modern architectures")
print("✓ Still effective for simple tasks!")
\`\`\`

## AlexNet (2012) - The Deep Learning Revolution

### Architecture

AlexNet, by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton, won ImageNet 2012 by a huge margin, igniting the deep learning revolution.

**Innovations**:
1. **ReLU activation** instead of tanh (faster training, mitigates vanishing gradients)
2. **Dropout regularization** (prevent overfitting)
3. **Data augmentation** (translations, flips, color jittering)
4. **GPU training** (dual GPU architecture)
5. **Local Response Normalization** (later replaced by BatchNorm)

**Structure**:
\`\`\`
INPUT(224×224×3) → CONV1(55×55×96) → POOL → 
CONV2(27×27×256) → POOL → 
CONV3(13×13×384) → CONV4(13×13×384) → CONV5(13×13×256) → POOL →
FC(4096) → FC(4096) → OUTPUT(1000)
\`\`\`

- ~60M parameters
- 650K neurons
- Top-5 error: 15.3% (vs 26.2% runner-up)

\`\`\`python
class AlexNet (nn.Module):
    """
    AlexNet Architecture (2012)
    Paper: "ImageNet Classification with Deep Convolutional Neural Networks"
    """
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        
        # Feature extraction
        self.features = nn.Sequential(
            # Conv1: 224×224×3 → 55×55×96
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d (kernel_size=3, stride=2),
            
            # Conv2: 55×55×96 → 27×27×256
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d (kernel_size=3, stride=2),
            
            # Conv3: 27×27×256 → 13×13×384
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv4: 13×13×384 → 13×13×384
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv5: 13×13×384 → 13×13×256
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d (kernel_size=3, stride=2),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    
    def forward (self, x):
        x = self.features (x)
        x = x.view (x.size(0), -1)  # Flatten
        x = self.classifier (x)
        return x

# Create and analyze
alexnet = AlexNet (num_classes=1000)
print("AlexNet Architecture:")
print(f"Parameters: {count_params (alexnet):,}")

# Test
x = torch.randn(1, 3, 224, 224)
output = alexnet (x)
print(f"\\nInput: {x.shape}")
print(f"Output: {output.shape}")

print("\\nAlexNet Innovations:")
print("✓ ReLU activation (10× faster than tanh)")
print("✓ Dropout (p=0.5 in FC layers)")
print("✓ Heavy data augmentation")
print("✓ GPU parallelization")
print("✓ Won ImageNet 2012 by huge margin")
\`\`\`

## VGGNet (2014) - Deeper is Better

### Architecture

VGG (Visual Geometry Group, Oxford) showed that **depth matters** by using very deep networks with small 3×3 filters.

**Key Insights**:
1. **Small filters** (3×3) throughout entire network
2. **Homogeneous architecture** - easy to understand and modify
3. **Stacking small filters** = larger receptive field with fewer parameters
4. **Two 3×3 convs** = one 5×5 receptive field, but 18 params vs 25

**VGG-16 Structure**:
\`\`\`
INPUT → [CONV3-64] × 2 → POOL →
[CONV3-128] × 2 → POOL →
[CONV3-256] × 3 → POOL →
[CONV3-512] × 3 → POOL →
[CONV3-512] × 3 → POOL →
FC-4096 → FC-4096 → FC-1000
\`\`\`

- 16 weight layers (13 conv + 3 FC)
- ~138M parameters (mostly in FC layers!)
- Top-5 error: 7.3%

\`\`\`python
class VGG16(nn.Module):
    """
    VGG-16 Architecture (2014)
    Paper: "Very Deep Convolutional Networks for Large-Scale Image Recognition"
    """
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()
        
        # Feature extraction
        self.features = nn.Sequential(
            # Block 1: 224×224×3 → 112×112×64
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d (kernel_size=2, stride=2),
            
            # Block 2: 112×112×64 → 56×56×128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d (kernel_size=2, stride=2),
            
            # Block 3: 56×56×128 → 28×28×256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d (kernel_size=2, stride=2),
            
            # Block 4: 28×28×256 → 14×14×512
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d (kernel_size=2, stride=2),
            
            # Block 5: 14×14×512 → 7×7×512
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d (kernel_size=2, stride=2),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )
    
    def forward (self, x):
        x = self.features (x)
        x = x.view (x.size(0), -1)
        x = self.classifier (x)
        return x

# Create and analyze
vgg16 = VGG16()
print("VGG-16 Architecture:")
print(f"Parameters: {count_params (vgg16):,}")

# Parameter breakdown
feature_params = sum (p.numel() for p in vgg16.features.parameters())
classifier_params = sum (p.numel() for p in vgg16.classifier.parameters())

print(f"\\nParameter Breakdown:")
print(f"  Features (conv layers): {feature_params:,}")
print(f"  Classifier (FC layers): {classifier_params:,}")
print(f"  FC layers are {classifier_params/feature_params:.1f}× more parameters!")

print("\\nVGG Insights:")
print("✓ Small 3×3 filters throughout")
print("✓ Deeper networks learn better features")
print("✓ Simple, homogeneous architecture")
print("✗ Too many parameters (138M)")
print("✗ Slow training and inference")
\`\`\`

## ResNet (2015) - Residual Learning

### The Problem: Degradation

**Surprising Discovery**: Adding more layers to plain networks made them **worse**, even on training data! This wasn't overfitting - deeper networks had higher training error than shallow ones.

**Hypothesis**: Deep plain networks are hard to optimize. The problem is **gradient vanishing/exploding** in very deep networks.

### The Solution: Residual Connections

**Key Idea**: Instead of learning mapping H(x), learn the **residual** F(x) = H(x) - x, then add back x:
\`\`\`
H(x) = F(x) + x
\`\`\`

**Why This Works**:
1. **Identity shortcut**: If optimal mapping is identity, just set F(x) = 0 (easy to learn)
2. **Gradient flow**: Gradients flow directly through shortcuts (no vanishing)
3. **Easier optimization**: Learning residuals easier than learning full mapping

\`\`\`python
class ResidualBlock (nn.Module):
    """
    Basic Residual Block for ResNet
    
    x → [Conv→BN→ReLU→Conv→BN] → + → ReLU
    └────────── skip connection ────────┘
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # Main path
        self.conv1 = nn.Conv2d (in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d (out_channels)
        self.conv2 = nn.Conv2d (out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d (out_channels)
        
        # Shortcut connection (if dimensions change)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d (in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d (out_channels)
            )
    
    def forward (self, x):
        # Main path
        out = F.relu (self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Add shortcut (residual connection)
        out += self.shortcut (x)
        out = F.relu (out)
        
        return out

class ResNet18(nn.Module):
    """
    ResNet-18 Architecture (2015)
    Paper: "Deep Residual Learning for Image Recognition"
    """
    def __init__(self, num_classes=1000):
        super(ResNet18, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d (kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer (self, in_channels, out_channels, num_blocks, stride):
        layers = []
        # First block may downsample
        layers.append(ResidualBlock (in_channels, out_channels, stride))
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock (out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward (self, x):
        # Initial layers: 224×224×3 → 56×56×64
        x = F.relu (self.bn1(self.conv1(x)))
        x = self.maxpool (x)
        
        # Residual blocks
        x = self.layer1(x)  # 56×56×64
        x = self.layer2(x)  # 28×28×128
        x = self.layer3(x)  # 14×14×256
        x = self.layer4(x)  # 7×7×512
        
        # Global average pooling and classification
        x = self.avgpool (x)  # 1×1×512
        x = x.view (x.size(0), -1)  # 512
        x = self.fc (x)  # num_classes
        
        return x

# Create and analyze
resnet18 = ResNet18()
print("ResNet-18 Architecture:")
print(f"Parameters: {count_params (resnet18):,}")

# Test residual block
print("\\nResidual Block Test:")
block = ResidualBlock(64, 64)
x = torch.randn(1, 64, 56, 56)
output = block (x)
print(f"Input: {x.shape}")
print(f"Output: {output.shape}")
print("✓ Dimensions preserved with identity shortcut")

# Compare gradient flow
print("\\nGradient Flow Comparison:")
print("Plain Network:")
print("  x → conv1 → conv2 → ... → conv50")
print("  Gradient vanishes over 50 multiplications")
print("\\nResNet:")
print("  x → [block1 + x] → [block2 + x] → ... → [block50 + x]")
print("  Gradient flows directly through shortcuts!")

print("\\nResNet Impact:")
print("✓ Enabled training of 100+ layer networks")
print("✓ ResNet-152 won ImageNet 2015 (3.57% error)")
print("✓ Residual connections now ubiquitous")
print("✓ Foundation for modern architectures")
\`\`\`

### ResNet Variants

**ResNet Family**:
- **ResNet-18**: 18 layers, 11M params
- **ResNet-34**: 34 layers, 21M params
- **ResNet-50**: 50 layers, 25M params (uses bottleneck blocks)
- **ResNet-101**: 101 layers, 44M params
- **ResNet-152**: 152 layers, 60M params

**Bottleneck Block** (for deeper ResNets):
\`\`\`
x → [1×1 Conv (reduce)] → [3×3 Conv] → [1×1 Conv (expand)] + x
    256 → 64           → 64          → 256
\`\`\`
Reduces computation by first reducing channels, then expanding.

## Inception (GoogLeNet) (2014) - Multi-Scale Features

### The Inception Module

**Motivation**: Objects can appear at different scales. Use multiple filter sizes in parallel!

**Inception Module**:
\`\`\`
            ┌─ 1×1 Conv ─┐
            │            │
Input ──────┼─ 3×3 Conv ─┼──→ Concatenate → Output
            │            │
            ├─ 5×5 Conv ─┤
            │            │
            └─ MaxPool ──┘
\`\`\`

**Problem**: Computational explosion!
**Solution**: Use 1×1 convolutions to reduce channels before expensive operations

\`\`\`python
class InceptionModule (nn.Module):
    """
    Inception Module (with dimension reduction)
    """
    def __init__(self, in_channels, ch1x1, ch3x3_reduce, ch3x3, 
                 ch5x5_reduce, ch5x5, pool_proj):
        super(InceptionModule, self).__init__()
        
        # 1×1 convolution branch
        self.branch1 = nn.Sequential(
            nn.Conv2d (in_channels, ch1x1, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        # 3×3 convolution branch (with 1×1 reduction)
        self.branch2 = nn.Sequential(
            nn.Conv2d (in_channels, ch3x3_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d (ch3x3_reduce, ch3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 5×5 convolution branch (with 1×1 reduction)
        self.branch3 = nn.Sequential(
            nn.Conv2d (in_channels, ch5x5_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d (ch5x5_reduce, ch5x5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        
        # Max pooling branch (with 1×1 projection)
        self.branch4 = nn.Sequential(
            nn.MaxPool2d (kernel_size=3, stride=1, padding=1),
            nn.Conv2d (in_channels, pool_proj, kernel_size=1),
            nn.ReLU(inplace=True)
        )
    
    def forward (self, x):
        # Execute all branches in parallel
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        # Concatenate along channel dimension
        outputs = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        return outputs

# Test Inception module
print("Inception Module Test:")
inception = InceptionModule(
    in_channels=192,
    ch1x1=64,
    ch3x3_reduce=96, ch3x3=128,
    ch5x5_reduce=16, ch5x5=32,
    pool_proj=32
)

x = torch.randn(1, 192, 28, 28)
output = inception (x)
print(f"Input: {x.shape}")
print(f"Output: {output.shape}")
print(f"Output channels: 64 + 128 + 32 + 32 = {64+128+32+32}")

print("\\nInception Advantages:")
print("✓ Multi-scale feature extraction")
print("✓ 1×1 convs reduce computation")
print("✓ Efficient parameter usage")
print("✓ Won ImageNet 2014")
\`\`\`

## EfficientNet (2019) - Compound Scaling

### The Scaling Problem

**Question**: How should we scale up CNNs?
- More depth?
- More width (channels)?
- Higher resolution inputs?

**EfficientNet Answer**: Scale **all three** in balanced way!

**Compound Scaling**:
\`\`\`
depth: d = α^φ
width: w = β^φ
resolution: r = γ^φ

where α · β² · γ² ≈ 2 (constraint)
φ = compound coefficient
\`\`\`

\`\`\`python
# EfficientNet comparison

architectures = {
    'EfficientNet-B0': {'depth': 1.0, 'width': 1.0, 'resolution': 224, 
                        'params': '5.3M', 'top1': '77.3%'},
    'EfficientNet-B1': {'depth': 1.1, 'width': 1.0, 'resolution': 240,
                        'params': '7.8M', 'top1': '79.2%'},
    'EfficientNet-B7': {'depth': 2.0, 'width': 2.0, 'resolution': 600,
                        'params': '66M', 'top1': '84.4%'},
}

print("EfficientNet Family:")
print("=" * 80)
for name, specs in architectures.items():
    print(f"{name}:")
    print(f"  Depth scale: {specs['depth']}, Width scale: {specs['width']}")
    print(f"  Input: {specs['resolution']}×{specs['resolution']}")
    print(f"  Parameters: {specs['params']}, Top-1: {specs['top1']}\\n")

print("Key Insight:")
print("Compound scaling achieves better accuracy with fewer parameters")
print("than scaling depth, width, or resolution alone!")
\`\`\`

## Architecture Comparison

\`\`\`python
import pandas as pd

# Comparison table
comparison = pd.DataFrame([
    {
        'Architecture': 'LeNet-5',
        'Year': 1998,
        'Depth': 5,
        'Parameters': '60K',
        'Top-5 Error': 'N/A',
        'Innovation': 'First successful CNN'
    },
    {
        'Architecture': 'AlexNet',
        'Year': 2012,
        'Depth': 8,
        'Parameters': '60M',
        'Top-5 Error': '15.3%',
        'Innovation': 'ReLU, Dropout, GPU training'
    },
    {
        'Architecture': 'VGG-16',
        'Year': 2014,
        'Depth': 16,
        'Parameters': '138M',
        'Top-5 Error': '7.3%',
        'Innovation': 'Small 3×3 filters, deeper'
    },
    {
        'Architecture': 'GoogLeNet',
        'Year': 2014,
        'Depth': 22,
        'Parameters': '6.8M',
        'Top-5 Error': '6.7%',
        'Innovation': 'Inception modules, 1×1 convs'
    },
    {
        'Architecture': 'ResNet-152',
        'Year': 2015,
        'Depth': 152,
        'Parameters': '60M',
        'Top-5 Error': '3.57%',
        'Innovation': 'Residual connections'
    },
    {
        'Architecture': 'EfficientNet-B7',
        'Year': 2019,
        'Depth': 'Variable',
        'Parameters': '66M',
        'Top-5 Error': '2.0%',
        'Innovation': 'Compound scaling'
    },
])

print("CNN Architecture Evolution:")
print("=" * 100)
print(comparison.to_string (index=False))

print("\\n\\nKey Trends:")
print("1. Increasing depth: 5 → 152+ layers")
print("2. More efficient designs: VGG (138M) → GoogLeNet (6.8M) similar performance")
print("3. Better optimization: Residual connections enable ultra-deep networks")
print("4. Principled scaling: EfficientNet\'s compound scaling")
print("5. Accuracy improvement: 15.3% → 2.0% top-5 error in 7 years!")
\`\`\`

## Design Principles

### What We've Learned

**1. Depth Matters** (VGG, ResNet):
- Deeper networks learn richer representations
- But require special techniques (residual connections, normalization)

**2. Skip Connections** (ResNet):
- Enable gradient flow in deep networks
- Allow learning identity mappings
- Now standard in most architectures

**3. Multi-Scale Features** (Inception):
- Different filter sizes capture different scales
- 1×1 convolutions for dimension reduction
- Parallel branches increase expressiveness

**4. Efficient Design** (GoogLeNet, EfficientNet):
- More parameters ≠ better performance
- Balanced scaling (depth, width, resolution) is key
- Architecture search finds optimal designs

**5. Batch Normalization** (not covered in detail):
- Stabilizes training
- Allows higher learning rates
- Enables deeper networks

**6. Global Average Pooling** (ResNet, EfficientNet):
- Replaces FC layers
- Reduces parameters dramatically
- Better generalization

## Choosing an Architecture

**For Your Task**:

**Use pretrained ResNet/EfficientNet if**:
- You have limited data (transfer learning)
- Want proven, reliable architecture
- Need good accuracy with reasonable compute

**Use VGG if**:
- Simple architecture needed
- Interpretability matters
- Prototyping/teaching

**Build custom architecture if**:
- Very specific domain (medical imaging, satellite)
- Extreme constraints (mobile, edge devices)
- Research/exploration

**General Guidelines**:
1. **Start with pretrained models** (transfer learning)
2. **ResNet is safe default** (good accuracy, widely supported)
3. **EfficientNet for efficiency** (mobile, production)
4. **Consider compute budget** (params, FLOPs, latency)

## Key Takeaways

1. **CNN architectures evolved rapidly**: LeNet → AlexNet → VGG → ResNet → EfficientNet

2. **Major innovations**:
   - AlexNet: ReLU, dropout, GPU training
   - VGG: Depth with small filters
   - ResNet: Residual connections (breakthrough!)
   - Inception: Multi-scale features, 1×1 convs
   - EfficientNet: Compound scaling

3. **Residual connections** revolutionized deep learning (enabled 100+ layer networks)

4. **Efficiency matters**: GoogLeNet achieved VGG-level accuracy with 20× fewer parameters

5. **Transfer learning**: Use pretrained architectures for most tasks

6. **Design principles** from these architectures apply broadly across deep learning

## Coming Next

In the next section, we'll explore **Image Processing with CNNs** - applying these architectures to real-world vision tasks like object detection, segmentation, and more!
`,
};
