/**
 * Image Processing with CNNs Section
 */

export const imageProcessingWithCnnsSection = {
  id: 'image-processing-with-cnns',
  title: 'Image Processing with CNNs',
  content: `# Image Processing with CNNs

## Introduction

Beyond basic image classification, CNNs power a wide array of computer vision tasks: object detection, image segmentation, style transfer, and more. Understanding these applications reveals how CNNs can be adapted to different problem structures.

**Key Applications**:
- **Object Detection**: Locate and classify multiple objects in images
- **Semantic Segmentation**: Classify every pixel
- **Instance Segmentation**: Separate individual object instances
- **Image Generation**: GANs, style transfer
- **Image Enhancement**: Super-resolution, denoising

## Data Augmentation

### Why Data Augmentation?

Deep CNNs need **lots** of data to avoid overfitting. Data augmentation artificially increases dataset size by creating modified versions of existing images that preserve the label.

**Benefits**:
- Reduces overfitting
- Makes model robust to variations
- Improves generalization
- Simulates real-world conditions

\`\`\`python
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Common augmentation techniques
augmentations = transforms.Compose([
    transforms.RandomHorizontalFlip (p=0.5),      # Flip horizontally
    transforms.RandomRotation (degrees=15),        # Rotate ±15°
    transforms.ColorJitter(                       # Color variations
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.RandomResizedCrop(                 # Random crop + resize
        size=224,
        scale=(0.8, 1.0),
        ratio=(0.9, 1.1)
    ),
    transforms.RandomAffine(                      # Affine transformations
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1)
    ),
])

# Load and augment image
def show_augmentations (image_path, num_samples=8):
    """Display original image and augmented versions."""
    original = Image.open (image_path)
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.ravel()
    
    # Show original
    axes[0].imshow (original)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Show augmented versions
    for i in range(1, 9):
        augmented = augmentations (original)
        axes[i].imshow (augmented)
        axes[i].set_title (f'Augmented {i}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Advanced augmentation with albumentations
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    # More sophisticated augmentations
    advanced_transform = A.Compose([
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
        A.OneOf([
            A.GaussNoise(),
            A.GaussianBlur(),
            A.MotionBlur(),
        ], p=0.2),
        A.OneOf([
            A.OpticalDistortion(),
            A.GridDistortion(),
            A.ElasticTransform(),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.Sharpen(),
            A.Emboss(),
            A.RandomBrightnessContrast(),
        ], p=0.3),
        A.HueSaturationValue (p=0.3),
    ])
    
    print("Advanced augmentation pipeline loaded!")
    print("Includes: Geometric transforms, noise, blur, distortions, color adjustments")
    
except ImportError:
    print("albumentations not installed. Use: pip install albumentations")

# Best practices for augmentation
print("\\nData Augmentation Best Practices:")
print("=" * 60)
print("1. ✓ Use augmentations that preserve semantic meaning")
print("   (horizontal flip for cats ✓, vertical flip ✗)")
print("\\n2. ✓ Match augmentations to real-world variations")
print("   (rotation for handwriting ✓, for text orientation ✗)")
print("\\n3. ✓ Apply different augmentations at different stages")
print("   (strong aug for training, no aug for validation)")
print("\\n4. ✓ Start with simple augmentations, add complexity gradually")
print("\\n5. ✓ Monitor validation performance (too much aug can hurt)")
\`\`\`

## Transfer Learning and Fine-Tuning

### The Power of Pretrained Models

Training CNNs from scratch requires huge datasets and computational resources. **Transfer learning** leverages models pretrained on large datasets (like ImageNet) and adapts them to your task.

**Why Transfer Learning Works**:
- Early layers learn general features (edges, textures)
- These features transfer across domains
- Fine-tuning adapts high-level features to your task

\`\`\`python
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

# Load pretrained ResNet
def create_transfer_model (num_classes, freeze_features=True):
    """
    Create transfer learning model from pretrained ResNet.
    
    Args:
        num_classes: Number of classes in your dataset
        freeze_features: If True, freeze convolutional layers
    """
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=True)
    
    print(f"Loaded ResNet-18 pretrained on ImageNet")
    print(f"Original classifier: {model.fc}")
    
    # Freeze feature extraction layers
    if freeze_features:
        for param in model.parameters():
            param.requires_grad = False
        print("\\n✓ Froze all layers (feature extraction only)")
    
    # Replace final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear (num_features, num_classes)
    print(f"\\n✓ Replaced classifier: {num_features} → {num_classes} classes")
    
    return model

# Example: Fine-tuning for 10-class dataset
model = create_transfer_model (num_classes=10, freeze_features=True)

# Count trainable parameters
trainable_params = sum (p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum (p.numel() for p in model.parameters())

print(f"\\nTrainable parameters: {trainable_params:,}")
print(f"Total parameters: {total_params:,}")
print(f"Frozen: {(total_params - trainable_params) / total_params * 100:.1f}%")

# Training strategy
print("\\n" + "=" * 60)
print("Transfer Learning Strategy:")
print("=" * 60)
print("\\nPhase 1: Train only classifier (features frozen)")
print("  - Fast training (few parameters)")
print("  - High learning rate OK (won't damage pretrained features)")
print("  - 5-10 epochs usually sufficient")
print("\\nPhase 2: Fine-tune top layers")
print("  - Unfreeze last few conv blocks")
print("  - Lower learning rate (don't destroy pretrained weights)")
print("  - 10-20 additional epochs")
print("\\nPhase 3: (Optional) Fine-tune entire network")
print("  - Unfreeze all layers")
print("  - Very low learning rate")
print("  - Small datasets: skip this step")

# Implement progressive unfreezing
def set_parameter_requires_grad (model, feature_extracting):
    """Set requires_grad for model parameters."""
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def unfreeze_layers (model, num_layers):
    """Unfreeze last N layers of model."""
    # Get all layers
    layers = list (model.children())
    
    # Freeze all
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze last num_layers
    for layer in layers[-num_layers:]:
        for param in layer.parameters():
            param.requires_grad = True
    
    trainable = sum (p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Unfroze last {num_layers} layers: {trainable:,} trainable params")

# Example: Progressive fine-tuning
print("\\n" + "=" * 60)
print("Progressive Fine-Tuning Example:")
print("=" * 60)

# Phase 1: Train classifier only
optimizer = optim.Adam (model.fc.parameters(), lr=0.001)
print("\\nPhase 1: Training classifier only (lr=0.001)")

# Phase 2: Fine-tune last layer
unfreeze_layers (model, num_layers=2)
optimizer = optim.Adam(
    filter (lambda p: p.requires_grad, model.parameters()),
    lr=0.0001
)
print("Phase 2: Fine-tuning last 2 layers (lr=0.0001)")
\`\`\`

## Object Detection

### From Classification to Detection

**Classification**: What\'s in the image?
**Detection**: What's in the image and **where**?

**Challenges**:
- Variable number of objects
- Different object sizes
- Precise localization required
- Real-time inference often needed

### YOLO (You Only Look Once)

YOLO treats detection as a regression problem: predict bounding boxes and class probabilities directly.

**Architecture**:
1. Divide image into S×S grid
2. Each grid cell predicts B bounding boxes
3. Each box predicts: (x, y, w, h, confidence, class_probs)
4. Single forward pass (very fast!)

\`\`\`python
# Simplified YOLO concept
class SimpleYOLO(nn.Module):
    """
    Simplified YOLO for educational purposes.
    Real YOLO is much more complex!
    """
    def __init__(self, num_classes=20, num_boxes=2, grid_size=7):
        super(SimpleYOLO, self).__init__()
        
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.grid_size = grid_size
        
        # Feature extraction (simplified)
        self.features = models.resnet18(pretrained=True)
        self.features.fc = nn.Identity()  # Remove classifier
        
        # Detection head
        # Output: grid_size × grid_size × (num_boxes * 5 + num_classes)
        # Each box: (x, y, w, h, confidence)
        output_size = grid_size * grid_size * (num_boxes * 5 + num_classes)
        self.detector = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, output_size),
        )
    
    def forward (self, x):
        # Extract features
        features = self.features (x)
        
        # Predict detections
        detections = self.detector (features)
        
        # Reshape to grid format
        batch_size = x.size(0)
        detections = detections.view(
            batch_size,
            self.grid_size,
            self.grid_size,
            self.num_boxes * 5 + self.num_classes
        )
        
        return detections

# Create model
yolo = SimpleYOLO(num_classes=20, num_boxes=2, grid_size=7)
print("Simplified YOLO Architecture:")
print(f"Grid size: 7×7")
print(f"Boxes per cell: 2")
print(f"Classes: 20")
print(f"Output shape per image: (7, 7, {2*5 + 20})")
print("  → 7×7 grid, each cell predicts 2 boxes (x,y,w,h,conf) + 20 class probs")

# Test
x = torch.randn(1, 3, 224, 224)
output = yolo (x)
print(f"\\nInput: {x.shape}")
print(f"Output: {output.shape}")

print("\\n" + "=" * 60)
print("YOLO Advantages:")
print("  ✓ Very fast (real-time detection)")
print("  ✓ Single forward pass")
print("  ✓ Global context (sees entire image)")
print("\\nYOLO Limitations:")
print("  ✗ Struggles with small objects")
print("  ✗ Limited boxes per grid cell")
print("  ✗ Less accurate than two-stage detectors (R-CNN family)")
\`\`\`

## Semantic Segmentation

### Pixel-Level Classification

Semantic segmentation classifies **every pixel** in the image. Unlike detection (bounding boxes), segmentation provides precise object boundaries.

**Applications**:
- Autonomous driving (road, cars, pedestrians)
- Medical imaging (tumor segmentation)
- Satellite imagery (land classification)
- Background removal

### U-Net Architecture

U-Net is the most popular architecture for segmentation, especially in medical imaging.

**Structure**:
\`\`\`
Encoder (Downsampling)    Decoder (Upsampling)
       ↓                        ↑
    [Conv]  ←─ skip connection ─→ [Conv]
       ↓                        ↑
    [Pool]                  [Upsample]
       ↓                        ↑
    [Conv]  ←─ skip connection ─→ [Conv]
\`\`\`

\`\`\`python
class UNet (nn.Module):
    """
    U-Net for semantic segmentation.
    """
    def __init__(self, in_channels=3, num_classes=1):
        super(UNet, self).__init__()
        
        # Encoder (downsampling)
        self.enc1 = self.conv_block (in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder (upsampling)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)  # 1024 because of skip connection
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        # Final classifier
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)
        
        # Pooling
        self.pool = nn.MaxPool2d (kernel_size=2, stride=2)
    
    def conv_block (self, in_channels, out_channels):
        """Two conv layers with ReLU."""
        return nn.Sequential(
            nn.Conv2d (in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d (out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    
    def forward (self, x):
        # Encoder
        enc1 = self.enc1(x)           # 64 channels
        enc2 = self.enc2(self.pool (enc1))  # 128 channels, /2 spatial
        enc3 = self.enc3(self.pool (enc2))  # 256 channels, /4 spatial
        enc4 = self.enc4(self.pool (enc3))  # 512 channels, /8 spatial
        
        # Bottleneck
        bottleneck = self.bottleneck (self.pool (enc4))  # 1024 channels, /16 spatial
        
        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)  # Skip connection
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Final classification
        out = self.out (dec1)
        
        return out

# Create U-Net
unet = UNet (in_channels=3, num_classes=1)  # Binary segmentation
print("U-Net Architecture:")
print(unet)

# Test
x = torch.randn(1, 3, 256, 256)
output = unet (x)
print(f"\\nInput: {x.shape}")
print(f"Output: {output.shape}")
print("Output is same spatial size as input! (pixel-wise prediction)")

print("\\n" + "=" * 60)
print("U-Net Key Features:")
print("  ✓ Skip connections preserve spatial information")
print("  ✓ Encoder: Extract high-level features")
print("  ✓ Decoder: Upsample to original resolution")
print("  ✓ Works great with small datasets")
print("  ✓ State-of-the-art for medical imaging")
\`\`\`

## Style Transfer

### Neural Style Transfer

Transfer the artistic style of one image to the content of another using CNN features.

**Key Idea**: 
- **Content** represented by deep layer activations (high-level features)
- **Style** represented by correlations between features (Gram matrices)
- Optimize new image to match both content and style

\`\`\`python
class StyleTransfer:
    """
    Neural Style Transfer using pretrained VGG.
    Based on Gatys et al. (2015)
    """
    def __init__(self, device='cpu'):
        self.device = device
        
        # Use pretrained VGG for feature extraction
        vgg = models.vgg19(pretrained=True).features.to (device).eval()
        
        # Freeze parameters
        for param in vgg.parameters():
            param.requires_grad = False
        
        self.vgg = vgg
        
        # Layers for style and content
        self.content_layers = ['conv4_2']
        self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    
    def get_features (self, image):
        """Extract features from image."""
        features = {}
        x = image
        
        layer_names = {
            '0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1',
            '19': 'conv4_1', '21': 'conv4_2', '28': 'conv5_1'
        }
        
        for name, layer in self.vgg._modules.items():
            x = layer (x)
            if name in layer_names:
                features[layer_names[name]] = x
        
        return features
    
    def gram_matrix (self, tensor):
        """
        Compute Gram matrix for style representation.
        Measures correlations between feature maps.
        """
        batch, channels, height, width = tensor.size()
        
        # Reshape to (channels, height*width)
        features = tensor.view (batch * channels, height * width)
        
        # Compute Gram matrix: G = F * F^T
        gram = torch.mm (features, features.t())
        
        # Normalize
        return gram / (batch * channels * height * width)
    
    def style_loss (self, style_grams, generated_grams):
        """Compute style loss (difference in Gram matrices)."""
        loss = 0
        for layer in self.style_layers:
            loss += F.mse_loss (generated_grams[layer], style_grams[layer])
        return loss
    
    def content_loss (self, content_features, generated_features):
        """Compute content loss."""
        loss = 0
        for layer in self.content_layers:
            loss += F.mse_loss (generated_features[layer], content_features[layer])
        return loss

print("Neural Style Transfer:")
print("=" * 60)
print("1. Extract features from content image (what)")
print("2. Extract features from style image (how)")
print("3. Start with content image (or random noise)")
print("4. Optimize pixel values to minimize:")
print("   - Content loss (preserve what)")
print("   - Style loss (match how)")
print("\\nResult: Content image painted in style of style image!")

print("\\nExample:")
print("Content: Photo of dog")
print("Style: Van Gogh\'s Starry Night")
print("Output: Dog painted in Van Gogh style ✨")
\`\`\`

## Visualization and Interpretability

### Understanding What CNNs Learn

Visualizing CNN internals helps understand and debug models.

\`\`\`python
# Visualize filters
def visualize_filters (model, layer_name, num_filters=16):
    """Visualize learned filters."""
    # Get specific layer
    layer = dict (model.named_modules())[layer_name]
    
    # Get weights
    weights = layer.weight.data.cpu()
    
    # Visualize
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()
    
    for i in range (min (num_filters, 16)):
        # Get filter
        filt = weights[i, 0]  # First channel
        
        # Normalize for visualization
        filt = (filt - filt.min()) / (filt.max() - filt.min())
        
        axes[i].imshow (filt, cmap='gray')
        axes[i].set_title (f'Filter {i}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Activation maximization
def visualize_activation (model, layer, filter_index, size=224, iterations=100):
    """
    Generate image that maximally activates a specific filter.
    Shows what the filter is 'looking for'.
    """
    # Random starting image
    img = torch.randn(1, 3, size, size, requires_grad=True)
    
    optimizer = optim.Adam([img], lr=0.1)
    
    for i in range (iterations):
        optimizer.zero_grad()
        
        # Forward pass
        features = model (img)
        
        # Loss: negative activation (we want to maximize)
        loss = -features[0, filter_index].mean()
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Optional: Add regularization (total variation)
        # to encourage spatial smoothness
    
    return img.detach()

print("Visualization Techniques:")
print("=" * 60)
print("1. Filter Visualization: Show learned kernel weights")
print("2. Activation Maps: Which parts of image activate filters")
print("3. Activation Maximization: Generate optimal input for filter")
print("4. Grad-CAM: Highlight regions important for classification")
print("5. t-SNE on features: Visualize learned representations")
\`\`\`

## Key Takeaways

1. **Data augmentation** is crucial for training robust CNNs with limited data

2. **Transfer learning** leverages pretrained models:
   - Freeze features + train classifier (small datasets)
   - Fine-tune top layers (medium datasets)
   - Fine-tune entire network (large datasets)

3. **Object detection** extends classification with localization:
   - YOLO: Fast, single-stage detector
   - R-CNN family: Slower but more accurate

4. **Semantic segmentation** provides pixel-level classification:
   - U-Net: Encoder-decoder with skip connections
   - Essential for medical imaging, autonomous driving

5. **Style transfer** demonstrates CNN features capture both content and style

6. **Visualization** helps understand and debug CNN behavior

## Coming Next

In the next section, we'll explore **Recurrent Neural Networks (RNNs)** - specialized architectures for sequential data like time series and text!
`,
};
