/**
 * Transfer Learning Content
 */

export const transferLearningSection = {
  id: 'transfer-learning',
  title: 'Transfer Learning',
  content: `
# Transfer Learning

## Introduction

**Transfer Learning** is the practice of using knowledge learned from one task to improve performance on a related task. In deep learning, this typically means:

1. **Pre-training**: Train a model on a large dataset (e.g., ImageNet, Wikipedia)
2. **Fine-tuning**: Adapt the pre-trained model to your specific task with smaller dataset

**Why Transfer Learning?**

- **Data efficiency**: Achieve high performance with limited labeled data (hundreds vs. millions of examples)
- **Training speed**: Fine-tuning is much faster than training from scratch
- **Better performance**: Pre-trained features often outperform task-specific training
- **Accessibility**: Use state-of-the-art models without massive compute

**Analogy**: Like learning to play guitar after learning piano - the musical knowledge transfers, even though the instruments differ.

---

## Core Concepts

### 1. Feature Extraction vs. Fine-Tuning

**Feature Extraction** (Frozen Backbone):
- Keep pre-trained weights **frozen** (no gradient updates)
- Only train new layers on top
- Fast, requires little data
- Use when: Target task very similar to pre-training task, very small dataset

**Fine-Tuning** (Updating Pre-trained Weights):
- **Unfreeze** some or all pre-trained layers
- Continue training with lower learning rate
- Slower, needs more data, but better performance
- Use when: Moderate dataset size, task somewhat different from pre-training

**Full Training** (From Scratch):
- Random initialization, train all weights
- Very slow, needs massive dataset
- Use when: Very large dataset, task very different from pre-training

---

### 2. The Transfer Learning Pipeline

**Step 1: Choose Pre-trained Model**
- Match architecture to task (CNN for vision, Transformer for NLP)
- Consider model size vs. dataset size
- Check pre-training dataset relevance

**Step 2: Modify Architecture**
- Replace task-specific head (classifier, decoder)
- Add layers if needed for your task
- Ensure input/output dimensions match

**Step 3: Freeze Appropriate Layers**
- Early layers: General features (edges, textures, words)
- Later layers: Task-specific features
- Strategy depends on dataset size and task similarity

**Step 4: Train with Appropriate Settings**
- **Learning rate**: Lower than from-scratch training (e.g., 1e-5 vs. 1e-3)
- **Epochs**: Fewer needed (10-50 vs. 100-300)
- **Batch size**: Can often use larger batches
- **Regularization**: Less dropout (model already regularized by pre-training)

---

## Transfer Learning in Computer Vision

### Pre-trained Models

**ImageNet Pre-training**: Most CV models pre-trained on ImageNet (1.4M images, 1000 classes)

**Popular architectures**:
1. **ResNet** (50, 101, 152 layers): Residual connections, 25-60M parameters
2. **EfficientNet** (B0-B7): Optimized accuracy/efficiency, 5-66M parameters
3. **Vision Transformer (ViT)**: Attention-based, 86-632M parameters
4. **DenseNet**: Dense connections, 8-33M parameters

### Implementation: Image Classification

\`\`\`python
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Load pre-trained model
model = models.resnet50(pretrained=True)
print(f"Original classifier: {model.fc}")
# Original: Linear (in_features=2048, out_features=1000)  # ImageNet classes

# 2. Replace classifier for your task (e.g., 10 classes)
num_classes = 10
model.fc = nn.Linear (model.fc.in_features, num_classes)
print(f"New classifier: {model.fc}")

# 3. Strategy A: Feature Extraction (freeze backbone)
for param in model.parameters():
    param.requires_grad = False  # Freeze all layers

# Unfreeze only the new classifier
for param in model.fc.parameters():
    param.requires_grad = True

# Count trainable parameters
trainable_params = sum (p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum (p.numel() for p in model.parameters())
print(f"Trainable: {trainable_params:,} / {total_params:,} "
      f"({100*trainable_params/total_params:.1f}%)")
# Only ~20K trainable (new classifier) out of 25M total

# 4. Prepare data
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),  # ResNet expects 224×224
    transforms.ToTensor(),
    transforms.Normalize (mean=[0.485, 0.456, 0.406],  # ImageNet statistics
                        std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder('path/to/train', transform=transform)
train_loader = DataLoader (train_dataset, batch_size=32, shuffle=True)

# 5. Training setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam (model.fc.parameters(), lr=1e-3)
# Only optimize the classifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to (device)

# 6. Training loop
num_epochs = 10
model.train()

for epoch in range (num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to (device), labels.to (device)
        
        # Forward pass
        outputs = model (images)
        loss = criterion (outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq (labels).sum().item()
    
    epoch_loss = running_loss / len (train_loader)
    epoch_acc = 100. * correct / total
    print(f"Epoch {epoch+1}/{num_epochs}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.2f}%")

print("Feature extraction training complete!")
\`\`\`

### Fine-Tuning Strategy

\`\`\`python
# Strategy B: Gradual Unfreezing
model = models.resnet50(pretrained=True)
model.fc = nn.Linear (model.fc.in_features, num_classes)

# Phase 1: Train only classifier (5 epochs)
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam (model.fc.parameters(), lr=1e-3)
# ... train for 5 epochs ...

# Phase 2: Unfreeze last residual block (5 epochs)
for param in model.layer4.parameters():  # Last block
    param.requires_grad = True

optimizer = torch.optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-4},  # Lower LR for pre-trained
    {'params': model.fc.parameters(), 'lr': 1e-3}       # Higher LR for new layers
], lr=1e-4)
# ... train for 5 epochs ...

# Phase 3: Unfreeze all (fine-tune entire network)
for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam([
    {'params': model.layer1.parameters(), 'lr': 1e-5},  # Very low LR for early layers
    {'params': model.layer2.parameters(), 'lr': 1e-5},
    {'params': model.layer3.parameters(), 'lr': 3e-5},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3}
], lr=1e-4)
# ... train for 10 epochs ...

print("Gradual fine-tuning complete!")
\`\`\`

### Discriminative Learning Rates

Different learning rates for different layers (lower for early layers):

\`\`\`python
def get_discriminative_lr_groups (model, base_lr=1e-3, multiplier=0.5):
    """
    Create parameter groups with exponentially decreasing learning rates.
    Early layers get lower LR, later layers get higher LR.
    
    Args:
        model: ResNet model
        base_lr: Learning rate for last layer
        multiplier: LR multiplier for each earlier layer
    
    Returns:
        List of parameter groups for optimizer
    """
    # Identify layer groups (early to late)
    layer_groups = [
        model.conv1,
        model.bn1,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
        model.fc
    ]
    
    # Calculate LR for each group (exponentially increasing)
    num_groups = len (layer_groups)
    param_groups = []
    
    for i, layer in enumerate (layer_groups):
        # Earlier layers get smaller LR
        lr_factor = multiplier ** (num_groups - i - 1)
        lr = base_lr * lr_factor
        
        param_groups.append({
            'params': layer.parameters(),
            'lr': lr
        })
        
        print(f"Layer {i} ({layer.__class__.__name__}): LR = {lr:.2e}")
    
    return param_groups


# Example usage
model = models.resnet50(pretrained=True)
model.fc = nn.Linear (model.fc.in_features, 10)

param_groups = get_discriminative_lr_groups (model, base_lr=1e-3, multiplier=0.5)
optimizer = torch.optim.Adam (param_groups)

# Output:
# Layer 0 (Conv2d): LR = 1.56e-05
# Layer 1 (BatchNorm2d): LR = 3.12e-05
# Layer 2 (Sequential): LR = 6.25e-05
# Layer 3 (Sequential): LR = 1.25e-04
# Layer 4 (Sequential): LR = 2.50e-04
# Layer 5 (Sequential): LR = 5.00e-04
# Layer 6 (Linear): LR = 1.00e-03
\`\`\`

---

## Transfer Learning in NLP

### Pre-trained Language Models

**Evolution of pre-training**:
1. **Word embeddings**: Word2Vec, GloVe (2013-2014)
2. **Contextual embeddings**: ELMo (2018)
3. **Transformer pre-training**: BERT, GPT (2018-2019)
4. **Large language models**: GPT-3, T5, PaLM (2020+)

**Popular models**:
- **BERT**: Bidirectional encoder, good for understanding (classification, QA)
- **GPT**: Unidirectional decoder, good for generation (text completion, dialogue)
- **T5**: Encoder-decoder, treats all tasks as text-to-text
- **RoBERTa**: Improved BERT training
- **DistilBERT**: Smaller, faster BERT (40% fewer parameters, 60% faster)

### Implementation: Text Classification with BERT

\`\`\`python
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

# 1. Load pre-trained BERT
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained (model_name)
model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2  # Binary classification
)

print(f"Model: {model_name}")
print(f"Parameters: {sum (p.numel() for p in model.parameters()):,}")
# ~110M parameters

# 2. Prepare dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len (self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor (label, dtype=torch.long)
        }


# Example data
texts = [
    "This movie was fantastic!",
    "Terrible film, waste of time.",
    "Best movie I've seen this year!",
    # ... more examples ...
]
labels = [1, 0, 1, ...]  # 1=positive, 0=negative

train_dataset = TextDataset (texts, labels, tokenizer)

# 3. Training configuration
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,  # Much lower than from-scratch
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# 4. Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    # eval_dataset=val_dataset,  # Add validation set
)

# 5. Fine-tune
trainer.train()

# 6. Inference
def predict (text):
    inputs = tokenizer (text, return_tensors='pt', truncation=True, max_length=128)
    outputs = model(**inputs)
    probs = torch.softmax (outputs.logits, dim=1)
    predicted_class = torch.argmax (probs, dim=1).item()
    confidence = probs[0, predicted_class].item()
    
    return predicted_class, confidence

# Test
text = "This product exceeded my expectations!"
label, conf = predict (text)
print(f"Text: {text}")
print(f"Prediction: {'Positive' if label == 1 else 'Negative'} ({conf:.2%} confidence)")
\`\`\`

### Adapter Layers (Parameter-Efficient Fine-Tuning)

Instead of fine-tuning all parameters, add small **adapter** layers:

\`\`\`python
import torch.nn as nn

class AdapterLayer (nn.Module):
    def __init__(self, hidden_size, adapter_size=64):
        """
        Adapter: down-project → non-linearity → up-project + residual
        
        Args:
            hidden_size: Original model dimension (e.g., 768 for BERT-base)
            adapter_size: Bottleneck dimension (much smaller, e.g., 64)
        """
        super().__init__()
        self.down_project = nn.Linear (hidden_size, adapter_size)
        self.up_project = nn.Linear (adapter_size, hidden_size)
        self.activation = nn.ReLU()
        
        # Initialize to near-identity
        nn.init.normal_(self.down_project.weight, std=1e-3)
        nn.init.normal_(self.up_project.weight, std=1e-3)
        nn.init.zeros_(self.down_project.bias)
        nn.init.zeros_(self.up_project.bias)
    
    def forward (self, x):
        # Bottleneck transformation
        residual = x
        x = self.down_project (x)
        x = self.activation (x)
        x = self.up_project (x)
        
        # Residual connection
        return x + residual


# Add adapters to BERT
def add_adapters_to_bert (model, adapter_size=64):
    """Insert adapter layers after each attention and FFN"""
    hidden_size = model.config.hidden_size
    
    for layer in model.bert.encoder.layer:
        # After attention
        original_attention_output = layer.attention.output
        layer.attention.output = nn.Sequential(
            original_attention_output,
            AdapterLayer (hidden_size, adapter_size)
        )
        
        # After feed-forward
        original_ffn_output = layer.output
        layer.output = nn.Sequential(
            original_ffn_output,
            AdapterLayer (hidden_size, adapter_size)
        )
    
    return model


# Freeze BERT, train only adapters
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model = add_adapters_to_bert (model, adapter_size=64)

# Freeze all parameters except adapters
for name, param in model.named_parameters():
    if 'adapter' not in name.lower():
        param.requires_grad = False

trainable = sum (p.numel() for p in model.parameters() if p.requires_grad)
total = sum (p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
# Only ~1-3% of parameters trainable!

# Train as usual
# ... training code ...
\`\`\`

**Advantages of Adapters**:
- **Efficiency**: Train only 1-3% of parameters
- **Speed**: Faster training, less memory
- **Modularity**: Can swap adapters for different tasks
- **Performance**: Often comparable to full fine-tuning

---

## When to Use Transfer Learning

### Decision Matrix

| Your Data Size | Task Similarity | Strategy | Learning Rate |
|----------------|----------------|----------|---------------|
| Very small (<100) | High | Feature extraction only | 1e-3 |
| Small (100-1K) | High | Feature extraction | 1e-3 |
| Small (100-1K) | Medium | Fine-tune last layers | 1e-4 |
| Small (100-1K) | Low | Fine-tune all, high augmentation | 1e-5 |
| Medium (1K-100K) | High | Fine-tune last layers | 1e-4 |
| Medium (1K-100K) | Medium | Fine-tune all | 1e-5 |
| Medium (1K-100K) | Low | Fine-tune all + augmentation | 1e-5 |
| Large (>100K) | Any | Fine-tune all or from scratch | 1e-4 to 1e-3 |

**Task similarity examples**:
- **High**: ImageNet → dog breed classification (both object recognition)
- **Medium**: ImageNet → medical image classification (objects vs. anatomy)
- **Low**: ImageNet → satellite image segmentation (natural vs. aerial, classification vs. segmentation)

---

## Domain Adaptation

**Challenge**: Pre-trained model (source domain) applied to different domain (target domain)

**Example**: Model trained on natural images → medical X-rays

**Domain shift problems**:
- Different visual appearance
- Different statistical properties
- Different class distributions

**Solutions**:

### 1. Gradual Unfreezing

Start with frozen layers, gradually unfreeze from top to bottom.

### 2. Data Augmentation

Apply augmentations matching target domain:

\`\`\`python
# Medical imaging augmentations
transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine (degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter (brightness=0.2, contrast=0.2),
    transforms.RandomHorizontalFlip(),
    # Medical-specific: simulate different scanner settings
    transforms.GaussianBlur (kernel_size=3),
    transforms.ToTensor(),
    transforms.Normalize (mean=[0.485], std=[0.229])  # Grayscale
])
\`\`\`

### 3. Domain Adversarial Training

Train model to be **invariant** to domain:

\`\`\`python
class DomainAdversarialNetwork (nn.Module):
    def __init__(self, feature_extractor, task_classifier, domain_classifier):
        """
        Architecture:
        - Feature extractor: Shared by both classifiers
        - Task classifier: Predicts task labels (e.g., disease present/absent)
        - Domain classifier: Predicts domain (e.g., source/target dataset)
        
        Training: Maximize task accuracy, minimize domain classification accuracy
        → Forces features to be domain-invariant
        """
        super().__init__()
        self.feature_extractor = feature_extractor
        self.task_classifier = task_classifier
        self.domain_classifier = domain_classifier
        self.gradient_reversal = GradientReversalLayer()
    
    def forward (self, x):
        features = self.feature_extractor (x)
        
        # Task prediction (normal forward)
        task_output = self.task_classifier (features)
        
        # Domain prediction (reverse gradients)
        domain_features = self.gradient_reversal (features)
        domain_output = self.domain_classifier (domain_features)
        
        return task_output, domain_output


class GradientReversalLayer (torch.autograd.Function):
    """Reverses gradients during backprop"""
    @staticmethod
    def forward (ctx, x, lambda_=1.0):
        ctx.lambda_ = lambda_
        return x.view_as (x)
    
    @staticmethod
    def backward (ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


# Training: Minimize task loss, maximize domain loss (confuse domain classifier)
task_loss = criterion (task_output, task_labels)
domain_loss = criterion (domain_output, domain_labels)

total_loss = task_loss - 0.1 * domain_loss  # Negative sign for adversarial
total_loss.backward()
\`\`\`

---

## Real-World Example: Medical Image Classification

\`\`\`python
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np

class MedicalImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len (self.image_paths)
    
    def __getitem__(self, idx):
        # Load image (grayscale X-ray)
        image = Image.open (self.image_paths[idx]).convert('L')  # Grayscale
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform (image)
        
        return image, label


# Data augmentation for medical images
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip (p=0.5),
    transforms.ColorJitter (brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    # Convert grayscale to 3-channel (for ImageNet pre-trained models)
    transforms.Lambda (lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize (mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Lambda (lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize (mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = MedicalImageDataset (train_paths, train_labels, train_transform)
val_dataset = MedicalImageDataset (val_paths, val_labels, val_transform)

train_loader = DataLoader (train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader (val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Load pre-trained model
model = models.densenet121(pretrained=True)
num_features = model.classifier.in_features
model.classifier = nn.Linear (num_features, 2)  # Binary: disease present/absent

# Transfer to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to (device)

# Training setup with discriminative learning rates
optimizer = torch.optim.Adam([
    {'params': model.features.parameters(), 'lr': 1e-5},  # Pre-trained: low LR
    {'params': model.classifier.parameters(), 'lr': 1e-3}  # New layer: high LR
])

criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3, verbose=True
)

# Training loop with validation
best_val_acc = 0.0
num_epochs = 30

for epoch in range (num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for images, labels in train_loader:
        images, labels = images.to (device), labels.to (device)
        
        optimizer.zero_grad()
        outputs = model (images)
        loss = criterion (outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq (labels).sum().item()
    
    train_acc = 100. * train_correct / train_total
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to (device), labels.to (device)
            outputs = model (images)
            loss = criterion (outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq (labels).sum().item()
    
    val_acc = 100. * val_correct / val_total
    
    # Logging
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss/len (train_loader):.4f}, Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss/len (val_loader):.4f}, Acc: {val_acc:.2f}%")
    
    # Learning rate scheduling
    scheduler.step (val_acc)
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save (model.state_dict(), 'best_medical_model.pth')
        print(f"Saved new best model with val accuracy: {val_acc:.2f}%")
    
    print("-" * 60)

print(f"Training complete! Best validation accuracy: {best_val_acc:.2f}%")
\`\`\`

**Results** (typical):
- From scratch: 75-80% accuracy (with 1000 images)
- Transfer learning: 85-92% accuracy (with same 1000 images)
- **10-15% improvement** from pre-training!

---

## Discussion Questions

1. **Why does transfer learning work better with fewer examples than training from scratch?**
   - Consider what the pre-trained model has already learned

2. **When fine-tuning, why should earlier layers have lower learning rates than later layers?**
   - Think about what different layers represent

3. **A medical imaging model trained on chest X-rays performs poorly on brain MRIs. Why might transfer learning fail here?**
   - Consider domain similarity and feature relevance

4. **How does the size of your target dataset affect the choice between feature extraction and fine-tuning?**
   - Think about overfitting risk and training stability

5. **Adapters achieve 95% of full fine-tuning performance with only 2% of trainable parameters. Why doesn't everyone use adapters?**
   - Consider trade-offs in performance, flexibility, and engineering complexity

---

## Key Takeaways

- **Transfer learning** leverages pre-trained models to achieve high performance with limited data
- **Feature extraction** (frozen backbone) for very small datasets, **fine-tuning** (unfrozen layers) for moderate datasets
- **Discriminative learning rates**: Lower LR for early layers (general features), higher for late layers (task-specific)
- **Gradual unfreezing**: Train classifier first, then gradually unfreeze deeper layers
- **Pre-trained models**: ImageNet for vision (ResNet, EfficientNet, ViT), Wikipedia/Books for NLP (BERT, GPT, T5)
- **Domain adaptation** addresses distribution shift between pre-training and target domains
- **Adapter layers**: Parameter-efficient alternative to full fine-tuning (1-3% parameters)
- **Typical improvement**: 10-20% accuracy gain vs. from-scratch training with limited data
- **Best practices**: Match task similarity, use appropriate learning rates, validate carefully
- **Modern paradigm**: Pre-train on massive data, fine-tune for specific tasks (democratizes deep learning)

---

## Practical Tips

1. **Always start with pre-trained model** unless you have >1M labeled examples

2. **Check pre-training data**: Closer to your task = better transfer

3. **Freeze strategically**: Small data → freeze more, large data → freeze less

4. **Use lower learning rates**: 10-100× smaller than from-scratch training

5. **Monitor both train and val**: Overfitting happens faster with fine-tuning

6. **Try gradual unfreezing**: Often better than unfreezing all at once

7. **Augment heavily**: Helps model adapt to new domain

8. **Consider adapters**: Especially if training multiple tasks or limited compute

---

## Further Reading

- ["How transferable are features in deep neural networks?"](https://arxiv.org/abs/1411.1792) - Yosinski et al., 2014
- ["Universal Language Model Fine-tuning for Text Classification"](https://arxiv.org/abs/1801.06146) - Howard & Ruder, 2018
- ["Parameter-Efficient Transfer Learning for NLP"](https://arxiv.org/abs/1902.00751) - Houlsby et al., 2019 (Adapters)
- [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [Hugging Face Fine-tuning Guide](https://huggingface.co/docs/transformers/training)

---

*Next Section: Autoencoders - Learn unsupervised learning with neural networks for dimensionality reduction, denoising, and generative modeling!*
`,
};
