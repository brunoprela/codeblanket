/**
 * Section: Efficient Training Techniques
 * Module: Deep Learning Fundamentals
 *
 * Covers gradient accumulation, mixed precision training, gradient checkpointing,
 * distributed training, and modern techniques for training large models efficiently
 */

export const efficientTrainingTechniquesSection = {
  id: 'efficient-training-techniques',
  title: 'Efficient Training Techniques',
  content: `
# Efficient Training Techniques

## Introduction

Modern deep learning pushes the boundaries of scale. Training billion-parameter models requires advanced techniques beyond standard backpropagation.

**What You'll Learn:**
- Gradient accumulation (effective larger batches)
- Mixed precision training (FP16)
- Gradient checkpointing (memory efficiency)
- Data parallelism (multiple GPUs)
- Model parallelism (models too large for one GPU)
- Pipeline parallelism
- Distributed training frameworks

## Gradient Accumulation

**Problem**: Want large batch size (better gradients) but GPU memory is limited.

**Solution**: Accumulate gradients over multiple mini-batches before updating weights.

\`\`\`python
# PyTorch example
accumulation_steps = 4  # Effective batch size = batch_size * accumulation_steps

model.train()
optimizer.zero_grad()

for i, (data, target) in enumerate(train_loader):
    # Forward pass
    output = model(data)
    loss = criterion(output, target)
    
    # Normalize loss (important!)
    loss = loss / accumulation_steps
    
    # Backward pass (accumulate gradients)
    loss.backward()
    
    # Update weights every accumulation_steps
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# Don't forget final update if not divisible
if (i + 1) % accumulation_steps != 0:
    optimizer.step()
    optimizer.zero_grad()
\`\`\`

**Key Points**:
- Divide loss by \`accumulation_steps\` to get correct gradient scale
- Memory usage: same as small batch
- Effective batch size: \`batch_size * accumulation_steps\`
- No speed benefit, but enables large effective batches

## Mixed Precision Training

**Idea**: Use FP16 (half precision) instead of FP32 (full precision) for 2x speed, 2x memory savings.

**Challenges**:
- FP16 has smaller range (can overflow/underflow)
- Gradients often very small (vanish in FP16)

**Solution**: Automatic Mixed Precision (AMP)
- Forward/backward in FP16
- Master weights in FP32
- Loss scaling to prevent underflow

### PyTorch AMP

\`\`\`python
import torch
from torch.cuda.amp import autocast, GradScaler

# Create gradient scaler
scaler = GradScaler()

model.train()
for epoch in range(num_epochs):
    for data, target in train_loader:
        data, target = data.cuda(), target.cuda()
        
        optimizer.zero_grad()
        
        # Forward pass with autocast
        with autocast():
            output = model(data)
            loss = criterion(output, target)
        
        # Backward pass with scaling
        scaler.scale(loss).backward()
        
        # Update weights
        scaler.step(optimizer)
        scaler.update()

print("Mixed precision training complete!")
\`\`\`

**Benefits**:
- 2x faster training (on compatible GPUs: V100, A100, RTX 20xx+)
- 2x memory reduction
- Same accuracy (if done correctly)
- Minimal code changes

### TensorFlow Mixed Precision

\`\`\`python
from tensorflow import keras

# Enable mixed precision
policy = keras.mixed_precision.Policy('mixed_float16')
keras.mixed_precision.set_global_policy(policy)

# Build model (automatically uses FP16)
model = keras.Sequential([
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(10)
])

# Important: output layer should use FP32 for numerical stability
model.add(keras.layers.Activation('softmax', dtype='float32'))

# Compile with loss scaling
optimizer = keras.optimizers.Adam()
optimizer = keras.mixed_precision.LossScaleOptimizer(optimizer)

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')
model.fit(X_train, y_train, epochs=10)
\`\`\`

## Gradient Checkpointing

**Problem**: Deep models store all intermediate activations for backprop → huge memory.

**Solution**: Trade compute for memory - recompute activations during backward pass instead of storing them.

\`\`\`python
import torch.utils.checkpoint as checkpoint

class CheckpointedBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
    
    def forward(self, x):
        # Use checkpointing for this block
        return checkpoint.checkpoint(self.block, x)

# Example: ResNet with checkpointing
class CheckpointedResNet(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            CheckpointedBlock(ResidualBlock()) 
            for _ in range(num_layers)
        ])
        self.output = nn.Linear(512, 10)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output(x)

# Now can train much deeper models in same memory
model = CheckpointedResNet(num_layers=200)
\`\`\`

**Trade-off**:
- Memory: ~O(√n) instead of O(n) where n = number of layers
- Compute: ~1.5x slower (recompute during backward)
- Enables training 2-3x deeper models

## Data Parallelism

**Idea**: Replicate model on multiple GPUs, split batch across GPUs, average gradients.

### PyTorch DataParallel

\`\`\`python
# Simple but less efficient
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

model = model.cuda()
model.fit(train_loader)
\`\`\`

### PyTorch DistributedDataParallel (Recommended)

\`\`\`python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    """Initialize distributed training"""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def train_ddp(rank, world_size):
    setup(rank, world_size)
    
    # Create model and move to GPU
    model = MyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    # Create distributed sampler
    train_sampler = torch.utils.data.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=train_sampler
    )
    
    # Training loop
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)  # Shuffle differently each epoch
        
        for data, target in train_loader:
            data, target = data.to(rank), target.to(rank)
            
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

# Launch with torch.multiprocessing
import torch.multiprocessing as mp

world_size = torch.cuda.device_count()
mp.spawn(train_ddp, args=(world_size,), nprocs=world_size)
\`\`\`

**Benefits**:
- Linear speedup (2 GPUs → 2x faster, 4 GPUs → 4x faster)
- Each GPU has full model copy
- Efficient gradient synchronization
- Industry standard for multi-GPU training

## Model Parallelism

**When**: Model too large to fit on single GPU.

**Idea**: Split model across multiple GPUs (different layers on different GPUs).

\`\`\`python
class ModelParallelNet(nn.Module):
    def __init__(self):
        super().__init__()
        # First half on GPU 0
        self.layer1 = nn.Linear(1000, 1000).to('cuda:0')
        self.layer2 = nn.Linear(1000, 1000).to('cuda:0')
        
        # Second half on GPU 1
        self.layer3 = nn.Linear(1000, 1000).to('cuda:1')
        self.layer4 = nn.Linear(1000, 10).to('cuda:1')
    
    def forward(self, x):
        # Move through GPUs
        x = x.to('cuda:0')
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        
        x = x.to('cuda:1')  # Transfer to GPU 1
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        return x

model = ModelParallelNet()
\`\`\`

**Drawback**: GPUs idle during forward pass (GPU 1 waits for GPU 0). Use **Pipeline Parallelism** to fix this.

## Pipeline Parallelism

**Idea**: Split batch into micro-batches, pipeline them through model partitions.

\`\`\`python
# Conceptual (use libraries like FairScale or DeepSpeed in practice)

# Split model into stages
stage1 = nn.Sequential(layer1, layer2).to('cuda:0')
stage2 = nn.Sequential(layer3, layer4).to('cuda:1')

# Split batch into micro-batches
micro_batch_size = 8
micro_batches = batch.split(micro_batch_size)

# Pipeline execution
outputs = []
for i, micro_batch in enumerate(micro_batches):
    # Stage 1 processes micro-batch i
    intermediate = stage1(micro_batch.to('cuda:0'))
    
    # Meanwhile, stage 2 processes micro-batch i-1
    if i > 0:
        output = stage2(intermediates[i-1].to('cuda:1'))
        outputs.append(output)
    
    intermediates.append(intermediate)

# Process final micro-batches
for intermediate in intermediates[-len(micro_batches)+1:]:
    output = stage2(intermediate.to('cuda:1'))
    outputs.append(output)
\`\`\`

**Benefit**: Better GPU utilization (both GPUs busy most of the time).

## Modern Frameworks

### DeepSpeed (Microsoft)

\`\`\`python
import deepspeed

# Configure DeepSpeed
ds_config = {
    "train_batch_size": 64,
    "gradient_accumulation_steps": 4,
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 2  # Optimizer state partitioning
    }
}

# Initialize
model_engine, optimizer, train_loader, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    training_data=train_dataset,
    config=ds_config
)

# Training loop
for data, target in train_loader:
    outputs = model_engine(data)
    loss = criterion(outputs, target)
    model_engine.backward(loss)
    model_engine.step()
\`\`\`

**DeepSpeed ZeRO** (Zero Redundancy Optimizer):
- Stage 1: Partition optimizer states (4x memory reduction)
- Stage 2: Partition gradients (8x memory reduction)  
- Stage 3: Partition model parameters (N_gpu × memory reduction)
- Enables training 100B+ parameter models

### Hugging Face Accelerate

\`\`\`python
from accelerate import Accelerator

# Handles device placement, mixed precision, distributed training
accelerator = Accelerator(mixed_precision="fp16")

# Prepare model, optimizer, dataloader
model, optimizer, train_loader = accelerator.prepare(
    model, optimizer, train_loader
)

# Training loop (same code for 1 GPU or 8 GPUs!)
for data, target in train_loader:
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, target)
    accelerator.backward(loss)  # Handles scaling
    optimizer.step()
\`\`\`

**Accelerate benefits**:
- Write once, run anywhere (single GPU, multi-GPU, TPU)
- Automatic device management
- Unified API for all backends

## Practical Recommendations

### Training on Single GPU
1. Use mixed precision (autocast + GradScaler)
2. Gradient accumulation if need larger batch
3. Gradient checkpointing if running out of memory
4. Profile to find bottlenecks

### Training on Multiple GPUs (Data Parallel)
1. Use DistributedDataParallel (not DataParallel)
2. Mixed precision essential for speed
3. Scale learning rate linearly with # GPUs
4. Use NCCL backend for GPU communication

### Training Very Large Models
1. Use DeepSpeed ZeRO for memory efficiency
2. Combine with gradient checkpointing
3. Pipeline parallelism for extreme cases
4. Consider model architecture (MoE, sparse models)

## Performance Optimization Checklist

\`\`\`python
# 1. Use mixed precision
with torch.cuda.amp.autocast():
    output = model(data)

# 2. Efficient data loading
train_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,      # Parallel data loading
    pin_memory=True,    # Faster GPU transfer
    persistent_workers=True  # Keep workers alive
)

# 3. Gradient accumulation for larger effective batch
for i, batch in enumerate(train_loader):
    loss = loss / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 4. Compile model (PyTorch 2.0+)
model = torch.compile(model)  # 2x faster!

# 5. Profile to find bottlenecks
with torch.profiler.profile() as prof:
    for _ in range(10):
        output = model(data)
        loss.backward()
print(prof.key_averages().table())
\`\`\`

## Key Takeaways

1. **Mixed precision** - 2x speed/memory with minimal code change
2. **Gradient accumulation** - larger effective batch without memory increase
3. **Gradient checkpointing** - trade 1.5x compute for 2-3x memory
4. **Data parallelism** - linear speedup with multiple GPUs
5. **Model parallelism** - for models too large for one GPU
6. **Modern frameworks** - DeepSpeed, Accelerate simplify scaling
7. **Profile first** - measure before optimizing

## Conclusion

You've now mastered deep learning fundamentals:
- **Theory**: Forward prop, backprop, optimization
- **Architecture**: Layers, activations, regularization
- **Training**: Initialization, LR scheduling, monitoring
- **Frameworks**: PyTorch, TensorFlow/Keras
- **Best practices**: Debugging, tuning, production
- **Efficiency**: Scaling to large models and datasets

**Next**: Apply these to specialized architectures - CNNs, RNNs, Transformers!
`,
};
