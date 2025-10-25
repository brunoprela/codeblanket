import { QuizQuestion } from '../../../types';

export const efficientTrainingTechniquesQuiz: QuizQuestion[] = [
  {
    id: 'efficient-training-q1',
    question:
      'Explain gradient accumulation: what problem does it solve, how does it work, and what is the key implementation detail to get correct? Compare memory and compute costs.',
    sampleAnswer: `Gradient accumulation is a technique to simulate larger batch sizes when GPU memory is limited:

**Problem It Solves:**
Want large batch size (e.g., 512) but GPU only fits 64
- Large batches: more stable gradients, better convergence
- Memory constraint: can't fit large batch

**How It Works:**
Instead of updating after each batch, accumulate gradients over multiple batches:

\`\`\`python
accumulation_steps = 8  # Effective batch = 64 * 8 = 512

optimizer.zero_grad()
for i, (data, target) in enumerate (train_loader):
    output = model (data)
    loss = criterion (output, target)
    
    # KEY: Divide by accumulation steps!
    loss = loss / accumulation_steps
    
    loss.backward()  # Accumulate gradients
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()  # Update weights
        optimizer.zero_grad()  # Reset
\`\`\`

**Critical Detail: Loss Scaling**
MUST divide loss by accumulation_steps to maintain correct gradient scale!

Without division:
- Gradients 8x too large
- Effective learning rate 8x higher
- Training unstable/diverges

**Memory Cost:**
- Same as small batch (64)
- Only one sub-batch in memory at a time
- Gradients accumulate (small overhead)

**Compute Cost:**
- Same total FLOPs as large batch
- Slightly slower (Python overhead)
- No speed benefit, only memory efficiency

**When to Use:**
- GPU memory limited
- Want large batch benefits
- Can afford slightly slower training

**Key Insight:** Gradient accumulation enables large effective batch sizes without additional memory, at the cost of slightly slower training.`,
    keyPoints: [
      'Solves: want large batch, GPU memory too small',
      'Accumulate gradients over multiple mini-batches',
      'CRITICAL: divide loss by accumulation_steps',
      'Memory: same as small batch',
      'Compute: same total, but slightly slower',
      'No speed benefit, only memory efficiency',
      'Effective batch = batch_size × accumulation_steps',
    ],
  },
  {
    id: 'efficient-training-q2',
    question:
      'Describe mixed precision training (FP16): what are the benefits and challenges? How does loss scaling address the challenges?',
    sampleAnswer: `Mixed precision training uses 16-bit floating point (FP16) instead of 32-bit (FP32) for significant speed and memory benefits:

**Benefits of FP16:**
- 2x faster computation (Tensor Cores)
- 2x memory savings
- Can train larger models or batches
- Minimal accuracy loss

**Challenges:**
1. **Smaller Range:** FP16 range: ~6e-8 to 6e4 (vs FP32: ~1e-38 to 1e38)
2. **Underflow:** Small gradients vanish to zero
3. **Overflow:** Large values become infinity

**Loss Scaling Solution:**

Multiply loss by large factor before backward:
\`\`\`python
# PyTorch AMP
scaler = GradScaler()

optimizer.zero_grad()
with autocast():  # FP16 forward pass
    output = model (data)
    loss = criterion (output, target)

scaler.scale (loss).backward()  # Scale up before backward
scaler.step (optimizer)  # Scale down and update
scaler.update()  # Adjust scale factor
\`\`\`

**How Loss Scaling Works:**
1. Multiply loss by scale factor (e.g., 1024)
2. Gradients also multiplied by 1024
3. Prevents underflow (small grads don't vanish)
4. Scale gradients back down before weight update
5. Dynamic scaling adjusts factor automatically

**Why It Works:**
- Small gradient: 1e-7 × 1024 = 1e-4 (representable in FP16)
- Without scaling: 1e-7 → underflow to 0
- With scaling: preserved through backward pass

**Master Weights:**
- Maintain FP32 copy of weights
- Compute in FP16 (fast)
- Update in FP32 (accurate)

**Requirements:**
- Modern GPU (Volta+, RTX 20xx+)
- PyTorch/TensorFlow AMP support
- Minimal code changes

**Typical Results:**
- 1.5-2x speedup
- 2x memory reduction
- <0.1% accuracy difference

**Key Insight:** Loss scaling enables FP16 training by preventing gradient underflow, achieving 2x speed/memory with negligible accuracy loss.`,
    keyPoints: [
      'FP16: 2x speed, 2x memory savings',
      'Challenge: smaller range causes underflow/overflow',
      'Loss scaling: multiply loss by large factor before backward',
      'Prevents small gradients from vanishing to zero',
      'Scale gradients back down before weight update',
      'Master weights in FP32 for precision',
      'Requires modern GPU (Tensor Cores)',
      'Minimal code changes with automatic mixed precision',
    ],
  },
  {
    id: 'efficient-training-q3',
    question:
      'Compare data parallelism, model parallelism, and pipeline parallelism. What problems does each solve, and when would you use each approach?',
    sampleAnswer: `Different parallelism strategies address different bottlenecks in distributed training:

**Data Parallelism:**
Replicate model on each GPU, split batch across GPUs:
\`\`\`python
# Each GPU has full model
GPU 0: batch[0:32]
GPU 1: batch[32:64]
GPU 2: batch[64:96]
GPU 3: batch[96:128]
# Synchronize gradients, update all models
\`\`\`

**Pros:**
- Simple implementation
- Linear speedup (4 GPUs = 4x faster)
- Most common approach

**Cons:**
- Model must fit on single GPU
- Gradient communication overhead
- Limited by model size

**When to Use:**
- Model fits on one GPU
- Have multiple GPUs
- Want training speedup
- **Most common scenario**

**Model Parallelism:**
Split model layers across GPUs:
\`\`\`python
GPU 0: layers 1-25
GPU 1: layers 26-50
GPU 2: layers 51-75
GPU 3: layers 76-100
\`\`\`

**Pros:**
- Can train models too large for one GPU
- No gradient synchronization

**Cons:**
- GPUs idle during sequential execution
- GPU 1 waits for GPU 0
- Poor utilization

**When to Use:**
- Model doesn't fit on single GPU
- Model too large even with FP16

**Pipeline Parallelism:**
Split model AND batch into micro-batches:
\`\`\`python
Time 1: GPU0(micro-batch1)
Time 2: GPU0(mb2), GPU1(mb1)
Time 3: GPU0(mb3), GPU1(mb2), GPU2(mb1)
Time 4: GPU0(mb4), GPU1(mb3), GPU2(mb2), GPU3(mb1)
\`\`\`

**Pros:**
- Better GPU utilization than model parallelism
- Can train very large models
- Pipelined execution reduces idle time

**Cons:**
- Complex implementation
- Some idle time remains
- Requires micro-batch tuning

**When to Use:**
- Very large models (billions of parameters)
- Model parallelism too inefficient
- Have framework support (DeepSpeed, FairScale)

**Modern Practice:**
- **Data parallel**: 95% of use cases
- **Model parallel**: Models > 20GB
- **Pipeline parallel**: Models > 100GB (GPT-3 scale)
- **3D parallelism**: Combine all three for extreme scale

**Key Decision:**
- Model fits on GPU → Data Parallelism
- Model doesn't fit → Model/Pipeline Parallelism

**DeepSpeed ZeRO:** Modern alternative that partitions optimizer states, gradients, and parameters across GPUs, enabling much larger models with data parallelism approach.`,
    keyPoints: [
      'Data parallel: replicate model, split batch, linear speedup',
      'Model parallel: split model across GPUs, sequential execution',
      'Pipeline parallel: model split + micro-batches, better utilization',
      'Data parallel: most common (95% of cases)',
      "Model parallel: when model doesn't fit single GPU",
      'Pipeline parallel: very large models with better efficiency',
      'DeepSpeed ZeRO: modern approach partitioning optimizer/gradients',
      'Choose based on model size relative to GPU memory',
    ],
  },
];
