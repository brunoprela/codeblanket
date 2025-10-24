export const llmTrainingProcess = {
  title: 'LLM Training Process',
  id: 'llm-training-process',
  content: `
# LLM Training Process

## Introduction

Training large language models requires massive compute, careful data curation, distributed systems, and sophisticated optimization techniques. From data preparation through evaluation, every step affects final model quality. This section covers the complete training pipeline: data collection and filtering, tokenization, distributed training, mixed precision, optimization strategies, and monitoring.

### Training at Scale

**Compute Requirements**: Millions of GPU-hours for large models
**Data Scale**: Trillions of tokens from diverse sources
**Distributed Training**: Across hundreds of GPUs/TPUs
**Optimization**: Mix of techniques for efficiency and stability
**Monitoring**: Track dozens of metrics in real-time

---

## Data Collection and Curation

### Building Training Datasets

\`\`\`python
"""
Data collection for LLM training
"""

class TrainingDataPipeline:
    """
    Complete pipeline for curating training data
    """
    
    def __init__(self):
        self.sources = []
        self.filters = []
        self.stats = {}
    
    # Step 1: Data sources
    def collect_data_sources(self):
        """
        Common data sources for LLM training
        """
        sources = {
            "web_crawl": {
                "source": "Common Crawl",
                "size": "~250TB (raw HTML)",
                "quality": "Mixed",
                "cost": "Free"
            },
            "books": {
                "source": "Books3, BookCorpus",
                "size": "~100GB",
                "quality": "High",
                "cost": "Legal issues"
            },
            "wikipedia": {
                "source": "Wikipedia dumps",
                "size": "~20GB (English)",
                "quality": "High",
                "cost": "Free"
            },
            "github": {
                "source": "Public repositories",
                "size": "~1TB",
                "quality": "Variable",
                "cost": "Free (with attribution)"
            },
            "academic": {
                "source": "ArXiv, PubMed",
                "size": "~100GB",
                "quality": "Very high",
                "cost": "Free"
            },
            "news": {
                "source": "News archives",
                "size": "~50GB",
                "quality": "Good",
                "cost": "Varies"
            }
        }
        
        return sources
    
    # Step 2: Data filtering
    def filter_low_quality(self, text):
        """
        Remove low-quality content
        
        Filters:
        1. Language detection (keep target language)
        2. Remove adult content
        3. Remove duplicate content
        4. Filter by quality signals
        """
        
        # 1. Language detection
        if not self.is_target_language(text):
            return False
        
        # 2. Adult content filter
        if self.contains_adult_content(text):
            return False
        
        # 3. Quality heuristics
        if not self.passes_quality_checks(text):
            return False
        
        # 4. Deduplication
        if self.is_duplicate(text):
            return False
        
        return True
    
    def is_target_language(self, text, target='en'):
        """
        Detect language
        """
        from langdetect import detect
        
        try:
            lang = detect(text)
            return lang == target
        except:
            return False
    
    def contains_adult_content(self, text):
        """
        Filter adult/harmful content
        """
        # Use blocklist or ML classifier
        adult_keywords = set([...])  # Blocklist
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in adult_keywords)
    
    def passes_quality_checks(self, text):
        """
        Quality heuristics
        """
        lines = text.split('\\n')
        
        # 1. Minimum length
        if len(text) < 100:
            return False
        
        # 2. Maximum line length (filter tables, logs)
        avg_line_length = sum(len(line) for line in lines) / len(lines)
        if avg_line_length > 200:  # Likely not prose
            return False
        
        # 3. Ratio of alphabetic characters
        alpha_ratio = sum(c.isalpha() for c in text) / len(text)
        if alpha_ratio < 0.6:  # Too many numbers/symbols
            return False
        
        # 4. Word count
        words = text.split()
        if len(words) < 20:
            return False
        
        # 5. Unique word ratio (filter repetitive text)
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:  # Too repetitive
            return False
        
        # 6. Stop word ratio (natural language check)
        stop_words = {'the', 'a', 'an', 'in', 'on', 'at', ...}
        stop_word_count = sum(1 for w in words if w.lower() in stop_words)
        stop_word_ratio = stop_word_count / len(words)
        if stop_word_ratio < 0.1:  # Not natural language
            return False
        
        return True
    
    def is_duplicate(self, text):
        """
        Deduplication using MinHash LSH
        """
        from datasketch import MinHash, MinHashLSH
        
        # Create MinHash
        m = MinHash(num_perm=128)
        for word in text.split():
            m.update(word.encode('utf-8'))
        
        # Check against LSH index
        result = self.lsh.query(m)
        
        if len(result) > 0:
            return True  # Duplicate found
        
        # Add to index
        self.lsh.insert(hash(text), m)
        return False

# Example: Process Common Crawl
import warc
from bs4 import BeautifulSoup

class CommonCrawlProcessor:
    """
    Extract text from Common Crawl WARC files
    """
    
    def __init__(self):
        self.pipeline = TrainingDataPipeline()
    
    def process_warc_file(self, warc_path):
        """
        Extract clean text from WARC
        """
        clean_texts = []
        
        with open(warc_path, 'rb') as f:
            for record in warc.WARCFile(fileobj=f):
                if record.type == 'response':
                    # Extract HTML
                    html = record.payload.read()
                    
                    # Parse with BeautifulSoup
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Remove scripts, styles
                    for script in soup(["script", "style"]):
                        script.decompose()
                    
                    # Get text
                    text = soup.get_text()
                    
                    # Clean whitespace
                    text = ' '.join(text.split())
                    
                    # Filter
                    if self.pipeline.filter_low_quality(text):
                        clean_texts.append(text)
        
        return clean_texts

# Data statistics
def analyze_dataset(texts):
    """
    Compute statistics on training data
    """
    import numpy as np
    
    stats = {
        "total_documents": len(texts),
        "total_tokens": sum(len(text.split()) for text in texts),
        "avg_length": np.mean([len(text.split()) for text in texts]),
        "median_length": np.median([len(text.split()) for text in texts]),
        "min_length": min(len(text.split()) for text in texts),
        "max_length": max(len(text.split()) for text in texts),
    }
    
    return stats

# Example dataset sizes
datasets = {
    "GPT-3": {
        "tokens": "300B",
        "sources": "Common Crawl (60%), WebText2 (22%), Books (16%), Wikipedia (3%)",
        "filtering": "Extensive quality filtering"
    },
    "LLaMA": {
        "tokens": "1.4T",
        "sources": "Common Crawl (67%), C4 (15%), GitHub (4.5%), Wikipedia (4.5%), Books (4.5%), ArXiv (2.5%), StackExchange (2%)",
        "filtering": "CCNet pipeline"
    },
    "GPT-4": {
        "tokens": "Unknown (likely >1T)",
        "sources": "Proprietary + licensed data",
        "filtering": "Advanced quality and safety filtering"
    }
}
\`\`\`

---

## Tokenization

### Subword Tokenization

\`\`\`python
"""
Tokenization for LLMs
"""

# BPE (Byte Pair Encoding) - Used by GPT models
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

class BPETokenizer:
    """
    Byte Pair Encoding tokenizer
    
    Algorithm:
    1. Start with character vocabulary
    2. Find most frequent adjacent pair
    3. Merge that pair into new token
    4. Repeat until desired vocab size
    """
    
    def train(self, texts, vocab_size=50000):
        """
        Train BPE tokenizer
        """
        # Initialize
        tokenizer = Tokenizer(models.BPE())
        
        # Pre-tokenization (split on whitespace/punctuation)
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        
        # Trainer
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<s>", "</s>", "<unk>", "<pad>", "<mask>"]
        )
        
        # Train
        tokenizer.train_from_iterator(texts, trainer)
        
        return tokenizer
    
    def example_bpe_merging(self):
        """
        How BPE learns merges
        """
        # Start: Character level
        # Text: "low low low lowest"
        # Tokens: ['l', 'o', 'w', ' ', 'l', 'o', 'w', ...]
        
        # Most frequent pair: 'l' + 'o' → Merge to 'lo'
        # Tokens: ['lo', 'w', ' ', 'lo', 'w', ...]
        
        # Next most frequent: 'lo' + 'w' → Merge to 'low'
        # Tokens: ['low', ' ', 'low', ' ', 'low', ' ', 'lo', 'w', 'e', 's', 't']
        
        # Continue until vocab_size reached
        pass

# Using GPT-2 tokenizer
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Tokenize text
text = "Hello, how are you?"
tokens = tokenizer.encode(text)
print(tokens)  # [15496, 11, 703, 389, 345, 30]

# Decode
decoded = tokenizer.decode(tokens)
print(decoded)  # "Hello, how are you?"

# Token analysis
text = "The cat sat on the mat"
tokens = tokenizer.tokenize(text)
print(tokens)  # ['The', 'Ġcat', 'Ġsat', 'Ġon', 'Ġthe', 'Ġmat']
# Note: Ġ represents space

# Subword splitting
text = "unbelievable"
tokens = tokenizer.tokenize(text)
print(tokens)  # ['un', 'believ', 'able']

# Advantages of BPE:
# 1. No unknown tokens (can represent any text)
# 2. Efficient: common words = 1 token, rare words = multiple tokens
# 3. Works across languages
# 4. Handles morphology (prefixes, suffixes)

# Token counting (important for API costs)
def count_tokens(text, tokenizer):
    """
    Count tokens in text
    """
    tokens = tokenizer.encode(text)
    return len(tokens)

# Example
text = "This is a sentence."
n_tokens = count_tokens(text, tokenizer)
print(f"Tokens: {n_tokens}")  # 5 tokens

# Rule of thumb: ~1 token per 4 characters in English
def estimate_tokens(text):
    """
    Quick estimation without tokenizer
    """
    return len(text) // 4

# SentencePiece (used by LLaMA, BERT)
import sentencepiece as spm

class SentencePieceTokenizer:
    """
    Language-agnostic tokenization
    """
    
    def train(self, input_file, vocab_size=32000):
        """
        Train SentencePiece model
        """
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix='tokenizer',
            vocab_size=vocab_size,
            character_coverage=0.9995,  # Cover 99.95% of characters
            model_type='bpe'  # Or 'unigram'
        )
    
    def load_and_use(self):
        """
        Load and use trained tokenizer
        """
        sp = spm.SentencePieceProcessor()
        sp.load('tokenizer.model')
        
        # Encode
        tokens = sp.encode_as_pieces("Hello world")
        print(tokens)  # ['▁Hello', '▁world']
        
        # Note: ▁ represents space (different from GPT-2's Ġ)
        
        ids = sp.encode_as_ids("Hello world")
        print(ids)  # [1234, 5678]
        
        # Decode
        text = sp.decode_ids(ids)
        print(text)  # "Hello world"

# Tokenizer vocabulary size impact
vocab_sizes = {
    "GPT-2": 50257,
    "GPT-3": 50257,  # Same as GPT-2
    "LLaMA": 32000,
    "Claude": 100000,  # Estimated
}

# Larger vocab:
# + Fewer tokens per text (more efficient)
# + Better handling of rare words
# - Larger embedding matrix
# - More parameters to train
\`\`\`

---

## Distributed Training

### Data and Model Parallelism

\`\`\`python
"""
Training LLMs across multiple GPUs
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# 1. Data Parallelism
class DataParallelTraining:
    """
    Split data across GPUs, replicate model
    
    Each GPU:
    - Has full model copy
    - Processes different batch
    - Computes gradients
    - Syncs gradients across GPUs
    - Updates model
    """
    
    def setup(self, rank, world_size):
        """
        Initialize distributed training
        """
        dist.init_process_group(
            backend='nccl',  # NVIDIA GPUs
            init_method='tcp://localhost:23456',
            rank=rank,
            world_size=world_size
        )
    
    def train(self, rank, world_size, model, dataset):
        """
        Train with DDP
        """
        # Setup
        self.setup(rank, world_size)
        
        # Move model to GPU
        model = model.to(rank)
        
        # Wrap with DDP
        model = DDP(model, device_ids=[rank])
        
        # Distributed sampler (splits data across GPUs)
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        
        # DataLoader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            sampler=sampler
        )
        
        # Training loop
        for epoch in range(num_epochs):
            sampler.set_epoch(epoch)  # Shuffle differently each epoch
            
            for batch in dataloader:
                inputs = batch['input_ids'].to(rank)
                labels = batch['labels'].to(rank)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # DDP automatically syncs gradients across GPUs here
                
                # Update
                optimizer.step()
                optimizer.zero_grad()
        
        # Cleanup
        dist.destroy_process_group()

# Launch distributed training
import torch.multiprocessing as mp

def main():
    world_size = 8  # 8 GPUs
    mp.spawn(
        DataParallelTraining().train,
        args=(world_size, model, dataset),
        nprocs=world_size,
        join=True
    )

# 2. Model Parallelism
class ModelParallelTraining:
    """
    Split model across GPUs (for models too large for single GPU)
    
    Example: 100B parameter model
    - GPU 0: Layers 0-24
    - GPU 1: Layers 25-49
    - GPU 2: Layers 50-74
    - GPU 3: Layers 75-99
    """
    
    def split_model_across_gpus(self, model, num_gpus=4):
        """
        Split layers across GPUs
        """
        layers_per_gpu = len(model.layers) // num_gpus
        
        for i, layer in enumerate(model.layers):
            gpu_id = i // layers_per_gpu
            layer.to(f'cuda:{gpu_id}')
        
        return model
    
    def forward_with_model_parallel(self, model, x):
        """
        Forward pass moves between GPUs
        """
        # Start on GPU 0
        x = x.to('cuda:0')
        
        for i, layer in enumerate(model.layers):
            # Move to correct GPU
            gpu_id = i // layers_per_gpu
            x = x.to(f'cuda:{gpu_id}')
            
            # Forward through layer
            x = layer(x)
        
        return x

# 3. Pipeline Parallelism
class PipelineParallelism:
    """
    Combine model parallelism with micro-batching
    
    Split batch into micro-batches, pipeline through GPUs
    Reduces idle time (GPUs don't wait for each other)
    """
    
    def forward_with_pipeline(self, model, batch, num_microbatches=4):
        """
        Pipeline execution
        """
        microbatch_size = len(batch) // num_microbatches
        microbatches = [
            batch[i:i+microbatch_size]
            for i in range(0, len(batch), microbatch_size)
        ]
        
        # Pipeline: As soon as GPU 0 finishes microbatch 1,
        # it starts microbatch 2 while GPU 1 processes microbatch 1
        
        outputs = []
        for mb in microbatches:
            out = self.forward_microbatch(model, mb)
            outputs.append(out)
        
        return torch.cat(outputs)

# 4. ZeRO (Zero Redundancy Optimizer) - DeepSpeed
from deepspeed import DeepSpeedConfig, initialize

class ZeROTraining:
    """
    DeepSpeed ZeRO: Memory-efficient distributed training
    
    ZeRO stages:
    - Stage 1: Partition optimizer states across GPUs (4x memory reduction)
    - Stage 2: + Partition gradients (8x memory reduction)
    - Stage 3: + Partition parameters (linear memory reduction)
    """
    
    def setup_deepspeed(self, model):
        """
        Setup DeepSpeed with ZeRO-3
        """
        ds_config = {
            "train_batch_size": 256,
            "train_micro_batch_size_per_gpu": 32,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": 3e-4,
                    "betas": [0.9, 0.95],
                    "eps": 1e-8,
                    "weight_decay": 0.1
                }
            },
            "fp16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 3,  # ZeRO-3: Partition everything
                "offload_optimizer": {
                    "device": "cpu"  # Offload to CPU RAM
                },
                "offload_param": {
                    "device": "cpu"
                }
            }
        }
        
        model_engine, optimizer, _, _ = initialize(
            model=model,
            config=ds_config
        )
        
        return model_engine, optimizer

# Training GPT-3 (175B parameters)
training_config = {
    "gpus": 1024,  # 1024 A100 GPUs (80GB each)
    "strategy": "3D parallelism",
    "data_parallel": 64,  # 64-way data parallelism
    "tensor_parallel": 8,  # 8-way tensor parallelism
    "pipeline_parallel": 2,  # 2-way pipeline parallelism
    "batch_size": 4096,  # Global batch size
    "microbatch_size": 1,  # Per GPU
    "time": "~34 days",
    "cost": "~$4.6M"
}
\`\`\`

---

## Mixed Precision Training

### FP16 and BF16

\`\`\`python
"""
Train faster with reduced precision
"""

import torch
from torch.cuda.amp import autocast, GradScaler

class MixedPrecisionTraining:
    """
    Use FP16 for speed, FP32 for stability
    
    Benefits:
    - 2x faster training
    - 2x less memory
    - 2x larger batch sizes
    """
    
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.scaler = GradScaler()  # Prevents underflow
    
    def training_step(self, batch):
        """
        Mixed precision training step
        """
        inputs = batch['input_ids']
        labels = batch['labels']
        
        # Forward pass in FP16
        with autocast():
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
        
        # Backward pass: Scale loss to prevent underflow
        self.scaler.scale(loss).backward()
        
        # Unscale gradients before clipping
        self.scaler.unscale_(self.optimizer)
        
        # Gradient clipping (in FP32)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update weights
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        self.optimizer.zero_grad()
        
        return loss.item()

# BFloat16 (BF16) - Better for LLMs
class BFloat16Training:
    """
    BF16: Better dynamic range than FP16
    
    FP16:  1 sign bit + 5 exponent + 10 mantissa
    BF16:  1 sign bit + 8 exponent + 7 mantissa
    
    BF16 advantages:
    - Same range as FP32 (no overflow/underflow issues)
    - No need for loss scaling
    - Slightly less precise, but fine for LLMs
    """
    
    def training_step(self, batch):
        """
        BF16 training (simpler than FP16)
        """
        inputs = batch['input_ids']
        labels = batch['labels']
        
        # Convert to BF16
        with autocast(dtype=torch.bfloat16):
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
        
        # No scaling needed!
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # Update
        optimizer.step()
        optimizer.zero_grad()
        
        return loss.item()

# Precision comparison
precision_comparison = {
    "FP32": {
        "speed": "1x (baseline)",
        "memory": "1x",
        "stability": "Best",
        "use": "Small models, debugging"
    },
    "FP16": {
        "speed": "2-3x",
        "memory": "0.5x",
        "stability": "Needs loss scaling",
        "use": "Inference, some training"
    },
    "BF16": {
        "speed": "2-3x",
        "memory": "0.5x",
        "stability": "Good (no scaling needed)",
        "use": "LLM training (preferred)"
    },
    "INT8": {
        "speed": "4x",
        "memory": "0.25x",
        "stability": "Requires quantization-aware training",
        "use": "Inference only"
    }
}
\`\`\`

---

## Optimization and Learning Rate Schedules

### Training Stability

\`\`\`python
"""
Optimization techniques for LLM training
"""

import torch
import torch.optim as optim
import math

# 1. AdamW (Adam with decoupled weight decay)
class AdamWOptimizer:
    """
    AdamW: Standard for LLM training
    
    Key hyperparameters:
    - lr: 1e-4 to 6e-4 (depends on model size)
    - betas: (0.9, 0.95) or (0.9, 0.999)
    - eps: 1e-8
    - weight_decay: 0.1
    """
    
    def __init__(self, model, lr=3e-4):
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.1
        )

# 2. Learning rate schedule
class LearningRateScheduler:
    """
    Warmup + Cosine decay
    """
    
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
    
    def get_lr(self, step):
        """
        Learning rate schedule
        
        1. Linear warmup: 0 → max_lr (first 2-10% of training)
        2. Cosine decay: max_lr → min_lr (rest of training)
        """
        if step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (step / self.warmup_steps)
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        return lr
    
    def step(self, step):
        """
        Update learning rate
        """
        lr = self.get_lr(step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

# Visualize LR schedule
import matplotlib.pyplot as plt

def plot_lr_schedule(warmup_steps=2000, total_steps=100000, base_lr=3e-4):
    scheduler = LearningRateScheduler(
        optimizer=None,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr=base_lr * 0.1
    )
    
    steps = range(total_steps)
    lrs = [scheduler.get_lr(step) for step in steps]
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, lrs)
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule (Warmup + Cosine Decay)')
    plt.grid(True)
    plt.show()

# 3. Gradient clipping
def clip_gradients(model, max_norm=1.0):
    """
    Prevent exploding gradients
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

# 4. Gradient accumulation
class GradientAccumulation:
    """
    Simulate larger batch sizes with limited memory
    
    Example: Want batch_size=256 but can only fit 32
    → Accumulate gradients over 8 steps (256/32=8)
    """
    
    def train_with_accumulation(self, model, dataloader, accumulation_steps=8):
        optimizer.zero_grad()
        
        for i, batch in enumerate(dataloader):
            # Forward pass
            outputs = model(batch['input_ids'])
            loss = criterion(outputs, batch['labels'])
            
            # Scale loss
            loss = loss / accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update every accumulation_steps
            if (i + 1) % accumulation_steps == 0:
                # Clip gradients
                clip_gradients(model)
                
                # Update
                optimizer.step()
                optimizer.zero_grad()

# Training hyperparameters for different model sizes
hyperparameters = {
    "125M": {
        "batch_size": 512,
        "lr": 6e-4,
        "warmup_steps": 2000,
        "weight_decay": 0.1,
        "gradient_clip": 1.0
    },
    "1.3B": {
        "batch_size": 1024,
        "lr": 3e-4,
        "warmup_steps": 5000,
        "weight_decay": 0.1,
        "gradient_clip": 1.0
    },
    "13B": {
        "batch_size": 2048,
        "lr": 1.5e-4,
        "warmup_steps": 10000,
        "weight_decay": 0.1,
        "gradient_clip": 1.0
    },
    "70B": {
        "batch_size": 4096,
        "lr": 1e-4,
        "warmup_steps": 20000,
        "weight_decay": 0.1,
        "gradient_clip": 1.0
    }
}
\`\`\`

---

## Training Monitoring

### Metrics and Checkpointing

\`\`\`python
"""
Monitor training and save checkpoints
"""

import wandb
from pathlib import Path

class TrainingMonitor:
    """
    Track metrics and save checkpoints
    """
    
    def __init__(self, project_name, run_name):
        # Initialize Weights & Biases
        wandb.init(project=project_name, name=run_name)
        
        self.step = 0
        self.best_val_loss = float('inf')
    
    def log_metrics(self, metrics):
        """
        Log to wandb
        """
        wandb.log(metrics, step=self.step)
        self.step += 1
    
    def should_save_checkpoint(self, val_loss):
        """
        Save if validation loss improved
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            return True
        return False
    
    def save_checkpoint(self, model, optimizer, scheduler, path):
        """
        Save training state
        """
        checkpoint = {
            'step': self.step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': model.config
        }
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, model, optimizer, scheduler, path):
        """
        Resume training
        """
        checkpoint = torch.load(path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.step = checkpoint['step']
        self.best_val_loss = checkpoint['best_val_loss']

# Complete training loop
def train_llm(
    model,
    train_dataloader,
    val_dataloader,
    num_steps=100000,
    checkpoint_every=5000,
    eval_every=1000
):
    """
    Full training loop with monitoring
    """
    # Setup
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = LearningRateScheduler(optimizer, warmup_steps=2000, total_steps=num_steps)
    scaler = GradScaler()
    monitor = TrainingMonitor(project_name="llm-training", run_name="gpt-125m")
    
    model.train()
    global_step = 0
    
    while global_step < num_steps:
        for batch in train_dataloader:
            # Training step
            inputs = batch['input_ids'].cuda()
            labels = batch['labels'].cuda()
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            scheduler.step(global_step)
            
            # Log metrics
            if global_step % 10 == 0:
                monitor.log_metrics({
                    'train/loss': loss.item(),
                    'train/perplexity': math.exp(loss.item()),
                    'train/lr': scheduler.get_lr(global_step),
                    'train/step': global_step
                })
            
            # Validation
            if global_step % eval_every == 0:
                val_loss = evaluate(model, val_dataloader)
                monitor.log_metrics({
                    'val/loss': val_loss,
                    'val/perplexity': math.exp(val_loss)
                })
                
                # Save checkpoint if improved
                if monitor.should_save_checkpoint(val_loss):
                    monitor.save_checkpoint(
                        model, optimizer, scheduler,
                        f'checkpoint_step_{global_step}.pt'
                    )
            
            # Regular checkpoint
            if global_step % checkpoint_every == 0:
                monitor.save_checkpoint(
                    model, optimizer, scheduler,
                    f'checkpoint_step_{global_step}.pt'
                )
            
            global_step += 1
            
            if global_step >= num_steps:
                break
    
    return model

def evaluate(model, dataloader):
    """
    Compute validation loss
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input_ids'].cuda()
            labels = batch['labels'].cuda()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0)
            total_tokens += inputs.size(0)
    
    model.train()
    return total_loss / total_tokens
\`\`\`

---

## Conclusion

Training LLMs requires:

1. **Data**: Trillions of tokens, carefully filtered and deduplicated
2. **Compute**: Hundreds of GPUs, distributed training strategies
3. **Optimization**: Mixed precision, gradient clipping, careful LR schedules
4. **Monitoring**: Track metrics, save checkpoints, evaluate regularly

**Key Techniques**:
- Data parallelism for scaling across GPUs
- Model/pipeline parallelism for large models
- ZeRO for memory efficiency
- BF16 for speed without instability
- AdamW + cosine LR schedule
- Gradient clipping and accumulation

**Practical Takeaways**:
- Start with quality data filtering
- Use BPE/SentencePiece tokenization
- Train with BF16 on modern GPUs
- Monitor training closely
- Save checkpoints frequently
- Test on validation set regularly

Training from scratch is expensive ($1M+ for large models). Most practitioners fine-tune existing models or use APIs.
`,
};
