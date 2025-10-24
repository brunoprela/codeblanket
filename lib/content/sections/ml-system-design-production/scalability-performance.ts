export const scalabilityPerformance = {
  title: 'Scalability & Performance',
  id: 'scalability-performance',
  content: `
# Scalability & Performance

## Introduction

**"Premature optimization is the root of all evil."** - Donald Knuth  
**"But late optimization is just as bad."** - Production Engineers

Building ML systems that scale requires careful optimization at every level—from model inference to data pipelines. A model that works great in development can fail spectacularly in production if it can't handle real traffic.

**Performance Challenges**:
- High latency kills user experience
- Low throughput limits scale
- Memory leaks crash systems
- CPU/GPU bottlenecks increase costs
- Network I/O slows everything down

This section covers optimizing ML systems for production scale, from model-level optimizations to infrastructure scaling.

### Performance Hierarchy

\`\`\`
Application Level → Model Level → Infrastructure Level
       ↓                 ↓                  ↓
   Caching          Quantization      Horizontal Scaling
   Batching         Pruning           Load Balancing
   Async            Distillation      Auto-scaling
\`\`\`

By the end of this section, you'll understand:
- Model optimization techniques
- Caching strategies
- Horizontal and vertical scaling
- Load balancing and auto-scaling
- Performance profiling and debugging

---

## Model Optimization

### Model Quantization

\`\`\`python
"""
Model Quantization for Faster Inference
"""

import torch
import torch.nn as nn
import time
import numpy as np

class SimpleModel(nn.Module):
    """Example model for quantization"""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(100, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 10)
        )
    
    def forward(self, x):
        return self.layers(x)


def benchmark_model(model, input_data, n_runs=1000):
    """
    Benchmark model inference speed
    """
    # Warm-up
    for _ in range(10):
        _ = model(input_data)
    
    # Benchmark
    start = time.time()
    for _ in range(n_runs):
        with torch.no_grad():
            _ = model(input_data)
    duration = time.time() - start
    
    latency_ms = (duration / n_runs) * 1000
    throughput = n_runs / duration
    
    return {
        'latency_ms': latency_ms,
        'throughput': throughput,
        'total_time': duration
    }


# Create model
model = SimpleModel()
model.eval()

# Test input
input_data = torch.randn(1, 100)

# Benchmark original model
print("=== Model Quantization Comparison ===\\n")
print("Original Model (FP32):")
fp32_metrics = benchmark_model(model, input_data)
print(f"  Latency: {fp32_metrics['latency_ms']:.3f}ms")
print(f"  Throughput: {fp32_metrics['throughput']:.0f} inferences/sec")

# Get model size
fp32_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
print(f"  Model size: {fp32_size:.2f} MB")

# Dynamic Quantization (INT8)
quantized_dynamic = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear},  # Quantize Linear layers
    dtype=torch.qint8
)

print("\\nDynamic Quantization (INT8):")
int8_metrics = benchmark_model(quantized_dynamic, input_data)
print(f"  Latency: {int8_metrics['latency_ms']:.3f}ms ({fp32_metrics['latency_ms']/int8_metrics['latency_ms']:.1f}x faster)")
print(f"  Throughput: {int8_metrics['throughput']:.0f} inferences/sec")

int8_size = sum(p.numel() * p.element_size() for p in quantized_dynamic.parameters()) / 1024 / 1024
print(f"  Model size: {int8_size:.2f} MB ({fp32_size/int8_size:.1f}x smaller)")

print("\\n✓ Quantization benefits:")
print(f"  - {fp32_metrics['latency_ms']/int8_metrics['latency_ms']:.1f}x faster inference")
print(f"  - {fp32_size/int8_size:.1f}x smaller model size")
print(f"  - Lower memory bandwidth requirements")
\`\`\`

### Model Pruning

\`\`\`python
"""
Model Pruning - Remove Unnecessary Weights
"""

import torch
import torch.nn.utils.prune as prune

def prune_model(model, amount=0.3):
    """
    Prune model by removing smallest weights
    
    Args:
        amount: Fraction of weights to prune (0.3 = 30%)
    """
    print(f"\\n=== Pruning Model ({amount*100}% of weights) ===\\n")
    
    # Count original parameters
    original_params = sum(p.numel() for p in model.parameters())
    
    # Prune each Linear layer
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')  # Make pruning permanent
    
    # Count remaining non-zero parameters
    remaining_params = sum((p != 0).sum().item() for p in model.parameters())
    
    print(f"Original parameters: {original_params:,}")
    print(f"Remaining parameters: {remaining_params:,}")
    print(f"Pruned: {original_params - remaining_params:,} ({(1-remaining_params/original_params)*100:.1f}%)")
    
    return model


# Create and prune model
pruned_model = SimpleModel()
pruned_model.eval()
pruned_model = prune_model(pruned_model, amount=0.5)  # Prune 50%

# Benchmark
pruned_metrics = benchmark_model(pruned_model, input_data)
print(f"\\nPruned Model Performance:")
print(f"  Latency: {pruned_metrics['latency_ms']:.3f}ms")
print(f"  Throughput: {pruned_metrics['throughput']:.0f} inferences/sec")
\`\`\`

### Knowledge Distillation

\`\`\`python
"""
Knowledge Distillation - Train Small Model from Large Model
"""

import torch.nn.functional as F
import torch.optim as optim

class TeacherModel(nn.Module):
    """Large, accurate model"""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(100, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 10)
        )
    
    def forward(self, x):
        return self.layers(x)


class StudentModel(nn.Module):
    """Small, fast model"""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 10)
        )
    
    def forward(self, x):
        return self.layers(x)


def distillation_loss(student_logits, teacher_logits, labels, temperature=3.0, alpha=0.7):
    """
    Combined loss for knowledge distillation
    
    Args:
        temperature: Softens probability distributions
        alpha: Weight for distillation loss (1-alpha for hard labels)
    """
    # Soft targets from teacher
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction='batchmean'
    ) * (temperature ** 2)
    
    # Hard targets (true labels)
    hard_loss = F.cross_entropy(student_logits, labels)
    
    # Combined loss
    return alpha * soft_loss + (1 - alpha) * hard_loss


def train_with_distillation(teacher, student, train_loader, epochs=5):
    """
    Train student model with knowledge distillation
    """
    teacher.eval()  # Teacher in eval mode
    student.train()
    
    optimizer = optim.Adam(student.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            # Get teacher predictions (no gradients needed)
            with torch.no_grad():
                teacher_logits = teacher(batch_X)
            
            # Student predictions
            student_logits = student(batch_X)
            
            # Distillation loss
            loss = distillation_loss(
                student_logits,
                teacher_logits,
                batch_y,
                temperature=3.0,
                alpha=0.7
            )
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return student


# Example setup
teacher = TeacherModel()
student = StudentModel()

teacher_params = sum(p.numel() for p in teacher.parameters())
student_params = sum(p.numel() for p in student.parameters())

print(f"\\n=== Knowledge Distillation ===")
print(f"Teacher parameters: {teacher_params:,}")
print(f"Student parameters: {student_params:,}")
print(f"Compression: {teacher_params/student_params:.1f}x smaller")

# In practice: train_with_distillation(teacher, student, train_loader)
\`\`\`

---

## Caching Strategies

### Multi-Level Caching

\`\`\`python
"""
Multi-Level Caching for ML Predictions
"""

import redis
from functools import lru_cache
import pickle
import hashlib
from typing import Any, Optional
import time

class PredictionCache:
    """
    Multi-level caching system
    
    Levels:
    1. In-memory (LRU cache) - fastest
    2. Redis (shared cache) - fast
    3. Model prediction - slowest
    """
    
    def __init__(self, redis_client=None, ttl=3600):
        self.redis_client = redis_client
        self.ttl = ttl  # Time-to-live in seconds
        
        # Statistics
        self.stats = {
            'memory_hits': 0,
            'redis_hits': 0,
            'misses': 0,
            'total_requests': 0
        }
    
    def _generate_key(self, features: list) -> str:
        """Generate cache key from features"""
        features_str = str(sorted(features))
        return hashlib.md5(features_str.encode()).hexdigest()
    
    @lru_cache(maxsize=1000)
    def _memory_cache_get(self, key: str) -> Optional[float]:
        """In-memory cache (LRU)"""
        # LRU cache is handled by decorator
        return None
    
    def get_prediction(self, features: list, model) -> dict:
        """
        Get prediction with multi-level caching
        """
        self.stats['total_requests'] += 1
        
        key = self._generate_key(features)
        start = time.time()
        
        # Level 1: In-memory cache
        cached = self._memory_cache_get(key)
        if cached is not None:
            self.stats['memory_hits'] += 1
            return {
                'prediction': cached,
                'cache_level': 'memory',
                'latency_ms': (time.time() - start) * 1000
            }
        
        # Level 2: Redis cache
        if self.redis_client:
            try:
                cached = self.redis_client.get(key)
                if cached:
                    prediction = pickle.loads(cached)
                    self.stats['redis_hits'] += 1
                    
                    # Populate memory cache
                    self._memory_cache_get(key)  # This will cache it
                    
                    return {
                        'prediction': prediction,
                        'cache_level': 'redis',
                        'latency_ms': (time.time() - start) * 1000
                    }
            except Exception as e:
                print(f"Redis error: {e}")
        
        # Level 3: Model prediction (cache miss)
        self.stats['misses'] += 1
        
        import numpy as np
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)[0]
        
        # Cache in Redis
        if self.redis_client:
            try:
                self.redis_client.setex(
                    key,
                    self.ttl,
                    pickle.dumps(prediction)
                )
            except Exception as e:
                print(f"Redis cache write error: {e}")
        
        # Cache in memory (via LRU)
        self._memory_cache_get(key)
        
        return {
            'prediction': prediction,
            'cache_level': 'model',
            'latency_ms': (time.time() - start) * 1000
        }
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        total = self.stats['total_requests']
        if total == 0:
            return self.stats
        
        hit_rate = (self.stats['memory_hits'] + self.stats['redis_hits']) / total * 100
        
        return {
            **self.stats,
            'hit_rate': hit_rate,
            'miss_rate': 100 - hit_rate
        }


# Example usage (without actual Redis)
from sklearn.ensemble import RandomForestRegressor

# Train simple model
model = RandomForestRegressor(n_estimators=100)
X_train = np.random.randn(1000, 10)
y_train = np.random.randn(1000)
model.fit(X_train, y_train)

# Create cache (without Redis for demo)
cache = PredictionCache(redis_client=None)

# Simulate requests
print("\\n=== Caching Performance ===\\n")

# First request (cache miss)
result1 = cache.get_prediction([1.0] * 10, model)
print(f"Request 1: {result1['cache_level']} ({result1['latency_ms']:.2f}ms)")

# Second request (cache hit)
result2 = cache.get_prediction([1.0] * 10, model)
print(f"Request 2: {result2['cache_level']} ({result2['latency_ms']:.2f}ms)")

# Simulate 1000 requests with 80% repeated
for i in range(1000):
    # 80% of requests are from 20 unique feature sets
    if np.random.random() < 0.8:
        features = [float(np.random.randint(0, 20))] * 10
    else:
        features = [float(np.random.randn())] * 10
    
    cache.get_prediction(features, model)

# Statistics
stats = cache.get_stats()
print(f"\\n=== Cache Statistics ===")
print(f"Total requests: {stats['total_requests']:,}")
print(f"Memory hits: {stats['memory_hits']:,}")
print(f"Redis hits: {stats['redis_hits']:,}")
print(f"Misses: {stats['misses']:,}")
print(f"Hit rate: {stats['hit_rate']:.1f}%")
\`\`\`

### Feature Caching

\`\`\`python
"""
Cache Expensive Feature Computations
"""

import functools
from typing import Dict
import pandas as pd

class FeatureCache:
    """
    Cache expensive feature computations
    """
    
    def __init__(self):
        self.cache = {}
        self.compute_times = []
    
    def cached_feature(self, key_func):
        """
        Decorator for caching feature computations
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                key = key_func(*args, **kwargs)
                
                # Check cache
                if key in self.cache:
                    return self.cache[key]
                
                # Compute feature
                start = time.time()
                result = func(*args, **kwargs)
                compute_time = time.time() - start
                
                # Cache result
                self.cache[key] = result
                self.compute_times.append(compute_time)
                
                return result
            
            return wrapper
        return decorator


# Example: Cache technical indicators
feature_cache = FeatureCache()

@feature_cache.cached_feature(lambda symbol, date: f"{symbol}:{date}")
def compute_technical_indicators(symbol: str, date: str) -> Dict[str, float]:
    """
    Expensive feature computation
    """
    # Simulate expensive computation
    time.sleep(0.1)  # 100ms
    
    return {
        'sma_20': np.random.randn(),
        'rsi': np.random.uniform(0, 100),
        'macd': np.random.randn(),
        'bollinger_upper': np.random.randn(),
        'bollinger_lower': np.random.randn()
    }


print("\\n=== Feature Caching ===\\n")

# First call (cache miss)
start = time.time()
features = compute_technical_indicators('AAPL', '2024-01-15')
print(f"First call: {(time.time() - start)*1000:.1f}ms (cache miss)")

# Second call (cache hit)
start = time.time()
features = compute_technical_indicators('AAPL', '2024-01-15')
print(f"Second call: {(time.time() - start)*1000:.3f}ms (cache hit)")

print(f"\\nSpeedup: {0.1 / (time.time() - start):.0f}x faster")
\`\`\`

---

## Horizontal Scaling

### Load Balancing

\`\`\`python
"""
Load Balancer for ML Services
"""

from typing import List
import random
from collections import defaultdict

class LoadBalancer:
    """
    Load balancer for distributing requests
    """
    
    def __init__(self, servers: List[str], strategy='round_robin'):
        self.servers = servers
        self.strategy = strategy
        self.current_index = 0
        
        # Track server load
        self.request_counts = defaultdict(int)
        self.response_times = defaultdict(list)
    
    def get_server(self) -> str:
        """
        Select server based on strategy
        """
        if self.strategy == 'round_robin':
            return self._round_robin()
        elif self.strategy == 'least_connections':
            return self._least_connections()
        elif self.strategy == 'least_response_time':
            return self._least_response_time()
        elif self.strategy == 'random':
            return random.choice(self.servers)
        else:
            return self.servers[0]
    
    def _round_robin(self) -> str:
        """Round-robin selection"""
        server = self.servers[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.servers)
        return server
    
    def _least_connections(self) -> str:
        """Select server with fewest active connections"""
        return min(self.servers, key=lambda s: self.request_counts[s])
    
    def _least_response_time(self) -> str:
        """Select server with lowest average response time"""
        def avg_response_time(server):
            times = self.response_times[server]
            return np.mean(times) if times else 0
        
        return min(self.servers, key=avg_response_time)
    
    def record_request(self, server: str, response_time_ms: float):
        """Record request completion"""
        self.request_counts[server] += 1
        self.response_times[server].append(response_time_ms)
        
        # Keep only recent 100 response times
        if len(self.response_times[server]) > 100:
            self.response_times[server] = self.response_times[server][-100:]
    
    def get_stats(self) -> dict:
        """Get load balancer statistics"""
        stats = {}
        
        for server in self.servers:
            times = self.response_times[server]
            
            stats[server] = {
                'requests': self.request_counts[server],
                'avg_response_ms': np.mean(times) if times else 0,
                'p95_response_ms': np.percentile(times, 95) if times else 0
            }
        
        return stats


# Example usage
lb = LoadBalancer(
    servers=['server1', 'server2', 'server3'],
    strategy='round_robin'
)

print("=== Load Balancing ===\\n")

# Simulate 1000 requests
for i in range(1000):
    server = lb.get_server()
    
    # Simulate variable response times
    response_time = np.random.uniform(10, 50)
    
    lb.record_request(server, response_time)

# Statistics
stats = lb.get_stats()
print("Server Statistics:")
for server, stat in stats.items():
    print(f"\\n{server}:")
    print(f"  Requests: {stat['requests']}")
    print(f"  Avg response: {stat['avg_response_ms']:.2f}ms")
    print(f"  P95 response: {stat['p95_response_ms']:.2f}ms")

# Compare strategies
print("\\n=== Strategy Comparison ===")

strategies = ['round_robin', 'least_connections', 'random']

for strategy in strategies:
    lb_test = LoadBalancer(servers=['s1', 's2', 's3'], strategy=strategy)
    
    # Simulate uneven server performance
    server_latencies = {'s1': 20, 's2': 40, 's3': 60}  # ms
    
    for _ in range(1000):
        server = lb_test.get_server()
        latency = server_latencies[server] + np.random.uniform(-5, 5)
        lb_test.record_request(server, latency)
    
    stats = lb_test.get_stats()
    
    avg_latency = np.mean([s['avg_response_ms'] for s in stats.values()])
    
    print(f"{strategy}: avg latency = {avg_latency:.2f}ms")
\`\`\`

### Auto-Scaling

\`\`\`python
"""
Auto-Scaling Based on Load
"""

from datetime import datetime, timedelta

class AutoScaler:
    """
    Auto-scale ML service based on metrics
    """
    
    def __init__(
        self,
        min_instances=2,
        max_instances=10,
        target_cpu_percent=70,
        target_latency_ms=100
    ):
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.target_cpu = target_cpu_percent
        self.target_latency = target_latency_ms
        
        self.current_instances = min_instances
        self.scale_history = []
    
    def should_scale_up(self, cpu_percent: float, latency_ms: float) -> bool:
        """
        Determine if should scale up
        """
        return (
            cpu_percent > self.target_cpu or
            latency_ms > self.target_latency
        ) and self.current_instances < self.max_instances
    
    def should_scale_down(self, cpu_percent: float, latency_ms: float) -> bool:
        """
        Determine if should scale down
        """
        return (
            cpu_percent < self.target_cpu * 0.5 and
            latency_ms < self.target_latency * 0.5
        ) and self.current_instances > self.min_instances
    
    def scale(self, cpu_percent: float, latency_ms: float, requests_per_sec: float):
        """
        Make scaling decision
        """
        old_instances = self.current_instances
        
        if self.should_scale_up(cpu_percent, latency_ms):
            # Scale up by 50%
            new_instances = min(
                int(self.current_instances * 1.5),
                self.max_instances
            )
            self.current_instances = new_instances
            
            action = 'SCALE_UP'
        
        elif self.should_scale_down(cpu_percent, latency_ms):
            # Scale down by 25%
            new_instances = max(
                int(self.current_instances * 0.75),
                self.min_instances
            )
            self.current_instances = new_instances
            
            action = 'SCALE_DOWN'
        
        else:
            action = 'NO_CHANGE'
        
        # Record decision
        if action != 'NO_CHANGE':
            self.scale_history.append({
                'timestamp': datetime.now(),
                'action': action,
                'old_instances': old_instances,
                'new_instances': self.current_instances,
                'cpu_percent': cpu_percent,
                'latency_ms': latency_ms,
                'requests_per_sec': requests_per_sec
            })
            
            print(f"[{action}] {old_instances} → {self.current_instances} instances")
            print(f"  CPU: {cpu_percent:.1f}%, Latency: {latency_ms:.1f}ms, RPS: {requests_per_sec:.0f}")
        
        return action


# Simulate auto-scaling
print("\\n=== Auto-Scaling Simulation ===\\n")

scaler = AutoScaler(
    min_instances=2,
    max_instances=10,
    target_cpu_percent=70,
    target_latency_ms=100
)

# Simulate load patterns
print("Simulating traffic spike...")

for minute in range(30):
    # Traffic spike between minute 5-15
    if 5 <= minute <= 15:
        cpu = np.random.uniform(75, 95)
        latency = np.random.uniform(110, 150)
        rps = np.random.uniform(800, 1200)
    else:
        cpu = np.random.uniform(30, 60)
        latency = np.random.uniform(40, 80)
        rps = np.random.uniform(200, 400)
    
    scaler.scale(cpu, latency, rps)

print(f"\\nFinal instances: {scaler.current_instances}")
print(f"Scaling events: {len(scaler.scale_history)}")
\`\`\`

---

## Performance Profiling

### Python Profiling

\`\`\`python
"""
Profile ML Code Performance
"""

import cProfile
import pstats
from functools import wraps
import time

def profile(func):
    """
    Decorator to profile function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        
        result = func(*args, **kwargs)
        
        profiler.disable()
        
        # Print stats
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        print(f"\\n=== Profile for {func.__name__} ===")
        stats.print_stats(10)  # Top 10 functions
        
        return result
    
    return wrapper


@profile
def expensive_prediction_pipeline(n_samples=1000):
    """
    Example prediction pipeline with bottlenecks
    """
    # Load data (slow)
    data = np.random.randn(n_samples, 100)
    
    # Feature engineering (expensive)
    features = []
    for i in range(n_samples):
        row_features = []
        for j in range(100):
            # Expensive computation
            row_features.append(np.sum(data[i] ** 2))
        features.append(row_features)
    
    features = np.array(features)
    
    # Model prediction
    predictions = model.predict(features)
    
    return predictions


# Run profiled function
# expensive_prediction_pipeline(n_samples=100)

print("Profiling example defined (uncomment to run)")
\`\`\`

### Memory Profiling

\`\`\`python
"""
Memory Profiling
"""

from memory_profiler import profile as mem_profile
import sys

@mem_profile
def memory_intensive_function():
    """
    Function with high memory usage
    """
    # Allocate large arrays
    large_array = np.random.randn(10000, 10000)  # ~800 MB
    
    # Process
    result = large_array ** 2
    
    # More allocations
    temp = result + large_array
    
    return temp.sum()


# Monitor memory usage
def get_memory_usage():
    """Get current memory usage in MB"""
    import psutil
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


print(f"\\nCurrent memory usage: {get_memory_usage():.2f} MB")

# In practice: use memory_profiler
# python -m memory_profiler script.py
\`\`\`

---

## Batch Processing

### Request Batching

\`\`\`python
"""
Batch Requests for Better Throughput
"""

import asyncio
from typing import List
from collections import deque
import time

class RequestBatcher:
    """
    Batch requests for efficient processing
    """
    
    def __init__(self, batch_size=32, max_wait_ms=10):
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        
        self.queue = deque()
        self.processing = False
    
    async def add_request(self, features: list) -> float:
        """
        Add request to batch
        """
        # Create future for this request
        future = asyncio.Future()
        
        self.queue.append((features, future))
        
        # Start processing if not already
        if not self.processing:
            asyncio.create_task(self._process_batch())
        
        # Wait for result
        return await future
    
    async def _process_batch(self):
        """
        Process accumulated batch
        """
        self.processing = True
        
        # Wait for batch to accumulate or timeout
        start = time.time()
        
        while len(self.queue) < self.batch_size:
            await asyncio.sleep(0.001)  # 1ms
            
            # Timeout
            if (time.time() - start) * 1000 > self.max_wait_ms:
                break
        
        if not self.queue:
            self.processing = False
            return
        
        # Collect batch
        batch = []
        futures = []
        
        while self.queue and len(batch) < self.batch_size:
            features, future = self.queue.popleft()
            batch.append(features)
            futures.append(future)
        
        # Process batch
        batch_array = np.array(batch)
        predictions = model.predict(batch_array)
        
        # Return results
        for future, prediction in zip(futures, predictions):
            future.set_result(prediction)
        
        self.processing = False
        
        # Process next batch if queue not empty
        if self.queue:
            asyncio.create_task(self._process_batch())


# Example usage
async def test_batching():
    batcher = RequestBatcher(batch_size=32, max_wait_ms=10)
    
    print("\\n=== Request Batching ===\\n")
    
    # Send 100 requests
    tasks = []
    start = time.time()
    
    for i in range(100):
        task = batcher.add_request([np.random.randn() for _ in range(10)])
        tasks.append(task)
    
    # Wait for all
    results = await asyncio.gather(*tasks)
    
    duration = time.time() - start
    
    print(f"Processed 100 requests in {duration*1000:.2f}ms")
    print(f"Throughput: {100/duration:.0f} requests/sec")
    print(f"Avg latency: {duration/100*1000:.2f}ms per request")


# Run async example
# asyncio.run(test_batching())
print("Batching example defined (uncomment to run)")
\`\`\`

---

## Key Takeaways

1. **Model Optimization**: Quantization (4x smaller), pruning (remove weights), distillation
2. **Caching**: Multi-level (memory + Redis), feature caching
3. **Horizontal Scaling**: Load balancing, auto-scaling based on metrics
4. **Profiling**: Identify bottlenecks with cProfile, memory_profiler
5. **Batching**: Process multiple requests together for better throughput

**Performance Targets (Trading)**:
- **Latency**: P95 < 50ms, P99 < 100ms
- **Throughput**: 10,000+ predictions/sec
- **Availability**: 99.9% uptime
- **Cost**: Optimize GPU/CPU usage

**Trading-Specific**:
- Cache technical indicators (expensive to compute)
- Batch predictions during market hours
- Scale up before market open
- Monitor latency closely (affects trade execution)

**Next Steps**: With performance optimized, we'll explore AutoML and Neural Architecture Search.
`,
};
