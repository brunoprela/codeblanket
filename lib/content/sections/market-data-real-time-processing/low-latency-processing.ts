export const lowLatencyProcessing = {
  title: 'Low-Latency Data Processing',
  id: 'low-latency-processing',
  content: `
# Low-Latency Data Processing

## Introduction

In high-frequency trading (HFT), microseconds determine profitability. A trading system that processes market data 10 microseconds faster than competitors can capture arbitrage opportunities worth millions annually. Sub-millisecond processing requires optimization at every level: algorithms, data structures, programming languages, operating systems, and hardware.

**Why Latency Matters:**
- **HFT Profit**: Firms co-locate servers near exchanges to save 4-5 microseconds (half the round-trip light speed from NYC to NJ)
- **Market Impact**: A 1μs advantage can mean the difference between profit and loss on every trade
- **Competition**: Top HFT firms achieve end-to-end latency of 10-50 microseconds
- **Economics**: $10M investment in low-latency infrastructure can generate $100M+ annual returns

**Latency Breakdown (Typical HFT System):**
- Market data ingestion: 1-5μs
- Order book update: 2-10μs
- Strategy logic: 5-20μs
- Order generation: 1-5μs
- Network to exchange: 50-200μs
- **Total**: 60-240μs (exchange → decision → order back to exchange)

**Real-World Examples:**
- **Citadel Securities**: Sub-10μs order book updates
- **Jump Trading**: Custom FPGA hardware for < 1μs processing
- **Virtu Financial**: 99%+ profitable trading days (latency advantage)

This section covers algorithm optimization, data structure selection, Cython/C++ integration, zero-copy techniques, and hardware acceleration.

---

## Algorithm Optimization

### O(1) vs O(log n) vs O(n) - Microseconds Matter

\`\`\`python
import time
from collections import deque
from sortedcontainers import SortedDict

class LatencyComparison:
    """Compare latency of different data structures"""
    
    def __init__(self, size: int = 10000):
        self.size = size
        
        # Different order book implementations
        self.dict_book = {}  # O(1) lookup, no ordering
        self.sorted_book = SortedDict()  # O(log n) operations
        self.list_book = []  # O(n) operations
    
    def benchmark_insert(self, iterations: int = 1000):
        """Benchmark insert performance"""
        
        # Dict insert (O(1))
        start = time.perf_counter_ns()
        for i in range(iterations):
            self.dict_book[i] = i * 100
        dict_time = (time.perf_counter_ns() - start) / iterations
        
        # SortedDict insert (O(log n))
        start = time.perf_counter_ns()
        for i in range(iterations):
            self.sorted_book[i] = i * 100
        sorted_time = (time.perf_counter_ns() - start) / iterations
        
        # List insert (O(n) - must maintain order)
        start = time.perf_counter_ns()
        for i in range(iterations):
            # Binary search insertion (still O(n) due to shift)
            self.list_book.append(i)
            self.list_book.sort()
        list_time = (time.perf_counter_ns() - start) / iterations
        
        return {
            'dict_ns': dict_time,
            'sorted_dict_ns': sorted_time,
            'list_ns': list_time
        }

# Results (typical):
# dict_ns: 50ns
# sorted_dict_ns: 200ns  
# list_ns: 5000ns

# Lesson: For price lookups, use dict (50ns). 
# For ordered book (need best bid/ask), use SortedDict (200ns).
# Never use list for order books in production (100× slower).
\`\`\`

### Pre-Allocated Memory

\`\`\`python
import numpy as np

class MemoryEfficientOrderBook:
    """Pre-allocated order book (no runtime allocations)"""
    
    def __init__(self, max_levels: int = 1000):
        # Pre-allocate arrays
        self.max_levels = max_levels
        
        # Bid book (price, size)
        self.bid_prices = np.zeros(max_levels, dtype=np.float64)
        self.bid_sizes = np.zeros(max_levels, dtype=np.int32)
        self.bid_count = 0
        
        # Ask book
        self.ask_prices = np.zeros(max_levels, dtype=np.float64)
        self.ask_sizes = np.zeros(max_levels, dtype=np.int32)
        self.ask_count = 0
    
    def update_bid(self, price: float, size: int):
        """Update bid (no memory allocation)"""
        # Find price level (binary search - O(log n))
        idx = np.searchsorted(self.bid_prices[:self.bid_count], -price)
        
        if idx < self.bid_count and self.bid_prices[idx] == -price:
            # Update existing level
            if size == 0:
                # Delete level (shift array)
                self.bid_prices[idx:self.bid_count-1] = self.bid_prices[idx+1:self.bid_count]
                self.bid_sizes[idx:self.bid_count-1] = self.bid_sizes[idx+1:self.bid_count]
                self.bid_count -= 1
            else:
                # Update size
                self.bid_sizes[idx] = size
        else:
            # Insert new level
            if self.bid_count < self.max_levels:
                # Shift elements
                self.bid_prices[idx+1:self.bid_count+1] = self.bid_prices[idx:self.bid_count]
                self.bid_sizes[idx+1:self.bid_count+1] = self.bid_sizes[idx:self.bid_count]
                
                # Insert
                self.bid_prices[idx] = -price  # Negative for descending order
                self.bid_sizes[idx] = size
                self.bid_count += 1
    
    def get_best_bid(self) -> tuple:
        """Get best bid in O(1)"""
        if self.bid_count > 0:
            return (-self.bid_prices[0], self.bid_sizes[0])
        return (None, None)

# Advantage: Zero malloc() calls during updates (10× faster than dict reallocation)
# Disadvantage: Fixed size, complex insertion logic
# Use case: Ultra-low-latency HFT where every nanosecond counts
\`\`\`

---

## Cython Optimization (10× Speedup)

### Python vs Cython Comparison

\`\`\`python
# Pure Python (baseline)
def calculate_vwap_python(prices, volumes):
    """Calculate VWAP in pure Python"""
    total_pv = 0.0
    total_vol = 0
    for i in range(len(prices)):
        total_pv += prices[i] * volumes[i]
        total_vol += volumes[i]
    return total_pv / total_vol if total_vol > 0 else 0.0

# Typical performance: 10μs for 100 ticks
\`\`\`

\`\`\`cython
# Cython (10× faster)
# File: vwap_fast.pyx

import numpy as np
cimport numpy as cnp
from libc.math cimport fabs

def calculate_vwap_cython(cnp.ndarray[cnp.float64_t, ndim=1] prices,
                          cnp.ndarray[cnp.int32_t, ndim=1] volumes):
    """Calculate VWAP in Cython with type annotations"""
    cdef double total_pv = 0.0
    cdef long total_vol = 0
    cdef int i, n = len(prices)
    
    for i in range(n):
        total_pv += prices[i] * volumes[i]
        total_vol += volumes[i]
    
    return total_pv / total_vol if total_vol > 0 else 0.0

# Typical performance: 1μs for 100 ticks (10× faster)

# Compilation:
# python setup.py build_ext --inplace
\`\`\`

**Setup for Cython:**

\`\`\`python
# setup.py
from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("vwap_fast.pyx"),
    include_dirs=[numpy.get_include()]
)

# Build: python setup.py build_ext --inplace
# Import: from vwap_fast import calculate_vwap_cython
\`\`\`

---

## Zero-Copy Techniques

### Memory Mapping (mmap)

\`\`\`python
import mmap
import struct

class ZeroCopyTickReader:
    """Read tick data with zero-copy (mmap)"""
    
    def __init__(self, filename: str):
        self.file = open(filename, 'r+b')
        self.mmap = mmap.mmap(self.file.fileno(), 0)
        
        # Tick format: timestamp(8), symbol(4), price(8), size(4) = 24 bytes
        self.tick_size = 24
        self.tick_count = len(self.mmap) // self.tick_size
    
    def get_tick(self, index: int) -> dict:
        """Get tick without copying memory"""
        offset = index * self.tick_size
        
        # Read directly from mapped memory (no copy)
        tick_bytes = self.mmap[offset:offset+self.tick_size]
        
        # Unpack (still requires some processing, but no malloc)
        timestamp, symbol, price, size = struct.unpack('Qidf', tick_bytes)
        
        return {
            'timestamp': timestamp,
            'symbol': symbol,
            'price': price,
            'size': size
        }
    
    def close(self):
        self.mmap.close()
        self.file.close()

# Advantage: No malloc for each tick (5-10× faster than file.read())
# Reading 1M ticks: 100ms (mmap) vs 1000ms (file.read())
\`\`\`

### NumPy Views (Zero-Copy Arrays)

\`\`\`python
import numpy as np

class TickArray:
    """Process ticks using NumPy views (zero-copy)"""
    
    def __init__(self, capacity: int = 100000):
        # Pre-allocate structured array
        self.dtype = np.dtype([
            ('timestamp', np.uint64),
            ('symbol', np.int32),
            ('price', np.float64),
            ('size', np.int32)
        ])
        
        self.ticks = np.zeros(capacity, dtype=self.dtype)
        self.count = 0
    
    def add_tick(self, timestamp, symbol, price, size):
        """Add tick (writes to pre-allocated memory)"""
        self.ticks[self.count] = (timestamp, symbol, price, size)
        self.count += 1
    
    def get_prices(self):
        """Get price array (zero-copy view)"""
        # No memory allocation - returns view
        return self.ticks['price'][:self.count]
    
    def calculate_vwap(self):
        """Calculate VWAP using vectorized operations"""
        prices = self.ticks['price'][:self.count]
        sizes = self.ticks['size'][:self.count]
        
        # Vectorized (no loops in Python)
        return np.sum(prices * sizes) / np.sum(sizes)

# Performance: VWAP on 10K ticks = 100μs (vectorized) vs 10ms (Python loop)
\`\`\`

---

## Hardware Optimization

### CPU Affinity (Pin Threads to Cores)

\`\`\`python
import os
import psutil

def set_cpu_affinity(core: int):
    """Pin process to specific CPU core"""
    process = psutil.Process(os.getpid())
    process.cpu_affinity([core])
    
    print(f"Process pinned to CPU core {core}")

# Usage: Pin market data thread to isolated core (no context switches)
set_cpu_affinity(core=2)  # Use core 2 exclusively

# Benefit: Reduces context switching latency from 10μs to < 1μs
# Requires: CPU isolation via kernel boot parameters (isolcpus=2,3)
\`\`\`

### Huge Pages (Reduce TLB Misses)

\`\`\`bash
# Enable huge pages (Linux)
# Add to /etc/sysctl.conf:
vm.nr_hugepages = 1024  # 2GB huge pages

# Mount hugetlbfs
mkdir /mnt/huge
mount -t hugetlbfs nodev /mnt/huge

# Benefit: 2MB pages instead of 4KB (512× fewer TLB entries)
# Latency reduction: 10-20% for memory-intensive operations
\`\`\`

### DPDK (Kernel Bypass Networking)

\`\`\`python
# DPDK bypasses kernel network stack
# Direct NIC → User space (no context switches)

# Traditional: NIC → Kernel → User space = 10-50μs
# DPDK: NIC → User space = 1-5μs

# Setup (requires DPDK-compatible NIC and drivers)
# EAL initialization code (C/C++)

# Benefit: 10× lower network latency
# Trade-off: Complex setup, requires dedicated NICs
\`\`\`

---

## Production Low-Latency Patterns

### Lock-Free Data Structures

\`\`\`python
from queue import Queue
import threading

class LockFreeQueue:
    """Lock-free queue using atomic operations"""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.head = 0
        self.tail = 0
    
    def push(self, item) -> bool:
        """Push item (single producer)"""
        next_tail = (self.tail + 1) % self.capacity
        
        if next_tail == self.head:
            return False  # Queue full
        
        self.buffer[self.tail] = item
        self.tail = next_tail
        return True
    
    def pop(self):
        """Pop item (single consumer)"""
        if self.head == self.tail:
            return None  # Queue empty
        
        item = self.buffer[self.head]
        self.head = (self.head + 1) % self.capacity
        return item

# Advantage: No mutex locks (10× faster than Queue for single producer/consumer)
# Latency: 50ns (lock-free) vs 500ns (mutex)
\`\`\`

---

## Latency Measurement

\`\`\`python
import time

class LatencyProfiler:
    """Profile latency of critical code paths"""
    
    def __init__(self):
        self.measurements = {}
    
    def measure(self, name: str):
        """Context manager for measuring code block"""
        class Timer:
            def __init__(self, profiler, name):
                self.profiler = profiler
                self.name = name
            
            def __enter__(self):
                self.start = time.perf_counter_ns()
                return self
            
            def __exit__(self, *args):
                elapsed_ns = time.perf_counter_ns() - self.start
                
                if self.name not in self.profiler.measurements:
                    self.profiler.measurements[self.name] = []
                
                self.profiler.measurements[self.name].append(elapsed_ns)
        
        return Timer(self, name)
    
    def report(self):
        """Generate latency report"""
        import numpy as np
        
        print("\\nLatency Profile:")
        print("-" * 60)
        
        for name, measurements in self.measurements.items():
            arr = np.array(measurements)
            print(f"{name}:")
            print(f"  Count: {len(arr)}")
            print(f"  Mean: {np.mean(arr)/1000:.2f}μs")
            print(f"  P50: {np.percentile(arr, 50)/1000:.2f}μs")
            print(f"  P99: {np.percentile(arr, 99)/1000:.2f}μs")
            print(f"  Max: {np.max(arr)/1000:.2f}μs")

# Usage
profiler = LatencyProfiler()

for _ in range(10000):
    with profiler.measure('order_book_update'):
        # Critical code path
        update_order_book()
    
    with profiler.measure('strategy_logic'):
        # Strategy execution
        execute_strategy()

profiler.report()

# Target latencies (HFT):
# - Order book update: < 10μs
# - Strategy logic: < 20μs
# - Order generation: < 5μs
\`\`\`

---

## Best Practices

1. **Profile first** - Measure before optimizing (don't guess)
2. **O(1) algorithms** - Avoid O(n) operations in hot paths
3. **Pre-allocate** - No malloc() in critical loops
4. **Use Cython** - 10× faster than pure Python
5. **Zero-copy** - mmap, NumPy views, shared memory
6. **CPU affinity** - Pin threads to isolated cores
7. **Avoid GC** - Use fixed-size structures, disable GC during trading

Now you can build sub-millisecond trading systems!
`,
};
