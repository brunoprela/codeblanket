export const lowLatencyProgramming = {
    title: 'Low-Latency Programming Techniques',
    id: 'low-latency-programming',
    content: `
# Low-Latency Programming Techniques

## Introduction

**Low-latency programming** optimizes code for microsecond/nanosecond performance. Critical for:
- **High-frequency trading**: Compete on speed (μs advantage = millions in profit)
- **Market making**: Quote faster than competitors
- **Arbitrage**: Execute before price discrepancies close

**Latency targets by system type:**
- **Ultra-low latency HFT**: Order to market <10μs (Citadel, Jump Trading)
- **Low latency market making**: Order to market <100μs (Jane Street, Virtu)
- **Standard algorithmic trading**: Order to market <1ms (typical quant funds)
- **Retail trading**: Order to market <100ms (Interactive Brokers, TD Ameritrade)

**Real-World Performance:**
- **Citadel Securities**: ~10-50 nanoseconds processing time (FPGA + custom ASICs)
- **Virtu Financial**: Sub-microsecond execution (C++ + kernel bypass)
- **Jane Street**: <100μs for complex options strategies (OCaml + C++)
- **Two Sigma**: <1ms for ML-driven strategies (C++ + Python)

This section covers production techniques for ultra-low latency systems, from Python optimization to C++ and FPGA.

---

## Latency Measurement and Profiling

\`\`\`python
"""
Accurate Latency Measurement
"""

import time
from contextlib import contextmanager
from typing import Dict, List
import statistics

class LatencyProfiler:
    """
    Measure microsecond-level latency
    
    Uses CLOCK_MONOTONIC_RAW for accuracy
    """
    
    def __init__(self):
        self.measurements: Dict[str, List[float]] = {}
    
    @contextmanager
    def measure(self, name: str):
        """Context manager for measuring latency"""
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            end = time.perf_counter_ns()
            latency_ns = end - start
            
            if name not in self.measurements:
                self.measurements[name] = []
            self.measurements[name].append(latency_ns)
    
    def get_statistics(self) -> Dict[str, Dict]:
        """Get latency statistics"""
        stats = {}
        for name, measurements in self.measurements.items():
            stats[name] = {
                'count': len(measurements),
                'mean_ns': statistics.mean(measurements),
                'median_ns': statistics.median(measurements),
                'p50_ns': statistics.quantiles(measurements, n=2)[0],
                'p95_ns': statistics.quantiles(measurements, n=20)[18],
                'p99_ns': statistics.quantiles(measurements, n=100)[98],
                'min_ns': min(measurements),
                'max_ns': max(measurements),
            }
        return stats
    
    def print_report(self):
        """Print latency report"""
        print("\\n" + "=" * 80)
        print("LATENCY PROFILE")
        print("=" * 80)
        
        stats = self.get_statistics()
        for name, s in stats.items():
            print(f"\\n{name}:")
            print(f"  Count:  {s['count']:,}")
            print(f"  Mean:   {s['mean_ns']/1000:.2f} μs")
            print(f"  Median: {s['median_ns']/1000:.2f} μs")
            print(f"  P95:    {s['p95_ns']/1000:.2f} μs")
            print(f"  P99:    {s['p99_ns']/1000:.2f} μs")
            print(f"  Min:    {s['min_ns']/1000:.2f} μs")
            print(f"  Max:    {s['max_ns']/1000:.2f} μs")


# Example usage
profiler = LatencyProfiler()

for i in range(1000):
    with profiler.measure("order_processing"):
        # Simulate order processing
        result = sum(range(100))

profiler.print_report()
\`\`\`

---

## Memory Optimization

\`\`\`python
"""
Memory-Efficient Order Book
"""

import numpy as np
from dataclasses import dataclass
from typing import List

class PreAllocatedOrderBook:
    """
    Order book with pre-allocated memory
    
    Avoids heap allocations during runtime
    """
    
    def __init__(self, max_levels: int = 100):
        # Pre-allocate arrays for bid/ask levels
        self.max_levels = max_levels
        
        # Bid prices and sizes (numpy for cache locality)
        self.bid_prices = np.zeros(max_levels, dtype=np.float64)
        self.bid_sizes = np.zeros(max_levels, dtype=np.int64)
        self.bid_count = 0
        
        # Ask prices and sizes
        self.ask_prices = np.zeros(max_levels, dtype=np.float64)
        self.ask_sizes = np.zeros(max_levels, dtype=np.int64)
        self.ask_count = 0
    
    def update_bid(self, price: float, size: int):
        """Update bid level (no allocation)"""
        # Binary search for insertion point
        idx = np.searchsorted(self.bid_prices[:self.bid_count], price)
        
        if idx < self.bid_count and self.bid_prices[idx] == price:
            # Update existing level
            if size == 0:
                # Remove level
                self.bid_prices[idx:self.bid_count-1] = self.bid_prices[idx+1:self.bid_count]
                self.bid_sizes[idx:self.bid_count-1] = self.bid_sizes[idx+1:self.bid_count]
                self.bid_count -= 1
            else:
                self.bid_sizes[idx] = size
        else:
            # Insert new level
            if self.bid_count < self.max_levels:
                self.bid_prices[idx+1:self.bid_count+1] = self.bid_prices[idx:self.bid_count]
                self.bid_sizes[idx+1:self.bid_count+1] = self.bid_sizes[idx:self.bid_count]
                self.bid_prices[idx] = price
                self.bid_sizes[idx] = size
                self.bid_count += 1
    
    def get_best_bid(self) -> tuple[float, int]:
        """Get best bid (no allocation)"""
        if self.bid_count > 0:
            return (self.bid_prices[self.bid_count-1], self.bid_sizes[self.bid_count-1])
        return (0.0, 0)
    
    def get_best_ask(self) -> tuple[float, int]:
        """Get best ask (no allocation)"""
        if self.ask_count > 0:
            return (self.ask_prices[0], self.ask_sizes[0])
        return (0.0, 0)


class ObjectPool:
    """
    Object pooling to avoid allocations
    
    Pre-allocate objects and reuse them
    """
    
    def __init__(self, object_class, pool_size: int = 10000):
        self.object_class = object_class
        self.pool_size = pool_size
        
        # Pre-allocate objects
        self.pool = [object_class() for _ in range(pool_size)]
        self.available = list(range(pool_size))
    
    def acquire(self):
        """Get object from pool"""
        if not self.available:
            raise RuntimeError("Pool exhausted")
        
        idx = self.available.pop()
        return self.pool[idx]
    
    def release(self, obj):
        """Return object to pool"""
        # Reset object state
        obj.reset()
        
        # Find index and mark available
        idx = self.pool.index(obj)
        self.available.append(idx)


@dataclass
class Order:
    """Reusable order object"""
    order_id: int = 0
    symbol: str = ""
    side: str = ""
    quantity: int = 0
    price: float = 0.0
    
    def reset(self):
        """Reset for reuse"""
        self.order_id = 0
        self.symbol = ""
        self.side = ""
        self.quantity = 0
        self.price = 0.0


# Example: Object pool
order_pool = ObjectPool(Order, pool_size=1000)

# Acquire order (no allocation)
order = order_pool.acquire()
order.order_id = 12345
order.symbol = "AAPL"

# Process order...

# Release order (reuse)
order_pool.release(order)
\`\`\`

---

## CPU Optimization

\`\`\`python
"""
CPU-Level Optimizations
"""

import os
import psutil
from typing import List

class CPUOptimizer:
    """
    CPU-level optimizations for low latency
    """
    
    @staticmethod
    def pin_to_core(core_id: int):
        """
        Pin process to specific CPU core
        
        Benefits:
        - Avoid context switches
        - Keep L1/L2 cache hot
        - Predictable performance
        """
        import os
        
        # Linux only
        os.sched_setaffinity(0, {core_id})
        print(f"[CPU] Pinned to core {core_id}")
    
    @staticmethod
    def set_high_priority():
        """
        Set process to high priority
        
        Requires root/admin permissions
        """
        import os
        
        try:
            # Set to real-time priority (Linux)
            os.nice(-20)  # Highest priority
            print("[CPU] Set to high priority")
        except PermissionError:
            print("[CPU] Warning: Could not set high priority (need root)")
    
    @staticmethod
    def disable_frequency_scaling():
        """
        Disable CPU frequency scaling for consistent performance
        
        Prevents CPU throttling during low load
        """
        # On Linux: echo performance > /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
        print("[CPU] Disable frequency scaling (requires root)")
        print("      Run: echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor")
    
    @staticmethod
    def get_numa_info():
        """
        Get NUMA node information
        
        Important for multi-socket systems
        """
        try:
            import subprocess
            result = subprocess.run(['numactl', '--hardware'], capture_output=True, text=True)
            print("[CPU] NUMA Information:")
            print(result.stdout)
        except FileNotFoundError:
            print("[CPU] numactl not installed")
    
    @staticmethod
    def isolate_cores(core_list: List[int]):
        """
        Isolate CPU cores for trading threads
        
        Prevents OS from scheduling other tasks on these cores
        """
        cores_str = ','.join(map(str, core_list))
        print(f"[CPU] To isolate cores {cores_str}:")
        print(f"      Add to kernel boot params: isolcpus={cores_str}")
        print(f"      Edit /etc/default/grub and run update-grub")


# Example usage
optimizer = CPUOptimizer()

# Pin to core 0 (trading thread)
optimizer.pin_to_core(0)

# Set high priority
optimizer.set_high_priority()

# Get NUMA info
optimizer.get_numa_info()
\`\`\`

---

## Lock-Free Data Structures

\`\`\`python
"""
Lock-Free Circular Buffer (Python)

For true lock-free, use C++ with std::atomic
This is a simplified Python version
"""

import threading
from typing import Optional, Generic, TypeVar

T = TypeVar('T')

class LockFreeRingBuffer(Generic[T]):
    """
    Lock-free ring buffer for single producer, single consumer
    
    Benefits:
    - No mutex contention
    - Cache-line friendly
    - Bounded latency
    """
    
    def __init__(self, size: int = 1024):
        self.size = size
        self.buffer = [None] * size
        self.write_idx = 0
        self.read_idx = 0
    
    def push(self, item: T) -> bool:
        """
        Push item (producer only)
        
        Returns: True if successful, False if buffer full
        """
        next_write = (self.write_idx + 1) % self.size
        
        # Check if buffer full
        if next_write == self.read_idx:
            return False
        
        # Write item
        self.buffer[self.write_idx] = item
        
        # Update write index (atomic on Python due to GIL)
        self.write_idx = next_write
        
        return True
    
    def pop(self) -> Optional[T]:
        """
        Pop item (consumer only)
        
        Returns: Item or None if buffer empty
        """
        # Check if buffer empty
        if self.read_idx == self.write_idx:
            return None
        
        # Read item
        item = self.buffer[self.read_idx]
        
        # Update read index
        self.read_idx = (self.read_idx + 1) % self.size
        
        return item
    
    def is_empty(self) -> bool:
        """Check if buffer empty"""
        return self.read_idx == self.write_idx
    
    def is_full(self) -> bool:
        """Check if buffer full"""
        next_write = (self.write_idx + 1) % self.size
        return next_write == self.read_idx


# Example: Market data pipeline
market_data_buffer = LockFreeRingBuffer(size=10000)

# Producer thread (market data feed)
def producer():
    for i in range(100000):
        tick = {'symbol': 'AAPL', 'price': 150.0 + i * 0.01}
        while not market_data_buffer.push(tick):
            pass  # Spin until space available

# Consumer thread (strategy)
def consumer():
    while True:
        tick = market_data_buffer.pop()
        if tick:
            # Process tick
            pass
\`\`\`

---

## C++ Low-Latency Implementation

\`\`\`cpp
"""
Production C++ Low-Latency Order Handler

Achieves <1μs latency for order processing
"""

#include <iostream>
#include <chrono>
#include <atomic>
#include <array>
#include <cstring>
#include <immintrin.h>  // For CPU intrinsics

// Cache line size (64 bytes on x86-64)
constexpr size_t CACHE_LINE_SIZE = 64;

// Align to cache line to avoid false sharing
struct alignas(CACHE_LINE_SIZE) Order {
    uint64_t order_id;
    char symbol[8];      // Fixed size, no heap
    uint32_t quantity;
    double price;
    char side;           // 'B' or 'S'
    char order_type;     // 'M' = market, 'L' = limit
    
    // Padding to fill cache line
    char padding[CACHE_LINE_SIZE - sizeof(order_id) - 8 - sizeof(quantity) 
                  - sizeof(price) - 2];
};

// Lock-free ring buffer for order pipeline
template<typename T, size_t N>
class LockFreeQueue {
private:
    std::array<T, N> buffer;
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> write_idx{0};
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> read_idx{0};

public:
    bool push(const T& item) noexcept {
        const size_t current_write = write_idx.load(std::memory_order_relaxed);
        const size_t next_write = (current_write + 1) % N;
        
        // Check if full
        if (next_write == read_idx.load(std::memory_order_acquire)) {
            return false;
        }
        
        // Copy item
        buffer[current_write] = item;
        
        // Publish write
        write_idx.store(next_write, std::memory_order_release);
        
        return true;
    }
    
    bool pop(T& item) noexcept {
        const size_t current_read = read_idx.load(std::memory_order_relaxed);
        
        // Check if empty
        if (current_read == write_idx.load(std::memory_order_acquire)) {
            return false;
        }
        
        // Copy item
        item = buffer[current_read];
        
        // Publish read
        read_idx.store((current_read + 1) % N, std::memory_order_release);
        
        return true;
    }
};

// Ultra-low latency order processor
class OrderProcessor {
private:
    LockFreeQueue<Order, 10000> order_queue;
    
    // Pre-allocated order pool (no malloc)
    std::array<Order, 10000> order_pool;
    std::atomic<size_t> next_order_idx{0};
    
    // Statistics
    std::atomic<uint64_t> orders_processed{0};
    std::atomic<uint64_t> total_latency_ns{0};

public:
    // Process order in <1μs
    void process_order(const Order& order) noexcept {
        auto start = std::chrono::high_resolution_clock::now();
        
        // 1. Validate (branchless if possible)
        const bool valid = (order.quantity > 0) & (order.price > 0.0);
        
        // 2. Calculate notional
        const double notional = order.quantity * order.price;
        
        // 3. Risk check (inline)
        const bool risk_ok = notional < 1000000.0;  // $1M limit
        
        // 4. Send to exchange (zero-copy via shared memory)
        if (valid & risk_ok) {
            send_to_exchange(order);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        
        // Update stats (lock-free)
        orders_processed.fetch_add(1, std::memory_order_relaxed);
        total_latency_ns.fetch_add(latency.count(), std::memory_order_relaxed);
    }
    
    void print_statistics() const {
        const uint64_t count = orders_processed.load(std::memory_order_relaxed);
        const uint64_t total_ns = total_latency_ns.load(std::memory_order_relaxed);
        
        if (count > 0) {
            const double avg_latency_ns = static_cast<double>(total_ns) / count;
            std::cout << "Orders processed: " << count << "\\n";
            std::cout << "Average latency: " << avg_latency_ns << " ns\\n";
            std::cout << "Average latency: " << (avg_latency_ns / 1000.0) << " μs\\n";
        }
    }

private:
    void send_to_exchange(const Order& order) noexcept {
        // Write to memory-mapped exchange gateway
        // Exchange reads directly (zero-copy)
        
        // In production: DMA to network card
    }
};

// Main trading loop
void trading_loop() {
    OrderProcessor processor;
    
    // Pin to CPU core 0
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
    
    // Set real-time priority
    struct sched_param param;
    param.sched_priority = 99;  // Highest priority
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
    
    Order order;
    
    while (true) {
        // Read order from lock-free queue
        // In production: Read from shared memory or network
        
        // Process in <1μs
        processor.process_order(order);
    }
}

// Compile with:
// g++ -O3 -march=native -std=c++17 low_latency.cpp -o trading -lpthread
\`\`\`

---

## Kernel Bypass Networking

\`\`\`python
"""
Kernel Bypass for Ultra-Low Latency

DPDK (Data Plane Development Kit) bypasses Linux kernel
"""

# Conceptual Python wrapper (actual DPDK is C)

class DPDKInterface:
    """
    DPDK interface for kernel bypass
    
    Latency reduction: ~5-10μs saved by bypassing kernel
    
    Benefits:
    - Direct NIC access (no kernel)
    - Zero-copy packet processing
    - Poll-mode drivers (no interrupts)
    """
    
    def __init__(self, port_id: int = 0):
        self.port_id = port_id
        print(f"[DPDK] Initializing port {port_id}")
        
        # In production:
        # 1. Initialize DPDK EAL (Environment Abstraction Layer)
        # 2. Configure NIC port
        # 3. Setup memory pools
        # 4. Setup RX/TX queues
    
    def receive_packet(self) -> bytes:
        """
        Receive packet (zero-copy)
        
        Polls NIC directly, no kernel involvement
        """
        # In production: rte_eth_rx_burst()
        pass
    
    def send_packet(self, packet: bytes):
        """
        Send packet (zero-copy)
        
        Writes directly to NIC ring buffer
        """
        # In production: rte_eth_tx_burst()
        pass


# Alternative: Solarflare Onload (user-space TCP/IP stack)
class SolarflareOnload:
    """
    Solarflare Onload: User-space TCP/IP
    
    Benefits:
    - Full TCP/IP stack in user-space
    - No kernel for socket operations
    - Drop-in replacement for sockets API
    """
    
    def __init__(self):
        print("[Onload] Accelerated TCP/IP stack")
        
        # Usage:
        # 1. Install Solarflare Onload
        # 2. Run: onload ./trading_app
        # 3. All socket() calls automatically accelerated


# Production setup for ultra-low latency:
"""
1. Hardware:
   - Solarflare NIC (kernel bypass support)
   - Intel NIC with DPDK support
   - Direct connection to exchange (no switches)

2. Network:
   - 10/40/100 Gbps network
   - Co-location in exchange data center
   - Direct market data feed

3. Software:
   - DPDK for packet processing
   - Custom TCP/IP stack or Solarflare Onload
   - Lock-free queues for order pipeline
   - CPU pinning and real-time priority

4. Result:
   - Order to exchange: <10μs
   - Market data processing: <1μs
   - Total round-trip: <50μs
"""
\`\`\`

---

## Summary

**Low-Latency Essentials:**
1. **Memory optimization**: Pre-allocate, object pooling, zero-copy
2. **CPU optimization**: Core pinning, high priority, disable frequency scaling
3. **Lock-free**: Atomic operations, ring buffers, no mutexes
4. **Kernel bypass**: DPDK, Solarflare Onload for network
5. **Profiling**: Measure everything, p99 latency matters
6. **C++ when needed**: Python for strategy, C++ for order execution

**Latency Budget Example (100μs total):**
- Market data receive: 10μs (kernel bypass)
- Strategy calculation: 40μs (Python or C++)
- Risk check: 5μs (C++)
- Order serialization: 10μs (FIX)
- Network send: 5μs (kernel bypass)
- Exchange processing: 30μs (exchange-dependent)

**Real-World Trade-offs:**
- **Python**: Easy development, ~1-10ms latency (good for most strategies)
- **Python + Cython**: 10-100x faster critical paths, ~100μs-1ms
- **C++**: Full control, ~1-100μs, harder to develop
- **FPGA**: <1μs, very expensive, only for HFT

**Next Section**: Module 14.10 - Message Queues for Trading
`,
};
