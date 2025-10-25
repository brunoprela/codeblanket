export const highFrequencyTradingArchitecture = {
  title: 'High-Frequency Trading Architecture',
  id: 'high-frequency-trading-architecture',
  content: `
# High-Frequency Trading Architecture

## Introduction

**High-Frequency Trading (HFT)** operates at microsecond (μs) and even nanosecond (ns) timescales. At these speeds, traditional software architectures fail. HFT requires:

- **Ultra-low latency**: <100μs order-to-execution, ideally <10μs
- **Deterministic performance**: No garbage collection pauses, no OS interrupts
- **Co-location**: Servers physically near exchange
- **Specialized hardware**: FPGAs, kernel bypass networking
- **Extreme optimization**: Every nanosecond counts

### Speed of Light Constraints

Physics limits HFT:
- Light travels ~300 km/ms in fiber
- Chicago to New York: 1,200 km ≈ 4ms one-way
- Co-located servers: 100m ≈ 0.0003ms

### The Economics

At μs scale:
- 1μs advantage can be worth $millions/year
- Firms spend $10M+ on low-latency infrastructure
- Co-location fees: $10K-50K/month per rack

By the end of this section, you'll understand:
- FPGA-based trading systems
- Kernel bypass networking (DPDK, Solarflare)
- Sub-microsecond order routing
- Memory management for deterministic performance
- Co-location strategies
- Hardware optimizations

---

## Traditional vs HFT Architecture

### Traditional Software Stack

\`\`\`
Application (Python/Java)
    ↓ ~1-10ms
Operating System (Linux)
    ↓ ~0.1-1ms
Network Stack (TCP/IP)
    ↓ ~0.05-0.5ms
Network Card
    ↓
Exchange

Total: ~1-12ms
\`\`\`

**Bottlenecks**:
- System calls (user → kernel mode switch): ~1μs
- TCP stack processing: ~10-100μs
- Garbage collection (Java/Python): ~10ms pauses
- Context switches: ~1-10μs

### HFT Software Stack

\`\`\`
FPGA Logic (Verilog/VHDL)
    ↓ ~0.0001-0.001ms (100ns-1μs)
Kernel Bypass (DPDK/OpenOnload)
    ↓ ~0.001-0.010ms (1-10μs)
Specialized NIC
    ↓
Exchange (co-located)

Total: ~1-50μs
\`\`\`

---

## FPGA-Based Trading

### What is FPGA?

**Field-Programmable Gate Array**: Reconfigurable hardware chip programmed to perform specific operations in parallel.

**Advantages**:
- **Latency**: 100-500ns processing
- **Deterministic**: No OS, no garbage collection
- **Parallel**: Process multiple streams simultaneously
- **Low jitter**: <10ns variance

**Disadvantages**:
- **Development**: Complex (Verilog/VHDL), 6-12 months
- **Cost**: $50K-500K per FPGA card
- **Limited logic**: Can't fit complex strategies
- **Debugging**: Extremely difficult

### FPGA Trading Pipeline

\`\`\`
Market Data → FPGA Parser → Strategy Logic → Order Generator → Exchange
   (100ns)      (200ns)         (300ns)         (100ns)        (200ns)

Total: ~900ns = 0.9μs
\`\`\`

### Example: Simple Arbitrage in FPGA

\`\`\`verilog
/*
 * Simple arbitrage detector in Verilog
 * Detects price discrepancies between two exchanges
 * Latency: ~200ns
 */

module arbitrage_detector(
    input wire clk,  // 200 MHz clock (5ns period)
    input wire [63:0] price_exchange_a,  // Fixed-point price
    input wire [63:0] price_exchange_b,
    input wire valid_a,
    input wire valid_b,
    output reg [63:0] arb_opportunity,  // Price difference
    output reg arb_signal  // High when arbitrage detected
);

// Threshold: 0.1% price difference
localparam THRESHOLD = 64'd100;  // 0.1% in basis points

// Calculate price difference
reg [63:0] price_diff;

always @(posedge clk) begin
    if (valid_a && valid_b) begin
        // Calculate absolute difference
        if (price_exchange_a > price_exchange_b) begin
            price_diff <= price_exchange_a - price_exchange_b;
        end else begin
            price_diff <= price_exchange_b - price_exchange_a;
        end
        
        // Calculate percentage difference (price_diff / avg_price * 10000)
        arb_opportunity <= (price_diff * 64'd10000) / 
                          ((price_exchange_a + price_exchange_b) >> 1);
        
        // Signal if above threshold
        arb_signal <= (arb_opportunity > THRESHOLD) ? 1'b1 : 1'b0;
    end else begin
        arb_signal <= 1'b0;
    end
end

endmodule
\`\`\`

### FPGA Development Flow

\`\`\`python
"""
FPGA development is radically different from software
This is pseudocode showing the concept
"""

# 1. Design in Hardware Description Language (Verilog/VHDL)
# Write logic as circuits, not sequential code

# 2. Simulation
# Test logic before burning to hardware (very expensive to fix bugs)

# 3. Synthesis
# Convert HDL to gate-level netlist
# Takes 2-8 hours for large designs

# 4. Place and Route
# Map gates to physical FPGA resources
# Optimize for timing

# 5. Generate Bitstream
# Create configuration file for FPGA

# 6. Program FPGA
# Load bitstream onto hardware

# 7. Verification
# Test in live environment with replayed market data

# Development cycle: 1-2 weeks per iteration
# vs Software: minutes per iteration
\`\`\`

---

## Kernel Bypass Networking

### The Problem with Standard Networking

Traditional Linux network stack:

\`\`\`
Application
    ↓ (syscall: 1μs)
Kernel TCP/IP Stack
    ↓ (processing: 10-50μs)
Network Driver
    ↓
NIC Hardware

Total: ~20-100μs per packet
\`\`\`

**Bottlenecks**:
- Context switches
- Memory copies
- Interrupt handling
- TCP protocol overhead

### Kernel Bypass: DPDK

**DPDK (Data Plane Development Kit)**: Framework for fast packet processing in userspace.

\`\`\`
Application (with DPDK)
    ↓ (direct memory access: 0.1μs)
NIC Hardware (polled mode)

Total: ~1-5μs per packet
\`\`\`

\`\`\`python
"""
DPDK Packet Processing (C pseudocode)
Poll network card directly from userspace
"""

# This is conceptual—actual DPDK is in C

class DPDKReceiver:
    def __init__(self, port_id):
        # Initialize DPDK
        self.port = dpdk_init_port(port_id)
        
        # Allocate memory pools
        self.mbuf_pool = dpdk_pktmbuf_pool_create(
            name="mbuf_pool",
            n=8192,  # Number of buffers
            cache_size=256
        )
        
        # Configure port
        dpdk_eth_dev_configure(port_id, rx_queues=1, tx_queues=1)
        
    def poll_packets(self):
        """Poll for packets (no interrupts)"""
        while True:
            # Receive batch of packets
            packets = dpdk_eth_rx_burst(
                port_id=self.port,
                queue_id=0,
                rx_pkts=self.packet_buffer,
                nb_pkts=32  # Batch size
            )
            
            # Process each packet
            for pkt in packets[:packets_received]:
                self.process_packet(pkt)
    
    def process_packet(self, pkt):
        """Process packet with zero-copy"""
        # Parse directly from packet memory
        # No memcpy required
        eth_hdr = parse_eth_header(pkt.data)
        ip_hdr = parse_ip_header(pkt.data + 14)
        udp_hdr = parse_udp_header(pkt.data + 34)
        
        # Extract market data
        market_data = parse_market_data(pkt.data + 42)
        
        # Generate order if strategy triggers
        if self.strategy.should_trade(market_data):
            self.send_order(market_data)
    
    def send_order(self, market_data):
        """Send order with zero-copy"""
        # Allocate packet from memory pool
        pkt = dpdk_pktmbuf_alloc(self.mbuf_pool)
        
        # Write order directly to packet memory
        write_fix_message(pkt.data, market_data)
        
        # Transmit packet
        dpdk_eth_tx_burst(port_id=self.port, queue_id=0, tx_pkts=[pkt])

# Latency: ~2-5μs vs ~50-100μs with kernel stack
\`\`\`

### Solarflare OpenOnload

Alternative to DPDK with TCP support:

\`\`\`c
/*
 * OpenOnload: Kernel bypass with TCP
 * Maintains TCP state machine in userspace
 */

#include <onload/...>

int main() {
    // Create socket
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    
    // Enable OpenOnload
    onload_set_stackname(ONLOAD_ALL_THREADS, "hft_stack");
    
    // Connect to exchange
    connect(sock, ...);
    
    // Receive with low latency
    while (1) {
        // Poll mode (no blocking system calls)
        int len = recv(sock, buffer, sizeof(buffer), MSG_DONTWAIT);
        
        if (len > 0) {
            // Process market data
            process_market_data(buffer, len);
        }
    }
}

// Latency: ~5-10μs (includes TCP overhead)
// vs DPDK: 2-5μs (UDP only)
\`\`\`

---

## Memory Management for Low Latency

### The Problem

Standard memory allocation:

\`\`\`python
# Python
order = Order(...)  # Allocation: ~1μs, GC later

# Java
Order order = new Order(...);  # Allocation: ~0.5μs, GC pause: 10ms+

# C++
Order* order = new Order(...);  # Allocation: ~0.2μs, fragmentation over time
\`\`\`

**Garbage Collection Pauses**:
- Python: 10-100ms
- Java: 1-50ms (even with G1GC)
- Unacceptable for HFT

### Solution: Object Pools

\`\`\`cpp
/*
 * Object pool for zero-allocation trading
 * Pre-allocate objects at startup, reuse
 */

template<typename T, size_t PoolSize = 10000>
class ObjectPool {
private:
    // Pre-allocated memory
    alignas(64) T pool[PoolSize];  // Cache-line aligned
    
    // Free list (bit vector)
    std::bitset<PoolSize> available;
    
    // Fast allocation index
    std::atomic<size_t> next_free{0};

public:
    ObjectPool() {
        // Mark all as available
        available.set();
    }
    
    T* acquire() {
        // Lock-free allocation
        for (size_t i = 0; i < PoolSize; ++i) {
            size_t idx = (next_free.fetch_add(1) + i) % PoolSize;
            
            // Try to claim
            bool expected = true;
            if (available[idx].compare_exchange_strong(expected, false)) {
                // Reset object state
                new (&pool[idx]) T();  // Placement new
                return &pool[idx];
            }
        }
        
        // Pool exhausted (should never happen in production)
        return nullptr;
    }
    
    void release(T* obj) {
        // Return to pool
        size_t idx = obj - pool;
        obj->~T();  // Destroy
        available[idx] = true;
    }
};

// Usage
ObjectPool<Order> order_pool;

void on_market_data() {
    // Acquire from pool (~10ns)
    Order* order = order_pool.acquire();
    
    // Use order
    order->symbol = "AAPL";
    order->quantity = 100;
    
    // Send to exchange...
    
    // Release back to pool
    order_pool.release(order);
}
\`\`\`

### Huge Pages

Reduce TLB (Translation Lookaside Buffer) misses:

\`\`\`bash
# Configure huge pages (2MB pages vs 4KB)
echo 1024 > /proc/sys/vm/nr_hugepages

# Use in application
void* mem = mmap(NULL, size, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);

# Result: ~30% latency reduction for memory-intensive ops
\`\`\`

---

## CPU and System Optimization

### CPU Pinning

\`\`\`cpp
/*
 * Pin threads to specific CPU cores
 * Prevents OS from moving threads around
 */

#include <pthread.h>
#include <sched.h>

void pin_thread_to_core(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    
    pthread_t thread = pthread_self();
    pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
}

int main() {
    // Pin market data thread to core 0
    std::thread market_data_thread([]() {
        pin_thread_to_core(0);
        process_market_data();
    });
    
    // Pin strategy thread to core 1
    std::thread strategy_thread([]() {
        pin_thread_to_core(1);
        run_strategy();
    });
    
    // Result: No context switches, ~20% latency reduction
}
\`\`\`

### Disable Hyper-Threading

\`\`\`bash
# Hyper-threading adds latency variance
# Disable in BIOS or:
echo 0 > /sys/devices/system/cpu/cpu1/online  # Disable HT core

# Result: More predictable latency
\`\`\`

### Real-Time Linux

\`\`\`bash
# Use PREEMPT_RT patch for real-time Linux
# Reduces max latency from ~1ms to ~50μs

# Set real-time priority
chrt -f 99 ./hft_strategy

# Isolate CPUs (no other processes)
isolcpus=0,1,2,3  # In kernel boot params

# Disable CPU frequency scaling
cpufreq-set -g performance

# Result: Deterministic <100μs latency
\`\`\`

---

## Co-Location Strategy

### Proximity to Exchange

\`\`\`
Your Server → Exchange Server

Same rack:     ~10μs
Same datacenter:  ~50μs
Same city:     ~200μs
Different city:   ~5ms

Optimal: Same rack as exchange matching engine
\`\`\`

### Network Topology

\`\`\`
                 Exchange
                     |
        +-----------+-----------+
        |           |           |
    Your Server  Your Server  Your Server
      (Primary)   (Backup)   (Backup)

- Multiple servers for redundancy
- Cross-connected for failover
- Direct fiber to exchange
\`\`\`

---

## Production HFT System

\`\`\`cpp
/*
 * Complete HFT System (C++)
 * Sub-10μs order-to-exchange
 */

class HFTEngine {
private:
    // Pre-allocated pools
    ObjectPool<Order, 10000> order_pool;
    ObjectPool<MarketData, 10000> market_data_pool;
    
    // Lock-free queues
    SPSCQueue<MarketData*> market_data_queue;  // Single producer, single consumer
    SPSCQueue<Order*> order_queue;
    
    // DPDK port
    uint16_t dpdk_port_id;
    
    // Statistics
    std::atomic<uint64_t> orders_sent{0};
    std::atomic<uint64_t> latency_sum_ns{0};

public:
    void run() {
        // Pin threads
        std::thread market_data_thread([this]() {
            pin_thread_to_core(0);
            this->market_data_loop();
        });
        
        std::thread strategy_thread([this]() {
            pin_thread_to_core(1);
            this->strategy_loop();
        });
        
        std::thread order_thread([this]() {
            pin_thread_to_core(2);
            this->order_loop();
        });
        
        market_data_thread.join();
        strategy_thread.join();
        order_thread.join();
    }
    
    void market_data_loop() {
        // Receive packets with DPDK
        while (running) {
            auto packets = dpdk_receive_burst();
            
            for (auto& pkt : packets) {
                uint64_t t0 = rdtsc();  // CPU timestamp counter
                
                // Parse market data (zero-copy)
                MarketData* md = market_data_pool.acquire();
                parse_market_data(pkt.data, md);
                
                // Push to strategy (lock-free)
                market_data_queue.push(md);
                
                uint64_t t1 = rdtsc();
                // Latency: ~200ns
            }
        }
    }
    
    void strategy_loop() {
        while (running) {
            // Poll for market data (no blocking)
            MarketData* md;
            if (market_data_queue.pop(md)) {
                uint64_t t0 = rdtsc();
                
                // Run strategy
                if (should_trade(md)) {
                    Order* order = order_pool.acquire();
                    generate_order(md, order);
                    order_queue.push(order);
                }
                
                market_data_pool.release(md);
                
                uint64_t t1 = rdtsc();
                // Latency: ~500ns
            }
        }
    }
    
    void order_loop() {
        while (running) {
            Order* order;
            if (order_queue.pop(order)) {
                uint64_t t0 = rdtsc();
                
                // Send order with DPDK
                dpdk_send_order(order);
                
                orders_sent++;
                
                uint64_t t1 = rdtsc();
                latency_sum_ns += (t1 - t0) * cycles_to_ns;
                
                order_pool.release(order);
                
                // Latency: ~1-2μs
            }
        }
    }
    
    bool should_trade(MarketData* md) {
        // Simple strategy: price above VWAP
        return md->price > md->vwap;
    }
};

// Total latency: 200ns + 500ns + 2000ns = ~2.7μs
// Add network latency (~5μs to exchange)
// Total: ~8μs order-to-exchange
\`\`\`

---

## Economics of HFT

### Cost-Benefit Analysis

**Infrastructure costs**:
- Co-location: $30K/month
- FPGA cards: $200K (one-time)
- High-speed network: $10K/month
- Development: $500K-2M/year
- **Total**: $1-3M/year

**Revenue potential**:
- Latency advantage: 1μs
- Value per trade: $0.01-0.10
- Trades per day: 100K-1M
- **Total revenue**: $1K-100K/day = $250K-25M/year

**Break-even**: 1-6 months if successful strategy

---

## Summary

HFT requires extreme optimization:

1. **FPGAs**: 100-500ns processing, deterministic, expensive
2. **Kernel bypass**: DPDK/OpenOnload, 2-10μs latency
3. **Memory pools**: Zero allocation, no GC pauses
4. **CPU pinning**: No context switches, dedicated cores
5. **Co-location**: Same rack as exchange, 10μs proximity
6. **Lock-free data structures**: Atomic operations only

**Reality check**: HFT is an arms race. Your 10μs advantage today is obsolete in 6 months. Only firms with $10M+ budgets and top-tier engineering compete successfully.

In the next section, we'll design distributed trading systems for global reach.
`,
};
