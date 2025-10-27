import { DiscussionQuestion } from '@/lib/types';

export const orderBookSimulatorQuiz: DiscussionQuestion[] = [
  {
    id: 'order-book-simulator-dq-1',
    question:
      'Optimize the order book matching engine to handle 100,000+ orders per second with sub-microsecond median latency. Analyze the current implementation and propose specific optimizations: (1) Data structure improvements (priority queues vs balanced trees vs hash maps), (2) Memory management strategies (object pooling, pre-allocation, cache-friendly layouts), (3) Lock-free concurrency techniques for multi-threaded matching, (4) CPU optimization (SIMD, cache locality, branch prediction), and (5) Profiling methodology to identify bottlenecks. Include code examples demonstrating 10x performance improvement and explain trade-offs between throughput and latency.',
    sampleAnswer: `Achieving 100K+ orders/second with sub-microsecond latency requires aggressive optimization across data structures, memory management, and CPU utilization. Here's a systematic approach to 10x performance improvement:

**1. Data Structure Analysis and Optimization**

**Current Implementation Issues:**
- Python dict + list: O(n) for finding best price, memory fragmentation
- Heap operations: O(log n) but constant factor overhead
- Frequent allocations/deallocations

**Optimized Approach: Custom Price-Level Book**

\`\`\`cpp
// C++ implementation for maximum performance
#include <unordered_map>
#include <list>
#include <memory>

struct Order {
    uint64_t order_id;
    double price;
    uint32_t quantity;
    uint64_t timestamp_ns;
    Order* next;  // Intrusive linked list (no allocations)
};

struct PriceLevel {
    double price;
    uint32_t total_quantity;
    Order* first_order;  // FIFO queue at this price
    Order* last_order;
    PriceLevel* next_better;  // Pointer to next better price (sorted)
    PriceLevel* next_worse;
};

class OptimizedOrderBook {
private:
    // Hash map: price -> PriceLevel (O(1) lookup)
    std::unordered_map<double, PriceLevel*> price_levels;
    
    // Best prices (O(1) access)
    PriceLevel* best_bid;
    PriceLevel* best_ask;
    
    // Order lookup (O(1) for cancels/modifies)
    std::unordered_map<uint64_t, Order*> order_lookup;
    
    // Memory pools (eliminate allocations)
    OrderPool order_pool;
    PriceLevelPool price_level_pool;

public:
    // Add order: O(1) average case
    void add_order(Order* order) {
        double price = order->price;
        
        // Find or create price level
        PriceLevel* level = get_or_create_price_level(price);
        
        // Add order to end of queue at this price (FIFO)
        if (level->last_order) {
            level->last_order->next = order;
        } else {
            level->first_order = order;
        }
        level->last_order = order;
        order->next = nullptr;
        
        level->total_quantity += order->quantity;
        
        // Update order lookup
        order_lookup[order->order_id] = order;
        
        // Update best bid/ask if needed
        update_best_prices(level);
    }
    
    // Match orders: O(1) for single fill
    Trade* match_order(Order* incoming) {
        if (incoming->side == Side::BUY) {
            // Match against best ask
            if (!best_ask) return nullptr;
            
            PriceLevel* level = best_ask;
            Order* resting = level->first_order;
            
            uint32_t match_qty = std::min(incoming->quantity, resting->quantity);
            
            // Create trade (from pool)
            Trade* trade = trade_pool.allocate();
            trade->price = level->price;
            trade->quantity = match_qty;
            trade->timestamp_ns = rdtsc();  // Use CPU timestamp counter
            
            // Update quantities
            incoming->quantity -= match_qty;
            resting->quantity -= match_qty;
            level->total_quantity -= match_qty;
            
            // Remove filled order
            if (resting->quantity == 0) {
                remove_order_from_level(resting, level);
            }
            
            return trade;
        }
        // ... similar for SELL side
    }
    
private:
    inline uint64_t rdtsc() {
        unsigned int lo, hi;
        __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
        return ((uint64_t)hi << 32) | lo;
    }
};
\`\`\`

**Performance Comparison:**

| Implementation | Add Order | Match Order | Cancel Order | Best Price |
|---------------|-----------|-------------|--------------|------------|
| Python (baseline) | 5 μs | 8 μs | 3 μs | 1 μs |
| C++ naive | 2 μs | 3 μs | 2 μs | 0.5 μs |
| C++ optimized (above) | 0.3 μs | 0.5 μs | 0.2 μs | 0.05 μs |

**Speedup: 16x for add, 16x for match, 15x for cancel**

**2. Memory Management Optimization**

**Problem:** Frequent malloc/free causes:
- Fragmentation (cache misses)
- Allocator overhead (~100ns per allocation)
- Unpredictable latency (GC pauses in managed languages)

**Solution: Object Pooling + Arena Allocation**

\`\`\`cpp
template<typename T, size_t POOL_SIZE = 1000000>
class ObjectPool {
private:
    T* pool;
    std::vector<T*> free_list;
    size_t allocated_count;

public:
    ObjectPool() {
        // Pre-allocate entire pool (huge page for better TLB hit rate)
        pool = (T*)aligned_alloc(2 * 1024 * 1024,  // 2MB alignment (huge page)
                                 POOL_SIZE * sizeof(T));
        
        // Initialize free list
        free_list.reserve(POOL_SIZE);
        for (size_t i = 0; i < POOL_SIZE; ++i) {
            free_list.push_back(&pool[i]);
        }
        
        allocated_count = 0;
    }
    
    // Allocate: O(1), no system call
    T* allocate() {
        if (free_list.empty()) {
            // Pool exhausted - fall back to malloc (rare)
            return new T();
        }
        
        T* obj = free_list.back();
        free_list.pop_back();
        allocated_count++;
        
        // Placement new (construct in pre-allocated memory)
        return new(obj) T();
    }
    
    // Deallocate: O(1), no system call
    void deallocate(T* obj) {
        obj->~T();  // Call destructor
        free_list.push_back(obj);
        allocated_count--;
    }
    
    size_t get_allocated_count() const { return allocated_count; }
};

// Usage
OrderPool order_pool;
Order* order = order_pool.allocate();  // ~5ns vs ~100ns for malloc
// ... use order ...
order_pool.deallocate(order);  // ~5ns vs ~100ns for free
\`\`\`

**Cache-Friendly Layout:**

\`\`\`cpp
// BAD: Scattered memory (cache misses)
struct Order {
    uint64_t order_id;
    double price;
    uint32_t quantity;
    char* client_id;  // Pointer to heap (cache miss!)
    Order* next;
};

// GOOD: Contiguous memory (cache friendly)
struct Order {
    uint64_t order_id;
    double price;
    uint32_t quantity;
    uint64_t timestamp_ns;
    Order* next;
    char client_id[16];  // Inline (no pointer chase)
    uint8_t side;
    uint8_t padding[7];  // Align to 64 bytes (cache line size)
} __attribute__((aligned(64)));
\`\`\`

**Memory Layout Impact:**
- Scattered: 3-5 cache misses per order access (~300ns)
- Contiguous: 1 cache miss per order access (~100ns)
- **Speedup: 3-5x**

**3. Lock-Free Concurrency**

**Problem:** Traditional locks add 50-200ns latency per critical section.

**Solution: Lock-Free Single-Writer, Multiple-Reader**

\`\`\`cpp
#include <atomic>

class LockFreeOrderBook {
private:
    // Atomic pointer to current order book state
    std::atomic<OrderBookState*> current_state;
    
public:
    // Writer thread (matching engine)
    void add_order_lockfree(Order* order) {
        OrderBookState* old_state = current_state.load(std::memory_order_acquire);
        
        // Create new state (copy-on-write)
        OrderBookState* new_state = clone_state(old_state);
        new_state->add_order(order);
        
        // Atomic swap
        while (!current_state.compare_exchange_weak(old_state, new_state,
                                                    std::memory_order_release,
                                                    std::memory_order_acquire)) {
            // Retry if state changed (rare in single-writer scenario)
            delete new_state;
            old_state = current_state.load(std::memory_order_acquire);
            new_state = clone_state(old_state);
            new_state->add_order(order);
        }
        
        // Defer deletion of old state (RCU)
        schedule_deletion(old_state);
    }
    
    // Reader threads (market data subscribers)
    OrderBookState* get_snapshot() {
        return current_state.load(std::memory_order_acquire);
    }
};
\`\`\`

**Alternative: Single-Threaded Matching (Best for Ultra-Low Latency)**

Instead of multi-threading, use:
- Single matching thread pinned to isolated CPU core
- Lock-free queues for input/output
- No context switches, no locks

**Latency Comparison:**
- Multi-threaded with locks: 2-5 μs (lock contention)
- Lock-free: 1-2 μs (atomic operations)
- Single-threaded: 0.5-1 μs (no synchronization)

**4. CPU Optimization Techniques**

**A. SIMD (Single Instruction, Multiple Data)**

Process multiple orders in parallel:

\`\`\`cpp
#include <immintrin.h>  // AVX2 intrinsics

// Check 8 orders for match eligibility in parallel
void check_orders_simd(Order* orders[8], double price_threshold, bool* results) {
    // Load 8 prices into 256-bit register
    __m256d prices = _mm256_set_pd(
        orders[0]->price, orders[1]->price, orders[2]->price, orders[3]->price,
        orders[4]->price, orders[5]->price, orders[6]->price, orders[7]->price
    );
    
    __m256d thresholds = _mm256_set1_pd(price_threshold);
    
    // Compare all 8 prices at once
    __m256d comparison = _mm256_cmp_pd(prices, thresholds, _CMP_LE_OQ);
    
    // Extract results
    int mask = _mm256_movemask_pd(comparison);
    for (int i = 0; i < 8; ++i) {
        results[i] = (mask & (1 << i)) != 0;
    }
}

// Speedup: 8x for this operation
\`\`\`

**B. Cache Locality**

**Prefetching:**
\`\`\`cpp
void prefetch_next_order(Order* order) {
    __builtin_prefetch(order->next, 0, 3);  // Prefetch to L1 cache
}

// Use in tight loop
Order* current = level->first_order;
while (current) {
    if (current->next) {
        prefetch_next_order(current->next);  // Prefetch while processing current
    }
    process_order(current);
    current = current->next;
}
\`\`\`

**C. Branch Prediction**

\`\`\`cpp
// BAD: Unpredictable branches
if (order->side == Side::BUY) {
    match_against_asks(order);
} else {
    match_against_bids(order);
}

// GOOD: Use function pointers (no branch)
typedef void (*MatchFunction)(Order*);
MatchFunction matchers[2] = {match_against_asks, match_against_bids};
matchers[order->side](order);  // No branch, just indirect call

// BEST: Separate code paths entirely
void process_buy_orders(Order* orders[], size_t count) {
    for (size_t i = 0; i < count; ++i) {
        match_against_asks(orders[i]);  // No branches in loop
    }
}
\`\`\`

**5. Profiling Methodology**

**A. Linux Perf**

\`\`\`bash
# Profile CPU cycles
perf record -e cycles -g ./matching_engine

# Analyze hotspots
perf report

# Check cache misses
perf stat -e cache-references,cache-misses ./matching_engine

# Branch prediction
perf stat -e branches,branch-misses ./matching_engine
\`\`\`

**B. Intel VTune**

\`\`\`bash
# Hotspot analysis
vtune -collect hotspots -result-dir vtune_results ./matching_engine

# Memory access analysis
vtune -collect memory-access -result-dir vtune_results ./matching_engine
\`\`\`

**C. Custom Instrumentation**

\`\`\`cpp
class LatencyTracker {
    std::vector<uint64_t> samples;
    
public:
    void record_latency(uint64_t start_tsc, uint64_t end_tsc) {
        samples.push_back(end_tsc - start_tsc);
    }
    
    void print_statistics() {
        std::sort(samples.begin(), samples.end());
        
        size_t p50 = samples.size() / 2;
        size_t p99 = samples.size() * 99 / 100;
        size_t p999 = samples.size() * 999 / 1000;
        
        double tsc_to_ns = 1000000000.0 / get_tsc_frequency();
        
        printf("Latency (ns):\\n");
        printf("  P50:  %.0f\\n", samples[p50] * tsc_to_ns);
        printf("  P99:  %.0f\\n", samples[p99] * tsc_to_ns);
        printf("  P999: %.0f\\n", samples[p999] * tsc_to_ns);
    }
};
\`\`\`

**Bottleneck Identification Results:**

| Bottleneck | % Time (Before) | Optimization | % Time (After) |
|------------|-----------------|--------------|----------------|
| malloc/free | 35% | Object pooling | 2% |
| Cache misses | 25% | Contiguous layout | 8% |
| Lock contention | 15% | Lock-free/single-thread | 0% |
| Hash map lookups | 10% | Inline/prefetch | 5% |
| Branch mispredicts | 8% | Remove branches | 2% |
| Useful work | 7% | Same | 83% |

**6. Performance Results**

**Baseline (Python):**
- Throughput: 10,000 orders/second
- Median latency: 5 μs
- P99 latency: 50 μs

**Optimized (C++ with all techniques):**
- Throughput: 150,000 orders/second (**15x improvement**)
- Median latency: 0.4 μs (**12.5x improvement**)
- P99 latency: 1.2 μs (**42x improvement**)

**7. Throughput vs. Latency Trade-offs**

**Batching for Throughput:**
- Process orders in batches of 100
- Amortize fixed costs (lock acquisition, cache warming)
- **Result:** 200K orders/sec, but 20 μs median latency (batch delay)

**Single-Order for Latency:**
- Process each order immediately
- No batching delays
- **Result:** 100K orders/sec, but 0.4 μs median latency

**Hybrid Approach:**
- Small batches (10 orders) with 10 μs timeout
- Balance throughput and latency
- **Result:** 150K orders/sec, 2 μs median latency

**Recommendation:** Single-order for HFT market making, batching for exchange infrastructure.

This systematic optimization achieves 10-15x performance improvement through data structure redesign, memory pooling, cache optimization, and elimination of synchronization overhead.`,
    sampleAnswer: `Achieving 100K+ orders/second with sub-microsecond latency requires aggressive optimization across data structures, memory management, and CPU utilization. Here's a systematic approach to 10x performance improvement:

(Full expanded answer above - continuing with next question...)`,
  },
  {
    id: 'order-book-simulator-dq-2',
    question:
      'Design a comprehensive market data distribution system for your order book simulator that supports: (1) Level 1/2/3 data feeds with different subscription tiers, (2) Multicast UDP for lowest-latency distribution, (3) Guaranteed reliable TCP fallback for dropped packets, (4) Conflation strategies to reduce bandwidth during high volatility, and (5) Historical tick data replay for backtesting. Explain the trade-offs between UDP (fast, unreliable) and TCP (reliable, slower), how to implement sequence numbers for gap detection, and architecture for supporting 10,000+ simultaneous subscribers without impacting matching engine performance.',
    sampleAnswer: `Market data distribution must balance ultra-low latency with reliability and scalability. Here's a production-grade architecture:

**1. System Architecture**

\`\`\`
                    ┌─────────────────┐
                    │ Matching Engine │ (Primary system)
                    └────────┬────────┘
                             │ Market events
                             ↓
                    ┌─────────────────┐
                    │  Market Data    │
                    │  Publisher      │ (Separate process/server)
                    └────────┬────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
                ↓                         ↓
        ┌──────────────┐          ┌──────────────┐
        │ UDP Multicast│          │  TCP Server  │
        │  (Primary)   │          │  (Recovery)  │
        └──────┬───────┘          └──────┬───────┘
               │                         │
       ┌───────┴───────┬─────────────────┴─────────┐
       │               │                           │
       ↓               ↓                           ↓
[Level 1 Feed]  [Level 2 Feed]              [Level 3 Feed]
(NBBO only)     (Top 10 levels)             (Full book)
\`\`\`

**Key Design Principles:**
- Matching engine completely isolated (no subscriber impact)
- Publisher handles all distribution (separate CPU cores)
- UDP for speed, TCP for recovery (dual path)
- Tiered subscriptions (pay more for more data)

**2. Level 1/2/3 Data Feed Tiers**

\`\`\`python
from enum import Enum
from dataclasses import dataclass

class FeedLevel(Enum):
    LEVEL_1 = "L1"  # Best bid/ask only
    LEVEL_2 = "L2"  # Top 10 price levels
    LEVEL_3 = "L3"  # Full order book

@dataclass
class MarketDataMessage:
    """Market data message structure"""
    sequence_number: int  # Monotonic sequence (gap detection)
    timestamp_ns: int  # Exchange timestamp
    symbol: str
    message_type: str  # UPDATE, TRADE, NBBO
    level: FeedLevel
    data: dict  # Level-specific data

class MarketDataPublisher:
    """Publish market data to subscribers"""
    
    def __init__(self):
        self.sequence_number = 0
        self.subscribers_by_level = {
            FeedLevel.LEVEL_1: [],
            FeedLevel.LEVEL_2: [],
            FeedLevel.LEVEL_3: []
        }
        
        # UDP multicast addresses (one per level)
        self.multicast_addresses = {
            FeedLevel.LEVEL_1: ('239.1.1.1', 5000),  # Multicast group
            FeedLevel.LEVEL_2: ('239.1.1.2', 5001),
            FeedLevel.LEVEL_3: ('239.1.1.3', 5002)
        }
    
    def publish_nbbo_update(self, symbol: str, bid: float, ask: float):
        """Publish Level 1 data (NBBO)"""
        self.sequence_number += 1
        
        message = MarketDataMessage(
            sequence_number=self.sequence_number,
            timestamp_ns=time.perf_counter_ns(),
            symbol=symbol,
            message_type='NBBO',
            level=FeedLevel.LEVEL_1,
            data={'bid': bid, 'ask': ask}
        )
        
        # Send via UDP multicast
        self.send_udp_multicast(message, FeedLevel.LEVEL_1)
        
        # Store for TCP recovery
        self.message_history.append(message)
    
    def publish_order_book_update(self, symbol: str, order_book: Dict):
        """Publish Level 2 data (top 10 levels)"""
        self.sequence_number += 1
        
        # Extract top 10 bids and asks
        top_bids = sorted(order_book['bids'].items(), reverse=True)[:10]
        top_asks = sorted(order_book['asks'].items())[:10]
        
        message = MarketDataMessage(
            sequence_number=self.sequence_number,
            timestamp_ns=time.perf_counter_ns(),
            symbol=symbol,
            message_type='ORDER_BOOK',
            level=FeedLevel.LEVEL_2,
            data={'bids': top_bids, 'asks': top_asks}
        )
        
        self.send_udp_multicast(message, FeedLevel.LEVEL_2)
        self.message_history.append(message)
    
    def publish_full_order_book(self, symbol: str, all_orders: List[Order]):
        """Publish Level 3 data (full order book with order IDs)"""
        self.sequence_number += 1
        
        message = MarketDataMessage(
            sequence_number=self.sequence_number,
            timestamp_ns=time.perf_counter_ns(),
            symbol=symbol,
            message_type='FULL_BOOK',
            level=FeedLevel.LEVEL_3,
            data={'orders': [o.to_dict() for o in all_orders]}
        )
        
        self.send_udp_multicast(message, FeedLevel.LEVEL_3)
        self.message_history.append(message)

**Subscription Pricing:**
- Level 1: $500/month (retail traders, basic algos)
- Level 2: $2,000/month (quantitative traders, market makers)
- Level 3: $10,000/month (HFT firms, full transparency)

**3. UDP Multicast Implementation**

**Why UDP?**
- **Latency:** 50-200 μs vs TCP 500-2000 μs
- **Broadcast:** Single packet reaches all subscribers (efficient)
- **No handshaking:** No connection setup overhead

**Why Multicast?**
- Scales to 10,000+ subscribers with single packet
- Network handles fan-out (not publisher)
- Conserves bandwidth

\`\`\`python
import socket
import struct

class UDPMulticastSender:
    """Send market data via UDP multicast"""
    
    def __init__(self, multicast_group: str, port: int):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Set TTL (time-to-live) for multicast
        ttl = struct.pack('b', 10)  # 10 hops (datacenter)
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)
        
        self.multicast_address = (multicast_group, port)
    
    def send_message(self, message: MarketDataMessage):
        """Send message via UDP multicast"""
        # Serialize message (binary format for efficiency)
        data = self.serialize(message)
        
        # Send (non-blocking)
        self.sock.sendto(data, self.multicast_address)
        
        # NO acknowledgment (fire-and-forget)
    
    def serialize(self, message: MarketDataMessage) -> bytes:
        """Serialize to binary (faster than JSON)"""
        # Use Protocol Buffers, FlatBuffers, or custom binary
        import struct
        
        # Example: Fixed-size header + variable data
        header = struct.pack(
            'Q Q 8s H',  # 8-byte seq, 8-byte timestamp, 8-char symbol, 2-byte type
            message.sequence_number,
            message.timestamp_ns,
            message.symbol.encode('ascii'),
            ord(message.message_type[0])
        )
        
        # Serialize data portion (implementation specific)
        data_bytes = json.dumps(message.data).encode('utf-8')
        
        return header + data_bytes

class UDPMulticastReceiver:
    """Receive market data via UDP multicast"""
    
    def __init__(self, multicast_group: str, port: int):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Bind to port
        self.sock.bind(('', port))
        
        # Join multicast group
        mreq = struct.pack('4sL', socket.inet_aton(multicast_group), socket.INADDR_ANY)
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        
        self.last_sequence_number = 0
    
    def receive_message(self) -> Optional[MarketDataMessage]:
        """Receive message (non-blocking)"""
        try:
            data, address = self.sock.recvfrom(65507)  # Max UDP packet size
            
            message = self.deserialize(data)
            
            # Gap detection
            if message.sequence_number != self.last_sequence_number + 1:
                gap_start = self.last_sequence_number + 1
                gap_end = message.sequence_number - 1
                
                print(f"GAP DETECTED: sequences {gap_start}-{gap_end} missing")
                
                # Request recovery via TCP
                self.request_gap_fill(gap_start, gap_end)
            
            self.last_sequence_number = message.sequence_number
            
            return message
            
        except BlockingIOError:
            return None  # No data available

**4. TCP Fallback for Reliability**

**When TCP is Used:**
- Gap detection (missed UDP packets)
- Initial snapshot download (subscriber connects)
- Slow joiner recovery (lagging subscriber)

\`\`\`python
class TCPRecoveryServer:
    """TCP server for gap-fill requests"""
    
    def __init__(self, message_history: List[MarketDataMessage]):
        self.message_history = message_history
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(('0.0.0.0', 6000))
        self.server_socket.listen(100)
    
    def handle_gap_fill_request(self, client_socket, request):
        """Handle request for missing sequences"""
        # Request format: "GAP <start_seq> <end_seq>"
        parts = request.decode('utf-8').split()
        start_seq = int(parts[1])
        end_seq = int(parts[2])
        
        # Find messages in range
        missing_messages = [
            msg for msg in self.message_history
            if start_seq <= msg.sequence_number <= end_seq
        ]
        
        # Send via TCP (reliable)
        for msg in missing_messages:
            data = self.serialize(msg)
            client_socket.sendall(data)
        
        client_socket.close()

**UDP vs TCP Trade-offs:**

| Aspect | UDP Multicast | TCP |
|--------|---------------|-----|
| **Latency** | 50-200 μs | 500-2000 μs |
| **Reliability** | Best-effort (drops packets) | Guaranteed delivery |
| **Ordering** | No guarantee | In-order delivery |
| **Scalability** | Excellent (10K+ subscribers, 1 packet) | Poor (10K connections) |
| **Bandwidth** | Efficient (broadcast) | High (unicast to each) |
| **Use Case** | Real-time feed (primary) | Recovery, snapshots (fallback) |

**5. Sequence Numbers and Gap Detection**

**Sequence Number Strategy:**

\`\`\`python
class SequenceNumberManager:
    """Manage sequence numbers for gap detection"""
    
    def __init__(self):
        self.current_sequence = 0
        self.lock = threading.Lock()
    
    def get_next_sequence(self) -> int:
        """Thread-safe sequence number generation"""
        with self.lock:
            self.current_sequence += 1
            return self.current_sequence

class GapDetector:
    """Detect gaps in received sequences"""
    
    def __init__(self):
        self.expected_sequence = 1
        self.gaps = []
    
    def check_sequence(self, received_sequence: int) -> Optional[tuple]:
        """Check for gap, return (start, end) if gap detected"""
        if received_sequence == self.expected_sequence:
            self.expected_sequence += 1
            return None
        elif received_sequence > self.expected_sequence:
            gap = (self.expected_sequence, received_sequence - 1)
            self.gaps.append(gap)
            self.expected_sequence = received_sequence + 1
            return gap
        else:
            # Duplicate or out-of-order (ignore)
            return None

**6. Conflation (Bandwidth Reduction)**

**Problem:** During high volatility, order book updates can exceed 100,000/second, overwhelming slow subscribers.

**Solution: Conflation (Aggregate updates)**

\`\`\`python
class ConflationEngine:
    """Conflate updates to reduce bandwidth"""
    
    def __init__(self, conflation_window_ms: int = 10):
        self.conflation_window_ms = conflation_window_ms
        self.pending_updates = {}
        self.last_send_time = {}
    
    def add_update(self, symbol: str, update: Dict):
        """Add update to conflation buffer"""
        # Only keep latest update per symbol in window
        self.pending_updates[symbol] = update
        
        # Check if window elapsed
        now = time.time() * 1000
        last_send = self.last_send_time.get(symbol, 0)
        
        if now - last_send >= self.conflation_window_ms:
            self.flush_updates(symbol)
    
    def flush_updates(self, symbol: str):
        """Send conflated update"""
        if symbol in self.pending_updates:
            update = self.pending_updates.pop(symbol)
            self.send_update(update)
            self.last_send_time[symbol] = time.time() * 1000

**Conflation Impact:**
- Without: 100,000 updates/sec × 1KB = 100 MB/s per subscriber
- With (10ms window): 10,000 updates/sec × 1KB = 10 MB/s per subscriber
- **Bandwidth reduction: 10x**

**Trade-off:** 10ms additional latency (acceptable for non-HFT subscribers)

**7. Supporting 10,000+ Subscribers**

**Architecture for Scale:**

\`\`\`python
class ScalableMarketDataSystem:
    """Market data system supporting 10K+ subscribers"""
    
    def __init__(self):
        # Separate publisher process (isolated from matching engine)
        self.publisher_process = self.spawn_publisher_process()
        
        # Shared memory for matching engine -> publisher communication
        self.shared_memory = SharedMemoryRingBuffer(size_mb=100)
        
        # UDP multicast handles fan-out (network infrastructure)
        self.udp_senders = {
            FeedLevel.LEVEL_1: UDPMulticastSender('239.1.1.1', 5000),
            FeedLevel.LEVEL_2: UDPMulticastSender('239.1.1.2', 5001),
            FeedLevel.LEVEL_3: UDPMulticastSender('239.1.1.3', 5002)
        }
        
        # TCP recovery servers (multiple for load distribution)
        self.tcp_recovery_servers = [
            TCPRecoveryServer(port=6000 + i) for i in range(10)
        ]
    
    def matching_engine_callback(self, event: MarketEvent):
        """Called by matching engine when market changes"""
        # Write event to shared memory (lock-free, wait-free)
        self.shared_memory.write(event)
        
        # NO direct subscriber interaction (maintains matching engine performance)
    
    def publisher_main_loop(self):
        """Publisher process main loop (runs on separate CPU core)"""
        while True:
            # Read event from shared memory
            event = self.shared_memory.read()
            
            if event:
                # Generate market data messages for each level
                self.generate_and_send_messages(event)

**Key Scalability Techniques:**1. **Matching engine isolation:** Zero subscriber impact (separate process/cores)
2. **UDP multicast:** Network handles fan-out, not publisher
3. **CPU pinning:** Publisher on dedicated cores (no context switches)
4. **Zero-copy:** Shared memory avoids serialization overhead
5. **Load balancing:** Multiple TCP recovery servers (round-robin)

**8. Historical Tick Data Replay**

\`\`\`python
class HistoricalDataReplayer:
    """Replay historical tick data for backtesting"""
    
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.messages = self.load_messages()
    
    def load_messages(self) -> List[MarketDataMessage]:
        """Load historical messages from file"""
        # Messages stored in compressed binary format
        import gzip, pickle
        
        with gzip.open(self.data_file, 'rb') as f:
            messages = pickle.load(f)
        
        return messages
    
    def replay(self, speed_multiplier: float = 1.0):
        """Replay messages at real-time or faster"""
        start_time = time.perf_counter()
        first_msg_time = self.messages[0].timestamp_ns
        
        for msg in self.messages:
            # Calculate when this message should be sent
            msg_offset_ns = msg.timestamp_ns - first_msg_time
            target_time = start_time + (msg_offset_ns / 1e9) / speed_multiplier
            
            # Wait until target time
            while time.perf_counter() < target_time:
                time.sleep(0.0001)  # 100 μs sleep
            
            # Send message
            self.send_message(msg)

**Replay Use Cases:**
- Backtesting trading strategies
- Stress testing subscribers
- Training machine learning models
- Debugging market data issues

This architecture achieves <200μs UDP latency, supports 10,000+ subscribers via multicast, provides TCP fallback for reliability, and scales independently of matching engine performance.`,
  },
  {
    id: 'order-book-simulator-dq-3',
    question:
      'Implement comprehensive pre-trade and post-trade risk management for the order book simulator. Your system must: (1) Enforce position limits and prevent traders from exceeding maximum exposure, (2) Implement dynamic margin requirements that adjust during volatility, (3) Detect wash trading and self-matching (same participant on both sides), (4) Monitor for layering/spoofing patterns in real-time, and (5) Provide circuit breakers that halt trading when price moves exceed thresholds. Describe the architecture for real-time risk checks that add <10 microseconds to execution latency, how to integrate with the matching engine without blocking, and regulatory reporting requirements (CAT, MiFID II).',
    sampleAnswer: `Production risk management must protect market integrity without sacrificing performance. Here's a comprehensive implementation:

**1. Real-Time Risk Check Architecture (<10μs Latency)**

**Challenge:** Risk checks must be fast enough to not impact matching engine performance.

**Solution: Pre-computed Risk State + Fast Lookups**

\`\`\`cpp
struct TraderRiskState {
    // Position tracking (updated incrementally)
    int64_t net_position;  // Positive = long, negative = short
    int64_t position_limit;
    
    // Margin tracking
    double used_margin;
    double available_margin;
    double margin_requirement_multiplier;  // Adjusted for volatility
    
    // Order flow metrics (for spoofing detection)
    uint32_t orders_placed_last_minute;
    uint32_t orders_canceled_last_minute;
    uint32_t trades_last_minute;
    
    // Flags
    bool trading_halted;
    uint64_t last_update_timestamp_ns;
} __attribute__((aligned(64)));  // Cache-line aligned

class FastRiskChecker {
private:
    // Hash map: trader_id -> risk state (O(1) lookup)
    std::unordered_map<uint64_t, TraderRiskState*> trader_states;
    
    // Memory pool for risk states
    ObjectPool<TraderRiskState> risk_state_pool;
    
public:
    // Pre-trade risk check (TARGET: <5μs)
    bool check_order(uint64_t trader_id, Order* order) {
        uint64_t start_tsc = __rdtsc();
        
        // O(1) lookup
        TraderRiskState* state = trader_states[trader_id];
        
        if (state->trading_halted) {
            return false;  // Trader halted
        }
        
        // Check 1: Position limit (simple integer comparison)
        int64_t new_position = state->net_position + 
                              (order->side == Side::BUY ? order->quantity : -order->quantity);
        
        if (abs(new_position) > state->position_limit) {
            return false;  // Would exceed position limit
        }
        
        // Check 2: Margin requirement
        double order_margin = calculate_margin_fast(order, state->margin_requirement_multiplier);
        
        if (state->used_margin + order_margin > state->available_margin) {
            return false;  // Insufficient margin
        }
        
        // Check 3: Order rate limit (simple counter check)
        if (state->orders_placed_last_minute > 1000) {
            return false;  // Too many orders (potential spoofing)
        }
        
        uint64_t end_tsc = __rdtsc();
        uint64_t cycles = end_tsc - start_tsc;
        
        // On 3.5 GHz CPU: 3,500 cycles = 1 μs
        // Target: <15,000 cycles (<5 μs)
        
        return true;  // All checks passed
    }
    
    inline double calculate_margin_fast(Order* order, double multiplier) {
        // Simple margin calculation (no complex math)
        return order->price * order->quantity * multiplier;
    }
    
    // Post-trade update (executed after match, non-blocking)
    void update_position(uint64_t trader_id, Trade* trade) {
        TraderRiskState* state = trader_states[trader_id];
        
        // Update position
        if (trade->buy_trader_id == trader_id) {
            state->net_position += trade->quantity;
        } else {
            state->net_position -= trade->quantity;
        }
        
        // Update margin
        state->used_margin = calculate_used_margin(state);
        
        state->last_update_timestamp_ns = __rdtsc();
    }
};
\`\`\`

**Performance: 3-5 μs per risk check** (well under 10μs target)

**2. Dynamic Margin Requirements (Volatility-Adjusted)**

**Concept:** Increase margin during high volatility to protect against larger losses.

\`\`\`python
class DynamicMarginCalculator:
    """Calculate margin requirements adjusted for volatility"""
    
    def __init__(self):
        self.base_margin_rate = 0.05  # 5% base margin
        self.volatility_multipliers = {
            'low': 1.0,      # Normal market
            'medium': 1.5,   # Elevated volatility
            'high': 2.5,     # High volatility
            'extreme': 5.0   # Market stress
        }
        
        self.current_volatility_regime = 'low'
    
    def update_volatility_regime(self, symbol: str):
        """Update volatility regime based on recent price moves"""
        # Calculate realized volatility (standard deviation of returns)
        recent_prices = self.get_recent_prices(symbol, minutes=30)
        returns = np.diff(np.log(recent_prices))
        realized_vol_annual = np.std(returns) * np.sqrt(252 * 390)  # Annualized
        
        # Classify regime
        if realized_vol_annual < 0.15:
            self.current_volatility_regime = 'low'
        elif realized_vol_annual < 0.30:
            self.current_volatility_regime = 'medium'
        elif realized_vol_annual < 0.60:
            self.current_volatility_regime = 'high'
        else:
            self.current_volatility_regime = 'extreme'
        
        # Update all trader margin requirements
        multiplier = self.volatility_multipliers[self.current_volatility_regime]
        self.update_all_traders_margin(multiplier)
    
    def calculate_margin(self, order: Order) -> float:
        """Calculate margin requirement for order"""
        multiplier = self.volatility_multipliers[self.current_volatility_regime]
        notional = order.price * order.quantity
        return notional * self.base_margin_rate * multiplier

**Example:**
- Normal market (vol=15%): 5% margin on $100K position = $5K
- High volatility (vol=45%): 12.5% margin on $100K position = $12.5K

**3. Wash Trading and Self-Matching Detection**

\`\`\`python
class WashTradeDetector:
    """Detect wash trading and self-matching"""
    
    def pre_match_check(self, buy_order: Order, sell_order: Order) -> bool:
        """Check if match would be wash trade (before execution)"""
        
        # Check 1: Same trader on both sides (direct wash trade)
        if buy_order.trader_id == sell_order.trader_id:
            self.log_wash_trade_attempt({
                'type': 'SELF_MATCH',
                'trader_id': buy_order.trader_id,
                'buy_order': buy_order.order_id,
                'sell_order': sell_order.order_id,
                'severity': 'CRITICAL'
            })
            
            # REJECT match
            return False
        
        # Check 2: Same parent entity (indirect wash trade)
        buyer_entity = self.get_parent_entity(buy_order.trader_id)
        seller_entity = self.get_parent_entity(sell_order.trader_id)
        
        if buyer_entity == seller_entity:
            self.log_wash_trade_attempt({
                'type': 'ENTITY_WASH_TRADE',
                'entity': buyer_entity,
                'buy_trader': buy_order.trader_id,
                'sell_trader': sell_order.trader_id,
                'severity': 'HIGH'
            })
            
            # REJECT match
            return False
        
        # Check 3: Coordinated trading pattern (statistical detection)
        connection_score = self.check_trader_connection(
            buy_order.trader_id, 
            sell_order.trader_id
        )
        
        if connection_score > 0.9:  # Very high likelihood of coordination
            self.log_wash_trade_attempt({
                'type': 'COORDINATED_WASH_TRADE',
                'connection_score': connection_score,
                'buy_trader': buy_order.trader_id,
                'sell_trader': sell_order.trader_id,
                'severity': 'MEDIUM'
            })
            
            # Flag for review (don't auto-reject, but investigate)
            self.flag_for_review(buy_order, sell_order)
        
        return True  # Allow match
    
    def check_trader_connection(self, trader1: int, trader2: int) -> float:
        """Calculate likelihood that two traders are coordinating"""
        
        # Get historical trading between these two
        historical_trades = self.get_historical_trades(trader1, trader2, days=30)
        
        if not historical_trades:
            return 0.0
        
        # Metrics indicating coordination:
        # 1. High volume concentration (trade mostly with each other)
        trader1_total_volume = self.get_total_volume(trader1, days=30)
        pair_volume = sum(t['quantity'] for t in historical_trades)
        volume_concentration = pair_volume / trader1_total_volume
        
        # 2. Always on opposite sides (never compete)
        opposite_side_rate = sum(1 for t in historical_trades 
                                if (t['buyer_id'] == trader1) != (t['buyer_id'] == trader2)) / len(historical_trades)
        
        # 3. Similar timing patterns
        time_diffs_ms = [abs(t['buy_time'] - t['sell_time']).total_seconds() * 1000 
                        for t in historical_trades]
        avg_time_diff = np.mean(time_diffs_ms)
        timing_score = 1.0 if avg_time_diff < 100 else 0.0  # <100ms suggests coordination
        
        # Combine scores
        connection_score = (volume_concentration * 0.5 + 
                          opposite_side_rate * 0.3 + 
                          timing_score * 0.2)
        
        return connection_score

**Self-Matching Prevention:**
- **Pre-trade check:** Block order if would match against own order
- **Latency cost:** <1 μs (simple ID comparison)
- **Regulatory requirement:** Reg NMS, MiFID II both prohibit

**4. Layering/Spoofing Detection**

**Real-Time Monitoring:**

\`\`\`python
class RealTimeSpoofingDetector:
    """Detect spoofing patterns in real-time"""
    
    def on_order_placed(self, order: Order):
        """Track order placement for spoofing detection"""
        trader_id = order.trader_id
        
        # Update metrics
        self.trader_metrics[trader_id]['orders_placed'] += 1
        self.trader_metrics[trader_id]['recent_orders'].append(order)
        
        # Keep only last 1 minute of orders
        cutoff = time.time() - 60
        self.trader_metrics[trader_id]['recent_orders'] = [
            o for o in self.trader_metrics[trader_id]['recent_orders']
            if o.timestamp > cutoff
        ]
        
        # Check for spoofing pattern
        if self.detect_spoofing_pattern(trader_id):
            self.handle_spoofing_alert(trader_id)
    
    def detect_spoofing_pattern(self, trader_id: int) -> bool:
        """Detect if trader's recent activity matches spoofing pattern"""
        metrics = self.trader_metrics[trader_id]
        
        recent_orders = metrics['recent_orders']
        
        if len(recent_orders) < 10:
            return False  # Need sufficient data
        
        # Calculate metrics
        total_orders = len(recent_orders)
        canceled_orders = len([o for o in recent_orders if o.status == 'CANCELED'])
        large_orders = [o for o in recent_orders if o.quantity > 5000]
        large_canceled = [o for o in large_orders if o.status == 'CANCELED']
        
        cancel_rate = canceled_orders / total_orders
        large_cancel_rate = len(large_canceled) / max(len(large_orders), 1)
        
        # Fast cancels (< 500ms)
        fast_cancels = [
            o for o in large_canceled
            if (o.cancel_time - o.place_time).total_seconds() < 0.5
        ]
        
        # Spoofing indicators
        if (cancel_rate > 0.80 and  # >80% cancellation rate
            large_cancel_rate > 0.90 and  # >90% large orders canceled
            len(fast_cancels) > 5):  # 5+ fast cancels
            return True
        
        return False
    
    def handle_spoofing_alert(self, trader_id: int):
        """Handle detected spoofing"""
        # Auto-pause trader
        self.risk_checker.halt_trader(trader_id)
        
        # Alert compliance
        self.send_alert({
            'alert_type': 'SPOOFING_DETECTED',
            'trader_id': trader_id,
            'severity': 'CRITICAL',
            'action_taken': 'TRADING_HALTED',
            'metrics': self.trader_metrics[trader_id]
        })
        
        # Regulatory report
        self.generate_sar_report(trader_id, 'SPOOFING')

**5. Circuit Breakers (Price Movement Limits)**

\`\`\`python
class CircuitBreakerSystem:
    """Halt trading when price moves exceed thresholds"""
    
    def __init__(self):
        # LULD (Limit Up Limit Down) bands
        self.tier1_threshold = 0.05  # 5% for stocks > $3
        self.tier2_threshold = 0.10  # 10% for stocks $0.75-$3
        
        self.reference_prices = {}  # symbol -> reference price
        self.trading_halted = set()  # Set of halted symbols
    
    def check_price_movement(self, symbol: str, trade_price: float) -> bool:
        """Check if trade would trigger circuit breaker"""
        
        ref_price = self.reference_prices.get(symbol)
        if not ref_price:
            # Initialize reference price
            self.reference_prices[symbol] = trade_price
            return True
        
        # Calculate price change
        price_change_pct = abs(trade_price - ref_price) / ref_price
        
        # Determine threshold
        threshold = (self.tier1_threshold if ref_price > 3.0 
                    else self.tier2_threshold)
        
        # Check breach
        if price_change_pct > threshold:
            self.trigger_circuit_breaker(symbol, trade_price, ref_price, price_change_pct)
            return False  # Block trade
        
        return True  # Allow trade
    
    def trigger_circuit_breaker(self, symbol: str, price: float, 
                                ref_price: float, change_pct: float):
        """Trigger circuit breaker (halt trading)"""
        print(f"[CIRCUIT BREAKER] {symbol}: {change_pct*100:.1f}% move")
        print(f"  Reference: \${ref_price:.2f}, Current: \${price:.2f}")
        
        # Halt trading
        self.trading_halted.add(symbol)
        
        # Cancel all orders in this symbol
        self.matching_engine.cancel_all_orders(symbol)
        
        # Hold for 5 seconds (LULD rule)
        threading.Timer(5.0, self.resume_trading, args=[symbol]).start()
    
    def resume_trading(self, symbol: str):
        """Resume trading after circuit breaker pause"""
        self.trading_halted.remove(symbol)
        print(f"[CIRCUIT BREAKER] {symbol}: Trading resumed")

**6. Integration with Matching Engine**

**Non-Blocking Architecture:**

\`\`\`
Order Submission Flow:
1. Order arrives
2. Risk check (5 μs)
   - Position limit check
   - Margin check
   - Rate limit check
3. If approved → Matching engine
4. Trade execution
5. Post-trade update (async, non-blocking)
   - Update position
   - Update margin
   - Log for audit
\`\`\`

**Key: Risk check is synchronous (must pass before matching), but post-trade updates are asynchronous (don't block matching engine).**

**7. Regulatory Reporting**

**CAT (US) Reporting:**
\`\`\`python
def generate_cat_report(self, trade: Trade):
    """Generate CAT report for US regulators"""
    return {
        'reportingID': self.firm_cat_id,
        'tradeDate': trade.timestamp.strftime('%Y%m%d'),
        'tradeTime': trade.timestamp.strftime('%H%M%S%f')[:-3],  # Microseconds
        'eventType': 'MEOT',  # Matched Execution
        'orderID': trade.buy_order_id,
        'symbol': trade.symbol,
        'quantity': trade.quantity,
        'price': trade.price,
        'side': 'B' if trade.aggressor_side == 'BUY' else 'S',
        'firmDesignatedID': trade.trader_id
    }

**MiFID II (EU) Reporting:**
\`\`\`python
def generate_mifid_report(self, trade: Trade):
    """Generate MiFID II transaction report"""
    return {
        'tradingVenue': 'XLON',  # Exchange code
        'instrumentISIN': trade.isin,
        'buyerLEI': self.get_trader_lei(trade.buyer_id),
        'sellerLEI': self.get_trader_lei(trade.seller_id),
        'quantity': trade.quantity,
        'price': trade.price,
        'tradingDateTime': trade.timestamp.isoformat() + 'Z',  # Microsecond precision
        'tradingCapacity': 'PRIN'  # Principal
    }

This comprehensive risk management system maintains <10μs latency for pre-trade checks while providing robust protection against wash trading, spoofing, and excessive risk-taking, with full regulatory compliance.`,
  },
];
