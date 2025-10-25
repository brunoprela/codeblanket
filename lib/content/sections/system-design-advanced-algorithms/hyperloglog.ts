/**
 * HyperLogLog Section
 */

export const hyperloglogSection = {
  id: 'hyperloglog',
  title: 'HyperLogLog',
  content: `HyperLogLog is a probabilistic algorithm that solves a fundamental problem in distributed systems: **"How do we count unique elements (cardinality) in massive datasets using minimal memory?"**

## The Cardinality Estimation Problem

Imagine you need to answer:
- How many unique visitors did our website get today? (Analytics)
- How many distinct IP addresses hit our API? (Security)
- How many unique users opened our email campaign? (Marketing)
- How many distinct search queries were made? (Search engine)

**Naive approach**: Store all seen elements in a set

\`\`\`python
unique_visitors = set()

for visitor in visitors:
    unique_visitors.add (visitor)

count = len (unique_visitors)
\`\`\`

**Problem**: For 1 billion unique visitors:
- Set size: 1B × 16 bytes (UUID) = 16 GB memory
- Too expensive to track in memory
- Doesn't scale across distributed systems

**HyperLogLog solution**: Count 1 billion unique elements using only **1.5 KB**!

---

## What is HyperLogLog?

**HyperLogLog** is a probabilistic algorithm that estimates cardinality with:
- ✅ **Tiny memory**: 1.5 KB for billions of elements (0.81% standard error)
- ✅ **Fast**: O(1) add and count operations
- ✅ **Mergeable**: Combine multiple HyperLogLogs (distributed counting)
- ❌ **Approximate**: ~2% error (acceptable for most use cases)
- ❌ **Cannot list elements**: Only counts, cannot retrieve actual values

**Key insight**: Don't store elements, store *statistics* about their hashes.

---

## The Intuition Behind HyperLogLog

### The Coin Flip Analogy

Imagine flipping a fair coin:
- Probability of 1 head: 1/2
- Probability of 2 heads in a row: 1/4
- Probability of 3 heads in a row: 1/8
- Probability of N heads in a row: 1/2^N

**If you see 10 heads in a row**, you can estimate you've done **~2^10 = 1024 flips**.

**HyperLogLog does similar with hashes**:
1. Hash each element (produces random bits)
2. Count leading zeros in binary representation
3. Maximum leading zeros ≈ log₂(cardinality)

### Example

\`\`\`
Element: "user123"
Hash: 0x00ABC123  (binary: 0000000010101011...)
Leading zeros: 7

Element: "user456"  
Hash: 0x12345678  (binary: 00010010001101...)
Leading zeros: 3

Maximum leading zeros seen: 7
Estimated cardinality: 2^7 = 128 elements
\`\`\`

**Problem with naive approach**: High variance (one hash with many zeros skews estimate).

**HyperLogLog solution**: Use multiple "buckets" and harmonic mean to reduce variance.

---

## How HyperLogLog Works

### Step 1: Hash the Element

\`\`\`python
import hashlib

def hash_element (element):
    """Hash to 64-bit integer"""
    h = hashlib.sha256(str (element).encode())
    return int (h.hexdigest()[:16], 16)

hash_value = hash_element("user123")
# Example: 0x1A2B3C4D5E6F7890
\`\`\`

### Step 2: Split Hash into Bucket Index + Remainder

\`\`\`
Hash (64 bits): |-- 14 bits --|-- 50 bits --|
                  bucket_index    remainder

With 14 bits: 2^14 = 16,384 buckets (registers)
\`\`\`

### Step 3: Count Leading Zeros in Remainder

\`\`\`python
def count_leading_zeros (value):
    """Count leading zeros in binary representation"""
    if value == 0:
        return 64  # Max
    return (value.bit_length() - 1).bit_length()

# Example: remainder = 0x0000000012345678
# Binary: 00000000000000000001...
# Leading zeros: 15
\`\`\`

### Step 4: Update Register

\`\`\`python
registers = [0] * 16384  # 16K buckets

def add (element):
    hash_value = hash_element (element)
    
    # Split hash
    bucket = hash_value & 0x3FFF  # Last 14 bits
    remainder = hash_value >> 14  # Remaining bits
    
    # Count leading zeros
    leading_zeros = count_leading_zeros (remainder)
    
    # Keep maximum per bucket
    registers[bucket] = max (registers[bucket], leading_zeros)
\`\`\`

### Step 5: Estimate Cardinality

\`\`\`python
def estimate_cardinality():
    m = len (registers)  # Number of buckets
    alpha = 0.7213 / (1 + 1.079 / m)  # Bias correction
    
    # Harmonic mean of 2^register_values
    raw_estimate = alpha * m * m / sum(2**(-reg) for reg in registers)
    
    # Small/large range corrections (omitted for simplicity)
    return int (raw_estimate)
\`\`\`

**Why harmonic mean?** Resists outliers (one bucket with large value doesn't skew estimate).

---

## Complete Implementation

\`\`\`python
import hashlib
import math

class HyperLogLog:
    def __init__(self, precision=14):
        """
        precision: Number of bits for bucket index (4-16)
        - More buckets = better accuracy, more memory
        - 14 bits = 16K buckets = 16 KB memory
        """
        self.precision = precision
        self.m = 2 ** precision  # Number of buckets
        self.registers = [0] * self.m
        
        # Bias correction constant
        if self.m >= 128:
            self.alpha = 0.7213 / (1 + 1.079 / self.m)
        elif self.m >= 64:
            self.alpha = 0.709
        elif self.m >= 32:
            self.alpha = 0.697
        else:
            self.alpha = 0.673
    
    def _hash (self, element):
        """64-bit hash"""
        h = hashlib.sha256(str (element).encode())
        return int (h.hexdigest()[:16], 16)
    
    def _leading_zeros (self, value, max_width=50):
        """Count leading zeros in binary representation"""
        if value == 0:
            return max_width + 1
        
        return max_width - value.bit_length() + 1
    
    def add (self, element):
        """Add element to HyperLogLog"""
        # Hash element
        hash_value = self._hash (element)
        
        # Extract bucket index (first \`precision\` bits)
        bucket = hash_value & ((1 << self.precision) - 1)
        
        # Extract remainder (remaining bits)
        remainder = hash_value >> self.precision
        
        # Count leading zeros in remainder
        leading_zeros = self._leading_zeros (remainder)
        
        # Update register (keep maximum)
        self.registers[bucket] = max (self.registers[bucket], leading_zeros)
    
    def count (self):
        """Estimate cardinality"""
        # Calculate raw estimate (harmonic mean)
        raw_estimate = self.alpha * self.m * self.m / sum(
            2**(-register) for register in self.registers
        )
        
        # Small range correction
        if raw_estimate <= 2.5 * self.m:
            # Count zero registers
            zeros = self.registers.count(0)
            if zeros != 0:
                return self.m * math.log (self.m / float (zeros))
        
        # Large range correction
        if raw_estimate > (1/30) * (2**32):
            return -2**32 * math.log(1 - raw_estimate / 2**32)
        
        return int (raw_estimate)
    
    def merge (self, other):
        """Merge another HyperLogLog (for distributed counting)"""
        if self.m != other.m:
            raise ValueError("Cannot merge HLLs with different precision")
        
        for i in range (self.m):
            self.registers[i] = max (self.registers[i], other.registers[i])

# Usage
hll = HyperLogLog (precision=14)

# Add elements
for i in range(10000):
    hll.add (f"user_{i}")

estimated_count = hll.count()
actual_count = 10000
error = abs (estimated_count - actual_count) / actual_count * 100

print(f"Actual: {actual_count}")
print(f"Estimated: {estimated_count}")
print(f"Error: {error:.2f}%")
# Typical output: Error: ~1-2%
\`\`\`

---

## Accuracy and Memory Trade-offs

### Precision vs Memory vs Error

\`\`\`
Precision (bits) | Buckets (m) | Memory   | Std Error
----------------|-------------|----------|----------
10              | 1,024       | 1 KB     | 3.25%
12              | 4,096       | 4 KB     | 1.63%
14              | 16,384      | 16 KB    | 0.81%
16              | 65,536      | 64 KB    | 0.41%
\`\`\`

**Formula**: Standard error ≈ 1.04 / √m

**Typical choice**: Precision = 14 (16 KB, 0.81% error)

### Memory Comparison

For 1 billion unique elements:

\`\`\`
Exact counting (set):  16 GB
HyperLogLog (p=14):    16 KB

Space savings: 1,000,000x
\`\`\`

---

## Real-World Use Cases

### 1. Redis

**PFADD, PFCOUNT commands**:
\`\`\`redis
PFADD unique_visitors user:123 user:456 user:789
PFCOUNT unique_visitors
# Returns: approximate count

# Merge multiple HyperLogLogs
PFMERGE result hll1 hll2 hll3
PFCOUNT result
\`\`\`

**Use cases**:
- Unique visitors per day
- Distinct IP addresses
- Unique events per user

### 2. Google BigQuery

**APPROX_COUNT_DISTINCT**:
\`\`\`sql
SELECT APPROX_COUNT_DISTINCT(user_id) as unique_users
FROM events
WHERE date = '2024-01-01'

-- Much faster than COUNT(DISTINCT user_id)
-- ~1-2% error acceptable for analytics
\`\`\`

**Benefits**:
- 1000x faster than exact counting
- Scales to trillions of rows
- Distributed aggregation

### 3. Facebook

**Counting unique users per feature**:
- Daily active users (DAU)
- Weekly active users (WAU)  
- Unique content viewers

**Distributed HyperLogLog**:
- Each server maintains local HLL
- Periodically merge HLLs
- Global unique count with minimal data transfer

### 4. Amazon (DynamoDB, CloudWatch)

**Metrics aggregation**:
- Unique API callers
- Distinct error types
- Unique resources accessed

### 5. Presto / Apache Druid

**Fast approximate queries**:
\`\`\`sql
SELECT approx_distinct (user_id) FROM events
-- Returns in milliseconds vs minutes for exact count
\`\`\`

---

## Merging HyperLogLogs (Distributed Counting)

**Key property**: HyperLogLogs are mergeable!

\`\`\`python
# Server 1: Counts unique visitors in US
hll_us = HyperLogLog()
for visitor in us_visitors:
    hll_us.add (visitor)

# Server 2: Counts unique visitors in EU  
hll_eu = HyperLogLog()
for visitor in eu_visitors:
    hll_eu.add (visitor)

# Merge (take maximum of each register)
hll_global = HyperLogLog()
hll_global.merge (hll_us)
hll_global.merge (hll_eu)

global_unique_visitors = hll_global.count()
\`\`\`

**Merge operation**:
\`\`\`python
def merge (hll1, hll2):
    for i in range (len (hll1.registers)):
        hll1.registers[i] = max (hll1.registers[i], hll2.registers[i])
\`\`\`

**Benefits**:
- Combine counts from multiple servers
- Minimal data transfer (only 16 KB per HLL)
- Accurate global unique count

---

## HyperLogLog vs Alternatives

| Approach | Memory | Accuracy | Mergeable | Use Case |
|----------|--------|----------|-----------|----------|
| **Exact set** | O(N) | 100% | ❌ | Small datasets |
| **Probabilistic Counting** | O(log log N) | ~30% error | ❌ | Research |
| **Linear Counting** | O(N/10) | ~2% error | ❌ | Medium datasets |
| **HyperLogLog** | O(1) | ~1% error | ✅ | Large scale |
| **HyperLogLog++** | O(1) | ~0.5% error | ✅ | Production (Google) |

**HyperLogLog dominates** for large-scale cardinality estimation.

---

## Advanced: HyperLogLog++

**Improvements over HyperLogLog**:
1. **Better accuracy for small cardinalities** (0-1000 elements)
2. **Sparse representation** (uses less memory when few elements)
3. **Improved bias correction**

**Used by**: Google (BigQuery, Analytics)

**Sparse mode**:
\`\`\`
Small cardinalities: Store only non-zero registers
  - 100 unique elements: ~800 bytes
  - Automatically upgrades to dense mode at threshold
\`\`\`

---

## Limitations and Considerations

### Cannot Retrieve Elements

\`\`\`python
hll.add("user123")
hll.add("user456")

# Can count
count = hll.count()  # ✅ Works

# Cannot retrieve
users = hll.get_elements()  # ❌ Not possible
"user123" in hll  # ❌ Cannot test membership
\`\`\`

**Solution**: If you need both counting and membership testing, use Bloom filter + HyperLogLog.

### Deterministic Hashing

**Problem**: Same element must always hash to same value

\`\`\`python
hll.add("user123")  # Hash: 0x1A2B3C...
hll.add("user123")  # Must produce SAME hash

# Otherwise: double counting
\`\`\`

**Solution**: Use cryptographic hash (SHA-256, MurmurHash)

### Not for Exact Counts

\`\`\`
Use HyperLogLog when:
✅ ~1-2% error acceptable (analytics, monitoring)
✅ Massive scale (billions of elements)
✅ Memory constrained

Do NOT use when:
❌ Need exact count (financial transactions)
❌ Small datasets (<10,000 elements)
❌ Need to retrieve elements
\`\`\`

---

## Implementation Tips

### Choosing Precision

\`\`\`python
# Conservative (0.41% error, 64 KB)
hll = HyperLogLog (precision=16)

# Balanced (0.81% error, 16 KB)
hll = HyperLogLog (precision=14)  # Most common

# Memory-constrained (1.63% error, 4 KB)
hll = HyperLogLog (precision=12)
\`\`\`

### Serialization (for storage/transfer)

\`\`\`python
import pickle

# Serialize
serialized = pickle.dumps (hll.registers)

# Deserialize
registers = pickle.loads (serialized)
hll_restored = HyperLogLog (precision=14)
hll_restored.registers = registers
\`\`\`

**Better**: Use compact binary format (1 byte per register)

---

## Interview Tips

### Key Talking Points

1. **Problem**: Count unique elements in billions of items
2. **Solution**: Probabilistic algorithm using hash statistics
3. **Memory**: Constant O(1), typically 16 KB for 0.81% error
4. **Trade-off**: ~1-2% error for 1,000,000x memory savings
5. **Mergeable**: Combine HLLs from multiple servers
6. **Real-world**: Redis PFCOUNT, BigQuery APPROX_COUNT_DISTINCT

### Common Interview Questions

**"How does HyperLogLog achieve such small memory usage?"**
- Doesn't store elements, stores statistics (leading zeros)
- Uses harmonic mean of 2^leading_zeros across buckets
- Leading zeros ≈ log₂(cardinality)
- Fixed number of buckets (e.g., 16,384) regardless of data size

**"What's the error rate?"**
- Standard error ≈ 1.04 / √m where m = buckets
- With 16K buckets: 0.81% error
- Practical: ~1-2% error for most datasets
- Trade-off: More buckets = better accuracy, more memory

**"When would you NOT use HyperLogLog?"**
- Need exact count (financial systems)
- Small datasets (overhead not worth it)
- Need to retrieve elements (use set or Bloom filter)
- Need membership testing (use Bloom filter)

**"How do you merge HyperLogLogs?"**
- Take maximum of each register across HLLs
- registers_merged[i] = max (hll1.registers[i], hll2.registers[i])
- Enables distributed counting with minimal data transfer

### Design Exercise

Design a system to track daily active users (DAU) for Twitter:

\`\`\`
Requirement: Count unique users who tweet/view per day
Scale: 500M users, 1B daily events

Solution:
1. Each server maintains HyperLogLog (precision=14, 16 KB)
2. Add user_id to HLL on every event
3. Every hour, servers send HLL to aggregator (16 KB each)
4. Aggregator merges HLLs (take max of registers)
5. Final count with ~1% accuracy

Benefits:
- Memory: 16 KB per server vs 8 GB for exact count
- Network: Transfer 16 KB per server per hour
- Fast: O(1) add, O(1) merge
- Scalable: Linear with number of servers

Alternative (exact counting):
- Would require distributed set or database
- Much more complex, expensive, slower
- 1% error acceptable for DAU metric
\`\`\`

---

## Summary

**HyperLogLog** is a probabilistic algorithm that estimates cardinality (unique element count) using minimal memory.

**Key principles**:
- ✅ **Constant memory**: 16 KB for billions of elements (0.81% error)
- ✅ **Fast**: O(1) add and count
- ✅ **Mergeable**: Combine distributed HLLs
- ✅ **Accurate**: ~1-2% error (acceptable for analytics)
- ❌ **Approximate**: Not for exact counts
- ❌ **Cannot retrieve elements**: Only counts

**Industry adoption**: Redis (PFCOUNT), Google BigQuery, Facebook, Amazon, Presto, Druid—every large-scale analytics system uses HyperLogLog.

**Perfect for**: Analytics, monitoring, metrics where ~1% error is acceptable and exact counting is too expensive.

Understanding HyperLogLog is **essential** for designing scalable analytics and monitoring systems.`,
};
