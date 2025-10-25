/**
 * Bloom Filters Section
 */

export const bloomfiltersSection = {
  id: 'bloom-filters',
  title: 'Bloom Filters',
  content: `Bloom filters are one of the most elegant probabilistic data structures in computer science. They solve a critical problem: **"How do we test set membership when the set is too large to fit in memory?"**

## The Problem Bloom Filters Solve

Imagine you're building a system that needs to check:
- Has this URL already been crawled? (Web crawler with billions of URLs)
- Is this username taken? (Social platform with 1 billion users)
- Has this email been seen before? (Email spam filter)
- Is this IP address malicious? (Firewall with millions of bad IPs)

**Naive approaches**:
- ❌ Hash table: Requires storing entire set (100GB+ for billion items)
- ❌ Database query: Too slow (100ms per query)
- ❌ Cache: Still need to store complete data

**Bloom filter approach**:
- ✅ **Space efficient**: 10-20 bits per element (vs 100+ bytes)
- ✅ **Fast**: O(k) lookups where k is small constant
- ✅ **Trade-off**: Allows false positives, no false negatives

---

## What is a Bloom Filter?

A **Bloom filter** is a probabilistic data structure that tests whether an element is a member of a set.

**Key properties**:
- **May return false positives**: Says "maybe in set"
- **Never returns false negatives**: Says "definitely not in set"
- **Space efficient**: Uses bits instead of storing actual elements
- **Cannot delete elements**: Once added, cannot remove (standard Bloom filter)

**Real-world analogy**: 
Think of a bouncer at a club with a fuzzy memory. If you ask "Has John been here?", the bouncer might say:
- "Definitely NOT" (100% accurate)
- "Maybe" (could be wrong)

This is acceptable for many use cases where occasional false positives are tolerable.

---

## How Bloom Filters Work

### Data Structure

A Bloom filter consists of:
1. **Bit array** of size \`m\` (all bits initially 0)
2. **k hash functions** (h₁, h₂, ..., hₖ)

\`\`\`
Bit array (m = 16 bits):
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15

Hash functions: 
h1(x), h2(x), h3(x)  (k = 3)
\`\`\`

### Insertion: add (element)

To add element "alice":
1. Apply each hash function: h₁("alice"), h₂("alice"), h₃("alice")
2. Set bits at those positions to 1

\`\`\`
h1("alice") = 3
h2("alice") = 7  
h3("alice") = 11

After inserting "alice":
[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]
          ↑           ↑           ↑
\`\`\`

Now add "bob":
\`\`\`
h1("bob") = 2
h2("bob") = 7   (collision with alice!)
h3("bob") = 14

After inserting "bob":
[0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0]
       ↑  ↑           ↑           ↑        ↑
\`\`\`

**Time complexity**: O(k) where k = number of hash functions

### Lookup: contains (element)

To check if element "alice" is in the set:
1. Apply each hash function
2. Check if ALL corresponding bits are 1

\`\`\`
contains("alice"):
h1("alice") = 3  → bit[3] = 1 ✓
h2("alice") = 7  → bit[7] = 1 ✓
h3("alice") = 11 → bit[11] = 1 ✓

All bits are 1 → "Probably present" ✓
\`\`\`

Check if "charlie" is in the set:
\`\`\`
contains("charlie"):
h1("charlie") = 5  → bit[5] = 0 ✗

At least one bit is 0 → "Definitely NOT present" ✓
\`\`\`

**Time complexity**: O(k)

### False Positives

Now check "dave":
\`\`\`
contains("dave"):
h1("dave") = 2  → bit[2] = 1 ✓ (set by bob)
h2("dave") = 7  → bit[7] = 1 ✓ (set by alice & bob)
h3("dave") = 3  → bit[3] = 1 ✓ (set by alice)

All bits are 1 → "Probably present"
But "dave" was NEVER inserted! → FALSE POSITIVE
\`\`\`

This is the trade-off: As more elements are added, more bits are set to 1, increasing collision probability.

---

## False Positive Probability

The false positive rate depends on:
- **m**: Number of bits in array
- **n**: Number of elements inserted
- **k**: Number of hash functions

**Formula**:
\`\`\`
False positive probability ≈ (1 - e^(-kn/m))^k
\`\`\`

**Intuition**:
- More bits (larger m) → Lower false positive rate
- More elements (larger n) → Higher false positive rate
- Optimal k ≈ (m/n) * ln(2) ≈ 0.693 * (m/n)

**Example**:
- m = 10,000 bits (1.25 KB)
- n = 1,000 elements
- k = 7 hash functions
- False positive rate: ~0.82% (very low!)

**Size comparison**:
- Bloom filter: 10 bits per element
- Hash table: 100+ bytes per element
- **Space savings: 80x smaller!**

---

## Optimal Parameters

### Choosing m (bits per element)

For desired false positive rate \`p\`:
\`\`\`
m = -n * ln (p) / (ln(2))²

Example: For 1% false positive rate (p = 0.01):
m ≈ 9.6 bits per element
\`\`\`

Common configurations:
- **1% FPR**: ~10 bits/element, k=7
- **0.1% FPR**: ~15 bits/element, k=10
- **0.01% FPR**: ~20 bits/element, k=14

### Choosing k (number of hashes)

\`\`\`
k = (m/n) * ln(2)

Too few hashes: Not enough randomization
Too many hashes: More computation, more collisions
\`\`\`

---

## Implementation Considerations

### Hash Functions

Need k independent hash functions. Options:

**Option 1: Multiple hash functions**
\`\`\`python
import hashlib

def hash1(item):
    return int (hashlib.md5(item).hexdigest(), 16) % m

def hash2(item):
    return int (hashlib.sha1(item).hexdigest(), 16) % m
\`\`\`

**Option 2: Double hashing (more efficient)**
\`\`\`python
def get_hashes (item, k, m):
    h1 = hash (item) % m
    h2 = hash (item + "salt") % m
    
    # Generate k hashes from 2 hash functions
    return [(h1 + i * h2) % m for i in range (k)]
\`\`\`

### Python Implementation

\`\`\`python
class BloomFilter:
    def __init__(self, size, num_hashes):
        self.size = size
        self.num_hashes = num_hashes
        self.bit_array = [0] * size
    
    def _get_hashes (self, item):
        """Generate k hash values using double hashing"""
        h1 = hash (item) % self.size
        h2 = hash (str (item) + "salt") % self.size
        
        return [(h1 + i * h2) % self.size 
                for i in range (self.num_hashes)]
    
    def add (self, item):
        """Add item to Bloom filter"""
        for index in self._get_hashes (item):
            self.bit_array[index] = 1
    
    def contains (self, item):
        """Check if item might be in set"""
        for index in self._get_hashes (item):
            if self.bit_array[index] == 0:
                return False  # Definitely not in set
        return True  # Probably in set

# Usage
bf = BloomFilter (size=1000, num_hashes=7)

bf.add("alice")
bf.add("bob")

print(bf.contains("alice"))    # True (definitely added)
print(bf.contains("charlie"))  # False (definitely NOT added)
print(bf.contains("dave"))     # Maybe False (false positive possible)
\`\`\`

---

## Real-World Use Cases

### 1. Google BigTable

**Problem**: Check if data exists before expensive disk lookup

**Solution**: Bloom filter in memory for each SSTable
- Query: "Is row key X in SSTable Y?"
- Bloom filter says "No" → Skip disk read (saves 100ms)
- Bloom filter says "Maybe" → Read from disk

**Result**: 75% reduction in unnecessary disk reads

### 2. Cassandra

**Similar to BigTable**: Bloom filter per SSTable
- 0.01% false positive rate (20 bits per key)
- Dramatic read performance improvement
- Configurable per table

### 3. Medium (Blogging Platform)

**Problem**: "Don't show articles user has already read"

**Traditional approach**: 
- Store read articles in database
- Query: \`SELECT ... WHERE user_id = ? AND article_id = ?\`
- Slow, expensive

**Bloom filter approach**:
- Maintain Bloom filter per user (in Redis)
- Check filter before querying database
- False positives: User might miss 1% of new articles (acceptable)

**Result**: 90% reduction in database queries

### 4. Chrome Safe Browsing

**Problem**: Check if URL is malicious (millions of bad URLs)

**Traditional**: Download entire malicious URL database (100MB+)

**Bloom filter**: 
- Download small Bloom filter (2MB)
- Check locally: "Is this URL malicious?"
- If "Maybe" → Query Google servers for confirmation
- If "No" → Definitely safe

**Result**: Fast, private, offline checking

### 5. Akamai CDN

**Problem**: Detect duplicate web objects efficiently

**Solution**: Bloom filter to avoid re-caching same content
- Check if object already cached
- Reduces cache pollution

### 6. Bitcoin

**Problem**: Check if transaction has been processed

**Solution**: Bloom filter for wallet synchronization
- Mobile wallets use Bloom filters
- Download only relevant transactions (privacy + efficiency)

---

## Variants of Bloom Filters

### Counting Bloom Filter

**Problem**: Standard Bloom filters cannot delete elements

**Solution**: Use counters instead of bits
\`\`\`
Standard:  [0, 1, 1, 0, 1]  (bits)
Counting:  [0, 2, 1, 0, 3]  (counters)
\`\`\`

**Operations**:
- Insert: Increment counters
- Delete: Decrement counters
- Query: Check if all counters > 0

**Trade-off**: Uses 4x more space (4 bits per counter vs 1 bit)

### Scalable Bloom Filter

**Problem**: Fixed size Bloom filter degrades as n increases

**Solution**: Add new Bloom filters as needed
\`\`\`
BF1 (full)  → BF2 (active) → BF3 (ready)
\`\`\`

**Query**: Check all filters (OR operation)

### Cuckoo Filter

**Modern alternative** to Bloom filters:
- ✅ Supports deletion
- ✅ Better space efficiency
- ✅ Better cache locality
- ❌ More complex implementation

---

## Bloom Filters vs Alternatives

| Approach | Space | Time | False Positives | Deletions |
|----------|-------|------|----------------|-----------|
| **Hash table** | O(n) | O(1) | No | Yes |
| **Sorted array** | O(n) | O(log n) | No | Yes |
| **Bloom filter** | O(1) | O(k) | Yes (~1%) | No |
| **Counting BF** | O(1) | O(k) | Yes | Yes |
| **Cuckoo filter** | O(1) | O(1) | Yes (<1%) | Yes |

---

## When to Use Bloom Filters

### ✅ Good Use Cases

**Reduce expensive operations**:
- Database queries (check cache first)
- Disk reads (check memory first)
- Network requests (check locally first)

**Large scale membership tests**:
- Billions of elements
- Memory constrained
- False positives acceptable

**Performance critical paths**:
- Web crawlers (duplicate URL detection)
- CDNs (cache deduplication)
- Rate limiting (check if user exceeded limit)

### ❌ Bad Use Cases

**Exact membership required**:
- Financial transactions (no false positives!)
- Security critical (authentication, authorization)
- Legal/compliance data

**Need to delete elements**:
- Use counting Bloom filter or cuckoo filter instead

**Small sets**:
- Hash table is simpler for < 10,000 elements

---

## Practical Guidelines

### 1. Sizing Your Bloom Filter

\`\`\`python
def calculate_bloom_filter_size (n, false_positive_rate):
    """
    n: expected number of elements
    false_positive_rate: desired FPR (e.g., 0.01 for 1%)
    """
    m = -(n * math.log (false_positive_rate)) / (math.log(2) ** 2)
    k = (m / n) * math.log(2)
    
    return int (m), int (k)

# Example: 1 million URLs, 1% FPR
m, k = calculate_bloom_filter_size(1_000_000, 0.01)
# m = 9,585,059 bits ≈ 1.2 MB
# k = 7 hash functions
\`\`\`

### 2. Monitoring False Positive Rate

\`\`\`python
class MonitoredBloomFilter(BloomFilter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_elements = 0
    
    def add (self, item):
        super().add (item)
        self.num_elements += 1
    
    def get_fill_ratio (self):
        """Percentage of bits set to 1"""
        return sum (self.bit_array) / len (self.bit_array)
    
    def estimated_fpr (self):
        """Estimate current false positive rate"""
        ratio = self.get_fill_ratio()
        return ratio ** self.num_hashes

# Monitor and rebuild if needed
if bf.estimated_fpr() > 0.05:  # 5% threshold
    # Rebuild with larger size
    new_bf = BloomFilter (size=bf.size * 2, num_hashes=bf.num_hashes)
\`\`\`

### 3. Distributed Bloom Filters

For distributed systems, share Bloom filters via:

**Redis**:
\`\`\`python
import redis

r = redis.Redis()

def add_to_redis_bloom (key, item, k, m):
    """Add item to Redis-backed Bloom filter"""
    for i in get_hashes (item, k, m):
        r.setbit (key, i, 1)

def check_redis_bloom (key, item, k, m):
    """Check if item in Redis Bloom filter"""
    for i in get_hashes (item, k, m):
        if not r.getbit (key, i):
            return False
    return True
\`\`\`

**Note**: Redis has native Bloom filter support via RedisBloom module.

---

## Interview Tips

### Key Points to Mention

1. **Trade-off**: Space efficiency for occasional false positives
2. **No false negatives**: If it says "not present", definitely not present
3. **Cannot delete**: Standard variant doesn't support deletions
4. **Use cases**: Cache filtering, duplicate detection, pre-filtering
5. **Real-world**: Google BigTable, Cassandra, Chrome, Bitcoin

### Common Interview Questions

**"How does a Bloom filter work?"**
- Explain bit array + k hash functions
- Walk through insert and lookup
- Explain false positive scenario

**"How do you size a Bloom filter?"**
- Depends on n (elements) and desired false positive rate
- Formula: m = -n ln (p) / (ln 2)²
- Common: 10 bits per element for 1% FPR

**"When would you use a Bloom filter?"**
- Large-scale duplicate detection
- Pre-filtering before expensive operations
- Memory-constrained environments
- When false positives acceptable

**"What are alternatives?"**
- Hash table (no false positives, more memory)
- Cuckoo filter (supports deletion, better performance)
- Count-min sketch (for frequency counting)

### Whiteboard Exercise

Design a web crawler that avoids re-crawling URLs:

\`\`\`
1. Initialize Bloom filter (10 billion URLs, 1% FPR)
   - Size: ~10 billion * 10 bits = 12.5 GB

2. For each URL:
   - Check Bloom filter
   - If "definitely not seen": crawl it, add to filter
   - If "probably seen": skip (1% false negatives = miss 1% of new URLs)

3. Periodically rebuild filter as it fills up

Trade-off: Use 12.5 GB instead of 1 TB+ for hash table
Acceptable: Missing 1% of URLs is fine for web crawler
\`\`\`

---

## Summary

**Bloom filters** are probabilistic data structures that efficiently test set membership with the trade-off of false positives.

**Key characteristics**:
- ✅ Space efficient: 10-20 bits per element
- ✅ Fast: O(k) constant time operations  
- ✅ No false negatives
- ❌ False positives (tunable rate)
- ❌ Cannot delete elements (standard version)

**Perfect for**:
- Pre-filtering expensive operations
- Large-scale duplicate detection
- Cache optimization
- Memory-constrained environments

**Used by**: Google BigTable, Cassandra, Chrome, Medium, Bitcoin, Akamai

When you need to answer "Is X in this set?" for billions of elements with limited memory, and occasional false positives are acceptable, **Bloom filters are the answer**.`,
};
