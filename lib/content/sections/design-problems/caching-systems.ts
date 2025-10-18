/**
 * Caching Systems Section
 */

export const cachingsystemsSection = {
  id: 'caching-systems',
  title: 'Caching Systems',
  content: `Caching is one of the most important concepts in computer systems. **Caches** store frequently accessed data in fast memory to avoid expensive recomputation or slow data fetches.

**Why Caching Matters:**
- **Speed**: RAM access is 100,000x faster than disk
- **Scale**: Reduces load on databases and APIs
- **Cost**: Cheaper to serve from cache than recompute
- **Real-world**: Every major system uses caching (CDNs, databases, CPUs)

---

## Cache Eviction Policies

When cache is full, which item do we remove? Different policies serve different use cases:

### 1. LRU (Least Recently Used) ⭐ Most Common

**Policy**: Evict the item that hasn't been accessed for the longest time.

**Intuition**: Recently used items are likely to be used again soon (temporal locality).

**Use Cases:**
- Web browser cache (recent pages likely revisited)
- Database query cache
- OS page cache
- Most general-purpose caching

**Implementation**: HashMap + Doubly LinkedList
- HashMap: O(1) lookup
- LinkedList: Maintain access order
- On access: Move to front (most recent)
- On eviction: Remove from back (least recent)

\`\`\`python
# LRU Cache Structure
# HashMap: key -> Node (fast lookup)
# LinkedList: Head (most recent) <-> ... <-> Tail (least recent)
#
# get(key):
#   1. Lookup in HashMap - O(1)
#   2. Move node to front - O(1)
#   3. Return value
#
# put(key, val):
#   1. If exists: update and move to front
#   2. If new: add to front
#   3. If over capacity: remove from back (LRU)
\`\`\`

**Time Complexity**: O(1) for get and put  
**Space Complexity**: O(capacity)

**Why Doubly Linked List?**
- Need to remove arbitrary nodes (when accessing them)
- Singly linked list would require O(N) to find previous node
- Doubly linked: node.prev available immediately -> O(1) removal

### 2. LFU (Least Frequently Used)

**Policy**: Evict the item with the lowest access frequency.

**Intuition**: Frequently used items are valuable, keep them longer.

**Use Cases:**
- Video streaming (popular videos cached)
- Content delivery networks
- When access patterns are skewed (80/20 rule)

**Implementation**: HashMap + Frequency buckets + LinkedLists
- HashMap: key -> node
- Frequency map: freq -> LinkedList of nodes with that freq
- Track minimum frequency for O(1) eviction

**Time Complexity**: O(1) for get and put (with good design)  
**Space Complexity**: O(capacity) + O(distinct frequencies)

**Challenge**: More complex than LRU, need to track frequencies efficiently.

### 3. FIFO (First-In-First-Out)

**Policy**: Evict the oldest item by insertion time (not access time).

**Intuition**: Simple, but doesn't consider access patterns.

**Use Cases:**
- Simple caches where LRU overhead not justified
- Queue-like scenarios

**Implementation**: Queue/CircularBuffer

### 4. LRU vs LFU Comparison

| Aspect | LRU | LFU |
|--------|-----|-----|
| **Evicts** | Least recently used | Least frequently used |
| **Temporal locality** | ✅ Excellent | ❌ Poor |
| **Hot items** | May evict if temporarily not accessed | ✅ Keeps popular items |
| **Cache pollution** | ❌ One scan can evict everything | ✅ Resistant |
| **Complexity** | Simple | More complex |
| **Use case** | General purpose | Skewed access patterns |

**Example showing difference:**
\`\`\`
Access pattern: A, B, C, D, E, A, B, C, D, E, ... (repeating)
Capacity: 3

LRU Cache:
- After A,B,C,D: Evicts A (LRU), cache = {B,C,D}
- Next A: Miss! Evicts B, cache = {C,D,A}
- Poor performance on this pattern (0% hit rate after initial)

LFU Cache:
- Tracks frequencies: A=10, B=10, C=10, D=1, E=1
- Keeps A,B,C (high frequency)
- D and E evicted (low frequency)
- Better for repeating patterns
\`\`\`

---

## LRU Cache Deep Dive

Let's fully understand the most important cache design:

### Why HashMap + Doubly LinkedList?

**Requirement**: O(1) for both get() and put()

**Can't use HashMap alone:**
- ✅ O(1) lookup
- ❌ Can't track LRU order efficiently

**Can't use LinkedList alone:**
- ✅ O(1) add/remove at ends
- ❌ O(N) to find arbitrary nodes

**Combined:**
- HashMap: key -> node reference (O(1) lookup)
- LinkedList: maintains LRU order
- When accessing key: Use HashMap to find node in O(1), move it to front in O(1)

### LRU Cache Operations Walkthrough

\`\`\`python
# Capacity = 3
# Operations: put(1,A), put(2,B), put(3,C), get(1), put(4,D)

# After put(1,A), put(2,B), put(3,C):
# LinkedList: 3 <-> 2 <-> 1  (3 is most recent)
# HashMap: {1: node1, 2: node2, 3: node3}

# get(1):
# 1. Find node1 via HashMap - O(1)
# 2. Remove node1 from current position - O(1)
#    LinkedList: 3 <-> 2
# 3. Move node1 to front - O(1)
#    LinkedList: 1 <-> 3 <-> 2
# Result: return A

# put(4,D):
# 1. Cache is full (3/3)
# 2. Evict LRU: Remove from back (node2) - O(1)
#    LinkedList: 1 <-> 3
#    HashMap: {1: node1, 3: node3}
# 3. Add new node to front - O(1)
#    LinkedList: 4 <-> 1 <-> 3
#    HashMap: {1: node1, 3: node3, 4: node4}
\`\`\`

**Key Insight**: Every operation that touches a node (get or put) makes it "most recent" -> move to front.

### Dummy Head and Tail Trick

**Problem**: Adding/removing at list ends requires null checks:
\`\`\`python
# Without dummy nodes:
if head is None:  # Empty list
    head = tail = new_node
elif ...:  # Many edge cases!
\`\`\`

**Solution**: Use dummy head and tail nodes:
\`\`\`python
# With dummy nodes:
head = Node(0, 0)  # Dummy
tail = Node(0, 0)  # Dummy
head.next = tail
tail.prev = head

# Now adding after head is always:
new_node.next = head.next
new_node.prev = head
head.next.prev = new_node
head.next = new_node
# No edge cases! Always works.
\`\`\`

This simplifies code and eliminates edge case bugs.

---

## Cache Performance Metrics

**Hit Rate**: % of requests served from cache
\`\`\`
Hit Rate = Cache Hits / Total Requests
\`\`\`

**Example**: 80% hit rate means cache serves 80% of requests, only 20% go to slow storage.

**Impact**:
- 0% hit rate: Cache useless, all requests slow
- 50% hit rate: 2x improvement (half of requests fast)
- 90% hit rate: 10x improvement (only 10% slow)
- 99% hit rate: 100x improvement

**Tip**: In interviews, mention hit rate when discussing cache effectiveness.

---

## Interview Tips for Caching Problems

1. **Start with requirements**: "What's the capacity? What's eviction policy?"

2. **State the two needs**: "Need O(1) lookup AND O(1) order tracking"

3. **Explain why each structure**: "HashMap for O(1) access, LinkedList for O(1) order updates"

4. **Draw it**: Sketch HashMap pointing to LinkedList nodes

5. **Test with example**: Walk through put and get operations

6. **Discuss alternatives**: "Could use OrderedDict in Python, but shows less understanding"

**Common Mistakes:**
- Using singly linked list (can't remove arbitrary nodes in O(1))
- Forgetting to update order on get() (not just put())
- Not handling capacity edge cases (empty, size 1)
- Memory leaks (forgetting to remove from HashMap when evicting)`,
};
