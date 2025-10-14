import { Module } from '@/lib/types';

export const designProblemsModule: Module = {
  id: 'design-problems',
  title: 'Design Problems',
  description:
    'Master data structure and system design problems - LRU Cache, Min Stack, Rate Limiters, and more',
  icon: '🏗️',
  timeComplexity: 'Varies by problem - typically O(1) for core operations',
  spaceComplexity: 'O(N) where N is capacity or number of elements',
  sections: [
    {
      id: 'introduction',
      title: 'Introduction to Design Problems',
      content: `Design problems are a major category in technical interviews where you're asked to **implement a data structure or system** with specific requirements and constraints. Unlike algorithm problems, design problems focus on:

- **API design**: What methods should the class expose?
- **Data structure selection**: Which structures give the required time complexity?
- **Trade-offs**: Memory vs speed, simplicity vs performance
- **Real-world systems**: Caches, rate limiters, social media features

**Why Design Problems Matter:**
- **20-30% of FAANG interviews** include at least one design problem
- Test **engineering judgment** and system thinking
- Distinguish senior from junior candidates
- Directly applicable to real-world engineering

**Common Design Problem Types:**

1. **Caching Systems** (LRU Cache, LFU Cache)
   - Eviction policies
   - Fast lookup + order tracking
   - Most frequently asked

2. **Data Structure Implementations** (Min Stack, Queue using Stacks)
   - Constrained operations
   - Maintaining extra properties efficiently

3. **Rate Limiting** (Hit Counter, Rate Limiter)
   - Time-based constraints
   - Sliding windows

4. **Application Features** (Browser History, Twitter Timeline)
   - Real system components
   - Multiple operations interaction

---

## How to Approach Design Problems

### Step 1: Clarify Requirements 🎯
Ask clarifying questions:
- What operations are needed? (read, write, delete)
- What are the constraints? (capacity limits, time limits)
- What's the expected performance? (O(1)? O(log N)?)
- How should edge cases be handled? (empty, full, invalid)

**Example - LRU Cache:**
- Q: "What should get() return if key doesn't exist?"
- A: "Return -1"
- Q: "What happens when we put() beyond capacity?"
- A: "Evict the least recently used item"

### Step 2: Choose Data Structures 🧱

Match requirements to data structures:

| Requirement | Data Structure | Why |
|-------------|----------------|-----|
| O(1) lookup | HashMap | Direct key access |
| Order tracking | LinkedList / Deque | Efficient add/remove at ends |
| Min/Max in O(1) | Additional stack or variable | Track extremes |
| Fast sorted access | Heap / TreeMap | Maintain ordering |
| LRU ordering | HashMap + Doubly LinkedList | Fast access + order |

**Key Insight:** Design problems often require **combining 2+ data structures**.

### Step 3: Design the API 📝

Define clear method signatures:
\`\`\`python
class LRUCache:
    def __init__(self, capacity: int):
        """Initialize with given capacity"""
        pass
    
    def get(self, key: int) -> int:
        """Get value, return -1 if not exists"""
        pass
    
    def put(self, key: int, value: int) -> None:
        """Put key-value, evict LRU if full"""
        pass
\`\`\`

### Step 4: Implement and Test ✅

- Write clean, modular code
- Handle edge cases (empty, full, single element)
- Test thoroughly with examples
- Analyze time and space complexity

### Step 5: Discuss Trade-offs ⚖️

Be ready to discuss:
- "Could we use a different data structure?"
- "What if we had different constraints?"
- "How would this scale?"

---

## Common Patterns

### Pattern 1: HashMap + LinkedList
**When:** Need O(1) access AND maintain order (LRU Cache, LFU Cache)

**Why it works:**
- HashMap: O(1) lookup by key
- Doubly LinkedList: O(1) move to front/back, O(1) removal
- Together: Fast access + order tracking

**Example:**
\`\`\`python
class Node:
    def __init__(self, key, val):
        self.key, self.val = key, val
        self.prev = self.next = None

class LRUCache:
    def __init__(self, capacity):
        self.cache = {}  # key -> node
        # Dummy head/tail for easy insertion
        self.head = Node(0, 0)
        self.tail = Node(0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head
        self.capacity = capacity
\`\`\`

### Pattern 2: Stack with Extra Tracking
**When:** Need stack operations + additional property (Min Stack, Max Stack)

**Why it works:**
- Main stack: Regular push/pop
- Extra stack/variable: Track min/max at each level

**Example:**
\`\`\`python
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []  # Track min at each level
    
    def push(self, val):
        self.stack.append(val)
        # Push current min (could be this val)
        min_val = min(val, self.min_stack[-1] if self.min_stack else val)
        self.min_stack.append(min_val)
\`\`\`

### Pattern 3: Two Stacks/Queues
**When:** Implement one structure using another (Queue using Stacks)

**Why it works:**
- Reverse LIFO to FIFO (or vice versa)
- Amortized O(1) with lazy transfer

**Example:**
\`\`\`python
class QueueUsingStacks:
    def __init__(self):
        self.stack1 = []  # For enqueue
        self.stack2 = []  # For dequeue
    
    def enqueue(self, x):
        self.stack1.append(x)
    
    def dequeue(self):
        if not self.stack2:
            # Move all from stack1 to stack2 (reverses order)
            while self.stack1:
                self.stack2.append(self.stack1.pop())
        return self.stack2.pop()
\`\`\`

### Pattern 4: Sliding Window with Deque
**When:** Time-based problems, rate limiting (Hit Counter, Rate Limiter)

**Why it works:**
- Deque: O(1) add/remove from both ends
- Remove expired timestamps from front
- Add new timestamps at back

**Example:**
\`\`\`python
class HitCounter:
    def __init__(self):
        self.hits = deque()  # Store timestamps
    
    def hit(self, timestamp):
        self.hits.append(timestamp)
    
    def getHits(self, timestamp):
        # Remove hits older than 300 seconds
        while self.hits and self.hits[0] <= timestamp - 300:
            self.hits.popleft()
        return len(self.hits)
\`\`\`

---

## Design Problem Checklist

Before you say "done", verify:

- [ ] **All operations** meet required time complexity
- [ ] **Edge cases** handled (empty, full, single element, invalid input)
- [ ] **Space complexity** is acceptable
- [ ] **API** is clean and intuitive
- [ ] **Thread safety** considered (if relevant)
- [ ] **Scalability** discussed (can it handle 1M items?)

---

## Interview Tips

1. **Start with brute force**: Even if obvious, state it. Shows you understand the problem.

2. **Think out loud**: "I need O(1) access, so HashMap. But also need order, so..."

3. **Draw it**: Sketch the data structures and how they interact.

4. **Test as you go**: Don't wait until the end. Test each method with simple example.

5. **Discuss alternatives**: "We could also use X, but that would be O(N) instead of O(1)."

6. **Ask about constraints**: "What's the expected capacity? Should I optimize for reads or writes?"

**Common Mistakes to Avoid:**
- ❌ Jumping to code without clarifying requirements
- ❌ Not considering edge cases until asked
- ❌ Forgetting to analyze time/space complexity
- ❌ Not testing with examples
- ❌ Overcomplicating when simple solution exists

**Pro Tip:** LRU Cache is the most common design problem. Master it completely - understand why HashMap + Doubly LinkedList is needed, not just how to implement it.`,
      quiz: [
        {
          id: 'q1',
          question:
            'Why do design problems often require combining multiple data structures? Give an example.',
          sampleAnswer:
            "Design problems combine data structures because single structures rarely meet all requirements. For example, LRU Cache needs BOTH O(1) access by key AND O(1) update of access order. HashMap alone gives O(1) access but cannot track order efficiently. LinkedList alone tracks order but requires O(N) to find elements. Combined, HashMap stores key->node mappings for O(1) access, while doubly LinkedList maintains LRU order with O(1) move-to-front. Each structure compensates for the other's weakness. This pattern repeats: Min Stack needs stack operations + O(1) getMin (use extra stack), Rate Limiter needs fast lookup + time ordering (use HashMap + Deque). Real systems are complex and single data structures are too limited.",
          keyPoints: [
            'Single structures rarely meet all requirements',
            'LRU: HashMap for O(1) access + LinkedList for order',
            'Each structure compensates weakness of other',
            'Min Stack: main stack + tracking stack',
            'Pattern repeats across design problems',
          ],
        },
        {
          id: 'q2',
          question:
            'How do you decide between simplicity and optimal performance in design problems?',
          sampleAnswer:
            'I start by clarifying constraints: "What\'s the expected scale? Should I optimize for speed or maintainability?" If capacity is small (say 100 items), a simple list might suffice even if O(N) - it\'s readable and fast enough. But for production systems (10K+ items), I choose optimal structures even if more complex. For interviews, I state both: "Simple solution would be X with O(N), but optimal requires Y with O(1) - should I implement optimal?" This shows judgment. I also consider: Will code be maintained by others? (favor simplicity). Are there performance SLAs? (favor optimization). Can we profile first? (premature optimization warning). The answer is usually: start simple, optimize bottlenecks with data.',
          keyPoints: [
            'Clarify constraints and scale expectations',
            'Small scale: simplicity wins',
            'Large scale: optimal structures needed',
            'State both options in interviews',
            'Consider maintainability vs performance',
          ],
        },
        {
          id: 'q3',
          question:
            'What are the most important questions to ask when given a design problem?',
          sampleAnswer:
            'Most important questions: (1) What operations are needed and their expected frequency? This determines what to optimize. (2) What are the performance requirements? "Should get() be O(1)?" (3) What\'s the expected scale? 100 items vs 1M items changes approach. (4) How should edge cases behave? Return null, throw exception, or default value? (5) Are there space constraints or is unlimited memory OK? (6) Should it be thread-safe? These questions prevent building wrong solution. For LRU Cache: "What\'s capacity limit? What does get() return for missing keys? When exactly is something \'recently used\'?" Good questions show engineering maturity and prevent rework.',
          keyPoints: [
            'What operations and their frequency?',
            'Performance requirements (time complexity)?',
            'Expected scale (100 vs 1M items)?',
            'Edge case behavior?',
            'Thread-safety requirements?',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'Why is LRU Cache the most commonly asked design problem?',
          options: [
            'It is the easiest design problem',
            'It requires combining HashMap and LinkedList, testing data structure knowledge',
            'It only needs one data structure',
            'It has no edge cases',
          ],
          correctAnswer: 1,
          explanation:
            'LRU Cache is popular because it tests the ability to combine data structures optimally. It requires HashMap for O(1) access AND doubly LinkedList for O(1) order updates - neither alone suffices. This tests deep understanding of data structure trade-offs, which is exactly what distinguishes strong candidates.',
        },
        {
          id: 'mc2',
          question: 'What is the main challenge in design problems?',
          options: [
            'Writing syntactically correct code',
            'Choosing and combining data structures to meet time complexity requirements',
            'Memorizing algorithms',
            'Using the fewest lines of code',
          ],
          correctAnswer: 1,
          explanation:
            'The core challenge is selecting the right data structures to satisfy ALL requirements. LRU Cache needs O(1) for get AND put with eviction. Min Stack needs O(1) for push, pop, AND getMin. This requires understanding trade-offs and often combining multiple structures creatively.',
        },
        {
          id: 'mc3',
          question:
            'In design problems, when should you use a doubly linked list over a singly linked list?',
          options: [
            'Always, it is always better',
            'When you need to remove arbitrary nodes in O(1) time',
            'Never, singly linked lists are always sufficient',
            'Only for small datasets',
          ],
          correctAnswer: 1,
          explanation:
            'Doubly linked lists allow O(1) removal of nodes when you have a reference to that node, because you can access node.prev directly. Singly linked lists require O(N) to find the previous node. LRU Cache uses doubly linked list because we need to remove arbitrary nodes (when accessing them) in O(1).',
        },
        {
          id: 'mc4',
          question: 'What does "amortized O(1)" mean?',
          options: [
            'Always O(1) for every single operation',
            'Average O(1) over a sequence of operations, though some individual operations may be O(N)',
            'Worse than O(1)',
            'Only works for small inputs',
          ],
          correctAnswer: 1,
          explanation:
            'Amortized O(1) means that while some operations may take O(N) time occasionally, the average time over many operations is O(1). Example: Queue using two stacks - dequeue might occasionally transfer N items (O(N)), but each item is transferred at most once, so average is O(1).',
        },
        {
          id: 'mc5',
          question:
            'Why do many rate limiting problems use a deque (double-ended queue)?',
          options: [
            'Deques use less memory',
            'Deques allow O(1) removal from front (old timestamps) and O(1) addition to back (new timestamps)',
            'Deques automatically sort timestamps',
            'Deques prevent duplicates',
          ],
          correctAnswer: 1,
          explanation:
            'Rate limiters need to remove old timestamps (from front) and add new timestamps (to back), both in O(1). Deque supports O(1) operations at both ends, making it perfect for sliding window patterns. Regular lists have O(N) removal from front.',
        },
      ],
    },
    {
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
      quiz: [
        {
          id: 'q1',
          question:
            'Why does LRU Cache require a doubly linked list instead of a singly linked list? Explain in detail.',
          sampleAnswer:
            'LRU Cache requires doubly linked list because we need to remove arbitrary nodes from the middle in O(1) time. When we access a key via get(), we find the node using HashMap in O(1), then must move that node to the front. Moving requires: (1) Remove node from current position, (2) Insert at front. With doubly linked list, removal is O(1): node.prev.next = node.next; node.next.prev = node.prev - we have direct access to both neighbors. With singly linked list, we only have node.next, so to remove a node we need to find its previous node, which requires O(N) traversal from head. Since get() must be O(1), we cannot afford O(N) removal. The extra prev pointer in doubly linked list is essential.',
          keyPoints: [
            'Need to remove arbitrary nodes in O(1)',
            'get() finds node via HashMap, must move to front',
            'Doubly linked: node.prev gives O(1) removal',
            'Singly linked: need O(N) to find previous',
            'Extra prev pointer essential for O(1)',
          ],
        },
        {
          id: 'q2',
          question:
            'When would you choose LFU over LRU, and vice versa? Give concrete examples.',
          sampleAnswer:
            'Choose LRU for most general-purpose caching: web browsers (recently visited pages likely revisited), database query caches, API caches. LRU handles temporal locality well - if something was used recently, it will likely be used again soon. Choose LFU when access patterns are heavily skewed with "hot" items: video streaming (popular videos get 80% of views), CDNs (viral content), autocomplete (common words). LFU prevents cache pollution - one sequential scan won\'t evict all your hot items. For example, if scanning through IDs 1-1000 once but repeatedly accessing a few hot items, LRU would evict the hot items (temporarily not accessed), while LFU keeps them (high frequency). However, LFU is more complex to implement and can struggle with changing patterns.',
          keyPoints: [
            'LRU: general purpose, temporal locality',
            'LRU examples: browsers, query cache',
            'LFU: skewed access patterns, hot items',
            'LFU examples: video streaming, CDN',
            'LFU resistant to cache pollution',
          ],
        },
        {
          id: 'q3',
          question:
            'Explain why the "dummy head and tail" technique simplifies LRU Cache implementation.',
          sampleAnswer:
            'Dummy head and tail eliminate all edge case checks when adding/removing nodes. Without dummies, adding to an empty list requires "if head is None: head = tail = new_node", and removing last node requires "if head == tail: head = tail = None", etc. With dummies, the list is NEVER empty (always has head and tail), so adding after head is always "new_node.next = head.next; head.next.prev = new_node; head.next = new_node" with no conditionals. Removing is always "prev.next = next; next.prev = prev". The actual data nodes are between head and tail. This eliminates null checks, simplifies code, and prevents edge case bugs. It\'s a standard technique for doubly linked lists in production code.',
          keyPoints: [
            'Eliminates null checks and edge cases',
            'List never empty (always has dummies)',
            'Add/remove operations have no conditionals',
            'Data nodes always between head and tail',
            'Standard technique in production code',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What is the time complexity of LRU Cache get() operation?',
          options: ['O(N)', 'O(log N)', 'O(1)', 'O(N log N)'],
          correctAnswer: 2,
          explanation:
            'LRU Cache get() is O(1). We use HashMap to find the node in O(1), then use doubly linked list to move it to front in O(1). Both operations are constant time, so overall is O(1).',
        },
        {
          id: 'mc2',
          question: "Why can't we implement LRU Cache with just a HashMap?",
          options: [
            'HashMaps are too slow',
            'HashMaps cannot track the order of access efficiently',
            'HashMaps do not support deletion',
            'HashMaps use too much memory',
          ],
          correctAnswer: 1,
          explanation:
            'HashMap provides O(1) lookup but cannot efficiently track which item was least recently used. We would need to iterate through all items to find the LRU item, making eviction O(N). We need a linked list to maintain access order.',
        },
        {
          id: 'mc3',
          question:
            'In LRU Cache, when does an item become "most recently used"?',
          options: [
            'Only when we put() it',
            'Only when we get() it',
            'Both when we get() OR put() it',
            'Only when cache is full',
          ],
          correctAnswer: 2,
          explanation:
            "An item becomes most recently used on BOTH get() and put(). When we get(key), we're accessing it (used recently). When we put(key, val), we're either adding new (definitely recent) or updating existing (also recent). Both operations move the node to front.",
        },
        {
          id: 'mc4',
          question: 'What advantage does LFU have over LRU?',
          options: [
            'LFU is simpler to implement',
            'LFU is always faster',
            'LFU is resistant to cache pollution from sequential scans',
            'LFU uses less memory',
          ],
          correctAnswer: 2,
          explanation:
            'LFU resists cache pollution because it tracks frequency, not recency. A one-time sequential scan of 1000 items won\'t evict frequently-used items in LFU, but would evict everything in LRU (since the scanned items are now "most recent"). LFU is actually more complex and uses more memory.',
        },
        {
          id: 'mc5',
          question:
            'Why do we need to remove a node from HashMap when evicting it from LRU Cache?',
          options: [
            'To save memory',
            'To prevent memory leaks and incorrect lookups',
            'HashMap removal is required for LinkedList removal',
            'To make get() faster',
          ],
          correctAnswer: 1,
          explanation:
            "We must remove from HashMap to prevent: (1) Memory leak - HashMap holds reference to evicted node, preventing garbage collection. (2) Incorrect behavior - future get(key) would find stale node that's not in LinkedList anymore. Always maintain consistency: if node is evicted from list, it must be removed from HashMap.",
        },
      ],
    },
    {
      id: 'stack-queue-designs',
      title: 'Stack & Queue Designs',
      content: `Stack and queue design problems test your ability to implement data structures with **constrained operations** or using **limited primitives**. These problems appear frequently in interviews because they test fundamental understanding.

---

## Min Stack / Max Stack

**Problem**: Implement a stack that supports push, pop, top, and **getMin()/getMax() in O(1)**.

**Challenge**: Regular stack operations are easy, but how do we track min/max in O(1)?

### Approach 1: Two Stacks

**Idea**: Maintain two stacks:
1. **Main stack**: Stores all values
2. **Min stack**: Stores minimum at each level

\`\`\`python
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []
    
    def push(self, val):
        self.stack.append(val)
        # Push current minimum
        min_val = min(val, self.min_stack[-1] if self.min_stack else val)
        self.min_stack.append(min_val)
    
    def pop(self):
        self.stack.pop()
        self.min_stack.pop()  # Keep in sync
    
    def top(self):
        return self.stack[-1]
    
    def getMin(self):
        return self.min_stack[-1]  # O(1)!
\`\`\`

**Key Insight**: At each level, we remember "what was the min up to this point?"

**Example**:
\`\`\`
push(3): stack=[3],    min_stack=[3]  # min so far: 3
push(1): stack=[3,1],  min_stack=[3,1]  # min so far: 1
push(2): stack=[3,1,2], min_stack=[3,1,1]  # min so far: still 1
pop():   stack=[3,1],  min_stack=[3,1]
getMin(): returns 1  # O(1)
\`\`\`

**Time Complexity**: O(1) for all operations  
**Space Complexity**: O(N) - two stacks of size N

### Approach 2: Single Stack with Tuples

**Idea**: Each stack element stores (value, min_so_far)

\`\`\`python
class MinStack:
    def __init__(self):
        self.stack = []  # Store (val, min_so_far)
    
    def push(self, val):
        if not self.stack:
            self.stack.append((val, val))
        else:
            current_min = min(val, self.stack[-1][1])
            self.stack.append((val, current_min))
    
    def pop(self):
        self.stack.pop()
    
    def top(self):
        return self.stack[-1][0]
    
    def getMin(self):
        return self.stack[-1][1]  # O(1)!
\`\`\`

**Trade-off**: Simpler (one data structure) but each element uses more memory (tuple vs single value).

### Approach 3: Optimized Min Stack (Advanced)

**Idea**: Only store mins when they change

\`\`\`python
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []  # Only store (value, count)
    
    def push(self, val):
        self.stack.append(val)
        if not self.min_stack or val < self.min_stack[-1][0]:
            self.min_stack.append((val, 1))
        elif val == self.min_stack[-1][0]:
            self.min_stack[-1] = (val, self.min_stack[-1][1] + 1)
    
    def pop(self):
        val = self.stack.pop()
        if val == self.min_stack[-1][0]:
            if self.min_stack[-1][1] == 1:
                self.min_stack.pop()
            else:
                self.min_stack[-1] = (val, self.min_stack[-1][1] - 1)
\`\`\`

**Optimization**: If pushing many duplicates, saves space.

---

## Implement Queue Using Stacks

**Problem**: Implement FIFO queue using only stack operations (LIFO).

**Challenge**: Stack is LIFO, queue is FIFO - opposite orders!

### Approach: Two Stacks

**Key Idea**: Use one stack for input, one for output. Transfer reverses order.

\`\`\`python
class QueueUsingStacks:
    def __init__(self):
        self.stack_in = []   # For enqueue
        self.stack_out = []  # For dequeue
    
    def enqueue(self, x):
        self.stack_in.append(x)  # O(1)
    
    def dequeue(self):
        if not self.stack_out:
            # Transfer all from in to out (reverses order)
            while self.stack_in:
                self.stack_out.append(self.stack_in.pop())
        return self.stack_out.pop() if self.stack_out else None
    
    def peek(self):
        if not self.stack_out:
            while self.stack_in:
                self.stack_out.append(self.stack_in.pop())
        return self.stack_out[-1] if self.stack_out else None
\`\`\`

**How it works**:
\`\`\`
enqueue(1): stack_in=[1], stack_out=[]
enqueue(2): stack_in=[1,2], stack_out=[]
enqueue(3): stack_in=[1,2,3], stack_out=[]

dequeue():
  Transfer: stack_in=[], stack_out=[3,2,1]
  Pop from out: returns 1, stack_out=[3,2]

enqueue(4): stack_in=[4], stack_out=[3,2]

dequeue():
  stack_out not empty, just pop: returns 2, stack_out=[3]
\`\`\`

**Key Insight**: Transfer is lazy - only when stack_out is empty. Once transferred, elements are in correct order.

**Time Complexity**: 
- enqueue: O(1)
- dequeue: Amortized O(1) - each element moved at most once

**Space Complexity**: O(N)

**Why Amortized O(1)?**
- A single dequeue might move N elements: O(N)
- But each element is moved exactly once from in to out
- So N dequeues = N moves total = O(1) average per operation

---

## Implement Stack Using Queues

**Problem**: Implement LIFO stack using only queue operations (FIFO).

### Approach 1: Two Queues

**Idea**: Make push expensive, pop cheap.

\`\`\`python
class StackUsingQueues:
    def __init__(self):
        self.q1 = deque()
        self.q2 = deque()
    
    def push(self, x):
        # Add to q2
        self.q2.append(x)
        # Move all from q1 to q2 (x is now at front)
        while self.q1:
            self.q2.append(self.q1.popleft())
        # Swap names
        self.q1, self.q2 = self.q2, self.q1
    
    def pop(self):
        return self.q1.popleft() if self.q1 else None
    
    def top(self):
        return self.q1[0] if self.q1 else None
\`\`\`

**How it works**:
\`\`\`
push(1): q1=[1]
push(2): q2=[2,1], swap -> q1=[2,1]
push(3): q2=[3,2,1], swap -> q1=[3,2,1]
pop(): returns 3, q1=[2,1]  # LIFO maintained!
\`\`\`

**Time Complexity**:
- push: O(N)
- pop: O(1)

### Approach 2: Single Queue

**Idea**: Rotate queue after each push

\`\`\`python
class StackUsingQueues:
    def __init__(self):
        self.q = deque()
    
    def push(self, x):
        self.q.append(x)
        # Rotate: move all elements before x to after x
        for _ in range(len(self.q) - 1):
            self.q.append(self.q.popleft())
    
    def pop(self):
        return self.q.popleft() if self.q else None
\`\`\`

**Cleaner code, same complexity**.

---

## Pattern Recognition

### When to Use Min/Max Stack Pattern:
- Need stack operations + track min/max
- "Design a stack that supports X in O(1)"
- Examples: Min Stack, Max Stack, Stock Span Problem

**Key Technique**: Auxiliary stack tracking property at each level

### When to Use Two Stacks Pattern:
- Implement queue with stacks
- Reverse operation order
- Browser history (forward/back)

**Key Technique**: One for input, one for output, transfer reverses

### When to Use Two Queues Pattern:
- Implement stack with queues
- Make one end "special" by rotation

**Key Technique**: Reorder on push to maintain LIFO

---

## Interview Tips

1. **Clarify constraints**: "Can I use two stacks, or must it be one?"

2. **Explain the core insight**: 
   - Min Stack: "Track min at each level"
   - Queue with stacks: "Transfer reverses order"

3. **Analyze amortization**: "Each element moved at most once, so amortized O(1)"

4. **Test edge cases**: Empty, single element, repeated min values

5. **Discuss trade-offs**: Push expensive vs pop expensive

**Common Mistakes**:
- Forgetting to keep min_stack synchronized with main stack
- Not handling empty cases
- Not explaining amortization (saying O(N) instead of amortized O(1))
- Overcomplicating Min Stack (it's simpler than you think!)`,
      quiz: [
        {
          id: 'q1',
          question:
            'Explain how the Min Stack maintains O(1) getMin() while supporting push and pop.',
          sampleAnswer:
            'Min Stack maintains O(1) getMin by storing the minimum value at each level in a parallel min_stack. When we push(val), we compare val with current min and push the smaller one to min_stack. When we pop(), we pop from both stacks to keep them synchronized. The key insight: at any point, min_stack.top() tells us "what is the minimum among all elements currently in the stack?" When we pop an element, the min_stack also pops, revealing the min of the remaining elements. For example, after push(3), push(1), push(5): main=[3,1,5], min=[3,1,1]. getMin() returns min[-1]=1 in O(1). After pop(): main=[3,1], min=[3,1], getMin() still works correctly returning 1.',
          keyPoints: [
            'Parallel min_stack tracks min at each level',
            'Push: store min(new_val, current_min)',
            'Pop: remove from both stacks (stay synchronized)',
            'min_stack.top() = min of current elements',
            'Always O(1) - just array access',
          ],
        },
        {
          id: 'q2',
          question:
            'Why is Queue using Stacks amortized O(1) for dequeue? Explain the amortization.',
          sampleAnswer:
            "Individual dequeue can be O(N) when transferring all elements from stack_in to stack_out, but amortized cost is O(1) because each element is transferred at most once. Consider N enqueues followed by N dequeues: First dequeue transfers N elements (O(N)), but subsequent N-1 dequeues just pop from stack_out (O(1) each). Total: O(N) + O(N-1) = O(2N-1) for N dequeues = O(1) average. Key point: once an element moves to stack_out, it never moves back. You can't count O(N) for every dequeue - elements aren't transferred repeatedly. Amortized analysis spreads the expensive operation over all operations.",
          keyPoints: [
            'Individual dequeue can be O(N) (transfer)',
            'But each element transferred at most once',
            'N dequeues = O(N) total work = O(1) average',
            'Elements never move back to stack_in',
            'Spread expensive operation over all ops',
          ],
        },
        {
          id: 'q3',
          question:
            'Compare the two approaches for Min Stack: two stacks vs. stack with tuples. Which is better?',
          sampleAnswer:
            "Two-stack approach: Pros: cleaner separation, easier to understand, each stack stores single values. Cons: manage two data structures, remember to sync on pop. Stack-with-tuples approach: Pros: single data structure, impossible to desync, cleaner code. Cons: every element stores tuple (more memory per element), tuple unpacking overhead. In practice, I prefer two stacks for interviews because it's clearer to explain and reason about. For production, stack-with-tuples might be cleaner due to single data structure. Neither has better time complexity (both O(1)). Memory is similar (two stacks: 2N ints, tuples: N tuples). Choice is about code clarity and maintainability.",
          keyPoints: [
            'Two stacks: cleaner separation, easier to explain',
            'Tuples: single structure, cannot desync',
            'Both O(1) time complexity',
            'Memory similar (2N ints vs N tuples)',
            'Choice based on clarity, not performance',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What is the time complexity of getMin() in Min Stack?',
          options: ['O(N)', 'O(log N)', 'O(1)', 'Amortized O(1)'],
          correctAnswer: 2,
          explanation:
            'getMin() is exactly O(1), not amortized. We simply return min_stack[-1] which is a direct array access. No searching, no iteration - constant time every single call.',
        },
        {
          id: 'mc2',
          question:
            'Why can\'t we just track a single "current_min" variable in Min Stack?',
          options: [
            'Variables use too much memory',
            "After popping the min element, we don't know what the new min is",
            'Variables are slower than stacks',
            'It would work fine',
          ],
          correctAnswer: 1,
          explanation:
            "Single variable fails when we pop the minimum element - we lose information about what the previous min was. Example: push(1), push(2), current_min=1. Then pop() removes 2, current_min still 1 (correct). But if push(3), pop() removes 3, pop() removes 1 - now we don't know the min! Stack remembers history.",
        },
        {
          id: 'mc3',
          question:
            'In Queue using Stacks, why do we transfer elements from stack_in to stack_out?',
          options: [
            'To save memory',
            'To reverse the order from LIFO to FIFO',
            'To make enqueue faster',
            'It is not necessary',
          ],
          correctAnswer: 1,
          explanation:
            'Transfer reverses order! Elements in stack_in are in LIFO order (last pushed on top). Moving them to stack_out reverses this to FIFO order (first pushed now on top of stack_out). This is the core trick - double reversal (push to stack1, transfer to stack2) gives original order.',
        },
        {
          id: 'mc4',
          question:
            'What is the time complexity of push() in Stack using Queues (single queue approach)?',
          options: ['O(1)', 'O(log N)', 'O(N)', 'O(N log N)'],
          correctAnswer: 2,
          explanation:
            'Push is O(N) because we rotate the queue: append new element, then move all N-1 previous elements to the back. This reorders the queue so the newest element is at front (for LIFO pop). Every push touches all elements.',
        },
        {
          id: 'mc5',
          question:
            'When implementing Queue using Stacks, when should we transfer elements from stack_in to stack_out?',
          options: [
            'On every enqueue',
            'On every dequeue',
            'Only when stack_out is empty and we need to dequeue',
            'Never',
          ],
          correctAnswer: 2,
          explanation:
            "Transfer only when stack_out is empty and we need to dequeue (lazy transfer). If stack_out has elements, they're already in correct FIFO order - just pop from it. Transferring on every operation would be wasteful and destroy amortization. This lazy approach ensures each element is moved at most once.",
        },
      ],
    },
    {
      id: 'rate-limiting',
      title: 'Rate Limiting & Counters',
      content: `Rate limiting controls how frequently users can perform actions. It's critical for:
- **API protection**: Prevent abuse and overload
- **Fair usage**: Ensure all users get access
- **Cost control**: Limit expensive operations
- **Security**: Prevent brute force attacks

**Real-world examples**: Twitter limits tweets/hour, APIs limit requests/second, login attempts limited per minute.

---

## Design Hit Counter

**Problem**: Count hits in the last N seconds (typically 300 = 5 minutes).

**Operations**:
- \`hit(timestamp)\`: Record a hit at given time
- \`getHits(timestamp)\`: Return hits in last N seconds

### Approach 1: Queue/Deque (Simple)

**Idea**: Store all timestamps, remove old ones.

\`\`\`python
from collections import deque

class HitCounter:
    def __init__(self):
        self.hits = deque()  # Store timestamps
        self.window = 300  # 5 minutes
    
    def hit(self, timestamp):
        self.hits.append(timestamp)  # O(1)
    
    def getHits(self, timestamp):
        # Remove hits older than timestamp - window
        while self.hits and self.hits[0] <= timestamp - self.window:
            self.hits.popleft()  # O(1) per old hit
        return len(self.hits)  # O(1)
\`\`\`

**Time Complexity**:
- hit(): O(1)
- getHits(): O(N) worst case if many old hits to remove, but amortized O(1) per hit

**Space Complexity**: O(N) where N is total hits in window

**Pros**: Simple, exact count  
**Cons**: Memory grows with hit count

### Approach 2: Time Buckets (Optimized)

**Idea**: Divide time into buckets, store count per bucket.

\`\`\`python
class HitCounter:
    def __init__(self):
        self.buckets = [0] * 300  # 300 seconds
        self.timestamps = [0] * 300  # Last update time per bucket
    
    def hit(self, timestamp):
        idx = timestamp % 300
        # If bucket from old window, reset it
        if self.timestamps[idx] != timestamp:
            self.timestamps[idx] = timestamp
            self.buckets[idx] = 1
        else:
            self.buckets[idx] += 1
    
    def getHits(self, timestamp):
        total = 0
        for i in range(300):
            # Only count if timestamp is within window
            if timestamp - self.timestamps[i] < 300:
                total += self.buckets[i]
        return total
\`\`\`

**Time Complexity**:
- hit(): O(1)
- getHits(): O(300) = O(1) for fixed window

**Space Complexity**: O(300) = O(1)

**Pros**: Fixed memory, good for high traffic  
**Cons**: Less accurate (bucket granularity), getHits() scans all buckets

### Approach 3: Hybrid (Best of Both)

**Idea**: Use deque but with bucketing for extreme scale.

\`\`\`python
class HitCounter:
    def __init__(self):
        self.hits = deque()  # Store (timestamp, count) pairs
        self.window = 300
    
    def hit(self, timestamp):
        # If last hit is same second, increment count
        if self.hits and self.hits[-1][0] == timestamp:
            self.hits[-1] = (timestamp, self.hits[-1][1] + 1)
        else:
            self.hits.append((timestamp, 1))
    
    def getHits(self, timestamp):
        # Remove old entries
        while self.hits and self.hits[0][0] <= timestamp - self.window:
            self.hits.popleft()
        
        return sum(count for ts, count in self.hits)
\`\`\`

**Optimization**: Stores (timestamp, count) instead of individual hits. If 1000 hits in same second, stores once instead of 1000 times.

---

## Rate Limiter Algorithms

### 1. Fixed Window Counter

**Idea**: Count requests in fixed time windows (e.g., 0-60s, 60-120s).

\`\`\`python
class FixedWindowRateLimiter:
    def __init__(self, limit, window_size):
        self.limit = limit
        self.window_size = window_size
        self.window_start = 0
        self.count = 0
    
    def allow_request(self, timestamp):
        # Check if we're in a new window
        window_num = timestamp // self.window_size
        if window_num > self.window_start:
            # Reset for new window
            self.window_start = window_num
            self.count = 0
        
        # Check if under limit
        if self.count < self.limit:
            self.count += 1
            return True
        return False
\`\`\`

**Problem**: Boundary spike issue!
\`\`\`
Window 1: [0-60s] - 100 requests at t=59s (allowed)
Window 2: [60-120s] - 100 requests at t=60s (allowed)
→ 200 requests in 2 seconds! (burst at boundary)
\`\`\`

### 2. Sliding Window Log

**Idea**: Store timestamp of each request, count in sliding window.

\`\`\`python
class SlidingWindowLog:
    def __init__(self, limit, window_size):
        self.limit = limit
        self.window_size = window_size
        self.requests = deque()  # Store timestamps
    
    def allow_request(self, timestamp):
        # Remove old requests outside window
        cutoff = timestamp - self.window_size
        while self.requests and self.requests[0] <= cutoff:
            self.requests.popleft()
        
        # Check if under limit
        if len(self.requests) < self.limit:
            self.requests.append(timestamp)
            return True
        return False
\`\`\`

**Pros**: Accurate, no boundary issues  
**Cons**: O(N) memory for N requests

### 3. Sliding Window Counter (Best)

**Idea**: Weighted combination of previous and current window.

\`\`\`python
class SlidingWindowCounter:
    def __init__(self, limit, window_size):
        self.limit = limit
        self.window_size = window_size
        self.prev_count = 0
        self.curr_count = 0
        self.curr_window_start = 0
    
    def allow_request(self, timestamp):
        window_num = timestamp // self.window_size
        
        if window_num > self.curr_window_start:
            # Move to new window
            self.prev_count = self.curr_count
            self.curr_count = 0
            self.curr_window_start = window_num
        
        # Calculate weighted count
        elapsed = timestamp % self.window_size
        weight = 1 - (elapsed / self.window_size)
        estimated_count = self.prev_count * weight + self.curr_count
        
        if estimated_count < self.limit:
            self.curr_count += 1
            return True
        return False
\`\`\`

**Pros**: O(1) memory, accurate approximation  
**Cons**: Slightly complex math

### 4. Token Bucket (Industry Standard)

**Idea**: Bucket fills with tokens at fixed rate. Requests consume tokens.

\`\`\`python
class TokenBucket:
    def __init__(self, capacity, refill_rate):
        self.capacity = capacity  # Max tokens
        self.tokens = capacity  # Current tokens
        self.refill_rate = refill_rate  # Tokens per second
        self.last_refill = time.time()
    
    def allow_request(self):
        now = time.time()
        # Refill tokens based on time elapsed
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, 
                         self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
        
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False
\`\`\`

**Pros**: 
- Allows bursts (up to capacity)
- Smooth rate limiting
- Used by AWS, GCP, most APIs

**Example**: Capacity=10, rate=1/sec
- Can burst 10 requests immediately
- Then limited to 1 per second
- Unused capacity accumulates (up to 10)

---

## Comparison Table

| Algorithm | Memory | Accuracy | Burst Handling | Use Case |
|-----------|--------|----------|----------------|----------|
| **Fixed Window** | O(1) | ❌ Boundary spikes | ❌ Double at boundary | Simple systems |
| **Sliding Log** | O(N) | ✅ Perfect | ✅ Perfect | Low traffic |
| **Sliding Counter** | O(1) | ✅ Good | ✅ Good | High traffic |
| **Token Bucket** | O(1) | ✅ Good | ✅ Controlled | Production APIs |

---

## Design Considerations

### Distributed Systems

**Problem**: Rate limiting across multiple servers

**Solutions**:
1. **Redis with atomic counters**: INCR and EXPIRE
2. **Sliding window in Redis**: Sorted sets with timestamps
3. **Token bucket in Redis**: Store tokens and last_refill

\`\`\`python
# Redis-based rate limiter
def allow_request(user_id, redis_client):
    key = f"rate_limit:{user_id}"
    current = redis_client.incr(key)
    
    if current == 1:
        redis_client.expire(key, 60)  # 60 second window
    
    return current <= 100  # 100 requests per minute
\`\`\`

### Per-User vs Global

- **Per-user**: Each user has own limit (fair)
- **Global**: Total limit across all users (protect system)
- **Hybrid**: Per-user + global cap

### 429 Status Code

Return "429 Too Many Requests" with headers:
\`\`\`
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1617891234
Retry-After: 60
\`\`\`

---

## Interview Tips

1. **Clarify requirements**:
   - What's the rate? (100/min, 1000/hour?)
   - Per user or global?
   - Fixed or sliding window?
   - Distributed or single server?

2. **Start simple**: Fixed window, then discuss improvements

3. **Mention production**: "In production, I'd use Token Bucket with Redis"

4. **Discuss trade-offs**: Memory vs accuracy, simplicity vs perfect fairness

5. **Test edge cases**: Boundary times, burst traffic, user with 0 requests

**Common Mistake**: Not removing old timestamps in sliding window implementations!`,
      quiz: [
        {
          id: 'q1',
          question:
            'Explain the boundary spike problem in Fixed Window rate limiting and how Sliding Window fixes it.',
          sampleAnswer:
            'Fixed Window resets counter at fixed intervals (0-60s, 60-120s, etc.). The problem: a user can make 100 requests at t=59s (end of window 1) and another 100 at t=60s (start of window 2), getting 200 requests in 1 second while limit is 100/minute. Sliding Window fixes this by looking at any 60-second period ending at current time, not fixed boundaries. At t=60s with sliding window, we count requests from t=0 to t=60, including the 100 at t=59, so we\'d only allow 0 more. Sliding window gives true "100 per rolling minute" while fixed window gives "100 per calendar minute" which can be exploited at boundaries.',
          keyPoints: [
            'Fixed window: resets at boundaries',
            'Exploit: 100 at t=59s + 100 at t=60s',
            'Sliding window: any 60s period ending now',
            'Sliding counts from t-60 to t',
            'Prevents boundary exploitation',
          ],
        },
        {
          id: 'q2',
          question:
            'Why is Token Bucket the industry standard for rate limiting? What advantages does it have?',
          sampleAnswer:
            'Token Bucket is industry standard because it elegantly handles bursts while maintaining average rate. Key advantages: (1) Allows controlled bursts - if user hasn\'t used API for a while, they can burst up to capacity, which feels natural. (2) Smooth refill - tokens accumulate steadily, not all at once. (3) Simple to reason about - "you have N tokens, requests cost 1 token". (4) Easily distributed - can store in Redis. (5) Flexible - different costs for different operations. Real APIs (AWS, GCP, GitHub) use this because users expect occasional bursts without penalty, but sustained abuse is still blocked. Fixed window feels harsh (sudden reset), Token Bucket feels fair.',
          keyPoints: [
            'Allows controlled bursts (up to capacity)',
            'Smooth token refill over time',
            'Simple mental model (tokens = credits)',
            'Easily distributed via Redis',
            'Used by AWS, GCP, GitHub - proven',
          ],
        },
        {
          id: 'q3',
          question:
            'How would you implement rate limiting in a distributed system with multiple servers?',
          sampleAnswer:
            'In distributed systems, use centralized state in Redis or similar. Each server checks/updates Redis before allowing request. Implementation: (1) Use Redis INCR for atomic counter increment. (2) Set TTL with EXPIRE for automatic cleanup. (3) For sliding window, use sorted sets with scores=timestamps, ZREMRANGEBYSCORE to remove old entries. (4) For token bucket, store tokens and last_refill in Redis hash. (5) Handle Redis failures gracefully - either fail open (allow request) or fail closed (deny request) based on requirements. Alternative for extreme scale: use consistent hashing to shard users across Redis instances. Trade-off: slight inconsistency (could exceed limit by milliseconds) for massive scale.',
          keyPoints: [
            'Centralized state in Redis',
            'Atomic operations (INCR, EXPIRE)',
            'Sorted sets for sliding window',
            'Hash for token bucket state',
            'Handle Redis failures (fail open/closed)',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question:
            'What is the main disadvantage of Fixed Window rate limiting?',
          options: [
            'It is too slow',
            'It allows burst traffic at window boundaries',
            'It uses too much memory',
            'It cannot be implemented',
          ],
          correctAnswer: 1,
          explanation:
            'Fixed Window allows burst at boundaries. Users can make max requests at end of one window and max again at start of next window, potentially doubling the rate for a brief period. This is the classic boundary spike problem.',
        },
        {
          id: 'mc2',
          question:
            'Why do we use a deque for Hit Counter instead of a regular list?',
          options: [
            'Deques are faster for all operations',
            'Deques allow O(1) removal from front (for old timestamps)',
            'Deques use less memory',
            'Deques automatically sort data',
          ],
          correctAnswer: 1,
          explanation:
            'Deque (double-ended queue) allows O(1) popleft() to remove old timestamps from front. Regular list would need list.pop(0) which is O(N) because it shifts all elements. Since we frequently remove old timestamps, O(1) removal is critical.',
        },
        {
          id: 'mc3',
          question: 'In Token Bucket, what happens when tokens reach capacity?',
          options: [
            'The system crashes',
            'Tokens start decreasing',
            'Tokens stop accumulating (cap at capacity)',
            'Requests are denied',
          ],
          correctAnswer: 2,
          explanation:
            "When tokens reach capacity, they stop accumulating - there's a maximum burst allowed. This prevents unlimited token accumulation if API is unused for days. Example: capacity=100 means max burst of 100, even if unused for a year.",
        },
        {
          id: 'mc4',
          question:
            'What is the time complexity of getHits() in the bucket-based Hit Counter?',
          options: ['O(1)', 'O(log N)', 'O(N)', 'O(N log N)'],
          correctAnswer: 0,
          explanation:
            'Bucket-based getHits() is O(1) because it scans exactly 300 buckets (fixed), regardless of how many hits occurred. O(300) = O(1) for constant-size window. This is an advantage over deque approach which is O(hits_in_window).',
        },
        {
          id: 'mc5',
          question: 'Why is Sliding Window Counter better than Fixed Window?',
          options: [
            'It uses less memory',
            'It prevents boundary spikes while using O(1) memory',
            'It is simpler to implement',
            'It is always more accurate',
          ],
          correctAnswer: 1,
          explanation:
            'Sliding Window Counter (weighted approach) prevents boundary spikes like Sliding Log, but uses only O(1) memory like Fixed Window. It weighs previous and current window counts based on position in current window, giving good approximation with minimal memory.',
        },
      ],
    },
    {
      id: 'application-designs',
      title: 'Application Designs',
      content: `Application design problems ask you to implement real-world features or components. These test your ability to combine data structures, handle state, and design clean APIs.

---

## Design Browser History

**Problem**: Implement browser back/forward functionality.

**Operations**:
- \`visit(url)\`: Visit a URL, clearing forward history
- \`back(steps)\`: Go back \`steps\` pages
- \`forward(steps)\`: Go forward \`steps\` pages

### Approach 1: Two Stacks

**Idea**: Back stack for history, forward stack for forward navigation.

\`\`\`python
class BrowserHistory:
    def __init__(self, homepage):
        self.back_stack = [homepage]
        self.forward_stack = []
        self.current = homepage
    
    def visit(self, url):
        # Clear forward history
        self.forward_stack = []
        # Push current to back stack
        self.back_stack.append(self.current)
        self.current = url
    
    def back(self, steps):
        # Move from current/back to forward
        while steps > 0 and len(self.back_stack) > 1:
            self.forward_stack.append(self.current)
            self.current = self.back_stack.pop()
            steps -= 1
        return self.current
    
    def forward(self, steps):
        # Move from forward to current/back
        while steps > 0 and self.forward_stack:
            self.back_stack.append(self.current)
            self.current = self.forward_stack.pop()
            steps -= 1
        return self.current
\`\`\`

**Time**: O(1) per step  
**Space**: O(N) where N is number of visited pages

### Approach 2: Array with Pointer

**Idea**: Store all pages in array, track current position.

\`\`\`python
class BrowserHistory:
    def __init__(self, homepage):
        self.history = [homepage]
        self.current_idx = 0
    
    def visit(self, url):
        # Remove everything after current
        self.history = self.history[:self.current_idx + 1]
        self.history.append(url)
        self.current_idx += 1
    
    def back(self, steps):
        self.current_idx = max(0, self.current_idx - steps)
        return self.history[self.current_idx]
    
    def forward(self, steps):
        self.current_idx = min(len(self.history) - 1, 
                               self.current_idx + steps)
        return self.history[self.current_idx]
\`\`\`

**Simpler**, same complexity.

---

## Design Twitter (Simplified)

**Problem**: Implement core Twitter features.

**Operations**:
- \`postTweet(userId, tweetId)\`: User posts a tweet
- \`getNewsFeed(userId)\`: Get 10 most recent tweets from user + followees
- \`follow(followerId, followeeId)\`
- \`unfollow(followerId, followeeId)\`

### Solution: HashMap + Heap

\`\`\`python
from collections import defaultdict
import heapq

class Twitter:
    def __init__(self):
        self.tweets = defaultdict(list)  # userId -> [(timestamp, tweetId)]
        self.following = defaultdict(set)  # userId -> set of followees
        self.timestamp = 0
    
    def postTweet(self, userId, tweetId):
        self.tweets[userId].append((self.timestamp, tweetId))
        self.timestamp += 1
    
    def getNewsFeed(self, userId):
        # Get tweets from user + followees
        max_heap = []
        
        # Add own tweets (last 10)
        for tweet in self.tweets[userId][-10:]:
            heapq.heappush(max_heap, (-tweet[0], tweet[1]))
        
        # Add followees' tweets
        for followeeId in self.following[userId]:
            for tweet in self.tweets[followeeId][-10:]:
                heapq.heappush(max_heap, (-tweet[0], tweet[1]))
        
        # Extract top 10
        result = []
        while max_heap and len(result) < 10:
            result.append(heapq.heappop(max_heap)[1])
        
        return result
    
    def follow(self, followerId, followeeId):
        if followerId != followeeId:  # Can't follow self
            self.following[followerId].add(followeeId)
    
    def unfollow(self, followerId, followeeId):
        self.following[followerId].discard(followeeId)
\`\`\`

**Time Complexity**:
- postTweet: O(1)
- getNewsFeed: O(N log K) where N = total tweets to consider, K = feed size
- follow/unfollow: O(1)

**Space**: O(users * tweets)

**Key Insight**: Use max heap to merge sorted timelines efficiently. Each user's tweets are in timestamp order, so we can merge K sorted lists.

---

## Design Search Autocomplete

**Problem**: Implement search autocomplete with top K suggestions.

**Operations**:
- \`input(c)\`: User types character \`c\`, return top suggestions
- \`recordSearch(query)\`: Record completed search (update frequencies)

### Solution: Trie + Frequency Tracking

\`\`\`python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.frequency = 0  # How many times this search was completed
        self.query = ""

class AutocompleteSystem:
    def __init__(self, sentences, times):
        self.root = TrieNode()
        self.current_input = ""
        
        # Build trie with initial data
        for sentence, freq in zip(sentences, times):
            self.add_to_trie(sentence, freq)
    
    def add_to_trie(self, sentence, frequency):
        node = self.root
        for char in sentence:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
        node.query = sentence
        node.frequency += frequency
    
    def search_with_prefix(self, prefix):
        node = self.root
        # Navigate to prefix
        for char in prefix:
            if char not in node.children:
                return []  # No matches
            node = node.children[char]
        
        # Find all completions from this node
        results = []
        self.dfs_collect(node, results)
        
        # Sort by frequency (desc), then lexicographically
        results.sort(key=lambda x: (-x[1], x[0]))
        return [query for query, freq in results[:3]]  # Top 3
    
    def dfs_collect(self, node, results):
        if node.is_end:
            results.append((node.query, node.frequency))
        for child in node.children.values():
            self.dfs_collect(child, results)
    
    def input(self, c):
        if c == '#':
            # End of query, record it
            self.add_to_trie(self.current_input, 1)
            self.current_input = ""
            return []
        else:
            self.current_input += c
            return self.search_with_prefix(self.current_input)
\`\`\`

**Time Complexity**:
- input: O(p + m log m) where p = prefix length, m = matching queries
- Optimized: Store top K at each node (precompute)

**Space**: O(total characters in all queries)

**Optimization**: Store top K results at each Trie node:
\`\`\`python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.top_k = []  # Precomputed top K queries from this prefix
\`\`\`

Then input() is O(p) - just navigate to prefix and return cached top_k.

---

## Pattern Recognition

### When to Use Multiple Data Structures:
- Twitter: HashMap for users + Heap for timeline merge
- Autocomplete: Trie for prefix search + Sorting for ranking
- Browser: Stack for navigation + might add HashMap for bookmarks

### State Management:
- **Mutable state**: Tweets, follows, history
- **Derived state**: News feed, autocomplete suggestions
- **Transient state**: Current input, navigation position

### API Design Principles:
1. **Clear names**: \`postTweet\` not \`add\`, \`getNewsFeed\` not \`get\`
2. **Consistent return types**: Always list, always id
3. **Edge case handling**: Can't follow self, empty history
4. **Efficient operations**: What's called frequently? (getNewsFeed - optimize!)

---

## Interview Tips

1. **Clarify scale**: "How many users? Tweets per user? Follows per user?"

2. **Start with data structures**: "I'll use HashMap for users, list for tweets..."

3. **Think about queries**: "GetNewsFeed is called frequently, so I'll optimize that"

4. **Handle edge cases**: "Can user follow themselves? What if no tweets?"

5. **Discuss improvements**: "In production, we'd cache news feeds, use database..."

**Common Mistakes**:
- Forgetting to clear forward history on visit() in Browser History
- Not handling "can't follow self" in Twitter
- Inefficient autocomplete (linear search instead of Trie)
- Not sorting by frequency in autocomplete`,
      quiz: [
        {
          id: 'q1',
          question:
            'In Design Twitter, why do we use a heap to merge timelines instead of sorting all tweets?',
          sampleAnswer:
            'We use heap because we only need top K (usually 10) tweets, not all tweets sorted. If a user follows 1000 people with 1000 tweets each = 1M tweets, sorting all would be O(1M log 1M) = ~20M operations. With heap: take last 10 tweets from each user (10K tweets), build max heap O(10K), extract top 10 = O(10K log 10K) = ~130K operations, 150x faster! Heap is perfect for "top K from multiple sorted lists" pattern. We only sort the small result set, not everything. This is why Twitter/Facebook feeds load quickly - they don\'t sort your entire timeline, just enough to show the top.',
          keyPoints: [
            'Only need top K, not all sorted',
            'Sorting all: O(N log N) where N = all tweets',
            'Heap: O(N log K) where K = feed size',
            'Huge savings when N >> K',
            'Top K from sorted lists = heap pattern',
          ],
        },
        {
          id: 'q2',
          question:
            'Why is a Trie the best data structure for autocomplete? What alternatives did you consider?',
          sampleAnswer:
            'Trie is optimal for autocomplete because: (1) Prefix search is O(p) where p = prefix length, not dependent on number of queries. (2) All queries sharing prefix stored once (memory efficient). (3) Easy to collect all matches with DFS. Alternatives considered: (1) Array with linear search - O(N) per search, too slow. (2) Sorted array with binary search - O(log N) to find start, but still need to collect all with prefix. (3) HashMap - great for exact match O(1), but cannot efficiently find "all strings starting with X". (4) Suffix tree - overkill, used for substring search not prefix. Trie specializes in prefix operations, making it perfect for autocomplete, dictionary, and spell-check applications.',
          keyPoints: [
            'Prefix search O(p), independent of total queries',
            'Shared prefixes save memory',
            'Natural DFS to collect all matches',
            'Alternatives worse: array O(N), HashMap no prefix',
            'Trie = specialized tool for prefix problems',
          ],
        },
        {
          id: 'q3',
          question:
            'In Browser History, why does visiting a new page clear forward history?',
          sampleAnswer:
            "Visiting new page clears forward history to match expected browser behavior: when you're at page 3, go back to page 2, then visit new page 4, page 3 is no longer accessible via forward button - it's in an alternate timeline. This creates a tree structure where each visit creates a new branch. If we kept forward history, you could visit google.com, go back, visit facebook.com, then forward to google.com, which is confusing - forward should continue from where you were, not jump to alternate timeline. Clearing forward stack maintains linear history property. The alternative (keeping tree of all pages) is complex and not what users expect. Real browsers work this way.",
          keyPoints: [
            'Matches expected browser behavior',
            'Visit creates new branch, old forward is alternate timeline',
            'Maintains linear history (no confusion)',
            'Alternative: tree of pages (complex)',
            'Real browsers clear forward on visit',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question:
            'In Design Twitter, what is the time complexity of getNewsFeed?',
          options: [
            'O(1)',
            'O(F) where F is number of followees',
            'O(F log K) where K is feed size',
            'O(N log N) where N is all tweets',
          ],
          correctAnswer: 2,
          explanation:
            "getNewsFeed is O(F log K) where F = number of followees and K = feed size (typically 10). We look at last 10 tweets from each followee (~10F tweets), build heap, extract top K. The log K factor comes from heap operations. We don't need to look at all tweets ever, just recent ones.",
        },
        {
          id: 'mc2',
          question: 'Why does Browser History use two stacks instead of one?',
          options: [
            'One stack is not enough memory',
            'Two stacks allow both back and forward navigation efficiently',
            'Two stacks are faster',
            'It does not need two stacks',
          ],
          correctAnswer: 1,
          explanation:
            'Two stacks naturally represent back and forward navigation. Current page is at top of back_stack. Going back pops from back_stack and pushes to forward_stack. Going forward does the reverse. Single stack cannot efficiently support both directions.',
        },
        {
          id: 'mc3',
          question:
            'In Autocomplete, what is the time complexity of input(c) if we precompute top K at each Trie node?',
          options: [
            'O(1)',
            'O(p) where p is prefix length',
            'O(p + m log m) where m is matching queries',
            'O(N) where N is all queries',
          ],
          correctAnswer: 1,
          explanation:
            'With precomputed top K at each node, input(c) only needs to navigate to the prefix in O(p) time, then return the cached top_k list. No need to collect all matches and sort - that work was done during insertion. This optimization makes autocomplete very fast.',
        },
        {
          id: 'mc4',
          question: 'What data structure does Twitter use to track followees?',
          options: ['List', 'Set', 'Array', 'Stack'],
          correctAnswer: 1,
          explanation:
            'Set is used because: (1) O(1) follow/unfollow. (2) Prevents duplicate follows. (3) O(1) check if following. (4) No need for ordering. List would have O(N) for unfollow (need to find and remove).',
        },
        {
          id: 'mc5',
          question:
            'In Design Twitter, why do we check "if followerId != followeeId" in follow()?',
          options: [
            'To save memory',
            'To prevent users from following themselves',
            'To make it faster',
            'It is not necessary',
          ],
          correctAnswer: 1,
          explanation:
            "We prevent users from following themselves because: (1) It doesn't make logical sense. (2) Would cause duplicate tweets in news feed (own tweets appear twice). (3) Matches real Twitter behavior. This is an edge case that should be handled explicitly.",
        },
      ],
    },
    {
      id: 'system-design-basics',
      title: 'System Design Basics',
      content: `System design problems go beyond single machines, asking you to design **scalable, distributed systems**. While full system design is beyond scope here, these problems introduce key concepts.

---

## Design Parking Lot

**Problem**: Design a parking lot system that can:
- Park cars
- Remove cars  
- Track available spots
- Support different spot types (compact, large, handicapped)

This is an **Object-Oriented Design** problem testing class structure and relationships.

### Class Design

\`\`\`python
from enum import Enum
from abc import ABC, abstractmethod
import heapq

class VehicleType(Enum):
    COMPACT = 1
    LARGE = 2
    MOTORCYCLE = 3

class SpotType(Enum):
    COMPACT = 1
    LARGE = 2
    HANDICAPPED = 3
    MOTORCYCLE = 4

class Vehicle(ABC):
    def __init__(self, license_plate):
        self.license_plate = license_plate
        self.type = None
    
    @abstractmethod
    def can_fit_in(self, spot):
        pass

class Car(Vehicle):
    def __init__(self, license_plate):
        super().__init__(license_plate)
        self.type = VehicleType.COMPACT
    
    def can_fit_in(self, spot):
        return spot.type in [SpotType.COMPACT, SpotType.LARGE, 
                             SpotType.HANDICAPPED]

class Truck(Vehicle):
    def __init__(self, license_plate):
        super().__init__(license_plate)
        self.type = VehicleType.LARGE
    
    def can_fit_in(self, spot):
        return spot.type == SpotType.LARGE

class ParkingSpot:
    def __init__(self, spot_id, spot_type, level, row):
        self.id = spot_id
        self.type = spot_type
        self.level = level
        self.row = row
        self.vehicle = None
    
    def is_available(self):
        return self.vehicle is None
    
    def park_vehicle(self, vehicle):
        if not self.is_available():
            return False
        if not vehicle.can_fit_in(self):
            return False
        self.vehicle = vehicle
        return True
    
    def remove_vehicle(self):
        self.vehicle = None

class ParkingLot:
    def __init__(self):
        self.spots = {}  # spot_id -> ParkingSpot
        self.vehicle_to_spot = {}  # license_plate -> spot_id
        # Min heaps for each type (for O(1) nearest spot)
        self.available_spots = {
            SpotType.COMPACT: [],
            SpotType.LARGE: [],
            SpotType.HANDICAPPED: [],
            SpotType.MOTORCYCLE: []
        }
    
    def add_spot(self, spot):
        self.spots[spot.id] = spot
        heapq.heappush(self.available_spots[spot.type], 
                      (spot.level, spot.row, spot.id))
    
    def park_vehicle(self, vehicle):
        # Find best available spot
        spot_id = self._find_available_spot(vehicle)
        if not spot_id:
            return None  # No spots available
        
        spot = self.spots[spot_id]
        if spot.park_vehicle(vehicle):
            self.vehicle_to_spot[vehicle.license_plate] = spot_id
            return spot_id
        return None
    
    def _find_available_spot(self, vehicle):
        # Try to find spot (check each compatible type)
        for spot_type in [SpotType.COMPACT, SpotType.LARGE, 
                         SpotType.HANDICAPPED]:
            heap = self.available_spots[spot_type]
            while heap:
                level, row, spot_id = heap[0]
                spot = self.spots[spot_id]
                if spot.is_available() and vehicle.can_fit_in(spot):
                    heapq.heappop(heap)
                    return spot_id
                else:
                    heapq.heappop(heap)  # Remove stale entry
        return None
    
    def remove_vehicle(self, license_plate):
        if license_plate not in self.vehicle_to_spot:
            return False
        
        spot_id = self.vehicle_to_spot[license_plate]
        spot = self.spots[spot_id]
        spot.remove_vehicle()
        
        # Return spot to available pool
        heapq.heappush(self.available_spots[spot.type],
                      (spot.level, spot.row, spot_id))
        
        del self.vehicle_to_spot[license_plate]
        return True
\`\`\`

**Key Concepts**:
- **Inheritance**: Vehicle -> Car/Truck
- **Composition**: ParkingLot has many ParkingSpots
- **Encapsulation**: Spot manages its own state
- **Min Heap**: O(log N) find nearest spot

---

## Design URL Shortener

**Problem**: Design a service like bit.ly that:
- Creates short URLs from long URLs
- Redirects short URLs to original URLs
- Tracks click counts

### Approach 1: Hash-based

\`\`\`python
import hashlib

class URLShortener:
    def __init__(self):
        self.url_to_short = {}  # long -> short
        self.short_to_url = {}  # short -> long
        self.base_url = "http://short.url/"
    
    def shorten(self, long_url):
        if long_url in self.url_to_short:
            return self.base_url + self.url_to_short[long_url]
        
        # Generate short code using hash
        hash_val = hashlib.md5(long_url.encode()).hexdigest()
        short_code = hash_val[:7]  # Take first 7 chars
        
        # Handle collisions
        counter = 0
        while short_code in self.short_to_url:
            short_code = hash_val[:7] + str(counter)
            counter += 1
        
        self.url_to_short[long_url] = short_code
        self.short_to_url[short_code] = long_url
        
        return self.base_url + short_code
    
    def expand(self, short_url):
        short_code = short_url.replace(self.base_url, "")
        return self.short_to_url.get(short_code)
\`\`\`

### Approach 2: Counter-based (Better)

\`\`\`python
class URLShortener:
    def __init__(self):
        self.counter = 0
        self.short_to_url = {}
        self.url_to_short = {}
        self.base62 = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    def encode_base62(self, num):
        """Convert number to base62 string"""
        if num == 0:
            return self.base62[0]
        
        result = []
        while num:
            result.append(self.base62[num % 62])
            num //= 62
        return ''.join(reversed(result))
    
    def shorten(self, long_url):
        if long_url in self.url_to_short:
            return self.url_to_short[long_url]
        
        self.counter += 1
        short_code = self.encode_base62(self.counter)
        
        short_url = "http://short.url/" + short_code
        self.url_to_short[long_url] = short_url
        self.short_to_url[short_code] = long_url
        
        return short_url
    
    def expand(self, short_url):
        short_code = short_url.split("/")[-1]
        return self.short_to_url.get(short_code)
\`\`\`

**Why Base62?**
- 62 characters: 0-9, a-z, A-Z
- 62^7 = 3.5 trillion possible URLs with 7 characters
- Short and URL-safe (no special characters)

**Time**: O(1) for both shorten and expand  
**Space**: O(N) where N is number of URLs

### Production Considerations

1. **Database**: Store URLs in database (Redis for cache, SQL for persistence)

2. **Distributed Counter**: Use Redis INCR or database auto-increment

3. **Custom Short Codes**: Allow users to choose (e.g., bit.ly/mylink)

4. **Expiration**: Auto-delete old URLs

5. **Analytics**: Track clicks, referrers, locations

6. **Caching**: Cache popular URLs in memory/CDN

7. **Rate Limiting**: Prevent abuse (use Token Bucket!)

---

## Key System Design Concepts

### Scalability
- **Vertical**: Bigger machine (limited, expensive)
- **Horizontal**: More machines (unlimited, complex)

### Load Balancing
- Distribute requests across servers
- Round-robin, least connections, consistent hashing

### Caching
- Application cache (Redis, Memcached)
- CDN (CloudFlare, Akamai)
- Browser cache

### Database
- **SQL**: Relational, ACID, complex queries (PostgreSQL, MySQL)
- **NoSQL**: Scalable, eventual consistency, simple queries (MongoDB, Cassandra)
- **Key-Value**: Ultra-fast, simple (Redis, DynamoDB)

### CAP Theorem
You can only have 2 of 3:
- **Consistency**: All nodes see same data
- **Availability**: System always responds
- **Partition Tolerance**: System works despite network failures

---

## Interview Strategy

1. **Clarify requirements**:
   - Scale: 100 users or 100M users?
   - Read-heavy or write-heavy?
   - Consistency requirements?

2. **Start simple**: Single server, then discuss scaling

3. **Identify bottlenecks**: "Database would be bottleneck at 100M users..."

4. **Propose solutions**: "We could shard the database by user ID..."

5. **Discuss trade-offs**: "NoSQL is faster but less consistent than SQL"

6. **Draw diagrams**: Show client -> load balancer -> servers -> database

**Common Mistakes**:
- Jumping to distributed system without starting simple
- Not asking clarifying questions
- Ignoring trade-offs (everything is perfect!)
- Not considering failure modes`,
      quiz: [
        {
          id: 'q1',
          question:
            'In Parking Lot design, why do we use a min heap for available spots instead of a simple list?',
          sampleAnswer:
            'Min heap gives us O(log N) access to the nearest spot (by level, then row). With a list, we\'d need O(N) to find the nearest available spot each time. As parking lot scales (1000+ spots), this matters significantly. Heap maintains spots sorted by (level, row), so heap[0] is always the closest available spot. When we park, we pop from heap (O(log N)). When a spot becomes available, we push to heap (O(log N)). Alternative: sorted list with binary search would be O(N) for insertion/deletion. Heap is the right tool for "repeatedly get min from dynamic set" pattern. This improves user experience - customers get nearest parking.',
          keyPoints: [
            'Min heap: O(log N) get nearest spot',
            'List would be O(N) linear search',
            'Heap maintains sorted order dynamically',
            'Pop gives closest spot instantly',
            'Scales well (1000+ spots)',
          ],
        },
        {
          id: 'q2',
          question:
            'Why is Base62 encoding better than using random strings for URL shortener?',
          sampleAnswer:
            'Base62 with counter is deterministic, collision-free, and shorter. With counter: URL 1 -> "1", URL 1000 -> "g8", URL 1M -> "4c92". No collisions ever because counter is unique. Random strings need collision checking: generate random, check if exists, retry if collision - could take multiple attempts. Base62 uses 62 chars (0-9,a-z,A-Z) vs UUID\'s 36 chars (0-9,a-f), so shorter codes for same space: 62^7 = 3.5T URLs vs 36^7 = 78B. Random also needs cryptographically secure RNG (slower). Base62 is: guaranteed unique, fast (just math), shortest possible, predictable length. Only downside: sequential codes are guessable, but usually not a security concern for public URLs.',
          keyPoints: [
            'Deterministic: no collision checking',
            'Counter guarantees uniqueness',
            'Shorter: 62 chars vs 36 (UUID)',
            'Fast: just math, no random generation',
            'Predictable: know how many URLs from code length',
          ],
        },
        {
          id: 'q3',
          question:
            'How would you scale URL shortener to handle 10,000 requests per second?',
          sampleAnswer:
            'For 10K req/s: (1) Multiple application servers behind load balancer for horizontal scaling. (2) Redis cache for hot URLs (top 10% get 90% of traffic) - check cache before DB, O(1) lookups. (3) Database: Use master-replica setup - master for writes (shorten), replicas for reads (expand). Reads are 90% of traffic so this helps. (4) Generate short codes in advance and store in queue - servers pop from queue when needed, separate service refills queue. This avoids counter bottleneck. (5) Database sharding by hash of short code (e.g., short_code[0] determines shard). (6) CDN for static content and caching. (7) Rate limiting per IP (Token Bucket). (8) Monitoring and auto-scaling. Key: cache everything, scale horizontally, separate reads from writes.',
          keyPoints: [
            'Load balancer + multiple app servers',
            'Redis cache for hot URLs',
            'Master-replica DB (reads vs writes)',
            'Pre-generate codes (avoid counter bottleneck)',
            'Database sharding by short code',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question:
            'What is the main advantage of using Base62 encoding in URL shortener?',
          options: [
            'It is more secure',
            'It generates shorter codes than other encodings',
            'It is faster than all alternatives',
            'It uses less memory',
          ],
          correctAnswer: 1,
          explanation:
            'Base62 uses 62 characters (0-9, a-z, A-Z) instead of Base10 (10 chars) or Base16 (16 chars), so it generates much shorter codes. Example: 1,000,000 in Base10 = "1000000" (7 chars), Base62 = "4c92" (4 chars). More compact = shorter URLs.',
        },
        {
          id: 'mc2',
          question:
            'In Parking Lot design, what design pattern is used for Car, Truck, Motorcycle?',
          options: [
            'Singleton',
            'Factory',
            'Inheritance (polymorphism)',
            'Observer',
          ],
          correctAnswer: 2,
          explanation:
            'Car, Truck, Motorcycle all inherit from Vehicle abstract class. This is polymorphism - each vehicle type implements can_fit_in() differently. ParkingLot code works with Vehicle interface without knowing specific type.',
        },
        {
          id: 'mc3',
          question:
            'What is the time complexity of shortening a URL with counter-based approach?',
          options: ['O(1)', 'O(log N)', 'O(N)', 'O(N log N)'],
          correctAnswer: 0,
          explanation:
            'Shortening is O(1): increment counter (O(1)), encode to Base62 (O(log counter) = O(1) for practical sizes), insert to HashMap (O(1)). No collision checking needed since counter is unique.',
        },
        {
          id: 'mc4',
          question:
            'According to CAP theorem, which two properties does a typical SQL database prioritize?',
          options: [
            'Consistency and Availability',
            'Consistency and Partition Tolerance',
            'Availability and Partition Tolerance',
            'All three',
          ],
          correctAnswer: 1,
          explanation:
            'Traditional SQL databases (PostgreSQL, MySQL) prioritize Consistency and Partition Tolerance (CP). During network partition, they may sacrifice Availability to maintain consistency. NoSQL databases often choose AP (Available and Partition Tolerant) with eventual consistency.',
        },
        {
          id: 'mc5',
          question:
            'Why do we need both url_to_short and short_to_url HashMaps in URL shortener?',
          options: [
            'One HashMap is not enough memory',
            'To support both directions: shorten() needs url_to_short, expand() needs short_to_url',
            'To make it faster',
            'We only need one',
          ],
          correctAnswer: 1,
          explanation:
            'We need both for O(1) operations in both directions. shorten(long) checks url_to_short to avoid duplicates and returns existing short code. expand(short) looks up short_to_url to get original. Single HashMap would only support one direction efficiently.',
        },
      ],
    },
  ],
  keyTakeaways: [
    'Design problems test ability to combine data structures to meet multiple requirements',
    'LRU Cache uses HashMap + Doubly LinkedList for O(1) get and put operations',
    'Min Stack tracks minimum at each level with auxiliary stack',
    'Queue using Stacks achieves amortized O(1) through lazy element transfer',
    'Rate limiting: Token Bucket is industry standard, allows controlled bursts',
    'Hit Counter uses deque for O(1) sliding window timestamp management',
    'Object-oriented design: use inheritance, composition, and encapsulation properly',
    'URL Shortener: Base62 encoding with counter is better than random strings',
    'System design: start simple, identify bottlenecks, discuss trade-offs',
    'Always clarify requirements: scale, performance, consistency needs',
  ],
  relatedProblems: [
    'lru-cache',
    'min-stack',
    'implement-queue-using-stacks',
    'design-hit-counter',
  ],
};
