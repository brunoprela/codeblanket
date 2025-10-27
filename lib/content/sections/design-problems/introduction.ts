/**
 * Introduction to Design Problems Section
 */

export const introductionSection = {
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

**Common Design Problem Types:**1. **Caching Systems** (LRU Cache, LFU Cache)
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
- What\'s the expected performance? (O(1)? O(log N)?)
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
    
    def get (self, key: int) -> int:
        """Get value, return -1 if not exists"""
        pass
    
    def put (self, key: int, value: int) -> None:
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
    
    def push (self, val):
        self.stack.append (val)
        # Push current min (could be this val)
        min_val = min (val, self.min_stack[-1] if self.min_stack else val)
        self.min_stack.append (min_val)
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
    
    def enqueue (self, x):
        self.stack1.append (x)
    
    def dequeue (self):
        if not self.stack2:
            # Move all from stack1 to stack2 (reverses order)
            while self.stack1:
                self.stack2.append (self.stack1.pop())
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
    
    def hit (self, timestamp):
        self.hits.append (timestamp)
    
    def getHits (self, timestamp):
        # Remove hits older than 300 seconds
        while self.hits and self.hits[0] <= timestamp - 300:
            self.hits.popleft()
        return len (self.hits)
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
};
