/**
 * Queue Interview Strategy Section
 */

export const interviewstrategySection = {
  id: 'interview-strategy',
  title: 'Queue Interview Strategy',
  content: `## Mastering Queue Interview Problems

Queues are common in interviews, especially for BFS and level-order traversal problems.

---

## Recognizing Queue Problems

**Look for these keywords:**

1. **"Level by level"** → Level-order traversal with queue
2. **"Breadth-first"** → BFS with queue
3. **"Order of arrival"** → FIFO queue
4. **"First-come, first-served"** → Queue scheduling
5. **"Shortest path" (unweighted)** → BFS with queue
6. **"Minimum depth/steps"** → BFS with queue
7. **"Process in order"** → Queue

---

## Common Queue Interview Questions

### Easy:
1. Implement Queue using Stacks
2. Implement Stack using Queues
3. Binary Tree Level Order Traversal
4. Number of Recent Calls
5. First Unique Character in String (with queue)

### Medium:
6. Binary Tree Right Side View
7. Binary Tree Zigzag Level Order
8. Rotting Oranges (BFS)
9. Sliding Window Maximum
10. Design Circular Queue
11. Shortest Path in Binary Matrix
12. Walls and Gates

### Hard:
13. Sliding Window Median
14. Max Value of Equation
15. Shortest Path to Get All Keys

---

## Problem-Solving Framework

### Step 1: Identify if Queue is Needed

Ask yourself:
- Do I need to process items in order they arrive? (FIFO)
- Am I doing BFS or level-order traversal?
- Do I need to track a sliding window?
- Is this about shortest path (unweighted)?

### Step 2: Choose Queue Type

\`\`\`python
from collections import deque

# Most common: regular deque
queue = deque()

# Circular queue with size limit
class CircularQueue:
    def __init__(self, k):
        self.queue = [None] * k
        # ...

# Priority queue (not really a queue!)
import heapq
pq = []  # Use heap

# Deque for sliding window
dq = deque()  # Can use both ends
\`\`\`

### Step 3: Write Solution Template

**BFS Template:**
\`\`\`python
from collections import deque

def bfs (start):
    """Standard BFS template"""
    queue = deque([start])
    visited = {start}
    
    while queue:
        current = queue.popleft()
        
        # Process current
        # ...
        
        # Add neighbors
        for neighbor in get_neighbors (current):
            if neighbor not in visited:
                visited.add (neighbor)
                queue.append (neighbor)
    
    return result
\`\`\`

**Level-Order Template:**
\`\`\`python
def level_order (root):
    """Process tree level by level"""
    if not root:
        return []
    
    queue = deque([root])
    result = []
    
    while queue:
        level_size = len (queue)  # Key: capture level size
        level = []
        
        for _ in range (level_size):  # Process entire level
            node = queue.popleft()
            level.append (node.val)
            
            if node.left:
                queue.append (node.left)
            if node.right:
                queue.append (node.right)
        
        result.append (level)
    
    return result
\`\`\`

**Sliding Window Template:**
\`\`\`python
def sliding_window (nums, k):
    """Sliding window with deque"""
    dq = deque()
    result = []
    
    for i in range (len (nums)):
        # Remove out-of-window elements
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        
        # Maintain deque property (e.g., decreasing for max)
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        
        dq.append (i)
        
        # Add to result once window is full
        if i >= k - 1:
            result.append (nums[dq[0]])
    
    return result
\`\`\`

---

## Communication Tips

**What to say during the interview:**

1. **Identify the pattern:**
   > "This is a level-order traversal problem, so I'll use a queue for BFS."

2. **Explain FIFO:**
   > "I'm using a queue because we need to process nodes in the order we discover them."

3. **Discuss implementation:**
   > "I'll use collections.deque for O(1) enqueue and dequeue operations."

4. **Walk through example:**
   > "Starting with the root in the queue, we dequeue it, process it, then enqueue its children..."

5. **Mention complexity:**
   > "Time is O(n) since we visit each node once. Space is O(w) where w is the maximum width."

---

## Common Mistakes to Avoid

❌ **Mistake 1:** Using list instead of deque
\`\`\`python
# ❌ BAD - O(n) per dequeue
queue = []
queue.pop(0)

# ✅ GOOD - O(1) per dequeue
from collections import deque
queue = deque()
queue.popleft()
\`\`\`

❌ **Mistake 2:** Forgetting to track level size
\`\`\`python
# ❌ BAD - can't separate levels
while queue:
    node = queue.popleft()
    # Process and add children

# ✅ GOOD - process level by level
while queue:
    level_size = len (queue)  # Capture current level
    for _ in range (level_size):
        node = queue.popleft()
        # Process and add children
\`\`\`

❌ **Mistake 3:** Not handling empty queue
\`\`\`python
# ❌ BAD - may raise exception
item = queue.popleft()

# ✅ GOOD - check first
if queue:
    item = queue.popleft()
\`\`\`

❌ **Mistake 4:** Confusing queue with stack
\`\`\`python
# ❌ BAD - this is a stack (LIFO)
queue.append (item)
queue.pop()  # Removes last item

# ✅ GOOD - this is a queue (FIFO)
queue.append (item)
queue.popleft()  # Removes first item
\`\`\`

---

## Complexity Analysis Tips

**BFS/Level-Order:**
- Time: O(n) where n is number of nodes
- Space: O(w) where w is maximum width
- "w can be O(n) for wide trees"

**Sliding Window with Deque:**
- Time: O(n) - each element added/removed once
- Space: O(k) - deque size bounded by window

**Queue Implementation:**
- Time: O(1) for all operations (with deque)
- Space: O(n) for n elements

---

## Practice Strategy

**Week 1: Basics**
- Implement queue with different methods
- Basic enqueue/dequeue problems

**Week 2: BFS**
- Binary tree BFS
- Level-order traversal
- Graph BFS

**Week 3: Advanced Patterns**
- Sliding window with deque
- Circular queue
- Queue with special properties

**Week 4: Hard Problems**
- Multiple queues
- Queue + other data structures
- Optimization problems

---

## Final Checklist

Before submitting, verify:

- [ ] Using deque, not list (unless required otherwise)
- [ ] Handling empty queue correctly
- [ ] Checking queue.popleft() not queue.pop()
- [ ] For level-order: capturing level size before loop
- [ ] Initializing queue with starting nodes
- [ ] Marking nodes as visited (for graphs)
- [ ] Returning correct result format
- [ ] Complexity analysis is correct

**Remember:** Queue + BFS is one of the most important patterns in coding interviews. Master level-order traversal and you'll ace many tree/graph problems!`,
};
