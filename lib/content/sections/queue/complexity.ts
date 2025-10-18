/**
 * Time & Space Complexity Section
 */

export const complexitySection = {
  id: 'complexity',
  title: 'Time & Space Complexity',
  content: `## Queue Complexity Analysis

Understanding the performance characteristics of queue operations is crucial for choosing the right implementation.

---

## Operation Complexities

### Using collections.deque (Recommended):

| Operation | Time | Explanation |
|-----------|------|-------------|
| **Enqueue** | O(1) | \`append()\` adds to rear in constant time |
| **Dequeue** | O(1) | \`popleft()\` removes from front in constant time |
| **Front/Peek** | O(1) | Access first element directly |
| **isEmpty** | O(1) | Check length |
| **Size** | O(1) | Return length |
| **Space** | O(n) | n elements stored |

**All operations are O(1)** - This is why deque is the best choice!

---

### Using List (NOT Recommended):

| Operation | Time | Explanation |
|-----------|------|-------------|
| **Enqueue** | O(1) | \`append()\` is O(1) |
| **Dequeue** | ❌ **O(n)** | \`pop(0)\` shifts all elements |
| **Front** | O(1) | \`list[0]\` is O(1) |
| **Space** | O(n) | n elements stored |

**Dequeue is O(n)** - This makes list unsuitable for queues!

---

### Using Two Stacks:

| Operation | Time | Amortized | Explanation |
|-----------|------|-----------|-------------|
| **Enqueue** | O(1) | O(1) | Push to stack_in |
| **Dequeue** | O(n) worst | **O(1)** amortized | Pop from stack_out, transfer if needed |
| **Front** | O(n) worst | **O(1)** amortized | Peek at stack_out, transfer if needed |
| **Space** | O(n) | O(n) | Two stacks |

**Amortized O(1)** - Expensive transfers are rare, averaged out over time.

---

## Amortized Analysis Example

Why is two-stacks queue O(1) amortized?

\`\`\`python
# Worst case scenario:
queue = QueueWithStacks()

# 1. Enqueue n items - O(n) total
for i in range(n):
    queue.enqueue(i)  # Each O(1)

# stack_in: [0,1,2,...,n-1]
# stack_out: []

# 2. First dequeue - O(n) because transfer all n items
queue.dequeue()  # Moves all n items from stack_in to stack_out

# stack_in: []
# stack_out: [n-1,...,2,1,0]

# 3. Next n-1 dequeues - O(1) each!
for _ in range(n-1):
    queue.dequeue()  # Just pop from stack_out

# Total cost: n enqueues + n dequeues = n + (n transfer + n pops) = 3n
# Average per operation: 3n / 2n = 1.5 = O(1) amortized!
\`\`\`

**Key Insight:** Each element is moved at most twice (stack_in → stack_out), so total work is O(1) per element on average.

---

## Space Complexity

### Regular Queue:
\`\`\`python
space = O(n)  # n elements in queue
\`\`\`

### BFS with Queue:
\`\`\`python
def bfs(root):
    queue = deque([root])  # Space: O(width of tree)
    # ...

# Worst case: O(n) if tree is very wide
# Example: complete binary tree last level has n/2 nodes
\`\`\`

### Queue in Sliding Window:
\`\`\`python
def sliding_window_max(nums, k):
    dq = deque()  # Space: O(k) max
    # ...

# Space is bounded by window size k, not array size n
\`\`\`

---

## Performance Tips

**1. Use deque, not list:**
\`\`\`python
# ❌ BAD - O(n) dequeue
queue = []
queue.append(1)
first = queue.pop(0)  # O(n)!

# ✅ GOOD - O(1) dequeue
from collections import deque
queue = deque()
queue.append(1)
first = queue.popleft()  # O(1)!
\`\`\`

**2. Pre-allocate for fixed-size queues:**
\`\`\`python
# For circular queue, pre-allocate array
queue = [None] * capacity  # Better than dynamic growth
\`\`\`

**3. Batch operations when possible:**
\`\`\`python
# If adding multiple items
queue.extend([1, 2, 3, 4])  # More efficient than 4 individual appends
\`\`\`

---

## Common Complexity Mistakes

❌ **Mistake 1:** Thinking list is O(1) for queue
\`\`\`python
queue = []
queue.pop(0)  # This is O(n), not O(1)!
\`\`\`

❌ **Mistake 2:** Forgetting BFS space complexity
\`\`\`python
# BFS can have O(n) space if tree is wide!
# Not always O(log n) like DFS
\`\`\`

❌ **Mistake 3:** Creating new deque in loop
\`\`\`python
# ❌ BAD - O(n²) total
for i in range(n):
    queue = deque()  # Creates new deque each time!

# ✅ GOOD - O(n) total
queue = deque()
for i in range(n):
    # Reuse same queue
\`\`\`

---

## Complexity Cheat Sheet

**Basic Operations:**
- Enqueue: O(1) with deque
- Dequeue: O(1) with deque
- Peek: O(1)

**BFS Complexity:**
- Time: O(V + E) for graphs (visit all vertices and edges)
- Space: O(W) where W is maximum width

**Level-Order Traversal:**
- Time: O(n) - visit each node once
- Space: O(w) - w is maximum width of level

**Sliding Window:**
- Time: O(n) - process each element once
- Space: O(k) - k is window size

**Remember:** Always use \`collections.deque\` for O(1) operations on both ends!`,
};
