/**
 * Common Queue Problems & Patterns Section
 */

export const commonproblemsSection = {
  id: 'common-problems',
  title: 'Common Queue Problems & Patterns',
  content: `## Essential Queue Patterns

Recognizing these patterns helps you identify when to use a queue.

---

## Pattern 1: Breadth-First Search (BFS)

**Most important queue application!**

BFS explores nodes level by level, using a queue to track which nodes to visit next.

\`\`\`python
from collections import deque

def bfs_tree (root):
    """BFS traversal of binary tree"""
    if not root:
        return []
    
    result = []
    queue = deque([root])  # Start with root
    
    while queue:
        # Process current level
        node = queue.popleft()  # FIFO!
        result.append (node.val)
        
        # Add children to queue (next level)
        if node.left:
            queue.append (node.left)
        if node.right:
            queue.append (node.right)
    
    return result

# Why queue works:
# Process nodes in order discovered
# Level 0: [root]
# Level 1: [root.left, root.right]
# Level 2: [children of level 1 nodes]
\`\`\`

**Key Insight:** Queue ensures we process all nodes at depth d before any node at depth d+1.

---

## Pattern 2: Level-Order Traversal

Process tree level by level, keeping levels separate.

\`\`\`python
def level_order (root):
    """Return list of lists, one per level"""
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len (queue)  # Current level size
        level = []
        
        # Process entire level
        for _ in range (level_size):
            node = queue.popleft()
            level.append (node.val)
            
            # Add next level
            if node.left:
                queue.append (node.left)
            if node.right:
                queue.append (node.right)
        
        result.append (level)
    
    return result

# Example:
#       1
#      / \\
#     2   3
#    / \\
#   4   5
#
# Returns: [[1], [2, 3], [4, 5]]
\`\`\`

**Pattern:** Use \`len (queue)\` to know how many nodes are at current level.

---

## Pattern 3: Sliding Window Maximum

Use deque to efficiently track maximum in sliding window.

\`\`\`python
from collections import deque

def max_sliding_window (nums, k):
    """Find max in each window of size k"""
    result = []
    dq = deque()  # Store indices
    
    for i in range (len (nums)):
        # Remove indices outside window
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        
        # Remove smaller elements (they can't be max)
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        
        dq.append (i)
        
        # Add to result once window is full
        if i >= k - 1:
            result.append (nums[dq[0]])  # Front is max
    
    return result

# Example: nums = [1,3,-1,-3,5,3,6,7], k = 3
# Result: [3, 3, 5, 5, 6, 7]
\`\`\`

**Key:** Deque maintains decreasing order - front is always maximum.

---

## Pattern 4: Recent Counter

Count requests in last N milliseconds using queue.

\`\`\`python
from collections import deque

class RecentCounter:
    """Count requests in last 3000ms"""
    
    def __init__(self):
        self.requests = deque()
    
    def ping (self, t):
        """Add request at time t, return count in [t-3000, t]"""
        self.requests.append (t)
        
        # Remove requests older than t - 3000
        while self.requests and self.requests[0] < t - 3000:
            self.requests.popleft()
        
        return len (self.requests)

# Usage
counter = RecentCounter()
print(counter.ping(1))     # 1 (request at t=1)
print(counter.ping(100))   # 2 (requests at t=1, 100)
print(counter.ping(3001))  # 3 (all within window)
print(counter.ping(3002))  # 3 (t=1 is now outside window)
\`\`\`

**Pattern:** Use queue as sliding time window - remove old entries as you add new ones.

---

## Pattern 5: Implement Stack Using Queues

Classic interview problem!

\`\`\`python
from collections import deque

class StackUsingQueues:
    """Implement stack using two queues"""
    
    def __init__(self):
        self.q1 = deque()
        self.q2 = deque()
    
    def push (self, x):
        """Push to stack - O(1)"""
        self.q1.append (x)
    
    def pop (self):
        """Pop from stack - O(n)"""
        # Move all but last to q2
        while len (self.q1) > 1:
            self.q2.append (self.q1.popleft())
        
        # Remove last (top of stack)
        result = self.q1.popleft()
        
        # Swap queues
        self.q1, self.q2 = self.q2, self.q1
        
        return result
    
    def top (self):
        """Peek at top - O(n)"""
        # Similar to pop but add back
        while len (self.q1) > 1:
            self.q2.append (self.q1.popleft())
        
        result = self.q1[0]
        self.q2.append (self.q1.popleft())
        
        self.q1, self.q2 = self.q2, self.q1
        
        return result

# Usage
stack = StackUsingQueues()
stack.push(1)
stack.push(2)
stack.push(3)
print(stack.pop())  # 3 (LIFO!)
print(stack.top())  # 2
\`\`\`

**Trick:** Use two queues and rotate to access last element.

---

## Pattern Recognition Guide

**Use Queue when:**
- ✅ Processing in order of arrival
- ✅ BFS / level-order traversal
- ✅ Sliding window with FIFO
- ✅ Task scheduling
- ✅ Buffering / streaming

**Use Stack when:**
- ❌ Need LIFO (last-in-first-out)
- ❌ DFS / backtracking
- ❌ Function calls / recursion
- ❌ Undo/redo (if single-ended)

**Use Deque when:**
- ✅ Need efficiency at both ends
- ✅ Sliding window maximum/minimum
- ✅ Palindrome checking
- ✅ Can't decide between queue and stack!

**Remember:** If the problem mentions "level by level", "breadth-first", "order of arrival", or "FIFO" → think Queue!`,
};
