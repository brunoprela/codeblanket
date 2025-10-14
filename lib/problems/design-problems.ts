import { Problem } from '@/lib/types';

export const designProblemsProblems: Problem[] = [
  {
    id: 'lru-cache',
    title: 'LRU Cache',
    difficulty: 'Medium',
    topic: 'Design Problems',
    description: `Design a data structure that follows the constraints of a **Least Recently Used (LRU) cache**.

Implement the \`LRUCache\` class:

- \`LRUCache(int capacity)\` Initialize the LRU cache with **positive** size \`capacity\`.
- \`int get(int key)\` Return the value of the \`key\` if the key exists, otherwise return \`-1\`.
- \`void put(int key, int value)\` Update the value of the \`key\` if the \`key\` exists. Otherwise, add the \`key-value\` pair to the cache. If the number of keys exceeds the \`capacity\` from this operation, **evict** the least recently used key.

The functions \`get\` and \`put\` must each run in **O(1)** average time complexity.`,
    hints: [
      'Need O(1) operations - what data structure gives O(1) access?',
      'HashMap for lookup, but how to track order?',
      'Doubly linked list allows O(1) removal from any position',
      'Combine HashMap (key->node) with Doubly LinkedList (LRU order)',
      'Use dummy head and tail to simplify edge cases',
    ],
    approach: `## Intuition

LRU Cache needs:
1. **O(1) access** to any element by key → **HashMap**
2. **O(1) update** of access order → **Doubly Linked List**

Neither structure alone suffices. **Combined**, they give O(1) for both operations.

---

## Why Doubly Linked List?

When we access a key, we need to move it to front (most recent). This requires:
1. **Remove** node from current position - O(1) with doubly linked list
2. **Insert** at front - O(1)

Singly linked list would need O(N) to find previous node for removal.

---

## Approach

**Data Structures:**
- \`HashMap<key, Node>\`: Fast lookup by key
- \`Doubly LinkedList\`: Maintain LRU order
  - Head: Most recently used
  - Tail: Least recently used

**Operations:**

### get(key):
1. If key not in HashMap → return -1
2. Find node via HashMap (O(1))
3. Move node to front (O(1))
4. Return node.value

### put(key, value):
1. If key exists:
   - Update value
   - Move to front
2. If key is new:
   - Create node
   - Add to HashMap
   - Insert at front
   - If over capacity:
     - Remove tail node (LRU)
     - Delete from HashMap

**Dummy Head/Tail Trick:**
Use dummy nodes to avoid null checks:
\`\`\`
head <-> node1 <-> node2 <-> tail
\`\`\`
Always insert after head, remove before tail.

---

## Time Complexity: O(1) for both get and put
- HashMap operations: O(1)
- LinkedList add/remove with node reference: O(1)

## Space Complexity: O(capacity)
- HashMap: O(capacity)  
- LinkedList: O(capacity)`,
    testCases: [
      {
        input: [
          ['LRUCache', 2],
          ['put', 1, 1],
          ['put', 2, 2],
          ['get', 1],
          ['put', 3, 3],
          ['get', 2],
          ['put', 4, 4],
          ['get', 1],
          ['get', 3],
          ['get', 4],
        ],
        expected: [null, null, null, 1, null, -1, null, -1, 3, 4],
      },
    ],
    solution: `class ListNode:
    """Node in doubly linked list"""
    def __init__(self, key=0, val=0):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # key -> ListNode
        
        # Dummy head and tail for easy insertion/deletion
        self.head = ListNode()
        self.tail = ListNode()
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _remove(self, node: ListNode) -> None:
        """Remove node from linked list - O(1)"""
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node
    
    def _add_to_front(self, node: ListNode) -> None:
        """Add node right after head (most recent) - O(1)"""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node
    
    def get(self, key: int) -> int:
        """Get value and mark as recently used"""
        if key not in self.cache:
            return -1
        
        node = self.cache[key]
        # Move to front (most recently used)
        self._remove(node)
        self._add_to_front(node)
        return node.val
    
    def put(self, key: int, value: int) -> None:
        """Add/update key-value pair"""
        if key in self.cache:
            # Key exists - update and move to front
            node = self.cache[key]
            node.val = value
            self._remove(node)
            self._add_to_front(node)
        else:
            # New key
            new_node = ListNode(key, value)
            self.cache[key] = new_node
            self._add_to_front(new_node)
            
            # Check capacity
            if len(self.cache) > self.capacity:
                # Evict LRU (node before tail)
                lru = self.tail.prev
                self._remove(lru)
                del self.cache[lru.key]  # Don't forget HashMap!

# Example usage:
# cache = LRUCache(2)
# cache.put(1, 1)  # cache = {1=1}
# cache.put(2, 2)  # cache = {1=1, 2=2}
# cache.get(1)     # returns 1, cache = {2=2, 1=1}
# cache.put(3, 3)  # evicts key 2, cache = {1=1, 3=3}
# cache.get(2)     # returns -1 (not found)`,
    timeComplexity: 'O(1) for both get() and put()',
    spaceComplexity:
      'O(capacity) - HashMap + LinkedList both store capacity elements',
    patterns: ['HashMap', 'Doubly Linked List', 'Design', 'Cache'],
    companies: [
      'Amazon',
      'Google',
      'Microsoft',
      'Facebook',
      'Apple',
      'Bloomberg',
      'Uber',
      'LinkedIn',
    ],
  },
  {
    id: 'min-stack',
    title: 'Min Stack',
    difficulty: 'Medium',
    topic: 'Design Problems',
    description: `Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

Implement the \`MinStack\` class:

- \`MinStack()\` initializes the stack object.
- \`void push(int val)\` pushes the element \`val\` onto the stack.
- \`void pop()\` removes the element on the top of the stack.
- \`int top()\` gets the top element of the stack.
- \`int getMin()\` retrieves the minimum element in the stack.

You must implement a solution with **O(1)** time complexity for each function.`,
    hints: [
      'Cannot iterate through stack to find min (would be O(N))',
      'Track minimum at each level - what was min when this element was pushed?',
      'Use a second stack to track minimums',
      'Alternatively, store (value, current_min) tuples in single stack',
    ],
    approach: `## Intuition

Regular stack operations are O(1), but finding minimum typically requires O(N) scan. How can we make getMin() O(1)?

**Key Insight**: Track the minimum *at each level* as we build the stack.

---

## Approach 1: Two Stacks

Maintain two parallel stacks:
1. **Main stack**: Stores all values
2. **Min stack**: At each level, stores the minimum value seen so far

**Example:**
\`\`\`
push(3): main=[3],    min=[3]  # min so far: 3
push(5): main=[3,5],  min=[3,3]  # min so far: still 3
push(1): main=[3,5,1], min=[3,3,1]  # min so far: 1
push(2): main=[3,5,1,2], min=[3,3,1,1]  # min so far: still 1
pop():   main=[3,5,1], min=[3,3,1]
getMin(): returns min[-1] = 1
\`\`\`

When we pop, we pop from both stacks simultaneously, so min_stack.top() always reflects the minimum of remaining elements.

---

## Approach 2: Single Stack with Tuples

Each element stores \`(value, min_at_this_level)\`:

\`\`\`
push(3): stack=[(3,3)]
push(5): stack=[(3,3), (5,3)]  # min is 3
push(1): stack=[(3,3), (5,3), (1,1)]  # min is now 1
\`\`\`

**Trade-off**: Simpler (one data structure) but each element uses more memory.

---

## Time Complexity: O(1) for all operations
- push/pop/top: O(1) standard stack operations
- getMin: O(1) - just return min_stack.top() or current tuple

## Space Complexity: O(N)
- Approach 1: Two stacks of size N
- Approach 2: One stack of size N (but tuples)`,
    testCases: [
      {
        input: [
          ['MinStack'],
          ['push', -2],
          ['push', 0],
          ['push', -3],
          ['getMin'],
          ['pop'],
          ['top'],
          ['getMin'],
        ],
        expected: [null, null, null, null, -3, null, 0, -2],
      },
    ],
    solution: `# Approach 1: Two Stacks (Recommended)
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []
    
    def push(self, val: int) -> None:
        self.stack.append(val)
        # Push current minimum to min_stack
        min_val = min(val, self.min_stack[-1] if self.min_stack else val)
        self.min_stack.append(min_val)
    
    def pop(self) -> None:
        self.stack.pop()
        self.min_stack.pop()  # Keep in sync!
    
    def top(self) -> int:
        return self.stack[-1]
    
    def getMin(self) -> int:
        return self.min_stack[-1]  # O(1)!


# Approach 2: Single Stack with Tuples
class MinStack:
    def __init__(self):
        self.stack = []  # Store (val, min_so_far) tuples
    
    def push(self, val: int) -> None:
        if not self.stack:
            self.stack.append((val, val))
        else:
            current_min = min(val, self.stack[-1][1])
            self.stack.append((val, current_min))
    
    def pop(self) -> None:
        self.stack.pop()
    
    def top(self) -> int:
        return self.stack[-1][0]
    
    def getMin(self) -> int:
        return self.stack[-1][1]


# Approach 3: Optimized (Space) - Only store min when it changes
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []  # Store (min_val, count)
    
    def push(self, val: int) -> None:
        self.stack.append(val)
        
        if not self.min_stack or val < self.min_stack[-1][0]:
            # New minimum
            self.min_stack.append((val, 1))
        elif val == self.min_stack[-1][0]:
            # Another instance of current minimum
            self.min_stack[-1] = (val, self.min_stack[-1][1] + 1)
    
    def pop(self) -> None:
        val = self.stack.pop()
        
        if val == self.min_stack[-1][0]:
            # Popping a minimum
            if self.min_stack[-1][1] == 1:
                self.min_stack.pop()
            else:
                self.min_stack[-1] = (val, self.min_stack[-1][1] - 1)
    
    def top(self) -> int:
        return self.stack[-1]
    
    def getMin(self) -> int:
        return self.min_stack[-1][0]`,
    timeComplexity: 'O(1) for all operations - push, pop, top, getMin',
    spaceComplexity: 'O(N) where N is number of elements in stack',
    patterns: ['Stack', 'Design', 'Monotonic'],
    companies: ['Amazon', 'Microsoft', 'Google', 'Bloomberg', 'Adobe'],
  },
  {
    id: 'queue-using-stacks',
    title: 'Implement Queue using Stacks',
    difficulty: 'Easy',
    topic: 'Design Problems',
    description: `Implement a first in first out (FIFO) queue using only two stacks. The implemented queue should support all the functions of a normal queue (\`push\`, \`peek\`, \`pop\`, and \`empty\`).

Implement the \`MyQueue\` class:

- \`void push(int x)\` Pushes element x to the back of the queue.
- \`int pop()\` Removes the element from the front of the queue and returns it.
- \`int peek()\` Returns the element at the front of the queue.
- \`boolean empty()\` Returns \`true\` if the queue is empty, \`false\` otherwise.

**Notes:**
- You must use **only** standard operations of a stack, which means only \`push to top\`, \`peek/pop from top\`, \`size\`, and \`is empty\` operations are valid.`,
    hints: [
      'Stack is LIFO, Queue is FIFO - opposite orders!',
      'Use one stack for input (enqueue), one for output (dequeue)',
      'Transfer from input to output reverses the order',
      'Lazy transfer - only when output stack is empty',
      'Each element is moved at most once → amortized O(1)',
    ],
    approach: `## Intuition

Stack = LIFO (Last-In-First-Out)  
Queue = FIFO (First-In-First-Out)

How to get FIFO from LIFO? **Reverse twice!**

---

## Approach: Two Stacks

Use two stacks:
- **stack_in**: For enqueue operations
- **stack_out**: For dequeue operations

**Key Idea**: Transferring from stack_in to stack_out reverses the order!

### Example:

\`\`\`
enqueue(1): stack_in=[1], stack_out=[]
enqueue(2): stack_in=[1,2], stack_out=[]
enqueue(3): stack_in=[1,2,3], stack_out=[]

dequeue():
  Transfer: stack_in=[], stack_out=[3,2,1]  # Reversed!
  Pop from out: returns 1, stack_out=[3,2]

enqueue(4): stack_in=[4], stack_out=[3,2]

dequeue():
  stack_out not empty, just pop: returns 2, stack_out=[3]

dequeue():
  returns 3, stack_out=[]

dequeue():
  Transfer: stack_in=[], stack_out=[4]
  returns 4
\`\`\`

**Lazy Transfer**: Only move elements when stack_out is empty. This ensures each element is transferred exactly once.

---

## Why Amortized O(1)?

- Individual dequeue might take O(N) (transfer N elements)
- But each element is transferred exactly once from in → out
- Over N operations, total work = O(N) → O(1) average per operation

**Analysis:**
- N enqueues: O(N) total
- N dequeues with lazy transfer: O(N) total (each element moved once)
- Total: O(2N) = O(1) amortized per operation

---

## Time Complexity:
- push: O(1)
- pop/peek: Amortized O(1)

## Space Complexity: O(N) where N is number of elements`,
    testCases: [
      {
        input: [
          ['MyQueue'],
          ['push', 1],
          ['push', 2],
          ['peek'],
          ['pop'],
          ['empty'],
        ],
        expected: [null, null, null, 1, 1, false],
      },
    ],
    solution: `class MyQueue:
    def __init__(self):
        self.stack_in = []   # For enqueue
        self.stack_out = []  # For dequeue
    
    def push(self, x: int) -> None:
        """Add element to back of queue - O(1)"""
        self.stack_in.append(x)
    
    def _transfer(self) -> None:
        """Transfer elements from in to out (reverses order)"""
        while self.stack_in:
            self.stack_out.append(self.stack_in.pop())
    
    def pop(self) -> int:
        """Remove and return front element - Amortized O(1)"""
        if not self.stack_out:
            self._transfer()  # Lazy transfer
        return self.stack_out.pop() if self.stack_out else None
    
    def peek(self) -> int:
        """Return front element without removing - Amortized O(1)"""
        if not self.stack_out:
            self._transfer()
        return self.stack_out[-1] if self.stack_out else None
    
    def empty(self) -> bool:
        """Check if queue is empty - O(1)"""
        return not self.stack_in and not self.stack_out

# Example usage:
# queue = MyQueue()
# queue.push(1)  # in=[1], out=[]
# queue.push(2)  # in=[1,2], out=[]
# queue.peek()   # returns 1, transfers: in=[], out=[2,1]
# queue.pop()    # returns 1, out=[2]
# queue.empty()  # returns False`,
    timeComplexity:
      'push: O(1), pop/peek: Amortized O(1) - each element moved at most once',
    spaceComplexity:
      'O(N) where N is total number of elements across both stacks',
    patterns: ['Stack', 'Queue', 'Design', 'Amortized Analysis'],
    companies: ['Bloomberg', 'Amazon', 'Microsoft', 'Apple'],
  },
  {
    id: 'stack-using-queues',
    title: 'Implement Stack using Queues',
    difficulty: 'Easy',
    topic: 'Design Problems',
    description: `Implement a last-in-first-out (LIFO) stack using only two queues. The implemented stack should support all the functions of a normal stack (\`push\`, \`top\`, \`pop\`, and \`empty\`).

Implement the \`MyStack\` class:

- \`void push(int x)\` Pushes element x to the top of the stack.
- \`int pop()\` Removes the element on the top of the stack and returns it.
- \`int top()\` Returns the element on the top of the stack.
- \`boolean empty()\` Returns \`true\` if the stack is empty, \`false\` otherwise.

**Follow-up**: Can you implement the stack using only one queue?`,
    hints: [
      'Queue is FIFO, Stack is LIFO - need to reverse order',
      'After adding new element, rotate all previous elements to come after it',
      'Single queue: after push, rotate queue so new element is at front',
      'Rotation: for size-1 times, dequeue and enqueue',
    ],
    approach: `## Intuition

Queue = FIFO (First-In-First-Out)  
Stack = LIFO (Last-In-First-Out)

To implement stack, newest element must be dequeued first. We need to **reorder** the queue after each push.

---

## Approach 1: Single Queue (Elegant!)

**Key Idea**: After pushing element X, rotate the queue so X is at front.

### Example:

\`\`\`
push(1): q=[1]
  Rotate 0 times

push(2): q=[1,2]
  Rotate 1 time: dequeue 1, enqueue 1
  Result: q=[2,1]

push(3): q=[2,1,3]
  Rotate 2 times: dequeue 2, enqueue 2 → [1,3,2]
                  dequeue 1, enqueue 1 → [3,2,1]
  Result: q=[3,2,1]

pop(): dequeue → returns 3, q=[2,1]  # LIFO maintained!
\`\`\`

**Rotation logic**: After push, rotate \`size-1\` times:
\`\`\`python
# After pushing x:
for _ in range(len(q) - 1):
    q.append(q.popleft())  # Move front to back
\`\`\`

---

## Approach 2: Two Queues

Use two queues, always keep data in q1:

**push(x)**:
1. Add x to q2
2. Move all from q1 to q2 (x is now at front)
3. Swap q1 and q2

---

## Time Complexity:
- **Approach 1**: push O(N), pop/top O(1)
- **Approach 2**: push O(N), pop/top O(1)

## Space Complexity: O(N)

**Comparison with Queue using Stacks:**
- Queue using Stacks: Amortized O(1) for all ops
- Stack using Queues: O(N) push, O(1) pop
- **Trade-off**: Made push expensive to keep pop cheap`,
    testCases: [
      {
        input: [
          ['MyStack'],
          ['push', 1],
          ['push', 2],
          ['top'],
          ['pop'],
          ['empty'],
        ],
        expected: [null, null, null, 2, 2, false],
      },
    ],
    solution: `from collections import deque

# Approach 1: Single Queue (Recommended)
class MyStack:
    def __init__(self):
        self.q = deque()
    
    def push(self, x: int) -> None:
        """Add element to top - O(N)"""
        self.q.append(x)
        # Rotate: move all previous elements to back
        for _ in range(len(self.q) - 1):
            self.q.append(self.q.popleft())
    
    def pop(self) -> int:
        """Remove and return top - O(1)"""
        return self.q.popleft() if self.q else None
    
    def top(self) -> int:
        """Return top without removing - O(1)"""
        return self.q[0] if self.q else None
    
    def empty(self) -> bool:
        """Check if empty - O(1)"""
        return len(self.q) == 0


# Approach 2: Two Queues
class MyStack:
    def __init__(self):
        self.q1 = deque()
        self.q2 = deque()
    
    def push(self, x: int) -> None:
        """Add element to top - O(N)"""
        # Add to q2
        self.q2.append(x)
        # Move all from q1 to q2 (x is now at front)
        while self.q1:
            self.q2.append(self.q1.popleft())
        # Swap names
        self.q1, self.q2 = self.q2, self.q1
    
    def pop(self) -> int:
        """Remove and return top - O(1)"""
        return self.q1.popleft() if self.q1 else None
    
    def top(self) -> int:
        """Return top without removing - O(1)"""
        return self.q1[0] if self.q1 else None
    
    def empty(self) -> bool:
        """Check if empty - O(1)"""
        return len(self.q1) == 0

# Example usage:
# stack = MyStack()
# stack.push(1)  # q=[1]
# stack.push(2)  # q=[2,1] after rotation
# stack.top()    # returns 2
# stack.pop()    # returns 2, q=[1]
# stack.empty()  # returns False`,
    timeComplexity: 'push: O(N) due to rotation, pop/top/empty: O(1)',
    spaceComplexity: 'O(N) where N is number of elements',
    patterns: ['Queue', 'Stack', 'Design'],
    companies: ['Bloomberg', 'Amazon', 'Microsoft'],
  },
  {
    id: 'hit-counter',
    title: 'Design Hit Counter',
    difficulty: 'Medium',
    topic: 'Design Problems',
    description: `Design a hit counter which counts the number of hits received in the past **5 minutes** (i.e., the past **300 seconds**).

Your system should accept a \`timestamp\` parameter (in seconds granularity), and you may assume that calls are being made to the system in chronological order (i.e., \`timestamp\` is monotonically increasing). Several hits may arrive at the same \`timestamp\`.

Implement the \`HitCounter\` class:

- \`HitCounter()\` Initializes the object of the hit counter system.
- \`void hit(int timestamp)\` Records a hit that happened at \`timestamp\` (in seconds). Several hits may happen at the same \`timestamp\`.
- \`int getHits(int timestamp)\` Returns the number of hits in the past 5 minutes from \`timestamp\` (i.e., from \`timestamp - 300\` to \`timestamp\`).`,
    hints: [
      'Need to remove old timestamps (older than 300 seconds)',
      'Deque allows O(1) removal from front (old timestamps)',
      'Optimization: store (timestamp, count) instead of individual hits',
      'Alternative: bucket timestamps into 300 buckets for O(1) space',
    ],
    approach: `## Intuition

Track hits in a **sliding window** of 300 seconds. Old hits should be automatically removed.

---

## Approach 1: Deque (Simple, Exact)

Store all hit timestamps in deque:
- **hit(t)**: Append timestamp to deque - O(1)
- **getHits(t)**: Remove timestamps ≤ t-300 from front, return count - O(old_hits)

\`\`\`
hit(1):   deque=[1]
hit(2):   deque=[1,2]
hit(3):   deque=[1,2,3]
getHits(4): Remove none, return 3
getHits(300): Remove none, return 3
hit(301): deque=[1,2,3,301]
getHits(301): Remove 1 (301-300=1), deque=[2,3,301], return 3
\`\`\`

**Pros**: Exact count, simple  
**Cons**: Memory grows with hit count

---

## Approach 2: Bucketing (O(1) Space)

Use 300 buckets (one per second):
- buckets[i] = count of hits at timestamp % 300
- timestamps[i] = last timestamp that updated bucket i

\`\`\`python
def hit(timestamp):
    idx = timestamp % 300
    if timestamps[idx] != timestamp:
        # Old window, reset bucket
        timestamps[idx] = timestamp
        buckets[idx] = 1
    else:
        buckets[idx] += 1

def getHits(timestamp):
    total = 0
    for i in range(300):
        if timestamp - timestamps[i] < 300:
            total += buckets[i]
    return total
\`\`\`

**Pros**: O(300) = O(1) space  
**Cons**: getHits() scans all 300 buckets

---

## Approach 3: Hybrid (Best)

Store (timestamp, count) pairs instead of individual timestamps:

\`\`\`python
# If 1000 hits at t=5, store (5, 1000) once
# Not 1000 individual entries
\`\`\`

**Pros**: Compact if many hits per second, exact count  
**Cons**: Still O(N) in worst case (one hit per second)

---

## Time Complexity:
- **Approach 1**: hit O(1), getHits O(N) worst case, amortized O(1)
- **Approach 2**: hit O(1), getHits O(300) = O(1)

## Space Complexity:
- **Approach 1**: O(N) where N = hits in window
- **Approach 2**: O(300) = O(1)`,
    testCases: [
      {
        input: [
          ['HitCounter'],
          ['hit', 1],
          ['hit', 2],
          ['hit', 3],
          ['getHits', 4],
          ['hit', 300],
          ['getHits', 300],
          ['getHits', 301],
        ],
        expected: [null, null, null, null, 3, null, 4, 3],
      },
    ],
    solution: `from collections import deque

# Approach 1: Deque with timestamps (Exact Count)
class HitCounter:
    def __init__(self):
        self.hits = deque()
        self.window = 300
    
    def hit(self, timestamp: int) -> None:
        """Record a hit - O(1)"""
        self.hits.append(timestamp)
    
    def getHits(self, timestamp: int) -> int:
        """Get hits in last 300 seconds - Amortized O(1)"""
        # Remove old hits (outside window)
        while self.hits and self.hits[0] <= timestamp - self.window:
            self.hits.popleft()
        return len(self.hits)


# Approach 2: Bucketing (O(1) Space)
class HitCounter:
    def __init__(self):
        self.buckets = [0] * 300
        self.timestamps = [0] * 300
        self.window = 300
    
    def hit(self, timestamp: int) -> None:
        """Record a hit - O(1)"""
        idx = timestamp % 300
        if self.timestamps[idx] != timestamp:
            # Bucket from old window, reset it
            self.timestamps[idx] = timestamp
            self.buckets[idx] = 1
        else:
            self.buckets[idx] += 1
    
    def getHits(self, timestamp: int) -> int:
        """Get hits in last 300 seconds - O(300) = O(1)"""
        total = 0
        for i in range(300):
            if timestamp - self.timestamps[i] < self.window:
                total += self.buckets[i]
        return total


# Approach 3: Hybrid (timestamp, count) pairs
class HitCounter:
    def __init__(self):
        self.hits = deque()  # Store (timestamp, count)
        self.window = 300
    
    def hit(self, timestamp: int) -> None:
        """Record a hit - O(1)"""
        if self.hits and self.hits[-1][0] == timestamp:
            # Same timestamp, increment count
            self.hits[-1] = (timestamp, self.hits[-1][1] + 1)
        else:
            self.hits.append((timestamp, 1))
    
    def getHits(self, timestamp: int) -> int:
        """Get hits in last 300 seconds"""
        # Remove old entries
        while self.hits and self.hits[0][0] <= timestamp - self.window:
            self.hits.popleft()
        
        return sum(count for ts, count in self.hits)

# Example usage:
# counter = HitCounter()
# counter.hit(1)       # hits=[(1,1)]
# counter.hit(2)       # hits=[(1,1), (2,1)]
# counter.hit(2)       # hits=[(1,1), (2,2)] - count incremented
# counter.getHits(5)   # returns 3
# counter.hit(302)     # hits=[(1,1), (2,2), (302,1)]
# counter.getHits(302) # removes (1,1), returns 2`,
    timeComplexity:
      'Approach 1: hit O(1), getHits amortized O(1). Approach 2: both O(1).',
    spaceComplexity:
      'Approach 1: O(N) where N=hits in window. Approach 2: O(1).',
    patterns: ['Design', 'Deque', 'Sliding Window', 'Hash Table'],
    companies: ['Google', 'Dropbox', 'Uber'],
  },
  {
    id: 'browser-history',
    title: 'Design Browser History',
    difficulty: 'Medium',
    topic: 'Design Problems',
    description: `You have a **browser** of one tab where you start on the \`homepage\` and you can visit another \`url\`, get back in the history number of \`steps\` or move forward in the history number of \`steps\`.

Implement the \`BrowserHistory\` class:

- \`BrowserHistory(string homepage)\` Initializes the object with the \`homepage\` of the browser.
- \`void visit(string url)\` Visits \`url\` from the current page. It clears up all the forward history.
- \`string back(int steps)\` Move \`steps\` back in history. If you can only return \`x\` steps in the history and \`steps > x\`, you will return only \`x\` steps. Return the current \`url\` after moving back in history **at most** \`steps\`.
- \`string forward(int steps)\` Move \`steps\` forward in history. If you can only forward \`x\` steps in the history and \`steps > x\`, you will forward only \`x\` steps. Return the current \`url\` after forwarding in history **at most** \`steps\`.`,
    hints: [
      'Think of browser history as linear timeline with current position',
      'Two stacks: back_stack for history, forward_stack for forward pages',
      'Alternative: array with current_index pointer',
      'visit() must clear forward history (creates new branch)',
    ],
    approach: `## Intuition

Browser maintains **linear history** with current position. We need:
- Go back: Move position backward
- Go forward: Move position forward  
- Visit: Add to history, clear forward

---

## Approach 1: Two Stacks

Use two stacks to track position:
- **back_stack**: Pages behind current
- **forward_stack**: Pages ahead of current
- **current**: Current page

### Example:

\`\`\`
BrowserHistory("home"):
  back=[], current="home", forward=[]

visit("page1"):
  back=["home"], current="page1", forward=[]

visit("page2"):
  back=["home", "page1"], current="page2", forward=[]

back(1):
  Move current to forward, pop from back
  back=["home"], current="page1", forward=["page2"]

back(1):
  back=[], current="home", forward=["page2", "page1"]

forward(1):
  back=["home"], current="page1", forward=["page2"]

visit("page3"):
  CLEARS forward! (new branch)
  back=["home", "page1"], current="page3", forward=[]
\`\`\`

---

## Approach 2: Array with Pointer

Store all history in array, track current index:

\`\`\`python
history = ["home"]
current_idx = 0

visit("page1"):
  # Remove everything after current
  history = history[:current_idx + 1]
  history.append("page1")
  current_idx += 1
  # history = ["home", "page1"], idx = 1

back(1):
  current_idx = max(0, current_idx - 1)
  return history[current_idx]
\`\`\`

**Simpler!** No need to manage two stacks.

---

## Why Clear Forward on Visit?

When you visit new page from middle of history, forward pages become alternate timeline:

\`\`\`
Timeline: A -> B -> C (at B, visit D)
Result:   A -> B -> D (C is lost, new branch)
\`\`\`

Matches real browser behavior.

---

## Time Complexity: O(1) per step for back/forward
## Space Complexity: O(N) where N = total pages visited`,
    testCases: [
      {
        input: [
          ['BrowserHistory', 'home'],
          ['visit', 'page1'],
          ['visit', 'page2'],
          ['back', 1],
          ['back', 1],
          ['forward', 1],
          ['visit', 'page3'],
          ['forward', 2],
          ['back', 2],
          ['back', 7],
        ],
        expected: [null, null, null, 'page1', 'home', 'page1', null, 'page1', 'home', 'home'],
      },
    ],
    solution: `# Approach 1: Two Stacks
class BrowserHistory:
    def __init__(self, homepage: str):
        self.back_stack = []
        self.forward_stack = []
        self.current = homepage
    
    def visit(self, url: str) -> None:
        """Visit new URL, clear forward history - O(1)"""
        self.back_stack.append(self.current)
        self.current = url
        self.forward_stack = []  # Clear forward!
    
    def back(self, steps: int) -> str:
        """Go back steps - O(steps)"""
        while steps > 0 and self.back_stack:
            self.forward_stack.append(self.current)
            self.current = self.back_stack.pop()
            steps -= 1
        return self.current
    
    def forward(self, steps: int) -> str:
        """Go forward steps - O(steps)"""
        while steps > 0 and self.forward_stack:
            self.back_stack.append(self.current)
            self.current = self.forward_stack.pop()
            steps -= 1
        return self.current


# Approach 2: Array with Pointer (Recommended - Simpler!)
class BrowserHistory:
    def __init__(self, homepage: str):
        self.history = [homepage]
        self.current_idx = 0
    
    def visit(self, url: str) -> None:
        """Visit new URL - O(1)"""
        # Remove everything after current (clear forward)
        self.history = self.history[:self.current_idx + 1]
        self.history.append(url)
        self.current_idx += 1
    
    def back(self, steps: int) -> str:
        """Go back steps - O(1)"""
        self.current_idx = max(0, self.current_idx - steps)
        return self.history[self.current_idx]
    
    def forward(self, steps: int) -> str:
        """Go forward steps - O(1)"""
        self.current_idx = min(len(self.history) - 1, 
                               self.current_idx + steps)
        return self.history[self.current_idx]

# Example usage:
# browser = BrowserHistory("home")
# browser.visit("page1")  # history=["home", "page1"], idx=1
# browser.visit("page2")  # history=["home", "page1", "page2"], idx=2
# browser.back(1)         # idx=1, returns "page1"
# browser.forward(1)      # idx=2, returns "page2"
# browser.visit("page3")  # history=["home", "page1", "page2", "page3"], idx=3`,
    timeComplexity: 'visit: O(1), back/forward: O(1) per step (or O(steps))',
    spaceComplexity: 'O(N) where N is number of unique pages visited',
    patterns: ['Stack', 'Design', 'Array'],
    companies: ['Google', 'Amazon', 'Facebook'],
  },
  {
    id: 'lfu-cache',
    title: 'LFU Cache',
    difficulty: 'Hard',
    topic: 'Design Problems',
    description: `Design and implement a data structure for a **Least Frequently Used (LFU)** cache.

Implement the \`LFUCache\` class:

- \`LFUCache(int capacity)\` Initializes the object with the \`capacity\` of the data structure.
- \`int get(int key)\` Gets the value of the \`key\` if the \`key\` exists in the cache. Otherwise, returns \`-1\`.
- \`void put(int key, int value)\` Update the value of the \`key\` if present, or inserts the \`key\` if not already present. When the cache reaches its \`capacity\`, it should invalidate and remove the **least frequently used** key before inserting a new item. For this problem, when there is a **tie** (i.e., two or more keys with the same frequency), the **least recently used** \`key\` would be invalidated.

To determine the least frequently used key, a **use counter** is maintained for each key in the cache. The key with the smallest **use counter** is the least frequently used key.

When a key is first inserted into the cache, its **use counter** is set to \`1\` (due to the \`put\` operation). The use counter for a key in the cache is incremented either a \`get\` or \`put\` operation is called on it.

The functions \`get\` and \`put\` must each run in **O(1)** average time complexity.`,
    hints: [
      'Track frequency of each key - HashMap: key -> frequency',
      'Group keys by frequency - HashMap: frequency -> list of keys',
      'Within same frequency, LRU order - use doubly linked list',
      'Track minimum frequency for O(1) eviction',
    ],
    approach: `## Intuition

LFU evicts **least frequently used** key. Ties broken by LRU.

Need to track:
1. **Frequency** of each key
2. **Keys at each frequency** (for eviction)
3. **Within frequency, LRU order** (for tie-breaking)

---

## Approach: HashMap + Frequency Buckets

**Data Structures:**

1. \`key_to_val\`: key -> value
2. \`key_to_freq\`: key -> frequency
3. \`freq_to_keys\`: frequency -> OrderedDict of keys (LRU order)
4. \`min_freq\`: Track minimum frequency for O(1) eviction

### Example:

\`\`\`
put(1, 1): freq=1
  key_to_freq = {1: 1}
  freq_to_keys = {1: OrderedDict([1])}
  min_freq = 1

put(2, 2): freq=1
  key_to_freq = {1: 1, 2: 1}
  freq_to_keys = {1: OrderedDict([1, 2])}
  min_freq = 1

get(1): returns 1, frequency → 2
  key_to_freq = {1: 2, 2: 1}
  freq_to_keys = {1: OrderedDict([2]), 2: OrderedDict([1])}
  min_freq = 1

put(3, 3): Cache full, evict key 2 (freq=1, LRU)
  key_to_freq = {1: 2, 3: 1}
  freq_to_keys = {1: OrderedDict([3]), 2: OrderedDict([1])}
\`\`\`

### Operations:

**get(key):**
1. If not exists → return -1
2. Increment frequency
3. Move key from freq to freq+1
4. Update min_freq if needed
5. Return value

**put(key, val):**
1. If capacity is 0 → do nothing
2. If key exists:
   - Update value
   - Increment frequency (same as get)
3. If key is new:
   - If at capacity: evict LFU key (from freq_to_keys[min_freq], first item)
   - Add key with freq=1
   - Set min_freq=1

---

## Time Complexity: O(1) for both get and put
## Space Complexity: O(capacity)`,
    testCases: [
      {
        input: [
          ['LFUCache', 2],
          ['put', 1, 1],
          ['put', 2, 2],
          ['get', 1],
          ['put', 3, 3],
          ['get', 2],
          ['get', 3],
          ['put', 4, 4],
          ['get', 1],
          ['get', 3],
          ['get', 4],
        ],
        expected: [null, null, null, 1, null, -1, 3, null, -1, 3, 4],
      },
    ],
    solution: `from collections import defaultdict, OrderedDict

class LFUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.min_freq = 0
        self.key_to_val = {}  # key -> value
        self.key_to_freq = {}  # key -> frequency
        self.freq_to_keys = defaultdict(OrderedDict)  # freq -> OrderedDict of keys
    
    def _update_freq(self, key: int) -> None:
        """Increment frequency of key"""
        freq = self.key_to_freq[key]
        
        # Remove from current frequency bucket
        del self.freq_to_keys[freq][key]
        
        # Update min_freq if this frequency bucket is now empty
        if not self.freq_to_keys[freq] and freq == self.min_freq:
            self.min_freq += 1
        
        # Increment frequency
        self.key_to_freq[key] = freq + 1
        
        # Add to new frequency bucket
        self.freq_to_keys[freq + 1][key] = None  # OrderedDict maintains insertion order
    
    def get(self, key: int) -> int:
        """Get value and increment frequency"""
        if key not in self.key_to_val:
            return -1
        
        self._update_freq(key)
        return self.key_to_val[key]
    
    def put(self, key: int, value: int) -> None:
        """Put key-value pair"""
        if self.capacity == 0:
            return
        
        if key in self.key_to_val:
            # Key exists - update value and frequency
            self.key_to_val[key] = value
            self._update_freq(key)
        else:
            # New key
            if len(self.key_to_val) >= self.capacity:
                # Evict LFU key (first key in min_freq bucket)
                lfu_key, _ = self.freq_to_keys[self.min_freq].popitem(last=False)
                del self.key_to_val[lfu_key]
                del self.key_to_freq[lfu_key]
            
            # Add new key with frequency 1
            self.key_to_val[key] = value
            self.key_to_freq[key] = 1
            self.freq_to_keys[1][key] = None
            self.min_freq = 1

# Example usage:
# cache = LFUCache(2)
# cache.put(1, 1)  # freq: {1:1}
# cache.put(2, 2)  # freq: {1:1, 2:1}
# cache.get(1)     # freq: {1:2, 2:1}, returns 1
# cache.put(3, 3)  # Evicts key 2 (freq=1), freq: {1:2, 3:1}`,
    timeComplexity: 'O(1) for both get() and put() operations',
    spaceComplexity: 'O(capacity) for all data structures',
    patterns: ['HashMap', 'Design', 'OrderedDict', 'LFU'],
    companies: ['Amazon', 'Google', 'Facebook', 'Apple'],
  },
  {
    id: 'design-twitter',
    title: 'Design Twitter',
    difficulty: 'Medium',
    topic: 'Design Problems',
    description: `Design a simplified version of Twitter where users can post tweets, follow/unfollow another user, and is able to see the \`10\` most recent tweets in the user's news feed.

Implement the \`Twitter\` class:

- \`Twitter()\` Initializes your twitter object.
- \`void postTweet(int userId, int tweetId)\` Composes a new tweet with ID \`tweetId\` by the user \`userId\`. Each call to this function will be made with a unique \`tweetId\`.
- \`List<Integer> getNewsFeed(int userId)\` Retrieves the \`10\` most recent tweet IDs in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user themself. Tweets must be **ordered from most recent to least recent**.
- \`void follow(int followerId, int followeeId)\` The user with ID \`followerId\` started following the user with ID \`followeeId\`.
- \`void unfollow(int followerId, int followeeId)\` The user with ID \`followerId\` started unfollowing the user with ID \`followeeId\`.`,
    hints: [
      'Store tweets per user in chronological order',
      'Use timestamp to order tweets across users',
      'Max heap to merge K sorted lists (user timelines)',
      'Use set for following (O(1) add/remove)',
    ],
    approach: `## Intuition

Twitter features:
1. **Post tweet**: Add to user's timeline
2. **Get news feed**: Merge timelines from user + followees
3. **Follow/Unfollow**: Manage following relationship

**Challenge**: Efficiently get top 10 recent tweets from multiple users.

---

## Approach: HashMap + Heap for Merging

**Data Structures:**
1. \`tweets\`: userId -> list of (timestamp, tweetId)
2. \`following\`: userId -> set of followeeIds
3. \`timestamp\`: Global counter for ordering

### getNewsFeed - Merge K Sorted Lists:

Each user's tweets are in timestamp order (sorted). Need to merge multiple sorted lists efficiently.

**Naive**: Collect all tweets, sort → O(N log N)  
**Better**: Max heap → O(K log K) where K = total recent tweets to consider

\`\`\`python
# Get last 10 tweets from user + each followee
# Use max heap to keep top 10 by timestamp
\`\`\`

**Optimization**: Only look at last 10 tweets per user (not all history).

---

## Example:

\`\`\`
postTweet(1, 5):  # User 1 tweets 5
  tweets = {1: [(0, 5)]}

follow(1, 2):     # User 1 follows user 2
  following = {1: {2}}

postTweet(2, 6):  # User 2 tweets 6
  tweets = {1: [(0, 5)], 2: [(1, 6)]}

getNewsFeed(1):   # Get feed for user 1
  # Collect from user 1 and user 2 (followee)
  # Tweets: [(0, 5), (1, 6)]
  # Sort by timestamp desc: [6, 5]
  returns [6, 5]
\`\`\`

---

## Time Complexity:
- postTweet: O(1)
- getNewsFeed: O(F * 10 * log K) where F = followees, K = feed size
- follow/unfollow: O(1)

## Space Complexity: O(U * T) where U = users, T = tweets per user`,
    testCases: [
      {
        input: [
          ['Twitter'],
          ['postTweet', 1, 5],
          ['getNewsFeed', 1],
          ['follow', 1, 2],
          ['postTweet', 2, 6],
          ['getNewsFeed', 1],
          ['unfollow', 1, 2],
          ['getNewsFeed', 1],
        ],
        expected: [null, null, [5], null, null, [6, 5], null, [5]],
      },
    ],
    solution: `from collections import defaultdict
import heapq

class Twitter:
    def __init__(self):
        self.tweets = defaultdict(list)  # userId -> [(timestamp, tweetId)]
        self.following = defaultdict(set)  # userId -> set of followeeIds
        self.timestamp = 0  # Global timestamp for ordering
    
    def postTweet(self, userId: int, tweetId: int) -> None:
        """Post a tweet - O(1)"""
        self.tweets[userId].append((self.timestamp, tweetId))
        self.timestamp += 1
    
    def getNewsFeed(self, userId: int) -> list[int]:
        """Get 10 most recent tweets from user + followees"""
        max_heap = []
        
        # Add own tweets (last 10)
        for timestamp, tweetId in self.tweets[userId][-10:]:
            heapq.heappush(max_heap, (-timestamp, tweetId))
        
        # Add followees' tweets (last 10 each)
        for followeeId in self.following[userId]:
            for timestamp, tweetId in self.tweets[followeeId][-10:]:
                heapq.heappush(max_heap, (-timestamp, tweetId))
        
        # Extract top 10
        result = []
        while max_heap and len(result) < 10:
            _, tweetId = heapq.heappop(max_heap)
            result.append(tweetId)
        
        return result
    
    def follow(self, followerId: int, followeeId: int) -> None:
        """Follow a user - O(1)"""
        if followerId != followeeId:  # Can't follow yourself
            self.following[followerId].add(followeeId)
    
    def unfollow(self, followerId: int, followeeId: int) -> None:
        """Unfollow a user - O(1)"""
        self.following[followerId].discard(followeeId)

# Example usage:
# twitter = Twitter()
# twitter.postTweet(1, 5)
# twitter.getNewsFeed(1)  # returns [5]
# twitter.follow(1, 2)
# twitter.postTweet(2, 6)
# twitter.getNewsFeed(1)  # returns [6, 5]
# twitter.unfollow(1, 2)
# twitter.getNewsFeed(1)  # returns [5]`,
    timeComplexity:
      'postTweet O(1), getNewsFeed O(F * 10 * log K), follow/unfollow O(1)',
    spaceComplexity: 'O(U * T) where U=users, T=average tweets per user',
    patterns: ['HashMap', 'Heap', 'Design', 'Merge K Sorted Lists'],
    companies: ['Twitter', 'Amazon', 'Microsoft', 'Facebook'],
  },
  {
    id: 'search-autocomplete',
    title: 'Design Search Autocomplete System',
    difficulty: 'Hard',
    topic: 'Design Problems',
    description: `Design a search autocomplete system for a search engine. Users may input a sentence (at least one word and end with a special character \`'#'\`).

You are given a string array \`sentences\` and an integer array \`times\` both of length \`n\` where \`sentences[i]\` is a previously typed sentence and \`times[i]\` is the corresponding number of times the sentence was typed. For each input character except \`'#'\`, return the top \`3\` historical hot sentences that have the same prefix as the part of the sentence already typed.

Here are the specific rules:

- The hot degree for a sentence is defined as the number of times a user typed the exactly same sentence before.
- The returned top \`3\` hot sentences should be sorted by hot degree (The first is the hottest one). If several sentences have the same hot degree, use ASCII-code order (smaller one appears first).
- If less than 3 hot sentences exist, return as many as you can.
- When the input is a special character, it means the sentence ends, and in this case, you need to return an empty list.

Implement the \`AutocompleteSystem\` class:

- \`AutocompleteSystem(String[] sentences, int[] times)\` Initializes the object with the \`sentences\` and \`times\` arrays.
- \`List<String> input(char c)\` This indicates that the user typed the character \`c\`.
  - Returns an empty array \`[]\` if \`c == '#'\`.
  - Otherwise, returns the top \`3\` historical hot sentences that have the same prefix as the part of the sentence already typed. If less than 3 exist, return as many as you can.`,
    hints: [
      'Trie for prefix search - O(p) where p = prefix length',
      'Store frequency at each end-of-word node',
      'DFS to collect all completions from current prefix',
      'Sort by frequency (desc), then lexicographically',
      'Optimization: Precompute top K at each Trie node',
    ],
    approach: `## Intuition

Autocomplete needs **prefix search** - perfect for **Trie**!

Each input character extends the prefix. We need to:
1. Navigate Trie to prefix
2. Find all sentences with that prefix
3. Return top 3 by frequency

---

## Approach: Trie + Frequency Tracking

**Trie Structure:**
\`\`\`
class TrieNode:
    children: dict[char -> TrieNode]
    is_end: bool
    frequency: int  # How many times this sentence was completed
    sentence: str   # Full sentence (stored at end node)
\`\`\`

### Operations:

**input(c):**
1. If c == '#':
   - End of sentence, save it with freq+1
   - Reset current input
   - Return []
2. Else:
   - Append c to current input
   - Navigate Trie to prefix
   - DFS to collect all completions
   - Sort by (frequency desc, lexicographical)
   - Return top 3

**Example:**
\`\`\`
Sentences: ["i love you", "island", "ironman"], times: [5, 3, 2]

input('i'):
  Navigate to 'i', collect all:
  - "i love you" (freq 5)
  - "island" (freq 3)
  - "ironman" (freq 2)
  Return ["i love you", "island", "ironman"]

input(' '):
  Navigate to 'i ', collect:
  - "i love you" (freq 5)
  Return ["i love you"]

input('a'):
  Navigate to 'i a', no matches
  Return []
\`\`\`

---

## Optimization: Precompute Top K

Instead of DFS + sort on every input, precompute top K at each node during insertion.

**Trade-off**: Slower insert, faster query.

---

## Time Complexity:
- Without optimization: O(p + m log m) where p=prefix, m=matching sentences
- With optimization: O(p) - just navigate and return cached

## Space Complexity: O(N * L) where N=sentences, L=avg length`,
    testCases: [
      {
        input: [
          [
            'AutocompleteSystem',
            ['i love you', 'island', 'iroman', 'i love leetcode'],
            [5, 3, 2, 2],
          ],
          ['input', 'i'],
          ['input', ' '],
          ['input', 'a'],
          ['input', '#'],
        ],
        expected: [
          null,
          ['i love you', 'island', 'i love leetcode'],
          ['i love you', 'i love leetcode'],
          [],
          [],
        ],
      },
    ],
    solution: `class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.frequency = 0
        self.sentence = ""

class AutocompleteSystem:
    def __init__(self, sentences: list[str], times: list[int]):
        self.root = TrieNode()
        self.current_input = ""
        
        # Build trie with initial data
        for sentence, freq in zip(sentences, times):
            self._add_to_trie(sentence, freq)
    
    def _add_to_trie(self, sentence: str, frequency: int) -> None:
        """Add sentence to trie with given frequency"""
        node = self.root
        for char in sentence:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        node.is_end = True
        node.sentence = sentence
        node.frequency += frequency
    
    def _search_with_prefix(self, prefix: str) -> list[str]:
        """Find all sentences with given prefix"""
        node = self.root
        
        # Navigate to prefix
        for char in prefix:
            if char not in node.children:
                return []  # No matches
            node = node.children[char]
        
        # Collect all completions from this node
        results = []
        self._dfs_collect(node, results)
        
        # Sort by frequency (desc), then lexicographically (asc)
        results.sort(key=lambda x: (-x[1], x[0]))
        
        # Return top 3 sentences
        return [sentence for sentence, freq in results[:3]]
    
    def _dfs_collect(self, node: TrieNode, results: list) -> None:
        """DFS to collect all sentences from this node"""
        if node.is_end:
            results.append((node.sentence, node.frequency))
        
        for child in node.children.values():
            self._dfs_collect(child, results)
    
    def input(self, c: str) -> list[str]:
        """Process input character"""
        if c == '#':
            # End of sentence - save it
            self._add_to_trie(self.current_input, 1)
            self.current_input = ""
            return []
        else:
            # Extend current input
            self.current_input += c
            return self._search_with_prefix(self.current_input)

# Example usage:
# autocomplete = AutocompleteSystem(["i love you", "island", "ironman"], [5, 3, 2])
# autocomplete.input('i')  # ["i love you", "island", "ironman"]
# autocomplete.input(' ')  # ["i love you"]
# autocomplete.input('a')  # []
# autocomplete.input('#')  # []`,
    timeComplexity:
      'O(p + m log m) per input, where p=prefix length, m=matching sentences',
    spaceComplexity:
      'O(N * L) where N=total sentences, L=average sentence length',
    patterns: ['Trie', 'DFS', 'Design', 'Sorting'],
    companies: ['Google', 'Amazon', 'Microsoft', 'Facebook'],
  },
  {
    id: 'rate-limiter',
    title: 'Design Rate Limiter',
    difficulty: 'Medium',
    topic: 'Design Problems',
    description: `Design a rate limiter that limits the number of requests a user can make to an API within a certain time window.

Implement the \`RateLimiter\` class:

- \`RateLimiter(int maxRequests, int timeWindow)\` Initializes the rate limiter with maximum \`maxRequests\` allowed in \`timeWindow\` seconds.
- \`boolean shouldAllow(int userId, int timestamp)\` Returns \`true\` if the user with \`userId\` is allowed to make a request at the given \`timestamp\`, otherwise returns \`false\`.

The rate limiter should handle multiple users, and timestamps are given in chronological order (monotonically increasing).`,
    hints: [
      'Track request timestamps per user',
      'Remove timestamps outside the time window',
      'Deque allows O(1) removal from front',
      'Alternative: Fixed window with counter (simpler but less accurate)',
      'Token bucket algorithm for smooth rate limiting',
    ],
    approach: `## Intuition

Rate limiting ensures users don't exceed a request quota within a time window.

**Challenge**: Efficiently track and count requests in sliding time window.

---

## Approach 1: Sliding Window Log (Exact)

For each user, store request timestamps in deque:

\`\`\`python
def shouldAllow(userId, timestamp):
    # Remove old requests (outside window)
    while requests[userId] and requests[userId][0] <= timestamp - timeWindow:
        requests[userId].popleft()
    
    # Check if under limit
    if len(requests[userId]) < maxRequests:
        requests[userId].append(timestamp)
        return True
    return False
\`\`\`

**Pros**: Exact count, no boundary issues  
**Cons**: O(N) memory per user

---

## Approach 2: Fixed Window Counter (Simple)

Track count per time window:

\`\`\`python
def shouldAllow(userId, timestamp):
    window = timestamp // timeWindow
    
    if user_windows[userId] != window:
        # New window, reset
        user_windows[userId] = window
        user_counts[userId] = 0
    
    if user_counts[userId] < maxRequests:
        user_counts[userId] += 1
        return True
    return False
\`\`\`

**Pros**: O(1) memory per user  
**Cons**: Boundary spike problem (2x rate at boundaries)

---

## Approach 3: Token Bucket (Industry Standard)

Each user has bucket that refills at constant rate:

\`\`\`python
def shouldAllow(userId, timestamp):
    # Refill tokens based on time elapsed
    elapsed = timestamp - last_refill[userId]
    tokens[userId] = min(capacity, tokens[userId] + elapsed * refill_rate)
    last_refill[userId] = timestamp
    
    if tokens[userId] >= 1:
        tokens[userId] -= 1
        return True
    return False
\`\`\`

**Pros**: Smooth rate limiting, allows bursts  
**Cons**: Slightly more complex

---

## Time Complexity:
- Approach 1: O(1) amortized per request
- Approach 2: O(1) per request
- Approach 3: O(1) per request

## Space Complexity: O(U) where U = number of users`,
    testCases: [
      {
        input: [
          ['RateLimiter', 3, 60],
          ['shouldAllow', 1, 0],
          ['shouldAllow', 1, 10],
          ['shouldAllow', 1, 20],
          ['shouldAllow', 1, 30],
          ['shouldAllow', 1, 70],
        ],
        expected: [null, true, true, true, false, true],
      },
    ],
    solution: `from collections import defaultdict, deque

# Approach 1: Sliding Window Log (Exact Count)
class RateLimiter:
    def __init__(self, maxRequests: int, timeWindow: int):
        self.maxRequests = maxRequests
        self.timeWindow = timeWindow
        self.user_requests = defaultdict(deque)  # userId -> deque of timestamps
    
    def shouldAllow(self, userId: int, timestamp: int) -> bool:
        """Check if request should be allowed"""
        requests = self.user_requests[userId]
        
        # Remove requests outside time window
        while requests and requests[0] <= timestamp - self.timeWindow:
            requests.popleft()
        
        # Check if under limit
        if len(requests) < self.maxRequests:
            requests.append(timestamp)
            return True
        return False


# Approach 2: Fixed Window Counter (Simple)
class RateLimiter:
    def __init__(self, maxRequests: int, timeWindow: int):
        self.maxRequests = maxRequests
        self.timeWindow = timeWindow
        self.user_windows = {}  # userId -> current window number
        self.user_counts = defaultdict(int)  # userId -> count in window
    
    def shouldAllow(self, userId: int, timestamp: int) -> bool:
        """Check if request should be allowed"""
        window = timestamp // self.timeWindow
        
        if self.user_windows.get(userId) != window:
            # New window - reset count
            self.user_windows[userId] = window
            self.user_counts[userId] = 0
        
        if self.user_counts[userId] < self.maxRequests:
            self.user_counts[userId] += 1
            return True
        return False


# Approach 3: Token Bucket (Industry Standard)
class RateLimiter:
    def __init__(self, maxRequests: int, timeWindow: int):
        self.capacity = maxRequests
        self.refill_rate = maxRequests / timeWindow  # tokens per second
        self.user_tokens = {}  # userId -> tokens
        self.user_last_refill = {}  # userId -> last refill timestamp
    
    def shouldAllow(self, userId: int, timestamp: int) -> bool:
        """Check if request should be allowed"""
        # Initialize user if first request
        if userId not in self.user_tokens:
            self.user_tokens[userId] = self.capacity
            self.user_last_refill[userId] = timestamp
        
        # Refill tokens based on time elapsed
        elapsed = timestamp - self.user_last_refill[userId]
        self.user_tokens[userId] = min(
            self.capacity,
            self.user_tokens[userId] + elapsed * self.refill_rate
        )
        self.user_last_refill[userId] = timestamp
        
        # Check if enough tokens
        if self.user_tokens[userId] >= 1:
            self.user_tokens[userId] -= 1
            return True
        return False

# Example usage:
# limiter = RateLimiter(3, 60)  # 3 requests per 60 seconds
# limiter.shouldAllow(1, 0)    # True
# limiter.shouldAllow(1, 10)   # True
# limiter.shouldAllow(1, 20)   # True
# limiter.shouldAllow(1, 30)   # False (limit reached)
# limiter.shouldAllow(1, 70)   # True (new window)`,
    timeComplexity:
      'O(1) amortized for sliding window, O(1) for fixed window and token bucket',
    spaceComplexity: 'O(U) where U is number of users',
    patterns: ['Deque', 'Sliding Window', 'Design', 'Hash Table'],
    companies: ['Amazon', 'Google', 'Microsoft', 'Stripe', 'Cloudflare'],
  },
  {
    id: 'parking-lot',
    title: 'Design Parking Lot',
    difficulty: 'Hard',
    topic: 'Design Problems',
    description: `Design a parking lot system that can:

1. Park vehicles (cars, trucks, motorcycles) in available spots
2. Remove vehicles from parking spots
3. Track available spots by type (compact, large, handicapped, motorcycle)
4. Find nearest available spot efficiently

The parking lot has multiple levels, and each spot has a type. Vehicles can only park in compatible spot types:
- Motorcycles can park in motorcycle spots
- Cars can park in compact, large, or handicapped spots
- Trucks can only park in large spots

Implement the \`ParkingLot\` class with proper object-oriented design principles.`,
    hints: [
      'Use inheritance for vehicle types (Vehicle → Car/Truck/Motorcycle)',
      'Each spot type has different constraints',
      'Min heap to efficiently find nearest available spot (by level, row)',
      'HashMap to track which spot each vehicle is in',
      'Encapsulate spot state management within ParkingSpot class',
    ],
    approach: `## Intuition

This is an **Object-Oriented Design** problem testing:
- Class hierarchy (inheritance)
- Encapsulation
- Composition
- Data structure selection

---

## Class Design

### 1. Vehicle Hierarchy

\`\`\`python
class Vehicle (ABC):
    - license_plate
    - type
    - can_fit_in(spot) → bool  # Abstract method

class Car(Vehicle):
    can_fit_in() → compact, large, handicapped spots

class Truck(Vehicle):
    can_fit_in() → only large spots

class Motorcycle(Vehicle):
    can_fit_in() → only motorcycle spots
\`\`\`

**Polymorphism**: \`vehicle.can_fit_in(spot)\` works for any vehicle type.

### 2. ParkingSpot

\`\`\`python
class ParkingSpot:
    - id, type, level, row
    - vehicle (currently parked)
    
    - is_available() → bool
    - park_vehicle(vehicle) → bool
    - remove_vehicle()
\`\`\`

**Encapsulation**: Spot manages its own state.

### 3. ParkingLot

\`\`\`python
class ParkingLot:
    - spots: HashMap(spot_id → ParkingSpot)
    - vehicle_to_spot: HashMap(license_plate → spot_id)
    - available_spots: HashMap(spot_type → MinHeap)
    
    - park_vehicle(vehicle) → spot_id
    - remove_vehicle(license_plate) → bool
\`\`\`

**Key Design**: Min heap per spot type for O(log N) nearest spot.

---

## Why Min Heap?

Need to find **nearest** available spot (by level, then row).

- Without heap: O(N) linear search through all spots
- With heap: O(log N) to get nearest spot

Heap maintains spots sorted by (level, row), so \`heap[0]\` is always closest.

---

## Time Complexity:
- park_vehicle: O(log N) where N = spots of compatible type
- remove_vehicle: O(log N) to re-add spot to heap

## Space Complexity: O(S + V) where S = spots, V = vehicles`,
    testCases: [
      {
        input: [
          ['ParkingLot'],
          ['add_spot', 'compact', 1, 1],
          ['add_spot', 'large', 1, 2],
          ['park_vehicle', 'car'],
          ['park_vehicle', 'truck'],
          ['remove_vehicle', 'car'],
        ],
        expected: 'Creates parking lot with 2 spots, parks car in compact, truck in large, removes car',
      },
    ],
    solution: `from enum import Enum
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
    """Abstract vehicle class"""
    def __init__(self, license_plate: str):
        self.license_plate = license_plate
        self.type = None
    
    @abstractmethod
    def can_fit_in(self, spot) -> bool:
        """Check if vehicle can fit in given spot"""
        pass

class Car(Vehicle):
    def __init__(self, license_plate: str):
        super().__init__(license_plate)
        self.type = VehicleType.COMPACT
    
    def can_fit_in(self, spot) -> bool:
        return spot.type in [SpotType.COMPACT, SpotType.LARGE, SpotType.HANDICAPPED]

class Truck(Vehicle):
    def __init__(self, license_plate: str):
        super().__init__(license_plate)
        self.type = VehicleType.LARGE
    
    def can_fit_in(self, spot) -> bool:
        return spot.type == SpotType.LARGE

class Motorcycle(Vehicle):
    def __init__(self, license_plate: str):
        super().__init__(license_plate)
        self.type = VehicleType.MOTORCYCLE
    
    def can_fit_in(self, spot) -> bool:
        return spot.type == SpotType.MOTORCYCLE

class ParkingSpot:
    """Parking spot that manages its own state"""
    def __init__(self, spot_id: int, spot_type: SpotType, level: int, row: int):
        self.id = spot_id
        self.type = spot_type
        self.level = level
        self.row = row
        self.vehicle = None
    
    def is_available(self) -> bool:
        return self.vehicle is None
    
    def park_vehicle(self, vehicle: Vehicle) -> bool:
        """Park vehicle if compatible and spot available"""
        if not self.is_available():
            return False
        if not vehicle.can_fit_in(self):
            return False
        self.vehicle = vehicle
        return True
    
    def remove_vehicle(self) -> None:
        self.vehicle = None

class ParkingLot:
    """Main parking lot system"""
    def __init__(self):
        self.spots = {}  # spot_id -> ParkingSpot
        self.vehicle_to_spot = {}  # license_plate -> spot_id
        # Min heaps for each spot type (sorted by level, row)
        self.available_spots = {
            SpotType.COMPACT: [],
            SpotType.LARGE: [],
            SpotType.HANDICAPPED: [],
            SpotType.MOTORCYCLE: []
        }
    
    def add_spot(self, spot: ParkingSpot) -> None:
        """Add spot to parking lot"""
        self.spots[spot.id] = spot
        heapq.heappush(
            self.available_spots[spot.type],
            (spot.level, spot.row, spot.id)
        )
    
    def park_vehicle(self, vehicle: Vehicle):
        """Park vehicle, returns spot_id or None"""
        spot_id = self._find_available_spot(vehicle)
        if not spot_id:
            return None  # No compatible spots available
        
        spot = self.spots[spot_id]
        if spot.park_vehicle(vehicle):
            self.vehicle_to_spot[vehicle.license_plate] = spot_id
            return spot_id
        return None
    
    def _find_available_spot(self, vehicle: Vehicle):
        """Find nearest available compatible spot"""
        # Check all compatible spot types
        compatible_types = []
        if isinstance(vehicle, Car):
            compatible_types = [SpotType.COMPACT, SpotType.LARGE, SpotType.HANDICAPPED]
        elif isinstance(vehicle, Truck):
            compatible_types = [SpotType.LARGE]
        elif isinstance(vehicle, Motorcycle):
            compatible_types = [SpotType.MOTORCYCLE]
        
        # Try each compatible type, find nearest
        for spot_type in compatible_types:
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
    
    def remove_vehicle(self, license_plate: str) -> bool:
        """Remove vehicle from parking lot"""
        if license_plate not in self.vehicle_to_spot:
            return False
        
        spot_id = self.vehicle_to_spot[license_plate]
        spot = self.spots[spot_id]
        spot.remove_vehicle()
        
        # Return spot to available pool
        heapq.heappush(
            self.available_spots[spot.type],
            (spot.level, spot.row, spot_id)
        )
        
        del self.vehicle_to_spot[license_plate]
        return True

# Example usage:
# lot = ParkingLot()
# lot.add_spot(ParkingSpot(1, SpotType.COMPACT, level=1, row=1))
# lot.add_spot(ParkingSpot(2, SpotType.LARGE, level=1, row=2))
# car = Car("ABC123")
# spot_id = lot.park_vehicle(car)  # Parks in spot 1
# lot.remove_vehicle("ABC123")     # Frees spot 1`,
    timeComplexity:
      'park_vehicle: O(log N), remove_vehicle: O(log N) where N = spots of type',
    spaceComplexity:
      'O(S + V) where S = total spots, V = currently parked vehicles',
    patterns: ['OOP Design', 'Inheritance', 'Heap', 'HashMap'],
    companies: ['Amazon', 'Microsoft', 'Google', 'Uber'],
  },
  {
    id: 'url-shortener',
    title: 'Design URL Shortener',
    difficulty: 'Medium',
    topic: 'Design Problems',
    description: `Design a URL shortener service like bit.ly that:

1. Takes a long URL and returns a shortened URL
2. Takes a shortened URL and redirects to the original long URL
3. Tracks the number of times each short URL has been accessed

Implement the \`URLShortener\` class:

- \`URLShortener()\` Initializes the URL shortener system.
- \`String shorten(String longUrl)\` Takes a long URL and returns a shortened URL. The short URL should be unique and as short as possible.
- \`String expand(String shortUrl)\` Takes a short URL and returns the original long URL, or empty string if not found.
- \`int getClickCount(String shortUrl)\` Returns the number of times the short URL has been accessed.

The shortened URL should be in the format: \`http://short.url/{code}\` where \`{code}\` is a unique identifier.`,
    hints: [
      'Use counter for unique IDs (no collisions)',
      'Convert counter to Base62 (0-9, a-z, A-Z) for shorter codes',
      'Two HashMaps: long→short and short→long',
      'Base62: 62^7 = 3.5 trillion possible 7-character codes',
      'Track click counts in separate HashMap',
    ],
    approach: `## Intuition

URL shortener needs:
1. **Unique short codes** for each long URL
2. **Bidirectional mapping**: long ↔ short
3. **Short as possible** codes

---

## Approach: Counter + Base62 Encoding

### Why Counter?

- **Guaranteed unique**: Increment for each URL
- **No collisions**: Unlike hashing
- **Predictable length**: Know code length for given # URLs

### Why Base62?

Convert counter to base 62 (0-9, a-z, A-Z):

\`\`\`
Counter  Base62
0        0
1        1
10       a
61       Z
62       10
1000     g8
1,000,000  4c92
\`\`\`

**Comparison:**
- Base10 (decimal): 1,000,000 = "1000000" (7 chars)
- Base62: 1,000,000 = "4c92" (4 chars)

**Capacity**: 62^7 = 3.5 trillion URLs with 7 characters

---

## Data Structures:

1. \`counter\`: Global counter for unique IDs
2. \`short_to_long\`: HashMap(short_code → long_url)
3. \`long_to_short\`: HashMap(long_url → short_url) *for deduplication*
4. \`clicks\`: HashMap(short_code → click_count)

---

## Example:

\`\`\`
shorten("https://example.com/very/long/url"):
  counter = 1
  short_code = encode_base62(1) = "1"
  return "http://short.url/1"

shorten("https://another.com/long/url"):
  counter = 2
  short_code = encode_base62(2) = "2"
  return "http://short.url/2"

expand("http://short.url/1"):
  returns "https://example.com/very/long/url"
  clicks["1"] += 1
\`\`\`

---

## Time Complexity: O(1) for shorten, expand, getClickCount
## Space Complexity: O(N) where N = number of URLs`,
    testCases: [
      {
        input: [
          ['URLShortener'],
          ['shorten', 'https://leetcode.com/problems/design'],
          ['expand', 'result'],
          ['getClickCount', 'code'],
        ],
        expected: [null, 'http://short.url/1', 'https://leetcode.com/problems/design', 1],
      },
    ],
    solution: `class URLShortener:
    def __init__(self):
        self.counter = 0  # Global counter for unique IDs
        self.short_to_long = {}  # short_code -> long_url
        self.long_to_short = {}  # long_url -> short_url (deduplication)
        self.clicks = {}  # short_code -> click count
        self.base62 = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.base_url = "http://short.url/"
    
    def encode_base62(self, num: int) -> str:
        """Convert number to base62 string"""
        if num == 0:
            return self.base62[0]
        
        result = []
        while num:
            result.append(self.base62[num % 62])
            num //= 62
        
        return ''.join(reversed(result))
    
    def shorten(self, long_url: str) -> str:
        """Shorten long URL - O(1)"""
        # Check if already shortened (deduplication)
        if long_url in self.long_to_short:
            return self.long_to_short[long_url]
        
        # Generate new short code
        self.counter += 1
        short_code = self.encode_base62(self.counter)
        short_url = self.base_url + short_code
        
        # Store mappings
        self.short_to_long[short_code] = long_url
        self.long_to_short[long_url] = short_url
        self.clicks[short_code] = 0
        
        return short_url
    
    def expand(self, short_url: str) -> str:
        """Expand short URL to long URL - O(1)"""
        # Extract short code from URL
        short_code = short_url.replace(self.base_url, "")
        
        if short_code in self.short_to_long:
            # Track click
            self.clicks[short_code] += 1
            return self.short_to_long[short_code]
        
        return ""  # Not found
    
    def getClickCount(self, short_url: str) -> int:
        """Get click count for short URL - O(1)"""
        short_code = short_url.replace(self.base_url, "")
        return self.clicks.get(short_code, 0)

# Example usage:
# shortener = URLShortener()
# 
# short1 = shortener.shorten("https://leetcode.com/problems/design")
# # Returns "http://short.url/1"
# 
# long1 = shortener.expand(short1)
# # Returns "https://leetcode.com/problems/design"
# # Click count incremented
# 
# clicks = shortener.getClickCount(short1)
# # Returns 1
# 
# # Duplicate URL returns same short code
# short2 = shortener.shorten("https://leetcode.com/problems/design")
# # Returns "http://short.url/1" (same as short1)`,
    timeComplexity: 'O(1) for shorten(), expand(), and getClickCount()',
    spaceComplexity: 'O(N) where N is number of unique URLs',
    patterns: ['HashMap', 'Design', 'Base62 Encoding', 'Counter'],
    companies: ['Google', 'Amazon', 'Microsoft', 'Facebook', 'Bitly'],
  },
];
