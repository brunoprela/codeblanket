/**
 * Queue Variations Section
 */

export const variationsSection = {
  id: 'variations',
  title: 'Queue Variations',
  content: `## Common Queue Variations

Beyond the basic queue, there are several important variations used in different scenarios.

---

## 1. Circular Queue (Ring Buffer)

A fixed-size queue where the rear connects back to the front, reusing space efficiently.

**Use Case:** Fixed-size buffers, streaming data

\`\`\`python
class CircularQueue:
    """Fixed-size circular queue"""
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.queue = [None] * capacity
        self.front = 0
        self.rear = 0
        self.size = 0
    
    def enqueue(self, item):
        """Add to rear"""
        if self.is_full():
            raise IndexError("Queue is full")
        
        self.queue[self.rear] = item
        self.rear = (self.rear + 1) % self.capacity  # Wrap around
        self.size += 1
    
    def dequeue(self):
        """Remove from front"""
        if self.is_empty():
            raise IndexError("Queue is empty")
        
        item = self.queue[self.front]
        self.front = (self.front + 1) % self.capacity  # Wrap around
        self.size -= 1
        return item
    
    def is_empty(self):
        return self.size == 0
    
    def is_full(self):
        return self.size == self.capacity
    
    def __repr__(self):
        """Show queue contents"""
        if self.is_empty():
            return "CircularQueue([])"
        
        items = []
        index = self.front
        for _ in range(self.size):
            items.append(self.queue[index])
            index = (index + 1) % self.capacity
        
        return f"CircularQueue({items})"

# Usage
cq = CircularQueue(3)
cq.enqueue(1)
cq.enqueue(2)
cq.enqueue(3)
print(cq)  # CircularQueue([1, 2, 3])

print(cq.dequeue())  # 1
cq.enqueue(4)  # Reuses space from 1!
print(cq)  # CircularQueue([2, 3, 4])
\`\`\`

**Key Feature:** Wraps around using modulo: \`(index + 1) % capacity\`

---

## 2. Priority Queue

Elements have priorities; highest priority element is dequeued first, not FIFO.

**Use Case:** Task scheduling, Dijkstra's algorithm, A* search

\`\`\`python
import heapq

class PriorityQueue:
    """Priority queue using heap"""
    
    def __init__(self):
        self.heap = []
        self.count = 0  # For tie-breaking
    
    def enqueue(self, item, priority):
        """Add with priority (lower number = higher priority)"""
        # Tuple: (priority, count, item)
        # count ensures FIFO for same priority
        heapq.heappush(self.heap, (priority, self.count, item))
        self.count += 1
    
    def dequeue(self):
        """Remove highest priority item"""
        if self.is_empty():
            raise IndexError("Queue is empty")
        
        priority, _, item = heapq.heappop(self.heap)
        return item
    
    def is_empty(self):
        return len(self.heap) == 0
    
    def size(self):
        return len(self.heap)

# Usage
pq = PriorityQueue()
pq.enqueue("Low priority task", priority=3)
pq.enqueue("High priority task", priority=1)
pq.enqueue("Medium priority task", priority=2)

print(pq.dequeue())  # "High priority task" (priority 1)
print(pq.dequeue())  # "Medium priority task" (priority 2)
print(pq.dequeue())  # "Low priority task" (priority 3)
\`\`\`

**Note:** Priority queue is technically a heap, not a queue! But it's called a queue because of the similar interface.

---

## 3. Deque (Double-Ended Queue)

Can add/remove from both ends efficiently.

**Use Case:** Sliding window problems, undo/redo, palindrome checking

\`\`\`python
from collections import deque

# Deque supports operations on both ends!
dq = deque([1, 2, 3])

# Add to right (rear)
dq.append(4)          # [1, 2, 3, 4]

# Add to left (front)
dq.appendleft(0)      # [0, 1, 2, 3, 4]

# Remove from right
dq.pop()              # Returns 4 → [0, 1, 2, 3]

# Remove from left
dq.popleft()          # Returns 0 → [1, 2, 3]

# All operations are O(1)!
\`\`\`

**Key Point:** Regular queue is one-way (rear in, front out). Deque is two-way (both ends).

---

## 4. Blocking Queue (Thread-Safe)

Thread-safe queue for concurrent programming.

**Use Case:** Producer-consumer pattern, multithreading

\`\`\`python
from queue import Queue  # Python's thread-safe queue
import threading

# Thread-safe queue
q = Queue(maxsize=10)  # Optional size limit

# Producer thread
def producer():
    for i in range(5):
        q.put(i)  # Blocks if queue is full
        print(f"Produced {i}")

# Consumer thread
def consumer():
    for _ in range(5):
        item = q.get()  # Blocks if queue is empty
        print(f"Consumed {item}")
        q.task_done()

# Start threads
t1 = threading.Thread(target=producer)
t2 = threading.Thread(target=consumer)
t1.start()
t2.start()
t1.join()
t2.join()
\`\`\`

**Features:**
- \`put()\` blocks when full
- \`get()\` blocks when empty
- Thread-safe operations
- Good for producer-consumer patterns

---

## Comparison Table

| Type | Add | Remove | Use Case |
|------|-----|--------|----------|
| **Regular Queue** | Rear | Front | BFS, basic FIFO |
| **Circular Queue** | Rear (wrap) | Front (wrap) | Fixed buffers |
| **Priority Queue** | By priority | Highest priority | Scheduling, Dijkstra |
| **Deque** | Both ends | Both ends | Sliding window |
| **Blocking Queue** | Thread-safe | Thread-safe | Multithreading |

---

## When to Use Each

**Regular Queue:** Default choice for BFS, task processing

**Circular Queue:** When you have fixed memory and want to reuse space

**Priority Queue:** When some items are more important than others

**Deque:** When you need efficiency at both ends (sliding window, palindrome)

**Blocking Queue:** When coordinating between threads

**Pro Tip:** For most LeetCode/interview problems, \`collections.deque\` is your best friend!`,
};
