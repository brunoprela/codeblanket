/**
 * Queue module content
 */

import { Module } from '@/lib/types';

export const queueModule: Module = {
    id: 'queue',
    title: 'Queue',
    description:
        'Master queue data structure and FIFO operations - essential for BFS, scheduling, and many algorithms.',
    icon: '📬',
    sections: [
        {
            id: 'introduction',
            title: 'Introduction to Queues',
            content: `A **Queue** is a linear data structure that follows the **FIFO (First-In-First-Out)** principle. Think of it like a line at a store - the first person in line is the first to be served.

**Why Queues Matter:**
- **Natural Model:** Represents real-world waiting scenarios
- **BFS Algorithm:** Essential for breadth-first search in trees and graphs
- **Task Scheduling:** Used in operating systems, printers, and job schedulers
- **Data Processing:** Buffering, streaming, message queues

**Real-World Applications:**
- **Operating Systems:** Process scheduling, task queues
- **Web Servers:** Request handling queues
- **Printers:** Print job queues
- **Customer Service:** Call center queues
- **Gaming:** Event queues, multiplayer matchmaking
- **Networking:** Packet queues in routers

**Key Insight:**
While Stack is Last-In-First-Out (LIFO), Queue is First-In-First-Out (FIFO). This makes queues perfect for scenarios where order of arrival matters and fairness is important.`,
            quiz: [
                {
                    id: 'q1',
                    question:
                        "Explain the FIFO principle and how it differs from a Stack's LIFO principle.",
                    sampleAnswer:
                        'FIFO (First-In-First-Out) means the first element added to the queue is the first one removed, like a line at a store. Stack uses LIFO (Last-In-First-Out) where the most recently added element is removed first, like a stack of plates. In a queue, elements are added at the rear and removed from the front. In a stack, elements are added and removed from the same end (top). Queue preserves order of arrival, while stack reverses it. Example: Queue [1,2,3] removes 1 first; Stack [1,2,3] removes 3 first.',
                    keyPoints: [
                        'Queue: First-In-First-Out (FIFO)',
                        'Stack: Last-In-First-Out (LIFO)',
                        'Queue: add rear, remove front',
                        'Stack: add/remove from top',
                        'Queue preserves order, stack reverses',
                    ],
                },
                {
                    id: 'q2',
                    question:
                        'Give three real-world examples where a queue is the natural data structure choice.',
                    sampleAnswer:
                        '(1) Printer queue - print jobs are processed in order they were submitted, ensuring fairness. (2) Customer service call center - customers are helped in order they called, first caller gets helped first. (3) BFS algorithm in graphs - explores nodes level by level, processing neighbors in order they were discovered. Other examples: process scheduling in OS, breadth-first tree traversal, keyboard buffer, network packet routing.',
                    keyPoints: [
                        'Any "first-come, first-served" scenario',
                        'BFS algorithm (level-order traversal)',
                        'Task scheduling and job processing',
                        'Buffering and data streaming',
                        'Order matters and fairness is important',
                    ],
                },
                {
                    id: 'q3',
                    question: 'What are the performance characteristics of queue operations with deque vs list?',
                    sampleAnswer: 'With deque: enqueue (append) is O(1), dequeue (popleft) is O(1). With list: enqueue (append) is O(1), but dequeue (pop(0)) is O(n) because all remaining elements must be shifted. For queue operations, deque is significantly better because both operations are O(1). List as queue causes O(n) dequeue, making it inefficient for large queues. Always use collections.deque for queue implementation in Python.',
                    keyPoints: [
                        'deque: O(1) for both enqueue and dequeue',
                        'list: O(1) enqueue, O(n) dequeue (pop(0))',
                        'list.pop(0) shifts all elements',
                        'deque is optimized for both ends',
                        'Use deque for efficient queues',
                    ],
                },
            ],
            multipleChoice: [
                {
                    id: 'mc1',
                    question: 'What does FIFO stand for?',
                    options: [
                        'First-In-First-Out',
                        'Fast-In-Fast-Out',
                        'Final-In-Final-Out',
                        'First-Item-First-Operation',
                    ],
                    correctAnswer: 0,
                    explanation:
                        'FIFO stands for First-In-First-Out, meaning the first element added is the first one removed, like people in a line.',
                },
                {
                    id: 'mc2',
                    question: 'Where are elements added and removed in a queue?',
                    options: [
                        'Added and removed from the front',
                        'Added and removed from the rear',
                        'Added at rear, removed from front',
                        'Added at front, removed from rear',
                    ],
                    correctAnswer: 2,
                    explanation:
                        'Elements are enqueued (added) at the rear and dequeued (removed) from the front, maintaining FIFO order.',
                },
                {
                    id: 'mc3',
                    question: 'Which real-world scenario best represents a queue?',
                    options: [
                        'Stack of plates',
                        'Pile of books',
                        'Line at a store checkout',
                        'Undo button in editor',
                    ],
                    correctAnswer: 2,
                    explanation:
                        'A checkout line is a perfect queue analogy - first person in line is first served (FIFO). Stacks of plates/books use LIFO.',
                },
                {
                    id: 'mc4',
                    question: 'What are the two primary operations of a queue?',
                    options: [
                        'Push and Pop',
                        'Enqueue and Dequeue',
                        'Insert and Delete',
                        'Add and Remove',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'The two primary queue operations are enqueue (add to rear) and dequeue (remove from front).',
                },
                {
                    id: 'mc5',
                    question:
                        'Which data structure is the opposite of a queue in terms of ordering?',
                    options: ['Array', 'Linked List', 'Stack', 'Tree'],
                    correctAnswer: 2,
                    explanation:
                        'A stack is the opposite of a queue: queue is FIFO (first in, first out) while stack is LIFO (last in, first out).',
                },
            ],
        },
        {
            id: 'operations',
            title: 'Queue Operations & Implementation',
            content: `## Core Queue Operations

A queue supports two fundamental operations:

### 1. **Enqueue (Add to rear)**
Add an element to the back of the queue.
\`\`\`python
queue.enqueue(item)  # Add item to rear
\`\`\`

### 2. **Dequeue (Remove from front)**
Remove and return the front element.
\`\`\`python
item = queue.dequeue()  # Remove from front
\`\`\`

### 3. **Peek/Front** (Optional but common)
View the front element without removing it.
\`\`\`python
item = queue.front()  # View front element
\`\`\`

### 4. **isEmpty** (Check if empty)
\`\`\`python
if queue.is_empty():
    print("Queue is empty")
\`\`\`

### 5. **Size** (Get number of elements)
\`\`\`python
length = queue.size()
\`\`\`

---

## Implementation Methods

There are three main ways to implement a queue in Python:

### Method 1: Using List (Simple but Inefficient)

❌ **Not recommended for production** - dequeue is O(n)

\`\`\`python
class QueueWithList:
    """Queue using Python list (inefficient)"""
    
    def __init__(self):
        self.items = []
    
    def enqueue(self, item):
        """Add to rear - O(1)"""
        self.items.append(item)
    
    def dequeue(self):
        """Remove from front - O(n) ❌ SLOW!"""
        if self.is_empty():
            raise IndexError("Dequeue from empty queue")
        return self.items.pop(0)  # O(n) - shifts all elements!
    
    def front(self):
        """Peek at front - O(1)"""
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.items[0]
    
    def is_empty(self):
        """Check if empty - O(1)"""
        return len(self.items) == 0
    
    def size(self):
        """Get size - O(1)"""
        return len(self.items)

# Usage
queue = QueueWithList()
queue.enqueue(1)
queue.enqueue(2)
queue.enqueue(3)
print(queue.dequeue())  # 1 (FIFO!)
print(queue.front())    # 2
\`\`\`

**Problem:** \`pop(0)\` is O(n) because it shifts all remaining elements left!

---

### Method 2: Using collections.deque (Recommended) ✅

**Best choice** - All operations O(1)

\`\`\`python
from collections import deque

class Queue:
    """Queue using deque (double-ended queue) - BEST IMPLEMENTATION"""
    
    def __init__(self):
        self.items = deque()  # Optimized for both ends
    
    def enqueue(self, item):
        """Add to rear - O(1)"""
        self.items.append(item)  # Append to right
    
    def dequeue(self):
        """Remove from front - O(1) ✅ FAST!"""
        if self.is_empty():
            raise IndexError("Dequeue from empty queue")
        return self.items.popleft()  # Pop from left - O(1)!
    
    def front(self):
        """Peek at front - O(1)"""
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.items[0]
    
    def is_empty(self):
        """Check if empty - O(1)"""
        return len(self.items) == 0
    
    def size(self):
        """Get size - O(1)"""
        return len(self.items)
    
    def __repr__(self):
        """String representation"""
        return f"Queue({list(self.items)})"

# Usage
queue = Queue()
queue.enqueue(1)
queue.enqueue(2)
queue.enqueue(3)
print(queue)            # Queue([1, 2, 3])
print(queue.dequeue())  # 1
print(queue.dequeue())  # 2
print(queue.front())    # 3
\`\`\`

**Why deque is best:**
- \`append()\` is O(1) for adding to rear
- \`popleft()\` is O(1) for removing from front
- Implemented as doubly-linked list internally
- This is what professionals use!

---

### Method 3: Using Two Stacks (Interview Favorite)

Sometimes asked in interviews: "Implement a queue using two stacks"

\`\`\`python
class QueueWithStacks:
    """Queue implemented using two stacks"""
    
    def __init__(self):
        self.stack_in = []   # For enqueue
        self.stack_out = []  # For dequeue
    
    def enqueue(self, item):
        """Add to rear - O(1)"""
        self.stack_in.append(item)
    
    def dequeue(self):
        """Remove from front - O(1) amortized"""
        # Move elements from stack_in to stack_out if needed
        if not self.stack_out:
            while self.stack_in:
                self.stack_out.append(self.stack_in.pop())
        
        if not self.stack_out:
            raise IndexError("Dequeue from empty queue")
        
        return self.stack_out.pop()
    
    def front(self):
        """Peek at front - O(1) amortized"""
        if not self.stack_out:
            while self.stack_in:
                self.stack_out.append(self.stack_in.pop())
        
        if not self.stack_out:
            raise IndexError("Queue is empty")
        
        return self.stack_out[-1]
    
    def is_empty(self):
        """Check if empty - O(1)"""
        return not self.stack_in and not self.stack_out

# How it works:
# stack_in:  [1, 2, 3] (top is 3)
# stack_out: []
# 
# On dequeue:
# 1. Pop all from stack_in to stack_out: [3, 2, 1] (top is 1)
# 2. Pop from stack_out returns 1 (first enqueued!)
# 
# Subsequent dequeues just pop from stack_out until empty
\`\`\`

---

## Complexity Comparison

| Operation | List | deque | Two Stacks |
|-----------|------|-------|------------|
| Enqueue | O(1) | **O(1)** | O(1) |
| Dequeue | ❌ O(n) | **O(1)** | O(1) amortized |
| Front | O(1) | **O(1)** | O(1) amortized |
| Space | O(n) | **O(n)** | O(n) |

**Recommendation:** Use \`collections.deque\` for real applications. Know the two-stacks approach for interviews.`,
            quiz: [
                {
                    id: 'q1',
                    question:
                        'Why is using a regular Python list for a queue inefficient? What specific operation causes the problem?',
                    sampleAnswer:
                        'Using a list for a queue is inefficient because dequeue (removing from front) requires pop(0), which is O(n). When we remove the first element, Python must shift all remaining n-1 elements one position to the left to fill the gap. So if the queue has 1000 elements, removing one requires 999 shift operations. This makes dequeue O(n) instead of O(1). In contrast, enqueue using append() is O(1) because it adds to the end. For a proper queue, use collections.deque which has O(1) operations for both ends.',
                    keyPoints: [
                        'pop(0) is O(n) - shifts all elements',
                        'Must shift n-1 elements after removal',
                        'Makes queue operations slow',
                        'deque has O(1) for both ends',
                        'List good for stack, bad for queue',
                    ],
                },
                {
                    id: 'q2',
                    question: 'How do you implement a queue using two stacks? Explain the amortized O(1) complexity.',
                    sampleAnswer: 'Use two stacks: stack_in for enqueue, stack_out for dequeue. Enqueue: push to stack_in (O(1)). Dequeue: if stack_out is empty, transfer all elements from stack_in to stack_out (reversing order), then pop from stack_out. This transfer happens rarely. Amortized analysis: each element is pushed once to stack_in and moved once to stack_out over its lifetime, giving O(1) amortized per operation. Example: enqueue(1,2,3) → stack_in=[3,2,1]. Dequeue → transfer to stack_out=[1,2,3], pop 1.',
                    keyPoints: [
                        'Two stacks: stack_in and stack_out',
                        'Enqueue pushes to stack_in',
                        'Dequeue pops from stack_out',
                        'Transfer from stack_in when stack_out empty',
                        'Amortized O(1) per operation',
                        'Each element moved at most twice total',
                    ],
                },
                {
                    id: 'q3',
                    question: 'What is a circular queue and when would you use it?',
                    sampleAnswer: 'A circular queue uses a fixed-size array with two pointers (front and rear) that wrap around to the beginning when reaching the end. It prevents wasted space from regular array queue where front pointer moves right, leaving unused space. Use cases: 1) Fixed buffer size known (streaming data, print spooler), 2) Prevent memory fragmentation, 3) Efficient ring buffer implementation. Implementation: use modulo operator for wrapping: rear = (rear + 1) % capacity. Must track size to distinguish empty (size==0) from full (size==capacity).',
                    keyPoints: [
                        'Fixed-size array with wraparound',
                        'Two pointers: front and rear',
                        'Use modulo for wraparound: (index + 1) % capacity',
                        'Prevents wasted space from linear array queue',
                        'Ideal for fixed buffer size',
                        'Track size to distinguish empty vs full',
                    ],
                },
            ],
            multipleChoice: [
                {
                    id: 'mc1',
                    question:
                        'What is the time complexity of enqueue and dequeue in a properly implemented queue?',
                    options: [
                        'Both O(1)',
                        'Both O(n)',
                        'Enqueue O(1), Dequeue O(n)',
                        'Enqueue O(n), Dequeue O(1)',
                    ],
                    correctAnswer: 0,
                    explanation:
                        'A properly implemented queue (using deque or linked list) has O(1) time complexity for both enqueue and dequeue operations.',
                },
                {
                    id: 'mc2',
                    question: 'Why is using a Python list as a queue inefficient?',
                    options: [
                        'Enqueue (append) is O(n)',
                        'Dequeue (pop(0)) is O(n) due to shifting',
                        'It uses too much memory',
                        'Lists cannot hold queue data',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'pop(0) removes the first element and shifts all remaining elements left, making it O(n). Use collections.deque for O(1) operations.',
                },
                {
                    id: 'mc3',
                    question:
                        "What is Python's recommended data structure for implementing a queue?",
                    options: ['list', 'tuple', 'collections.deque', 'set'],
                    correctAnswer: 2,
                    explanation:
                        'collections.deque (double-ended queue) provides O(1) append and popleft operations, making it ideal for queues.',
                },
                {
                    id: 'mc4',
                    question: 'How would you implement a queue using two stacks?',
                    options: [
                        'Use one stack for everything',
                        'Push to stack1, pop from stack1',
                        'Push to stack1, transfer to stack2 when popping',
                        "It's impossible",
                    ],
                    correctAnswer: 2,
                    explanation:
                        'Push all elements to stack1. When dequeuing, if stack2 is empty, transfer all from stack1 to stack2 (reversing order), then pop from stack2.',
                },
                {
                    id: 'mc5',
                    question:
                        'What should happen when you try to dequeue from an empty queue?',
                    options: [
                        'Return None',
                        'Return 0',
                        'Raise an exception or return error indicator',
                        'Do nothing',
                    ],
                    correctAnswer: 2,
                    explanation:
                        'Dequeuing from an empty queue should raise an exception (like IndexError) or return a special error indicator to prevent invalid operations.',
                },
            ],
        },
        {
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
            quiz: [
                {
                    id: 'q1',
                    question:
                        'Explain how a circular queue works and why it uses the modulo operator.',
                    sampleAnswer:
                        'A circular queue uses a fixed-size array where the rear wraps back to the front when it reaches the end. It uses modulo operator (index + 1) % capacity to wrap indices around. For example, in a queue of capacity 5, after index 4, the next index is (4 + 1) % 5 = 0. This reuses space from dequeued elements instead of shifting array elements or growing the array. Both front and rear pointers wrap around. This is efficient for fixed-size buffers where we want O(1) enqueue/dequeue without wasted space or expensive shifts.',
                    keyPoints: [
                        'Fixed-size array that wraps around',
                        'Modulo operator: (index + 1) % capacity',
                        'Reuses space from dequeued elements',
                        'Both front and rear wrap around',
                        'O(1) operations, no shifting or growing',
                    ],
                },
                {
                    id: 'q2',
                    question:
                        'What is the difference between a regular queue and a priority queue? When would you use each?',
                    sampleAnswer:
                        'Regular queue follows strict FIFO - first in is first out. Priority queue dequeues elements by priority, not insertion order. Regular queue uses deque with O(1) enqueue/dequeue. Priority queue uses a heap with O(log n) enqueue/dequeue. Use regular queue for BFS, task processing where order matters (like print queue). Use priority queue when some elements are more important: Dijkstra\'s shortest path (process closest node first), CPU scheduling (high priority tasks first), A* search, event simulation with timestamps. Key difference: FIFO vs priority-based.',
                    keyPoints: [
                        'Regular: strict FIFO order',
                        'Priority: dequeue by priority, not order',
                        'Regular: O(1) operations with deque',
                        'Priority: O(log n) with heap',
                        'Use priority when importance varies',
                    ],
                },
                {
                    id: 'q3',
                    question:
                        'Why is a deque better than a regular queue for sliding window problems?',
                    sampleAnswer:
                        'Sliding window problems often need to add/remove from both ends efficiently. Deque provides O(1) operations at both ends: append (add right), appendleft (add left), pop (remove right), popleft (remove left). Regular queue only efficiently removes from front. For example, in "sliding window maximum," we maintain indices in decreasing order of values. When window slides: remove expired indices from front (popleft), remove smaller values from rear before adding new (pop). Both operations are O(1) with deque. Using a regular list would be O(n) for removing from front. Deque is implemented as a doubly-linked list, enabling efficient operations on both ends.',
                    keyPoints: [
                        'Deque: O(1) add/remove from both ends',
                        'Regular queue: only efficient at one end',
                        'Sliding window: remove old (front), add new (rear)',
                        'Maintain order/constraints by modifying both ends',
                        'Deque is doubly-linked list internally',
                    ],
                },
            ],
            multipleChoice: [
                {
                    id: 'mc1',
                    question: 'What is the main advantage of a circular queue over a regular array-based queue?',
                    options: [
                        'It is faster',
                        'It reuses space from dequeued elements without shifting',
                        'It uses less memory',
                        'It maintains sorted order',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Circular queues reuse space by wrapping the rear pointer back to the front using modulo arithmetic. This avoids the need to shift elements or grow the array when the rear reaches the end.',
                },
                {
                    id: 'mc2',
                    question: 'In a priority queue, what data structure is typically used for implementation?',
                    options: [
                        'Array',
                        'Linked List',
                        'Heap',
                        'Hash Table',
                    ],
                    correctAnswer: 2,
                    explanation:
                        'Priority queues are typically implemented using a heap (min-heap or max-heap) which provides O(log n) enqueue and dequeue operations based on priority.',
                },
                {
                    id: 'mc3',
                    question: 'Which queue variation allows efficient addition and removal from both ends?',
                    options: [
                        'Circular Queue',
                        'Priority Queue',
                        'Deque (Double-Ended Queue)',
                        'Blocking Queue',
                    ],
                    correctAnswer: 2,
                    explanation:
                        'Deque (double-ended queue) supports O(1) operations at both ends: append, appendleft, pop, and popleft.',
                },
                {
                    id: 'mc4',
                    question: 'What is the time complexity of dequeue operation in a priority queue implemented with a heap?',
                    options: [
                        'O(1)',
                        'O(log n)',
                        'O(n)',
                        'O(n log n)',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Dequeue (removing the highest priority element) in a heap-based priority queue requires removing the root and reheapifying, which takes O(log n) time.',
                },
                {
                    id: 'mc5',
                    question: 'Which Python module provides a thread-safe blocking queue?',
                    options: [
                        'collections',
                        'queue',
                        'threading',
                        'asyncio',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'The queue module in Python provides Queue, which is a thread-safe blocking queue useful for producer-consumer patterns in multithreaded applications.',
                },
            ],
        },
        {
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

def bfs_tree(root):
    """BFS traversal of binary tree"""
    if not root:
        return []
    
    result = []
    queue = deque([root])  # Start with root
    
    while queue:
        # Process current level
        node = queue.popleft()  # FIFO!
        result.append(node.val)
        
        # Add children to queue (next level)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    
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
def level_order(root):
    """Return list of lists, one per level"""
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)  # Current level size
        level = []
        
        # Process entire level
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            
            # Add next level
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level)
    
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

**Pattern:** Use \`len(queue)\` to know how many nodes are at current level.

---

## Pattern 3: Sliding Window Maximum

Use deque to efficiently track maximum in sliding window.

\`\`\`python
from collections import deque

def max_sliding_window(nums, k):
    """Find max in each window of size k"""
    result = []
    dq = deque()  # Store indices
    
    for i in range(len(nums)):
        # Remove indices outside window
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        
        # Remove smaller elements (they can't be max)
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        
        dq.append(i)
        
        # Add to result once window is full
        if i >= k - 1:
            result.append(nums[dq[0]])  # Front is max
    
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
    
    def ping(self, t):
        """Add request at time t, return count in [t-3000, t]"""
        self.requests.append(t)
        
        # Remove requests older than t - 3000
        while self.requests and self.requests[0] < t - 3000:
            self.requests.popleft()
        
        return len(self.requests)

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
    
    def push(self, x):
        """Push to stack - O(1)"""
        self.q1.append(x)
    
    def pop(self):
        """Pop from stack - O(n)"""
        # Move all but last to q2
        while len(self.q1) > 1:
            self.q2.append(self.q1.popleft())
        
        # Remove last (top of stack)
        result = self.q1.popleft()
        
        # Swap queues
        self.q1, self.q2 = self.q2, self.q1
        
        return result
    
    def top(self):
        """Peek at top - O(n)"""
        # Similar to pop but add back
        while len(self.q1) > 1:
            self.q2.append(self.q1.popleft())
        
        result = self.q1[0]
        self.q2.append(self.q1.popleft())
        
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
            quiz: [
                {
                    id: 'q1',
                    question:
                        'Explain why BFS uses a queue instead of a stack. What would happen if you used a stack?',
                    sampleAnswer:
                        'BFS uses a queue because FIFO order ensures we explore all nodes at depth d before any node at depth d+1. Queue processes nodes in the order they were discovered. If we used a stack (LIFO), we would get DFS instead: we would go as deep as possible before backtracking. For example, in a tree, BFS with queue visits root, then all level 1 nodes, then all level 2 nodes (level-order). With a stack, we would visit root, then immediately go deep down one branch before exploring other branches at level 1. The traversal order would be completely different - depth-first rather than breadth-first.',
                    keyPoints: [
                        'Queue FIFO: process nodes in discovery order',
                        'Ensures all depth d before depth d+1',
                        'Stack would give DFS (depth-first)',
                        'Stack goes deep immediately, queue goes wide',
                        'Different traversal order: level-by-level vs deep-first',
                    ],
                },
                {
                    id: 'q2',
                    question:
                        'How do you implement a queue using two stacks? Explain the enqueue and dequeue operations.',
                    sampleAnswer:
                        'Two-stacks queue uses stack_in for enqueue and stack_out for dequeue. Enqueue: push to stack_in - O(1). Dequeue: if stack_out is empty, transfer all from stack_in to stack_out (reversing order), then pop from stack_out. Transfer is expensive O(n), but it\'s amortized O(1) because each element is transferred at most once. Example: enqueue 1,2,3 to stack_in [1,2,3]. First dequeue transfers to stack_out [3,2,1], then pops 1. Next dequeues just pop from stack_out (2, then 3) without transfers. The key insight: two reversals (stack_in to stack_out) restore FIFO order.',
                    keyPoints: [
                        'stack_in for enqueue, stack_out for dequeue',
                        'Enqueue: push to stack_in, O(1)',
                        'Dequeue: pop from stack_out, transfer if empty',
                        'Transfer is O(n) but amortized O(1)',
                        'Two stack reversals restore FIFO order',
                    ],
                },
                {
                    id: 'q3',
                    question:
                        'What is the pattern for "sliding window maximum" and why does it use a deque instead of a regular queue?',
                    sampleAnswer:
                        'Sliding window maximum finds the max element in each window of size k as it slides. We maintain a deque of indices in decreasing order of their values. For each element: (1) Remove indices outside window from front (dequeue from left), (2) Remove indices with smaller values from back (we pop from right because current element makes them useless), (3) Add current index to back, (4) Front of deque is the max for this window. We need deque not regular queue because we remove from BOTH ends: old indices from front (outside window), useless smaller indices from back (won\'t be max). This is O(n) because each element enters and leaves deque at most once.',
                    keyPoints: [
                        'Maintain deque of indices in decreasing value order',
                        'Remove old indices from front (outside window)',
                        'Remove smaller indices from back (useless)',
                        'Need both-end operations: deque not queue',
                        'O(n): each element processed once',
                    ],
                },
            ],
            multipleChoice: [
                {
                    id: 'mc1',
                    question: 'What is the most common application of queues in algorithms?',
                    options: [
                        'Sorting',
                        'Breadth-First Search (BFS)',
                        'Binary search',
                        'Finding duplicates',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Queues are essential for BFS, which explores nodes level by level. The queue maintains the order of nodes to visit, ensuring breadth-first traversal.',
                },
                {
                    id: 'mc2',
                    question: 'In level-order tree traversal, what determines when to start processing a new level?',
                    options: [
                        'When the queue is empty',
                        'By counting len(queue) at the start of each level',
                        'When we see a None node',
                        'After processing the root',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'We capture len(queue) at the start of each level iteration. This tells us exactly how many nodes are in the current level, allowing us to process them separately.',
                },
                {
                    id: 'mc3',
                    question: 'What is the amortized time complexity of dequeue in a two-stack queue implementation?',
                    options: [
                        'O(1)',
                        'O(log n)',
                        'O(n)',
                        'O(n²)',
                    ],
                    correctAnswer: 0,
                    explanation:
                        'While individual dequeue operations can be O(n) when transferring elements, the amortized complexity is O(1) because each element is transferred at most once.',
                },
                {
                    id: 'mc4',
                    question: 'For the "shortest path in unweighted graph" problem, which algorithm should you use?',
                    options: [
                        'DFS',
                        'BFS with queue',
                        'Dijkstra',
                        'Binary search',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'BFS with a queue finds the shortest path in unweighted graphs because it explores nodes level by level, guaranteeing the first time you reach a node is via the shortest path.',
                },
                {
                    id: 'mc5',
                    question: 'In sliding window maximum, why do we need a deque instead of a priority queue?',
                    options: [
                        'Deque is faster',
                        'Priority queue cannot maintain window order',
                        'We need O(1) removal from both ends, priority queue is O(log n)',
                        'Deque uses less memory',
                    ],
                    correctAnswer: 2,
                    explanation:
                        'We need to efficiently remove elements from both ends: old elements from the front (outside window) and smaller elements from the back. Deque provides O(1) for both, while priority queue would be O(log n).',
                },
            ],
        },
        {
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
            quiz: [
                {
                    id: 'q1',
                    question:
                        'Why is list.pop(0) O(n) but deque.popleft() O(1)? Explain the underlying data structure difference.',
                    sampleAnswer:
                        'Lists in Python are implemented as dynamic arrays (contiguous memory). When you pop(0) from the front, all remaining elements must shift left by one position to fill the gap - this is O(n) copying. Deque is implemented as a doubly-linked list of blocks (not a single array). Each block contains multiple elements. Removing from the left just adjusts the head pointer to the next block or next element within the block - no shifting needed. This makes popleft() O(1). Similarly, appendleft() is O(1) because we just add a new block or element at the head. Lists optimize for index access O(1), deques optimize for both-end operations O(1).',
                    keyPoints: [
                        'List: dynamic array, contiguous memory',
                        'list.pop(0): shift all elements, O(n)',
                        'Deque: doubly-linked list of blocks',
                        'deque.popleft(): adjust head pointer, O(1)',
                        'Lists for indexing, deques for both-end operations',
                    ],
                },
                {
                    id: 'q2',
                    question:
                        'What is amortized O(1) and why does the two-stack queue implementation achieve it?',
                    sampleAnswer:
                        'Amortized O(1) means that while some operations are expensive, the average cost per operation over many operations is O(1). In two-stack queue, most dequeues are O(1) (just pop from stack_out). Occasionally, when stack_out is empty, we must transfer all n elements from stack_in to stack_out - this single operation is O(n). But here\'s the key: each element is enqueued once, transferred at most once, and dequeued once. So for n operations total, we do at most 3n single-element operations, which averages to O(1) per operation. The expensive transfers are rare and their cost is "spread out" (amortized) over many cheap operations.',
                    keyPoints: [
                        'Most operations cheap, occasional expensive one',
                        'Average cost over many operations is O(1)',
                        'Each element: enqueued once, transferred once, dequeued once',
                        'Total 3n operations for n elements = O(1) average',
                        'Expensive transfers are rare and cost is spread out',
                    ],
                },
                {
                    id: 'q3',
                    question:
                        'Compare the space complexity of BFS vs DFS. Which uses more memory and why?',
                    sampleAnswer:
                        'For a tree of depth d with branching factor b, BFS uses O(b^d) space (width of last level) because the queue holds all nodes at the current level. DFS uses O(d) space for the recursion stack or explicit stack (height of tree). BFS is wider, DFS is deeper. For a binary tree of depth 10, BFS can have 2^10 = 1024 nodes in queue at the deepest level. DFS only needs 10 stack frames. BFS uses exponentially more space as depth increases for trees with high branching factor. However, for graphs with cycles, DFS also needs O(V) visited set. In general: BFS = O(width), DFS = O(depth). Use DFS if memory is tight and tree is wide.',
                    keyPoints: [
                        'BFS: O(b^d) space, holds entire level in queue',
                        'DFS: O(d) space, holds path in stack',
                        'BFS wider, DFS deeper',
                        'BFS can use exponentially more memory',
                        'Choose DFS if memory tight and tree is wide',
                    ],
                },
            ],
            multipleChoice: [
                {
                    id: 'mc1',
                    question: 'What is the time complexity of enqueue operation using collections.deque?',
                    options: [
                        'O(1)',
                        'O(log n)',
                        'O(n)',
                        'O(n log n)',
                    ],
                    correctAnswer: 0,
                    explanation:
                        'deque.append() (enqueue) is O(1) because deque is implemented as a doubly-linked list of blocks, allowing constant-time addition to either end.',
                },
                {
                    id: 'mc2',
                    question: 'Why should you avoid using list.pop(0) for queue operations?',
                    options: [
                        'It does not work correctly',
                        'It is O(n) because all elements must shift',
                        'It only works for small lists',
                        'It uses too much memory',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'list.pop(0) is O(n) because removing the first element requires shifting all remaining elements left by one position in the underlying array.',
                },
                {
                    id: 'mc3',
                    question: 'What is the space complexity of BFS traversal on a binary tree with n nodes?',
                    options: [
                        'O(1)',
                        'O(log n)',
                        'O(n)',
                        'O(n²)',
                    ],
                    correctAnswer: 2,
                    explanation:
                        'In the worst case, the queue will hold all nodes at the deepest level. For a complete binary tree, this can be up to n/2 nodes, which is O(n) space.',
                },
                {
                    id: 'mc4',
                    question: 'In the two-stack queue implementation, what is the amortized time complexity of dequeue?',
                    options: [
                        'O(1)',
                        'O(log n)',
                        'O(n)',
                        'O(n log n)',
                    ],
                    correctAnswer: 0,
                    explanation:
                        'While a single dequeue can be O(n) when transferring elements, the amortized complexity is O(1) because each element is transferred at most once over all operations.',
                },
                {
                    id: 'mc5',
                    question: 'Which operation is NOT O(1) with collections.deque?',
                    options: [
                        'append (add to right)',
                        'appendleft (add to left)',
                        'pop (remove from right)',
                        'Accessing middle element by index',
                    ],
                    correctAnswer: 3,
                    explanation:
                        'Deque is optimized for both-end operations (all O(1)), but random access to middle elements is O(n) because it is a linked structure, not an array.',
                },
            ],
        },
        {
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

def bfs(start):
    """Standard BFS template"""
    queue = deque([start])
    visited = {start}
    
    while queue:
        current = queue.popleft()
        
        # Process current
        # ...
        
        # Add neighbors
        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return result
\`\`\`

**Level-Order Template:**
\`\`\`python
def level_order(root):
    """Process tree level by level"""
    if not root:
        return []
    
    queue = deque([root])
    result = []
    
    while queue:
        level_size = len(queue)  # Key: capture level size
        level = []
        
        for _ in range(level_size):  # Process entire level
            node = queue.popleft()
            level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level)
    
    return result
\`\`\`

**Sliding Window Template:**
\`\`\`python
def sliding_window(nums, k):
    """Sliding window with deque"""
    dq = deque()
    result = []
    
    for i in range(len(nums)):
        # Remove out-of-window elements
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        
        # Maintain deque property (e.g., decreasing for max)
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        
        dq.append(i)
        
        # Add to result once window is full
        if i >= k - 1:
            result.append(nums[dq[0]])
    
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
    level_size = len(queue)  # Capture current level
    for _ in range(level_size):
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
queue.append(item)
queue.pop()  # Removes last item

# ✅ GOOD - this is a queue (FIFO)
queue.append(item)
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
            quiz: [
                {
                    id: 'q1',
                    question:
                        'What are the key indicators that a problem requires BFS with a queue rather than DFS?',
                    sampleAnswer:
                        'Use BFS with queue when: (1) Problem asks for "shortest path" in unweighted graph - BFS guarantees finding shortest path first, (2) Need "level by level" or "layer by layer" processing, (3) Problem mentions "minimum depth/steps/moves", (4) Need to find nodes at distance k from source, (5) "Closest" or "nearest" in unweighted context. Use DFS when: detecting cycles, finding any path (not shortest), exploring all possibilities with backtracking, topological sort. Key difference: BFS explores breadth-first (all neighbors before going deeper), DFS explores depth-first (go as deep as possible). BFS with queue is also better when tree is very deep but we need shallow answer.',
                    keyPoints: [
                        'BFS: shortest path in unweighted graphs',
                        'BFS: level-by-level processing',
                        'BFS: minimum depth/steps',
                        'DFS: any path, backtracking, cycles',
                        'BFS better for shallow answers in deep trees',
                    ],
                },
                {
                    id: 'q2',
                    question:
                        'Walk through the level-order traversal pattern. Why is capturing len(queue) crucial?',
                    sampleAnswer:
                        'Level-order processes tree level by level. At start of each iteration, len(queue) tells us how many nodes are in the current level. We loop exactly that many times to process the current level, adding children to queue for the next level. Without capturing len(queue), we would process children immediately instead of waiting for next level. Example: level 0 has root (1 node), len(queue)=1, process 1 node, add 2 children. Level 1: len(queue)=2, process 2 nodes, add 4 children. Level 2: len(queue)=4. The captured length separates levels. Alternative: use sentinel value (like None) to mark level boundaries, but len(queue) is cleaner.',
                    keyPoints: [
                        'len(queue) at start = nodes in current level',
                        'Loop that many times to process just this level',
                        'Children added for next level, not processed now',
                        'Separates levels cleanly',
                        'Without it, would process children immediately',
                    ],
                },
                {
                    id: 'q3',
                    question:
                        'What are the common mistakes when implementing queue-based solutions in interviews?',
                    sampleAnswer:
                        'Common mistakes: (1) Using list.pop(0) instead of deque.popleft() - O(n) vs O(1), (2) Forgetting to mark nodes as visited in graph BFS - causes infinite loops, (3) Not capturing len(queue) before level loop - mixes levels, (4) Adding node to queue after checking visited instead of when marking visited - adds duplicates, (5) Using queue.pop() instead of popleft() - processes LIFO not FIFO, (6) Not checking empty queue before popleft() - IndexError, (7) Initializing queue incorrectly (forgetting starting nodes). Prevention: use templates, test with simple example, check visited logic, verify FIFO order.',
                    keyPoints: [
                        'Use deque.popleft(), not list.pop(0)',
                        'Mark visited when adding to queue',
                        'Capture len(queue) for level-order',
                        'Use popleft() not pop()',
                        'Check empty before popleft()',
                    ],
                },
            ],
            multipleChoice: [
                {
                    id: 'mc1',
                    question: 'Which keyword in a problem description most strongly indicates using BFS with a queue?',
                    options: [
                        'Find all paths',
                        'Shortest path in unweighted graph',
                        'Detect cycles',
                        'Generate permutations',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'BFS with a queue is optimal for finding shortest paths in unweighted graphs because it explores nodes level by level, guaranteeing the first time you reach a node is via the shortest path.',
                },
                {
                    id: 'mc2',
                    question: 'In the standard BFS template, when should you mark a node as visited?',
                    options: [
                        'After dequeuing it',
                        'When adding it to the queue',
                        'After processing all its neighbors',
                        'Before the while loop starts',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Mark nodes as visited when adding them to the queue, not when dequeuing. This prevents adding the same node multiple times to the queue before it is processed.',
                },
                {
                    id: 'mc3',
                    question: 'What is the purpose of capturing len(queue) in level-order traversal?',
                    options: [
                        'To check if the queue is empty',
                        'To know how many nodes are in the current level',
                        'To calculate total nodes in tree',
                        'To prevent infinite loops',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Capturing len(queue) at the start tells us exactly how many nodes are in the current level, allowing us to process them separately from the next level.',
                },
                {
                    id: 'mc4',
                    question: 'Which queue problem pattern is typically categorized as Hard difficulty?',
                    options: [
                        'Binary Tree Level Order Traversal',
                        'Implement Queue using Stacks',
                        'Sliding Window Median',
                        'Number of Recent Calls',
                    ],
                    correctAnswer: 2,
                    explanation:
                        'Sliding Window Median requires maintaining two heaps (or balanced data structure) along with a sliding window, making it a Hard problem. Simple level-order and queue implementations are Easy/Medium.',
                },
                {
                    id: 'mc5',
                    question: 'What is the most important thing to verify before saying your solution is complete in a queue interview?',
                    options: [
                        'The code compiles',
                        'You used the right imports',
                        'You analyzed time/space complexity correctly',
                        'You added comments',
                    ],
                    correctAnswer: 2,
                    explanation:
                        'Always analyze and verify your time and space complexity. Interviewers want to see you understand not just that it works, but why it works and how efficient it is. This demonstrates algorithmic thinking.',
                },
            ],
        },
    ],
    keyTakeaways: [
        'Queue is a FIFO (First-In-First-Out) data structure',
        'Two main operations: enqueue (add to rear) and dequeue (remove from front)',
        'Python: use collections.deque for O(1) operations; avoid list for queue',
        'Essential for BFS (Breadth-First Search) algorithms',
        'Common applications: task scheduling, buffering, level-order traversal',
        'Circular queue uses modular arithmetic to reuse space efficiently',
        'Priority queue: elements dequeued based on priority, not FIFO order',
        'Typical complexity: O(1) for enqueue and dequeue operations',
    ],
    relatedProblems: [
        'implement-queue-using-stacks',
        'design-circular-queue',
        'sliding-window-maximum',
    ],
};
