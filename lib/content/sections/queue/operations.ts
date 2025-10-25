/**
 * Queue Operations & Implementation Section
 */

export const operationsSection = {
  id: 'operations',
  title: 'Queue Operations & Implementation',
  content: `## Core Queue Operations

A queue supports two fundamental operations:

### 1. **Enqueue (Add to rear)**
Add an element to the back of the queue.
\`\`\`python
queue.enqueue (item)  # Add item to rear
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
    
    def enqueue (self, item):
        """Add to rear - O(1)"""
        self.items.append (item)
    
    def dequeue (self):
        """Remove from front - O(n) ❌ SLOW!"""
        if self.is_empty():
            raise IndexError("Dequeue from empty queue")
        return self.items.pop(0)  # O(n) - shifts all elements!
    
    def front (self):
        """Peek at front - O(1)"""
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.items[0]
    
    def is_empty (self):
        """Check if empty - O(1)"""
        return len (self.items) == 0
    
    def size (self):
        """Get size - O(1)"""
        return len (self.items)

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
    
    def enqueue (self, item):
        """Add to rear - O(1)"""
        self.items.append (item)  # Append to right
    
    def dequeue (self):
        """Remove from front - O(1) ✅ FAST!"""
        if self.is_empty():
            raise IndexError("Dequeue from empty queue")
        return self.items.popleft()  # Pop from left - O(1)!
    
    def front (self):
        """Peek at front - O(1)"""
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.items[0]
    
    def is_empty (self):
        """Check if empty - O(1)"""
        return len (self.items) == 0
    
    def size (self):
        """Get size - O(1)"""
        return len (self.items)
    
    def __repr__(self):
        """String representation"""
        return f"Queue({list (self.items)})"

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
    
    def enqueue (self, item):
        """Add to rear - O(1)"""
        self.stack_in.append (item)
    
    def dequeue (self):
        """Remove from front - O(1) amortized"""
        # Move elements from stack_in to stack_out if needed
        if not self.stack_out:
            while self.stack_in:
                self.stack_out.append (self.stack_in.pop())
        
        if not self.stack_out:
            raise IndexError("Dequeue from empty queue")
        
        return self.stack_out.pop()
    
    def front (self):
        """Peek at front - O(1) amortized"""
        if not self.stack_out:
            while self.stack_in:
                self.stack_out.append (self.stack_in.pop())
        
        if not self.stack_out:
            raise IndexError("Queue is empty")
        
        return self.stack_out[-1]
    
    def is_empty (self):
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
};
