/**
 * Stack & Queue Designs Section
 */

export const stackqueuedesignsSection = {
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
    
    def push (self, val):
        self.stack.append (val)
        # Push current minimum
        min_val = min (val, self.min_stack[-1] if self.min_stack else val)
        self.min_stack.append (min_val)
    
    def pop (self):
        self.stack.pop()
        self.min_stack.pop()  # Keep in sync
    
    def top (self):
        return self.stack[-1]
    
    def getMin (self):
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
    
    def push (self, val):
        if not self.stack:
            self.stack.append((val, val))
        else:
            current_min = min (val, self.stack[-1][1])
            self.stack.append((val, current_min))
    
    def pop (self):
        self.stack.pop()
    
    def top (self):
        return self.stack[-1][0]
    
    def getMin (self):
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
    
    def push (self, val):
        self.stack.append (val)
        if not self.min_stack or val < self.min_stack[-1][0]:
            self.min_stack.append((val, 1))
        elif val == self.min_stack[-1][0]:
            self.min_stack[-1] = (val, self.min_stack[-1][1] + 1)
    
    def pop (self):
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
    
    def enqueue (self, x):
        self.stack_in.append (x)  # O(1)
    
    def dequeue (self):
        if not self.stack_out:
            # Transfer all from in to out (reverses order)
            while self.stack_in:
                self.stack_out.append (self.stack_in.pop())
        return self.stack_out.pop() if self.stack_out else None
    
    def peek (self):
        if not self.stack_out:
            while self.stack_in:
                self.stack_out.append (self.stack_in.pop())
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
    
    def push (self, x):
        # Add to q2
        self.q2.append (x)
        # Move all from q1 to q2 (x is now at front)
        while self.q1:
            self.q2.append (self.q1.popleft())
        # Swap names
        self.q1, self.q2 = self.q2, self.q1
    
    def pop (self):
        return self.q1.popleft() if self.q1 else None
    
    def top (self):
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
    
    def push (self, x):
        self.q.append (x)
        # Rotate: move all elements before x to after x
        for _ in range (len (self.q) - 1):
            self.q.append (self.q.popleft())
    
    def pop (self):
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
};
