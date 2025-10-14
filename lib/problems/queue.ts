/**
 * Queue problems
 */

import { Problem } from '../types';

export const queueProblems: Problem[] = [
  {
    id: 'queue-implement-queue',
    title: 'Implement Queue using Stacks',
    difficulty: 'Easy',
    topic: 'Queue',
    description: `Implement a first in first out (FIFO) queue using only two stacks.

The implemented queue should support all the functions of a normal queue (push, peek, pop, and empty).

**Requirements:**
- \`push(x)\` - Pushes element x to the back of queue
- \`pop()\` - Removes the element from front of queue and returns it
- \`peek()\` - Returns the element at front of queue
- \`empty()\` - Returns true if queue is empty, false otherwise

**Constraints:**
- You must use only standard stack operations (push, pop, peek/top, size, is empty)
- Depending on your language, stack may not be supported natively. You can simulate a stack using a list.

This is a classic problem that tests understanding of both stacks and queues!`,
    examples: [
      {
        input: 'push(1), push(2), peek(), pop(), empty()',
        output: '1, 1, false',
      },
    ],
    constraints: [
      '1 <= x <= 9',
      'At most 100 calls will be made',
      'All pop and peek calls are valid',
    ],
    hints: [
      'Use two stacks: one for input, one for output',
      'Push always goes to the input stack',
      'When popping/peeking, transfer elements from input to output stack if needed',
      'The transfer reverses order, giving us FIFO',
      'Amortized O(1) for all operations',
    ],
    starterCode: `class MyQueue:
    """
    Implement Queue using two stacks.
    """
    
    def __init__(self):
        """Initialize your data structure here."""
        pass
    
    def push(self, x: int) -> None:
        """Push element x to the back of queue."""
        pass
    
    def pop(self) -> int:
        """Remove and return element from front of queue."""
        pass
    
    def peek(self) -> int:
        """Get the front element."""
        pass
    
    def empty(self) -> bool:
        """Return whether queue is empty."""
        pass


# Test code
queue = MyQueue()
queue.push(1)
queue.push(2)
print(queue.peek())   # Expected: 1
print(queue.pop())    # Expected: 1
print(queue.empty())  # Expected: False
`,
    testCases: [
      {
        input: [
          ['push', 'push', 'peek', 'pop', 'empty'],
          [[1], [2], [], [], []],
        ],
        expected: [null, null, 1, 1, false],
      },
    ],
    solution: `class MyQueue:
    """Implement Queue using two stacks"""
    
    def __init__(self):
        self.stack_in = []   # For push operations
        self.stack_out = []  # For pop/peek operations
    
    def push(self, x: int) -> None:
        """Push to back - O(1)"""
        self.stack_in.append(x)
    
    def pop(self) -> int:
        """Pop from front - O(1) amortized"""
        self._move()
        return self.stack_out.pop()
    
    def peek(self) -> int:
        """Peek at front - O(1) amortized"""
        self._move()
        return self.stack_out[-1]
    
    def empty(self) -> bool:
        """Check if empty - O(1)"""
        return not self.stack_in and not self.stack_out
    
    def _move(self):
        """Move elements from stack_in to stack_out if needed"""
        if not self.stack_out:
            while self.stack_in:
                self.stack_out.append(self.stack_in.pop())


# How it works:
# stack_in:  [1, 2, 3]  (top is 3)
# stack_out: []
# 
# On pop/peek:
# Transfer all from stack_in to stack_out: [3, 2, 1] (top is 1)
# Now pop from stack_out returns 1 (first enqueued!)
# 
# Time: O(1) amortized - each element moved at most twice
# Space: O(n) for n elements`,
    timeComplexity: 'O(1) amortized for all operations',
    spaceComplexity: 'O(n) for n elements',
    followUp: [
      'Can you implement this with just one stack?',
      'What is the amortized time complexity analysis?',
      'How would you implement a stack using queues?',
    ],
  },
  {
    id: 'queue-recent-calls',
    title: 'Number of Recent Calls',
    difficulty: 'Easy',
    topic: 'Queue',
    description: `You have a \`RecentCounter\` class which counts the number of recent requests within a certain time frame.

Implement the \`RecentCounter\` class:
- \`RecentCounter()\` - Initializes the counter with zero recent requests
- \`ping(int t)\` - Adds a new request at time \`t\`, where \`t\` represents some time in milliseconds, and returns the number of requests that have happened in the past 3000 milliseconds (including the new request). Specifically, return the number of requests that have happened in the inclusive range [t - 3000, t].

It is **guaranteed** that every call to \`ping\` uses a strictly larger value of \`t\` than before.`,
    examples: [
      {
        input: 'RecentCounter(), ping(1), ping(100), ping(3001), ping(3002)',
        output: '1, 2, 3, 3',
      },
    ],
    constraints: [
      '1 <= t <= 10⁹',
      'Each test case will call ping with strictly increasing values',
      'At most 10⁴ calls will be made to ping',
    ],
    hints: [
      'Use a queue to store request timestamps',
      'For each new request, remove timestamps older than t - 3000',
      'Return the size of the queue',
      'Old requests can be removed from the front',
    ],
    starterCode: `class RecentCounter:
    """
    Count number of requests in last 3000ms.
    """
    
    def __init__(self):
        """Initialize counter."""
        pass
    
    def ping(self, t: int) -> int:
        """
        Add request at time t and return count in [t-3000, t].
        
        Args:
            t: Time in milliseconds
            
        Returns:
            Number of requests in last 3000ms
        """
        pass


# Test code
counter = RecentCounter()
print(counter.ping(1))     # Expected: 1
print(counter.ping(100))   # Expected: 2
print(counter.ping(3001))  # Expected: 3
print(counter.ping(3002))  # Expected: 3
`,
    testCases: [
      {
        input: [
          ['ping', 'ping', 'ping', 'ping'],
          [[1], [100], [3001], [3002]],
        ],
        expected: [1, 2, 3, 3],
      },
    ],
    solution: `from collections import deque

class RecentCounter:
    """Count requests in sliding time window"""
    
    def __init__(self):
        self.requests = deque()  # Store timestamps
    
    def ping(self, t: int) -> int:
        """Add request and count recent ones"""
        # Add new request
        self.requests.append(t)
        
        # Remove requests older than t - 3000
        while self.requests and self.requests[0] < t - 3000:
            self.requests.popleft()
        
        # Return count of requests in window
        return len(self.requests)


# Example walkthrough:
# ping(1):    queue=[1]                    → count=1
# ping(100):  queue=[1,100]                → count=2
# ping(3001): queue=[1,100,3001]           → count=3
# ping(3002): queue=[100,3001,3002]        → count=3
#             (removed 1, it's < 3002-3000=2)

# Time Complexity: O(1) amortized - each element added/removed once
# Space Complexity: O(W) where W is window size (at most 3000ms of requests)`,
    timeComplexity: 'O(1) amortized per ping',
    spaceComplexity: 'O(W) where W is window size',
    followUp: [
      'How would you modify this for a variable time window?',
      'What if requests could come out of order?',
      'Can you solve this without using a queue?',
    ],
  },
  {
    id: 'queue-circular-queue',
    title: 'Design Circular Queue',
    difficulty: 'Medium',
    topic: 'Queue',
    description: `Design your implementation of the circular queue. The circular queue is a linear data structure where the last position connects back to the first position to make a circle.

A circular queue has the advantage of reusing empty space. When the queue is full and we want to add a new element, we can check if there's any empty space at the beginning.

Implement the \`MyCircularQueue\` class:
- \`MyCircularQueue(k)\` - Initializes the queue with size k
- \`enQueue(value)\` - Inserts an element into the circular queue. Return true if successful
- \`deQueue()\` - Deletes an element from the circular queue. Return true if successful
- \`Front()\` - Gets the front item. Return -1 if empty
- \`Rear()\` - Gets the last item. Return -1 if empty
- \`isEmpty()\` - Checks if queue is empty
- \`isFull()\` - Checks if queue is full`,
    examples: [
      {
        input:
          'MyCircularQueue(3), enQueue(1), enQueue(2), enQueue(3), enQueue(4), Rear(), isFull(), deQueue(), enQueue(4), Rear()',
        output: 'true, true, true, false, 3, true, true, true, 4',
      },
    ],
    constraints: ['1 <= k <= 1000', '0 <= value <= 1000', 'At most 3000 calls'],
    hints: [
      'Use an array of size k to store elements',
      'Track front and rear pointers',
      'Use modulo arithmetic to wrap around: (index + 1) % k',
      'Track size or use a flag to distinguish empty from full',
    ],
    starterCode: `class MyCircularQueue:
    """
    Fixed-size circular queue implementation.
    """
    
    def __init__(self, k: int):
        """Initialize with size k."""
        pass
    
    def enQueue(self, value: int) -> bool:
        """Insert element. Return true if successful."""
        pass
    
    def deQueue(self) -> bool:
        """Delete element. Return true if successful."""
        pass
    
    def Front(self) -> int:
        """Get front item. Return -1 if empty."""
        pass
    
    def Rear(self) -> int:
        """Get last item. Return -1 if empty."""
        pass
    
    def isEmpty(self) -> bool:
        """Check if queue is empty."""
        pass
    
    def isFull(self) -> bool:
        """Check if queue is full."""
        pass


# Test code
queue = MyCircularQueue(3)
print(queue.enQueue(1))  # True
print(queue.enQueue(2))  # True
print(queue.enQueue(3))  # True
print(queue.enQueue(4))  # False (full)
print(queue.Rear())      # 3
print(queue.isFull())    # True
print(queue.deQueue())   # True
print(queue.enQueue(4))  # True (now has space)
print(queue.Rear())      # 4
`,
    testCases: [
      {
        input: [
          [
            'MyCircularQueue',
            'enQueue',
            'enQueue',
            'enQueue',
            'enQueue',
            'Rear',
            'isFull',
            'deQueue',
            'enQueue',
            'Rear',
          ],
          [[3], [1], [2], [3], [4], [], [], [], [4], []],
        ],
        expected: [null, true, true, true, false, 3, true, true, true, 4],
      },
    ],
    solution: `class MyCircularQueue:
    """Circular queue using fixed-size array"""
    
    def __init__(self, k: int):
        self.queue = [0] * k  # Fixed-size array
        self.capacity = k
        self.front = 0        # Front pointer
        self.size = 0         # Current size
    
    def enQueue(self, value: int) -> bool:
        """Add to rear"""
        if self.isFull():
            return False
        
        # Calculate rear position
        rear = (self.front + self.size) % self.capacity
        self.queue[rear] = value
        self.size += 1
        return True
    
    def deQueue(self) -> bool:
        """Remove from front"""
        if self.isEmpty():
            return False
        
        self.front = (self.front + 1) % self.capacity  # Move front
        self.size -= 1
        return True
    
    def Front(self) -> int:
        """Get front element"""
        return -1 if self.isEmpty() else self.queue[self.front]
    
    def Rear(self) -> int:
        """Get rear element"""
        if self.isEmpty():
            return -1
        rear = (self.front + self.size - 1) % self.capacity
        return self.queue[rear]
    
    def isEmpty(self) -> bool:
        return self.size == 0
    
    def isFull(self) -> bool:
        return self.size == self.capacity


# Key insight: Use modulo to wrap around
# front: points to first element
# rear: (front + size) % capacity
# Circular wrapping: (index + 1) % capacity

# Time Complexity: O(1) for all operations
# Space Complexity: O(k) for the array`,
    timeComplexity: 'O(1) for all operations',
    spaceComplexity: 'O(k) for capacity k',
    followUp: [
      'How would you implement with a linked list?',
      'What if you needed to support dynamic resizing?',
      'Can you optimize space usage?',
    ],
  },
  {
    id: 'queue-moving-average',
    title: 'Moving Average from Data Stream',
    difficulty: 'Easy',
    topic: 'Queue',
    description: `Given a stream of integers and a window size, calculate the moving average of all integers in the sliding window.

Implement the \`MovingAverage\` class:
- \`MovingAverage(int size)\` - Initializes the object with the size of the window
- \`double next(int val)\` - Returns the moving average of the last \`size\` values of the stream

**Moving Average Example:**
For stream [1, 10, 3, 5] with window size 3:
- After 1: avg = 1/1 = 1.0
- After 10: avg = (1+10)/2 = 5.5
- After 3: avg = (1+10+3)/3 = 4.67
- After 5: avg = (10+3+5)/3 = 6.0 (1 is out of window)`,
    examples: [
      {
        input: 'MovingAverage(3), next(1), next(10), next(3), next(5)',
        output: '1.0, 5.5, 4.67, 6.0',
      },
    ],
    constraints: [
      '1 <= size <= 1000',
      '-10⁵ <= val <= 10⁵',
      'At most 10⁴ calls',
    ],
    hints: [
      'Use a queue to store the last "size" elements',
      'Keep a running sum to avoid recalculating',
      'When queue is full, remove oldest element and subtract from sum',
      'Add new element, update sum, calculate average',
    ],
    starterCode: `class MovingAverage:
    """
    Calculate moving average of fixed window size.
    """
    
    def __init__(self, size: int):
        """Initialize with window size."""
        pass
    
    def next(self, val: int) -> float:
        """
        Add value and return moving average.
        
        Args:
            val: New value in stream
            
        Returns:
            Moving average of last size values
        """
        pass


# Test code
ma = MovingAverage(3)
print(ma.next(1))   # Expected: 1.0
print(ma.next(10))  # Expected: 5.5
print(ma.next(3))   # Expected: 4.666...
print(ma.next(5))   # Expected: 6.0
`,
    testCases: [
      {
        input: [[3], [1], [10], [3], [5]],
        expected: [null, 1.0, 5.5, 4.666666666666667, 6.0],
      },
    ],
    solution: `from collections import deque

class MovingAverage:
    """Moving average using queue and running sum"""
    
    def __init__(self, size: int):
        self.size = size
        self.queue = deque()
        self.window_sum = 0
    
    def next(self, val: int) -> float:
        """Add value and compute average"""
        # Add new value
        self.queue.append(val)
        self.window_sum += val
        
        # Remove oldest if window exceeded
        if len(self.queue) > self.size:
            removed = self.queue.popleft()
            self.window_sum -= removed
        
        # Calculate average
        return self.window_sum / len(self.queue)


# Walkthrough for MovingAverage(3):
# next(1):  queue=[1], sum=1      → avg=1.0
# next(10): queue=[1,10], sum=11  → avg=5.5
# next(3):  queue=[1,10,3], sum=14 → avg=4.67
# next(5):  queue=[10,3,5], sum=18 → avg=6.0
#           (removed 1, sum=14-1+5=18)

# Time Complexity: O(1) per next operation
# Space Complexity: O(size) for the queue`,
    timeComplexity: 'O(1) per operation',
    spaceComplexity: 'O(size)',
    followUp: [
      'How would you handle weighted moving average?',
      'Can you support variable window sizes?',
      'What about exponential moving average?',
    ],
  },
  {
    id: 'queue-perfect-squares',
    title: 'Perfect Squares (BFS)',
    difficulty: 'Medium',
    topic: 'Queue',
    description: `Given an integer n, return the least number of perfect square numbers that sum to n.

A **perfect square** is an integer that is the square of an integer; in other words, it is the product of some integer with itself. For example, 1, 4, 9, and 16 are perfect squares while 3 and 11 are not.

**Examples:**
- n = 12 → 3 (12 = 4 + 4 + 4)
- n = 13 → 2 (13 = 4 + 9)

Use BFS with a queue to find the shortest path (minimum number of squares).`,
    examples: [
      { input: 'n = 12', output: '3 (4+4+4)' },
      { input: 'n = 13', output: '2 (4+9)' },
    ],
    constraints: ['1 <= n <= 10⁴'],
    hints: [
      'Think of this as a graph problem: find shortest path from n to 0',
      'Each node n can go to n-1², n-2², n-3², ... where k² <= n',
      'Use BFS to find minimum steps',
      'Use a queue and track visited nodes',
      'The level where you reach 0 is your answer',
    ],
    starterCode: `def num_squares(n):
    """
    Find minimum number of perfect squares that sum to n using BFS.
    
    Args:
        n: Target sum
        
    Returns:
        Minimum count of perfect squares
        
    Examples:
        >>> num_squares(12)
        3
        >>> num_squares(13)
        2
    """
    pass


# Test cases
print(num_squares(12))  # Expected: 3
print(num_squares(13))  # Expected: 2
`,
    testCases: [
      { input: [12], expected: 3 },
      { input: [13], expected: 2 },
      { input: [1], expected: 1 },
      { input: [2], expected: 2 },
      { input: [4], expected: 1 },
    ],
    solution: `from collections import deque
import math

def num_squares(n):
    """BFS to find minimum perfect squares"""
    if n == 0:
        return 0
    
    # BFS
    queue = deque([(n, 0)])  # (remaining, steps)
    visited = {n}
    
    while queue:
        remaining, steps = queue.popleft()
        
        # Try all perfect squares <= remaining
        for i in range(1, int(math.sqrt(remaining)) + 1):
            square = i * i
            next_remaining = remaining - square
            
            # Found answer
            if next_remaining == 0:
                return steps + 1
            
            # Explore further
            if next_remaining not in visited:
                visited.add(next_remaining)
                queue.append((next_remaining, steps + 1))
    
    return -1  # Should never reach here


# BFS finds shortest path:
# n=12: Level 0: [12]
#       Level 1: [11,8,3] (subtract 1,4,9)
#       Level 2: [10,7,2,4,0] (subtract from Level 1)
#       Found 0 at level 2+1=3 steps
# Answer: We needed 3 perfect squares (4+4+4)

# Alternative: Dynamic Programming
def num_squares_dp(n):
    """DP solution - also O(n√n)"""
    dp = [float('inf')] * (n + 1)
    dp[0] = 0
    
    for i in range(1, n + 1):
        j = 1
        while j * j <= i:
            dp[i] = min(dp[i], dp[i - j*j] + 1)
            j += 1
    
    return dp[n]

# Time Complexity: O(n√n) - for each of n nodes, check √n squares
# Space Complexity: O(n) - queue and visited set`,
    timeComplexity: 'O(n√n)',
    spaceComplexity: 'O(n)',
    followUp: [
      'Can you solve this with dynamic programming?',
      'How would you optimize for very large n?',
      'What about finding all possible combinations?',
    ],
  },
  {
    id: 'queue-walls-gates',
    title: 'Walls and Gates (Multi-Source BFS)',
    difficulty: 'Medium',
    topic: 'Queue',
    description: `You are given an m x n grid initialized with these three possible values:
- **-1** - A wall or an obstacle
- **0** - A gate
- **INF** - An empty room (represented as 2³¹ - 1 = 2147483647)

Fill each empty room with the distance to its nearest gate. If it is impossible to reach a gate, it should be filled with INF.

Use multi-source BFS: start from all gates simultaneously and propagate distances outward.`,
    examples: [
      {
        input:
          '[[INF,-1,0,INF],[INF,INF,INF,-1],[INF,-1,INF,-1],[0,-1,INF,INF]]',
        output: '[[3,-1,0,1],[2,2,1,-1],[1,-1,2,-1],[0,-1,3,4]]',
      },
    ],
    constraints: [
      'm == rooms.length',
      'n == rooms[i].length',
      '1 <= m, n <= 250',
    ],
    hints: [
      'Multi-source BFS: add all gates to queue initially',
      'Process level by level, updating distances',
      'Each cell visited once, so O(mn) time',
      'Directions: up, down, left, right',
    ],
    starterCode: `def walls_and_gates(rooms):
    """
    Fill each empty room with distance to nearest gate using BFS.
    
    Args:
        rooms: 2D grid where -1=wall, 0=gate, INF=empty room
        
    Returns:
        None (modifies rooms in-place)
        
    Note: INF = 2147483647
    """
    pass


# Test case
INF = 2147483647
rooms = [
  [INF, -1, 0, INF],
  [INF, INF, INF, -1],
  [INF, -1, INF, -1],
  [0, -1, INF, INF]
]
walls_and_gates(rooms)
# Expected result:
# [[3, -1, 0, 1],
#  [2, 2, 1, -1],
#  [1, -1, 2, -1],
#  [0, -1, 3, 4]]
`,
    testCases: [
      {
        input: [
          [
            [2147483647, -1, 0, 2147483647],
            [2147483647, 2147483647, 2147483647, -1],
            [2147483647, -1, 2147483647, -1],
            [0, -1, 2147483647, 2147483647],
          ],
        ],
        expected: [
          [3, -1, 0, 1],
          [2, 2, 1, -1],
          [1, -1, 2, -1],
          [0, -1, 3, 4],
        ],
      },
    ],
    solution: `from collections import deque

def walls_and_gates(rooms):
    """Multi-source BFS from all gates"""
    if not rooms or not rooms[0]:
        return
    
    m, n = len(rooms), len(rooms[0])
    INF = 2147483647
    queue = deque()
    
    # Add all gates to queue (multi-source BFS)
    for i in range(m):
        for j in range(n):
            if rooms[i][j] == 0:
                queue.append((i, j, 0))  # (row, col, distance)
    
    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # BFS
    while queue:
        row, col, dist = queue.popleft()
        
        # Explore 4 directions
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            # Check bounds and if cell is empty room
            if (0 <= new_row < m and 0 <= new_col < n and 
                rooms[new_row][new_col] == INF):
                
                # Update distance
                rooms[new_row][new_col] = dist + 1
                
                # Add to queue for further exploration
                queue.append((new_row, new_col, dist + 1))


# Why multi-source BFS?
# Start from all gates simultaneously
# First time we reach a room = shortest distance
# No need to compare distances - BFS guarantees shortest

# Example walkthrough:
# Initial gates at (0,2) and (3,0)
# Level 0: process gates
# Level 1: cells 1 step from gates
# Level 2: cells 2 steps from gates
# Continue until all reachable cells visited

# Time Complexity: O(m*n) - each cell visited once
# Space Complexity: O(m*n) - queue can have all cells`,
    timeComplexity: 'O(m*n)',
    spaceComplexity: 'O(m*n)',
    followUp: [
      'How would you modify for diagonal movement?',
      'What if walls could be broken (with cost)?',
      'Can you solve with dynamic programming?',
    ],
  },
  {
    id: 'queue-open-lock',
    title: 'Open the Lock (BFS)',
    difficulty: 'Medium',
    topic: 'Queue',
    description: `You have a lock with 4 circular wheels. Each wheel has 10 slots: '0' to '9'. The wheels can rotate freely and wrap around: '9' -> '0' and '0' -> '9'.

The lock initially starts at '0000'.

You are given a list of \`deadends\` - if the lock displays any of these codes, the wheels stop turning and you cannot open it.

Given a \`target\` representing the value to unlock, return the minimum total number of turns required to open the lock, or -1 if impossible.

This is a shortest-path problem - perfect for BFS!`,
    examples: [
      {
        input:
          'deadends = ["0201","0101","0102","1212","2002"], target = "0202"',
        output: '6',
      },
      {
        input: 'deadends = ["8888"], target = "0009"',
        output: '1',
      },
    ],
    constraints: [
      '1 <= deadends.length <= 500',
      'deadends[i].length == 4',
      'target.length == 4',
      'target will not be in deadends',
      '0000 will not be in deadends',
    ],
    hints: [
      'Model as graph: each combination is a node',
      'Edges connect combinations differing by 1 turn',
      'BFS finds shortest path from "0000" to target',
      'From each state, you can turn any of 4 wheels up or down',
      'Skip deadends and visited states',
    ],
    starterCode: `def open_lock(deadends, target):
    """
    Find minimum turns to open lock using BFS.
    
    Args:
        deadends: List of forbidden combinations
        target: Target combination string
        
    Returns:
        Minimum turns, or -1 if impossible
        
    Examples:
        >>> open_lock(["0201","0101","0102","1212","2002"], "0202")
        6
    """
    pass


# Test cases
print(open_lock(["0201","0101","0102","1212","2002"], "0202"))  # 6
print(open_lock(["8888"], "0009"))  # 1
`,
    testCases: [
      {
        input: [['0201', '0101', '0102', '1212', '2002'], '0202'],
        expected: 6,
      },
      {
        input: [['8888'], '0009'],
        expected: 1,
      },
      {
        input: [['0000'], '8888'],
        expected: -1,
      },
    ],
    solution: `from collections import deque

def open_lock(deadends, target):
    """BFS to find minimum turns"""
    dead_set = set(deadends)
    
    # Check if start or target is a deadend
    if "0000" in dead_set or target in dead_set:
        return -1
    
    if target == "0000":
        return 0
    
    # BFS
    queue = deque([("0000", 0)])  # (combination, turns)
    visited = {"0000"}
    
    while queue:
        combo, turns = queue.popleft()
        
        # Try all possible moves (8 total: 4 wheels × 2 directions)
        for i in range(4):
            digit = int(combo[i])
            
            # Turn wheel up and down
            for direction in [-1, 1]:
                new_digit = (digit + direction) % 10
                new_combo = combo[:i] + str(new_digit) + combo[i+1:]
                
                # Found target
                if new_combo == target:
                    return turns + 1
                
                # Skip if deadend or visited
                if new_combo in dead_set or new_combo in visited:
                    continue
                
                # Add to queue
                visited.add(new_combo)
                queue.append((new_combo, turns + 1))
    
    return -1  # Target unreachable


# Helper function to generate neighbors
def get_neighbors(combo):
    """Generate all 8 possible next combinations"""
    neighbors = []
    for i in range(4):
        digit = int(combo[i])
        # Turn up
        neighbors.append(combo[:i] + str((digit + 1) % 10) + combo[i+1:])
        # Turn down  
        neighbors.append(combo[:i] + str((digit - 1) % 10) + combo[i+1:])
    return neighbors


# Time Complexity: O(10⁴) = O(10000) - at most 10000 possible combinations
# Space Complexity: O(10⁴) - queue and visited set`,
    timeComplexity: 'O(10⁴) = O(10000) combinations',
    spaceComplexity: 'O(10⁴)',
    followUp: [
      'Can you optimize with bidirectional BFS?',
      'What if there were more/fewer wheels?',
      'How would you find all shortest paths?',
    ],
  },
  {
    id: 'queue-sliding-window-max',
    title: 'Sliding Window Maximum',
    difficulty: 'Hard',
    topic: 'Queue',
    description: `You are given an array of integers \`nums\` and an integer \`k\`. There is a sliding window of size \`k\` which is moving from the left to the right. You can only see the \`k\` numbers in the window. Each time the sliding window moves right by one position.

Return the max sliding window (array of maximums for each window position).

**Example:**
Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
Output: [3,3,5,5,6,7]

Window positions:
- [1 3 -1] -3 5 3 6 7 → max = 3
- 1 [3 -1 -3] 5 3 6 7 → max = 3
- 1 3 [-1 -3 5] 3 6 7 → max = 5
- 1 3 -1 [-3 5 3] 6 7 → max = 5
- 1 3 -1 -3 [5 3 6] 7 → max = 6
- 1 3 -1 -3 5 [3 6 7] → max = 7

Use a **monotonic deque** for O(n) solution!`,
    examples: [
      { input: 'nums = [1,3,-1,-3,5,3,6,7], k = 3', output: '[3,3,5,5,6,7]' },
      { input: 'nums = [1], k = 1', output: '[1]' },
    ],
    constraints: [
      '1 <= nums.length <= 10⁵',
      '-10⁴ <= nums[i] <= 10⁴',
      '1 <= k <= nums.length',
    ],
    hints: [
      'Naive O(nk) is too slow for large inputs',
      'Use deque to maintain potential maximum candidates',
      'Keep deque in decreasing order (monotonic decreasing)',
      'Front of deque is always the maximum',
      'Remove indices outside window from front',
      'Remove smaller elements from back before adding new element',
    ],
    starterCode: `def max_sliding_window(nums, k):
    """
    Find maximum in each sliding window using monotonic deque.
    
    Args:
        nums: Array of integers
        k: Window size
        
    Returns:
        Array of maximums for each window
        
    Examples:
        >>> max_sliding_window([1,3,-1,-3,5,3,6,7], 3)
        [3, 3, 5, 5, 6, 7]
    """
    pass


# Test cases
print(max_sliding_window([1,3,-1,-3,5,3,6,7], 3))  # [3,3,5,5,6,7]
print(max_sliding_window([1], 1))  # [1]
`,
    testCases: [
      { input: [[1, 3, -1, -3, 5, 3, 6, 7], 3], expected: [3, 3, 5, 5, 6, 7] },
      { input: [[1], 1], expected: [1] },
      { input: [[1, -1], 1], expected: [1, -1] },
      { input: [[9, 11], 2], expected: [11] },
    ],
    solution: `from collections import deque

def max_sliding_window(nums, k):
    """Monotonic decreasing deque solution"""
    result = []
    dq = deque()  # Store indices, not values
    
    for i in range(len(nums)):
        # Remove indices outside current window
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        
        # Remove smaller elements from back
        # They can never be maximum while nums[i] is in window
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        
        # Add current index
        dq.append(i)
        
        # Add to result once window is full
        if i >= k - 1:
            result.append(nums[dq[0]])  # Front is maximum
    
    return result


# Why it works:
# Deque maintains indices in decreasing order of values
# Front = index of maximum in current window
# When adding new element:
#   - Remove expired indices (outside window)
#   - Remove smaller elements (can't be max anymore)
#   - Add new element
# Deque always has potential maximum candidates

# Example: [1,3,-1,-3,5,3,6,7], k=3
# i=0: dq=[0], nums[0]=1
# i=1: dq=[1], nums[1]=3 (removed 0 as 1<3)
# i=2: dq=[1,2], add max=nums[1]=3
# i=3: dq=[1,2,3], add max=nums[1]=3
# i=4: dq=[4], nums[4]=5 (removed all as 5 is largest)
# ...

# Time Complexity: O(n) - each element added/removed once
# Space Complexity: O(k) - deque size bounded by k`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(k)',
    followUp: [
      'Can you find minimum sliding window instead?',
      'What if you need both max and min?',
      'How would you handle duplicates specially?',
    ],
  },
];
