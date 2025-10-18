/**
 * Design Circular Queue
 * Problem ID: queue-circular-queue
 * Order: 3
 */

import { Problem } from '../../../types';

export const circular_queueProblem: Problem = {
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
};
