/**
 * Moving Average from Data Stream
 * Problem ID: queue-moving-average
 * Order: 4
 */

import { Problem } from '../../../types';

export const moving_averageProblem: Problem = {
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
  constraints: ['1 <= size <= 1000', '-10⁵ <= val <= 10⁵', 'At most 10⁴ calls'],
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
};
