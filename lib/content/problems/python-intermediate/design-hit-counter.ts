/**
 * Design Hit Counter
 * Problem ID: design-hit-counter
 * Order: 27
 */

import { Problem } from '../../../types';

export const design_hit_counterProblem: Problem = {
  id: 'design-hit-counter',
  title: 'Design Hit Counter',
  difficulty: 'Medium',
  category: 'python-intermediate',
  description: `Design a hit counter that counts the number of hits received in the past 5 minutes (300 seconds).

Implement \`HitCounter\` class using \`deque\`:
- \`HitCounter()\`: Initialize
- \`void hit(int timestamp)\`: Record a hit at given timestamp
- \`int getHits(int timestamp)\`: Return number of hits in past 5 minutes from timestamp

**Example:**
\`\`\`python
counter = HitCounter()
counter.hit(1)        # hit at timestamp 1
counter.hit(2)        # hit at timestamp 2
counter.hit(3)        # hit at timestamp 3
counter.getHits(4)    # returns 3 (hits at 1,2,3 are within 5 mins)
counter.hit(300)      # hit at timestamp 300
counter.getHits(300)  # returns 4 (hits at 1,2,3,300)
counter.getHits(301)  # returns 3 (hit at 1 is outside 5 min window)
\`\`\`

**Follow-up:** What if hit rate is very high? How would you optimize?`,
  starterCode: `from collections import deque

class HitCounter:
    def __init__(self):
        """Initialize hit counter."""
        pass
    
    def hit(self, timestamp: int) -> None:
        """Record a hit at timestamp."""
        pass
    
    def getHits(self, timestamp: int) -> int:
        """Get hits in past 300 seconds."""
        pass`,
  testCases: [
    {
      input: [
        ['HitCounter'],
        ['hit', 1],
        ['hit', 2],
        ['hit', 3],
        ['getHits', 4],
      ],
      expected: [null, null, null, null, 3],
    },
    {
      input: [
        ['HitCounter'],
        ['hit', 1],
        ['hit', 2],
        ['hit', 3],
        ['hit', 300],
        ['getHits', 300],
      ],
      expected: [null, null, null, null, null, 4],
    },
    {
      input: [
        ['HitCounter'],
        ['hit', 1],
        ['hit', 2],
        ['hit', 3],
        ['hit', 300],
        ['getHits', 301],
      ],
      expected: [null, null, null, null, null, 3],
    },
  ],
  hints: [
    'Use deque to store timestamps',
    'Remove old timestamps when checking hits',
    'Timestamps older than 300 seconds are outside window',
  ],
  solution: `from collections import deque

class HitCounter:
    def __init__(self):
        """Initialize hit counter."""
        self.hits = deque()
    
    def hit(self, timestamp: int) -> None:
        """Record a hit at timestamp."""
        self.hits.append(timestamp)
    
    def getHits(self, timestamp: int) -> int:
        """Get hits in past 300 seconds."""
        # Remove hits outside 5-minute window
        while self.hits and self.hits[0] <= timestamp - 300:
            self.hits.popleft()
        
        return len(self.hits)


# Optimized version for high hit rate (using buckets)
class HitCounterOptimized:
    """
    For very high hit rates, store (timestamp, count) pairs
    instead of individual timestamps.
    """
    def __init__(self):
        self.hits = deque()  # Store (timestamp, count) tuples
    
    def hit(self, timestamp: int) -> None:
        if self.hits and self.hits[-1][0] == timestamp:
            # Increment count for current timestamp
            self.hits[-1] = (timestamp, self.hits[-1][1] + 1)
        else:
            # New timestamp
            self.hits.append((timestamp, 1))
    
    def getHits(self, timestamp: int) -> int:
        # Remove old timestamps
        while self.hits and self.hits[0][0] <= timestamp - 300:
            self.hits.popleft()
        
        # Sum all counts
        return sum(count for ts, count in self.hits)


# Test
counter = HitCounter()
counter.hit(1)
counter.hit(2)
counter.hit(3)
print(counter.getHits(4))    # 3
counter.hit(300)
print(counter.getHits(300))  # 4
print(counter.getHits(301))  # 3`,
  timeComplexity: 'O(1) for hit, O(n) for getHits where n is hits in window',
  spaceComplexity: 'O(n) where n is hits in window',
  order: 27,
  topic: 'Python Intermediate',
};
