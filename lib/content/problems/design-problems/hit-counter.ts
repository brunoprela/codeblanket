/**
 * Design Hit Counter
 * Problem ID: hit-counter
 * Order: 5
 */

import { Problem } from '../../../types';

export const hit_counterProblem: Problem = {
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
  spaceComplexity: 'Approach 1: O(N) where N=hits in window. Approach 2: O(1).',
  patterns: ['Design', 'Deque', 'Sliding Window', 'Hash Table'],
  companies: ['Google', 'Dropbox', 'Uber'],
};
