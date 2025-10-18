/**
 * Number of Recent Calls
 * Problem ID: queue-recent-calls
 * Order: 2
 */

import { Problem } from '../../../types';

export const recent_callsProblem: Problem = {
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
};
