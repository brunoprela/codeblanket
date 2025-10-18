/**
 * Perfect Squares (BFS)
 * Problem ID: queue-perfect-squares
 * Order: 5
 */

import { Problem } from '../../../types';

export const perfect_squaresProblem: Problem = {
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
};
