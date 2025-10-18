/**
 * Open the Lock (BFS)
 * Problem ID: queue-open-lock
 * Order: 7
 */

import { Problem } from '../../../types';

export const open_lockProblem: Problem = {
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
      input: 'deadends = ["0201","0101","0102","1212","2002"], target = "0202"',
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
};
