/**
 * Gas Station
 * Problem ID: gas-station
 * Order: 3
 */

import { Problem } from '../../../types';

export const gas_stationProblem: Problem = {
  id: 'gas-station',
  title: 'Gas Station',
  difficulty: 'Hard',
  description: `There are \`n\` gas stations along a circular route, where the amount of gas at the \`ith\` station is \`gas[i]\`.

You have a car with an unlimited gas tank and it costs \`cost[i]\` of gas to travel from the \`ith\` station to its next \`(i + 1)th\` station. You begin the journey with an empty tank at one of the gas stations.

Given two integer arrays \`gas\` and \`cost\`, return the starting gas station's index if you can travel around the circuit once in the clockwise direction, otherwise return \`-1\`. If there exists a solution, it is **guaranteed** to be **unique**.


**Greedy Approach:**1. If total gas < total cost, impossible
2. Track current gas tank. If it goes negative, start from next station
3. Key insight: If we cannot reach station j from station i, we also cannot reach j from any station between i and j

**Key Insight:**
If there is a solution, we can find it in one pass by resetting start whenever tank goes negative.`,
  examples: [
    {
      input: 'gas = [1,2,3,4,5], cost = [3,4,5,1,2]',
      output: '3',
      explanation:
        'Start at station 3. Tank = 0 + 4 - 1 = 3. Tank = 3 + 5 - 2 = 6. Tank = 6 + 1 - 3 = 4. Tank = 4 + 2 - 4 = 2. Tank = 2 + 3 - 5 = 0. Complete the circuit.',
    },
    {
      input: 'gas = [2,3,4], cost = [3,4,3]',
      output: '-1',
      explanation:
        'Cannot start at any station. Total gas = 9, total cost = 10.',
    },
  ],
  constraints: [
    'n == gas.length == cost.length',
    '1 <= n <= 10^5',
    '0 <= gas[i], cost[i] <= 10^4',
  ],
  hints: [
    'If total gas < total cost, impossible to complete circuit',
    'Track current tank as you go',
    'If tank goes negative at position i, cannot start from 0..i',
    'Reset start to i+1 when tank goes negative',
    'Greedy: if solution exists, one pass finds it',
    'Key: If stuck at j starting from i, also stuck starting from i+1..j-1',
  ],
  starterCode: `from typing import List

def can_complete_circuit(gas: List[int], cost: List[int]) -> int:
    """
    Find starting gas station to complete circuit.
    
    Args:
        gas: Gas available at each station
        cost: Cost to travel to next station
        
    Returns:
        Starting station index, or -1 if impossible
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [
        [1, 2, 3, 4, 5],
        [3, 4, 5, 1, 2],
      ],
      expected: 3,
    },
    {
      input: [
        [2, 3, 4],
        [3, 4, 3],
      ],
      expected: -1,
    },
    {
      input: [[5], [4]],
      expected: 0,
    },
  ],
  solution: `from typing import List


def can_complete_circuit(gas: List[int], cost: List[int]) -> int:
    """
    Greedy one-pass solution.
    Time: O(n), Space: O(1)
    """
    # Quick check: total gas must >= total cost
    if sum(gas) < sum(cost):
        return -1
    
    start = 0
    tank = 0
    
    for i in range(len(gas)):
        tank += gas[i] - cost[i]
        
        # Can't reach next station from current start
        if tank < 0:
            # Try starting from next station
            start = i + 1
            tank = 0
    
    return start


# Alternative: With explicit total check
def can_complete_circuit_explicit(gas: List[int], cost: List[int]) -> int:
    """
    More explicit version showing both checks.
    """
    total_tank = 0
    current_tank = 0
    start = 0
    
    for i in range(len(gas)):
        total_tank += gas[i] - cost[i]
        current_tank += gas[i] - cost[i]
        
        if current_tank < 0:
            start = i + 1
            current_tank = 0
    
    return start if total_tank >= 0 else -1


# Alternative: Two-pass (easier to understand)
def can_complete_circuit_two_pass(gas: List[int], cost: List[int]) -> int:
    """
    Two-pass: check total first, then find start.
    """
    n = len(gas)
    
    # Pass 1: Check if solution exists
    if sum(gas) < sum(cost):
        return -1
    
    # Pass 2: Find starting station
    tank = 0
    start = 0
    
    for i in range(n):
        tank += gas[i] - cost[i]
        if tank < 0:
            start = i + 1
            tank = 0
    
    return start`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',

  leetcodeUrl: 'https://leetcode.com/problems/gas-station/',
  youtubeUrl: 'https://www.youtube.com/watch?v=lJwbPZGo05A',
  order: 3,
  topic: 'Greedy',
};
