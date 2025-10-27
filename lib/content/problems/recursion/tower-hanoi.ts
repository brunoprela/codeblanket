/**
 * Tower of Hanoi
 * Problem ID: recursion-tower-hanoi
 * Order: 11
 */

import { Problem } from '../../../types';

export const tower_hanoiProblem: Problem = {
  id: 'recursion-tower-hanoi',
  title: 'Tower of Hanoi',
  difficulty: 'Medium',
  topic: 'Recursion',
  description: `Solve the Tower of Hanoi puzzle using recursion.

**Problem:**
- 3 rods: source, auxiliary, target
- n disks of different sizes on source rod
- Move all disks to target rod
- Rules: Only move one disk at a time, never place larger disk on smaller disk

**Solution Pattern:**1. Move n-1 disks from source to auxiliary (using target)
2. Move largest disk from source to target
3. Move n-1 disks from auxiliary to target (using source)

Return the sequence of moves as a list of tuples (from_rod, to_rod).

This is a classic recursive problem that elegantly demonstrates the power of recursion!`,
  examples: [
    {
      input: 'n = 2, source = "A", aux = "B", target = "C"',
      output: '[("A","B"), ("A","C"), ("B","C")]',
    },
    {
      input: 'n = 3',
      output: '7 moves total',
    },
  ],
  constraints: ['1 <= n <= 15', 'Number of moves = 2^n - 1'],
  hints: [
    'Base case: n = 1, just move disk from source to target',
    'Recursive case: break into 3 steps',
    'Step 1: Move n-1 disks from source to auxiliary',
    'Step 2: Move disk n from source to target',
    'Step 3: Move n-1 disks from auxiliary to target',
    'Notice how the "helper" rod changes in each recursive call',
  ],
  starterCode: `def tower_of_hanoi(n, source='A', auxiliary='B', target='C'):
    """
    Solve Tower of Hanoi puzzle.
    
    Args:
        n: Number of disks
        source: Source rod name
        auxiliary: Auxiliary (helper) rod name
        target: Target rod name
        
    Returns:
        List of moves as tuples (from_rod, to_rod)
        
    Examples:
        >>> tower_of_hanoi(2)
        [('A', 'B'), ('A', 'C'), ('B', 'C')]
    """
    pass


# Test cases
moves = tower_of_hanoi(2)
print(f"Moves for n=2: {moves}")
print(f"Total moves: {len(moves)}")  # Should be 3 (2^2 - 1)
`,
  testCases: [
    {
      input: [1, 'A', 'B', 'C'],
      expected: [['A', 'C']],
    },
    {
      input: [2, 'A', 'B', 'C'],
      expected: [
        ['A', 'B'],
        ['A', 'C'],
        ['B', 'C'],
      ],
    },
    {
      input: [3, 'A', 'B', 'C'],
      expected: [
        ['A', 'C'],
        ['A', 'B'],
        ['C', 'B'],
        ['A', 'C'],
        ['B', 'A'],
        ['B', 'C'],
        ['A', 'C'],
      ],
    },
  ],
  solution: `def tower_of_hanoi(n, source='A', auxiliary='B', target='C'):
    """
    Solve Tower of Hanoi puzzle recursively.
    
    The key insight: To move n disks from source to target:
    1. Move n-1 disks from source to auxiliary (using target as helper)
    2. Move largest disk from source to target
    3. Move n-1 disks from auxiliary to target (using source as helper)
    """
    # Base case: only one disk
    if n == 1:
        return [(source, target)]
    
    moves = []
    
    # Step 1: Move n-1 disks from source to auxiliary (using target)
    moves.extend(tower_of_hanoi(n - 1, source, target, auxiliary))
    
    # Step 2: Move largest disk from source to target
    moves.append((source, target))
    
    # Step 3: Move n-1 disks from auxiliary to target (using source)
    moves.extend(tower_of_hanoi(n - 1, auxiliary, source, target))
    
    return moves


# Verification function
def verify_tower_of_hanoi(moves, n):
    """Verify solution is correct"""
    rods = {'A': list(range(n, 0, -1)), 'B': [], 'C': []}
    
    for from_rod, to_rod in moves:
        if not rods[from_rod]:
            return False
        disk = rods[from_rod].pop()
        if rods[to_rod] and rods[to_rod][-1] < disk:
            return False  # Larger disk on smaller
        rods[to_rod].append(disk)
    
    # All disks should be on rod C
    return rods['C'] == list(range(n, 0, -1))


# Time Complexity: O(2^n) - exponential (exactly 2^n - 1 moves)
# Space Complexity: O(n) - call stack depth`,
  timeComplexity: 'O(2^n)',
  spaceComplexity: 'O(n)',
  followUp: [
    'Why does this problem require 2^n - 1 moves?',
    'Can it be solved iteratively?',
    'What if you have 4 rods instead of 3?',
  ],
};
