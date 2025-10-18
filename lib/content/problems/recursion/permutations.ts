/**
 * Generate All Permutations
 * Problem ID: recursion-permutations
 * Order: 12
 */

import { Problem } from '../../../types';

export const permutationsProblem: Problem = {
  id: 'recursion-permutations',
  title: 'Generate All Permutations',
  difficulty: 'Medium',
  topic: 'Recursion',
  description: `Generate all possible permutations of a list of distinct integers using recursion.

A permutation is an arrangement of all elements where order matters.

**Example:**
- Input: [1, 2, 3]
- Output: [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]

For n elements, there are n! (factorial) permutations.

This is a classic backtracking problem that demonstrates recursive exploration of all possibilities.`,
  examples: [
    {
      input: 'nums = [1, 2, 3]',
      output: '[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]',
    },
    { input: 'nums = [1, 2]', output: '[[1,2],[2,1]]' },
    { input: 'nums = [1]', output: '[[1]]' },
  ],
  constraints: [
    '1 <= nums.length <= 6',
    'All integers are distinct',
    '-10 <= nums[i] <= 10',
  ],
  hints: [
    'Base case: if current permutation has all elements, add to result',
    'Try adding each unused number to current permutation',
    'Recursively build rest of permutation',
    'Backtrack by removing number after exploring',
    'Use a set or list to track which numbers are used',
  ],
  starterCode: `def permutations(nums):
    """
    Generate all permutations of nums.
    
    Args:
        nums: List of distinct integers
        
    Returns:
        List of all permutations
        
    Examples:
        >>> permutations([1, 2, 3])
        [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
    """
    pass


# Test cases
result = permutations([1, 2, 3])
print(f"Permutations: {result}")
print(f"Count: {len(result)}")  # Should be 3! = 6
`,
  testCases: [
    {
      input: [[1, 2]],
      expected: [
        [1, 2],
        [2, 1],
      ],
    },
    {
      input: [[1]],
      expected: [[1]],
    },
    {
      input: [[1, 2, 3]],
      expected: [
        [1, 2, 3],
        [1, 3, 2],
        [2, 1, 3],
        [2, 3, 1],
        [3, 1, 2],
        [3, 2, 1],
      ],
    },
  ],
  solution: `def permutations(nums):
    """Generate all permutations using backtracking"""
    result = []
    
    def backtrack(current, remaining):
        # Base case: no more numbers to add
        if not remaining:
            result.append(current[:])  # Make a copy!
            return
        
        # Try each remaining number
        for i in range(len(remaining)):
            # Choose: add remaining[i] to current permutation
            current.append(remaining[i])
            
            # Explore: recurse with remaining numbers
            new_remaining = remaining[:i] + remaining[i+1:]
            backtrack(current, new_remaining)
            
            # Unchoose: backtrack
            current.pop()
    
    backtrack([], nums)
    return result


# Alternative approach using sets for tracking:
def permutations_alt(nums):
    """Generate permutations using set for tracking"""
    result = []
    used = set()
    
    def backtrack(current):
        # Base case: permutation is complete
        if len(current) == len(nums):
            result.append(current[:])
            return
        
        # Try each unused number
        for num in nums:
            if num not in used:
                # Choose
                current.append(num)
                used.add(num)
                
                # Explore
                backtrack(current)
                
                # Unchoose
                current.pop()
                used.remove(num)
    
    backtrack([])
    return result


# Time Complexity: O(n! * n) - n! permutations, each takes O(n) to build
# Space Complexity: O(n) - recursion depth`,
  timeComplexity: 'O(n! * n)',
  spaceComplexity: 'O(n)',
  followUp: [
    'How would you handle duplicate elements?',
    'Can you generate permutations in lexicographic order?',
    'What if you only want permutations of length k?',
  ],
};
