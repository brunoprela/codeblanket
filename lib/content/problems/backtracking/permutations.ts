/**
 * Permutations
 * Problem ID: permutations
 * Order: 2
 */

import { Problem } from '../../../types';

export const permutationsProblem: Problem = {
  id: 'permutations',
  title: 'Permutations',
  difficulty: 'Medium',
  description: `Given an array \`nums\` of **distinct** integers, return **all possible permutations**. You can return the answer in **any order**.


**Approach:**
Use backtracking. At each step, try adding each unused number to the current permutation. When we've used all numbers, we have a complete permutation.

**Key Difference from Subsets:**
- Subsets: partial selections (can stop at any size)
- Permutations: use ALL elements, order matters

**Example for [1,2,3]:**
\`\`\`
Start with []
Try 1: [1]
  Try 2: [1,2]
    Try 3: [1,2,3] ← complete permutation
  Try 3: [1,3]
    Try 2: [1,3,2] ← complete permutation
Try 2: [2]
  Try 1: [2,1]
    Try 3: [2,1,3]
  ...
\`\`\``,
  examples: [
    {
      input: 'nums = [1,2,3]',
      output: '[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]',
    },
    {
      input: 'nums = [0,1]',
      output: '[[0,1],[1,0]]',
    },
    {
      input: 'nums = [1]',
      output: '[[1]]',
    },
  ],
  constraints: [
    '1 <= nums.length <= 6',
    '-10 <= nums[i] <= 10',
    'All the integers of nums are unique',
  ],
  hints: [
    'Use backtracking to build permutations',
    'Track which elements have been used (visited set or check if in current path)',
    'When current path length equals input length, add to results',
    'Try each unused element at each position',
    'Time: O(N! * N) - N! permutations, O(N) to copy each',
  ],
  starterCode: `from typing import List

def permute(nums: List[int]) -> List[List[int]]:
    """
    Generate all permutations.
    
    Args:
        nums: Array of distinct integers
        
    Returns:
        List of all permutations
    """
    # Write your code here
    pass
`,
  testCases: [
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
    {
      input: [[0, 1]],
      expected: [
        [0, 1],
        [1, 0],
      ],
    },
    {
      input: [[1]],
      expected: [[1]],
    },
  ],
  solution: `from typing import List


def permute(nums: List[int]) -> List[List[int]]:
    """
    Backtracking with 'in path' check.
    Time: O(N! * N), Space: O(N) recursion
    """
    result = []
    
    def backtrack(path):
        # Base case: used all numbers
        if len(path) == len(nums):
            result.append(path[:])
            return
        
        # Try each number
        for num in nums:
            if num in path:  # Already used
                continue
            
            path.append(num)      # Choose
            backtrack(path)       # Explore
            path.pop()            # Unchoose (backtrack)
    
    backtrack([])
    return result


# Alternative: Using visited set (more efficient)
def permute_visited(nums: List[int]) -> List[List[int]]:
    """
    Backtracking with visited set.
    More efficient than 'in path' check.
    """
    result = []
    visited = set()
    
    def backtrack(path):
        if len(path) == len(nums):
            result.append(path[:])
            return
        
        for num in nums:
            if num not in visited:
                visited.add(num)
                path.append(num)
                
                backtrack(path)
                
                path.pop()
                visited.remove(num)
    
    backtrack([])
    return result


# Alternative: Swap-based approach (in-place)
def permute_swap(nums: List[int]) -> List[List[int]]:
    """
    Backtracking with swapping.
    Generates permutations in-place.
    """
    result = []
    
    def backtrack(start):
        if start == len(nums):
            result.append(nums[:])
            return
        
        for i in range(start, len(nums)):
            # Swap
            nums[start], nums[i] = nums[i], nums[start]
            
            # Recurse
            backtrack(start + 1)
            
            # Swap back (backtrack)
            nums[start], nums[i] = nums[i], nums[start]
    
    backtrack(0)
    return result


# Alternative: Using remaining list
def permute_remaining(nums: List[int]) -> List[List[int]]:
    """
    Backtracking by passing remaining elements.
    """
    result = []
    
    def backtrack(path, remaining):
        if not remaining:
            result.append(path)
            return
        
        for i in range(len(remaining)):
            backtrack(
                path + [remaining[i]],
                remaining[:i] + remaining[i+1:]
            )
    
    backtrack([], nums)
    return result`,
  timeComplexity: 'O(N! * N)',
  spaceComplexity: 'O(N) for recursion depth',
  order: 2,
  topic: 'Backtracking',
  leetcodeUrl: 'https://leetcode.com/problems/permutations/',
  youtubeUrl: 'https://www.youtube.com/watch?v=s7AvT7cGdSo',
};
