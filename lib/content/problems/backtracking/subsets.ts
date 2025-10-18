/**
 * Subsets
 * Problem ID: subsets
 * Order: 1
 */

import { Problem } from '../../../types';

export const subsetsProblem: Problem = {
  id: 'subsets',
  title: 'Subsets',
  difficulty: 'Easy',
  description: `Given an integer array \`nums\` of **unique** elements, return **all possible subsets** (the power set).

The solution set **must not** contain duplicate subsets. Return the solution in **any order**.


**Approach:**
Use backtracking to build subsets incrementally. At each element, we have two choices: include it or exclude it. Use a \`start\` index to avoid generating duplicate subsets.

**Example Decision Tree for [1,2,3]:**
\`\`\`
                    []
          /                   \\
        [1]                    []
       /   \\                 /    \\
    [1,2]  [1]            [2]      []
    /  \\   /  \\          /  \\     /  \\
[1,2,3][1,2][1,3][1] [2,3][2]  [3] []
\`\`\``,
  examples: [
    {
      input: 'nums = [1,2,3]',
      output: '[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]',
      explanation: 'All possible subsets.',
    },
    {
      input: 'nums = [0]',
      output: '[[],[0]]',
    },
  ],
  constraints: [
    '1 <= nums.length <= 10',
    '-10 <= nums[i] <= 10',
    'All the numbers of nums are unique',
  ],
  hints: [
    'Use backtracking to explore all possibilities',
    'At each position, you can either include or exclude the element',
    'Use a start index to avoid generating duplicates ([1,2] vs [2,1])',
    'Add the current subset to results at every recursive call',
    'Time complexity: O(2^N * N) where N is array length',
  ],
  starterCode: `from typing import List

def subsets(nums: List[int]) -> List[List[int]]:
    """
    Generate all subsets (power set).
    
    Args:
        nums: Array of unique integers
        
    Returns:
        List of all possible subsets
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[1, 2, 3]],
      expected: [[], [1], [2], [1, 2], [3], [1, 3], [2, 3], [1, 2, 3]],
    },
    {
      input: [[0]],
      expected: [[], [0]],
    },
    {
      input: [[1, 2]],
      expected: [[], [1], [2], [1, 2]],
    },
  ],
  solution: `from typing import List


def subsets(nums: List[int]) -> List[List[int]]:
    """
    Backtracking approach.
    Time: O(2^N * N), Space: O(N) recursion depth
    """
    result = []
    
    def backtrack(start, path):
        # Add current subset (at every level)
        result.append(path[:])
        
        # Try including each element from start onward
        for i in range(start, len(nums)):
            path.append(nums[i])      # Include nums[i]
            backtrack(i + 1, path)    # Explore with nums[i]
            path.pop()                # Backtrack (exclude nums[i])
    
    backtrack(0, [])
    return result


# Alternative: Iterative approach
def subsets_iterative(nums: List[int]) -> List[List[int]]:
    """
    Iterative approach: build subsets by adding each element.
    Time: O(2^N * N), Space: O(1) if we don't count output
    """
    result = [[]]
    
    for num in nums:
        # Add num to all existing subsets
        result += [subset + [num] for subset in result]
    
    return result


# Alternative: Bit manipulation
def subsets_bitmask(nums: List[int]) -> List[List[int]]:
    """
    Bit manipulation approach: each subset corresponds to a bitmask.
    Time: O(2^N * N), Space: O(1) if we don't count output
    """
    n = len(nums)
    result = []
    
    # Iterate through all possible bitmasks (0 to 2^n - 1)
    for mask in range(1 << n):  # 2^n subsets
        subset = []
        for i in range(n):
            # Check if i-th bit is set
            if mask & (1 << i):
                subset.append(nums[i])
        result.append(subset)
    
    return result`,
  timeComplexity: 'O(2^N * N)',
  spaceComplexity: 'O(N) for recursion depth',
  order: 1,
  topic: 'Backtracking',
  leetcodeUrl: 'https://leetcode.com/problems/subsets/',
  youtubeUrl: 'https://www.youtube.com/watch?v=REOH22Xwdkk',
};
