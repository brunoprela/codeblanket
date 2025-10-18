/**
 * Count of Smaller Numbers After Self
 * Problem ID: count-smaller
 * Order: 6
 */

import { Problem } from '../../../types';

export const count_smallerProblem: Problem = {
  id: 'count-smaller',
  title: 'Count of Smaller Numbers After Self',
  difficulty: 'Hard',
  topic: 'Sorting Algorithms',

  leetcodeUrl:
    'https://leetcode.com/problems/count-of-smaller-numbers-after-self/',
  youtubeUrl: 'https://www.youtube.com/watch?v=2SVLYsq5W8M',
  order: 6,
  description: `Given an integer array \`nums\`, return an integer array \`counts\` where \`counts[i]\` is the number of smaller elements to the right of \`nums[i]\`.

**Challenge:** Can you do better than O(n²)?

**Hint:** Modified merge sort can count inversions during the merge process!`,
  examples: [
    {
      input: 'nums = [5,2,6,1]',
      output: '[2,1,1,0]',
      explanation: `To the right of 5 there are 2 smaller elements (2 and 1).
To the right of 2 there is 1 smaller element (1).
To the right of 6 there is 1 smaller element (1).
To the right of 1 there is 0 smaller element.`,
    },
    {
      input: 'nums = [-1]',
      output: '[0]',
    },
    {
      input: 'nums = [-1,-1]',
      output: '[0,0]',
    },
  ],
  constraints: ['1 <= nums.length <= 10^5', '-10^4 <= nums[i] <= 10^4'],
  hints: [
    'Brute force: for each element, count how many after it are smaller - O(n²)',
    'Key insight: counting inversions is similar to merge sort',
    'During merge, track original indices',
  ],
  starterCode: `from typing import List

def count_smaller(nums: List[int]) -> List[int]:
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[5, 2, 6, 1]],
      expected: [2, 1, 1, 0],
    },
    {
      input: [[-1]],
      expected: [0],
    },
    {
      input: [[-1, -1]],
      expected: [0, 0],
    },
    {
      input: [[1, 2, 3, 4, 5]],
      expected: [0, 0, 0, 0, 0],
    },
    {
      input: [[5, 4, 3, 2, 1]],
      expected: [4, 3, 2, 1, 0],
    },
  ],
  solution: `# Optimal: Modified Merge Sort - O(n log n)
def count_smaller(nums):
    def merge_sort(arr):
        if len(arr) <= 1:
            return arr
        
        mid = len(arr) // 2
        left = merge_sort(arr[:mid])
        right = merge_sort(arr[mid:])
        
        return merge(left, right)
    
    def merge(left, right):
        result = []
        i = j = 0
        
        while i < len(left) and j < len(right):
            if left[i][1] <= right[j][1]:
                # Count how many from right are smaller
                counts[left[i][0]] += j
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        
        # Process remaining left elements
        while i < len(left):
            counts[left[i][0]] += j
            result.append(left[i])
            i += 1
        
        # Process remaining right elements
        while j < len(right):
            result.append(right[j])
            j += 1
        
        return result
    
    # Create array of (index, value) pairs
    indexed_nums = [(i, num) for i, num in enumerate(nums)]
    counts = [0] * len(nums)
    
    merge_sort(indexed_nums)
    
    return counts

# Brute Force: O(n²) - for comparison
def count_smaller_brute(nums):
    counts = []
    for i in range(len(nums)):
        count = 0
        for j in range(i + 1, len(nums)):
            if nums[j] < nums[i]:
                count += 1
        counts.append(count)
    return counts
`,
  timeComplexity:
    'O(n log n) with modified merge sort vs O(n²) brute force - demonstrates advanced sorting applications',
  spaceComplexity: 'O(n) for auxiliary arrays during merge sort',
};
