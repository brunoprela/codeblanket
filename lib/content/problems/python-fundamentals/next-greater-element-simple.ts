/**
 * Next Greater Element (Simple)
 * Problem ID: fundamentals-next-greater-element-simple
 * Order: 99
 */

import { Problem } from '../../../types';

export const next_greater_element_simpleProblem: Problem = {
  id: 'fundamentals-next-greater-element-simple',
  title: 'Next Greater Element (Simple)',
  difficulty: 'Easy',
  description: `For each element in nums1, find the next greater element in nums2.

nums1 is subset of nums2.
Next greater = first greater element to the right in nums2.

**Example:** nums1=[4,1,2], nums2=[1,3,4,2]
→ [-1,3,-1] (4: none, 1→3, 2: none)

This tests:
- Array traversal
- Finding next greater
- Hash map usage`,
  examples: [
    {
      input: 'nums1 = [4,1,2], nums2 = [1,3,4,2]',
      output: '[-1,3,-1]',
    },
  ],
  constraints: ['1 <= len(nums1), len(nums2) <= 1000'],
  hints: [
    'Build map of next greater in nums2',
    'Use stack for efficient next greater',
    'Look up each nums1 element',
  ],
  starterCode: `def next_greater_element(nums1, nums2):
    """
    Find next greater elements.
    
    Args:
        nums1: Query array
        nums2: Search array
        
    Returns:
        Array of next greater elements
        
    Examples:
        >>> next_greater_element([4,1,2], [1,3,4,2])
        [-1, 3, -1]
    """
    pass


# Test
print(next_greater_element([4,1,2], [1,3,4,2]))
`,
  testCases: [
    {
      input: [
        [4, 1, 2],
        [1, 3, 4, 2],
      ],
      expected: [-1, 3, -1],
    },
    {
      input: [
        [2, 4],
        [1, 2, 3, 4],
      ],
      expected: [3, -1],
    },
  ],
  solution: `def next_greater_element(nums1, nums2):
    # Build next greater map for nums2
    next_greater = {}
    stack = []
    
    for num in nums2:
        while stack and stack[-1] < num:
            next_greater[stack.pop()] = num
        stack.append(num)
    
    # Query for nums1
    return [next_greater.get(num, -1) for num in nums1]


# Brute force O(n*m)
def next_greater_element_brute(nums1, nums2):
    result = []
    
    for num in nums1:
        idx = nums2.index(num)
        found = -1
        for i in range(idx + 1, len(nums2)):
            if nums2[i] > num:
                found = nums2[i]
                break
        result.append(found)
    
    return result`,
  timeComplexity: 'O(n + m)',
  spaceComplexity: 'O(n)',
  order: 99,
  topic: 'Python Fundamentals',
};
