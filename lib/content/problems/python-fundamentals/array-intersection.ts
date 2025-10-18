/**
 * Intersection of Two Arrays
 * Problem ID: fundamentals-array-intersection
 * Order: 43
 */

import { Problem } from '../../../types';

export const array_intersectionProblem: Problem = {
  id: 'fundamentals-array-intersection',
  title: 'Intersection of Two Arrays',
  difficulty: 'Easy',
  description: `Find the intersection of two arrays.

Return elements that appear in both arrays.
- Each element should appear as many times as it shows in both arrays
- Result can be in any order

**Example:** [1,2,2,1] ∩ [2,2] = [2,2]

This tests:
- Set operations
- Counter usage
- Array manipulation`,
  examples: [
    {
      input: 'nums1 = [1,2,2,1], nums2 = [2,2]',
      output: '[2,2]',
    },
    {
      input: 'nums1 = [4,9,5], nums2 = [9,4,9,8,4]',
      output: '[4,9] or [9,4]',
    },
  ],
  constraints: ['1 <= len(nums1), len(nums2) <= 1000'],
  hints: [
    'Use Counter or dictionary',
    'Count occurrences in both arrays',
    'Take minimum count for each element',
  ],
  starterCode: `def array_intersection(nums1, nums2):
    """
    Find intersection of two arrays.
    
    Args:
        nums1: First array
        nums2: Second array
        
    Returns:
        Array of intersecting elements
        
    Examples:
        >>> array_intersection([1,2,2,1], [2,2])
        [2, 2]
    """
    pass


# Test
print(array_intersection([4,9,5], [9,4,9,8,4]))
`,
  testCases: [
    {
      input: [
        [1, 2, 2, 1],
        [2, 2],
      ],
      expected: [2, 2],
    },
    {
      input: [
        [4, 9, 5],
        [9, 4, 9, 8, 4],
      ],
      expected: [9, 4],
    },
  ],
  solution: `def array_intersection(nums1, nums2):
    from collections import Counter
    
    count1 = Counter(nums1)
    count2 = Counter(nums2)
    
    result = []
    for num in count1:
        if num in count2:
            result.extend([num] * min(count1[num], count2[num]))
    
    return result


# Alternative using set and list
def array_intersection_simple(nums1, nums2):
    result = []
    nums2_copy = nums2.copy()
    
    for num in nums1:
        if num in nums2_copy:
            result.append(num)
            nums2_copy.remove(num)
    
    return result`,
  timeComplexity: 'O(n + m)',
  spaceComplexity: 'O(min(n, m))',
  order: 43,
  topic: 'Python Fundamentals',
};
