/**
 * Majority Element
 * Problem ID: fundamentals-majority-element
 * Order: 49
 */

import { Problem } from '../../../types';

export const majority_elementProblem: Problem = {
  id: 'fundamentals-majority-element',
  title: 'Majority Element',
  difficulty: 'Easy',
  description: `Find the majority element in an array.

The majority element appears more than ⌊n/2⌋ times.

**Guaranteed:** The majority element always exists in the array.

**Boyer-Moore Algorithm:** Efficient O(1) space solution using voting.

This tests:
- Array traversal
- Counter or voting algorithm
- Majority logic`,
  examples: [
    {
      input: 'nums = [3,2,3]',
      output: '3',
    },
    {
      input: 'nums = [2,2,1,1,1,2,2]',
      output: '2',
    },
  ],
  constraints: ['1 <= len(nums) <= 5*10^4', 'Majority element always exists'],
  hints: [
    'Use Counter for simple solution',
    'Boyer-Moore voting for O(1) space',
    'Candidate changes when count reaches 0',
  ],
  starterCode: `def majority_element(nums):
    """
    Find majority element.
    
    Args:
        nums: Array of integers
        
    Returns:
        The majority element
        
    Examples:
        >>> majority_element([3,2,3])
        3
    """
    pass


# Test
print(majority_element([2,2,1,1,1,2,2]))
`,
  testCases: [
    {
      input: [[3, 2, 3]],
      expected: 3,
    },
    {
      input: [[2, 2, 1, 1, 1, 2, 2]],
      expected: 2,
    },
  ],
  solution: `def majority_element(nums):
    # Boyer-Moore Voting Algorithm
    candidate = None
    count = 0
    
    for num in nums:
        if count == 0:
            candidate = num
        count += 1 if num == candidate else -1
    
    return candidate


# Alternative using Counter
from collections import Counter

def majority_element_counter(nums):
    counts = Counter(nums)
    return counts.most_common(1)[0][0]`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1) with Boyer-Moore',
  order: 49,
  topic: 'Python Fundamentals',
};
