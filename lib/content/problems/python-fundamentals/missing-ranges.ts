/**
 * Missing Ranges
 * Problem ID: fundamentals-missing-ranges
 * Order: 51
 */

import { Problem } from '../../../types';

export const missing_rangesProblem: Problem = {
  id: 'fundamentals-missing-ranges',
  title: 'Missing Ranges',
  difficulty: 'Easy',
  description: `Find missing ranges in a sorted array given a lower and upper bound.

Return list of missing ranges as strings in format "start->end" or just "num" for single number.

**Example:** nums=[0,1,3,50,75], lower=0, upper=99
Missing: ["2", "4->49", "51->74", "76->99"]

This tests:
- Range detection
- String formatting
- Edge case handling`,
  examples: [
    {
      input: 'nums=[0,1,3,50,75], lower=0, upper=99',
      output: '["2", "4->49", "51->74", "76->99"]',
    },
  ],
  constraints: ['-10^9 <= lower <= upper <= 10^9'],
  hints: [
    'Check gaps between consecutive numbers',
    'Handle start and end boundaries',
    'Single number vs range formatting',
  ],
  starterCode: `def find_missing_ranges(nums, lower, upper):
    """
    Find missing ranges in sorted array.
    
    Args:
        nums: Sorted array
        lower: Lower bound
        upper: Upper bound
        
    Returns:
        List of missing range strings
        
    Examples:
        >>> find_missing_ranges([0,1,3,50,75], 0, 99)
        ["2", "4->49", "51->74", "76->99"]
    """
    pass


# Test
print(find_missing_ranges([0,1,3,50,75], 0, 99))
`,
  testCases: [
    {
      input: [[0, 1, 3, 50, 75], 0, 99],
      expected: ['2', '4->49', '51->74', '76->99'],
    },
    {
      input: [[], 1, 1],
      expected: ['1'],
    },
  ],
  solution: `def find_missing_ranges(nums, lower, upper):
    def format_range(start, end):
        return str(start) if start == end else f"{start}->{end}"
    
    result = []
    prev = lower - 1
    
    for num in nums + [upper + 1]:
        if num > prev + 1:
            result.append(format_range(prev + 1, num - 1))
        prev = num
    
    return result`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  order: 51,
  topic: 'Python Fundamentals',
};
