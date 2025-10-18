/**
 * First Bad Version
 * Problem ID: first-bad-version
 * Order: 5
 */

import { Problem } from '../../../types';

export const first_bad_versionProblem: Problem = {
  id: 'first-bad-version',
  title: 'First Bad Version',
  difficulty: 'Easy',
  topic: 'Binary Search',
  order: 5,
  description: `You are a product manager and currently leading a team to develop a new product. Unfortunately, the latest version of your product fails the quality check. Since each version is developed based on the previous version, all the versions after a bad version are also bad.

Suppose you have \`n\` versions \`[1, 2, ..., n]\` and you want to find out the first bad one, which causes all the following ones to be bad.

You are given an API \`def is_bad_version(version):\` which returns whether \`version\` is bad. Implement a function to find the first bad version. You should minimize the number of calls to the API.`,
  examples: [
    {
      input: 'n = 5, bad = 4',
      output: '4',
      explanation:
        'is_bad_version(3) -> False, is_bad_version(4) -> True, so 4 is the first bad version.',
    },
    {
      input: 'n = 1, bad = 1',
      output: '1',
    },
  ],
  constraints: ['1 <= bad <= n <= 2^31 - 1'],
  hints: [
    'Use binary search to minimize API calls',
    'If version is bad, search left half (including current)',
    'If version is good, search right half',
  ],
  starterCode: `# The is_bad_version API is already defined for you.
def is_bad_version(version: int) -> bool:
    pass

def first_bad_version(n: int) -> int:
    """
    Find the first bad version.
    
    Args:
        n: Total number of versions
        
    Returns:
        The first bad version number
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [5],
      expected: 4,
    },
    {
      input: [1],
      expected: 1,
    },
    {
      input: [10],
      expected: 7,
    },
  ],
  solution: `def is_bad_version(version: int) -> bool:
    # This is a stub - actual implementation provided by API
    pass

def first_bad_version(n: int) -> int:
    """
    Binary search for first bad version.
    Time: O(log n), Space: O(1)
    """
    left, right = 1, n
    
    while left < right:
        mid = (left + right) // 2
        
        if is_bad_version(mid):
            # First bad is at mid or before
            right = mid
        else:
            # First bad is after mid
            left = mid + 1
    
    return left
`,
  timeComplexity: 'O(log n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/first-bad-version/',
  youtubeUrl: 'https://www.youtube.com/watch?v=GJVO2BTdBZw',
};
