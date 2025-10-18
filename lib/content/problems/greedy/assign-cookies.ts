/**
 * Assign Cookies
 * Problem ID: assign-cookies
 * Order: 4
 */

import { Problem } from '../../../types';

export const assign_cookiesProblem: Problem = {
  id: 'assign-cookies',
  title: 'Assign Cookies',
  difficulty: 'Easy',
  topic: 'Greedy',
  description: `Assume you are an awesome parent and want to give your children some cookies. But, you should give each child at most one cookie.

Each child \`i\` has a greed factor \`g[i]\`, which is the minimum size of a cookie that the child will be content with; and each cookie \`j\` has a size \`s[j]\`. If \`s[j] >= g[i]\`, we can assign the cookie \`j\` to the child \`i\`, and the child \`i\` will be content. Your goal is to maximize the number of your content children and output the maximum number.`,
  examples: [
    {
      input: 'g = [1,2,3], s = [1,1]',
      output: '1',
    },
    {
      input: 'g = [1,2], s = [1,2,3]',
      output: '2',
    },
  ],
  constraints: [
    '1 <= g.length <= 3 * 10^4',
    '0 <= s.length <= 3 * 10^4',
    '1 <= g[i], s[j] <= 2^31 - 1',
  ],
  hints: [
    'Sort both arrays',
    'Try to satisfy smallest greed with smallest cookie',
  ],
  starterCode: `from typing import List

def find_content_children(g: List[int], s: List[int]) -> int:
    """
    Find maximum number of content children.
    
    Args:
        g: Array of greed factors
        s: Array of cookie sizes
        
    Returns:
        Maximum content children
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [
        [1, 2, 3],
        [1, 1],
      ],
      expected: 1,
    },
    {
      input: [
        [1, 2],
        [1, 2, 3],
      ],
      expected: 2,
    },
  ],
  timeComplexity: 'O(n log n + m log m)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/assign-cookies/',
  youtubeUrl: 'https://www.youtube.com/watch?v=DIX2p7vb9co',
};
