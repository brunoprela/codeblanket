/**
 * Guess Number Higher or Lower
 * Problem ID: guess-number-higher-or-lower
 * Order: 10
 */

import { Problem } from '../../../types';

export const guess_number_higher_or_lowerProblem: Problem = {
  id: 'guess-number-higher-or-lower',
  title: 'Guess Number Higher or Lower',
  difficulty: 'Easy',
  topic: 'Binary Search',
  description: `We are playing the Guess Game. The game is as follows:

I pick a number from \`1\` to \`n\`. You have to guess which number I picked.

Every time you guess wrong, I will tell you whether the number I picked is higher or lower than your guess.

You call a pre-defined API \`int guess(int num)\`, which returns three possible results:
- \`-1\`: Your guess is higher than the number I picked (i.e. \`num > pick\`)
- \`1\`: Your guess is lower than the number I picked (i.e. \`num < pick\`)
- \`0\`: your guess is equal to the number I picked (i.e. \`num == pick\`)

Return the number that I picked.`,
  examples: [
    {
      input: 'n = 10, pick = 6',
      output: '6',
    },
    {
      input: 'n = 1, pick = 1',
      output: '1',
    },
  ],
  constraints: ['1 <= n <= 2^31 - 1', '1 <= pick <= n'],
  hints: ['Use binary search', 'Call guess() to narrow down the search space'],
  starterCode: `# The guess API is already defined for you.
# @param num, your guess
# @return -1 if num is higher than the picked number
#          1 if num is lower than the picked number
#          otherwise return 0
# def guess(num: int) -> int:

def guess_number(n: int) -> int:
    """
    Guess the picked number using binary search.
    
    Args:
        n: Upper bound of range [1, n]
        
    Returns:
        The picked number
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [10],
      expected: 6,
    },
    {
      input: [1],
      expected: 1,
    },
    {
      input: [2],
      expected: 1,
    },
  ],
  timeComplexity: 'O(log n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/guess-number-higher-or-lower/',
  youtubeUrl: 'https://www.youtube.com/watch?v=xW4QsTtaCa4',
};
