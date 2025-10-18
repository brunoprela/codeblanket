/**
 * Relative Ranks
 * Problem ID: relative-ranks
 * Order: 3
 */

import { Problem } from '../../../types';

export const relative_ranksProblem: Problem = {
  id: 'relative-ranks',
  title: 'Relative Ranks',
  difficulty: 'Easy',
  topic: 'Heap / Priority Queue',
  description: `You are given an integer array \`score\` of size \`n\`, where \`score[i]\` is the score of the \`i-th\` athlete in a competition. All the scores are guaranteed to be **unique**.

The athletes are **placed** based on their scores, where the \`1st\` place athlete has the highest score, the \`2nd\` place athlete has the \`2nd\` highest score, and so on. The placement of each athlete determines their rank:

- The \`1st\` place athlete's rank is \`"Gold Medal"\`.
- The \`2nd\` place athlete's rank is \`"Silver Medal"\`.
- The \`3rd\` place athlete's rank is \`"Bronze Medal"\`.
- For the \`4th\` place to the \`n-th\` place athlete, their rank is their placement number (i.e., the \`x-th\` place athlete's rank is \`"x"\`).

Return an array \`answer\` of size \`n\` where \`answer[i]\` is the **rank** of the \`i-th\` athlete.`,
  examples: [
    {
      input: 'score = [5,4,3,2,1]',
      output: '["Gold Medal","Silver Medal","Bronze Medal","4","5"]',
    },
    {
      input: 'score = [10,3,8,9,4]',
      output: '["Gold Medal","5","Bronze Medal","Silver Medal","4"]',
    },
  ],
  constraints: [
    'n == score.length',
    '1 <= n <= 10^4',
    '0 <= score[i] <= 10^6',
    'All the values in score are unique',
  ],
  hints: [
    'Create pairs of (score, index)',
    'Sort by score descending',
    'Assign ranks based on sorted order',
  ],
  starterCode: `from typing import List

def find_relative_ranks(score: List[int]) -> List[str]:
    """
    Assign ranks to athletes based on scores.
    
    Args:
        score: Array of athlete scores
        
    Returns:
        Array of rank strings
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[5, 4, 3, 2, 1]],
      expected: ['Gold Medal', 'Silver Medal', 'Bronze Medal', '4', '5'],
    },
    {
      input: [[10, 3, 8, 9, 4]],
      expected: ['Gold Medal', '5', 'Bronze Medal', 'Silver Medal', '4'],
    },
  ],
  timeComplexity: 'O(n log n)',
  spaceComplexity: 'O(n)',
  leetcodeUrl: 'https://leetcode.com/problems/relative-ranks/',
  youtubeUrl: 'https://www.youtube.com/watch?v=qFKI9TKXRIs',
};
