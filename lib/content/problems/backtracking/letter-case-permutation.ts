/**
 * Letter Case Permutation
 * Problem ID: letter-case-permutation
 * Order: 4
 */

import { Problem } from '../../../types';

export const letter_case_permutationProblem: Problem = {
  id: 'letter-case-permutation',
  title: 'Letter Case Permutation',
  difficulty: 'Easy',
  topic: 'Backtracking',
  description: `Given a string \`s\`, you can transform every letter individually to be lowercase or uppercase to create another string.

Return a list of all possible strings we could create. Return the output in **any order**.`,
  examples: [
    {
      input: 's = "a1b2"',
      output: '["a1b2","a1B2","A1b2","A1B2"]',
    },
    {
      input: 's = "3z4"',
      output: '["3z4","3Z4"]',
    },
  ],
  constraints: [
    '1 <= s.length <= 12',
    's consists of lowercase English letters, uppercase English letters, and digits',
  ],
  hints: [
    'For each letter, branch to try both cases',
    'For digits, just continue with same char',
  ],
  starterCode: `from typing import List

def letter_case_permutation(s: str) -> List[str]:
    """
    Generate all letter case permutations.
    
    Args:
        s: Input string
        
    Returns:
        List of all permutations
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: ['a1b2'],
      expected: ['a1b2', 'a1B2', 'A1b2', 'A1B2'],
    },
    {
      input: ['3z4'],
      expected: ['3z4', '3Z4'],
    },
    {
      input: ['C'],
      expected: ['c', 'C'],
    },
  ],
  timeComplexity: 'O(2^n * n)',
  spaceComplexity: 'O(2^n * n)',
  leetcodeUrl: 'https://leetcode.com/problems/letter-case-permutation/',
  youtubeUrl: 'https://www.youtube.com/watch?v=ZCqRY1JbVAo',
};
