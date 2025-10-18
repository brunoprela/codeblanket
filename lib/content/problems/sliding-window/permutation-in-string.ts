/**
 * Permutation in String
 * Problem ID: permutation-in-string
 * Order: 12
 */

import { Problem } from '../../../types';

export const permutation_in_stringProblem: Problem = {
  id: 'permutation-in-string',
  title: 'Permutation in String',
  difficulty: 'Medium',
  topic: 'Sliding Window',
  description: `Given two strings \`s1\` and \`s2\`, return \`true\` if \`s2\` contains a permutation of \`s1\`, or \`false\` otherwise.

In other words, return \`true\` if one of \`s1\`'s permutations is the substring of \`s2\`.`,
  examples: [
    {
      input: 's1 = "ab", s2 = "eidbaooo"',
      output: 'true',
      explanation: 's2 contains one permutation of s1 ("ba").',
    },
    {
      input: 's1 = "ab", s2 = "eidboaoo"',
      output: 'false',
    },
  ],
  constraints: [
    '1 <= s1.length, s2.length <= 10^4',
    's1 and s2 consist of lowercase English letters',
  ],
  hints: [
    'Use a sliding window of size len(s1)',
    'Compare character frequencies',
  ],
  starterCode: `def check_inclusion(s1: str, s2: str) -> bool:
    """
    Check if s2 contains a permutation of s1.
    
    Args:
        s1: Pattern string
        s2: String to search in
        
    Returns:
        True if s2 contains permutation of s1
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: ['ab', 'eidbaooo'],
      expected: true,
    },
    {
      input: ['ab', 'eidboaoo'],
      expected: false,
    },
    {
      input: ['adc', 'dcda'],
      expected: true,
    },
  ],
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/permutation-in-string/',
  youtubeUrl: 'https://www.youtube.com/watch?v=UbyhOgBN834',
};
