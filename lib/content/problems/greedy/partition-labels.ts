/**
 * Partition Labels
 * Problem ID: partition-labels
 * Order: 8
 */

import { Problem } from '../../../types';

export const partition_labelsProblem: Problem = {
  id: 'partition-labels',
  title: 'Partition Labels',
  difficulty: 'Medium',
  topic: 'Greedy',
  description: `You are given a string \`s\`. We want to partition the string into as many parts as possible so that each letter appears in at most one part.

Note that the partition is done so that after concatenating all the parts in order, the resultant string should be \`s\`.

Return a list of integers representing the size of these parts.`,
  examples: [
    {
      input: 's = "ababcbacadefegdehijhklij"',
      output: '[9,7,8]',
      explanation: 'The partition is "ababcbaca", "defegde", "hijhklij".',
    },
    {
      input: 's = "eccbbbbdec"',
      output: '[10]',
    },
  ],
  constraints: [
    '1 <= s.length <= 500',
    's consists of lowercase English letters',
  ],
  hints: [
    'Find last occurrence of each character',
    'Extend partition to include all occurrences',
  ],
  starterCode: `from typing import List

def partition_labels(s: str) -> List[int]:
    """
    Partition string into maximum parts.
    
    Args:
        s: Input string
        
    Returns:
        List of partition sizes
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: ['ababcbacadefegdehijhklij'],
      expected: [9, 7, 8],
    },
    {
      input: ['eccbbbbdec'],
      expected: [10],
    },
  ],
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/partition-labels/',
  youtubeUrl: 'https://www.youtube.com/watch?v=B7m8UmZE-vw',
};
