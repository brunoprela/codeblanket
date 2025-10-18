/**
 * Find the Town Judge
 * Problem ID: find-town-judge
 * Order: 6
 */

import { Problem } from '../../../types';

export const find_town_judgeProblem: Problem = {
  id: 'find-town-judge',
  title: 'Find the Town Judge',
  difficulty: 'Easy',
  topic: 'Graphs',
  description: `In a town, there are \`n\` people labeled from \`1\` to \`n\`. There is a rumor that one of these people is secretly the town judge.

If the town judge exists, then:

1. The town judge trusts nobody.
2. Everybody (except for the town judge) trusts the town judge.
3. There is exactly one person that satisfies properties 1 and 2.

You are given an array \`trust\` where \`trust[i] = [ai, bi]\` representing that the person labeled \`ai\` trusts the person labeled \`bi\`. If a trust relationship does not exist in \`trust\` array, then such a trust relationship does not exist.

Return the label of the town judge if the town judge exists and can be identified, or return \`-1\` otherwise.`,
  examples: [
    {
      input: 'n = 2, trust = [[1,2]]',
      output: '2',
    },
    {
      input: 'n = 3, trust = [[1,3],[2,3]]',
      output: '3',
    },
    {
      input: 'n = 3, trust = [[1,3],[2,3],[3,1]]',
      output: '-1',
    },
  ],
  constraints: [
    '1 <= n <= 1000',
    '0 <= trust.length <= 10^4',
    'trust[i].length == 2',
    'All the pairs of trust are unique',
    'ai != bi',
    '1 <= ai, bi <= n',
  ],
  hints: [
    'Count in-degree and out-degree for each person',
    'Judge has in-degree = n-1, out-degree = 0',
  ],
  starterCode: `from typing import List

def find_judge(n: int, trust: List[List[int]]) -> int:
    """
    Find the town judge.
    
    Args:
        n: Number of people
        trust: Trust relationships
        
    Returns:
        Judge label or -1
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [2, [[1, 2]]],
      expected: 2,
    },
    {
      input: [
        3,
        [
          [1, 3],
          [2, 3],
        ],
      ],
      expected: 3,
    },
    {
      input: [
        3,
        [
          [1, 3],
          [2, 3],
          [3, 1],
        ],
      ],
      expected: -1,
    },
  ],
  timeComplexity: 'O(E)',
  spaceComplexity: 'O(N)',
  leetcodeUrl: 'https://leetcode.com/problems/find-the-town-judge/',
  youtubeUrl: 'https://www.youtube.com/watch?v=ZwFjogT5HuQ',
};
