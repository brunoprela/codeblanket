/**
 * Minimum Genetic Mutation
 * Problem ID: minimum-genetic-mutation
 * Order: 8
 */

import { Problem } from '../../../types';

export const minimum_genetic_mutationProblem: Problem = {
  id: 'minimum-genetic-mutation',
  title: 'Minimum Genetic Mutation',
  difficulty: 'Medium',
  topic: 'Breadth-First Search (BFS)',
  description: `A gene string can be represented by an 8-character long string, with choices from \`'A'\`, \`'C'\`, \`'G'\`, and \`'T'\`.

Suppose we need to investigate a mutation from a gene string \`startGene\` to a gene string \`endGene\` where one mutation is defined as one single character changed in the gene string.

- For example, \`"AACCGGTT" --> "AACCGGTA"\` is one mutation.

There is also a gene bank \`bank\` that records all the valid gene mutations. A gene must be in \`bank\` to make it a valid gene string.

Given the two gene strings \`startGene\` and \`endGene\` and the gene bank \`bank\`, return the minimum number of mutations needed to mutate from \`startGene\` to \`endGene\`. If there is no such a mutation, return \`-1\`.

Note that the starting point is assumed to be valid, so it might not be included in the bank.`,
  examples: [
    {
      input:
        'startGene = "AACCGGTT", endGene = "AACCGGTA", bank = ["AACCGGTA"]',
      output: '1',
    },
    {
      input:
        'startGene = "AACCGGTT", endGene = "AAACGGTA", bank = ["AACCGGTA","AACCGCTA","AAACGGTA"]',
      output: '2',
    },
  ],
  constraints: [
    '0 <= bank.length <= 10',
    'startGene.length == endGene.length == bank[i].length == 8',
    "startGene, endGene, and bank[i] consist of only the characters ['A', 'C', 'G', 'T']",
  ],
  hints: [
    'BFS from startGene',
    'Try changing each position to A, C, G, T',
    'Only proceed if mutation in bank',
  ],
  starterCode: `from typing import List
from collections import deque

def min_mutation(start_gene: str, end_gene: str, bank: List[str]) -> int:
    """
    Find minimum mutations to reach end gene.
    
    Args:
        start_gene: Starting gene sequence
        end_gene: Target gene sequence
        bank: Valid mutations
        
    Returns:
        Minimum mutations or -1
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: ['AACCGGTT', 'AACCGGTA', ['AACCGGTA']],
      expected: 1,
    },
    {
      input: ['AACCGGTT', 'AAACGGTA', ['AACCGGTA', 'AACCGCTA', 'AAACGGTA']],
      expected: 2,
    },
  ],
  timeComplexity: 'O(B * 8 * 4) where B is bank size',
  spaceComplexity: 'O(B)',
  leetcodeUrl: 'https://leetcode.com/problems/minimum-genetic-mutation/',
  youtubeUrl: 'https://www.youtube.com/watch?v=rZUv6R7Q8Mo',
};
