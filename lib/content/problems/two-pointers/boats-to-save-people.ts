/**
 * Boats to Save People
 * Problem ID: boats-to-save-people
 * Order: 14
 */

import { Problem } from '../../../types';

export const boats_to_save_peopleProblem: Problem = {
  id: 'boats-to-save-people',
  title: 'Boats to Save People',
  difficulty: 'Medium',
  topic: 'Two Pointers',
  description: `You are given an array \`people\` where \`people[i]\` is the weight of the \`i-th\` person, and an infinite number of boats where each boat can carry a maximum weight of \`limit\`. Each boat carries at most two people at the same time, provided the sum of the weight of those people is at most \`limit\`.

Return the minimum number of boats to carry every given person.`,
  examples: [
    {
      input: 'people = [1,2], limit = 3',
      output: '1',
      explanation: '1 boat (1, 2)',
    },
    {
      input: 'people = [3,2,2,1], limit = 3',
      output: '3',
      explanation: '3 boats (1, 2), (2) and (3)',
    },
    {
      input: 'people = [3,5,3,4], limit = 5',
      output: '4',
      explanation: '4 boats (3), (3), (4), (5)',
    },
  ],
  constraints: [
    '1 <= people.length <= 5 * 10^4',
    '1 <= people[i] <= limit <= 3 * 10^4',
  ],
  hints: [
    'Sort the people by weight',
    'Use two pointers: one for lightest, one for heaviest',
    'Try to pair the heaviest with the lightest',
  ],
  starterCode: `from typing import List

def num_rescue_boats(people: List[int], limit: int) -> int:
    """
    Find minimum boats needed to rescue all people.
    
    Args:
        people: Array of weights
        limit: Maximum weight per boat
        
    Returns:
        Minimum number of boats needed
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[1, 2], 3],
      expected: 1,
    },
    {
      input: [[3, 2, 2, 1], 3],
      expected: 3,
    },
    {
      input: [[3, 5, 3, 4], 5],
      expected: 4,
    },
    {
      input: [[5, 1, 4, 2], 6],
      expected: 2,
    },
  ],
  timeComplexity: 'O(n log n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/boats-to-save-people/',
  youtubeUrl: 'https://www.youtube.com/watch?v=XbaxWuHIWUs',
};
