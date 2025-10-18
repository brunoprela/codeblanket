/**
 * Relative Ranks
 * Problem ID: fundamentals-relative-ranks
 * Order: 93
 */

import { Problem } from '../../../types';

export const relative_ranksProblem: Problem = {
  id: 'fundamentals-relative-ranks',
  title: 'Relative Ranks',
  difficulty: 'Easy',
  description: `Given scores, assign ranks with special medals.

Ranks:
- 1st place: "Gold Medal"
- 2nd place: "Silver Medal"
- 3rd place: "Bronze Medal"
- Others: "4", "5", etc.

**Example:** [5,4,3,2,1] → ["Gold Medal","Silver Medal","Bronze Medal","4","5"]

This tests:
- Sorting with indices
- Ranking
- String formatting`,
  examples: [
    {
      input: 'score = [5,4,3,2,1]',
      output: '["Gold Medal","Silver Medal","Bronze Medal","4","5"]',
    },
  ],
  constraints: ['1 <= len(score) <= 10^4', 'All scores unique'],
  hints: [
    'Sort with original indices',
    'Map ranks to medals/numbers',
    'Restore original order',
  ],
  starterCode: `def find_relative_ranks(score):
    """
    Assign ranks with medals.
    
    Args:
        score: Array of scores
        
    Returns:
        Array of rank strings
        
    Examples:
        >>> find_relative_ranks([5,4,3,2,1])
        ["Gold Medal","Silver Medal","Bronze Medal","4","5"]
    """
    pass


# Test
print(find_relative_ranks([10,3,8,9,4]))
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
  solution: `def find_relative_ranks(score):
    n = len(score)
    # Create list of (score, index) and sort by score descending
    sorted_scores = sorted(enumerate(score), key=lambda x: x[1], reverse=True)
    
    medals = ["Gold Medal", "Silver Medal", "Bronze Medal"]
    result = [""] * n
    
    for rank, (original_idx, _) in enumerate(sorted_scores):
        if rank < 3:
            result[original_idx] = medals[rank]
        else:
            result[original_idx] = str(rank + 1)
    
    return result`,
  timeComplexity: 'O(n log n)',
  spaceComplexity: 'O(n)',
  order: 93,
  topic: 'Python Fundamentals',
};
