/**
 * Group Data with defaultdict
 * Problem ID: intermediate-defaultdict-grouping
 * Order: 37
 */

import { Problem } from '../../../types';

export const intermediate_defaultdict_groupingProblem: Problem = {
  id: 'intermediate-defaultdict-grouping',
  title: 'Group Data with defaultdict',
  difficulty: 'Easy',
  description: `Use defaultdict to group data efficiently.

defaultdict features:
- No KeyError for missing keys
- Auto-creates default values
- Cleaner than dict.get()

**Use Case:** Grouping, counting, aggregation

This tests:
- collections.defaultdict
- Data grouping
- Default value types`,
  examples: [
    {
      input: 'Group students by grade',
      output: 'Dict of grade: [students]',
    },
  ],
  constraints: ['Use defaultdict', 'Group by key'],
  hints: [
    'from collections import defaultdict',
    'defaultdict(list) for grouping',
    'No need to check if key exists',
  ],
  starterCode: `from collections import defaultdict

def group_by_first_letter(words):
    """
    Group words by first letter.
    
    Args:
        words: List of strings
        
    Returns:
        Dict of letter: [words]
        
    Examples:
        >>> group_by_first_letter(['apple', 'ant', 'banana', 'bear'])
        {'a': ['apple', 'ant'], 'b': ['banana', 'bear']}
    """
    pass


# Test
result = group_by_first_letter(['cat', 'dog', 'cow', 'duck'])
print(result)
`,
  testCases: [
    {
      input: [['cat', 'dog', 'cow', 'duck']],
      expected: { c: ['cat', 'cow'], d: ['dog', 'duck'] },
    },
    {
      input: [['apple', 'ant', 'banana']],
      expected: { a: ['apple', 'ant'], b: ['banana'] },
    },
  ],
  solution: `from collections import defaultdict

def group_by_first_letter(words):
    groups = defaultdict(list)
    for word in words:
        groups[word[0]].append(word)
    return dict(groups)


# For counting
from collections import defaultdict

def count_occurrences(items):
    counts = defaultdict(int)
    for item in items:
        counts[item] += 1
    return dict(counts)`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',
  order: 37,
  topic: 'Python Intermediate',
};
