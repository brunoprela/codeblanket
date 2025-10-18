/**
 * Dict Comprehension with Filtering
 * Problem ID: intermediate-dict-comprehension-filter
 * Order: 45
 */

import { Problem } from '../../../types';

export const intermediate_dict_comprehension_filterProblem: Problem = {
  id: 'intermediate-dict-comprehension-filter',
  title: 'Dict Comprehension with Filtering',
  difficulty: 'Easy',
  description: `Use dict comprehensions with conditional filtering.

**Syntax:**
\`\`\`python
{k: v for k, v in items if condition}
\`\`\`

This tests:
- Dict comprehensions
- Filtering
- Key-value transformation`,
  examples: [
    {
      input: 'Filter dict by value > 10',
      output: 'New dict with only matching items',
    },
  ],
  constraints: ['Use dict comprehension', 'Add condition'],
  hints: ['{k: v for ...}', 'Add if condition', 'Can transform k and v'],
  starterCode: `def filter_scores(scores, min_score):
    """
    Filter scores above minimum.
    
    Args:
        scores: Dict of name: score
        min_score: Minimum passing score
        
    Returns:
        Dict with only passing scores
        
    Examples:
        >>> filter_scores({'Alice': 85, 'Bob': 65, 'Charlie': 95}, 70)
        {'Alice': 85, 'Charlie': 95}
    """
    pass


# Test
print(filter_scores({'A': 90, 'B': 60, 'C': 75, 'D': 50}, 70))
`,
  testCases: [
    {
      input: [{ A: 90, B: 60, C: 75, D: 50 }, 70],
      expected: { A: 90, C: 75 },
    },
    {
      input: [{ x: 10, y: 20, z: 5 }, 10],
      expected: { x: 10, y: 20 },
    },
  ],
  solution: `def filter_scores(scores, min_score):
    return {name: score for name, score in scores.items() if score >= min_score}


# Transform keys and values
def uppercase_keys_double_values(d):
    return {k.upper(): v * 2 for k, v in d.items()}`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',
  order: 45,
  topic: 'Python Intermediate',
};
