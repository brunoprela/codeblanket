/**
 * Dict Merge Operators (| and |=)
 * Problem ID: intermediate-dict-merge-operators
 * Order: 40
 */

import { Problem } from '../../../types';

export const intermediate_dict_merge_operatorsProblem: Problem = {
  id: 'intermediate-dict-merge-operators',
  title: 'Dict Merge Operators (| and |=)',
  difficulty: 'Easy',
  description: `Use Python 3.9+ dict merge operators.

**Operators:**
- | : Merge (like union)
- |= : In-place merge

**Example:**
\`\`\`python
d1 = {'a': 1, 'b': 2}
d2 = {'b': 3, 'c': 4}
merged = d1 | d2  # {'a': 1, 'b': 3, 'c': 4}
\`\`\`

This tests:
- Dict operators
- Merge behavior
- Modern Python syntax`,
  examples: [
    {
      input: 'd1 | d2',
      output: 'Merged dict, d2 values win',
    },
  ],
  constraints: ['Use | operator', 'Right dict values take precedence'],
  hints: [
    'd1 | d2 creates new dict',
    'd1 |= d2 modifies d1',
    'Later values override',
  ],
  starterCode: `def merge_configs(defaults, overrides):
    """
    Merge configuration dicts.
    
    Args:
        defaults: Default config
        overrides: Override values
        
    Returns:
        Merged dict (overrides take precedence)
        
    Examples:
        >>> defaults = {'color': 'blue', 'size': 10}
        >>> overrides = {'size': 20}
        >>> merge_configs(defaults, overrides)
        {'color': 'blue', 'size': 20}
    """
    pass


# Test
print(merge_configs({'a': 1, 'b': 2}, {'b': 3, 'c': 4}))
`,
  testCases: [
    {
      input: [
        { a: 1, b: 2 },
        { b: 3, c: 4 },
      ],
      expected: { a: 1, b: 3, c: 4 },
    },
    {
      input: [{ x: 10 }, { y: 20 }],
      expected: { x: 10, y: 20 },
    },
  ],
  solution: `def merge_configs(defaults, overrides):
    return defaults | overrides


# In-place merge
def merge_configs_inplace(defaults, overrides):
    defaults |= overrides
    return defaults


# Pre-3.9 alternative
def merge_configs_old(defaults, overrides):
    return {**defaults, **overrides}`,
  timeComplexity: 'O(n + m)',
  spaceComplexity: 'O(n + m)',
  order: 40,
  topic: 'Python Intermediate',
};
