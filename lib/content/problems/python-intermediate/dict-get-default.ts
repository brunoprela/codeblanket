/**
 * Dict get() with Default Value
 * Problem ID: intermediate-dict-get-default
 * Order: 42
 */

import { Problem } from '../../../types';

export const intermediate_dict_get_defaultProblem: Problem = {
  id: 'intermediate-dict-get-default',
  title: 'Dict get() with Default Value',
  difficulty: 'Easy',
  description: `Use dict.get() to safely access dictionary values with defaults.

**Syntax:**
\`\`\`python
value = d.get(key, default)
\`\`\`

Safer than d[key] which raises KeyError.

This tests:
- Safe dict access
- Default values
- Avoiding KeyError`,
  examples: [
    {
      input: 'd.get("missing", 0)',
      output: '0 (no KeyError)',
    },
  ],
  constraints: ['Use .get() method', 'Provide defaults'],
  hints: [
    'dict.get(key, default)',
    'Returns default if key missing',
    'No exception raised',
  ],
  starterCode: `def count_items(items):
    """
    Count occurrences of each item.
    
    Args:
        items: List of items
        
    Returns:
        Dict of item: count
        
    Examples:
        >>> count_items(['a', 'b', 'a', 'c', 'b', 'a'])
        {'a': 3, 'b': 2, 'c': 1}
    """
    counts = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return counts


# Test
print(count_items(['x', 'y', 'x', 'z', 'y', 'x']))
`,
  testCases: [
    {
      input: [['x', 'y', 'x', 'z', 'y', 'x']],
      expected: { x: 3, y: 2, z: 1 },
    },
    {
      input: [['a', 'a', 'a']],
      expected: { a: 3 },
    },
  ],
  solution: `def count_items(items):
    counts = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return counts


# Alternative with setdefault
def count_items_setdefault(items):
    counts = {}
    for item in items:
        counts.setdefault(item, 0)
        counts[item] += 1
    return counts`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(k) where k is unique items',
  order: 42,
  topic: 'Python Intermediate',
};
