/**
 * Merge Dictionaries
 * Problem ID: fundamentals-merge-dicts
 * Order: 9
 */

import { Problem } from '../../../types';

export const merge_dictsProblem: Problem = {
  id: 'fundamentals-merge-dicts',
  title: 'Merge Dictionaries',
  difficulty: 'Easy',
  description: `Merge multiple dictionaries with conflict resolution.

**Requirements:**
- Take a list of dictionaries and merge them into one
- For duplicate keys, use a strategy:
  - "last": Keep value from the last dictionary (default)
  - "first": Keep value from the first dictionary
  - "sum": Sum all values (assumes numeric values)
  - "list": Collect all values in a list

**Example:**
\`\`\`python
dicts = [{"a": 1, "b": 2}, {"b": 3, "c": 4}]
merge(dicts, "last") → {"a": 1, "b": 3, "c": 4}
merge(dicts, "sum") → {"a": 1, "b": 5, "c": 4}
\`\`\``,
  examples: [
    {
      input: 'dicts = [{"a": 1, "b": 2}, {"b": 3, "c": 4}], strategy = "last"',
      output: '{"a": 1, "b": 3, "c": 4}',
    },
  ],
  constraints: [
    'List contains 1 to 100 dictionaries',
    'Keys are strings, values are integers',
  ],
  hints: [
    'Iterate through all dictionaries',
    'Use a result dictionary to accumulate values',
    'Handle each strategy differently',
  ],
  starterCode: `def merge_dicts(dicts, strategy="last"):
    """
    Merge multiple dictionaries with conflict resolution.
    
    Args:
        dicts: List of dictionaries to merge
        strategy: How to handle duplicate keys
                  "last", "first", "sum", or "list"
        
    Returns:
        Merged dictionary
        
    Examples:
        >>> merge_dicts([{"a": 1}, {"a": 2}], "last")
        {"a": 2}
        >>> merge_dicts([{"a": 1}, {"a": 2}], "sum")
        {"a": 3}
    """
    pass


# Test
dicts = [{"a": 1, "b": 2}, {"b": 3, "c": 4}, {"a": 5}]
print(merge_dicts(dicts, "last"))
print(merge_dicts(dicts, "sum"))
`,
  testCases: [
    {
      input: [
        [
          { a: 1, b: 2 },
          { b: 3, c: 4 },
        ],
        'last',
      ],
      expected: { a: 1, b: 3, c: 4 },
    },
    {
      input: [
        [
          { a: 1, b: 2 },
          { b: 3, c: 4 },
        ],
        'sum',
      ],
      expected: { a: 1, b: 5, c: 4 },
    },
  ],
  solution: `def merge_dicts(dicts, strategy="last"):
    result = {}
    
    for d in dicts:
        for key, value in d.items():
            if key not in result:
                # First occurrence
                result[key] = value if strategy != "list" else [value]
            else:
                # Duplicate key
                if strategy == "last":
                    result[key] = value
                elif strategy == "first":
                    pass  # Keep existing value
                elif strategy == "sum":
                    result[key] += value
                elif strategy == "list":
                    result[key].append(value)
    
    return result


# Using Python 3.9+ union operator
def merge_dicts_modern(dicts, strategy="last"):
    if strategy == "last":
        result = {}
        for d in dicts:
            result = result | d  # Python 3.9+
        return result
    else:
        return merge_dicts(dicts, strategy)


# Using ChainMap (for reading, preserves separate dicts)
from collections import ChainMap

def merge_dicts_chainmap(dicts):
    return dict(ChainMap(*reversed(dicts)))`,
  timeComplexity: 'O(n*k) where n is number of dicts, k is avg keys per dict',
  spaceComplexity: 'O(k) for result dictionary',
  order: 9,
  topic: 'Python Fundamentals',
};
