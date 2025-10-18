/**
 * Longest Common Prefix
 * Problem ID: fundamentals-longest-common-prefix
 * Order: 29
 */

import { Problem } from '../../../types';

export const longest_common_prefixProblem: Problem = {
  id: 'fundamentals-longest-common-prefix',
  title: 'Longest Common Prefix',
  difficulty: 'Easy',
  description: `Find the longest common prefix string amongst an array of strings.

Return empty string if no common prefix.

**Example:** ["flower", "flow", "flight"] → "fl"

This problem tests:
- String comparison
- Iteration strategies
- Edge case handling`,
  examples: [
    {
      input: 'strs = ["flower", "flow", "flight"]',
      output: '"fl"',
    },
    {
      input: 'strs = ["dog", "racecar", "car"]',
      output: '""',
      explanation: 'No common prefix',
    },
  ],
  constraints: ['1 <= len(strs) <= 200', '0 <= len(strs[i]) <= 200'],
  hints: [
    'Compare characters at same position across all strings',
    'Stop when characters differ',
    'Handle empty strings',
  ],
  starterCode: `def longest_common_prefix(strs):
    """
    Find longest common prefix in array of strings.
    
    Args:
        strs: List of strings
        
    Returns:
        Longest common prefix string
        
    Examples:
        >>> longest_common_prefix(["flower", "flow", "flight"])
        "fl"
    """
    pass`,
  testCases: [
    {
      input: [['flower', 'flow', 'flight']],
      expected: 'fl',
    },
    {
      input: [['dog', 'racecar', 'car']],
      expected: '',
    },
    {
      input: [['']],
      expected: '',
    },
    {
      input: [['a']],
      expected: 'a',
    },
  ],
  solution: `def longest_common_prefix(strs):
    if not strs:
        return ""
    
    # Take first string as reference
    prefix = strs[0]
    
    # Compare with each string
    for string in strs[1:]:
        # Reduce prefix until it matches beginning of string
        while not string.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    
    return prefix

# Vertical scanning approach
def longest_common_prefix_vertical(strs):
    if not strs:
        return ""
    
    # Check each character position
    for i in range(len(strs[0])):
        char = strs[0][i]
        # Check if all strings have same char at position i
        for string in strs[1:]:
            if i >= len(string) or string[i] != char:
                return strs[0][:i]
    
    return strs[0]

# Using zip
def longest_common_prefix_zip(strs):
    if not strs:
        return ""
    
    result = []
    for chars in zip(*strs):
        if len(set(chars)) == 1:
            result.append(chars[0])
        else:
            break
    
    return ''.join(result)`,
  timeComplexity: 'O(S) where S is sum of all characters',
  spaceComplexity: 'O(1)',
  order: 29,
  topic: 'Python Fundamentals',
};
