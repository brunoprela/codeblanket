/**
 * Group Anagrams (Simple)
 * Problem ID: fundamentals-group-anagrams-simple
 * Order: 62
 */

import { Problem } from '../../../types';

export const group_anagrams_simpleProblem: Problem = {
  id: 'fundamentals-group-anagrams-simple',
  title: 'Group Anagrams (Simple)',
  difficulty: 'Medium',
  description: `Group strings that are anagrams of each other.

Anagrams are words made by rearranging letters.

**Example:** ["eat","tea","tan","ate","nat","bat"]
→ [["bat"],["nat","tan"],["ate","eat","tea"]]

This tests:
- String sorting
- Dictionary grouping
- List manipulation`,
  examples: [
    {
      input: 'strs = ["eat","tea","tan","ate","nat","bat"]',
      output: '[["bat"],["nat","tan"],["ate","eat","tea"]]',
    },
  ],
  constraints: ['1 <= len(strs) <= 10^4', '0 <= len(strs[i]) <= 100'],
  hints: [
    'Sort each string to get signature',
    'Group strings with same signature',
    'Use defaultdict(list)',
  ],
  starterCode: `def group_anagrams(strs):
    """
    Group anagram strings together.
    
    Args:
        strs: List of strings
        
    Returns:
        List of grouped anagrams
        
    Examples:
        >>> group_anagrams(["eat","tea","tan"])
        [["eat","tea"],["tan"]]
    """
    pass


# Test
print(group_anagrams(["eat","tea","tan","ate","nat","bat"]))
`,
  testCases: [
    {
      input: [['eat', 'tea', 'tan', 'ate', 'nat', 'bat']],
      expected: [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']],
    },
  ],
  solution: `def group_anagrams(strs):
    from collections import defaultdict
    
    groups = defaultdict(list)
    
    for s in strs:
        # Sort string to get signature
        key = '.join(sorted(s))
        groups[key].append(s)
    
    return list(groups.values())`,
  timeComplexity: 'O(n * k log k) where k is max string length',
  spaceComplexity: 'O(n * k)',
  order: 62,
  topic: 'Python Fundamentals',
};
