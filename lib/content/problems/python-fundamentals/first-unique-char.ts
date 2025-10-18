/**
 * First Unique Character
 * Problem ID: fundamentals-first-unique-char
 * Order: 54
 */

import { Problem } from '../../../types';

export const first_unique_charProblem: Problem = {
  id: 'fundamentals-first-unique-char',
  title: 'First Unique Character',
  difficulty: 'Easy',
  description: `Find the index of first non-repeating character in a string.

Return -1 if all characters repeat.

**Example:** "leetcode" → 0 (l appears once)
"loveleetcode" → 2 (v appears once)

This tests:
- Character frequency
- First occurrence tracking
- Two-pass algorithm`,
  examples: [
    {
      input: 's = "leetcode"',
      output: '0',
    },
    {
      input: 's = "loveleetcode"',
      output: '2',
    },
    {
      input: 's = "aabb"',
      output: '-1',
    },
  ],
  constraints: ['1 <= len(s) <= 10^5', 'Only lowercase letters'],
  hints: [
    'Count character frequencies first',
    'Then find first char with count=1',
    'Use Counter or dictionary',
  ],
  starterCode: `def first_uniq_char(s):
    """
    Find index of first unique character.
    
    Args:
        s: Input string
        
    Returns:
        Index of first unique char or -1
        
    Examples:
        >>> first_uniq_char("leetcode")
        0
    """
    pass


# Test
print(first_uniq_char("loveleetcode"))
`,
  testCases: [
    {
      input: ['leetcode'],
      expected: 0,
    },
    {
      input: ['loveleetcode'],
      expected: 2,
    },
    {
      input: ['aabb'],
      expected: -1,
    },
  ],
  solution: `def first_uniq_char(s):
    from collections import Counter
    
    counts = Counter(s)
    
    for i, char in enumerate(s):
        if counts[char] == 1:
            return i
    
    return -1


# Alternative without Counter
def first_uniq_char_dict(s):
    char_count = {}
    
    for char in s:
        char_count[char] = char_count.get(char, 0) + 1
    
    for i, char in enumerate(s):
        if char_count[char] == 1:
            return i
    
    return -1`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1) - at most 26 letters',
  order: 54,
  topic: 'Python Fundamentals',
};
