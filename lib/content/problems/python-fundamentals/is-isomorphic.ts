/**
 * Isomorphic Strings
 * Problem ID: fundamentals-is-isomorphic
 * Order: 57
 */

import { Problem } from '../../../types';

export const is_isomorphicProblem: Problem = {
  id: 'fundamentals-is-isomorphic',
  title: 'Isomorphic Strings',
  difficulty: 'Easy',
  description: `Check if two strings are isomorphic.

Strings are isomorphic if characters in s can be replaced to get t, maintaining:
- One-to-one character mapping
- Character order preserved

**Example:** "egg" and "add" → true (e→a, g→d)
"foo" and "bar" → false (o can't map to both a and r)

This tests:
- Character mapping
- Bidirectional checking
- Hash map usage`,
  examples: [
    {
      input: 's = "egg", t = "add"',
      output: 'True',
    },
    {
      input: 's = "foo", t = "bar"',
      output: 'False',
    },
  ],
  constraints: ['1 <= len(s), len(t) <= 5*10^4', 'Same length'],
  hints: [
    'Map each char in s to char in t',
    'Also ensure reverse mapping is unique',
    'Use two dictionaries',
  ],
  starterCode: `def is_isomorphic(s, t):
    """
    Check if strings are isomorphic.
    
    Args:
        s: First string
        t: Second string
        
    Returns:
        True if isomorphic
        
    Examples:
        >>> is_isomorphic("egg", "add")
        True
    """
    pass


# Test
print(is_isomorphic("egg", "add"))
`,
  testCases: [
    {
      input: ['egg', 'add'],
      expected: true,
    },
    {
      input: ['foo', 'bar'],
      expected: false,
    },
  ],
  solution: `def is_isomorphic(s, t):
    if len(s) != len(t):
        return False
    
    map_s_to_t = {}
    map_t_to_s = {}
    
    for char_s, char_t in zip(s, t):
        if char_s in map_s_to_t:
            if map_s_to_t[char_s] != char_t:
                return False
        else:
            map_s_to_t[char_s] = char_t
        
        if char_t in map_t_to_s:
            if map_t_to_s[char_t] != char_s:
                return False
        else:
            map_t_to_s[char_t] = char_s
    
    return True`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1) - at most 256 ASCII chars',
  order: 57,
  topic: 'Python Fundamentals',
};
