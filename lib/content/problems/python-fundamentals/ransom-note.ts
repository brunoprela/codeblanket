/**
 * Ransom Note
 * Problem ID: fundamentals-ransom-note
 * Order: 55
 */

import { Problem } from '../../../types';

export const ransom_noteProblem: Problem = {
  id: 'fundamentals-ransom-note',
  title: 'Ransom Note',
  difficulty: 'Easy',
  description: `Check if ransom note can be constructed from magazine characters.

Each character in magazine can be used only once.

**Example:** 
- ransomNote = "aa", magazine = "aab" → true
- ransomNote = "aa", magazine = "ab" → false

This tests:
- Character counting
- Availability checking
- Counter operations`,
  examples: [
    {
      input: 'ransomNote = "aa", magazine = "aab"',
      output: 'True',
    },
    {
      input: 'ransomNote = "aa", magazine = "ab"',
      output: 'False',
    },
  ],
  constraints: ['1 <= len(ransomNote), len(magazine) <= 10^5'],
  hints: [
    'Count characters in magazine',
    'Check if each ransom char available',
    'Use Counter subtraction',
  ],
  starterCode: `def can_construct(ransom_note, magazine):
    """
    Check if ransom note can be made from magazine.
    
    Args:
        ransom_note: Note to construct
        magazine: Available characters
        
    Returns:
        True if constructible
        
    Examples:
        >>> can_construct("aa", "aab")
        True
    """
    pass


# Test
print(can_construct("aa", "aab"))
`,
  testCases: [
    {
      input: ['aa', 'aab'],
      expected: true,
    },
    {
      input: ['aa', 'ab'],
      expected: false,
    },
  ],
  solution: `def can_construct(ransom_note, magazine):
    from collections import Counter
    
    ransom_count = Counter(ransom_note)
    magazine_count = Counter(magazine)
    
    for char, count in ransom_count.items():
        if magazine_count[char] < count:
            return False
    
    return True


# Alternative using subtraction
def can_construct_subtract(ransom_note, magazine):
    from collections import Counter
    
    ransom_count = Counter(ransom_note)
    magazine_count = Counter(magazine)
    
    return not (ransom_count - magazine_count)`,
  timeComplexity: 'O(m + n)',
  spaceComplexity: 'O(1) - at most 26 letters',
  order: 55,
  topic: 'Python Fundamentals',
};
