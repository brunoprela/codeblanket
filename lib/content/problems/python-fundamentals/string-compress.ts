/**
 * String Compression
 * Problem ID: fundamentals-string-compress
 * Order: 36
 */

import { Problem } from '../../../types';

export const string_compressProblem: Problem = {
  id: 'fundamentals-string-compress',
  title: 'String Compression',
  difficulty: 'Medium',
  description: `Compress a string using run-length encoding.

- Replace consecutive repeated characters with character + count
- If compressed string is not shorter, return original
- Only compress if result is shorter

**Example:** "aabcccccaaa" → "a2b1c5a3"

This tests:
- String building
- Counting consecutive characters
- Conditional logic`,
  examples: [
    {
      input: 's = "aabcccccaaa"',
      output: '"a2b1c5a3"',
    },
    {
      input: 's = "abcd"',
      output: '"abcd"',
      explanation: 'Compressed would be longer',
    },
  ],
  constraints: ['1 <= len(s) <= 1000', 'Only letters (a-z, A-Z)'],
  hints: [
    'Count consecutive characters',
    'Build compressed string as you go',
    'Compare lengths at the end',
  ],
  starterCode: `def compress_string(s):
    """
    Compress string using run-length encoding.
    
    Args:
        s: Input string
        
    Returns:
        Compressed string or original if not shorter
        
    Examples:
        >>> compress_string("aabcccccaaa")
        "a2b1c5a3"
    """
    pass


# Test
print(compress_string("aabcccccaaa"))
`,
  testCases: [
    {
      input: ['aabcccccaaa'],
      expected: 'a2b1c5a3',
    },
    {
      input: ['abcd'],
      expected: 'abcd',
    },
    {
      input: ['aaa'],
      expected: 'a3',
    },
  ],
  solution: `def compress_string(s):
    if not s:
        return s
    
    compressed = []
    count = 1
    
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            count += 1
        else:
            compressed.append(s[i - 1] + str(count))
            count = 1
    
    # Add last group
    compressed.append(s[-1] + str(count))
    
    result = '.join(compressed)
    return result if len(result) < len(s) else s`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',
  order: 36,
  topic: 'Python Fundamentals',
};
