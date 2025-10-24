/**
 * Convert to Title Case
 * Problem ID: fundamentals-title-case
 * Order: 30
 */

import { Problem } from '../../../types';

export const title_caseProblem: Problem = {
  id: 'fundamentals-title-case',
  title: 'Convert to Title Case',
  difficulty: 'Easy',
  description: `Convert a string to title case following these rules:
- First character of each word should be uppercase
- All other characters should be lowercase
- Handle multiple spaces between words

**Example:** "hello  world" → "Hello  World"

This problem tests:
- String manipulation
- Character case operations
- Space handling`,
  examples: [
    {
      input: 's = "hello world"',
      output: '"Hello World"',
    },
    {
      input: 's = "PYTHON programming"',
      output: '"Python Programming"',
    },
    {
      input: 's = "a  b  c"',
      output: '"A  B  C"',
      explanation: 'Preserves multiple spaces',
    },
  ],
  constraints: ['1 <= len(s) <= 10^4', 'Cannot use built-in title() method'],
  hints: [
    'Track if previous character was a space',
    'Capitalize first char and after spaces',
    'Preserve spacing',
  ],
  starterCode: `def to_title_case(s):
    """
    Convert string to title case.
    
    Args:
        s: Input string
        
    Returns:
        String in title case
        
    Examples:
        >>> to_title_case("hello world")
        "Hello World"
    """
    pass`,
  testCases: [
    {
      input: ['hello world'],
      expected: 'Hello World',
    },
    {
      input: ['PYTHON programming'],
      expected: 'Python Programming',
    },
    {
      input: ['a  b  c'],
      expected: 'A  B  C',
    },
    {
      input: ['the quick brown fox'],
      expected: 'The Quick Brown Fox',
    },
  ],
  solution: `def to_title_case(s):
    result = []
    capitalize_next = True
    
    for char in s:
        if char == ' ':
            result.append(char)
            capitalize_next = True
        elif capitalize_next:
            result.append(char.upper())
            capitalize_next = False
        else:
            result.append(char.lower())
    
    return '.join(result)

# Alternative using enumerate
def to_title_case_alt(s):
    result = list(s.lower())
    
    # Capitalize first character
    if result and result[0] != ' ':
        result[0] = result[0].upper()
    
    # Capitalize after spaces
    for i in range(1, len(result)):
        if result[i-1] == ' ' and result[i] != ' ':
            result[i] = result[i].upper()
    
    return '.join(result)`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',
  order: 30,
  topic: 'Python Fundamentals',
};
