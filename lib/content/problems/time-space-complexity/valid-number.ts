/**
 * Valid Number
 * Problem ID: valid-number
 * Order: 6
 */

import { Problem } from '../../../types';

export const valid_numberProblem: Problem = {
  id: 'valid-number',
  title: 'Valid Number',
  difficulty: 'Hard',
  topic: 'Time & Space Complexity',

  leetcodeUrl: 'https://leetcode.com/problems/valid-number/',
  youtubeUrl: 'https://www.youtube.com/watch?v=QfRSeibcugw',
  order: 6,
  description: `A **valid number** can be split up into these components (in order):

1. A **decimal number** or an **integer**.
2. (Optional) An \`'e'\` or \`'E'\`, followed by an **integer**.

A **decimal number** can be split up into these components (in order):
1. (Optional) A sign character (either \`'+'\` or \`'-'\`).
2. One of the following formats:
   - One or more digits, followed by a dot \`'.'\`.
   - One or more digits, followed by a dot \`'.'\`, followed by one or more digits.
   - A dot \`'.'\`, followed by one or more digits.

An **integer** can be split up into these components (in order):
1. (Optional) A sign character (either \`'+'\` or \`'-'\`).
2. One or more digits.

Given a string \`s\`, return \`true\` if \`s\` is a **valid number**.

**Complexity Focus:** This problem tests your ability to implement a state machine with O(n) time and O(1) space.`,
  examples: [
    {
      input: 's = "0"',
      output: 'true',
    },
    {
      input: 's = "e"',
      output: 'false',
    },
    {
      input: 's = "."',
      output: 'false',
    },
    {
      input: 's = "0.1"',
      output: 'true',
    },
  ],
  constraints: [
    '1 <= s.length <= 20',
    "s consists of only English letters (both uppercase and lowercase), digits (0-9), plus '+', minus '-', or dot '.'.",
  ],
  hints: [
    'Use a finite state machine',
    'Track: seen digit, seen dot, seen exponent',
    'Process character by character - O(n) time',
  ],
  starterCode: `def is_number(s: str) -> bool:
    # Write your code here
    pass
`,
  testCases: [
    {
      input: ['0'],
      expected: true,
    },
    {
      input: ['e'],
      expected: false,
    },
    {
      input: ['.'],
      expected: false,
    },
    {
      input: ['0.1'],
      expected: true,
    },
    {
      input: ['2e10'],
      expected: true,
    },
    {
      input: ['-90e3'],
      expected: true,
    },
    {
      input: ['1e'],
      expected: false,
    },
    {
      input: ['e3'],
      expected: false,
    },
    {
      input: ['6e-1'],
      expected: true,
    },
    {
      input: ['.1'],
      expected: true,
    },
  ],
  solution: `# State Machine - O(n) time, O(1) space
def is_number(s):
    seen_digit = False
    seen_exponent = False
    seen_dot = False
    
    for i, char in enumerate(s):
        if char.isdigit():
            seen_digit = True
        elif char in ['+', '-']:
            # Sign must be at start or right after exponent
            if i > 0 and s[i-1] not in ['e', 'E']:
                return False
        elif char in ['e', 'E']:
            # Can't have two exponents, must have seen digit before
            if seen_exponent or not seen_digit:
                return False
            seen_exponent = True
            seen_digit = False  # Must have digit after exponent
        elif char == '.':
            # Can't have two dots or dot after exponent
            if seen_dot or seen_exponent:
                return False
            seen_dot = True
        else:
            # Invalid character
            return False
    
    return seen_digit

# Alternative: Try parsing with try-except (simpler but less educational)
def is_number_simple(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
`,
  timeComplexity: 'O(n) - single pass through the string',
  spaceComplexity: 'O(1) - only tracking a few boolean flags',
};
