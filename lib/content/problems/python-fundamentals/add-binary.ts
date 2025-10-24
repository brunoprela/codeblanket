/**
 * Add Binary Strings
 * Problem ID: fundamentals-add-binary
 * Order: 40
 */

import { Problem } from '../../../types';

export const add_binaryProblem: Problem = {
  id: 'fundamentals-add-binary',
  title: 'Add Binary Strings',
  difficulty: 'Easy',
  description: `Add two binary strings and return the sum as a binary string.

Binary strings contain only '0' and '1' characters.

**Example:** "11" + "1" = "100" (3 + 1 = 4 in decimal)

This tests:
- String manipulation
- Binary arithmetic
- Carry handling`,
  examples: [
    {
      input: 'a = "11", b = "1"',
      output: '"100"',
    },
    {
      input: 'a = "1010", b = "1011"',
      output: '"10101"',
    },
  ],
  constraints: ['1 <= len(a), len(b) <= 10^4', "Only '0' and '1' characters"],
  hints: [
    'Process from right to left',
    'Keep track of carry',
    'Handle different lengths',
  ],
  starterCode: `def add_binary(a, b):
    """
    Add two binary strings.
    
    Args:
        a: First binary string
        b: Second binary string
        
    Returns:
        Sum as binary string
        
    Examples:
        >>> add_binary("11", "1")
        "100"
    """
    pass


# Test
print(add_binary("1010", "1011"))
`,
  testCases: [
    {
      input: ['11', '1'],
      expected: '100',
    },
    {
      input: ['1010', '1011'],
      expected: '10101',
    },
    {
      input: ['0', '0'],
      expected: '0',
    },
  ],
  solution: `def add_binary(a, b):
    result = []
    carry = 0
    i, j = len(a) - 1, len(b) - 1
    
    while i >= 0 or j >= 0 or carry:
        digit_a = int(a[i]) if i >= 0 else 0
        digit_b = int(b[j]) if j >= 0 else 0
        
        total = digit_a + digit_b + carry
        result.append(str(total % 2))
        carry = total // 2
        
        i -= 1
        j -= 1
    
    return '.join(reversed(result))


# Alternative using Python's int conversion
def add_binary_simple(a, b):
    return bin(int(a, 2) + int(b, 2))[2:]`,
  timeComplexity: 'O(max(len(a), len(b)))',
  spaceComplexity: 'O(max(len(a), len(b)))',
  order: 40,
  topic: 'Python Fundamentals',
};
