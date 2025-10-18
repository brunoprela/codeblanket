/**
 * Quiz questions for Bitwise Operators section
 */

export const operatorsQuiz = [
  {
    id: 'q1',
    question:
      'Explain each bitwise operator (AND, OR, XOR, NOT, shift). What are their primary use cases?',
    sampleAnswer:
      'AND (&): both bits 1 → result 1, else 0. Use: check if bit set (n & (1 << i)), clear bit (n & ~(1 << i)), mask to keep certain bits. OR (|): any bit 1 → result 1. Use: set bit (n | (1 << i)), combine flags. XOR (^): bits different → result 1. Use: toggle bit (n ^ (1 << i)), swap variables, find unique element. NOT (~): flip all bits. Use: create mask, ones complement. Left shift (<<): multiply by 2^k. Use: fast multiplication, create bit mask (1 << i). Right shift (>>): divide by 2^k. Use: fast division, extract high bits. For example: permissions check (hasPermission = userPerms & WRITE_PERM), toggle LED (ledState ^= 1), fast multiply by 8 (n << 3).',
    keyPoints: [
      'AND: check/clear bits, masking',
      'OR: set bits, combine flags',
      'XOR: toggle, swap, find unique',
      'NOT: flip all bits, create mask',
      'Shifts: fast multiply/divide by powers of 2',
    ],
  },
  {
    id: 'q2',
    question:
      'Describe left vs right shift. What is the difference between arithmetic and logical right shift?',
    sampleAnswer:
      'Left shift (<<): move bits left, fill right with 0s. Each left shift multiplies by 2. Example: 5 << 2 = 20 (101 → 10100). Right shift (>>): move bits right. Two types: logical (fill left with 0s) and arithmetic (fill left with sign bit, preserving sign). Logical: treats number as unsigned (5 >> 1 = 2, 101 → 010). Arithmetic: preserves sign for negative numbers (-8 >> 1 = -4, maintains negative). Language behavior: Python >> is arithmetic, Java >> is arithmetic and >>> is logical. Use cases: left shift for fast multiply by powers of 2, right shift for fast divide. Caution: shifting negative numbers can be tricky (sign extension). Shifting by 32+ bits is undefined in 32-bit integers.',
    keyPoints: [
      'Left shift: move left, fill 0s, multiply by 2^k',
      'Right shift: move right, divide by 2^k',
      'Logical: fill with 0s (unsigned)',
      'Arithmetic: fill with sign bit (signed)',
      'Language differences: Python/Java behavior',
    ],
  },
  {
    id: 'q3',
    question:
      'Walk me through operator precedence and common pitfalls with bitwise operators.',
    sampleAnswer:
      'Precedence (high to low): NOT (~), then shifts (<<, >>), then AND (&), then XOR (^), then OR (|). All bitwise ops have LOWER precedence than comparison operators (==, <, >). Common pitfall: x & 1 == 0 is parsed as x & (1 == 0) = x & 0 = 0, always false! Should be (x & 1) == 0. Another: if (flags & MASK) could be 0 (false) or non-zero (true) - works in boolean context. Mixing signed and unsigned: ~0 is -1 (signed) but 0xFFFFFFFF (unsigned). Overflow: left shift can overflow (5 << 30). Right shift on negative: sign extension vs zero fill. Best practice: use parentheses liberally, understand precedence, test with edge cases (0, -1, max values).',
    keyPoints: [
      'Precedence: ~, shifts, &, ^, |',
      'Lower precedence than comparisons',
      'Pitfall: x & 1 == 0 parsed wrong',
      'Use parentheses: (x & 1) == 0',
      'Careful: signed/unsigned, overflow, edge cases',
    ],
  },
];
