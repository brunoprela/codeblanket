/**
 * Quiz questions for Interview Strategy section
 */

export const interviewstrategyQuiz = [
  {
    id: 'q1',
    question:
      'How do you recognize a bit manipulation problem? What keywords and patterns signal these techniques?',
    sampleAnswer:
      'Keywords: "binary representation", "XOR", "bit", "power of 2", "single number", "subset", "mask", "toggle", "flip", "set/clear flag". Patterns: 1) Find unique element (XOR). 2) Check/set/clear flags (bit operations). 3) Subset generation (2^n enumeration). 4) Fast multiply/divide by 2 (shifts). 5) Space optimization (pack booleans). 6) Count set bits, reverse bits. For example, "find single number where all others appear twice" → XOR all. "Generate all subsets" → iterate 0 to 2^n-1. "Check if power of 2" → n & (n-1) == 0. "Optimize flag storage" → bitmask. "Count 1-bits" → Brian Kernighan. Signals: mention of binary, efficient flag storage, XOR properties, subset problems. These problems often have elegant bit solutions vs complex alternatives.',
    keyPoints: [
      'Keywords: XOR, binary, bit, power of 2, subset',
      'Patterns: unique element, flags, subsets, fast ops',
      'Examples: single number, power of 2, subset generation',
      'Signals: binary operations, flag storage',
      'Elegant bit solutions vs alternatives',
    ],
  },
  {
    id: 'q2',
    question:
      'Walk me through your interview approach for bit manipulation problems from recognition to implementation.',
    sampleAnswer:
      'First, recognize bit manipulation opportunity from keywords (XOR, binary, single number). Second, identify pattern: unique element? Flags? Subset? Power of 2? Third, choose technique: XOR for cancellation, masks for flags, n & (n-1) for power of 2, iteration for subsets. Fourth, consider edge cases: 0, negative numbers, overflow, 32 vs 64 bit. Fifth, code clearly with comments explaining bit logic. Sixth, test with examples: trace bits through operations. Finally, analyze complexity and discuss alternatives. For example, "find single number": recognize XOR pattern, explain a^a=0 property, code single line result = functools.reduce (operator.xor, nums), test [4,1,2,1,2] → 4, note O(n) time O(1) space vs hash O(n) space. Show understanding of both bit tricks and fundamentals.',
    keyPoints: [
      'Recognize: keywords, patterns',
      'Identify: XOR, masks, subsets, power of 2',
      'Consider: edge cases, overflow, negatives',
      'Code clearly with comments on bit logic',
      'Test: trace bits, analyze complexity',
      'Discuss alternatives',
    ],
  },
  {
    id: 'q3',
    question:
      'What are common mistakes in bit manipulation and how do you avoid them?',
    sampleAnswer:
      'First: operator precedence (x & 1 == 0 parsed wrong). Second: forgetting edge cases (0, negative, INT_MAX). Third: mixing signed/unsigned (overflow, sign extension). Fourth: wrong shift direction (left vs right). Fifth: off-by-one in bit positions (0-indexed). Sixth: not handling negative numbers correctly (arithmetic vs logical shift). Seventh: integer overflow from left shift. My strategy: 1) Always use parentheses: (x & 1) == 0. 2) Test edge cases: 0, -1, INT_MAX, INT_MIN. 3) Be explicit about signed/unsigned. 4) Comment which direction and why. 5) Use masks like (1 << i) with bounds check. 6) Trace example manually before coding. 7) Remember Python has arbitrary precision (no overflow) but other languages do not. Most mistakes from precedence and edge cases.',
    keyPoints: [
      'Precedence: use parentheses (x & 1) == 0',
      'Edge cases: 0, negative, max/min values',
      'Signed/unsigned: overflow, sign extension',
      'Direction: left vs right shift',
      'Test: trace manually, edge cases',
      'Language: Python arbitrary vs C overflow',
    ],
  },
];
