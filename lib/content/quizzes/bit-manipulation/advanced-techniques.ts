/**
 * Quiz questions for Advanced Techniques section
 */

export const advancedtechniquesQuiz = [
  {
    id: 'q1',
    question:
      'Explain bit masking and how it is used for subset generation. Why is it efficient?',
    sampleAnswer:
      'Bit masking uses integers as bit arrays to represent sets. Each bit position represents an element (bit i → element i). For n elements, there are 2^n subsets, represented by integers 0 to 2^n-1. For example, set {a,b,c}, binary 101 (5) represents subset {a,c}. Generate all subsets: iterate 0 to 2^n-1, check each bit to see which elements included. For {a,b,c}: 000→{}, 001→{c}, 010→{b}, 011→{b,c}, 100→{a}, 101→{a,c}, 110→{a,b}, 111→{a,b,c}. Efficient because: compact (32 bits in one int), fast operations (check/set/clear bits), natural enumeration. Applications: DP with bitmask (TSP, job assignment), subset sum, combination generation. This is why many DP problems use "mask" to represent state.',
    keyPoints: [
      'Integer as bit array, each bit = element',
      '2^n integers represent 2^n subsets',
      'Iterate 0 to 2^n-1, check bits',
      'Compact (32 in int), fast operations',
      'Uses: DP bitmask, subset problems',
    ],
  },
  {
    id: 'q2',
    question:
      'Describe bit manipulation tricks for arithmetic: multiply/divide by powers of 2, check even/odd, power of 2.',
    sampleAnswer:
      'Multiply by 2^k: n << k. Each left shift doubles (n << 3 = n * 8). Divide by 2^k: n >> k. Each right shift halves (n >> 2 = n / 4). Check even/odd: n & 1 (0=even, 1=odd). Check power of 2: (n & (n-1)) == 0 and n != 0. Power of 2 has only one bit set. Get rightmost set bit: n & -n. Round up to next power of 2: set all bits to right of highest bit, then add 1. For example, check 16 is power of 2: 16 & 15 = 10000 & 01111 = 0 (yes). Multiply 7 by 8: 7 << 3 = 56. Check 13 even/odd: 13 & 1 = 1 (odd). These tricks are single CPU instructions vs expensive arithmetic operations.',
    keyPoints: [
      'Multiply: n << k (n * 2^k)',
      'Divide: n >> k (n / 2^k)',
      'Even/odd: n & 1',
      'Power of 2: (n & (n-1)) == 0',
      'Single instruction vs arithmetic',
    ],
  },
  {
    id: 'q3',
    question:
      'Walk me through reversing bits in an integer. What are different approaches?',
    sampleAnswer:
      'Reverse 32-bit integer: 10110... → ...01101. Approach 1: iterate bits, extract from right, build from left. For each i from 0 to 31, check bit i in input, set bit (31-i) in output. O(32). Approach 2: swap pairs, then nibbles, then bytes (divide-and-conquer). Swap bits 0↔1, 2↔3, ..., then pairs 0-1↔2-3, etc. O(log 32) operations. Approach 3: lookup table for 8-bit chunks, reverse and swap positions of 4 bytes. O(1) with O(256) space. Example approach 1 for 43261596 (binary ...11010): bit 0 is 0, set bit 31=0; bit 1 is 0, set bit 30=0; ...; bit 31 is 0, set bit 0=0. Result: reversed bits. Choice depends on: one-time (approach 1) vs frequent (lookup table).',
    keyPoints: [
      'Approach 1: iterate, extract right, build left O(32)',
      'Approach 2: divide-and-conquer swaps O(log n)',
      'Approach 3: lookup table per byte O(1)',
      'Trade: simplicity vs speed vs space',
      'Frequent calls → lookup table',
    ],
  },
];
