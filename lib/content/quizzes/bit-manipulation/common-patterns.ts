/**
 * Quiz questions for Common Bit Manipulation Patterns section
 */

export const commonpatternsQuiz = [
  {
    id: 'q1',
    question:
      'Explain common bit manipulation patterns: check/set/clear/toggle bit. How do they work?',
    sampleAnswer:
      'Check bit i: (n & (1 << i)) != 0. Create mask with only bit i set (1 << i), AND with n. If result non-zero, bit is set. Set bit i: n | (1 << i). OR with mask sets bit to 1 without affecting others. Clear bit i: n & ~(1 << i). Create mask with all bits 1 except i, AND clears only bit i. Toggle bit i: n ^ (1 << i). XOR flips bit i. For example, n=10 (1010), check bit 1: 10 & (1 << 1) = 10 & 2 = 2 (true). Set bit 0: 10 | 1 = 11 (1011). Clear bit 3: 10 & ~8 = 10 & 7 = 2 (0010). Toggle bit 2: 10 ^ 4 = 14 (1110). These are building blocks for all bit manipulation algorithms.',
    keyPoints: [
      'Check: n & (1 << i) != 0',
      'Set: n | (1 << i)',
      'Clear: n & ~(1 << i)',
      'Toggle: n ^ (1 << i)',
      'Foundation for all bit manipulation',
    ],
  },
  {
    id: 'q2',
    question:
      'Describe the n & (n-1) trick. Why is it useful and what problems does it solve?',
    sampleAnswer:
      'n & (n-1) removes the rightmost set bit. How it works: n-1 flips all bits after rightmost 1 (including that 1). ANDing clears that bit. Example: n=12 (1100), n-1=11 (1011), 1100 & 1011 = 1000 (8). Applications: 1) Check power of 2: (n & (n-1)) == 0 (power of 2 has only one bit). 2) Count set bits: loop n = n & (n-1) until n=0. 3) Find rightmost set bit: n & ~(n-1) or n & -n. For example, check 8 is power of 2: 8 & 7 = 1000 & 0111 = 0 (yes). Count bits in 13 (1101): 13 & 12 = 12, 12 & 11 = 8, 8 & 7 = 0, so 3 bits. This trick is O(k) where k is number of set bits, more efficient than checking all 32 bits.',
    keyPoints: [
      'n & (n-1) removes rightmost set bit',
      'n-1 flips bits after rightmost 1',
      'Check power of 2: (n & (n-1)) == 0',
      'Count bits: loop until n=0',
      'O(k) where k = number of set bits',
    ],
  },
  {
    id: 'q3',
    question:
      'Walk me through counting set bits (Hamming weight). What are different approaches and their complexities?',
    sampleAnswer:
      'Counting set bits (1s in binary). Approach 1: check each bit with (n & (1 << i)), O(log n) or O(32) for 32-bit. Approach 2: n & (n-1) removes rightmost bit, count iterations until n=0, O(k) where k is set bits (better for sparse). Approach 3: lookup table for 8-bit chunks, O(1) with O(256) space. Approach 4: divide-and-conquer (bit tricks), O(1). Example for 13 (1101) using approach 2: 13 & 12 = 12 (1100), count=1; 12 & 11 = 8 (1000), count=2; 8 & 7 = 0, count=3. For 0b10101010 (sparse), approach 2 does 4 iterations vs 8 for approach 1. Best choice depends on: sparse vs dense bits, need constant time, space constraints.',
    keyPoints: [
      'Approach 1: check each bit O(log n)',
      'Approach 2: n & (n-1) loop O(k)',
      'Approach 3: lookup table O(1) with space',
      'Choose based on: sparse/dense, time/space',
      'Brian Kernighan algorithm: n & (n-1) most common',
    ],
  },
];
