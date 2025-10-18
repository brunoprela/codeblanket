/**
 * Quiz questions for Introduction to Bit Manipulation section
 */

export const introductionQuiz = [
  {
    id: 'q1',
    question:
      'Explain what bit manipulation is and why it is useful in programming. Give real-world examples.',
    sampleAnswer:
      'Bit manipulation is working directly with individual bits in binary representation of numbers using bitwise operators. Useful because: 1) Fast - bitwise operations are single CPU instructions, 2) Space-efficient - store multiple flags in single integer, 3) Mathematical tricks - check power of 2, swap variables without temp. Real-world examples: permissions (Unix: read=4, write=2, execute=1, combined with OR), IP addresses and subnet masks, image compression, cryptography, database indexes, embedded systems (limited memory). For example, file permissions 755 = rwxr-xr-x stored as bits. Graphics: RGB color as 24 bits (8 per channel). Networking: check if IP in subnet by masking. Games: entity flags (isAlive, canFly, isInvincible) in single int. Hardware control: set specific pin high/low.',
    keyPoints: [
      'Direct manipulation of binary representation',
      'Fast (single CPU ops), space-efficient',
      'Uses: permissions, flags, compression, crypto',
      'Examples: Unix permissions, RGB colors, network masks',
      'Store multiple booleans in single integer',
    ],
  },
  {
    id: 'q2',
    question:
      'Compare bit manipulation to regular arithmetic operations. When is bit manipulation preferred?',
    sampleAnswer:
      'Bit manipulation uses &, |, ^, ~, <<, >> operating on binary level. Regular arithmetic uses +, -, *, /, % operating on decimal level. Prefer bit manipulation when: checking even/odd (n & 1 vs n % 2), multiplying/dividing by powers of 2 (n << 1 vs n * 2), swapping (XOR swap vs temp variable), checking power of 2 (n & (n-1) vs division loop), setting/clearing flags (OR/AND vs multiple booleans). Speed: bit ops are O(1) single instruction, while division/modulo can be slower. Space: single int holds 32 flags vs 32 boolean variables. However, prefer readable code unless performance-critical. For example, in tight loop processing millions of pixels, n << 1 faster than n * 2. But in business logic, n * 2 is clearer.',
    keyPoints: [
      'Bit: &, |, ^, ~, <<, >>. Arithmetic: +, -, *, /, %',
      'Bit faster: single instruction vs complex ops',
      'Prefer for: powers of 2, flags, even/odd',
      'Space efficient: 32 flags in one int',
      'Tradeoff: performance vs readability',
    ],
  },
  {
    id: 'q3',
    question:
      'Walk me through the binary representation of positive and negative numbers. How does twos complement work?',
    sampleAnswer:
      'Positive numbers: standard binary (5 = 101). Negative numbers use twos complement: flip all bits (ones complement) then add 1. For example, -5 in 8-bit: 5 = 00000101, flip = 11111010, add 1 = 11111011. Why twos complement? Makes subtraction same as adding negative: 5 + (-5) = 5 + 11111011 = 00000000 (carry ignored). No separate subtraction circuit needed. The most significant bit indicates sign (0=positive, 1=negative). Range for n bits: -2^(n-1) to 2^(n-1)-1. For example, 8 bits: -128 to 127. One asymmetry: -128 exists but 128 does not. Benefit: single representation for zero (no +0 and -0). This is why ~5 gives -6 (flip 101 gives 11111010 which is -6).',
    keyPoints: [
      'Positive: standard binary',
      'Negative: flip bits, add 1 (twos complement)',
      'MSB indicates sign (1=negative)',
      'Makes addition/subtraction use same circuit',
      'Range: -2^(n-1) to 2^(n-1)-1',
    ],
  },
];
