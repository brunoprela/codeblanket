export default [
  {
    id: 'cp-m1-s9-q1',
    section: 'Bits, Bytes & Bitwise Operations',
    question: 'What does the expression (n & (n-1)) do to a number n?',
    options: [
      'It doubles the number',
      'It removes the rightmost set bit from n',
      'It counts the number of bits in n',
      'It checks if n is even',
    ],
    correctAnswer: 1,
    explanation:
      "`n & (n-1)` removes the rightmost set bit from n. Example: n=12 (1100 in binary), n-1=11 (1011), n & (n-1) = 1000 (8). This is useful for: (1) Counting bits: repeatedly apply until n becomes 0, (2) Checking if n is a power of 2: `n & (n-1) == 0` (powers of 2 have exactly one bit set). Algorithm: Brian Kernighan's bit counting - counts set bits in O(number of set bits) instead of O(log n). Code: `while(n) { count++; n = n & (n-1); }`. This is a fundamental bit manipulation trick used in many algorithms.",
    difficulty: 'intermediate',
  },
  {
    id: 'cp-m1-s9-q2',
    section: 'Bits, Bytes & Bitwise Operations',
    question:
      'How do you check if the i-th bit (0-indexed from right) of number n is set?',
    options: [
      'if(n & i)',
      'if(n & (1 << i))',
      'if(n | (1 << i))',
      'if(n << i)',
    ],
    correctAnswer: 1,
    explanation:
      'To check if the i-th bit is set, use `n & (1 << i)`. Explanation: `1 << i` creates a number with only the i-th bit set (e.g., 1 << 3 = 1000 in binary = 8). AND-ing with n isolates that bit. If result is non-zero, bit is set. Example: n=12 (1100), check bit 2: `12 & (1 << 2)` = `1100 & 0100` = `0100` (non-zero, so bit 2 is set). Related operations: Set bit i: `n | (1 << i)`, Clear bit i: `n & ~(1 << i)`, Toggle bit i: `n ^ (1 << i)`. These are fundamental for bitmask DP, subset enumeration, and many other algorithms.',
    difficulty: 'intermediate',
  },
  {
    id: 'cp-m1-s9-q3',
    section: 'Bits, Bytes & Bitwise Operations',
    question: 'What is the fastest way to check if a number n is a power of 2?',
    options: [
      'Repeatedly divide by 2 until reaching 1',
      'Check if n & (n-1) == 0 and n != 0',
      "Calculate log2(n) and check if it's an integer",
      'Check if n % 2 == 0',
    ],
    correctAnswer: 1,
    explanation:
      'Powers of 2 have exactly one bit set in binary (1, 2, 4, 8 = 1, 10, 100, 1000). Using `n & (n-1)` removes the rightmost set bit. If n is a power of 2, removing its only bit gives 0. Code: `(n & (n-1)) == 0 && n != 0`. The n != 0 check is necessary because 0 & (0-1) = 0, but 0 is not a power of 2. This is O(1) - single operation! Dividing repeatedly is O(log n), logarithm calculation risks floating-point errors, n % 2 == 0 only checks if even (not if power of 2). This trick is essential for problems involving powers of 2.',
    difficulty: 'intermediate',
  },
  {
    id: 'cp-m1-s9-q4',
    section: 'Bits, Bytes & Bitwise Operations',
    question:
      'What does the XOR operation (^) between two identical numbers produce?',
    options: [
      'The number doubled',
      'Zero',
      'The number itself',
      'All bits flipped',
    ],
    correctAnswer: 1,
    explanation:
      'XOR of identical numbers is always 0. Example: 5 ^ 5 = 0101 ^ 0101 = 0000 = 0. XOR properties: (1) a ^ a = 0, (2) a ^ 0 = a, (3) XOR is commutative and associative. This is useful for: Finding the unique element when all others appear twice: `result = 0; for(x : arr) result ^= x;` - all pairs cancel out, leaving only the unique element. Swapping without temp variable: `a ^= b; b ^= a; a ^= b;`. Checking if two numbers are equal: `a ^ b == 0`. XOR is also the basis for many cryptographic and error-detection algorithms.',
    difficulty: 'beginner',
  },
  {
    id: 'cp-m1-s9-q5',
    section: 'Bits, Bytes & Bitwise Operations',
    question:
      'How do you iterate through all subsets of a set represented as a bitmask?',
    options: [
      'for(int mask = 0; mask < (1 << n); mask++)',
      'for(int mask = 1; mask <= n; mask++)',
      'for(int mask = 0; mask <= (1 << n); mask *= 2)',
      'for(int mask = n; mask >= 0; mask--)',
    ],
    correctAnswer: 0,
    explanation:
      'To iterate through all 2^n subsets of an n-element set, use `for(int mask = 0; mask < (1 << n); mask++)`. Each number from 0 to 2^n-1 represents a subset (bit i indicates if element i is included). Example: n=3, subsets 0-7 represent {}, {0}, {1}, {0,1}, {2}, {0,2}, {1,2}, {0,1,2}. This is fundamental for: (1) Bitmask DP, (2) Subset enumeration problems, (3) Traveling Salesman Problem. To iterate through all subsets of a subset (submask iteration): `for(int sub = mask; sub > 0; sub = (sub-1) & mask)`. Time complexity: O(2^n) for all subsets. Essential technique for many CP problems.',
    difficulty: 'intermediate',
  },
] as const;
