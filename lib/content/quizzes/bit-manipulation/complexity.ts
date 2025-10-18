/**
 * Quiz questions for Time and Space Complexity section
 */

export const complexityQuiz = [
  {
    id: 'q1',
    question:
      'Analyze the time and space complexity of bit manipulation operations. Why are they considered fast?',
    sampleAnswer:
      'Individual bit operations (&, |, ^, ~, <<, >>) are O(1) time and O(1) space - single CPU instructions. Check/set/clear/toggle bit: O(1). Count set bits with n & (n-1): O(k) where k is set bits (better than O(log n) for sparse). Subset generation: O(2^n) to iterate all, O(2^n × n) to store. Bit masking DP: O(2^n × ...) for state space (exponential but compact). Lookup tables: O(1) query with O(2^k) space for k-bit chunks. Why fast? 1) Hardware support - CPU does in one cycle. 2) No memory access - registers only. 3) Predictable - no branches. For example, check even with n & 1 vs n % 2: both O(1) but bitwise is single AND instruction while modulo can be multiple instructions.',
    keyPoints: [
      'Bit operations: O(1) time, single CPU instruction',
      'n & (n-1): O(k) where k = set bits',
      'Subset generation: O(2^n) exponential',
      'Fast: hardware support, no memory, no branches',
      'vs Arithmetic: fewer CPU cycles',
    ],
  },
  {
    id: 'q2',
    question:
      'Compare space complexity of bit manipulation vs alternative approaches for flag storage.',
    sampleAnswer:
      'Bit manipulation stores 32 flags in single 32-bit int (4 bytes). Alternative: 32 boolean variables (32 bytes). Space saving: 8x for 32 flags. For n flags: bit manipulation ⌈n/32⌉ ints, booleans n bytes. Example: 1000 flags: bits need 32 ints (128 bytes), booleans need 1000 bytes. Trade-off: bit manipulation saves space but less readable. Best for: embedded systems (limited RAM), large datasets (millions of flags), performance-critical code. Not worth for: few flags (< 32), readability matters, premature optimization. Example: Linux file permissions (rwxrwxrwx) stored in 9 bits vs 9 booleans. Game entity flags: 32 states in one int vs 32 variables. Modern systems have lots of memory, prefer readability unless proven bottleneck.',
    keyPoints: [
      'Bit: 32 flags in 4 bytes (32-bit int)',
      'Boolean array: n flags in n bytes',
      'Space saving: 8x for 32 flags',
      'Use when: embedded, large datasets, proven bottleneck',
      'vs Readability: optimize only when necessary',
    ],
  },
  {
    id: 'q3',
    question:
      'Explain complexity of bitmask DP. When is exponential state space acceptable?',
    sampleAnswer:
      'Bitmask DP uses integer to represent state (each bit = element included/excluded). State space: 2^n for n elements. For example, TSP with n cities: O(n^2 × 2^n) time, O(n × 2^n) space. Exponential but: 1) Compact - 2^20 ≈ 1M states fits in memory. 2) Fast bit ops for state transitions. 3) Often best known algorithm (TSP, job assignment). Acceptable when n is small (n ≤ 20-22). For n=20: 2^20 × 20 = 20M operations - feasible. For n=30: 2^30 × 30 = 30B operations - too slow. Alternative: brute force is O(n!) which is worse (20! = 2.4×10^18). Bitmask DP trades exponential space for feasible solution. Used: TSP, subset sum, assignment problems. Key: recognize when n is small enough.',
    keyPoints: [
      'Bitmask DP: O(2^n) state space',
      'Compact: 2^20 ≈ 1M states fits memory',
      'Acceptable for n ≤ 20-22',
      'Often better than brute force O(n!)',
      'Uses: TSP, job assignment, subset problems',
    ],
  },
];
