/**
 * Quiz questions for Number Theory Fundamentals section
 */

export const numbertheoryQuiz = [
  {
    id: 'q1',
    question:
      'Explain prime number checking. Compare trial division vs Sieve of Eratosthenes approaches.',
    sampleAnswer:
      'Prime: number divisible only by 1 and itself. Trial division: check if n divisible by any number from 2 to sqrt(n). O(sqrt(n)) per check. Sieve of Eratosthenes: find all primes up to n by marking multiples. O(n log log n) for all primes up to n. Use trial division when: checking single number, n is small. Use Sieve when: need many primes, precompute up to limit. For example, check if 17 prime: test 2,3,4 (stop at sqrt(17)≈4) - none divide, so prime. For primes up to 100: Sieve marks multiples of 2 (4,6,8,...), then 3 (6,9,12,...), etc. Remaining unmarked are primes. Trial division: O(sqrt(n)) per check. Sieve: O(n log log n) total for all primes ≤ n. For multiple queries, Sieve precomputation wins.',
    keyPoints: [
      'Prime: divisible by 1 and self only',
      'Trial division: O(sqrt(n)) per check',
      'Sieve: O(n log log n) for all primes ≤ n',
      'Trial when: single check, small n',
      'Sieve when: multiple queries, precompute',
    ],
  },
  {
    id: 'q2',
    question:
      'Describe GCD (Greatest Common Divisor) and Euclidean algorithm. Why is it efficient?',
    sampleAnswer:
      'GCD: largest number dividing both a and b. Euclidean algorithm: gcd(a, b) = gcd(b, a % b), base case gcd(a, 0) = a. Works because: any divisor of a and b also divides (a - kb) for any k, including a % b. Repeat until remainder 0. For example, gcd(48, 18): 48 % 18 = 12, gcd(18, 12); 18 % 12 = 6, gcd(12, 6); 12 % 6 = 0, gcd(6, 0) = 6. Time complexity: O(log min(a,b)) - each step reduces numbers by at least half (Fibonacci numbers are worst case). Space: O(1) iterative or O(log n) recursive. Why efficient? Much faster than factorization. Used for: simplifying fractions, LCM (lcm = a*b/gcd), modular inverse. This is one of oldest algorithms (300 BC).',
    keyPoints: [
      'GCD: largest divisor of both numbers',
      'Euclidean: gcd(a, b) = gcd(b, a % b)',
      'O(log min(a,b)) - very fast',
      'Each step reduces by at least half',
      'Uses: fractions, LCM, modular inverse',
    ],
  },
  {
    id: 'q3',
    question:
      'Explain fast power (exponentiation by squaring). Why is it O(log n) instead of O(n)?',
    sampleAnswer:
      'Fast power computes a^n efficiently by squaring. Key insight: a^n = (a^(n/2))^2 if n even, a^n = a × (a^((n-1)/2))^2 if n odd. Instead of n multiplications (a×a×...×a), do log n by repeatedly squaring. For example, 2^10: 2^10 = (2^5)^2, 2^5 = 2×(2^2)^2, 2^2 = (2^1)^2, 2^1 = 2. Total 4 multiplications vs 10. With modulo (for large results): compute (a^n) % m by taking modulo at each step to keep numbers bounded. Algorithm: start with result=1, while n>0: if n odd, result *= a; a *= a; n /= 2. Why O(log n)? Each iteration halves n. Used for: large exponents, modular exponentiation (RSA cryptography), matrix exponentiation (Fibonacci).',
    keyPoints: [
      'Compute a^n by repeated squaring',
      'O(log n) vs O(n) naive multiplication',
      'Even: (a^(n/2))^2, Odd: a × (a^((n-1)/2))^2',
      'With modulo: keep numbers bounded',
      'Uses: large exponents, crypto, Fibonacci',
    ],
  },
];
