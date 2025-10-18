/**
 * Count Primes Below N
 * Problem ID: fundamentals-count-primes
 * Order: 68
 */

import { Problem } from '../../../types';

export const count_primesProblem: Problem = {
  id: 'fundamentals-count-primes',
  title: 'Count Primes Below N',
  difficulty: 'Medium',
  description: `Count the number of prime numbers less than n.

Use Sieve of Eratosthenes for efficient solution.

**Example:** n = 10 → 4 primes (2, 3, 5, 7)

This tests:
- Sieve algorithm
- Boolean array
- Prime generation`,
  examples: [
    {
      input: 'n = 10',
      output: '4',
      explanation: '2, 3, 5, 7',
    },
    {
      input: 'n = 0',
      output: '0',
    },
  ],
  constraints: ['0 <= n <= 5*10^6'],
  hints: [
    'Use Sieve of Eratosthenes',
    'Mark multiples of each prime',
    'Count unmarked numbers',
  ],
  starterCode: `def count_primes(n):
    """
    Count primes less than n.
    
    Args:
        n: Upper bound (exclusive)
        
    Returns:
        Count of primes < n
        
    Examples:
        >>> count_primes(10)
        4
    """
    pass


# Test
print(count_primes(10))
`,
  testCases: [
    {
      input: [10],
      expected: 4,
    },
    {
      input: [0],
      expected: 0,
    },
    {
      input: [1],
      expected: 0,
    },
  ],
  solution: `def count_primes(n):
    if n <= 2:
        return 0
    
    is_prime = [True] * n
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            # Mark multiples as not prime
            for j in range(i * i, n, i):
                is_prime[j] = False
    
    return sum(is_prime)`,
  timeComplexity: 'O(n log log n)',
  spaceComplexity: 'O(n)',
  order: 68,
  topic: 'Python Fundamentals',
};
