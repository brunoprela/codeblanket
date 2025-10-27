/**
 * Find Prime Numbers
 * Problem ID: fundamentals-prime-numbers
 * Order: 5
 */

import { Problem } from '../../../types';

export const prime_numbersProblem: Problem = {
  id: 'fundamentals-prime-numbers',
  title: 'Find Prime Numbers',
  difficulty: 'Easy',
  description: `Find all prime numbers up to a given number n using the Sieve of Eratosthenes.

**Prime Number:** A natural number greater than 1 that has no positive divisors other than 1 and itself.

**Sieve of Eratosthenes:**1. Create a list of consecutive integers from 2 to n
2. Start with the first number (2)
3. Mark all multiples of that number (except the number itself) as composite
4. Move to the next unmarked number and repeat

Return a list of all prime numbers up to and including n.`,
  examples: [
    {
      input: 'n = 10',
      output: '[2, 3, 5, 7]',
    },
  ],
  constraints: ['2 <= n <= 10^6', 'Return sorted list of primes'],
  hints: [
    'Create a boolean array to mark primes',
    'Start marking from 2',
    'Only check up to sqrt(n)',
  ],
  starterCode: `def find_primes(n):
    """
    Find all prime numbers up to n using Sieve of Eratosthenes.
    
    Args:
        n: Upper limit (inclusive)
        
    Returns:
        List of prime numbers up to n
        
    Examples:
        >>> find_primes(10)
        [2, 3, 5, 7]
    """
    pass


# Test
print(find_primes(30))
`,
  testCases: [
    {
      input: [10],
      expected: [2, 3, 5, 7],
    },
    {
      input: [20],
      expected: [2, 3, 5, 7, 11, 13, 17, 19],
    },
  ],
  solution: `def find_primes(n):
    if n < 2:
        return []
    
    # Create boolean array, initially all True
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    
    # Sieve of Eratosthenes
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            # Mark multiples of i as not prime
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    
    # Collect all prime numbers
    return [i for i in range(n + 1) if is_prime[i]]


# Simple approach (less efficient for large n)
def find_primes_simple(n):
    def is_prime(num):
        if num < 2:
            return False
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                return False
        return True
    
    return [i for i in range(2, n + 1) if is_prime(i)]`,
  timeComplexity: 'O(n log log n) - Sieve of Eratosthenes',
  spaceComplexity: 'O(n)',
  order: 5,
  topic: 'Python Fundamentals',
};
