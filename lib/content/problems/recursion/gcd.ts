/**
 * Greatest Common Divisor (GCD)
 * Problem ID: recursion-gcd
 * Order: 8
 */

import { Problem } from '../../../types';

export const gcdProblem: Problem = {
  id: 'recursion-gcd',
  title: 'Greatest Common Divisor (GCD)',
  difficulty: 'Easy',
  topic: 'Recursion',
  description: `Find the greatest common divisor (GCD) of two positive integers using Euclidean algorithm with recursion.

**Euclidean Algorithm:**
- GCD(a, b) = GCD(b, a mod b)
- Base case: GCD(a, 0) = a

**Examples:**
- GCD(48, 18) = 6
- GCD(100, 50) = 50
- GCD(7, 3) = 1

This is one of the oldest and most elegant recursive algorithms!`,
  examples: [
    { input: 'a = 48, b = 18', output: '6' },
    { input: 'a = 100, b = 50', output: '50' },
    { input: 'a = 7, b = 3', output: '1' },
  ],
  constraints: ['1 <= a, b <= 10⁹'],
  hints: [
    'Base case: if b is 0, return a',
    'Recursive case: gcd(a, b) = gcd(b, a % b)',
    'The algorithm always terminates because remainder gets smaller',
    'Works because GCD(a,b) = GCD(b, a mod b)',
  ],
  starterCode: `def gcd(a, b):
    """
    Calculate GCD using Euclidean algorithm with recursion.
    
    Args:
        a: First positive integer
        b: Second positive integer
        
    Returns:
        Greatest common divisor of a and b
        
    Examples:
        >>> gcd(48, 18)
        6
        >>> gcd(100, 50)
        50
    """
    pass


# Test cases
print(gcd(48, 18))   # Expected: 6
print(gcd(100, 50))  # Expected: 50
`,
  testCases: [
    { input: [48, 18], expected: 6 },
    { input: [100, 50], expected: 50 },
    { input: [7, 3], expected: 1 },
    { input: [1, 1], expected: 1 },
    { input: [20, 30], expected: 10 },
  ],
  solution: `def gcd(a, b):
    """GCD using Euclidean algorithm"""
    # Base case: when b is 0, GCD is a
    if b == 0:
        return a
    
    # Recursive case: GCD(a, b) = GCD(b, a mod b)
    return gcd(b, a % b)


# Example trace for gcd(48, 18):
# gcd(48, 18) = gcd(18, 48 % 18) = gcd(18, 12)
# gcd(18, 12) = gcd(12, 18 % 12) = gcd(12, 6)
# gcd(12, 6)  = gcd(6, 12 % 6)   = gcd(6, 0)
# gcd(6, 0)   = 6 ✓

# Time Complexity: O(log(min(a,b)))
# Space Complexity: O(log(min(a,b))) - call stack`,
  timeComplexity: 'O(log(min(a,b)))',
  spaceComplexity: 'O(log(min(a,b)))',
  followUp: [
    'How would you calculate LCM using GCD?',
    'Why does this algorithm work?',
    'Can you implement this iteratively?',
  ],
};
