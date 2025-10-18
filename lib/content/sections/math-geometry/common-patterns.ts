/**
 * Common Algorithm Patterns Section
 */

export const commonpatternsSection = {
  id: 'common-patterns',
  title: 'Common Algorithm Patterns',
  content: `**Pattern Recognition:**

**1. Happy Number / Cycle Detection:**
- Use Floyd's cycle detection (two pointers)
- Transform number repeatedly until cycle or target

**2. Power Function:**
- Fast exponentiation: O(log n)
- Handle negative exponents
- Consider overflow

**3. Digit Manipulation:**
- Extract digits: n % 10, n //= 10
- Reverse number: build from digits
- Sum of digit squares: for happy number

**4. Sieve Algorithms:**
- Sieve of Eratosthenes for primes
- Mark multiples as composite
- O(n log log n) time

**5. Mathematical Optimization:**
- Use math properties to reduce brute force
- Example: Check divisors only up to sqrt(n)
- Use formulas instead of loops when possible

**6. Modular Arithmetic:**
- Keep numbers bounded
- (a + b) % m = ((a % m) + (b % m)) % m
- Useful for large number problems`,
};
