/**
 * Number Theory Fundamentals Section
 */

export const numbertheorySection = {
  id: 'number-theory',
  title: 'Number Theory Fundamentals',
  content: `**Essential Number Theory Concepts:**

**1. Prime Numbers**
- Only divisible by 1 and itself
- Sieve of Eratosthenes for finding all primes up to n

**2. GCD (Greatest Common Divisor)**
- Euclidean Algorithm: O(log (min (a,b)))
\`\`\`python
def gcd (a, b):
    while b:
        a, b = b, a % b
    return a
\`\`\`

**3. LCM (Least Common Multiple)**
- Formula: LCM(a,b) = (a × b) / GCD(a,b)

**4. Factorization**
- Break number into prime factors
- Check divisors up to sqrt (n)

**5. Power and Exponentiation**
- Fast exponentiation: O(log n)
- Use for large powers efficiently

**6. Modular Arithmetic**
- (a + b) % m = ((a % m) + (b % m)) % m
- (a × b) % m = ((a % m) × (b % m)) % m
- Important for handling large numbers`,
};
