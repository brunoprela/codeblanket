/**
 * Time and Space Complexity Section
 */

export const complexitySection = {
  id: 'complexity',
  title: 'Time and Space Complexity',
  content: `**Common Complexities:**

**Number Theory:**
- GCD (Euclidean): O(log(min(a,b)))
- Prime check: O(sqrt(n))
- Sieve of Eratosthenes: O(n log log n)
- Factorization: O(sqrt(n))
- Fast exponentiation: O(log n)

**Matrix Operations:**
- Matrix traversal: O(m × n)
- Matrix transpose: O(m × n)
- Matrix rotation in-place: O(m × n), O(1) space
- Spiral traversal: O(m × n)

**Geometry:**
- Distance calculation: O(1)
- Point in polygon: O(n) for n vertices
- Convex hull (Graham scan): O(n log n)
- Line intersection: O(1)

**Combinatorics:**
- Factorial: O(n)
- Combinations (iterative): O(min(r, n-r))
- Fibonacci (iterative): O(n), O(1) space
- Fibonacci (matrix): O(log n)
- Catalan number: O(n²) with DP

**Space Optimization:**
- Most math problems can be solved with O(1) space
- Use iterative over recursive when possible
- Reuse variables instead of arrays`,
};
