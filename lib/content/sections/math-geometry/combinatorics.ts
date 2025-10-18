/**
 * Combinatorics and Sequences Section
 */

export const combinatoricsSection = {
  id: 'combinatorics',
  title: 'Combinatorics and Sequences',
  content: `**Common Combinatorial Concepts:**

**1. Permutations:**
- Order matters
- n! = n × (n-1) × ... × 1
- nPr = n!/(n-r)! for r items from n

**2. Combinations:**
- Order doesn't matter
- nCr = n!/(r!(n-r)!)
- Pascal's triangle: C(n,r) = C(n-1,r-1) + C(n-1,r)

**3. Fibonacci Sequence:**
- F(n) = F(n-1) + F(n-2)
- F(0) = 0, F(1) = 1
- Many DP problems use Fibonacci pattern

**4. Factorials:**
- n! grows very fast
- Often need modular arithmetic
- Trailing zeros in n! = count of 5s in factors

**5. Catalan Numbers:**
- C(n) = (2n)!/(n!(n+1)!)
- Count valid parentheses, BSTs, etc.
- C(n) = C(0)C(n-1) + C(1)C(n-2) + ... + C(n-1)C(0)`,
};
