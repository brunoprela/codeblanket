/**
 * Common Bit Manipulation Patterns Section
 */

export const commonpatternsSection = {
  id: 'common-patterns',
  title: 'Common Bit Manipulation Patterns',
  content: `**Essential Bit Manipulation Techniques:**

**1. Check if bit i is set:**
\`\`\`python
(n & (1 << i)) != 0
\`\`\`

**2. Set bit i:**
\`\`\`python
n | (1 << i)
\`\`\`

**3. Clear bit i:**
\`\`\`python
n & ~(1 << i)
\`\`\`

**4. Toggle bit i:**
\`\`\`python
n ^ (1 << i)
\`\`\`

**5. Check if power of 2:**
\`\`\`python
n > 0 and (n & (n - 1)) == 0
\`\`\`
Why? Powers of 2 have exactly one bit set. n-1 flips all bits after that bit, so AND gives 0.

**6. Get rightmost set bit:**
\`\`\`python
n & -n
\`\`\`
Why? -n is the two's complement (flip bits and add 1), isolating the rightmost 1.

**7. Remove rightmost set bit:**
\`\`\`python
n & (n - 1)
\`\`\`
Why? n-1 flips all bits from rightmost 1, AND removes it.

**8. Count set bits (Brian Kernighan's algorithm):**
\`\`\`python
count = 0
while n:
    n &= (n - 1)  # Remove rightmost set bit
    count += 1
\`\`\``,
};
