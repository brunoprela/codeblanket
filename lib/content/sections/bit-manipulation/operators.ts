/**
 * Bitwise Operators Section
 */

export const operatorsSection = {
  id: 'operators',
  title: 'Bitwise Operators',
  content: `**Core Bitwise Operators:**

**1. AND (&)**
- Result is 1 only if both bits are 1
- Use: Testing if bits are set, masking
\`\`\`
  1010  (10)
& 1100  (12)
------
  1000  (8)
\`\`\`

**2. OR (|)**
- Result is 1 if at least one bit is 1
- Use: Setting bits, combining flags
\`\`\`
  1010  (10)
| 1100  (12)
------
  1110  (14)
\`\`\`

**3. XOR (^)**
- Result is 1 if bits are different
- Use: Toggling bits, finding unique elements
\`\`\`
  1010  (10)
^ 1100  (12)
------
  0110  (6)
\`\`\`

**4. NOT (~)**
- Flips all bits (0→1, 1→0)
- Use: Inverting bits
\`\`\`
~1010 = 0101 (in 4-bit system)
\`\`\`

**5. Left Shift (<<)**
- Shifts bits left, fills with 0
- Effect: Multiplies by 2^n
\`\`\`
5 << 2 = 20
0101 << 2 = 10100
\`\`\`

**6. Right Shift (>>)**
- Shifts bits right, fills with sign bit
- Effect: Divides by 2^n (integer division)
\`\`\`
20 >> 2 = 5
10100 >> 2 = 00101
\`\`\``,
};
