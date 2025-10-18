/**
 * XOR Properties and Applications Section
 */

export const xorpropertiesSection = {
  id: 'xor-properties',
  title: 'XOR Properties and Applications',
  content: `**XOR (Exclusive OR) is a superstar operator** with unique properties that make it invaluable for many problems.

**Key XOR Properties:**

**1. Self-cancellation:**
\`\`\`python
a ^ a = 0  # Any number XOR itself equals 0
\`\`\`

**2. Identity:**
\`\`\`python
a ^ 0 = a  # Any number XOR 0 equals itself
\`\`\`

**3. Commutative:**
\`\`\`python
a ^ b = b ^ a
\`\`\`

**4. Associative:**
\`\`\`python
(a ^ b) ^ c = a ^ (b ^ c)
\`\`\`

**5. Self-inverse:**
\`\`\`python
a ^ b ^ b = a  # XOR twice with same value cancels out
\`\`\`

**Classic XOR Applications:**

**Finding Single Element:**
When every element appears twice except one, XOR all elements. Duplicates cancel to 0, leaving the unique element.

**Swapping Without Temp Variable:**
\`\`\`python
a ^= b  # a = a ^ b
b ^= a  # b = b ^ (a ^ b) = a
a ^= b  # a = (a ^ b) ^ a = b
\`\`\`

**Missing Number:**
XOR all indices and all array values. The missing index won't have a pair to cancel with.

**Parity Check:**
XOR all bits to check if count of 1s is odd or even.`,
};
