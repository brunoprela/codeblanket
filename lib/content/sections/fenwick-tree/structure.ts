/**
 * Fenwick Tree Structure Section
 */

export const structureSection = {
  id: 'structure',
  title: 'Fenwick Tree Structure',
  content: `**How it Works:**
Each index in BIT stores sum of a range of elements, not just one element.

**Index Responsibility:**
Index \`i\` stores sum of \`r(i)\` elements, where \`r(i)\` is the position of the last set bit.

**Example:** Array = [3, 2, -1, 6, 5, 4, -3, 3, 7, 2, 3]

\`\`\`
Index (binary)  Stores sum of
1  (0001)       arr[1]           (1 element)
2  (0010)       arr[1..2]        (2 elements)
3  (0011)       arr[3]           (1 element)
4  (0100)       arr[1..4]        (4 elements)
5  (0101)       arr[5]           (1 element)
6  (0110)       arr[5..6]        (2 elements)
7  (0111)       arr[7]           (1 element)
8  (1000)       arr[1..8]        (8 elements)
\`\`\`

**Key Operations:**

**1. Get Last Set Bit (LSB):**
\`\`\`python
def lsb(i):
    return i & (-i)
# Examples:
# 6 (110) & -6 (010) = 2 (010)
# 12 (1100) & -12 (0100) = 4 (0100)
\`\`\`

**2. Parent (for update):**
\`\`\`python
def parent(i):
    return i + (i & -i)
# Move to next index that needs updating
\`\`\`

**3. Prefix (for query):**
\`\`\`python
def prefix_parent(i):
    return i - (i & -i)
# Move to previous range to sum
\`\`\``,
};
