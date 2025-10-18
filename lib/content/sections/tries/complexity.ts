/**
 * Complexity Analysis Section
 */

export const complexitySection = {
  id: 'complexity',
  title: 'Complexity Analysis',
  content: `**Time Complexity:**

| Operation | Trie | Hash Table | Binary Search Tree |
|-----------|------|------------|-------------------|
| Insert | O(m) | O(m) avg | O(m log n) |
| Search | O(m) | O(m) avg | O(m log n) |
| Prefix Search | O(m + k) | O(n * m) | O(m log n + k) |
| Delete | O(m) | O(m) avg | O(m log n) |

Where:
- **m** = length of string
- **k** = number of strings with prefix
- **n** = number of strings stored

**Space Complexity:**

**Worst Case**: O(ALPHABET_SIZE * m * n)
- Each of n words, length m
- Each character needs array of size ALPHABET_SIZE

**Best Case**: O(m * n)
- All words share maximum common prefix

**Typical Case**: Depends on:
- Alphabet size (26 for lowercase, 256 for ASCII)
- Number of shared prefixes
- Storage method (array vs hash map)

---

**Space Optimization Techniques:**

**1. Use Hash Map Instead of Array**
\`\`\`python
children = {}  # Only store existing children
# vs
children = [None] * 26  # Allocate all 26 slots
\`\`\`

**2. Compressed Trie**
Merge single-child chains into one node.

**3. Bit-packed Storage**
Use bit manipulation for compact storage.

**4. Reference Counting**
Share nodes between tries where possible.

---

**Comparison Summary:**

**Trie Wins When:**
- Many prefix searches
- Autocomplete functionality
- Shared prefixes (space efficient)
- Need sorted order

**Hash Table Wins When:**
- Only exact searches
- Limited memory
- Random access patterns
- Simple implementation

**BST Wins When:**
- Need range queries
- Ordered iteration important
- Balanced operations`,
};
