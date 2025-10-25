/**
 * Trie Implementation Section
 */

export const implementationSection = {
  id: 'implementation',
  title: 'Trie Implementation',
  content: `**Basic Trie Node Structure:**

\`\`\`python
class TrieNode:
    def __init__(self):
        self.children = {}  # or [None] * 26 for lowercase letters
        self.is_end_of_word = False
\`\`\`

**Two Storage Approaches:**

**1. Hash Map (Dictionary)**
\`\`\`python
children = {}  # Flexible, any character
\`\`\`
**Pros**: Handles any character set, space efficient for sparse data
**Cons**: Slightly slower lookup (O(1) average vs O(1) guaranteed)

**2. Fixed Array**
\`\`\`python
children = [None] * 26  # Only lowercase a-z
\`\`\`
**Pros**: O(1) guaranteed lookup, cache-friendly
**Cons**: Wastes space if sparse, limited to fixed alphabet

---

**Complete Trie Class:**

\`\`\`python
class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert (self, word: str) -> None:
        """Insert a word into the trie."""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
    
    def search (self, word: str) -> bool:
        """Check if exact word exists."""
        node = self._search_prefix (word)
        return node is not None and node.is_end_of_word
    
    def starts_with (self, prefix: str) -> bool:
        """Check if any word starts with prefix."""
        return self._search_prefix (prefix) is not None
    
    def _search_prefix (self, prefix: str) -> TrieNode:
        """Helper: traverse to end of prefix."""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node
\`\`\`

---

**Operation Complexities:**

| Operation | Time | Space |
|-----------|------|-------|
| Insert | O(m) | O(m) |
| Search | O(m) | O(1) |
| Prefix Search | O(m) | O(1) |
| Delete | O(m) | O(1) |

Where **m** = length of string

---

**Memory Analysis:**

For **n** words with average length **m**:
- **Worst case**: O(ALPHABET_SIZE * m * n)
  - Each character has array of size 26
- **Best case**: O(m * n)
  - All words share common prefix
- **Average case**: Somewhere in between`,
};
