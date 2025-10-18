/**
 * Advanced Trie Techniques Section
 */

export const advancedSection = {
  id: 'advanced',
  title: 'Advanced Trie Techniques',
  content: `**Technique 1: Compressed Trie (Radix Tree)**

Merge nodes with single child to save space.

\`\`\`
Normal Trie:        Compressed Trie:
    c                   c
    |                   |
    a                   ar
   / \\                 /  \\
  r   t               (t)  (*)
 (*)                 

Saves space by storing "ar" instead of "a" -> "r"
\`\`\`

---

**Technique 2: Suffix Trie**

Build trie of all suffixes for pattern matching.

\`\`\`python
def build_suffix_trie(text: str):
    trie = Trie()
    for i in range(len(text)):
        trie.insert(text[i:])
    return trie

# Check if pattern exists in text
def contains_pattern(text, pattern):
    suffix_trie = build_suffix_trie(text)
    return suffix_trie.starts_with(pattern)
\`\`\`

---

**Technique 3: Ternary Search Tree**

Space-efficient alternative to trie.

Each node has 3 children: less, equal, greater.

\`\`\`python
class TSTNode:
    def __init__(self, char):
        self.char = char
        self.left = None   # char < node.char
        self.mid = None    # char == node.char (next)
        self.right = None  # char > node.char
        self.is_end = False
\`\`\`

**Advantages:**
- Uses less memory than standard trie
- Faster than hash table for small alphabets
- Supports ordered operations

---

**Technique 4: Persistent Trie**

Maintain multiple versions without copying entire structure.

**Use case**: Version control, undo/redo operations

---

**Technique 5: XOR Trie**

Store integers in binary trie for maximum XOR queries.

\`\`\`python
class XORTrie:
    def __init__(self):
        self.root = {}
    
    def insert(self, num):
        node = self.root
        for i in range(31, -1, -1):
            bit = (num >> i) & 1
            if bit not in node:
                node[bit] = {}
            node = node[bit]
    
    def max_xor(self, num):
        node = self.root
        xor = 0
        for i in range(31, -1, -1):
            bit = (num >> i) & 1
            # Try opposite bit for max XOR
            opposite = 1 - bit
            if opposite in node:
                xor |= (1 << i)
                node = node[opposite]
            else:
                node = node[bit]
        return xor
\`\`\``,
};
