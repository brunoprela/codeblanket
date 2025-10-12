import { Module } from '@/lib/types';

export const triesModule: Module = {
  id: 'tries',
  title: 'Tries',
  description:
    'Master the prefix tree data structure for efficient string operations and searches.',
  icon: '🌲',
  timeComplexity: 'O(m) where m is string length',
  spaceComplexity: 'O(ALPHABET_SIZE * m * n)',
  sections: [
    {
      id: 'introduction',
      title: 'Introduction to Tries',
      content: `A **Trie** (pronounced "try") is a tree-like data structure used to store strings efficiently. Also called a **prefix tree** or **digital tree**, it's particularly useful for string searches and prefix matching.

**Key Characteristics:**

- Each node represents a single character
- Root node is empty
- Path from root to node = string prefix
- Leaf nodes (or marked nodes) = complete words
- Children typically stored in array or hash map

**Trie Structure:**

\`\`\`
Example: Insert "cat", "car", "dog"

        (root)
       /      \\
      c        d
      |        |
      a        o
     / \\       |
    t   r      g
   (*)  (*)   (*)

(*) = end of word marker
\`\`\`

**Why Use Tries?**

**Advantages:**
- **Fast prefix searches**: O(m) where m = string length
- **Autocomplete**: Find all words with prefix
- **Spell checking**: Fast dictionary lookups
- **Space efficient**: Shared prefixes (e.g., "cat", "car" share "ca")
- **No hash collisions**: Unlike hash tables

**Disadvantages:**
- **Space overhead**: Each node needs storage for all possible children
- **Cache locality**: Tree structure less cache-friendly than arrays
- **Memory**: Can use more memory than hash tables for sparse data

---

**Common Use Cases:**

1. **Autocomplete / Type-ahead**
   - Google search suggestions
   - IDE code completion

2. **Spell Checker**
   - Dictionary lookup
   - Suggest corrections

3. **IP Routing**
   - Longest prefix matching
   - Network routing tables

4. **Phone Directory**
   - Search by name prefix
   - T9 predictive text

5. **Word Games**
   - Boggle solver
   - Scrabble word validation`,
    },
    {
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
    
    def insert(self, word: str) -> None:
        """Insert a word into the trie."""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
    
    def search(self, word: str) -> bool:
        """Check if exact word exists."""
        node = self._search_prefix(word)
        return node is not None and node.is_end_of_word
    
    def starts_with(self, prefix: str) -> bool:
        """Check if any word starts with prefix."""
        return self._search_prefix(prefix) is not None
    
    def _search_prefix(self, prefix: str) -> TrieNode:
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
      codeExample: `class TrieNode:
    """Node in the Trie structure."""
    
    def __init__(self):
        self.children = {}  # char -> TrieNode
        self.is_end_of_word = False


class Trie:
    """Prefix tree for efficient string operations."""
    
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word: str) -> None:
        """
        Insert a word into the trie.
        Time: O(m), Space: O(m) where m = len(word)
        """
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
    
    def search(self, word: str) -> bool:
        """
        Check if exact word exists in trie.
        Time: O(m), Space: O(1)
        """
        node = self._search_prefix(word)
        return node is not None and node.is_end_of_word
    
    def starts_with(self, prefix: str) -> bool:
        """
        Check if any word starts with given prefix.
        Time: O(m), Space: O(1)
        """
        return self._search_prefix(prefix) is not None
    
    def _search_prefix(self, prefix: str) -> TrieNode:
        """
        Helper to traverse to end of prefix.
        Returns node at end of prefix, or None if not found.
        """
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node
    
    def delete(self, word: str) -> bool:
        """
        Delete a word from the trie.
        Time: O(m), Space: O(m) for recursion
        """
        def _delete_helper(node, word, depth):
            if depth == len(word):
                # Word found, unmark it
                if not node.is_end_of_word:
                    return False  # Word doesn't exist
                node.is_end_of_word = False
                # Delete node if it has no children
                return len(node.children) == 0
            
            char = word[depth]
            if char not in node.children:
                return False  # Word doesn't exist
            
            child = node.children[char]
            should_delete_child = _delete_helper(child, word, depth + 1)
            
            if should_delete_child:
                del node.children[char]
                # Delete current node if:
                # - It's not end of another word
                # - It has no other children
                return not node.is_end_of_word and len(node.children) == 0
            
            return False
        
        return _delete_helper(self.root, word, 0)
    
    def get_all_words_with_prefix(self, prefix: str) -> list:
        """
        Get all words in trie starting with prefix.
        Useful for autocomplete.
        """
        result = []
        node = self._search_prefix(prefix)
        
        if node is None:
            return result
        
        def dfs(node, path):
            if node.is_end_of_word:
                result.append(prefix + path)
            
            for char, child in node.children.items():
                dfs(child, path + char)
        
        dfs(node, "")
        return result`,
    },
    {
      id: 'patterns',
      title: 'Common Trie Patterns',
      content: `**Pattern 1: Standard Dictionary**

Most basic use: Insert words and search.

\`\`\`python
trie = Trie()
trie.insert("apple")
trie.insert("app")

trie.search("app")      # True
trie.search("appl")     # False
trie.starts_with("app") # True
\`\`\`

---

**Pattern 2: Autocomplete / Type-ahead**

Find all words with given prefix.

\`\`\`python
def autocomplete(trie, prefix):
    # Find node at end of prefix
    node = trie._search_prefix(prefix)
    if not node:
        return []
    
    # DFS to collect all words from here
    words = []
    
    def dfs(node, path):
        if node.is_end_of_word:
            words.append(prefix + path)
        for char, child in node.children.items():
            dfs(child, path + char)
    
    dfs(node, "")
    return words
\`\`\`

---

**Pattern 3: Word Search with Wildcards**

Support '.' as wildcard (matches any character).

\`\`\`python
def search_with_wildcard(self, word: str) -> bool:
    def dfs(node, i):
        if i == len(word):
            return node.is_end_of_word
        
        char = word[i]
        
        if char == '.':
            # Try all possible children
            for child in node.children.values():
                if dfs(child, i + 1):
                    return True
            return False
        else:
            if char not in node.children:
                return False
            return dfs(node.children[char], i + 1)
    
    return dfs(self.root, 0)
\`\`\`

---

**Pattern 4: Word Break Problem**

Check if string can be segmented into dictionary words.

\`\`\`python
def word_break(s: str, wordDict: list) -> bool:
    # Build trie from dictionary
    trie = Trie()
    for word in wordDict:
        trie.insert(word)
    
    # DP with trie
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True
    
    for i in range(1, n + 1):
        node = trie.root
        for j in range(i - 1, -1, -1):
            if not dp[j]:
                continue
            
            char = s[j]
            if char not in node.children:
                break
            
            node = node.children[char]
            if node.is_end_of_word:
                dp[i] = True
                break
    
    return dp[n]
\`\`\`

---

**Pattern 5: Longest Common Prefix**

Find longest prefix shared by all strings.

\`\`\`python
def longest_common_prefix(strs: list) -> str:
    if not strs:
        return ""
    
    trie = Trie()
    for s in strs:
        trie.insert(s)
    
    # Traverse until branch or end
    prefix = []
    node = trie.root
    
    while len(node.children) == 1 and not node.is_end_of_word:
        char = list(node.children.keys())[0]
        prefix.append(char)
        node = node.children[char]
    
    return "".join(prefix)
\`\`\`

---

**Pattern 6: Count Prefixes**

Count how many words start with each prefix.

\`\`\`python
class TrieNodeWithCount:
    def __init__(self):
        self.children = {}
        self.count = 0  # Number of words passing through
        self.is_end = False

def count_prefixes(words: list, prefixes: list) -> list:
    trie = Trie()
    
    # Insert with counts
    for word in words:
        node = trie.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNodeWithCount()
            node = node.children[char]
            node.count += 1
        node.is_end = True
    
    # Query counts
    result = []
    for prefix in prefixes:
        node = trie._search_prefix(prefix)
        result.append(node.count if node else 0)
    
    return result
\`\`\``,
    },
    {
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
    },
    {
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
    },
    {
      id: 'templates',
      title: 'Code Templates',
      content: `**Template 1: Basic Trie**
\`\`\`python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
    
    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end
\`\`\`

---

**Template 2: Trie with Prefix Search**
\`\`\`python
class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
    
    def get_words_with_prefix(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        result = []
        def dfs(node, path):
            if node.is_end:
                result.append(prefix + path)
            for char, child in node.children.items():
                dfs(child, path + char)
        
        dfs(node, "")
        return result
\`\`\`

---

**Template 3: Trie with Wildcard Search**
\`\`\`python
def search_wildcard(self, word):
    def dfs(node, i):
        if i == len(word):
            return node.is_end
        
        if word[i] == '.':
            for child in node.children.values():
                if dfs(child, i + 1):
                    return True
            return False
        
        if word[i] not in node.children:
            return False
        return dfs(node.children[word[i]], i + 1)
    
    return dfs(self.root, 0)
\`\`\`

---

**Template 4: Delete from Trie**
\`\`\`python
def delete(self, word):
    def _delete(node, word, depth):
        if depth == len(word):
            if not node.is_end:
                return False
            node.is_end = False
            return len(node.children) == 0
        
        char = word[depth]
        if char not in node.children:
            return False
        
        should_delete = _delete(node.children[char], word, depth + 1)
        
        if should_delete:
            del node.children[char]
            return not node.is_end and len(node.children) == 0
        
        return False
    
    _delete(self.root, word, 0)
\`\`\``,
    },
    {
      id: 'interview',
      title: 'Interview Strategy',
      content: `**Recognition Signals:**

**Use Trie when you see:**
- "Prefix" mentioned
- "Autocomplete" or "Type-ahead"
- "Dictionary" of words
- "Word search" in grid
- Multiple string searches
- "Longest common prefix"
- "Add and search with wildcards"

---

**Problem-Solving Steps:**

**Step 1: Identify Trie Need (2 min)**
- Multiple string operations?
- Prefix-related queries?
- Dictionary lookups?

**Step 2: Choose Structure (2 min)**
- Standard trie (hash map)?
- Array-based (fixed alphabet)?
- Need counts? Modify node

**Step 3: Define Node (3 min)**
\`\`\`python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        # Add other fields if needed:
        # self.count = 0
        # self.word = None
\`\`\`

**Step 4: Implement Operations (10 min)**
- Insert: Always needed
- Search: Exact or prefix?
- Delete: Rarely needed
- Custom: Problem-specific

**Step 5: Test Edge Cases (3 min)**
- Empty string
- Single character
- No matching prefix
- All strings identical

---

**Interview Communication:**

**Example: Implement Trie**

*Interviewer: Implement a trie with insert, search, and startsWith.*

**You:**

1. **Clarify:**
   - "Should I handle only lowercase letters?"
   - "Do we need to support deletion?"
   - "Any constraints on number of words or length?"

2. **Explain:**
   - "I'll use a hash map for children - flexible and space efficient."
   - "Each node needs a flag to mark end of word."
   - "Root node will be empty."

3. **Structure:**
   \`\`\`python
   class TrieNode:
       def __init__(self):
           self.children = {}  # char -> TrieNode
           self.is_end = False
   \`\`\`

4. **Operations:**
   - "Insert: Traverse/create path, mark end."
   - "Search: Traverse path, check is_end."
   - "StartsWith: Just traverse path."

5. **Complexity:**
   - "All operations are O(m) time where m is string length."
   - "Space is O(n * m * ALPHABET) worst case."

---

**Common Mistakes:**

**1. Forgetting is_end Flag**
\`\`\`python
# Wrong: Just having node doesn't mean word exists
if node:
    return True

# Right: Check if it's marked as end
if node and node.is_end:
    return True
\`\`\`

**2. Not Handling Empty String**
Always consider edge case of empty input.

**3. Memory Leak in Delete**
Must carefully clean up unused nodes.

**4. Inefficient Prefix Collection**
Use DFS, not repeated searches.

---

**Practice Progression:**

**Week 1: Basics**
- Implement Trie
- Add and Search Word
- Replace Words

**Week 2: Applications**
- Word Search II
- Longest Word in Dictionary
- Design Search Autocomplete System

**Week 3: Advanced**
- Maximum XOR of Two Numbers
- Concatenated Words
- Stream of Characters`,
    },
  ],
  keyTakeaways: [
    'Trie is a tree structure where each path represents a string prefix',
    'All operations (insert, search, prefix) are O(m) where m = string length',
    'Excellent for autocomplete, spell check, and prefix queries',
    'Space: O(ALPHABET_SIZE * m * n) worst case, O(m * n) best case',
    'Use hash map for flexible alphabet, array for fixed (faster but more space)',
    'Each node needs is_end_of_word flag to distinguish complete words from prefixes',
    'Can be extended with counts, wildcards, or other metadata per node',
    'More space-efficient than storing all prefixes separately',
  ],
  relatedProblems: ['implement-trie', 'add-and-search-word', 'word-search-ii'],
};
