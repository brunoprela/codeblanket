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
      quiz: [
        {
          id: 'q1',
          question:
            'Explain what a Trie is and how it differs from other tree structures. What makes it special for string operations?',
          sampleAnswer:
            'A Trie (prefix tree) is a tree where each node represents a character and paths from root to nodes spell words. Unlike binary trees, each node can have up to 26 children (for lowercase English). Special for strings because it stores common prefixes once - if you have "cat" and "car", they share "ca". Each edge represents adding one character. Searching for word takes O(L) where L is word length, independent of how many words stored. Binary search tree would take O(L log N) where N is number of words. Trie trades space for time: uses more memory (26 pointers per node even if unused) but provides fast prefix operations. Perfect for autocomplete, spell check, IP routing.',
          keyPoints: [
            'Tree where nodes represent characters',
            'Paths spell words, shares common prefixes',
            'Search: O(L) independent of word count',
            'vs BST: O(L log N)',
            'Use cases: autocomplete, spell check, prefix search',
          ],
        },
        {
          id: 'q2',
          question:
            'Walk me through inserting a word into a Trie. What happens with overlapping prefixes?',
          sampleAnswer:
            'Start at root. For each character in word: check if child node for that character exists. If yes, move to that child. If no, create new child node for that character. At end of word, mark node as end-of-word. For overlapping prefixes: if inserting "cat" then "car", first insert creates nodes C→A→T with T marked as end. Second insert reuses C→A, creates R branch at A, marks R as end. The A node now has two children: T and R. This is key efficiency: shared prefix "ca" stored once, not duplicated. Each insert is O(L) where L is word length. The end-of-word marker distinguishes complete words from prefixes.',
          keyPoints: [
            'For each char: use existing child or create new',
            'Mark last node as end-of-word',
            'Overlapping prefixes: reuse existing nodes',
            'Example: "cat" then "car" shares "ca"',
            'O(L) per insert',
          ],
        },
        {
          id: 'q3',
          question:
            'Describe common Trie applications. Why is autocomplete a perfect use case?',
          sampleAnswer:
            'Autocomplete is perfect for Tries because after user types prefix, we just traverse to that prefix node, then DFS/BFS to collect all words in that subtree - all words starting with prefix. Dictionary operations: check if word exists in O(L) time. Spell check: find words within edit distance using Trie traversal with modifications. IP routing: store IP prefixes, match longest prefix efficiently. Word break: check if segments are valid words. Phone T9: map number sequences to possible words. The pattern: problems involving prefix matching, word validation, or collecting words with common prefix. Trie excels when you need fast prefix operations on large dictionary.',
          keyPoints: [
            'Autocomplete: traverse to prefix, collect subtree',
            'Dictionary: O(L) word lookup',
            'Spell check: edit distance traversal',
            'IP routing: longest prefix match',
            'Perfect for: prefix operations on large dictionary',
          ],
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Compare HashMap vs Array for storing Trie children. When would you choose each?',
          sampleAnswer:
            'HashMap (dict) is flexible: handles any character set (Unicode, symbols), space efficient for sparse data (only stores existing children). Array is fixed: typically size 26 for lowercase letters, uses index = char - ord("a") for O(1) access. Choose HashMap when: character set is large or unknown, data is sparse (few children per node), need flexibility. Choose Array when: character set is small and fixed (lowercase letters), performance critical (array access faster than hash lookup), memory layout matters (cache-friendly). For English words, array is common. For international text or mixed characters, use HashMap. The tradeoff: HashMap flexibility vs Array speed and simplicity.',
          keyPoints: [
            'HashMap: flexible, any char set, space efficient when sparse',
            'Array: fixed size (26), O(1) guaranteed, cache-friendly',
            'HashMap when: large/unknown charset, sparse data',
            'Array when: fixed small charset, performance critical',
            'Tradeoff: flexibility vs speed',
          ],
        },
        {
          id: 'q2',
          question:
            'Explain the is_end_of_word flag. Why cannot we just check if node has no children?',
          sampleAnswer:
            'is_end_of_word marks complete words vs prefixes. Cannot check "no children" because words can be prefixes of other words. Example: insert "car" then "carpet". The node at R (end of "car") has children (p→e→t), but "car" is still a complete word. Without the flag, searching "car" would fail because R has children. The flag distinguishes: node in middle of word, node at end of word but also prefix of longer words, node at end of only word (leaf). When searching, we need node to exist AND be marked as end. The flag is independent of having children - a node can be both end of word and have children.',
          keyPoints: [
            'Marks complete words vs prefixes only',
            'Cannot use "no children" check',
            'Words can be prefixes: "car" and "carpet"',
            'Node can be end-of-word AND have children',
            'Search needs: node exists AND is_end_of_word',
          ],
        },
        {
          id: 'q3',
          question:
            'Walk me through the autocomplete get_words_with_prefix implementation. How does DFS collect all words?',
          sampleAnswer:
            'Autocomplete has two phases. Phase 1: Navigate to prefix node. Traverse character by character from root. If any character missing, return empty list (no words with prefix). If reach end, we have the prefix subtree. Phase 2: DFS from prefix node to collect all words. Recursive DFS: if current node is_end_of_word, add accumulated path to results. Then recurse on all children, adding each child character to path. The DFS explores entire subtree under prefix node. For example, prefix "ca": navigate to node C→A, then DFS finds all paths: R (→car), R→P→E→T (→carpet), T (→cat). Return ["car", "carpet", "cat"]. The beauty: all words starting with prefix are in one subtree.',
          keyPoints: [
            'Phase 1: navigate to prefix node',
            'Phase 2: DFS to collect all words in subtree',
            'DFS: add word if is_end_of_word, recurse on children',
            'All words with prefix are in one subtree',
            'Example: "ca" → DFS finds car, carpet, cat',
          ],
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Explain the word search pattern. How does Trie make it more efficient than checking each word individually?',
          sampleAnswer:
            'Word search checks if string exists in dictionary. Naive approach: iterate through all words, compare each. For N words of length M, this is O(N×M) per search. With Trie: insert all words once in O(N×M) total, then each search is O(M) regardless of N. For example, dictionary with 100K words: naive search checks all 100K words every time. Trie search just traverses M nodes. The efficiency comes from shared prefixes: multiple words share path, so we do not redundantly check same prefix. If searching "apple", we traverse A→P→P→L→E once, check is_end_of_word. Dictionary operations (spell check, autocorrect) need many searches, so Trie amortizes cost across all words.',
          keyPoints: [
            'Naive: O(N×M) per search (check all words)',
            'Trie: O(N×M) build once, O(M) per search',
            'Shared prefixes avoid redundant checks',
            'Example: 100K words, Trie searches only M nodes',
            'Perfect for: many searches on same dictionary',
          ],
        },
        {
          id: 'q2',
          question:
            'Describe the word break pattern. How does Trie help with checking if segments are valid words?',
          sampleAnswer:
            'Word break determines if string can be segmented into dictionary words. Example: "leetcode" → "leet" + "code". Use DP with Trie for validation. DP: dp[i] = can we break s[0..i]. For each i, try all possible last words ending at i. Trie helps: from each position j, traverse Trie character by character toward i. If we hit end-of-word marker, s[j..i] is valid word, so dp[i] = dp[i] or dp[j]. Without Trie, checking if s[j..i] is valid word takes O(word_count × word_length). With Trie, traverse once character by character, finding all valid words simultaneously. The Trie acts as oracle: "is this prefix valid? is this word complete?" enabling efficient DP.',
          keyPoints: [
            'Segment string into dictionary words',
            'DP: dp[i] = can break s[0..i]',
            'Trie validates: is substring valid word?',
            'Traverse Trie from each position, find valid words',
            'More efficient than checking each word separately',
          ],
        },
        {
          id: 'q3',
          question:
            'Walk me through the prefix matching pattern for IP routing. Why is Trie ideal for longest prefix match?',
          sampleAnswer:
            'IP routing finds longest matching prefix for destination IP. Example: routes for 192.168.*, 192.168.1.*, 192.168.1.100. For IP 192.168.1.100, longest match is 192.168.1.100 (most specific). Build Trie of IP prefixes (each bit is node). Search by traversing IP bits. Track last end-of-word node seen (last valid route). At end, return longest match found. Trie is ideal because: naturally finds longest match by traversing deep as possible, shared prefixes stored once (192.168 shared by all), O(bits) lookup regardless of route count. In practice, 32-bit IPv4 gives O(32) lookup for millions of routes. The tree structure mirrors IP hierarchy: general to specific.',
          keyPoints: [
            'Find longest matching prefix for IP',
            'Build Trie of IP prefixes (bits as nodes)',
            'Traverse IP, track last valid route seen',
            'Naturally finds longest by going deep as possible',
            'O(bits) lookup for millions of routes',
          ],
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Explain compressed Trie (Radix Tree). How does it save space compared to standard Trie?',
          sampleAnswer:
            'Compressed Trie merges chains of single-child nodes into one node storing edge label (string). Standard Trie for "test" and "testing" has nodes T→E→S→T→I→N→G with "test" marked at first T. Compressed Trie stores edge "test" from root, then edge "ing" for "testing". Saves space by eliminating intermediate nodes that have only one child. Standard uses O(total characters), compressed uses O(number of words). For sparse dictionary where words share few prefixes, compressed Trie is much more space efficient. Tradeoff: implementation complexity increases (store strings on edges, not chars in nodes). Used in practice: Git uses Radix trees, file systems for path lookup. Best when: long strings, few shared prefixes.',
          keyPoints: [
            'Merge single-child chains into one node',
            'Store strings on edges, not chars',
            'Space: O(words) vs O(total chars)',
            'Best: sparse dictionary, long strings, few prefixes',
            'Tradeoff: space savings vs implementation complexity',
          ],
        },
        {
          id: 'q2',
          question:
            'Describe Trie with counts. What additional problems does it enable?',
          sampleAnswer:
            'Add count field to each node tracking number of words passing through. During insert, increment count on each node traversed. This enables: prefix frequency (how many words start with prefix = count at prefix node), word frequency (store count at end node), autocomplete with popularity (sort suggestions by count), stream problems (maintain counts as words arrive). For example, insert "cat" twice, "car" once. Node C has count 3, A has count 3, T has count 2 (end), R has count 1 (end). Query "ca" prefix: count at A is 3 words start with "ca". This augmentation adds minimal overhead (one integer per node) but enables frequency-based queries crucial for real-world applications like search suggestions.',
          keyPoints: [
            'Add count field: words passing through node',
            'Increment during insert, decrement during delete',
            'Enables: prefix frequency, word frequency',
            'Autocomplete with popularity ranking',
            'Minimal overhead, major functionality gain',
          ],
        },
        {
          id: 'q3',
          question:
            'Walk me through the XOR Trie for maximum XOR problem. Why use binary Trie?',
          sampleAnswer:
            'Maximum XOR finds two numbers with maximum XOR result. XOR maximized when bits differ. Build binary Trie: each node has children 0 and 1, insert numbers bit by bit (32 bits for integers). For each query number, traverse Trie trying opposite bit at each level (if number has 0, try to go 1; if has 1, try to go 0). This greedily maximizes XOR from most significant bit. If opposite bit path exists, take it. Otherwise, take same bit. At leaf, we have the number that gives maximum XOR with query. Example: numbers [3,10,5,25], query 2. Binary: 00010. Try to go opposite at each bit to maximize XOR. Binary Trie perfect because it organizes numbers by bit patterns, enabling greedy bit-by-bit maximization.',
          keyPoints: [
            'Binary Trie: nodes have children 0 and 1',
            'Insert numbers bit by bit (32 bits)',
            'Query: try opposite bit at each level',
            'Greedy maximizes XOR from MSB',
            'Perfect for: bit manipulation, XOR problems',
          ],
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Compare Trie vs Hash Table vs BST for dictionary operations. When would you choose Trie?',
          sampleAnswer:
            'For exact word lookup: Hash Table is O(m) average case (same as Trie), BST is O(m log n). For prefix operations: Trie excels with O(m), Hash Table cannot do prefix search efficiently, BST is O(m log n + k) for k results. Space: Trie uses most space (26 pointers per node), Hash Table is moderate, BST is least (two pointers per node). Choose Trie when: prefix operations needed (autocomplete, word games), many words share prefixes (amortizes space), need ordered traversal by prefix. Choose Hash Table when: only exact lookup, space matters, no prefix operations. Choose BST when: need ordered iteration, space critical, fewer words. The killer feature of Trie is prefix operations - nothing else does them efficiently.',
          keyPoints: [
            'Lookup: Trie O(m), Hash O(m) avg, BST O(m log n)',
            'Prefix: Trie O(m), Hash inefficient, BST O(m log n + k)',
            'Space: Trie highest, Hash moderate, BST lowest',
            'Choose Trie: prefix operations crucial',
            'Trie killer feature: efficient prefix queries',
          ],
        },
        {
          id: 'q2',
          question:
            'Explain Trie space complexity. Why is it O(ALPHABET_SIZE × m × n) worst case?',
          sampleAnswer:
            'Each node has array of ALPHABET_SIZE pointers (26 for lowercase). For n words of average length m, worst case has n × m nodes (no shared prefixes, all words completely different). Each node has 26 pointers, so total space is 26 × m × n pointers. For example, 1000 words of length 10: worst case is 26 × 10 × 1000 = 260K pointers. In practice, much better due to shared prefixes. If all words start with "a", first node is shared. Best case: all words identical, only m nodes for shared path. Average case depends on prefix overlap. Using HashMap children reduces to actual children count, not fixed 26. The alphabet size dominates: English (26) vs Unicode (100K+) makes huge difference.',
          keyPoints: [
            'Each node: ALPHABET_SIZE pointers',
            'Worst case: n words, m length, no sharing',
            'Space: ALPHABET_SIZE × m × n',
            'Best case: O(m) when all words identical',
            'HashMap children: only actual children, more efficient',
          ],
        },
        {
          id: 'q3',
          question: 'When should you NOT use a Trie? What are its limitations?',
          sampleAnswer:
            'Do NOT use Trie when: only exact lookups needed (Hash Table simpler and uses less space), space is critical constraint (Trie is space-heavy), alphabet is huge like Unicode (26^1000 becomes unreasonable), words are very long with no shared prefixes (defeats purpose of Trie), need additional operations like range queries (better suited for other structures). For example, storing random UUIDs: no shared prefixes, no prefix queries needed - Hash Table is better. Storing full DNA sequences: 4-letter alphabet is good, but sequences are millions long with little sharing - other structures might be better. Trie shines when: many prefix queries, moderate alphabet size, significant prefix overlap, moderate word lengths. Know when NOT to use it is as important as knowing when to use it.',
          keyPoints: [
            'Not for: only exact lookups, space critical',
            'Not for: huge alphabet, no prefix sharing',
            'Example: random UUIDs better in Hash Table',
            'Trie shines: prefix queries, moderate alphabet, sharing',
            'Wrong tool for job: wastes space, adds complexity',
          ],
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Walk me through the basic Trie template. What are the essential components?',
          sampleAnswer:
            'Basic Trie has two classes: TrieNode and Trie. TrieNode has children (dict or array) and is_end_of_word flag. Trie has root node and three methods. Insert: start at root, for each char, get or create child node, move to child, mark last as end. Search: traverse chars, return False if any missing, check is_end_of_word at end. StartsWith: same as search but ignore is_end_of_word. The pattern: all operations traverse character by character. Children dict is most flexible (any alphabet). Array (size 26) is faster but limited. The is_end_of_word flag is crucial for distinguishing complete words from prefixes. This template is foundation for all Trie problems - understand it deeply.',
          keyPoints: [
            'Two classes: TrieNode (children, is_end_of_word) and Trie (root, methods)',
            'Insert: traverse/create path, mark end',
            'Search: traverse, check exists and is_end_of_word',
            'StartsWith: traverse, ignore end flag',
            'Pattern: all ops traverse char by char',
          ],
        },
        {
          id: 'q2',
          question:
            'Explain the autocomplete template. How do you collect all words with a prefix?',
          sampleAnswer:
            'Autocomplete is two-phase. Phase 1: Navigate to prefix node using starts_with logic. If prefix does not exist, return empty. Phase 2: DFS from prefix node to collect all words. DFS function: takes current node and path built so far. If node is_end_of_word, add prefix + path to results. Recurse on all children, adding child char to path. This DFS explores entire subtree under prefix node. For example, prefix "ca", navigate to node at A (after C), then DFS finds all paths: R (car), R→P→E→T (carpet), T (cat). Return results. The key insight: all words with prefix are descendants of prefix node. DFS naturally collects them all.',
          keyPoints: [
            'Phase 1: navigate to prefix node',
            'Phase 2: DFS from prefix node',
            'DFS: add word if is_end, recurse on children',
            'All prefix words are in subtree',
            'Example: "ca" → DFS finds car, carpet, cat',
          ],
        },
        {
          id: 'q3',
          question:
            'Describe the Trie delete template. Why is it more complex than insert?',
          sampleAnswer:
            'Delete is complex because we must avoid breaking other words. Use recursive approach: delete(node, word, index). Base case: index equals word length, unmark is_end_of_word. Recursive: get child for current char, recursively delete from child. After recursion, check if child should be removed: child has no children and is not end of other word. Return whether current node should be deleted (no children, not end of word). The complexity: cannot just remove nodes blindly. If deleting "car" but "carpet" exists, cannot remove R node (has children). If deleting "carpet" but "car" exists, cannot remove R node (is end of word). Must carefully check each node bottom-up. This is why delete is rarely asked in interviews - too complex for 45 minutes.',
          keyPoints: [
            'Recursive: delete(node, word, index)',
            'Base: unmark is_end_of_word',
            'Recursive: delete from child, check if remove child',
            'Cannot blindly remove: might break other words',
            'Check: node has no children and not end of word',
          ],
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'How do you recognize that a problem needs a Trie? What keywords or patterns signal this?',
          sampleAnswer:
            'Several signals indicate Trie. Explicit: "prefix", "autocomplete", "dictionary", "word search", "spell check". Implicit: multiple words need efficient lookup, need to find all words with prefix, matching patterns in strings. Problem types: implement dictionary with prefix query, word break, design search autocomplete, replace words, stream of characters. If problem involves many words and prefix operations, Trie is likely. For example, "design search autocomplete system" screams Trie. "Check if word exists in dictionary" could be hash table unless prefix operations mentioned. Ask yourself: do I need prefix matching? Do multiple words share prefixes? Is dictionary static or changing? Trie excels at: changing dictionary with prefix queries.',
          keyPoints: [
            'Keywords: prefix, autocomplete, dictionary, spell check',
            'Multiple words with prefix operations',
            'Problems: autocomplete, word break, replace words',
            'Ask: prefix matching needed? Words share prefixes?',
            'Trie excels: dynamic dictionary with prefix queries',
          ],
        },
        {
          id: 'q2',
          question:
            'Walk me through your complete Trie interview approach from recognition to implementation.',
          sampleAnswer:
            'First, recognize Trie from keywords (prefix, autocomplete). Second, clarify: character set (lowercase only?), will words be deleted, max word length, number of words. Third, explain approach: build Trie, insert all words, for queries traverse Trie. Fourth, state complexity: insert O(m) per word, search O(m), space O(ALPHABET × m × n) worst case, better with shared prefixes. Fifth, discuss implementation: use dict for children (flexible) vs array (faster), is_end_of_word flag essential. Sixth, draw small example: insert "cat", "car", show shared prefix. Seventh, code clearly with TrieNode class and Trie class. Test with example. Finally, optimize if needed: compressed Trie for space, add counts for frequency queries.',
          keyPoints: [
            'Recognize from keywords, clarify requirements',
            'Explain: build Trie, traverse for queries',
            'State complexity with reasoning',
            'Discuss: dict vs array, is_end_of_word flag',
            'Draw example showing shared prefix',
            'Code clearly, test, optimize if needed',
          ],
        },
        {
          id: 'q3',
          question:
            'What are common Trie mistakes in interviews and how do you avoid them?',
          sampleAnswer:
            'First: forgetting is_end_of_word flag, causing failure when words are prefixes of others ("car" and "carpet"). Second: using fixed array without checking char is in range (crashes on non-lowercase). Third: not handling empty string edge case. Fourth: in autocomplete, forgetting to add prefix to collected words (returning "r" instead of "car" for prefix "ca"). Fifth: delete implementation breaking other words (removing shared nodes). Sixth: thinking Trie needed when Hash Table sufficient (if no prefix operations). My strategy: always include is_end_of_word, validate character range if using array, test with prefix words ("a", "ab", "abc"), draw tree to visualize structure, verify sharing works correctly.',
          keyPoints: [
            'Forgetting is_end_of_word (fails on prefix words)',
            'Array without range check (crashes)',
            'Empty string edge case',
            'Autocomplete: remember to add prefix',
            'Test with: prefix words, draw tree, verify sharing',
          ],
        },
      ],
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
