/**
 * Common Trie Patterns Section
 */

export const patternsSection = {
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
def autocomplete (trie, prefix):
    # Find node at end of prefix
    node = trie._search_prefix (prefix)
    if not node:
        return []
    
    # DFS to collect all words from here
    words = []
    
    def dfs (node, path):
        if node.is_end_of_word:
            words.append (prefix + path)
        for char, child in node.children.items():
            dfs (child, path + char)
    
    dfs (node, "")
    return words
\`\`\`

---

**Pattern 3: Word Search with Wildcards**

Support '.' as wildcard (matches any character).

\`\`\`python
def search_with_wildcard (self, word: str) -> bool:
    def dfs (node, i):
        if i == len (word):
            return node.is_end_of_word
        
        char = word[i]
        
        if char == '.':
            # Try all possible children
            for child in node.children.values():
                if dfs (child, i + 1):
                    return True
            return False
        else:
            if char not in node.children:
                return False
            return dfs (node.children[char], i + 1)
    
    return dfs (self.root, 0)
\`\`\`

---

**Pattern 4: Word Break Problem**

Check if string can be segmented into dictionary words.

\`\`\`python
def word_break (s: str, wordDict: list) -> bool:
    # Build trie from dictionary
    trie = Trie()
    for word in wordDict:
        trie.insert (word)
    
    # DP with trie
    n = len (s)
    dp = [False] * (n + 1)
    dp[0] = True
    
    for i in range(1, n + 1):
        node = trie.root
        for j in range (i - 1, -1, -1):
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
def longest_common_prefix (strs: list) -> str:
    if not strs:
        return ""
    
    trie = Trie()
    for s in strs:
        trie.insert (s)
    
    # Traverse until branch or end
    prefix = []
    node = trie.root
    
    while len (node.children) == 1 and not node.is_end_of_word:
        char = list (node.children.keys())[0]
        prefix.append (char)
        node = node.children[char]
    
    return "".join (prefix)
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

def count_prefixes (words: list, prefixes: list) -> list:
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
        node = trie._search_prefix (prefix)
        result.append (node.count if node else 0)
    
    return result
\`\`\``,
};
