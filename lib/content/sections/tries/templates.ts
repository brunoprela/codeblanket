/**
 * Code Templates Section
 */

export const templatesSection = {
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
};
