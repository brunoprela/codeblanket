/**
 * Interview Strategy Section
 */

export const interviewSection = {
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
};
