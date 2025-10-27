/**
 * Introduction to Tries Section
 */

export const introductionSection = {
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

**Common Use Cases:**1. **Autocomplete / Type-ahead**
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
};
