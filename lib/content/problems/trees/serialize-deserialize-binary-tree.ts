/**
 * Serialize and Deserialize Binary Tree
 * Problem ID: serialize-deserialize-binary-tree
 * Order: 10
 */

import { Problem } from '../../../types';

export const serialize_deserialize_binary_treeProblem: Problem = {
  id: 'serialize-deserialize-binary-tree',
  title: 'Serialize and Deserialize Binary Tree',
  difficulty: 'Hard',
  description: `Serialization is the process of converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer, or transmitted across a network connection link to be reconstructed later in the same or another computer environment.

Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how your serialization/deserialization algorithm should work. You just need to ensure that a binary tree can be serialized to a string and this string can be deserialized to the original tree structure.

**Clarification:** The input/output format is the same as how LeetCode serializes a binary tree. You do not necessarily need to follow this format, so please be creative and come up with different approaches yourself.


**Why This Problem Matters:**

This is one of the most asked tree problems at FAANG companies because it tests:
- Deep understanding of tree traversal
- String manipulation skills
- Edge case handling (null nodes, single nodes, etc.)
- Design skills (choosing the right format)

**Key Concepts:**1. **Serialization Strategy:** Convert tree to string representation
   - Can use pre-order, level-order, or post-order traversal
   - Must encode null nodes to preserve structure
   
2. **Deserialization Strategy:** Reconstruct tree from string
   - Parse the string and rebuild nodes
   - Must handle null markers correctly
   
3. **Format Choices:**
   - Delimiter-based: "1,2,3,null,null,4,5"
   - Bracket-based: "1(2)(3(4)(5))"
   - Level-order vs Pre-order

**Common Approaches:**

**Approach 1: Pre-order Traversal (Recommended)**
- Serialize: Root → Left → Right, mark nulls
- Deserialize: Build root, recursively build left & right
- Time: O(N), Space: O(N)
- Pros: Simple, intuitive, recursive

**Approach 2: Level-order Traversal (BFS)**
- Serialize: Use queue, process level by level
- Deserialize: Use queue, connect nodes level by level  
- Time: O(N), Space: O(N)
- Pros: Matches LeetCode format

**Approach 3: Post-order Traversal**
- Serialize: Left → Right → Root
- Deserialize: Build from end of string
- Time: O(N), Space: O(N)
- Less common in interviews`,
  examples: [
    {
      input: 'root = [1,2,3,null,null,4,5]',
      output: '[1,2,3,null,null,4,5]',
      explanation:
        'Tree structure is preserved after serialization and deserialization.',
    },
    {
      input: 'root = []',
      output: '[]',
      explanation: 'Empty tree case.',
    },
    {
      input: 'root = [1]',
      output: '[1]',
      explanation: 'Single node tree.',
    },
    {
      input: 'root = [1,2]',
      output: '[1,2]',
      explanation: 'Tree with only left child.',
    },
  ],
  constraints: [
    'The number of nodes in the tree is in the range [0, 10^4]',
    '-1000 <= Node.val <= 1000',
  ],
  hints: [
    'Use pre-order traversal for serialization (root, left, right)',
    'Mark null nodes explicitly with a special marker like "null" or "N"',
    'Use a delimiter like comma to separate values',
    'For deserialization, use an iterator or index to track position',
    'Build the tree recursively during deserialization',
    'Handle edge cases: empty tree, single node, only left/right children',
  ],
  starterCode: `# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Codec:
    def serialize(self, root: TreeNode) -> str:
        """Encodes a tree to a single string.
        
        Args:
            root: Root of binary tree
            
        Returns:
            String representation of the tree
        """
        pass
    
    def deserialize(self, data: str) -> TreeNode:
        """Decodes your encoded data to tree.
        
        Args:
            data: String representation from serialize()
            
        Returns:
            Root of reconstructed binary tree
        """
        pass

# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.deserialize(codec.serialize(root))`,
  testCases: [
    {
      input: [1, 2, 3, null, null, 4, 5],
      expected: [1, 2, 3, null, null, 4, 5],
    },
    {
      input: [],
      expected: [],
    },
    {
      input: [1],
      expected: [1],
    },
    {
      input: [1, 2],
      expected: [1, 2],
    },
  ],
  solution: `# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Codec:
    """
    Serialize and Deserialize Binary Tree using Pre-order Traversal.
    
    Time Complexity: O(N) for both operations
    Space Complexity: O(N) for storing the string and recursion stack
    
    Strategy:
    - Serialize: Pre-order traversal (root, left, right) with null markers
    - Deserialize: Build tree recursively using iterator
    
    Format: "1,2,N,N,3,4,N,N,5,N,N"
    Where N represents null nodes and commas are delimiters
    """
    
    def serialize(self, root: TreeNode) -> str:
        """
        Encodes a tree to a single string using pre-order traversal.
        
        Example: Tree [1,2,3,null,null,4,5]
                 1
                / \\
               2   3
                  / \\
                 4   5
        
        Serialize as: "1,2,N,N,3,4,N,N,5,N,N"
        Breakdown:
        - Visit 1 → "1"
        - Visit 2 → "1,2"
        - 2.left is null → "1,2,N"
        - 2.right is null → "1,2,N,N"
        - Visit 3 → "1,2,N,N,3"
        - Visit 4 → "1,2,N,N,3,4"
        - 4.left is null → "1,2,N,N,3,4,N"
        - 4.right is null → "1,2,N,N,3,4,N,N"
        - Visit 5 → "1,2,N,N,3,4,N,N,5"
        - 5.left is null → "1,2,N,N,3,4,N,N,5,N"
        - 5.right is null → "1,2,N,N,3,4,N,N,5,N,N"
        """
        def dfs(node):
            if not node:
                return "N"
            
            # Pre-order: Root → Left → Right
            return f"{node.val},{dfs(node.left)},{dfs(node.right)}"
        
        return dfs(root)
    
    def deserialize(self, data: str) -> TreeNode:
        """
        Decodes string to tree using recursive pre-order reconstruction.
        
        Example: data = "1,2,N,N,3,4,N,N,5,N,N"
        
        Parse as list: ["1", "2", "N", "N", "3", "4", "N", "N", "5", "N", "N"]
        Use iterator to track current position:
        
        1. Read "1" → Create node(1)
        2. Build left subtree:
           - Read "2" → Create node(2)
           - Build 2.left: Read "N" → None
           - Build 2.right: Read "N" → None
           - Return node(2)
        3. Build right subtree:
           - Read "3" → Create node(3)
           - Build 3.left:
             - Read "4" → Create node(4)
             - Build 4.left: Read "N" → None
             - Build 4.right: Read "N" → None
             - Return node(4)
           - Build 3.right:
             - Read "5" → Create node(5)
             - Build 5.left: Read "N" → None
             - Build 5.right: Read "N" → None
             - Return node(5)
           - Return node(3)
        4. Return node(1) as root
        """
        def dfs():
            val = next(vals)
            
            # Base case: null marker
            if val == "N":
                return None
            
            # Create node and recursively build subtrees
            node = TreeNode(int(val))
            node.left = dfs()   # Build left subtree
            node.right = dfs()  # Build right subtree
            
            return node
        
        # Convert string to iterator for sequential access
        vals = iter(data.split(","))
        return dfs()


# Alternative Solution: Level-order (BFS) approach
class CodecBFS:
    """
    Alternative using level-order traversal (matches LeetCode format).
    
    Time: O(N), Space: O(N)
    
    Pros: Matches standard tree representation
    Cons: Slightly more complex with queue management
    """
    
    def serialize(self, root: TreeNode) -> str:
        if not root:
            return ""
        
        result = []
        queue = [root]
        
        while queue:
            node = queue.pop(0)
            
            if node:
                result.append(str(node.val))
                queue.append(node.left)
                queue.append(node.right)
            else:
                result.append("N")
        
        return ",".join(result)
    
    def deserialize(self, data: str) -> TreeNode:
        if not data:
            return None
        
        vals = data.split(",")
        root = TreeNode(int(vals[0]))
        queue = [root]
        i = 1
        
        while queue and i < len(vals):
            node = queue.pop(0)
            
            # Process left child
            if vals[i] != "N":
                node.left = TreeNode(int(vals[i]))
                queue.append(node.left)
            i += 1
            
            # Process right child
            if i < len(vals) and vals[i] != "N":
                node.right = TreeNode(int(vals[i]))
                queue.append(node.right)
            i += 1
        
        return root


"""
Comparison of Approaches:

Pre-order (Recommended for interviews):
✅ Simple, clean recursive solution
✅ Easy to explain and code
✅ Natural tree traversal
✅ Minimal state tracking (just iterator)

Level-order:
✅ Matches LeetCode format
✅ Intuitive (processes level by level)
⚠️ More complex with queue management
⚠️ More variables to track

Interview Tips:
1. Start with pre-order - cleaner and easier to code under pressure
2. Explain your choice of delimiter and null marker
3. Handle edge cases: empty tree, single node
4. Test with asymmetric trees (only left or right children)
5. Time/Space complexity: Both O(N)

Common Mistakes:
❌ Forgetting to mark null nodes → can't reconstruct structure
❌ Not using delimiter → can't parse multi-digit numbers
❌ Off-by-one errors in level-order deserialization
❌ Not handling empty tree case
"""`,
  timeComplexity: 'O(N) for both serialize and deserialize',
  spaceComplexity: 'O(N) for the string and recursion stack',
  order: 10,
  topic: 'Trees',
  leetcodeUrl:
    'https://leetcode.com/problems/serialize-and-deserialize-binary-tree/',
  youtubeUrl: 'https://www.youtube.com/watch?v=u4JAi2JJhI8',
};
