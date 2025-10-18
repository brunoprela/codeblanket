/**
 * Introduction to Linked Lists Section
 */

export const introductionSection = {
  id: 'introduction',
  title: 'Introduction to Linked Lists',
  content: `A **linked list** is a linear data structure where elements (nodes) are stored non-contiguously in memory. Each node contains:
1. **Data** - the value stored
2. **Next pointer** - reference to the next node
3. **(Optional) Prev pointer** - for doubly linked lists

**Advantages over Arrays:**
- **Dynamic size**: Grows/shrinks as needed, no resizing
- **Efficient insertion/deletion**: O(1) at known positions
- **No wasted space**: Only allocates what's needed

**Disadvantages:**
- **No random access**: O(N) to access element at index i
- **Extra memory**: Pointers require additional space
- **Poor cache locality**: Nodes scattered in memory

**Types of Linked Lists:**

**1. Singly Linked List**
\`\`\`
[1] → [2] → [3] → [4] → None
\`\`\`
- One-directional traversal
- Each node has one \`next\` pointer

**2. Doubly Linked List**
\`\`\`
None ← [1] ⇄ [2] ⇄ [3] ⇄ [4] → None
\`\`\`
- Bidirectional traversal
- Each node has \`next\` and \`prev\` pointers
- More memory but more flexible

**3. Circular Linked List**
\`\`\`
[1] → [2] → [3] → [4] ⮌
\`\`\`
- Last node points back to first
- No \`None\` termination
- Useful for round-robin scheduling

**Python Implementation:**
\`\`\`python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# Create a linked list: 1 → 2 → 3
head = ListNode(1)
head.next = ListNode(2)
head.next.next = ListNode(3)
\`\`\`

**When to Use Linked Lists:**
- Frequent insertions/deletions at the beginning
- Unknown/dynamic size
- Implementing stacks, queues, hash tables (chaining)
- **Interview problems** involving pointer manipulation`,
};
