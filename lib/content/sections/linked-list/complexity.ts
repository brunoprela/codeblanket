/**
 * Complexity Analysis Section
 */

export const complexitySection = {
  id: 'complexity',
  title: 'Complexity Analysis',
  content: `**Linked List Operations:**

| Operation | Singly | Doubly | Array | Notes |
|-----------|--------|--------|-------|-------|
| Access by index | O(N) | O(N) | O(1) | Must traverse |
| Search | O(N) | O(N) | O(N) | Linear search |
| Insert at head | O(1) | O(1) | O(N) | Just update pointers |
| Insert at tail | O(N) | O(1)* | O(1) | *If tail pointer maintained |
| Insert at index | O(N) | O(N) | O(N) | Traverse + insert |
| Delete at head | O(1) | O(1) | O(N) | Just update pointers |
| Delete at tail | O(N) | O(1)* | O(1) | *If tail pointer maintained |
| Delete node | O(N) | O(1) | O(N) | Need prev for singly |

**Space Complexity:**

**Singly Linked List:**
- Per node: O(1) - one pointer
- Total: O(N) for N nodes

**Doubly Linked List:**
- Per node: O(2) - two pointers
- Total: O(2N) = O(N) for N nodes

**Recursive Operations:**
- Stack space: O(N) for call stack
- Example: Recursive reverse uses O(N) space

**Common Problem Complexities:**

**Reverse Linked List:**
- Iterative: Time O(N), Space O(1)
- Recursive: Time O(N), Space O(N)

**Detect Cycle:**
- Floyd's: Time O(N), Space O(1)
- Hash Set: Time O(N), Space O(N)

**Merge Two Lists:**
- Iterative: Time O(N+M), Space O(1)
- Recursive: Time O(N+M), Space O(N+M)

**Merge K Sorted Lists:**
- Naive: Time O(NK) where K is number of lists
- Min Heap: Time O(N log K), Space O(K)
- Divide & Conquer: Time O(N log K), Space O(log K)

**Find Middle:**
- Two pointers: Time O(N), Space O(1)
- Count then traverse: Time O(N), Space O(1)

**Key Insights:**
- Linked lists trade access time (O(N)) for insertion/deletion efficiency (O(1))
- Many problems can be solved in O(1) space using pointer manipulation
- Recursive solutions use O(N) extra space for call stack`,
};
