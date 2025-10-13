import { Module } from '@/lib/types';

export const linkedListModule: Module = {
  id: 'linked-list',
  title: 'Linked List',
  description:
    'Master linked list manipulation, pointer techniques, and common patterns for interview success.',
  icon: '🔗',
  timeComplexity:
    'O(N) for traversal, O(1) for insertion/deletion at known positions',
  spaceComplexity: 'O(1) for most operations',
  sections: [
    {
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
      quiz: [
        {
          id: 'q1',
          question:
            'Explain the fundamental difference between linked lists and arrays. What trade-offs do you make when choosing one over the other?',
          sampleAnswer:
            'The fundamental difference is memory layout and access pattern. Arrays store elements contiguously in memory with O(1) random access by index - I can jump to any element instantly. Linked lists store elements as nodes scattered in memory, connected by pointers, with O(n) access - I must traverse from head to reach an element. Arrays have better cache locality, making them faster for sequential access. Linked lists excel at insertions and deletions, especially at the beginning: O(1) vs O(n) for arrays which must shift elements. The trade-off: arrays for fast access and iteration, linked lists for dynamic size and frequent insertions/deletions. Memory overhead: arrays have minimal overhead, linked lists need extra pointer storage per node.',
          keyPoints: [
            'Arrays: contiguous memory, O(1) random access',
            'Linked lists: scattered memory, O(n) access',
            'Arrays: better cache locality',
            'Linked lists: O(1) insert/delete at beginning',
            'Trade-off: access speed vs insertion flexibility',
          ],
        },
        {
          id: 'q2',
          question:
            'Walk me through the mechanics of inserting a node in the middle of a linked list. Why is this O(1) once you have the reference?',
          sampleAnswer:
            'To insert a new node after a given node, I create the new node, set its next pointer to point to the current node next, then update the current node next to point to the new node. This is just two pointer updates, so O(1) time once I have the reference. The key is "once you have the reference" - finding the insertion point takes O(n) traversal. But if I already have a pointer to the node, insertion is constant time because I am only changing pointers, not shifting elements like in arrays. For example, to insert B between A and C: B.next = A.next (B points to C), then A.next = B (A points to B). No other nodes are affected.',
          keyPoints: [
            'Create new node',
            'New node points to current node next',
            'Update current node to point to new node',
            'Two pointer updates: O(1)',
            'Finding position: O(n), insertion itself: O(1)',
          ],
        },
        {
          id: 'q3',
          question:
            'Describe the concept of a dummy node. When would you use it and why does it simplify code?',
          sampleAnswer:
            'A dummy node is a placeholder node placed before the actual head of the list. Instead of tracking the real head, I work with dummy.next as the head. This simplifies edge cases because now the head is no longer special - I can insert or delete at the head using the same logic as any other position. Without a dummy, I need special case handling when the head changes. For example, deleting the first node requires updating the head pointer separately. With a dummy, deleting the first node is just dummy.next = dummy.next.next, same as any deletion. At the end, return dummy.next as the new head. This is a common interview trick to avoid messy conditional logic.',
          keyPoints: [
            'Placeholder node before real head',
            'Work with dummy.next as head',
            'Eliminates special case for head operations',
            'Same logic for all positions',
            'Return dummy.next at end',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What is the main advantage of linked lists over arrays?',
          options: [
            'Faster random access',
            'O(1) insertion/deletion at known positions',
            'Better cache locality',
            'Less memory usage',
          ],
          correctAnswer: 1,
          explanation:
            'Linked lists excel at O(1) insertion and deletion at known positions (just pointer updates), while arrays require O(N) shifting of elements. However, arrays have O(1) random access.',
        },
        {
          id: 'mc2',
          question: 'How do you access the 100th element in a linked list?',
          options: [
            'Use indexing: list[99]',
            'Traverse from head 100 times',
            'Use binary search',
            'Direct memory access',
          ],
          correctAnswer: 1,
          explanation:
            'Linked lists require O(N) traversal to access any element. You must start at the head and follow next pointers until you reach the desired node. No random access is available.',
        },
        {
          id: 'mc3',
          question: 'What is the time complexity of inserting at the beginning of a linked list?',
          options: [
            'O(N)',
            'O(1)',
            'O(log N)',
            'O(N²)',
          ],
          correctAnswer: 1,
          explanation:
            'Inserting at the beginning is O(1): create new node, set its next to current head, update head to new node. Only three operations regardless of list size.',
        },
        {
          id: 'mc4',
          question: 'What is a dummy node and why is it useful?',
          options: [
            'A node with value 0',
            'A placeholder node before the real head that simplifies edge cases',
            'The last node in the list',
            'A node used for sorting',
          ],
          correctAnswer: 1,
          explanation:
            'A dummy node is a placeholder before the actual head. It eliminates special-case handling for head operations since the head is now at dummy.next, treated like any other node.',
        },
        {
          id: 'mc5',
          question: 'In a doubly linked list, what does each node contain?',
          options: [
            'Only a value and next pointer',
            'A value, next pointer, and prev pointer',
            'Two values',
            'Only pointers, no value',
          ],
          correctAnswer: 1,
          explanation:
            'Doubly linked lists have three components per node: the data value, a next pointer to the following node, and a prev pointer to the previous node, enabling bidirectional traversal.',
        },
      ],
    },
    {
      id: 'patterns',
      title: 'Essential Linked List Patterns',
      content: `**Pattern 1: Two Pointers (Fast & Slow)**

**Use Cases:**
- Detect cycles
- Find middle of list
- Find kth from end

**Visualization - Find Middle:**
\`\`\`
Initial:
slow, fast
   ↓    ↓
[1] → [2] → [3] → [4] → [5] → None

Step 1:
     slow   fast
       ↓      ↓
[1] → [2] → [3] → [4] → [5] → None

Step 2:
          slow        fast
            ↓           ↓
[1] → [2] → [3] → [4] → [5] → None

fast reaches end, slow is at middle!
\`\`\`

**Cycle Detection (Floyd's Algorithm):**
\`\`\`
[1] → [2] → [3] → [4]
       ↑           ↓
       ←  ←  ←  ← [5]

Fast and slow will meet inside the cycle!
\`\`\`

---

**Pattern 2: Dummy Node**

Use a dummy/sentinel node to simplify edge cases.

**Without Dummy (Complex):**
\`\`\`python
if not head:
    return None
if head.val == target:
    return head.next
# ... more cases
\`\`\`

**With Dummy (Simple):**
\`\`\`python
dummy = ListNode(0)
dummy.next = head
prev = dummy
# ... uniform logic for all cases
return dummy.next
\`\`\`

**Visualization:**
\`\`\`
Original:  [1] → [2] → [3]
With dummy: [0] → [1] → [2] → [3]
             ↑
           dummy
\`\`\`

---

**Pattern 3: Reverse Pointers**

Reverse links by manipulating \`next\` pointers.

**Visualization:**
\`\`\`
Original:  [1] → [2] → [3] → None

Step 1:
None ← [1]  [2] → [3] → None
 ↑     ↑    ↑
prev  curr next

Step 2:
None ← [1] ← [2]  [3] → None
       ↑     ↑    ↑
      prev  curr next

Step 3:
None ← [1] ← [2] ← [3]
              ↑     ↑
             prev  curr

Result: [3] → [2] → [1] → None
\`\`\`

---

**Pattern 4: Runner Technique**

Use two pointers moving at different speeds or positions.

**Find Kth from End:**
\`\`\`
K = 2

Step 1: Move first pointer K steps ahead
[1] → [2] → [3] → [4] → [5] → None
 ↑           ↑
second     first

Step 2: Move both until first reaches end
[1] → [2] → [3] → [4] → [5] → None
            ↑                 ↑
          second            first

second is now K from end!
\`\`\`

---

**Pattern 5: Merge Technique**

Merge two sorted lists by comparing heads.

**Visualization:**
\`\`\`
List1: [1] → [3] → [5]
List2: [2] → [4] → [6]

Compare 1 vs 2: take 1
[1] → ...
Compare 3 vs 2: take 2
[1] → [2] → ...
Compare 3 vs 4: take 3
[1] → [2] → [3] → ...

Result: [1] → [2] → [3] → [4] → [5] → [6]
\`\`\``,
      codeExample: `class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def find_middle(head: ListNode) -> ListNode:
    """
    Find the middle node using fast & slow pointers.
    Time: O(N), Space: O(1)
    """
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    return slow  # slow is at middle


def has_cycle(head: ListNode) -> bool:
    """
    Detect if linked list has a cycle (Floyd's algorithm).
    Time: O(N), Space: O(1)
    """
    if not head:
        return False
    
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    
    return False


def reverse_list(head: ListNode) -> ListNode:
    """
    Reverse a linked list iteratively.
    Time: O(N), Space: O(1)
    """
    prev = None
    curr = head
    
    while curr:
        next_temp = curr.next  # Save next
        curr.next = prev       # Reverse pointer
        prev = curr            # Move prev forward
        curr = next_temp       # Move curr forward
    
    return prev  # New head


def remove_nth_from_end(head: ListNode, n: int) -> ListNode:
    """
    Remove nth node from end using runner technique.
    """
    dummy = ListNode(0)
    dummy.next = head
    first = second = dummy
    
    # Move first n+1 steps ahead
    for _ in range(n + 1):
        first = first.next
    
    # Move both until first reaches end
    while first:
        first = first.next
        second = second.next
    
    # Remove the nth node
    second.next = second.next.next
    
    return dummy.next`,
      quiz: [
        {
          id: 'q1',
          question:
            'Explain the fast and slow pointer technique. Why does the slow pointer end up at the middle when fast reaches the end?',
          sampleAnswer:
            'Fast and slow pointer uses two pointers moving at different speeds: slow moves one step per iteration, fast moves two steps. When fast reaches the end (or null), slow is at the middle. This works because fast travels twice the distance of slow. If the list has n nodes, when fast has moved n steps (reached end), slow has moved n/2 steps (at middle). For example, in list 1→2→3→4→5, when fast reaches 5, slow is at 3. This is elegant because we find the middle in one pass without knowing the length beforehand. Used for finding middle node, detecting cycles, or problems requiring simultaneous traversal at different speeds.',
          keyPoints: [
            'Slow moves one step, fast moves two steps',
            'Fast travels twice the distance of slow',
            'When fast at end, slow at middle',
            'One pass without knowing length',
            'Used for: middle, cycle detection, offset problems',
          ],
        },
        {
          id: 'q2',
          question:
            'Walk me through cycle detection with Floyd algorithm. How does meeting of two pointers prove a cycle exists?',
          sampleAnswer:
            'Floyd cycle detection uses fast and slow pointers. If there is a cycle, fast will eventually catch up to slow inside the cycle because fast gains one position per iteration. Think of it like a circular track: the faster runner will lap the slower runner. If no cycle, fast reaches null. Once they meet, we know a cycle exists. To find the cycle start, reset one pointer to head and move both one step at a time - they meet at the cycle entrance. This works due to mathematical properties of the distances. The brilliance is using O(1) space instead of O(n) hash set to track visited nodes. Time is O(n) as each pointer traverses at most n nodes.',
          keyPoints: [
            'Fast and slow pointers in cycle',
            'Fast eventually catches slow (laps on circular track)',
            'Meeting proves cycle exists',
            'To find start: reset one to head, both move one step',
            'O(1) space vs O(n) hash set',
          ],
        },
        {
          id: 'q3',
          question:
            'Describe the two-pointer technique for removing the nth node from end. Why move first pointer n steps ahead?',
          sampleAnswer:
            'To remove nth node from end, I use two pointers with n-node gap between them. First, move the first pointer n steps ahead. Then move both pointers together until first reaches the end. Now second pointer is n nodes from the end - exactly at the node before the one to delete. This works because maintaining constant gap of n nodes means when first reaches end (0 from end), second is n from end. I then do second.next = second.next.next to remove the target. Using a dummy node handles the edge case of removing the first node. This is one-pass solution, very elegant compared to two-pass (find length, then remove).',
          keyPoints: [
            'Two pointers with n-node gap',
            'Move first n steps ahead',
            'Move both until first at end',
            'Second now at node before target',
            'One pass vs two-pass solution',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'How does the fast and slow pointer technique detect a cycle in a linked list?',
          options: [
            'Fast pointer moves twice as fast; if there\'s a cycle, they will meet',
            'Both pointers move at same speed',
            'Fast pointer checks each node for duplicates',
            'Slow pointer marks visited nodes',
          ],
          correctAnswer: 0,
          explanation:
            'Fast pointer moves 2 steps while slow moves 1 step. In a cycle, fast will eventually lap slow and they will meet. If no cycle, fast reaches None.',
        },
        {
          id: 'mc2',
          question: 'What is the key insight for reversing a linked list iteratively?',
          options: [
            'Use a stack to store all nodes',
            'Reverse each node\'s pointer to point to the previous node',
            'Create a new list in reverse order',
            'Swap node values',
          ],
          correctAnswer: 1,
          explanation:
            'Iteratively reverse each pointer: save next, point current to previous, move forward. Requires three pointers: prev, current, and next. O(1) space.',
        },
        {
          id: 'mc3',
          question: 'To remove the nth node from the end, why do you move the first pointer n+1 steps ahead?',
          options: [
            'To make it faster',
            'So the second pointer stops at the node before the target',
            'To handle lists of length n',
            'It\'s not necessary',
          ],
          correctAnswer: 1,
          explanation:
            'Moving first pointer n+1 steps ahead (when using a dummy node) ensures that when first reaches the end, second is at the node before the target, allowing easy deletion with second.next = second.next.next.',
        },
        {
          id: 'mc4',
          question: 'What is Floyd\'s cycle detection algorithm also known as?',
          options: [
            'Binary search method',
            'Tortoise and hare algorithm',
            'Divide and conquer',
            'Greedy approach',
          ],
          correctAnswer: 1,
          explanation:
            'Floyd\'s cycle detection is called the "tortoise and hare" algorithm because of the slow (tortoise) and fast (hare) pointers moving at different speeds.',
        },
        {
          id: 'mc5',
          question: 'What is the time complexity of reversing a linked list iteratively?',
          options: [
            'O(N²)',
            'O(N)',
            'O(log N)',
            'O(1)',
          ],
          correctAnswer: 1,
          explanation:
            'Reversing a linked list iteratively requires a single pass through all N nodes, making it O(N) time complexity with O(1) space complexity.',
        },
      ],
    },
    {
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
      quiz: [
        {
          id: 'q1',
          question:
            'Compare the complexity of inserting at head vs tail in a singly linked list. Why is there such a difference?',
          sampleAnswer:
            'Inserting at head is O(1) because we have direct access to the head pointer - create new node, point it to current head, update head to new node. Done. Inserting at tail is O(n) because we must traverse the entire list to find the last node, then append. Each step of traversal takes constant time but we need n steps. However, if we maintain a tail pointer, insertion at tail becomes O(1) too - just update tail.next and move tail pointer. The difference comes from whether we need to search for the insertion point or have direct access to it. This is why doubly linked lists with tail pointers are more versatile.',
          keyPoints: [
            'Head insert: O(1) - direct access',
            'Tail insert: O(n) - must traverse',
            'With tail pointer: O(1) for tail insert',
            'Difference: need to search vs direct access',
            'Doubly linked with tail pointer: O(1) both ends',
          ],
        },
        {
          id: 'q2',
          question:
            'Explain why recursive linked list solutions use O(n) space. What exactly is stored on the call stack?',
          sampleAnswer:
            'Recursive solutions use O(n) space because each recursive call adds a stack frame to the call stack. For a list of n nodes, we make n recursive calls before starting to return. Each stack frame stores local variables like the current node pointer, return address, and any other local state. For example, recursive reverse: we recursively call on next node, storing current node on each frame, until we reach the end, then unwind the stack updating pointers. This is in contrast to iterative solutions which use O(1) space with just a few pointer variables. The space comes from the recursion depth matching list length, not from the data itself.',
          keyPoints: [
            'Each recursive call adds stack frame',
            'n nodes = n recursive calls',
            'Each frame: local variables, return address',
            'Stack depth matches list length',
            'Iterative: O(1) with pointer variables',
          ],
        },
        {
          id: 'q3',
          question:
            'Walk me through why Floyd cycle detection is O(n) time and O(1) space. How do the pointers traverse the list?',
          sampleAnswer:
            'Floyd cycle detection uses two pointers: slow (one step) and fast (two steps). In worst case with no cycle, fast reaches the end after n/2 iterations, giving O(n) time. With a cycle, once both enter the cycle, fast catches slow in at most cycle length iterations. Total is still O(n) because we traverse at most n nodes plus some cycle iterations. Space is O(1) because we only need two pointer variables, regardless of list size. Compare to hash set approach: also O(n) time but O(n) space to store visited nodes. The brilliance of Floyd is achieving cycle detection with constant space by using the mathematical property that faster pointer will catch slower in a cycle.',
          keyPoints: [
            'Two pointers: O(1) space',
            'No cycle: fast reaches end in n/2 iterations',
            'With cycle: fast catches slow within cycle',
            'Total: O(n) time',
            'vs Hash set: O(n) space',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What is the time complexity of accessing an element at a specific index in a linked list?',
          options: [
            'O(1)',
            'O(N)',
            'O(log N)',
            'O(N²)',
          ],
          correctAnswer: 1,
          explanation:
            'Accessing an element at index i requires traversing from the head through i nodes, taking O(N) time. Unlike arrays, linked lists do not support O(1) random access.',
        },
        {
          id: 'mc2',
          question: 'Why is inserting at the tail O(N) in a singly linked list without a tail pointer?',
          options: [
            'Because linked lists are slow',
            'Because you must traverse the entire list to find the last node',
            'Because insertion requires copying all nodes',
            'It is actually O(1)',
          ],
          correctAnswer: 1,
          explanation:
            'Without a tail pointer, you must traverse all N nodes to find the last node before inserting. With a tail pointer, it becomes O(1).',
        },
        {
          id: 'mc3',
          question: 'What is the space complexity of Floyd\'s cycle detection algorithm?',
          options: [
            'O(N)',
            'O(1)',
            'O(log N)',
            'O(N²)',
          ],
          correctAnswer: 1,
          explanation:
            'Floyd\'s algorithm uses only two pointers (slow and fast) regardless of list size, making it O(1) space. This is more efficient than hash set approaches which use O(N) space.',
        },
        {
          id: 'mc4',
          question: 'Why do recursive linked list solutions use O(N) space?',
          options: [
            'They create new nodes',
            'Each recursive call adds a stack frame to the call stack',
            'They use extra arrays',
            'They don\'t use O(N) space',
          ],
          correctAnswer: 1,
          explanation:
            'Recursive solutions make N recursive calls for N nodes, with each call adding a stack frame to the call stack. This results in O(N) space complexity, unlike iterative solutions which use O(1) space.',
        },
        {
          id: 'mc5',
          question: 'What is the time complexity of merging K sorted linked lists using a min heap?',
          options: [
            'O(NK)',
            'O(N log K)',
            'O(K log N)',
            'O(N²)',
          ],
          correctAnswer: 1,
          explanation:
            'Using a min heap of size K, each of the N total nodes is added and removed from the heap once, with each operation taking O(log K) time, giving O(N log K) total time.',
        },
      ],
    },
    {
      id: 'templates',
      title: 'Code Templates',
      content: `**Template 1: Basic Traversal**
\`\`\`python
def traverse(head: ListNode) -> None:
    """
    Traverse and process each node.
    """
    curr = head
    
    while curr:
        # Process curr.val
        print(curr.val)
        curr = curr.next
\`\`\`

**Template 2: Two Pointers (Fast & Slow)**
\`\`\`python
def two_pointer_pattern(head: ListNode) -> ListNode:
    """
    Generic fast & slow pointer template.
    """
    if not head:
        return None
    
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next          # Move 1 step
        fast = fast.next.next     # Move 2 steps
        
        # Check condition (e.g., cycle detection)
        if slow == fast:
            return True
    
    return slow  # Middle, or False for cycle
\`\`\`

**Template 3: Dummy Node Pattern**
\`\`\`python
def dummy_node_pattern(head: ListNode) -> ListNode:
    """
    Use dummy node to simplify edge cases.
    """
    dummy = ListNode(0)
    dummy.next = head
    
    prev = dummy
    curr = head
    
    while curr:
        # Perform operations
        if some_condition:
            # Remove curr
            prev.next = curr.next
        else:
            prev = curr
        curr = curr.next
    
    return dummy.next  # New head
\`\`\`

**Template 4: Reverse List (Iterative)**
\`\`\`python
def reverse_list(head: ListNode) -> ListNode:
    """
    Standard iterative reversal pattern.
    """
    prev = None
    curr = head
    
    while curr:
        next_temp = curr.next  # Save next
        curr.next = prev       # Reverse link
        prev = curr            # Move prev
        curr = next_temp       # Move curr
    
    return prev  # New head
\`\`\`

**Template 5: Reverse List (Recursive)**
\`\`\`python
def reverse_list_recursive(head: ListNode) -> ListNode:
    """
    Recursive reversal pattern.
    """
    # Base case
    if not head or not head.next:
        return head
    
    # Recursive case
    new_head = reverse_list_recursive(head.next)
    
    # Reverse the link
    head.next.next = head
    head.next = None
    
    return new_head
\`\`\`

**Template 6: Merge Two Lists**
\`\`\`python
def merge_two_lists(l1: ListNode, l2: ListNode) -> ListNode:
    """
    Merge two sorted lists.
    """
    dummy = ListNode(0)
    curr = dummy
    
    while l1 and l2:
        if l1.val <= l2.val:
            curr.next = l1
            l1 = l1.next
        else:
            curr.next = l2
            l2 = l2.next
        curr = curr.next
    
    # Attach remaining
    curr.next = l1 if l1 else l2
    
    return dummy.next
\`\`\`

**Template 7: Runner Technique (Kth from End)**
\`\`\`python
def kth_from_end(head: ListNode, k: int) -> ListNode:
    """
    Find kth node from the end.
    """
    first = second = head
    
    # Move first k steps ahead
    for _ in range(k):
        if not first:
            return None
        first = first.next
    
    # Move both until first reaches end
    while first:
        first = first.next
        second = second.next
    
    return second
\`\`\``,
      quiz: [
        {
          id: 'q1',
          question:
            'Walk me through the iterative reversal template. Why do we need three pointers (prev, curr, next)?',
          sampleAnswer:
            'For iterative reversal, prev tracks the reversed portion, curr is the node we are currently reversing, and next temporarily stores the rest of the list. The pattern: save curr.next in next (so we do not lose the list), reverse curr by setting curr.next = prev, move prev to curr, move curr to next. We need three because we must simultaneously reverse the current link and advance forward. If we only had two pointers and did curr.next = prev, we would lose access to the rest of the list. The next pointer saves that access. After the loop, prev points to the new head. This is O(n) time and O(1) space, elegant and efficient.',
          keyPoints: [
            'prev: reversed portion so far',
            'curr: node currently reversing',
            'next: temporarily saves rest of list',
            'Pattern: save next, reverse curr, advance both',
            'O(n) time, O(1) space',
          ],
        },
        {
          id: 'q2',
          question:
            'Explain the dummy node technique for merging sorted lists. Why does it make the code cleaner?',
          sampleAnswer:
            'Dummy node eliminates special case handling for the first node. Without it, I need to check if result list is empty and set head specially for the first node. With dummy, I start with dummy.next = None and use curr = dummy to track the tail of merged list. I just do curr.next = smaller node and advance curr. At the end, return dummy.next as the new head. This works because dummy acts as a placeholder before the real list, so even the first node is handled uniformly - no if statements for "is this the first node". The code becomes a simple loop without conditionals for initialization. This pattern appears in many linked list problems.',
          keyPoints: [
            'Placeholder before real head',
            'Eliminates first node special case',
            'curr tracks tail of merged list',
            'Uniform logic for all nodes',
            'Return dummy.next at end',
          ],
        },
        {
          id: 'q3',
          question:
            'Compare the iterative vs recursive approach to linked list problems. When would you choose each?',
          sampleAnswer:
            'Iterative uses explicit pointers and loops, giving O(1) space. Recursive uses call stack for implicit tracking, using O(n) space but often cleaner code. I choose iterative when space is critical or when the iterative logic is straightforward, like reversal or traversal. I choose recursive when the problem has natural recursive structure, like tree-like operations or when backtracking is needed. Recursive can be more elegant for complex pointer manipulation. In interviews, I often code iterative first as it shows space efficiency, then mention recursive as an alternative if asked. Some problems like reversing in groups feel more natural recursively.',
          keyPoints: [
            'Iterative: O(1) space, explicit pointers',
            'Recursive: O(n) space, cleaner for some problems',
            'Iterative: when space critical',
            'Recursive: natural structure, backtracking',
            'Interview strategy: start iterative, mention recursive',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'In the iterative reversal template, why do you need the next pointer?',
          options: [
            'To make it faster',
            'To temporarily save the rest of the list before reversing the current pointer',
            'To delete nodes',
            'It is not necessary',
          ],
          correctAnswer: 1,
          explanation:
            'The next pointer saves the rest of the list before you reverse curr.next = prev. Without it, you would lose access to the remaining nodes and couldn\'t continue traversal.',
        },
        {
          id: 'mc2',
          question: 'What does the dummy node technique eliminate in linked list problems?',
          options: [
            'All loops',
            'Special case handling for the head node',
            'The need for pointers',
            'Time complexity',
          ],
          correctAnswer: 1,
          explanation:
            'A dummy node acts as a placeholder before the real head, eliminating special case handling when the head changes (like insertion/deletion at the beginning). All nodes can be handled uniformly.',
        },
        {
          id: 'mc3',
          question: 'What is the space complexity advantage of iterative solutions over recursive ones?',
          options: [
            'Iterative uses O(N) space',
            'Iterative uses O(1) space instead of O(N) for recursive call stack',
            'They have the same space complexity',
            'Recursive is more space efficient',
          ],
          correctAnswer: 1,
          explanation:
            'Iterative solutions use O(1) space with just a few pointer variables, while recursive solutions use O(N) space for the call stack as each recursive call adds a stack frame.',
        },
        {
          id: 'mc4',
          question: 'In the runner technique for finding kth from end, how far ahead should the first pointer move?',
          options: [
            'k-1 steps',
            'k steps',
            'k+1 steps',
            '2k steps',
          ],
          correctAnswer: 1,
          explanation:
            'Move the first pointer k steps ahead. Then when both pointers move together until first reaches the end, the second pointer will be exactly k nodes from the end.',
        },
        {
          id: 'mc5',
          question: 'When merging two sorted lists, what should you do after the main loop?',
          options: [
            'Return immediately',
            'Attach the remaining portion of whichever list is not exhausted',
            'Delete remaining nodes',
            'Create new nodes',
          ],
          correctAnswer: 1,
          explanation:
            'After one list is exhausted, attach the remaining portion of the other list: curr.next = l1 if l1 else l2. This is more efficient than continuing to process node by node.',
        },
      ],
    },
    {
      id: 'advanced',
      title: 'Advanced Techniques',
      content: `**Technique 1: Finding Cycle Start**

After detecting cycle, find where it starts:

\`\`\`python
def detect_cycle_start(head: ListNode) -> ListNode:
    """
    Find the node where cycle begins.
    """
    # Phase 1: Detect cycle
    slow = fast = head
    has_cycle = False
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            has_cycle = True
            break
    
    if not has_cycle:
        return None
    
    # Phase 2: Find cycle start
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    
    return slow  # Cycle start
\`\`\`

**Why this works:**
- After meeting, if we reset slow to head and move both at same speed
- They'll meet at cycle start
- Mathematical proof: distance from head to start = distance from meeting point to start

---

**Technique 2: Reverse in Groups**

Reverse K nodes at a time:

\`\`\`python
def reverse_k_group(head: ListNode, k: int) -> ListNode:
    """
    Reverse every k nodes.
    """
    # Check if we have k nodes remaining
    curr = head
    count = 0
    while curr and count < k:
        curr = curr.next
        count += 1
    
    if count < k:
        return head  # Not enough nodes
    
    # Reverse first k nodes
    prev = None
    curr = head
    for _ in range(k):
        next_temp = curr.next
        curr.next = prev
        prev = curr
        curr = next_temp
    
    # Recursively reverse remaining
    head.next = reverse_k_group(curr, k)
    
    return prev
\`\`\`

---

**Technique 3: Palindrome Check**

Check if linked list is a palindrome:

\`\`\`python
def is_palindrome(head: ListNode) -> bool:
    """
    Check if linked list is palindrome.
    Time: O(N), Space: O(1)
    """
    # Find middle
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    # Reverse second half
    prev = None
    while slow:
        next_temp = slow.next
        slow.next = prev
        prev = slow
        slow = next_temp
    
    # Compare both halves
    left, right = head, prev
    while right:  # right is shorter or equal
        if left.val != right.val:
            return False
        left = left.next
        right = right.next
    
    return True
\`\`\`

**Steps:**
1. Find middle (fast & slow)
2. Reverse second half
3. Compare both halves
4. (Optional) Restore list

---

**Technique 4: Deep Copy with Random Pointer**

Clone a linked list with random pointers:

\`\`\`python
class Node:
    def __init__(self, val, next=None, random=None):
        self.val = val
        self.next = next
        self.random = random

def copy_random_list(head: Node) -> Node:
    """
    Deep copy linked list with random pointers.
    Time: O(N), Space: O(N)
    """
    if not head:
        return None
    
    # Step 1: Create copy nodes interleaved
    curr = head
    while curr:
        copy = Node(curr.val)
        copy.next = curr.next
        curr.next = copy
        curr = copy.next
    
    # Step 2: Set random pointers
    curr = head
    while curr:
        if curr.random:
            curr.next.random = curr.random.next
        curr = curr.next.next
    
    # Step 3: Separate lists
    dummy = Node(0)
    copy_curr = dummy
    curr = head
    
    while curr:
        copy_curr.next = curr.next
        curr.next = curr.next.next
        copy_curr = copy_curr.next
        curr = curr.next
    
    return dummy.next
\`\`\``,
      quiz: [
        {
          id: 'q1',
          question:
            'Explain how to find the cycle start after detecting a cycle. Why does the mathematical relationship work?',
          sampleAnswer:
            'After fast and slow meet in a cycle, reset one pointer to head and keep the other at meeting point. Move both one step at a time - they will meet at the cycle start. The math: let distance from head to cycle start be x, distance from start to meeting point be y, and remaining cycle be z. When they meet, slow traveled x+y, fast traveled x+y+z+y (went around once more). Since fast is twice as fast, 2(x+y) = x+y+z+y, simplifying to x = z. So distance from head to start equals distance from meeting point to start. This elegant property lets us find the start with O(1) space. Beautiful application of algebra to pointer manipulation.',
          keyPoints: [
            'After meeting: reset one to head',
            'Both move one step until meeting',
            'They meet at cycle start',
            'Math: x = z (head to start = meeting to start)',
            'O(1) space solution',
          ],
        },
        {
          id: 'q2',
          question:
            'Walk me through the approach to reverse nodes in k-groups. What makes this problem more complex than simple reversal?',
          sampleAnswer:
            'Reversing in k-groups requires tracking multiple boundaries. For each group of k nodes, I reverse that segment, then connect it back to the previous group and forward to the next group. The complexity comes from managing these connections: need to save the node before the group, reverse the k nodes, connect previous group tail to reversed group head, and connect reversed group tail to next group. If fewer than k nodes remain, leave them as-is. I typically use a helper function to reverse k nodes and return new head and tail, making the main logic cleaner. The challenge is correctly updating all pointer connections without losing references.',
          keyPoints: [
            'Reverse each k-node segment',
            'Track boundaries: before, after each group',
            'Connect reversed segments back together',
            'Handle remainder less than k',
            'Helper function for k-node reversal',
          ],
        },
        {
          id: 'q3',
          question:
            'Describe the technique for sorting a linked list. Why is merge sort preferred over quick sort for linked lists?',
          sampleAnswer:
            'Merge sort is ideal for linked lists because it does not require random access. The algorithm: find middle with fast-slow pointers, recursively sort left and right halves, merge sorted halves. This gives O(n log n) time and O(log n) space for recursion. Quick sort needs efficient partitioning which requires random access - linked lists lack this. Partitioning a linked list is O(n) but awkward with many pointer updates. Merge sort natural operations (split, merge) work elegantly with linked list structure. The merge step is especially clean with linked lists - just pointer manipulation, no array copying. This is why merge sort is the go-to for linked list sorting.',
          keyPoints: [
            'Merge sort: O(n log n) time, O(log n) space',
            'Works without random access',
            'Find middle, recursively sort, merge',
            'Quick sort needs random access for efficient partition',
            'Merge step elegant with pointers',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'After detecting a cycle, how do you find where the cycle starts?',
          options: [
            'Use a hash map',
            'Reset one pointer to head, move both one step at a time until they meet',
            'Count all nodes',
            'Reverse the list',
          ],
          correctAnswer: 1,
          explanation:
            'After fast and slow meet in the cycle, reset one pointer to head and keep the other at the meeting point. Move both one step at a time - they will meet at the cycle start due to the mathematical relationship x = z.',
        },
        {
          id: 'mc2',
          question: 'Why is merge sort preferred over quick sort for linked lists?',
          options: [
            'Quick sort is always slower',
            'Merge sort doesn\'t require random access which linked lists lack',
            'Merge sort uses less memory',
            'Quick sort doesn\'t work on lists',
          ],
          correctAnswer: 1,
          explanation:
            'Merge sort works well with linked lists because it doesn\'t require random access. It uses find-middle, recursively sort, and merge operations that work naturally with linked list structure. Quick sort needs efficient random access for partitioning.',
        },
        {
          id: 'mc3',
          question: 'When reversing nodes in k-groups, what is the main challenge?',
          options: [
            'Finding k nodes',
            'Managing boundary connections between reversed groups',
            'Counting nodes',
            'It is actually easy',
          ],
          correctAnswer: 1,
          explanation:
            'The main challenge is correctly managing connections: save node before group, reverse k nodes, connect previous group tail to reversed head, connect reversed tail to next group. Many pointers to track.',
        },
        {
          id: 'mc4',
          question: 'How can you copy a linked list with random pointers in O(N) time and O(1) space?',
          options: [
            'Use a hash map',
            'Interweave copied nodes with original nodes, then separate',
            'Use recursion',
            'It cannot be done in O(1) space',
          ],
          correctAnswer: 1,
          explanation:
            'Create new nodes and interweave them with originals (A->A\'->B->B\'), copy random pointers using the interweaved structure, then separate the two lists. This avoids the hash map.',
        },
        {
          id: 'mc5',
          question: 'What is the time complexity of finding the middle of a linked list using fast/slow pointers?',
          options: [
            'O(1)',
            'O(N)',
            'O(N²)',
            'O(log N)',
          ],
          correctAnswer: 1,
          explanation:
            'The fast/slow pointer technique requires traversing the list once. Fast moves 2 steps while slow moves 1 step, so they traverse N nodes total, giving O(N) time complexity.',
        },
      ],
    },
    {
      id: 'common-pitfalls',
      title: 'Common Pitfalls',
      content: `**Pitfall 1: Not Handling Empty List or Single Node**

❌ **Wrong:**
\`\`\`python
def reverse(head):
    prev = None
    curr = head
    while curr:  # Crashes if head is None
        next_temp = curr.next
        curr.next = prev
        prev = curr
        curr = next_temp
\`\`\`

✅ **Correct:**
\`\`\`python
def reverse(head):
    if not head or not head.next:  # Check edge cases
        return head
    
    prev = None
    curr = head
    while curr:
        next_temp = curr.next
        curr.next = prev
        prev = curr
        curr = next_temp
    return prev
\`\`\`

---

**Pitfall 2: Losing Reference to Next Node**

❌ **Wrong:**
\`\`\`python
curr.next = prev  # Lost reference to next!
curr = curr.next  # Now points to prev, not original next
\`\`\`

✅ **Correct:**
\`\`\`python
next_temp = curr.next  # Save next first
curr.next = prev       # Now safe to modify
curr = next_temp       # Move to original next
\`\`\`

---

**Pitfall 3: Forgetting to Update Head**

❌ **Wrong:**
\`\`\`python
def insert_at_head(head, val):
    new_node = ListNode(val)
    new_node.next = head
    # Forgot to update head!
    return head  # Still returns old head
\`\`\`

✅ **Correct:**
\`\`\`python
def insert_at_head(head, val):
    new_node = ListNode(val)
    new_node.next = head
    return new_node  # Return new head
\`\`\`

---

**Pitfall 4: Off-by-One in Two Pointers**

❌ **Wrong (Finding Kth from End):**
\`\`\`python
# Move first k steps
for _ in range(k):  # Should be k, not k-1 or k+1
    first = first.next
\`\`\`

**Test with example:**
- List: [1,2,3,4,5], k=2
- Should return [4]
- First must be k steps ahead (at None when second is at [4])

---

**Pitfall 5: Not Using Dummy Node When Needed**

❌ **Wrong (Removing Nodes):**
\`\`\`python
def remove_val(head, val):
    # Special case for head
    while head and head.val == val:
        head = head.next
    
    # Regular case
    curr = head
    while curr and curr.next:
        if curr.next.val == val:
            curr.next = curr.next.next
        else:
            curr = curr.next
    return head
\`\`\`

✅ **Correct (With Dummy):**
\`\`\`python
def remove_val(head, val):
    dummy = ListNode(0)
    dummy.next = head
    curr = dummy
    
    while curr.next:
        if curr.next.val == val:
            curr.next = curr.next.next
        else:
            curr = curr.next
    
    return dummy.next  # Uniform logic!
\`\`\`

---

**Pitfall 6: Modifying List While Returning It**

❌ **Wrong:**
\`\`\`python
def merge_lists(l1, l2):
    # Directly modifying input lists
    # Caller might not expect this!
\`\`\`

✅ **Better:**
- Document if you modify in-place
- Or create new nodes if pure function needed`,
      quiz: [
        {
          id: 'q1',
          question:
            'What happens if you forget to check for null before accessing node.next? How do you systematically avoid this?',
          sampleAnswer:
            'Forgetting null checks causes null pointer dereference errors - accessing .next on null crashes. This commonly happens in while loops: if I write "while curr.next" but then access "curr.next.next" without checking if curr.next is null first, I crash when curr.next is the last node. I systematically avoid this by always checking: "if curr and curr.next" before accessing curr.next.next. Another pattern: use "while curr" and only access curr properties, never going ahead without explicit checks. In interviews, I state assumptions: "assuming input is not null" or explicitly handle null at the start. Test mental edge cases: empty list, single node, two nodes.',
          keyPoints: [
            'Null access causes crash',
            'Common in while loops accessing ahead',
            'Check: "if curr and curr.next" before curr.next.next',
            'Use "while curr" and only access curr properties',
            'Test edge cases: empty, single, two nodes',
          ],
        },
        {
          id: 'q2',
          question:
            'Explain the mistake of losing the head pointer. How does a dummy node help prevent this?',
          sampleAnswer:
            'Losing head happens when you move head pointer during traversal or modification, then cannot return the original head. For example, moving head forward in a loop: "head = head.next" loses the start. Without dummy, you need a separate variable to save original head. Dummy node prevents this by keeping a fixed reference before the real head. I work with dummy.next and never move dummy itself. At the end, dummy.next is always the current head, whether modified or not. This is foolproof - I cannot lose the head because dummy always points to it. The dummy acts as an anchor before the list.',
          keyPoints: [
            'Losing head by moving head pointer',
            'Need separate variable without dummy',
            'Dummy: fixed reference before head',
            'Work with dummy.next, never move dummy',
            'Return dummy.next: always current head',
          ],
        },
        {
          id: 'q3',
          question:
            'Describe the pitfall of modifying in-place without considering the implications. When is it problematic?',
          sampleAnswer:
            'Modifying in-place changes the original list, which is problematic when the caller needs the original data, when the list is shared by multiple references, or when the function should be pure. For example, reversing a list in-place destroys the original order. If another part of code holds a reference to a middle node, that reference becomes invalid after modification. In interviews, I clarify: "should I modify in-place or create a new list?" In-place is usually preferred for space efficiency (O(1) vs O(n)), but I mention the trade-off. For some problems like detecting cycles, in-place is necessary. For others like copying, creating new nodes is required.',
          keyPoints: [
            'In-place modifies original: may not be desired',
            'Problematic if: caller needs original, shared references',
            'Destroys original data',
            'In-place: O(1) space, but loses original',
            'Clarify requirements in interview',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What is the most common mistake when working with linked lists?',
          options: [
            'Using too much memory',
            'Not checking for null/None before accessing node.next',
            'Making lists too long',
            'Using wrong variable names',
          ],
          correctAnswer: 1,
          explanation:
            'Forgetting null checks causes null pointer dereference errors. Always check "if curr and curr.next" before accessing curr.next.next to avoid crashes on empty lists or at list boundaries.',
        },
        {
          id: 'mc2',
          question: 'What happens if you reverse a pointer before saving the next reference?',
          options: [
            'Nothing, it works fine',
            'You lose access to the rest of the list',
            'The list gets sorted',
            'Memory leak occurs',
          ],
          correctAnswer: 1,
          explanation:
            'If you do curr.next = prev before saving the original curr.next, you lose the reference to the rest of the list. Always save next with next_temp = curr.next first.',
        },
        {
          id: 'mc3',
          question: 'Why should you use a dummy node when removing nodes from a linked list?',
          options: [
            'It makes the code slower',
            'It eliminates special case handling for removing the head node',
            'It is required by Python',
            'It reduces memory usage',
          ],
          correctAnswer: 1,
          explanation:
            'A dummy node eliminates the need for special case logic when removing the head node. With a dummy, all nodes (including the head at dummy.next) can be handled uniformly.',
        },
        {
          id: 'mc4',
          question: 'What is the off-by-one error in the runner technique for finding kth from end?',
          options: [
            'Moving first pointer k-1 steps instead of k',
            'Moving first pointer k steps instead of k+1 or k-1',
            'Not moving any pointer',
            'Moving both pointers',
          ],
          correctAnswer: 0,
          explanation:
            'The first pointer should move exactly k steps ahead. Common mistakes include k-1 (one too few) or k+1 (one too many). Test with a small example: list [1,2,3,4,5], k=2 should return [4].',
        },
        {
          id: 'mc5',
          question: 'Why might modifying a linked list in-place be problematic?',
          options: [
            'It is always wrong',
            'The caller might need the original data or other parts of code might hold references',
            'It uses more memory',
            'It is slower',
          ],
          correctAnswer: 1,
          explanation:
            'In-place modification destroys the original data, which is problematic if the caller needs it or if other code holds references to nodes. Always clarify whether in-place modification is acceptable.',
        },
      ],
    },
    {
      id: 'interview',
      title: 'Interview Strategy',
      content: `**Recognition Signals:**

**Use Linked List techniques when you see:**
- "Linked list" explicitly mentioned
- "Node" with "next" pointer
- Problems about "reversing", "merging", "detecting cycles"
- "In-place" with O(1) space requirement
- "Find middle", "kth from end"
- "Reorder" or "rearrange" nodes

---

**Problem-Solving Steps:**

**Step 1: Clarify the Problem**
- Singly or doubly linked?
- Can we modify in-place?
- What's the format of input/output?
- Any dummy nodes or sentinel values?

**Step 2: Identify the Pattern**
- **Two pointers?** → Fast & slow, cycle detection, middle finding
- **Reversal?** → Iterative or recursive reverse
- **Merging?** → Merge sorted lists
- **Reordering?** → Find pattern, possibly reverse part
- **Cycle?** → Floyd's algorithm
- **Kth from end?** → Runner technique

**Step 3: Handle Edge Cases**
- Empty list (\`head == None\`)
- Single node (\`head.next == None\`)
- Two nodes
- All same values
- Cycle vs. no cycle

**Step 4: Choose Approach**
- **Iterative** (usually O(1) space)
- **Recursive** (simpler but O(N) space)
- **With/without dummy node**

**Step 5: Trace Through Example**
Draw the list and pointers at each step:
\`\`\`
Initial: [1] → [2] → [3] → None
Step 1:  None ← [1]  [2] → [3] → None
Step 2:  None ← [1] ← [2]  [3] → None
Final:   None ← [1] ← [2] ← [3]
\`\`\`

---

**Interview Communication:**

1. **Clarify:**
   - "Is this a singly or doubly linked list?"
   - "Can I modify the list in place?"

2. **Explain approach:**
   - "I'll use two pointers: fast moves 2 steps, slow moves 1 step."
   - "When fast reaches the end, slow will be at the middle."

3. **Discuss alternatives:**
   - "Recursively would be cleaner but uses O(N) space."
   - "Iteratively is O(1) space but slightly more complex."

4. **Handle edge cases:**
   - "For empty list, I'll return None immediately."
   - "For single node, no work needed, return head."

5. **Complexity analysis:**
   - "Time: O(N) since we traverse once."
   - "Space: O(1) using only pointers, no extra data structures."

---

**Common Follow-ups:**

**Q: Can you do it in O(1) space?**
- Avoid hash sets/maps
- Use pointer manipulation instead

**Q: Can you do it without modifying the input?**
- Create new nodes
- Or note that you're modifying in-place

**Q: What if there are multiple cycles?**
- Typically only one cycle in problems
- But clarify assumptions

**Q: Can you do it recursively?**
- Show both iterative and recursive
- Discuss tradeoffs

---

**Practice Plan:**

1. **Basics (Day 1-2):**
   - Reverse Linked List (iterative & recursive)
   - Find Middle
   - Delete Node

2. **Two Pointers (Day 3-4):**
   - Linked List Cycle
   - Cycle Start
   - Remove Nth From End

3. **Merging/Rearranging (Day 5-6):**
   - Merge Two Lists
   - Merge K Lists
   - Reorder List

4. **Advanced (Day 7):**
   - Palindrome Check
   - Copy List with Random Pointer
   - Reverse in K Groups

5. **Resources:**
   - LeetCode Linked List tag (100+ problems)
   - Draw diagrams for every problem
   - Practice until pointer manipulation is intuitive`,
      quiz: [
        {
          id: 'q1',
          question:
            'How do you recognize that a problem requires linked list techniques? What signals in the problem description tell you?',
          sampleAnswer:
            'Several signals indicate linked list problems. First, obviously "linked list" in the problem statement. Second, sequential access patterns without random access needs - traversing one by one. Third, frequent insertions or deletions, especially at beginning or middle. Fourth, problems involving cycles, finding middle, or nth from end - these are classic linked list patterns. Fifth, memory-constrained problems where you cannot use arrays. The key question: does the problem require pointer manipulation or sequential traversal with O(1) space? If accessing by index is not needed and modifications are frequent, linked list techniques apply. Even if input is array, thinking in linked list terms can help.',
          keyPoints: [
            'Explicit: "linked list" mentioned',
            'Sequential access without random access',
            'Frequent insertions/deletions',
            'Classic patterns: cycles, middle, nth from end',
            'O(1) space pointer manipulation',
          ],
        },
        {
          id: 'q2',
          question:
            'Walk me through your approach to a linked list problem in an interview, from problem statement to explaining your solution.',
          sampleAnswer:
            'First, I clarify: is it singly or doubly linked? Can I modify in-place? Are there cycles? Then I identify the pattern: is it reversal, cycle detection, two pointers, or merging? I explain my approach: "I will use fast and slow pointers to find the middle in one pass". I discuss complexity: "O(n) time for one traversal, O(1) space with just two pointers". Then I draw a small example on paper or whiteboard, showing pointer movements step by step. While coding, I explain: "I initialize dummy node to handle edge cases, then..." After coding, I trace through edge cases: empty list, single node, two nodes. I mention optimizations or alternative approaches. Clear visualization and edge case handling are crucial.',
          keyPoints: [
            'Clarify: singly/doubly, in-place, cycles?',
            'Identify pattern',
            'Explain approach and complexity',
            'Draw example, show pointer movements',
            'Code with explanations',
            'Trace edge cases',
          ],
        },
        {
          id: 'q3',
          question:
            'What are the most common mistakes in linked list problems and how do you avoid them?',
          sampleAnswer:
            'First: null pointer errors from not checking before access. I always use "if curr and curr.next" before curr.next.next. Second: losing head pointer. I use dummy node or save original head. Third: off-by-one in pointer movement - moving too far or not far enough. I trace small examples to verify. Fourth: forgetting edge cases like empty list or single node. I test these mentally before finishing. Fifth: creating cycles accidentally by incorrect pointer updates. I carefully track which pointers I am changing. Sixth: using wrong loop condition, causing infinite loops. I ensure progress in every iteration. My strategy: use dummy nodes, test edge cases, draw examples, and double-check pointer updates.',
          keyPoints: [
            'Null checks: "if curr and curr.next"',
            'Use dummy node to avoid losing head',
            'Trace examples for off-by-one',
            'Test edge cases: empty, single node',
            'Careful pointer updates to avoid cycles',
            'Ensure loop progress to avoid infinite loops',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What signals in a problem description indicate you should use linked list techniques?',
          options: [
            'Array sorting required',
            '"Linked list", cycles, middle/nth from end, frequent insertions/deletions',
            'Matrix operations',
            'String manipulation',
          ],
          correctAnswer: 1,
          explanation:
            'Keywords like "linked list", "cycle", "middle", "nth from end", or scenarios with frequent insertions/deletions without random access needs indicate linked list techniques.',
        },
        {
          id: 'mc2',
          question: 'What should you clarify first in a linked list interview problem?',
          options: [
            'The programming language',
            'Whether it is singly or doubly linked and if in-place modification is allowed',
            'The interviewer\'s name',
            'The company size',
          ],
          correctAnswer: 1,
          explanation:
            'Always clarify whether the list is singly or doubly linked and whether in-place modification is allowed, as this affects your approach and space complexity.',
        },
        {
          id: 'mc3',
          question: 'Why should you draw diagrams when solving linked list problems?',
          options: [
            'To waste time',
            'To visualize pointer movements and avoid mistakes',
            'It is not necessary',
            'To impress the interviewer',
          ],
          correctAnswer: 1,
          explanation:
            'Drawing diagrams helps visualize pointer movements step by step, making it easier to track state and avoid common mistakes like null pointer errors or losing references.',
        },
        {
          id: 'mc4',
          question: 'When asked "Can you do it in O(1) space?", what does this typically mean for linked lists?',
          options: [
            'Use arrays instead',
            'Use pointer manipulation instead of hash sets/maps',
            'Use more memory',
            'It cannot be done',
          ],
          correctAnswer: 1,
          explanation:
            'O(1) space requirement means avoiding auxiliary data structures like hash sets/maps and instead using pointer manipulation techniques like fast/slow pointers or dummy nodes.',
        },
        {
          id: 'mc5',
          question: 'What is the recommended practice progression for linked list mastery?',
          options: [
            'Start with the hardest problems',
            'Start with basics (reverse, middle), then two pointers, then merging, then advanced',
            'Skip basics and do advanced',
            'Only practice one type',
          ],
          correctAnswer: 1,
          explanation:
            'Progress from basic operations (reverse, find middle) to two-pointer techniques (cycles, nth from end), then merging/rearranging, and finally advanced problems. This builds intuition incrementally.',
        },
      ],
    },
  ],
  keyTakeaways: [
    'Linked lists provide O(1) insertion/deletion at known positions but O(N) random access',
    'Two pointers (fast & slow) solve cycle detection, middle finding, and kth from end problems',
    'Dummy nodes simplify edge cases by providing a stable reference before the head',
    'Reversal pattern: save next, reverse current link, move pointers forward',
    'Always save the next reference before modifying curr.next to avoid losing the list',
    "Floyd's cycle detection: fast and slow pointers meet inside cycle if one exists",
    'Runner technique: move first pointer k steps ahead to find kth from end',
    'Consider iterative (O(1) space) vs recursive (O(N) space) based on requirements',
  ],
  relatedProblems: [
    'reverse-linked-list',
    'linked-list-cycle',
    'merge-k-sorted-lists',
  ],
};
