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
