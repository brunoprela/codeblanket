/**
 * Advanced Techniques Section
 */

export const advancedSection = {
  id: 'advanced',
  title: 'Advanced Techniques',
  content: `**Technique 1: Finding Cycle Start**

After detecting cycle, find where it starts:

\`\`\`python
def detect_cycle_start (head: ListNode) -> ListNode:
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
def reverse_k_group (head: ListNode, k: int) -> ListNode:
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
    for _ in range (k):
        next_temp = curr.next
        curr.next = prev
        prev = curr
        curr = next_temp
    
    # Recursively reverse remaining
    head.next = reverse_k_group (curr, k)
    
    return prev
\`\`\`

---

**Technique 3: Palindrome Check**

Check if linked list is a palindrome:

\`\`\`python
def is_palindrome (head: ListNode) -> bool:
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

def copy_random_list (head: Node) -> Node:
    """
    Deep copy linked list with random pointers.
    Time: O(N), Space: O(N)
    """
    if not head:
        return None
    
    # Step 1: Create copy nodes interleaved
    curr = head
    while curr:
        copy = Node (curr.val)
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
};
