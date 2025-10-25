/**
 * Code Templates Section
 */

export const templatesSection = {
  id: 'templates',
  title: 'Code Templates',
  content: `**Template 1: Basic Traversal**
\`\`\`python
def traverse (head: ListNode) -> None:
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
def two_pointer_pattern (head: ListNode) -> ListNode:
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
def dummy_node_pattern (head: ListNode) -> ListNode:
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
def reverse_list (head: ListNode) -> ListNode:
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
def reverse_list_recursive (head: ListNode) -> ListNode:
    """
    Recursive reversal pattern.
    """
    # Base case
    if not head or not head.next:
        return head
    
    # Recursive case
    new_head = reverse_list_recursive (head.next)
    
    # Reverse the link
    head.next.next = head
    head.next = None
    
    return new_head
\`\`\`

**Template 6: Merge Two Lists**
\`\`\`python
def merge_two_lists (l1: ListNode, l2: ListNode) -> ListNode:
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
def kth_from_end (head: ListNode, k: int) -> ListNode:
    """
    Find kth node from the end.
    """
    first = second = head
    
    # Move first k steps ahead
    for _ in range (k):
        if not first:
            return None
        first = first.next
    
    # Move both until first reaches end
    while first:
        first = first.next
        second = second.next
    
    return second
\`\`\``,
};
