/**
 * Common Pitfalls Section
 */

export const commonpitfallsSection = {
  id: 'common-pitfalls',
  title: 'Common Pitfalls',
  content: `**Pitfall 1: Not Handling Empty List or Single Node**

❌ **Wrong:**
\`\`\`python
def reverse (head):
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
def reverse (head):
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
def insert_at_head (head, val):
    new_node = ListNode (val)
    new_node.next = head
    # Forgot to update head!
    return head  # Still returns old head
\`\`\`

✅ **Correct:**
\`\`\`python
def insert_at_head (head, val):
    new_node = ListNode (val)
    new_node.next = head
    return new_node  # Return new head
\`\`\`

---

**Pitfall 4: Off-by-One in Two Pointers**

❌ **Wrong (Finding Kth from End):**
\`\`\`python
# Move first k steps
for _ in range (k):  # Should be k, not k-1 or k+1
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
def remove_val (head, val):
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
def remove_val (head, val):
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
def merge_lists (l1, l2):
    # Directly modifying input lists
    # Caller might not expect this!
\`\`\`

✅ **Better:**
- Document if you modify in-place
- Or create new nodes if pure function needed`,
};
