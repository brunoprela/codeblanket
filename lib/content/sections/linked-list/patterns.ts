/**
 * Essential Linked List Patterns Section
 */

export const patternsSection = {
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
};
