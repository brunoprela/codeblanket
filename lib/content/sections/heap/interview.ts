/**
 * Interview Strategy Section
 */

export const interviewSection = {
  id: 'interview',
  title: 'Interview Strategy',
  content: `**Recognition Signals:**

**Use Heap when you see:**
- "Kth largest/smallest"
- "Top K elements"
- "Find median"
- "Merge K sorted..."
- "Schedule/priority"
- "Continuous stream of data"
- "Maintain smallest/largest elements"
- Words like "priority", "efficiently maintain", "running"

---

**Problem-Solving Steps:**

**Step 1: Identify Heap Type**
- **Min heap**: For K largest elements, top K frequent
- **Max heap**: For K smallest elements (or negate in Python)
- **Two heaps**: For median, balance problems

**Step 2: Determine Heap Size**
- **Fixed K**: Maintain heap of size K
- **Growing**: Allow heap to grow (all elements)
- **Balanced**: Two heaps of similar size

**Step 3: Choose Pattern**
- **Top K?** → Single heap of size K
- **Median?** → Two heaps (max + min)
- **Merge K?** → Heap with (value, list_idx, elem_idx)
- **Stream?** → Add to heap continuously

**Step 4: Handle Ties**
- Some problems need secondary sort
- Use tuples: (priority, tie_breaker, data)

**Step 5: Optimize**
- Can you use heapify instead of N pushes?
- Do you need custom comparator?
- Can you avoid unnecessary operations?

---

**Interview Communication:**

**Example: Kth Largest Element**1. **Clarify:**
   - "Is the array sorted or unsorted?"
   - "Can I modify the input array?"
   - "What if k > array length?"

2. **Explain approach:**
   - "I'll use a min heap of size K."
   - "Keep only the K largest elements."
   - "The top of the heap is the Kth largest."

3. **Walk through example:**
   \`\`\`
   nums = [3,2,1,5,6,4], k = 2
   
   Process 3: heap = [3]
   Process 2: heap = [2, 3]
   Process 1: heap = [2, 3] (pop 1, heap full)
   Process 5: heap = [3, 5] (pop 2)
   Process 6: heap = [5, 6] (pop 3)
   Process 4: heap = [5, 6] (4 < 5, don't add)
   
   Result: heap[0] = 5 (2nd largest)
   \`\`\`

4. **Complexity:**
   - "Time: O(N log K) - N insertions, each O(log K)."
   - "Space: O(K) - heap stores K elements."

5. **Compare alternatives:**
   - "Sorting would be O(N log N) time."
   - "Quickselect would be O(N) average but O(N²) worst."
   - "Heap is good middle ground with guaranteed O(N log K)."

---

**Common Follow-ups:**

**Q: Can you optimize space?**
- For top K: Already optimal at O(K)
- For merge K: Can process without storing all

**Q: What if K = N?**
- Heap approach still works
- But simpler to just sort

**Q: Can you handle updates?**
- Yes, but need to re-heapify or track positions
- May need additional data structure

**Q: What about duplicates?**
- Heaps handle duplicates naturally
- Just ensure comparison is well-defined

---

**Practice Plan:**1. **Basics (Day 1-2):**
   - Kth Largest Element
   - Top K Frequent Elements
   - Last Stone Weight

2. **Two Heaps (Day 3-4):**
   - Find Median from Data Stream
   - Sliding Window Median

3. **Merge Problems (Day 5):**
   - Merge K Sorted Lists
   - Merge K Sorted Arrays

4. **Advanced (Day 6-7):**
   - Task Scheduler
   - Meeting Rooms II
   - Ugly Number II

5. **Resources:**
   - LeetCode Heap tag (100+ problems)
   - Understand heapify implementation
   - Practice both min and max heap patterns`,
};
