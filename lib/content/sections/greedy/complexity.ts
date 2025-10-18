/**
 * Complexity Analysis Section
 */

export const complexitySection = {
  id: 'complexity',
  title: 'Complexity Analysis',
  content: `**Time Complexity Patterns:**

| Pattern | Complexity | Why |
|---------|------------|-----|
| Sort + Greedy | O(n log n) | Sorting dominates |
| Heap-based | O(n log n) | n operations on heap |
| Single Pass | O(n) | One iteration |
| Two Pointer | O(n) | Linear scan |
| Priority Queue | O(n log k) | k heap operations |

---

**Space Complexity:**

**Usually O(1) to O(n):**
- O(1): In-place greedy (jump game)
- O(n): Sorting auxiliary space
- O(n): Heap for k elements

---

**Common Complexities:**

**Activity Selection:**
- Time: O(n log n) for sorting
- Space: O(1) or O(n) for sorting

**Fractional Knapsack:**
- Time: O(n log n) for sorting by ratio
- Space: O(n) for items

**Huffman Coding:**
- Time: O(n log n) for heap operations
- Space: O(n) for heap

**Jump Game:**
- Time: O(n) single pass
- Space: O(1) no extra space

**Two Pointer (Container):**
- Time: O(n) single pass
- Space: O(1)

---

**Optimization Tips:**

**1. Avoid Unnecessary Sorting**
If data already sorted, skip sort step.

**2. Use Heap Instead of Repeated Sorting**
Heap operations O(log n) vs re-sorting O(n log n).

**3. Early Termination**
Stop when answer found or impossible.

**4. In-place Operations**
Modify input instead of creating copies.`,
};
