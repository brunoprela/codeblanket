/**
 * Array Fundamentals & Patterns Section
 */

export const arraysSection = {
  id: 'arrays',
  title: 'Array Fundamentals & Patterns',
  content: `**What is an Array?**
A contiguous block of memory storing elements of the same type, with O(1) access by index.

**Core Operations & Complexity:**
- **Access by index:** O(1)
- **Insert/Delete at end:** O(1) amortized
- **Insert/Delete at beginning/middle:** O(n) - requires shifting
- **Search unsorted:** O(n)
- **Search sorted:** O(log n) with binary search

**Common Array Patterns:**

**1. Two Pointers** (covered in separate module)
- Opposite direction for sorted arrays
- Same direction for in-place modifications

**2. Sliding Window** (covered in separate module)
- Fixed or variable size subarrays
- Running calculations over ranges

**3. Prefix Sum**
Build cumulative sum array for range queries:
\`\`\`python
prefix[i] = prefix[i-1] + arr[i]
range_sum (l, r) = prefix[r] - prefix[l-1]
\`\`\`

**4. Kadane's Algorithm**
Maximum subarray sum in O(n):
\`\`\`python
max_current = max_global = arr[0]
for i in range(1, len (arr)):
    max_current = max (arr[i], max_current + arr[i])
    max_global = max (max_global, max_current)
\`\`\`

**5. In-Place Reversal**
\`\`\`python
left, right = 0, len (arr) - 1
while left < right:
    arr[left], arr[right] = arr[right], arr[left]
    left += 1
    right -= 1
\`\`\``,
};
