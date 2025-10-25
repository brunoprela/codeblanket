/**
 * Common Sorting Interview Patterns Section
 */

export const interviewproblemsSection = {
  id: 'interview-problems',
  title: 'Common Sorting Interview Patterns',
  content: `**Pattern 1: Custom Sorting / Comparators**

Often you need to sort by custom criteria:

\`\`\`python
# Sort by multiple criteria
students = [("Alice", 85), ("Bob", 90), ("Charlie", 85)]
# Sort by grade desc, then name asc
students.sort (key=lambda x: (-x[1], x[0]))
# Result: [("Bob", 90), ("Alice", 85), ("Charlie", 85)]
\`\`\`

**Pattern 2: Sorting as a Preprocessing Step**

Many problems become easier after sorting:

\`\`\`python
# Find if any two numbers sum to target
def two_sum_sorted (arr, target):
    arr.sort()  # O(n log n)
    left, right = 0, len (arr) - 1
    while left < right:  # O(n)
        curr_sum = arr[left] + arr[right]
        if curr_sum == target:
            return True
        elif curr_sum < target:
            left += 1
        else:
            right -= 1
    return False
# Total: O(n log n)
\`\`\`

**Pattern 3: Partial Sorting**

Sometimes you don't need a complete sort:

\`\`\`python
# Kth largest element
def findKthLargest (nums, k):
    import heapq
    return heapq.nlargest (k, nums)[-1]
# O(n log k) vs O(n log n) for full sort
\`\`\`

**Pattern 4: Merge Sorted Arrays/Lists**

\`\`\`python
def merge_sorted_arrays (arr1, arr2):
    result = []
    i = j = 0
    while i < len (arr1) and j < len (arr2):
        if arr1[i] <= arr2[j]:
            result.append (arr1[i])
            i += 1
        else:
            result.append (arr2[j])
            j += 1
    result.extend (arr1[i:])
    result.extend (arr2[j:])
    return result
# O(n + m) - linear time!
\`\`\`

**Pattern 5: In-Place Sorting with Constraints**

\`\`\`python
# Sort colors (Dutch National Flag)
# Array contains only 0s, 1s, 2s - sort in-place in one pass
def sort_colors (nums):
    low = mid = 0
    high = len (nums) - 1
    
    while mid <= high:
        if nums[mid] == 0:
            nums[low], nums[mid] = nums[mid], nums[low]
            low += 1
            mid += 1
        elif nums[mid] == 1:
            mid += 1
        else:  # nums[mid] == 2
            nums[mid], nums[high] = nums[high], nums[mid]
            high -= 1
    return nums
# O(n) time, O(1) space
\`\`\`

**Pattern 6: Finding Duplicates After Sorting**

\`\`\`python
def find_duplicates (arr):
    arr.sort()  # O(n log n)
    duplicates = []
    for i in range(1, len (arr)):  # O(n)
        if arr[i] == arr[i-1] and (not duplicates or duplicates[-1] != arr[i]):
            duplicates.append (arr[i])
    return duplicates
\`\`\`

**Pattern 7: Interval Problems**

\`\`\`python
# Merge overlapping intervals
def merge_intervals (intervals):
    if not intervals:
        return []
    
    intervals.sort (key=lambda x: x[0])  # Sort by start time
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:  # Overlapping
            last[1] = max (last[1], current[1])
        else:
            merged.append (current)
    
    return merged
\`\`\`

**Common Interview Questions:**

1. **"Implement quick sort"** → Know the partition logic
2. **"Find Kth largest element"** → Quick select or heap
3. **"Merge K sorted lists"** → Min heap
4. **"Sort array with limited values (0,1,2)"** → Counting or Dutch flag
5. **"Is array almost sorted?"** → Check inversions, consider insertion sort
6. **"Sort linked list"** → Merge sort (no random access)
7. **"External sort large file"** → External merge sort

**Red Flags to Avoid:**

❌ Using bubble sort in production code
❌ Not considering stability when it matters
❌ Sorting when a heap or partial sort would suffice
❌ Forgetting that built-in sort is often O(n log n)
❌ Not asking about data properties (size, range, distribution)

**Quick Reference:**

| Need | Algorithm | Complexity |
|------|-----------|------------|
| General purpose | Quick Sort / Timsort | O(n log n) |
| Stability required | Merge Sort / Timsort | O(n log n) |
| Memory constrained | Heap Sort | O(n log n), O(1) space |
| Small data | Insertion Sort | O(n²) but fast |
| Nearly sorted | Insertion Sort / Timsort | O(n) best |
| Top-k elements | Heap | O(n log k) |
| Small integer range | Counting Sort | O(n + k) |
| Fixed-length keys | Radix Sort | O(d × n) |`,
};
