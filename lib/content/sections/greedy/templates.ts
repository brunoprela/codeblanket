/**
 * Code Templates Section
 */

export const templatesSection = {
  id: 'templates',
  title: 'Code Templates',
  content: `**Template 1: Activity Selection**
\`\`\`python
def max_activities (intervals):
    intervals.sort (key=lambda x: x[1])  # Sort by end
    
    count = 1
    last_end = intervals[0][1]
    
    for start, end in intervals[1:]:
        if start >= last_end:
            count += 1
            last_end = end
    
    return count
\`\`\`

---

**Template 2: Greedy with Heap**
\`\`\`python
import heapq

def greedy_heap (items):
    heap = list (items)
    heapq.heapify (heap)
    
    result = 0
    while len (heap) > 1:
        smallest1 = heapq.heappop (heap)
        smallest2 = heapq.heappop (heap)
        
        combined = smallest1 + smallest2
        result += combined
        heapq.heappush (heap, combined)
    
    return result
\`\`\`

---

**Template 3: Jump Game (Max Reach)**
\`\`\`python
def can_jump (nums):
    max_reach = 0
    
    for i in range (len (nums)):
        if i > max_reach:
            return False
        max_reach = max (max_reach, i + nums[i])
    
    return max_reach >= len (nums) - 1
\`\`\`

---

**Template 4: Two Pointer Greedy**
\`\`\`python
def two_pointer_greedy (arr):
    left, right = 0, len (arr) - 1
    result = 0
    
    while left < right:
        # Calculate current value
        current = some_function (arr[left], arr[right])
        result = max (result, current)
        
        # Move pointer based on greedy choice
        if arr[left] < arr[right]:
            left += 1
        else:
            right -= 1
    
    return result
\`\`\`

---

**Template 5: Sort + Greedy**
\`\`\`python
def sort_greedy (items):
    # Sort by some criteria
    items.sort (key=lambda x: your_key (x))
    
    result = []
    for item in items:
        if can_take (item):
            result.append (item)
    
    return result
\`\`\``,
};
