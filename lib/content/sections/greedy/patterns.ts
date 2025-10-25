/**
 * Common Greedy Patterns Section
 */

export const patternsSection = {
  id: 'patterns',
  title: 'Common Greedy Patterns',
  content: `**Pattern 1: Activity Selection / Interval Scheduling**

Select maximum non-overlapping activities.

**Strategy**: Choose activity that finishes earliest.

\`\`\`python
def max_activities (activities):
    # Sort by end time
    activities.sort (key=lambda x: x[1])
    
    count = 1
    last_end = activities[0][1]
    
    for i in range(1, len (activities)):
        if activities[i][0] >= last_end:
            count += 1
            last_end = activities[i][1]
    
    return count
\`\`\`

**Why greedy works**: If we don't choose the earliest-finishing activity, we can always replace it without making things worse.

---

**Pattern 2: Fractional Knapsack**

Maximize value with weight limit (can take fractions).

**Strategy**: Take items with highest value/weight ratio.

\`\`\`python
def fractional_knapsack (items, capacity):
    # Sort by value/weight ratio (descending)
    items.sort (key=lambda x: x[1]/x[0], reverse=True)
    
    total_value = 0
    for weight, value in items:
        if capacity >= weight:
            # Take whole item
            total_value += value
            capacity -= weight
        else:
            # Take fraction
            total_value += value * (capacity / weight)
            break
    
    return total_value
\`\`\`

---

**Pattern 3: Huffman Coding / Minimum Cost**

Build optimal prefix-free code or minimize total cost.

**Strategy**: Always combine two smallest elements.

\`\`\`python
import heapq

def huffman_cost (frequencies):
    heap = list (frequencies)
    heapq.heapify (heap)
    
    total_cost = 0
    while len (heap) > 1:
        freq1 = heapq.heappop (heap)
        freq2 = heapq.heappop (heap)
        
        combined = freq1 + freq2
        total_cost += combined
        heapq.heappush (heap, combined)
    
    return total_cost
\`\`\`

---

**Pattern 4: Minimum Platforms / Jump Game**

Find minimum resources needed or maximum reach.

**Strategy**: Track current capacity and extend when needed.

\`\`\`python
def jump_game (nums):
    max_reach = 0
    for i in range (len (nums)):
        if i > max_reach:
            return False  # Can't reach i
        max_reach = max (max_reach, i + nums[i])
    return True
\`\`\`

---

**Pattern 5: Two-Pointer / Partition**

Partition array to maximize/minimize some property.

**Strategy**: Process from both ends, move based on condition.

\`\`\`python
def container_with_most_water (height):
    left, right = 0, len (height) - 1
    max_area = 0
    
    while left < right:
        width = right - left
        h = min (height[left], height[right])
        max_area = max (max_area, width * h)
        
        # Move pointer with smaller height
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    
    return max_area
\`\`\`

---

**Pattern 6: Sorting + Greedy**

Sort first, then make greedy choices.

**Common sorting criteria:**
- By end time (activity selection)
- By ratio (knapsack)
- By deadline (job scheduling)
- By value (various optimization)`,
};
