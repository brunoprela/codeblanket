import { Module } from '@/lib/types';

export const greedyModule: Module = {
  id: 'greedy',
  title: 'Greedy',
  description:
    'Master greedy algorithms that make locally optimal choices to find global optima.',
  icon: '🎯',
  timeComplexity: 'Often O(n log n) with sorting',
  spaceComplexity: 'Usually O(1) to O(n)',
  sections: [
    {
      id: 'introduction',
      title: 'Introduction to Greedy Algorithms',
      content: `A **greedy algorithm** makes the locally optimal choice at each step, hoping to find a global optimum. Unlike dynamic programming which considers all possibilities, greedy makes one choice and never looks back.

**Core Principle:**

> "Make the choice that looks best right now, without worrying about future consequences."

**Greedy Choice Property:**
A globally optimal solution can be arrived at by making locally optimal (greedy) choices.

**Optimal Substructure:**
An optimal solution contains optimal solutions to subproblems.

---

**When Greedy Works:**

Greedy algorithms work when:
1. **Greedy choice property** holds
2. **Optimal substructure** exists
3. No need to reconsider past choices

**Examples where greedy works:**
- Dijkstra's shortest path (non-negative weights)
- Minimum spanning tree (Prim's, Kruskal's)
- Huffman coding
- Activity selection
- Coin change (specific denominations)

**Examples where greedy fails:**
- Knapsack problem (fractional works, 0/1 doesn't)
- Coin change (arbitrary denominations)
- Longest path in graph
- Traveling salesman problem

---

**Greedy vs Dynamic Programming:**

| Aspect | Greedy | Dynamic Programming |
|--------|--------|---------------------|
| Approach | Make one choice | Try all choices |
| Look back | Never | Yes (memoization) |
| Guarantee | Sometimes optimal | Always optimal |
| Speed | Usually faster | Usually slower |
| Problems | Fewer | More general |

**Example - Coin Change:**

Greedy **works** for US coins [25, 10, 5, 1]:
- For 63 cents: 25 + 25 + 10 + 1 + 1 + 1 = 6 coins ✓

Greedy **fails** for coins [25, 20, 5, 1]:
- For 40 cents:
  - Greedy: 25 + 5 + 5 + 5 = 4 coins
  - Optimal: 20 + 20 = 2 coins ✗

---

**Common Greedy Strategies:**

1. **Earliest First**: Select task finishing earliest
2. **Latest First**: Select task starting latest
3. **Largest First**: Select largest available item
4. **Smallest First**: Select smallest available item
5. **Best Ratio**: Select best value/weight ratio
6. **Closest First**: Select nearest/closest element`,
    },
    {
      id: 'patterns',
      title: 'Common Greedy Patterns',
      content: `**Pattern 1: Activity Selection / Interval Scheduling**

Select maximum non-overlapping activities.

**Strategy**: Choose activity that finishes earliest.

\`\`\`python
def max_activities(activities):
    # Sort by end time
    activities.sort(key=lambda x: x[1])
    
    count = 1
    last_end = activities[0][1]
    
    for i in range(1, len(activities)):
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
def fractional_knapsack(items, capacity):
    # Sort by value/weight ratio (descending)
    items.sort(key=lambda x: x[1]/x[0], reverse=True)
    
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

def huffman_cost(frequencies):
    heap = list(frequencies)
    heapq.heapify(heap)
    
    total_cost = 0
    while len(heap) > 1:
        freq1 = heapq.heappop(heap)
        freq2 = heapq.heappop(heap)
        
        combined = freq1 + freq2
        total_cost += combined
        heapq.heappush(heap, combined)
    
    return total_cost
\`\`\`

---

**Pattern 4: Minimum Platforms / Jump Game**

Find minimum resources needed or maximum reach.

**Strategy**: Track current capacity and extend when needed.

\`\`\`python
def jump_game(nums):
    max_reach = 0
    for i in range(len(nums)):
        if i > max_reach:
            return False  # Can't reach i
        max_reach = max(max_reach, i + nums[i])
    return True
\`\`\`

---

**Pattern 5: Two-Pointer / Partition**

Partition array to maximize/minimize some property.

**Strategy**: Process from both ends, move based on condition.

\`\`\`python
def container_with_most_water(height):
    left, right = 0, len(height) - 1
    max_area = 0
    
    while left < right:
        width = right - left
        h = min(height[left], height[right])
        max_area = max(max_area, width * h)
        
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
      codeExample: `from typing import List
import heapq


def activity_selection(start: List[int], end: List[int]) -> int:
    """
    Maximum non-overlapping activities.
    Time: O(n log n), Space: O(n)
    """
    activities = list(zip(start, end))
    activities.sort(key=lambda x: x[1])
    
    count = 1
    last_end = activities[0][1]
    
    for s, e in activities[1:]:
        if s >= last_end:
            count += 1
            last_end = e
    
    return count


def fractional_knapsack(
    values: List[int], weights: List[int], capacity: int
) -> float:
    """
    Fractional knapsack - greedy works.
    Time: O(n log n), Space: O(n)
    """
    items = [(v/w, w, v) for v, w in zip(values, weights)]
    items.sort(reverse=True)
    
    total = 0.0
    for ratio, weight, value in items:
        if capacity >= weight:
            total += value
            capacity -= weight
        else:
            total += ratio * capacity
            break
    
    return total


def min_cost_connect_ropes(ropes: List[int]) -> int:
    """
    Huffman-style: always combine smallest.
    Time: O(n log n), Space: O(n)
    """
    heapq.heapify(ropes)
    total_cost = 0
    
    while len(ropes) > 1:
        first = heapq.heappop(ropes)
        second = heapq.heappop(ropes)
        
        cost = first + second
        total_cost += cost
        heapq.heappush(ropes, cost)
    
    return total_cost


def jump_game(nums: List[int]) -> bool:
    """
    Can reach end of array?
    Time: O(n), Space: O(1)
    """
    max_reach = 0
    for i in range(len(nums)):
        if i > max_reach:
            return False
        max_reach = max(max_reach, i + nums[i])
    return True


def jump_game_ii(nums: List[int]) -> int:
    """
    Minimum jumps to reach end.
    Time: O(n), Space: O(1)
    """
    jumps = 0
    current_end = 0
    farthest = 0
    
    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])
        
        if i == current_end:
            jumps += 1
            current_end = farthest
    
    return jumps`,
    },
    {
      id: 'proof',
      title: 'Proving Greedy Correctness',
      content: `**How to Prove Greedy Works:**

**Method 1: Exchange Argument**

Show that any non-greedy solution can be converted to greedy without losing quality.

**Example - Activity Selection:**

*Claim*: Choosing earliest-finishing activity is optimal.

*Proof*:
1. Let optimal solution select activities A = {a₁, a₂, ..., aₖ}
2. Let greedy select activities G = {g₁, g₂, ..., gₘ}
3. If a₁ ≠ g₁:
   - Replace a₁ with g₁ in A
   - g₁ finishes earlier, so no conflicts with rest
   - Still valid, same size
4. Repeat for all activities
5. Therefore greedy is optimal

---

**Method 2: Greedy Stays Ahead**

Show greedy is always "ahead" of any other solution.

**Example - Fractional Knapsack:**

*Claim*: Taking highest value/weight ratio first is optimal.

*Proof*:
1. At each step, greedy has ≥ value as any other solution
2. After k items, greedy value ≥ optimal value with same weight
3. Greedy "stays ahead" throughout
4. Therefore greedy is optimal

---

**Method 3: Structural Induction**

Show optimal solution has greedy structure.

**Steps:**
1. **Base case**: Greedy optimal for smallest input
2. **Induction**: If greedy optimal for size n, prove for n+1
3. Show adding greedy choice maintains optimality

---

**Common Greedy Proof Mistakes:**

**❌ Wrong: "Greedy looks optimal"**
Need formal proof, not intuition.

**❌ Wrong: "Greedy works for examples"**
Examples don't prove correctness.

**✓ Right: Exchange argument or stays-ahead proof**

---

**When Greedy Fails:**

If you can't prove greedy with above methods, it probably doesn't work. Use DP instead.

**Example - 0/1 Knapsack:**

Greedy (by ratio) fails:
- Items: (weight, value) = (10, 60), (20, 100), (30, 120)
- Capacity: 50
- Greedy by ratio: 60 + 100 = 160
- Optimal: 100 + 120 = 220 ✗

Need DP for 0/1 knapsack!`,
    },
    {
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
    },
    {
      id: 'templates',
      title: 'Code Templates',
      content: `**Template 1: Activity Selection**
\`\`\`python
def max_activities(intervals):
    intervals.sort(key=lambda x: x[1])  # Sort by end
    
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

def greedy_heap(items):
    heap = list(items)
    heapq.heapify(heap)
    
    result = 0
    while len(heap) > 1:
        smallest1 = heapq.heappop(heap)
        smallest2 = heapq.heappop(heap)
        
        combined = smallest1 + smallest2
        result += combined
        heapq.heappush(heap, combined)
    
    return result
\`\`\`

---

**Template 3: Jump Game (Max Reach)**
\`\`\`python
def can_jump(nums):
    max_reach = 0
    
    for i in range(len(nums)):
        if i > max_reach:
            return False
        max_reach = max(max_reach, i + nums[i])
    
    return max_reach >= len(nums) - 1
\`\`\`

---

**Template 4: Two Pointer Greedy**
\`\`\`python
def two_pointer_greedy(arr):
    left, right = 0, len(arr) - 1
    result = 0
    
    while left < right:
        # Calculate current value
        current = some_function(arr[left], arr[right])
        result = max(result, current)
        
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
def sort_greedy(items):
    # Sort by some criteria
    items.sort(key=lambda x: your_key(x))
    
    result = []
    for item in items:
        if can_take(item):
            result.append(item)
    
    return result
\`\`\``,
    },
    {
      id: 'interview',
      title: 'Interview Strategy',
      content: `**Recognition Signals:**

**Use Greedy when you see:**
- "Maximum", "minimum", "optimal"
- Scheduling, intervals
- "Can reach", "jump to"
- Sorted input or easy to sort
- Local choice seems to work
- No need to reconsider choices

---

**Problem-Solving Steps:**

**Step 1: Identify Greedy Potential (2 min)**
- Does local optimal lead to global?
- Can I make choice without looking back?
- Is there clear "best" choice each step?

**Step 2: Define Greedy Choice (3 min)**
- What's the greedy criterion?
  - Earliest finish?
  - Highest ratio?
  - Largest/smallest?
- How to select at each step?

**Step 3: Sort if Needed (1 min)**
- What sorting key?
- Ascending or descending?

**Step 4: Prove Correctness (5 min)**
- Exchange argument
- Stays ahead
- Can you find counterexample?

**Step 5: Implement (8 min)**

**Step 6: Test Edge Cases (3 min)**
- Empty input
- Single element
- All same
- Already sorted

---

**Interview Communication:**

**Example: Jump Game**

*Interviewer: Can you reach last index?*

**You:**

1. **Clarify:**
   - "Is array always valid (no negatives)?"
   - "Can values be 0?"

2. **Approach:**
   - "I'll use greedy - track maximum reachable index."
   - "At each position, update max reach."
   - "If current position > max reach, can't continue."

3. **Why Greedy:**
   - "Making locally optimal choice (max reach) is globally optimal."
   - "Never need to backtrack."

4. **Complexity:**
   - "Time: O(n) - single pass."
   - "Space: O(1) - only track one variable."

---

**Common Mistakes:**

**1. Assuming Greedy Works**
Always verify or prove!

**2. Wrong Sorting Key**
Activity selection: sort by END, not start.

**3. Not Considering Counterexamples**
Test greedy on small examples first.

**4. Confusing with DP**
If greedy fails, switch to DP.

---

**Red Flags (Greedy Won't Work):**

- "Longest path" (try DP)
- "All possible ways" (try DP)
- "Can't sort" (greedy harder)
- "Need to reconsider" (not greedy)

---

**Practice Progression:**

**Week 1: Basics**
- Jump Game
- Best Time to Buy/Sell Stock
- Maximum Subarray

**Week 2: Intervals**
- Meeting Rooms II
- Non-overlapping Intervals
- Minimum Arrows

**Week 3: Advanced**
- Gas Station
- Jump Game II
- Task Scheduler`,
    },
  ],
  keyTakeaways: [
    'Greedy makes locally optimal choice without reconsidering, hoping for global optimum',
    'Works when greedy choice property + optimal substructure exist',
    'Prove correctness with exchange argument or "stays ahead" method',
    'Often O(n log n) due to sorting, but some greedy algorithms are O(n)',
    'Activity selection: sort by end time, select non-overlapping earliest finish',
    'Fractional knapsack: sort by value/weight ratio, greedy works; 0/1 knapsack needs DP',
    'Huffman/merge problems: always combine two smallest (use min heap)',
    'Jump game patterns: track maximum reachable index in single pass',
  ],
  relatedProblems: ['jump-game', 'best-time-to-buy-sell-stock', 'gas-station'],
};
