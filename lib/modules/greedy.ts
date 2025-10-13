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
            quiz: [
                {
                    id: 'q1',
                    question:
                        'Explain what greedy algorithms are and when they produce optimal solutions. What makes a problem suitable for greedy?',
                    sampleAnswer:
                        'Greedy algorithms make locally optimal choices at each step hoping to find global optimum. At each decision point, choose what looks best right now without considering future consequences. Greedy works when problem has greedy choice property (local optimum leads to global) and optimal substructure (optimal solution contains optimal solutions to subproblems). For example, making change with coins: always take largest coin possible. Activity selection: always pick earliest ending activity. Greedy fails when local optimum does not guarantee global optimum. For example, 0/1 knapsack needs DP because taking highest value item first might leave no room for better combination. Key: greedy is fast O(n log n) but only works for specific problems. Must prove greedy choice is safe.',
                    keyPoints: [
                        'Make locally optimal choice at each step',
                        'Works when: greedy choice property + optimal substructure',
                        'Local optimum must lead to global optimum',
                        'Fast O(n log n) but limited applicability',
                        'Must prove correctness, does not always work',
                    ],
                },
                {
                    id: 'q2',
                    question:
                        'Compare greedy vs dynamic programming. When would you choose each approach?',
                    sampleAnswer:
                        'Greedy makes irreversible choice at each step, no backtracking. DP considers all options, stores results. Greedy is O(n) or O(n log n), DP is O(n²) or higher. Choose greedy when: local optimum provably leads to global (activity selection, Huffman coding), problem has greedy choice property. Choose DP when: need to consider all possibilities (knapsack, LCS), local choice might not be optimal. For example, coin change for making value: greedy (always largest coin) works for standard coins but fails for arbitrary denominations - need DP. Activity selection: greedy (earliest end) is provably optimal. The key: greedy is preferred when it works (faster, simpler) but DP is more general. Try to prove greedy works; if cannot, use DP.',
                    keyPoints: [
                        'Greedy: one choice per step, no backtrack',
                        'DP: try all options, memoize results',
                        'Greedy: O(n log n), DP: O(n²) or higher',
                        'Greedy when: local → global provably',
                        'DP when: need all possibilities',
                    ],
                },
                {
                    id: 'q3',
                    question:
                        'Walk me through common greedy patterns. How do you recognize which pattern to apply?',
                    sampleAnswer:
                        'Common patterns: Interval scheduling (earliest end time), knapsack fractional (best ratio), two-pointer (extremes first), sorting (process in order). Recognize by problem structure. Intervals + maximize count → sort by end time. Maximize/minimize with constraints → try sorting. Array with extremes → two pointers. Resource allocation → priority queue. For example, "maximum units on truck with capacity" → greedy by unit density (sort by units per box). "Minimum arrows to burst balloons" → sort intervals by end, count groups. "Assign cookies to children" → sort both, match smallest. The pattern: identify what to sort by and what greedy choice to make. Drawing examples often reveals the pattern.',
                    keyPoints: [
                        'Patterns: interval (end time), fractional (ratio), extremes (two pointers)',
                        'Intervals + count → sort by end',
                        'Maximize/minimize → try sorting',
                        'Resource allocation → priority queue',
                        'Draw examples to reveal pattern',
                    ],
                },
            ],
            multipleChoice: [
                {
                    id: 'mc1',
                    question: 'What is the core principle of greedy algorithms?',
                    options: [
                        'Try all possibilities',
                        'Make the locally optimal choice at each step without reconsidering',
                        'Use memoization',
                        'Divide and conquer',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Greedy algorithms make the locally optimal choice at each step (the choice that looks best right now) and never reconsider past decisions, hoping this leads to a global optimum.',
                },
                {
                    id: 'mc2',
                    question: 'What two properties must hold for a greedy algorithm to work?',
                    options: [
                        'Fast and simple',
                        'Greedy choice property and optimal substructure',
                        'Sorting and hashing',
                        'Recursion and iteration',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Greedy works when: 1) Greedy choice property - locally optimal choices lead to global optimum, 2) Optimal substructure - optimal solution contains optimal solutions to subproblems.',
                },
                {
                    id: 'mc3',
                    question: 'Why does greedy fail for coin change with arbitrary denominations?',
                    options: [
                        'Too slow',
                        'Locally optimal choice (largest coin) may not lead to global optimum',
                        'Uses too much memory',
                        'Cannot handle coins',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'For coins [25,20,5,1] and amount 40: greedy takes 25 first (locally best), needs 4 coins total. Optimal is 20+20=2 coins. The locally optimal choice doesn\'t guarantee global optimum.',
                },
                {
                    id: 'mc4',
                    question: 'How does greedy compare to dynamic programming?',
                    options: [
                        'Greedy is always better',
                        'Greedy is faster but works on fewer problems; DP is slower but always finds optimal',
                        'They are the same',
                        'DP is always better',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Greedy makes one choice per step (faster) but only works when greedy choice property holds. DP tries all choices (slower) but always finds optimal solution for problems with optimal substructure.',
                },
                {
                    id: 'mc5',
                    question: 'What is a common greedy strategy for activity/interval selection?',
                    options: [
                        'Select randomly',
                        'Select activity finishing earliest (earliest deadline first)',
                        'Select longest activity',
                        'Select activity starting latest',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Activity selection: sort by end time, greedily select activities finishing earliest. This maximizes remaining time for future activities, giving optimal non-overlapping selection.',
                },
            ],
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
            quiz: [
                {
                    id: 'q1',
                    question:
                        'Explain the activity selection pattern. Why does sorting by end time give optimal solution?',
                    sampleAnswer:
                        'Activity selection maximizes non-overlapping activities. Greedy: sort by end time, pick activity if it starts after last picked ends. Why optimal? Finishing early leaves maximum room for future activities. Proof by exchange argument: suppose optimal solution differs from greedy. Replace first activity in optimal with greedy choice (earliest ending). This cannot make solution worse because earliest ending leaves more room. Continue replacing - greedy matches optimal. For example, activities with ends [3,5,6,8]: picking 3 leaves room for 5,6,8. Picking 5 first blocks 3. Earliest end is provably best choice. This pattern extends to interval scheduling, meeting rooms.',
                    keyPoints: [
                        'Maximize non-overlapping activities',
                        'Sort by end time, greedy select',
                        'Earliest end leaves most room',
                        'Proof: exchange argument shows optimality',
                        'Pattern: interval problems with count',
                    ],
                },
                {
                    id: 'q2',
                    question:
                        'Describe the fractional knapsack pattern. How does it differ from 0/1 knapsack?',
                    sampleAnswer:
                        'Fractional knapsack: can take partial items, maximize value with weight limit. Greedy: sort by value/weight ratio, take items in order, take fraction of last if needed. This is optimal for fractional. 0/1 knapsack: must take whole items, greedy fails. For example, capacity 50, items: [60kg,$100], [10kg,$20], [20kg,$30]. Fractional: take all of 10kg ($20), all of 20kg ($30), 20kg of 60kg ($33.33) = $83.33 optimal. Greedy by ratio works. 0/1: cannot take fraction of 60kg item, need DP. Key difference: fractional allows splitting, making greedy safe. 0/1 needs considering combinations. Fractional is O(n log n), 0/1 is O(n×W).',
                    keyPoints: [
                        'Fractional: can split items, greedy optimal',
                        'Sort by value/weight ratio',
                        '0/1: must take whole, greedy fails, need DP',
                        'Splitting makes greedy safe',
                        'Fractional O(n log n), 0/1 O(n×W)',
                    ],
                },
                {
                    id: 'q3',
                    question:
                        'Walk me through the jump game greedy pattern. Why does tracking farthest reachable work?',
                    sampleAnswer:
                        'Jump game: reach end of array, each element is max jump length. Greedy: iterate, track farthest reachable from current position. At each index i, farthest = max(farthest, i + nums[i]). If current position exceeds farthest, cannot proceed. If farthest >= end, can reach. For minimum jumps: track current range end, when reach end of range, increment jumps and update range to farthest. Works because if we can reach index i, we can reach all indices before i. So tracking farthest from all reachable positions guarantees we find if end is reachable. For minimum jumps, greedily extending range as far as possible minimizes jump count. This is provably optimal.',
                    keyPoints: [
                        'Track farthest reachable position',
                        'At i: farthest = max(farthest, i + nums[i])',
                        'Can reach end if farthest >= end',
                        'Min jumps: greedily extend range',
                        'Provably optimal: can reach implies can reach all before',
                    ],
                },
            ],
            multipleChoice: [
                {
                    id: 'mc1',
                    question: 'In activity selection, why sort by end time instead of start time?',
                    options: [
                        'It\'s faster',
                        'Activities finishing earliest leave maximum time for remaining activities',
                        'Random choice',
                        'Easier to implement',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Sorting by end time and selecting earliest-finishing activities maximizes remaining time for future selections, giving optimal non-overlapping count. This is the greedy choice.',
                },
                {
                    id: 'mc2',
                    question: 'What is the greedy choice for fractional knapsack?',
                    options: [
                        'Pick heaviest items first',
                        'Pick items with best value/weight ratio first',
                        'Pick lightest items first',
                        'Random selection',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Fractional knapsack: sort by value/weight ratio, take items with best ratio first (can take fractions). This maximizes value per unit weight, giving optimal solution.',
                },
                {
                    id: 'mc3',
                    question: 'In two-pointer greedy problems, what is the typical strategy?',
                    options: [
                        'Move both pointers together',
                        'Process extremes first (largest/smallest), move pointers based on greedy criterion',
                        'Random pointer movement',
                        'Only move one pointer',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Two-pointer greedy: sort array, use pointers at extremes. Make greedy choice based on which extreme to take (e.g., container with most water: move pointer with smaller height).',
                },
                {
                    id: 'mc4',
                    question: 'What data structure is often used for greedy resource allocation?',
                    options: [
                        'Array',
                        'Priority queue (heap) to efficiently get best choice',
                        'Stack',
                        'Linked list',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Priority queue (heap) maintains best available choice at top, enabling O(log N) greedy selections. Used for scheduling, task allocation, and other resource problems.',
                },
                {
                    id: 'mc5',
                    question: 'Why does greedy work for Huffman coding?',
                    options: [
                        'It\'s simple',
                        'Merging two lowest-frequency nodes minimizes total encoding length (greedy choice property)',
                        'Random',
                        'Always works',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Huffman coding: repeatedly merge two lowest-frequency nodes. This greedy choice minimizes weighted path length (encoding cost), proven optimal by exchange argument.',
                },
            ],
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
            quiz: [
                {
                    id: 'q1',
                    question:
                        'How do you prove a greedy algorithm is correct? What techniques can you use?',
                    sampleAnswer:
                        'Three main proof techniques. First, greedy stays ahead: show greedy is always at least as good as optimal after each step. Second, exchange argument: suppose optimal differs from greedy, exchange optimal choice with greedy choice, show this does not worsen solution. Third, induction: prove greedy works for base case, then show if greedy works for k steps, it works for k+1. For activity selection: exchange argument - replacing earliest end in optimal with greedy choice leaves more room. For Huffman: induction - merging two lowest frequency nodes is optimal at each step. Without proof, greedy might be wrong. For example, greedy fails for 0/1 knapsack but works for fractional. Must prove correctness.',
                    keyPoints: [
                        'Three techniques: stays ahead, exchange, induction',
                        'Stays ahead: greedy ≥ optimal at each step',
                        'Exchange: swap optimal choice with greedy, no worse',
                        'Induction: prove base, then k → k+1',
                        'Without proof, greedy might be wrong',
                    ],
                },
                {
                    id: 'q2',
                    question:
                        'Explain why greedy fails for 0/1 knapsack. What property is missing?',
                    sampleAnswer:
                        'Greedy fails because local choice (highest ratio) does not guarantee global optimum when items are indivisible. Example: capacity 50, items: [10kg,$60 ratio=6], [20kg,$100 ratio=5], [30kg,$120 ratio=4]. Greedy picks 10kg then 20kg for $160. Optimal picks 20kg+30kg for $220. The problem: after picking 10kg item, cannot fit 30kg item, but 30kg item is part of optimal. Greedy lacks look-ahead. Missing property: greedy choice property does not hold - locally best choice (highest ratio) blocks better global combination. For fractional knapsack, can take partial items, so greedy works (take fractions to fill capacity). For 0/1, need DP to consider all item combinations.',
                    keyPoints: [
                        'Local best does not guarantee global when indivisible',
                        'Example: highest ratio blocks better combination',
                        'Lacks look-ahead to see better combos',
                        'Greedy choice property does not hold',
                        'Fractional works (can split), 0/1 needs DP',
                    ],
                },
                {
                    id: 'q3',
                    question:
                        'Describe when sorting enables greedy solutions. What should you sort by?',
                    sampleAnswer:
                        'Sorting reveals structure that makes greedy safe. Sort by: end time for intervals (activity selection), ratio for fractional knapsack, difficulty or deadline for scheduling, size for matching problems. Sorting orders choices so greedy can process best-first. For example, activity selection: after sorting by end time, earliest ending is provably best choice - picking it leaves most room. Without sorting, would need to search for earliest ending each time or might pick wrong activity. Sorting is O(n log n) but enables O(n) greedy processing. Pattern: if greedy should pick "best" choice but best is not obvious, sorting often reveals it. Sort puts related items together, enables greedy invariants.',
                    keyPoints: [
                        'Sorting reveals structure for greedy',
                        'Sort by: end time, ratio, deadline, size',
                        'Orders choices for best-first processing',
                        'Example: sort by end → earliest is provably best',
                        'O(n log n) sort enables O(n) greedy',
                    ],
                },
            ],
            multipleChoice: [
                {
                    id: 'mc1',
                    question: 'What is the exchange argument for proving greedy correctness?',
                    options: [
                        'Test all cases',
                        'Show any non-greedy solution can be converted to greedy without losing quality',
                        'Use induction',
                        'Random testing',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Exchange argument: take any optimal solution, replace non-greedy choices with greedy ones. If quality doesn\'t decrease, greedy is optimal. Shows greedy choice is always safe.',
                },
                {
                    id: 'mc2',
                    question: 'What does "greedy stays ahead" mean?',
                    options: [
                        'Greedy is faster',
                        'At each step, greedy maintains ≥ quality as any other solution',
                        'Greedy uses more memory',
                        'Random property',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Greedy stays ahead: after k steps, greedy solution has quality ≥ any other solution with same resources. Proves greedy remains optimal throughout.',
                },
                {
                    id: 'mc3',
                    question: 'Why does greedy fail for 0/1 knapsack?',
                    options: [
                        'Too slow',
                        'Cannot take fractions - greedy by ratio doesn\'t guarantee optimal',
                        'Uses too much memory',
                        'Cannot handle knapsacks',
                    ],
                    correctAnswer: 1,
                    explanation:
                        '0/1 knapsack: must take whole items. Greedy by ratio fails: might take high-ratio small item instead of optimal large item. Need DP to try all combinations.',
                },
                {
                    id: 'mc4',
                    question: 'What is structural induction for proving greedy?',
                    options: [
                        'Test examples',
                        'Base case + show greedy choice maintains optimality for n+1',
                        'Random testing',
                        'Brute force',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Structural induction: prove greedy optimal for base case, then show adding greedy choice maintains optimality for larger inputs. Builds proof incrementally.',
                },
                {
                    id: 'mc5',
                    question: 'How do you know when greedy won\'t work?',
                    options: [
                        'Always test first',
                        'Cannot prove with exchange/stays-ahead arguments - need DP instead',
                        'Random guess',
                        'Greedy always works',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'If you cannot prove greedy correctness with exchange argument or stays-ahead, greedy likely fails. Use DP for those problems (0/1 knapsack, longest path, etc.).',
                },
            ],
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
            quiz: [
                {
                    id: 'q1',
                    question:
                        'Compare greedy time complexity vs other approaches. Why is greedy usually faster?',
                    sampleAnswer:
                        'Greedy is typically O(n log n) from sorting + O(n) for one pass = O(n log n) total. DP is O(n²) or O(n×W) for knapsack. Backtracking is exponential O(2^n). Greedy is faster because: makes one irreversible choice per step, no backtracking or trying all options. For example, activity selection: greedy is O(n log n), brute force trying all combinations is O(2^n). Coin change: greedy is O(n) for standard coins, DP is O(n×amount). Greedy trades generality for speed - only works for specific problems but when it works, it is fastest. The one-pass nature after sorting makes it efficient. No memoization overhead, no recursion depth.',
                    keyPoints: [
                        'Greedy: O(n log n) sort + O(n) pass',
                        'DP: O(n²), Backtracking: O(2^n)',
                        'Faster: one choice, no backtracking',
                        'Example: activity O(n log n) vs brute O(2^n)',
                        'Trades generality for speed',
                    ],
                },
                {
                    id: 'q2',
                    question:
                        'Explain space complexity of greedy algorithms. Why is it usually O(1) or O(n)?',
                    sampleAnswer:
                        'Greedy usually uses O(1) extra space: just a few variables to track current choice, best value, etc. Sometimes O(n) for: sorting requires space (though often in-place), storing result array, using priority queue. For example, activity selection: O(1) extra (just track last end time). Jump game: O(1) (track farthest reachable). Huffman coding: O(n) for priority queue. Meeting rooms: O(n) for heap. Compared to DP which is O(n) or O(n²) for table, greedy is more space efficient. In-place sorting and single-pass processing keep space low. If problem allows modifying input or returning indices instead of copying data, can achieve O(1).',
                    keyPoints: [
                        'Usually O(1): few variables',
                        'Sometimes O(n): result array, priority queue',
                        'Examples: activity O(1), Huffman O(n)',
                        'vs DP: O(n) or O(n²) for table',
                        'In-place processing keeps space low',
                    ],
                },
                {
                    id: 'q3',
                    question:
                        'When is greedy not the right approach? What signals should make you reconsider?',
                    sampleAnswer:
                        'Reconsider greedy when: local optimum clearly does not lead to global (0/1 knapsack), problem asks for "all possible" not "optimal" (need backtracking), constraints prevent greedy choice (dependencies between choices), optimization involves combinations not sequences. Signals: cannot find greedy choice property proof, simple greedy gives wrong answer on examples, problem is known NP-hard (usually need approximation or DP). For example, if problem asks "count all ways", that is DP or backtracking not greedy. If simple greedy fails on small example, likely need DP. Test greedy on examples first - if fails, abandon greedy. Good practice: always test greedy candidate on edge cases before full implementation.',
                    keyPoints: [
                        'Reconsider: local ≠ global, need "all solutions"',
                        'Signals: cannot prove greedy, wrong on examples',
                        'Known NP-hard usually not greedy',
                        '"Count all ways" → DP or backtracking',
                        'Test on examples first before implementing',
                    ],
                },
            ],
            multipleChoice: [
                {
                    id: 'mc1',
                    question: 'What is the typical complexity of greedy algorithms with sorting?',
                    options: [
                        'O(N)',
                        'O(N log N) - sorting dominates',
                        'O(N²)',
                        'O(log N)',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Most greedy algorithms require O(N log N) sorting followed by O(N) greedy processing, giving O(N log N) total complexity.',
                },
                {
                    id: 'mc2',
                    question: 'What is the space complexity of most greedy algorithms?',
                    options: [
                        'O(N log N)',
                        'O(1) to O(N) - often in-place or linear auxiliary space',
                        'O(N²)',
                        'O(2^N)',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Greedy algorithms typically use O(1) space (in-place like jump game) or O(N) for sorting/auxiliary structures. Much better than DP\'s O(N²) space.',
                },
                {
                    id: 'mc3',
                    question: 'How can you avoid sorting to optimize greedy algorithms?',
                    options: [
                        'Cannot avoid sorting',
                        'Use heap/priority queue for dynamic best-choice selection',
                        'Use arrays only',
                        'Random selection',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'When greedy choice changes dynamically, use priority queue to maintain best choice in O(log N) instead of O(N log N) re-sorting. Useful for streaming/online problems.',
                },
                {
                    id: 'mc4',
                    question: 'What is the complexity of jump game with greedy?',
                    options: [
                        'O(N log N)',
                        'O(N) single pass, O(1) space',
                        'O(N²)',
                        'O(2^N)',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Jump game: one pass tracking farthest reachable position. O(N) time, O(1) space. No sorting needed. This is optimal.',
                },
                {
                    id: 'mc5',
                    question: 'How does greedy complexity compare to DP for same problems?',
                    options: [
                        'Always the same',
                        'Greedy usually faster: O(N log N) vs DP O(N²), but works on fewer problems',
                        'DP is always faster',
                        'Random',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'When greedy works, it\'s typically O(N log N) vs DP\'s O(N²). But greedy only works when greedy choice property holds - DP is more general.',
                },
            ],
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
            quiz: [
                {
                    id: 'q1',
                    question:
                        'Walk me through the greedy interval template. What are the key steps?',
                    sampleAnswer:
                        'Greedy interval template for maximizing count has four steps. First, sort intervals by end time. Second, initialize count and last_end (first interval end). Third, iterate from second interval: if current.start >= last_end (non-overlapping), increment count and update last_end to current.end. Fourth, return count. The key: sorting by end time ensures earliest ending is always processed first, which is provably optimal. For example, [[1,3], [2,5], [4,6]]: sort by end gives [[1,3], [2,5], [4,6]]. Pick [1,3], skip [2,5] (overlaps), pick [4,6] (4 >= 3). Count is 2. This template works for activity selection, non-overlapping intervals, minimum arrows to burst balloons.',
                    keyPoints: [
                        'Step 1: sort by end time',
                        'Step 2: track count and last_end',
                        'Step 3: if non-overlap, increment count',
                        'Earliest ending is provably optimal',
                        'Works: activity, non-overlap, arrows',
                    ],
                },
                {
                    id: 'q2',
                    question:
                        'Explain the two-pointer greedy template. When is this pattern applicable?',
                    sampleAnswer:
                        'Two-pointer greedy processes from both ends toward middle, making greedy choice at each step. Applicable when: array is sorted (or sortable), decision involves comparing extremes, want to match/pair/distribute elements. Template: sort array, left at start, right at end. While left < right: check condition, move pointer based on greedy choice. For example, assign cookies: sort children and cookies, try to satisfy smallest child with smallest adequate cookie. Two pointers enable greedy matching. Container with most water: try largest width first, move pointer with smaller height (greedy choice). The pattern: sorted + extremes + pairing → two pointers. Enables O(n) greedy after O(n log n) sort.',
                    keyPoints: [
                        'Process from both ends toward middle',
                        'Applicable: sorted, compare extremes, matching',
                        'Template: sort, left/right pointers, move based on choice',
                        'Example: assign cookies, container water',
                        'O(n) greedy after O(n log n) sort',
                    ],
                },
                {
                    id: 'q3',
                    question:
                        'Describe the priority queue greedy template. What problems benefit from this?',
                    sampleAnswer:
                        'Priority queue greedy processes items by priority, using heap for efficient access to next best choice. Use when: need to repeatedly pick best available item, items arrive over time, resource allocation with priorities. Template: create max or min heap, add items, repeatedly pop best and process. For example, Huffman coding: repeatedly merge two lowest frequency nodes (min heap). Meeting rooms II: track earliest ending meeting (min heap). Task scheduler: process most frequent task first (max heap). The heap maintains best choice in O(log n) rather than O(n) scan. This is crucial when many greedy choices need to be made. Total complexity: O(n log n) for n heap operations.',
                    keyPoints: [
                        'Use heap for next best choice',
                        'Applicable: repeated picks, arrivals, priorities',
                        'Template: heap, add items, pop best',
                        'Examples: Huffman, meeting rooms, task scheduler',
                        'O(log n) per choice vs O(n) scan',
                    ],
                },
            ],
            multipleChoice: [
                {
                    id: 'mc1',
                    question: 'What is the first step in the activity selection template?',
                    options: [
                        'Count activities',
                        'Sort intervals by end time',
                        'Find maximum',
                        'Random selection',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Activity selection template: 1) Sort by end time, 2) Greedily select non-overlapping activities. Sorting by end enables optimal greedy selection.',
                },
                {
                    id: 'mc2',
                    question: 'In fractional knapsack template, what do you sort by?',
                    options: [
                        'Weight',
                        'Value/weight ratio (descending)',
                        'Value',
                        'Random',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Fractional knapsack: sort by value/weight ratio in descending order. Take items with best ratio first (can take fractions of last item).',
                },
                {
                    id: 'mc3',
                    question: 'In two-pointer greedy template, how do you decide which pointer to move?',
                    options: [
                        'Always move left',
                        'Based on greedy criterion (e.g., move pointer with smaller value)',
                        'Random',
                        'Always move right',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Two-pointer greedy: after processing current pair, move pointer that enables best future choices. E.g., container problem: move pointer with smaller height.',
                },
                {
                    id: 'mc4',
                    question: 'What is common in all greedy templates?',
                    options: [
                        'Use hash maps',
                        'Sort or organize data, make locally optimal choice at each step',
                        'Use recursion',
                        'Random selection',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'All greedy templates: 1) Sort/organize data to reveal greedy structure, 2) Iterate making locally optimal choices. Pattern: prepare data, then greedy loop.',
                },
                {
                    id: 'mc5',
                    question: 'Why is the jump game template so efficient?',
                    options: [
                        'Uses sorting',
                        'Single pass tracking farthest reachable, no sorting needed - O(N) time, O(1) space',
                        'Uses heap',
                        'Random',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Jump game needs no sorting - just track farthest reachable position in one pass. O(N) time, O(1) space. Shows not all greedy needs sorting.',
                },
            ],
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
            quiz: [
                {
                    id: 'q1',
                    question:
                        'How do you recognize a greedy problem? What keywords or patterns signal this?',
                    sampleAnswer:
                        'Keywords: "maximize", "minimize", "optimal", "scheduling", "selection". Patterns: intervals with counting, resource allocation, matching/pairing, sequence optimization. Recognize by: can you make irreversible choice that is locally best? Does sorting reveal structure? For example, "maximize number of meetings" suggests interval greedy. "Minimum cost" with simple constraints suggests greedy. "Assign tasks optimally" suggests sorting and greedy matching. Contrast with: "count all ways" (DP), "find all combinations" (backtracking), "with capacity constraint and indivisible items" (DP knapsack). If problem feels like optimization with obvious best choice at each step, try greedy. But always verify with examples and proof.',
                    keyPoints: [
                        'Keywords: maximize, minimize, optimal, scheduling',
                        'Patterns: intervals, allocation, matching',
                        'Can make irreversible local best choice?',
                        'Does sorting help?',
                        'Verify with examples and proof',
                    ],
                },
                {
                    id: 'q2',
                    question:
                        'Walk me through your greedy interview approach from recognition to verification.',
                    sampleAnswer:
                        'First, recognize greedy from keywords and structure. Second, identify greedy choice: what should I pick at each step? Should I sort? Third, explain approach: sort by X, iterate, make greedy choice Y. Fourth, verify on examples: does it give correct answer? Fifth, attempt to prove: use exchange argument or stays ahead. Sixth, state complexity: O(n log n) sort + O(n) pass. Seventh, code clearly showing greedy choice. Eighth, test edge cases. Ninth, if greedy fails verification, discuss why (like 0/1 knapsack) and mention alternative (DP). This demonstrates: understanding greedy, ability to verify, knowing when it fails. Even failed greedy attempt followed by correct alternative shows depth.',
                    keyPoints: [
                        'Recognize, identify greedy choice',
                        'Explain: sort, iterate, choose',
                        'Verify on examples, attempt proof',
                        'State complexity, code clearly',
                        'If fails, explain why and give alternative',
                    ],
                },
                {
                    id: 'q3',
                    question:
                        'What are common greedy mistakes and how do you avoid them?',
                    sampleAnswer:
                        'First: assuming greedy works without proof (test examples first). Second: sorting by wrong criteria (e.g., start instead of end for activity selection). Third: making wrong greedy choice (not identifying true local optimum). Fourth: forgetting to update state after greedy choice. Fifth: applying greedy to problem that needs DP (0/1 knapsack). Sixth: not handling ties properly in sorting. My strategy: always test on 2-3 examples before implementing, try to prove or find counterexample, know classic failures (0/1 knapsack), sort by the right key (draw examples to verify), clearly identify and document greedy choice. Most mistakes come from wrong greedy choice or applying to wrong problem.',
                    keyPoints: [
                        'Assuming works without testing',
                        'Sorting by wrong key',
                        'Wrong greedy choice',
                        'Applying to problem needing DP',
                        'Test examples, attempt proof, know classic failures',
                    ],
                },
            ],
            multipleChoice: [
                {
                    id: 'mc1',
                    question: 'What keywords signal a greedy problem?',
                    options: [
                        'Array, list, tree',
                        'Maximum, minimum, optimal, scheduling, earliest/latest',
                        'Count all ways',
                        'Longest path',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Keywords like "maximum", "minimum", "optimal", "scheduling", "earliest", "latest", "best" suggest greedy. But verify greedy choice property before implementing!',
                },
                {
                    id: 'mc2',
                    question: 'What should you clarify first in a greedy interview problem?',
                    options: [
                        'Complexity only',
                        'Can I sort? What defines "best"? Any constraints on choices?',
                        'Language preference',
                        'Nothing',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Clarify: Can you sort (modifies input)? What makes a choice "best"? Are there dependencies between choices? These affect whether greedy works.',
                },
                {
                    id: 'mc3',
                    question: 'What is a common mistake when using greedy?',
                    options: [
                        'Sorting correctly',
                        'Assuming greedy works without proof - need exchange argument or stays-ahead',
                        'Good variable names',
                        'Using correct syntax',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Common mistake: assuming greedy works because it "looks right". Always verify with exchange argument, stays-ahead proof, or test counterexamples.',
                },
                {
                    id: 'mc4',
                    question: 'What are red flags that greedy won\'t work?',
                    options: [
                        'Optimization problems',
                        '"Longest path", "all possible ways", "need to reconsider past choices"',
                        'Scheduling problems',
                        'Interval problems',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Red flags: "longest path" (use DP), "all possible ways" (use DP/backtracking), "reconsider choices" (not greedy). Greedy makes irrevocable choices.',
                },
                {
                    id: 'mc5',
                    question: 'What is a good practice progression for greedy?',
                    options: [
                        'Start with hardest',
                        'Week 1: Basics (Jump Game), Week 2: Intervals, Week 3: Advanced (Huffman)',
                        'Random order',
                        'Skip practice',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Progress: Week 1 basics (jump game, stock) → Week 2 intervals (meetings, activity selection) → Week 3 advanced (Huffman, knapsack). Build proof skills.',
                },
            ],
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
