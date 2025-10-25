/**
 * Introduction to Greedy Algorithms Section
 */

export const introductionSection = {
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
- Dijkstra\'s shortest path (non-negative weights)
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
};
