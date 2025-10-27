/**
 * Proving Greedy Correctness Section
 */

export const proofSection = {
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

**Steps:**1. **Base case**: Greedy optimal for smallest input
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
};
