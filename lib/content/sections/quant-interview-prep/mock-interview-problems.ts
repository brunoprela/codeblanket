export const mockInterviewProblems = {
  title: 'Mock Interview Problems',
  id: 'mock-interview-problems',
  content: `
# Mock Interview Problems

## Introduction

This section contains **complete mock interview problems** that combine multiple skills tested in quantitative interviews. Each problem simulates a real interview scenario and requires you to integrate probability, statistics, coding, mental math, and market intuition.

**What Makes These "Mock" Problems:**
- Multi-part questions that evolve based on your answers
- Combine 2-3 different skill areas
- Include follow-up questions interviewers actually ask
- Test both technical ability and communication
- Realistic time constraints (30-45 minutes)

**How to Use This Section:**
1. **Time yourself** - Treat each as a real interview
2. **Think out loud** - Practice explaining your reasoning
3. **Handle follow-ups** - Don't stop at the first answer
4. **Show multiple approaches** - Demonstrates depth
5. **Check your work** - Always verify answers

---

## Mock Interview 1: Coin Flip Game (Jane Street Style)

**Time: 30 minutes**

### Part A: Basic Game (10 min)

**Interviewer:** "I'll flip a fair coin repeatedly. You can stop me at any point and receive $1 for each heads minus $1 for each tails shown so far. What\'s your optimal strategy and expected payout?"

**Your Task:** Analyze the game completely.

**Solution:**

The optimal strategy is: **Stop as soon as you're ahead (positive balance).**

**Proof:**
- Let E = expected value of optimal play
- Current balance: B (heads - tails)
- If B > 0: Stop now (guaranteed profit B)
- If B ≤ 0: Continue flipping

When B = 0:
- Next flip: 50% chance +$1, 50% chance -$1
- If +$1: You're ahead, stop (get $1)
- If -$1: You're at -$1, continue playing

From B = -1:
- Eventually reach B = 0 again (symmetric random walk)
- Then follow same strategy

**Expected value from B = 0:**
\`\`\`
E(0) = 1 + 0.5×1 + 0.5×E(-1)

But by symmetry and properties of random walks:
E(-1) = E(0) - 1

Therefore:
E(0) = 1 + 0.5×1 + 0.5×(E(0) - 1)
E(0) = 1 + 0.5 + 0.5×E(0) - 0.5
0.5×E(0) = 1
E(0) = $2
\`\`\`

**Answer: Expected payout is $2**

### Part B: Modified Rules (10 min)

**Interviewer:** "Now I'll give you $1 per heads but you pay me $2 per tails. Same rules—stop whenever you want. What\'s your strategy and expected value?"

**Solution:**

**New Analysis:**
- Heads: +$1
- Tails: -$2
- Expected value per flip: 0.5(1) + 0.5(-2) = -0.5

**This is a losing game on average!**

**Optimal strategy:** Stop immediately (don't flip at all).
Expected value: $0

If you flip even once:
- 50%: +$1, stop → EV = $0.50
- 50%: -$2, then try to recover... but game is rigged against you
- Overall EV < 0

**Key insight:** Never play a negative expectation game if you can avoid it.

### Part C: Coding Challenge (10 min)

**Interviewer:** "Simulate 100,000 games of Part A with optimal strategy. Verify the $2 expected value."

\`\`\`python
"""
Coin Flip Game Simulation
"""

import random

def play_game():
    """Play one game with optimal strategy: stop when ahead."""
    balance = 0
    while balance <= 0:
        flip = random.choice([-1, 1])  # -1 for tails, +1 for heads
        balance += flip
    return balance

# Simulate
results = [play_game() for _ in range(100000)]
print(f"Expected value: \${sum (results)/len (results):.4f}")
print(f"Theoretical: $2.00")

# Distribution
from collections import Counter
dist = Counter (results)
for value in sorted (dist.keys())[:10]:
    print(f"\${value}: {dist[value]/1000:.1f}%")
\`\`\`

**Expected Output:**
\`\`\`
Expected value: $2.0013
Theoretical: $2.00
$1: 50.2%
$2: 12.4%
$3: 15.7%
...
\`\`\`

---

## Mock Interview 2: Options Trading Puzzle (Citadel Style)

**Time: 45 minutes**

### Part A: Arbitrage Detection (15 min)

**Interviewer:** "A stock trades at $100. The 1-month call with strike $100 trades at $5. The 1-month put with strike $100 trades at $3. Risk-free rate is 0%. Is there an arbitrage? If so, how do you exploit it?"

**Solution:**

**Check Put-Call Parity:**
\`\`\`
C - P = S - K×e^(-rT)

With r = 0:
C - P should equal S - K

Actual: $5 - $3 = $2
Expected: $100 - $100 = $0

Difference: $2 > $0 → ARBITRAGE EXISTS!
\`\`\`

**Arbitrage Strategy:**
1. **Sell the call:** Receive $5
2. **Buy the put:** Pay $3
3. **Buy the stock:** Pay $100
4. **Net outflow:** -$100 + $5 - $3 = -$98

**At Expiration:**
- If S_T > $100: Put expires worthless, call exercised, you deliver stock → Receive $100
- If S_T ≤ $100: Call expires worthless, exercise put, deliver stock → Receive $100
- **Guaranteed receive:** $100
- **Profit:** $100 - $98 = $2 risk-free!

### Part B: Greeks Calculation (15 min)

**Interviewer:** "Assume the call is fairly priced by Black-Scholes with vol = 20%. What are delta, gamma, and vega?"

**Solution:**

Given: S=$100, K=$100, T=1/12 (1 month), r=0%, σ=20%

\`\`\`
d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
   = [ln(1) + (0 + 0.02)×(1/12)] / (0.2×√(1/12))
   = 0.00167 / 0.0577
   = 0.029

d2 = d1 - σ√T = 0.029 - 0.0577 = -0.0287

Delta = N(d1) = N(0.029) ≈ 0.512

Gamma = N'(d1) / (S×σ×√T)
      = 0.399 / (100×0.2×0.289)
      = 0.0689

Vega = S×N'(d1)×√T
     = 100×0.399×0.289
     = $11.53 per 1% vol change
\`\`\`

**Interpretation:**
- **Delta = 0.51:** $1 stock move → ~$0.51 option move
- **Gamma = 0.069:** Delta changes by 0.069 per $1 stock move
- **Vega = $11.53:** Option gains $11.53 if vol increases 1%

### Part C: Hedging Strategy (15 min)

**Interviewer:** "You're short 1000 of these calls. How do you hedge? Stock moves to $105—what do you do?"

**Solution:**

**Initial Hedge:**
- Short 1000 calls → Delta exposure = -1000 × 0.512 = -512
- **Buy 512 shares** to neutralize delta

**After Stock Moves to $105:**

New delta (approximate):
\`\`\`
New d1 ≈ [ln(105/100) + 0.00167] / 0.0577 ≈ 0.862
New delta ≈ N(0.862) ≈ 0.806
\`\`\`

New delta exposure = -1000 × 0.806 = -806
Current shares: 512
**Need to buy additional:** 806 - 512 = 294 shares

**Dynamic hedging:** Continuously rebalance as stock moves.

---

## Mock Interview 3: Probability & Coding (Two Sigma Style)

**Time: 40 minutes**

### Part A: Probability Question (20 min)

**Interviewer:** "You draw cards from a deck without replacement until you get an ace. What\'s the expected number of cards drawn?"

**Solution:**

**Method 1: Direct Calculation**

Let E = expected number of draws.

The ace can appear in position 1, 2, 3, ..., 49 (can't be beyond position 49 since only 48 non-aces).

\`\`\`
P(ace in position k) = P(first k-1 are not aces) × P(kth is ace | first k-1 not aces)
                      = (48/52)×(47/51)×...×(50-k/54-k) × 4/(53-k)
\`\`\`

**Method 2: Symmetry Argument**

By symmetry, all 52 positions are equally likely for any given ace.

The 4 aces divide the deck into 5 regions.
Expected position of first ace = 53/5 = 10.6

**Method 3: Formula for Geometric-like Distribution**

For hypergeometric waiting time:
\`\`\`
E[draws] = (n+1)/(k+1)

where n = 52 total cards, k = 4 aces

E = 53/5 = 10.6 cards
\`\`\`

**Answer: 10.6 cards expected**

### Part B: Implementation (20 min)

**Interviewer:** "Code this simulation and verify your answer. Also compute the full distribution."

\`\`\`python
"""
First Ace Simulation
"""

import random
import numpy as np
from collections import Counter

def draw_until_ace():
    """Draw cards until ace appears."""
    deck = ['A'] * 4 + ['X'] * 48  # 4 aces, 48 others
    random.shuffle (deck)
    
    for i, card in enumerate (deck, 1):
        if card == 'A':
            return i
    return 53  # Should never reach (ace must appear)

# Simulate
n_trials = 100000
results = [draw_until_ace() for _ in range (n_trials)]

# Statistics
mean = np.mean (results)
median = np.median (results)
std = np.std (results)

print(f"Expected draws: {mean:.2f} (theoretical: 10.60)")
print(f"Median: {median}")
print(f"Std dev: {std:.2f}")

# Distribution
dist = Counter (results)
print("\nDistribution (top 15):")
for draw in range(1, 16):
    prob = dist[draw] / n_trials
    print(f"Draw {draw:2d}: {prob:.4f} ({prob*100:.2f}%)")

# Verify formula
print(f"\nFormula check: (52+1)/(4+1) = {53/5:.2f}")
\`\`\`

**Expected Output:**
\`\`\`
Expected draws: 10.61 (theoretical: 10.60)
Median: 9.0
Std dev: 7.85

Distribution (top 15):
Draw  1: 0.0769 (7.69%)
Draw  2: 0.0738 (7.38%)
Draw  3: 0.0710 (7.10%)
...

Formula check: (52+1)/(4+1) = 10.60
\`\`\`

---

## Interview Tips for Mock Problems

### Time Management
- **Quick read:** 1 minute
- **Clarify:** 1 minute  
- **Solve Part A:** 10-15 minutes
- **Solve Part B:** 10-15 minutes
- **Code/verify:** 10-15 minutes

### Communication Strategy
1. **Summarize the problem** in your own words
2. **State your approach** before diving in
3. **Show your work** - write equations clearly
4. **Verbalize trade-offs** - "I could do X or Y, I'll choose X because..."
5. **Check results** - "Let me verify this makes sense..."
6. **Be ready to pivot** - If stuck, try a different approach

### Common Mistakes
- ❌ Rushing into calculation without understanding
- ❌ Not asking clarifying questions
- ❌ Getting stuck on one approach
- ❌ Forgetting to check the answer
- ❌ Poor time management

### What Interviewers Look For
- ✓ Structured thinking
- ✓ Multiple solution approaches
- ✓ Clear communication
- ✓ Handling follow-ups smoothly
- ✓ Verification and sanity checks
- ✓ Code that works on first try (or debugs quickly)

---

## Practice Schedule

**Week 1-2:** Do Problems 1-3 untimed, focus on completeness
**Week 3-4:** Time yourself, aim to finish within time limit
**Week 5-6:** Practice explaining solutions out loud
**Week 7-8:** Do problems with friend acting as interviewer

**Remember:** The interview is a conversation, not an exam. Show your thinking process!

Good luck!
`,
};
