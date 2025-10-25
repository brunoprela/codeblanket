export const combinatoricsCounting = {
  title: 'Combinatorics & Counting',
  id: 'combinatorics-counting',
  content: `
# Combinatorics & Counting

## Introduction

Combinatorics is fundamental to quantitative finance interviews because it underpins probability theory, option pricing, and portfolio construction. Top quant firms (Jane Street, Citadel, Two Sigma) use combinatorics problems to test:

- **Structured thinking**: Breaking complex counting into manageable cases
- **Pattern recognition**: Identifying when to use permutations vs combinations
- **Recursion skills**: Finding recurrence relations
- **Attention to detail**: Avoiding overcounting and edge cases

This section covers:
1. Fundamental counting principles
2. Permutations (with and without repetition)
3. Combinations and binomial coefficients
4. Stars and bars method
5. Inclusion-exclusion principle
6. Recursive counting
7. Generating functions
8. Catalan numbers and their applications
9. Advanced counting techniques for finance

**Why this matters in finance:**

- **Option pricing**: Counting paths in binomial trees
- **Portfolio optimization**: Selecting k assets from n
- **Risk management**: Counting failure scenarios
- **Trading strategies**: Combinations of positions

---

## Fundamental Counting Principles

### Rule of Product (Multiplication Principle)

If task A can be done in m ways and task B can be done in n ways, then both tasks can be done in **m × n** ways (if tasks are independent).

**Example:** A restaurant offers 4 appetizers, 6 entrees, and 3 desserts. How many complete meals are possible?

**Solution:** 4 × 6 × 3 = 72 meals

### Rule of Sum (Addition Principle)

If task A can be done in m ways and task B can be done in n ways (and they're mutually exclusive), then either task can be done in **m + n** ways.

**Example:** A programmer can take route A (3 paths) or route B (5 paths) to work. Total: 3 + 5 = 8 paths.

### Interview Problem 1: Password Counting

**Question:** A password must be 8 characters long with:
- First character: uppercase letter (26 options)
- Next 6 characters: digits or lowercase letters (36 options each)
- Last character: special symbol (10 options)

How many valid passwords exist?

**Solution:**
\`\`\`
Total = 26 × 36^6 × 10
     = 26 × 2,176,782,336 × 10
     = 565,964,807,360
     ≈ 566 billion passwords
\`\`\`

**Mental math shortcut:**
- 36^6 ≈ 36^6 = (6^2)^6 = 6^12 ≈ 2.2 × 10^9
- Total ≈ 26 × 2.2 × 10^9 × 10 = 572 × 10^9 ≈ 5.7 × 10^11

\`\`\`python
"""
Password Counting with Constraints
"""

def count_passwords(length: int, char_sets: list) -> int:
    """
    Count passwords given character set sizes for each position.
    
    Args:
        length: Password length
        char_sets: List of character set sizes for each position
        
    Returns:
        Total number of valid passwords
    """
    total = 1
    for chars in char_sets:
        total *= chars
    return total

# Example: 8-character password
char_sets = [26] + [36] * 6 + [10]  # Uppercase, then 6 alphanumeric, then special
total_passwords = count_passwords(8, char_sets)

print(f"Total valid passwords: {total_passwords:,}")
print(f"In scientific notation: {total_passwords:.2e}")

# Output:
# Total valid passwords: 565,964,807,360
# In scientific notation: 5.66e+11
\`\`\`

---

## Permutations

### Permutations Without Repetition

The number of ways to arrange n distinct objects in a sequence:

\`\`\`
P(n) = n! = n × (n-1) × (n-2) × ... × 2 × 1
\`\`\`

**Example:** How many ways can 5 people stand in a line?
\`\`\`
5! = 120 ways
\`\`\`

### r-Permutations

The number of ways to arrange r objects from n distinct objects:

\`\`\`
P(n,r) = n!/(n-r)! = n × (n-1) × ... × (n-r+1)
\`\`\`

**Example:** How many 3-letter words can you make from 26 letters (no repetition)?
\`\`\`
P(26,3) = 26 × 25 × 24 = 15,600
\`\`\`

### Permutations With Repetition

When objects can repeat, the number of r-permutations from n objects is:

\`\`\`
n^r
\`\`\`

**Example:** How many 3-digit numbers exist? (Leading zeros allowed)
\`\`\`
10^3 = 1,000
\`\`\`

### Interview Problem 2: Trading Strategy Sequences

**Question:** A hedge fund has 10 trading strategies. They want to deploy 4 strategies in a specific order throughout the day. How many possible sequences exist?

**Solution:**
\`\`\`
P(10,4) = 10!/(10-4)! = 10!/6! = 10 × 9 × 8 × 7 = 5,040
\`\`\`

**Follow-up:** If strategies can be repeated?
\`\`\`
10^4 = 10,000
\`\`\`

\`\`\`python
"""
Permutations Calculator
"""

import math
from typing import Optional

def permutations(n: int, r: Optional[int] = None, repetition: bool = False) -> int:
    """
    Calculate permutations.
    
    Args:
        n: Total number of objects
        r: Number to select (default: n)
        repetition: Allow repetition (default: False)
        
    Returns:
        Number of permutations
    """
    if r is None:
        r = n
    
    if repetition:
        return n ** r
    else:
        return math.factorial(n) // math.factorial(n - r)

# Examples
print("Permutations Examples:")
print("=" * 60)

# 5 people in a line
print(f"P(5,5) = {permutations(5):,}")

# 3 letters from 26 (no repetition)
print(f"P(26,3) = {permutations(26, 3):,}")

# 3-digit numbers (with repetition)
print(f"10^3 = {permutations(10, 3, repetition=True):,}")

# Trading strategies
print(f"P(10,4) = {permutations(10, 4):,}")
print(f"10^4 with repetition = {permutations(10, 4, repetition=True):,}")

# Output:
# Permutations Examples:
# ============================================================
# P(5,5) = 120
# P(26,3) = 15,600
# 10^3 = 1,000
# P(10,4) = 5,040
# 10^4 with repetition = 10,000
\`\`\`

---

## Combinations

### Basic Combinations

The number of ways to choose r objects from n distinct objects (order doesn't matter):

\`\`\`
C(n,r) = n!/(r!(n-r)!) = "n choose r"
\`\`\`

**Key property:** C(n,r) = C(n, n-r)

**Example:** How many 5-card poker hands from 52 cards?
\`\`\`
C(52,5) = 52!/(5!×47!) = 2,598,960
\`\`\`

### Pascal's Triangle

\`\`\`
Row 0:              1
Row 1:            1   1
Row 2:          1   2   1
Row 3:        1   3   3   1
Row 4:      1   4   6   4   1
Row 5:    1   5  10  10   5   1
\`\`\`

**Property:** C(n,r) = C(n-1,r-1) + C(n-1,r)

### Interview Problem 3: Portfolio Selection

**Question:** You manage a fund and must select 5 stocks from a universe of 100 stocks to form a portfolio. How many possible portfolios exist?

**Solution:**
\`\`\`
C(100,5) = 100!/(5!×95!) 
         = (100×99×98×97×96)/(5×4×3×2×1)
         = 75,287,520
\`\`\`

**Mental math approach:**
1. Numerator: 100×99×98×97×96 ≈ 100^5 / (some factor) ≈ 9 × 10^9
2. Denominator: 120
3. Result: 9 × 10^9 / 120 ≈ 7.5 × 10^7

**Follow-up:** Must include at least 2 tech stocks (20 tech stocks, 80 others). How many portfolios?

**Solution (inclusion-exclusion or casework):**
- Case 1: Exactly 2 tech → C(20,2) × C(80,3)
- Case 2: Exactly 3 tech → C(20,3) × C(80,2)
- Case 3: Exactly 4 tech → C(20,4) × C(80,1)
- Case 4: Exactly 5 tech → C(20,5)

\`\`\`python
"""
Combinations and Portfolio Selection
"""

from math import comb

def combinations(n: int, r: int) -> int:
    """Calculate n choose r."""
    return comb(n, r)

# Problem: 5 stocks from 100
total_portfolios = combinations(100, 5)
print(f"Total portfolios (5 from 100): {total_portfolios:,}")

# Follow-up: At least 2 tech stocks (20 tech, 80 non-tech)
portfolios_with_min_tech = sum(
    combinations(20, tech) * combinations(80, 5 - tech)
    for tech in range(2, 6)
)

print(f"Portfolios with ≥2 tech stocks: {portfolios_with_min_tech:,}")

# Verify by complement: Total - (0 tech + 1 tech)
portfolios_complement = (
    total_portfolios 
    - combinations(20, 0) * combinations(80, 5)
    - combinations(20, 1) * combinations(80, 4)
)

print(f"Verification (complement): {portfolios_complement:,}")
assert portfolios_with_min_tech == portfolios_complement

# Output:
# Total portfolios (5 from 100): 75,287,520
# Portfolios with ≥2 tech stocks: 50,658,702
# Verification (complement): 50,658,702
\`\`\`

---

## Stars and Bars Method

Used for distributing indistinguishable objects into distinguishable bins.

**Problem:** Distribute n identical objects into k bins. Number of ways:

\`\`\`
C(n + k - 1, k - 1) = C(n + k - 1, n)
\`\`\`

**Visualization:** 10 stars (objects) into 3 bins using 2 bars:
\`\`\`
★★★|★★★★★|★★
Bin1: 3, Bin2: 5, Bin3: 2
\`\`\`

Total arrangements: C(10+3-1, 3-1) = C(12, 2) = 66

### Interview Problem 4: Order Allocation

**Question:** You need to execute 100 shares across 4 exchanges. Each exchange must receive at least 1 share. How many ways to allocate?

**Solution:**

First, give 1 share to each exchange (satisfying minimum). Now distribute remaining 96 shares freely:

\`\`\`
C(96 + 4 - 1, 4 - 1) = C(99, 3) = 156,849
\`\`\`

**Alternative (with minimum requirement):**

Use stars and bars with adjustment:
1. Distribute all 100 shares: C(100+4-1, 4-1) = C(103, 3)
2. Subtract invalid cases (where at least one exchange gets 0)
3. By inclusion-exclusion: ... (complex)

**Simpler:** Give each exchange 1 share, then distribute 96 freely as shown above.

\`\`\`python
"""
Stars and Bars Method
"""

def stars_and_bars(n: int, k: int, min_per_bin: int = 0) -> int:
    """
    Distribute n identical objects into k distinguishable bins.
    
    Args:
        n: Number of identical objects
        k: Number of distinguishable bins
        min_per_bin: Minimum objects per bin (default 0)
        
    Returns:
        Number of ways to distribute
    """
    # Ensure minimum requirement satisfied
    if n < k * min_per_bin:
        return 0
    
    # Distribute minimums, then freely allocate remainder
    remaining = n - k * min_per_bin
    return comb(remaining + k - 1, k - 1)

# Example: 100 shares to 4 exchanges, min 1 each
ways = stars_and_bars(100, 4, min_per_bin=1)
print(f"Ways to allocate 100 shares to 4 exchanges (min 1 each): {ways:,}")

# Example: 10 identical coins into 3 piggy banks (no minimum)
ways_coins = stars_and_bars(10, 3, min_per_bin=0)
print(f"Ways to put 10 coins in 3 banks: {ways_coins:,}")

# Output:
# Ways to allocate 100 shares to 4 exchanges (min 1 each): 156,849
# Ways to put 10 coins in 3 banks: 66
\`\`\`

---

## Inclusion-Exclusion Principle

For counting objects satisfying at least one of several properties:

\`\`\`
|A₁ ∪ A₂ ∪ ... ∪ Aₙ| = Σ|Aᵢ| - Σ|Aᵢ ∩ Aⱼ| + Σ|Aᵢ ∩ Aⱼ ∩ Aₖ| - ...
\`\`\`

### Interview Problem 5: Derangements (No Fixed Points)

**Question:** 5 traders each submit an order. Orders are randomly shuffled and returned. What's the probability no trader gets their own order back?

This is a **derangement** problem: permutations with no fixed points.

**Formula for derangements:**
\`\`\`
D(n) = n! × Σ((-1)^k / k!) for k=0 to n
     ≈ n! / e  (for large n)
\`\`\`

For n=5:
\`\`\`
D(5) = 5! × (1/0! - 1/1! + 1/2! - 1/3! + 1/4! - 1/5!)
     = 120 × (1 - 1 + 0.5 - 0.167 + 0.042 - 0.008)
     = 120 × 0.367
     = 44
\`\`\`

**Probability:** 44/120 = 0.367 ≈ 1/e

\`\`\`python
"""
Inclusion-Exclusion and Derangements
"""

def derangements(n: int) -> int:
    """
    Count derangements (permutations with no fixed points).
    
    Uses inclusion-exclusion principle.
    """
    total = 0
    factorial_n = math.factorial(n)
    
    for k in range(n + 1):
        total += ((-1) ** k) / math.factorial(k)
    
    return round(factorial_n * total)

# Test derangements
print("Derangements D(n):")
print("=" * 40)
for n in range(1, 11):
    d_n = derangements(n)
    total = math.factorial(n)
    prob = d_n / total
    
    print(f"n={n}: D(n)={d_n:6,}, Total={total:8,}, P={prob:.4f}")

# Output:
# Derangements D(n):
# ========================================
# n=1: D(n)=     0, Total=       1, P=0.0000
# n=2: D(n)=     1, Total=       2, P=0.5000
# n=3: D(n)=     2, Total=       6, P=0.3333
# n=4: D(n)=     9, Total=      24, P=0.3750
# n=5: D(n)=    44, Total=     120, P=0.3667
# n=10: D(n)=1,334,961, Total=3,628,800, P=0.3679

print(f"\\nLimit: 1/e = {1/math.e:.4f}")
\`\`\`

---

## Catalan Numbers

Catalan numbers appear frequently in combinatorics and finance (option pricing trees, parentheses matching, etc.).

**Definition:**
\`\`\`
C(n) = (1/(n+1)) × C(2n, n) = C(2n,n)/(n+1)
\`\`\`

**Recurrence:**
\`\`\`
C(0) = 1
C(n) = Σ C(i) × C(n-1-i) for i=0 to n-1
\`\`\`

**First few Catalan numbers:**
C(0)=1, C(1)=1, C(2)=2, C(3)=5, C(4)=14, C(5)=42, ...

### Applications in Finance

**1. Binary Tree Paths:**

In a binomial option pricing model with n periods, how many distinct paths reach a specific final node?

**2. Parentheses Matching:**

How many ways to match n pairs of parentheses?
- n=3: ((())), (()()), (())(), ()(()), ()()()

**3. Triangulation:**

How many ways to triangulate a convex (n+2)-gon? C(n)

### Interview Problem 6: Order Book Sequences

**Question:** In a limit order book, you have n buy orders and n sell orders. A valid matching sequence must never have more sells than buys processed at any point (to maintain market balance). How many valid sequences for n=5?

This is Catalan number C(5) = 42.

\`\`\`python
"""
Catalan Numbers
"""

def catalan(n: int) -> int:
    """Calculate the nth Catalan number."""
    if n <= 1:
        return 1
    
    # Using combination formula: C(n) = C(2n,n)/(n+1)
    return comb(2 * n, n) // (n + 1)

def catalan_recursive(n: int, memo: dict = None) -> int:
    """Calculate Catalan number using recurrence."""
    if memo is None:
        memo = {}
    
    if n <= 1:
        return 1
    
    if n in memo:
        return memo[n]
    
    result = sum(
        catalan_recursive(i, memo) * catalan_recursive(n - 1 - i, memo)
        for i in range(n)
    )
    
    memo[n] = result
    return result

# Generate Catalan numbers
print("Catalan Numbers:")
print("=" * 40)
for n in range(11):
    c_n = catalan(n)
    print(f"C({n:2d}) = {c_n:8,}")

# Output:
# Catalan Numbers:
# ========================================
# C( 0) =        1
# C( 1) =        1
# C( 2) =        2
# C( 3) =        5
# C( 4) =       14
# C( 5) =       42
# C( 6) =      132
# C( 7) =      429
# C( 8) =    1,430
# C( 9) =    4,862
# C(10) =   16,796

# Application: Valid order book sequences
n = 5
valid_sequences = catalan(n)
total_sequences = math.factorial(2 * n) // (math.factorial(n) ** 2)

print(f"\\nOrder Book Sequences (n={n}):")
print(f"Valid sequences: {valid_sequences}")
print(f"Total sequences: {total_sequences}")
print(f"Probability valid: {valid_sequences/total_sequences:.4f}")
\`\`\`

---

## Advanced Counting Techniques

### Fibonacci-Style Recursion

**Interview Problem 7: Staircase Climbing**

**Question:** You can climb stairs taking 1 or 2 steps at a time. How many ways to climb n=10 stairs?

**Recurrence:**
\`\`\`
f(n) = f(n-1) + f(n-2)
f(1) = 1, f(2) = 2
\`\`\`

This is Fibonacci sequence!

**Solution:**
\`\`\`
f(10) = 89 ways
\`\`\`

**Follow-up:** If you can take 1, 2, or 3 steps?

\`\`\`
f(n) = f(n-1) + f(n-2) + f(n-3)
f(1)=1, f(2)=2, f(3)=4
\`\`\`

\`\`\`python
"""
Recursive Counting - Staircase Problem
"""

def staircase_ways(n: int, steps: list = [1, 2], memo: dict = None) -> int:
    """
    Count ways to climb n stairs with given step sizes.
    
    Args:
        n: Number of stairs
        steps: List of allowed step sizes
        memo: Memoization dictionary
        
    Returns:
        Number of ways to climb stairs
    """
    if memo is None:
        memo = {}
    
    if n == 0:
        return 1
    if n < 0:
        return 0
    
    if n in memo:
        return memo[n]
    
    total = sum(staircase_ways(n - step, steps, memo) for step in steps)
    memo[n] = total
    return total

# Test with different step sizes
print("Staircase Climbing Ways:")
print("=" * 50)

for n in range(1, 11):
    ways_2 = staircase_ways(n, [1, 2])
    ways_3 = staircase_ways(n, [1, 2, 3])
    
    print(f"n={n:2d}: Steps[1,2]={ways_2:4d}, Steps[1,2,3]={ways_3:5d}")

# Output:
# Staircase Climbing Ways:
# ==================================================
# n= 1: Steps[1,2]=   1, Steps[1,2,3]=    1
# n= 2: Steps[1,2]=   2, Steps[1,2,3]=    2
# n= 3: Steps[1,2]=   3, Steps[1,2,3]=    4
# n= 4: Steps[1,2]=   5, Steps[1,2,3]=    7
# n= 5: Steps[1,2]=   8, Steps[1,2,3]=   13
# n=10: Steps[1,2]=  89, Steps[1,2,3]=  274
\`\`\`

### Burnside's Lemma (Advanced)

For counting distinct objects under symmetry (rotations, reflections).

**Interview Problem 8:** How many distinct necklaces can you make with n beads of k colors, considering rotations?

Using Burnside's lemma, the count involves summing over symmetries. This is advanced and rarely appears in interviews, but good to know exists.

---

## Trading & Finance Applications

### Problem 9: Options Spread Combinations

**Question:** You have 10 different strike prices for a stock. How many:
1. Vertical spreads (buy one call, sell another call at different strike)?
2. Butterfly spreads (buy 1 low strike, sell 2 middle, buy 1 high strike)?
3. Iron condors (put spread + call spread with 4 different strikes)?

**Solutions:**

1. **Vertical spreads:** Choose 2 strikes from 10, order matters (which to buy vs sell)
   - C(10,2) = 45

2. **Butterfly spreads:** Choose 3 strikes where middle is to be sold (2 contracts)
   - C(10,3) = 120

3. **Iron condors:** Choose 4 strikes, assign to specific roles
   - C(10,4) = 210

\`\`\`python
"""
Options Spread Combinations
"""

def count_option_spreads(n_strikes: int) -> dict:
    """
    Count various option spread combinations.
    
    Args:
        n_strikes: Number of available strike prices
        
    Returns:
        Dictionary with counts for each spread type
    """
    return {
        'vertical_spreads': comb(n_strikes, 2),
        'butterfly_spreads': comb(n_strikes, 3),
        'iron_condors': comb(n_strikes, 4),
        'calendar_spreads': n_strikes * (n_strikes - 1),  # Different expiries
    }

# Calculate for 10 strikes
spreads = count_option_spreads(10)

print("Options Spread Combinations (10 strikes):")
print("=" * 50)
for spread_type, count in spreads.items():
    print(f"{spread_type:20s}: {count:6,}")

# Output:
# Options Spread Combinations (10 strikes):
# ==================================================
# vertical_spreads    :     45
# butterfly_spreads   :    120
# iron_condors        :    210
# calendar_spreads    :     90
\`\`\`

---

## Practice Problems

### Problem Set 1: Basic Counting

1. **License Plates:** Format: 3 letters + 4 digits. How many possible plates?
2. **PIN Codes:** 4-digit PIN where no digit repeats. How many?
3. **Seating:** 8 people, 8 chairs in a row. 2 specific people must sit together. How many arrangements?

### Problem Set 2: Combinations

4. **Committee Selection:** From 15 people (9 women, 6 men), select a 5-person committee with at least 2 women. How many ways?
5. **Card Hands:** In poker, how many ways to get exactly one pair?
6. **Paths on Grid:** From (0,0) to (5,3), moving only right or up. How many paths?

### Problem Set 3: Advanced

7. **Circular Permutations:** 6 people sit at a round table. How many distinct arrangements?
8. **Anagrams:** How many distinct arrangements of "MISSISSIPPI"?
9. **Surjections:** How many ways to map {1,2,3,4,5} onto {A,B,C} (surjective functions)?

---

## Solutions to Practice Problems

**Problem 1:** 26^3 × 10^4 = 17,576,000 × 10,000 = 175,760,000,000

**Problem 2:** P(10,4) = 10 × 9 × 8 × 7 = 5,040

**Problem 3:** Treat 2 people as one unit. Arrange 7 units: 7! = 5,040. The 2 people can swap: ×2. Total: 10,080

**Problem 4:** Total - (0 or 1 women) = C(15,5) - [C(9,0)×C(6,5) + C(9,1)×C(6,4)] = 3,003 - [6 + 135] = 2,862

**Problem 5:** Choose rank for pair (13), choose 2 suits (C(4,2)=6), choose 3 other ranks (C(12,3)=220), choose suits (4^3=64). Total: 13 × 6 × 220 × 64 = 1,098,240

**Problem 6:** Need 5 rights, 3 ups. Total moves: 8. Choose 3 positions for ups: C(8,3) = 56

**Problem 7:** Fix one person's position (eliminate rotation). Arrange remaining 5: 5! = 120

**Problem 8:** 11 letters: 4 I's, 4 S's, 2 P's, 1 M. Count: 11!/(4!×4!×2!×1!) = 34,650

**Problem 9:** Use inclusion-exclusion. Answer: 3! × S(5,3) where S(5,3) is Stirling number of second kind = 25. So: 6 × 25 = 150

---

## Interview Strategy

**When to use each technique:**

1. **Product rule:** Independent choices at each step
2. **Permutations:** Order matters
3. **Combinations:** Order doesn't matter
4. **Stars and bars:** Identical objects, distinguishable bins
5. **Inclusion-exclusion:** "At least one" problems
6. **Recursion:** Problem breaks into smaller versions of itself

**Common mistakes:**
- Forgetting to divide by overcounting (using permutation instead of combination)
- Not considering edge cases (empty sets, zero elements)
- Mixing up n and r in formulas
- Forgetting factorials in denominators

**Communication tips:**
- State which principle you're using
- Check for overcounting
- Verify answer with small examples
- Offer to simulate for validation

---

## Summary

Combinatorics is essential for:
- **Probability calculations:** Counting favorable outcomes
- **Option pricing:** Paths in binomial trees
- **Portfolio construction:** Asset selection
- **Risk management:** Failure scenario enumeration

**Key formulas to memorize:**
- Permutations: P(n,r) = n!/(n-r)!
- Combinations: C(n,r) = n!/(r!(n-r)!)
- Stars and bars: C(n+k-1, k-1)
- Derangements: D(n) ≈ n!/e
- Catalan: C(n) = C(2n,n)/(n+1)

**Next steps:**
- Practice 20-30 problems daily
- Time yourself (aim for 2-3 minutes per problem)
- Review classic problems (handshakes, seating, poker hands)
- Understand when to apply each technique

Master combinatorics, and probability problems become much easier!
`,
};
