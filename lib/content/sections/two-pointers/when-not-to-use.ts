/**
 * When NOT to Use Two Pointers Section
 */

export const whennottouseSection = {
  id: 'when-not-to-use',
  title: 'When NOT to Use Two Pointers',
  content: `## 🚫 When Two Pointers Won't Work

Two pointers is powerful, but it's not a universal solution. Here's when to **avoid** it and what to use instead:

---

## ❌ Pattern 1: When You Need All Pairs/Combinations

**Problem:** Two pointers finds ONE solution efficiently, not ALL solutions (unless specifically designed for it).

\`\`\`python
# ❌ BAD: Using two pointers to find all pairs summing to target
def findAllPairs(nums, target):
    nums.sort()
    left, right = 0, len(nums) - 1
    result = []
    
    while left < right:
        current_sum = nums[left] + nums[right]
        if current_sum == target:
            result.append([nums[left], nums[right]])
            left += 1  # ❌ WRONG! Might miss pairs with duplicates
            right -= 1
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    
    return result

# Example: nums = [1, 1, 2, 2, 3], target = 4
# After finding (1,3), moves both pointers
# Misses (2,2) pair!

# ✅ BETTER: Use hash map for all pairs
def findAllPairs(nums, target):
    seen = {}
    result = []
    
    for num in nums:
        complement = target - num
        if complement in seen:
            # Add all combinations with this complement
            for _ in range(seen[complement]):
                result.append([complement, num])
        seen[num] = seen.get(num, 0) + 1
    
    return result

# Alternative: Nested loops if you need ALL combinations (not just pairs)
\`\`\`

**When this happens:**
- "Find all pairs/triplets/combinations..."
- Need to track multiple valid solutions with duplicates
- Counting problems where order matters

**What to use instead:** Hash maps, nested loops (if small), or backtracking.

---

## ❌ Pattern 2: When Array is Unsorted and Sorting Would Lose Information

**Problem:** Two pointers often requires sorting, but sorting destroys original order or indices.

\`\`\`python
# ❌ BAD: Sorting when you need original indices
def twoSum(nums, target):
    # Two Sum asks for INDICES of the two numbers
    nums.sort()  # ❌ WRONG! Lost original indices
    left, right = 0, len(nums) - 1
    
    while left < right:
        current_sum = nums[left] + nums[right]
        if current_sum == target:
            return [left, right]  # ❌ These are sorted indices, not original!
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    
    return []

# Example: nums = [3, 2, 4], target = 6
# Need to return [1, 2] (indices of 2 and 4)
# After sorting: nums = [2, 3, 4], returns [0, 2]
# WRONG! These aren't the original indices.

# ✅ GOOD: Use hash map to preserve indices
def twoSum(nums, target):
    seen = {}  # value -> original index
    
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]  # ✅ Original indices!
        seen[num] = i
    
    return []
\`\`\`

**When this happens:**
- Problem asks for original indices
- Relative order matters in output
- Need to track element positions

**What to use instead:** Hash map to maintain index information.

---

## ❌ Pattern 3: When You Need O(1) Lookup of Specific Values

**Problem:** Two pointers traverses sequentially; can't jump to specific values efficiently.

\`\`\`python
# ❌ BAD: Two pointers when you need fast lookup
def hasTripleSumDivisibleByThree(nums):
    # Check if any three numbers sum to value divisible by 3
    nums.sort()
    
    for i in range(len(nums)):
        left, right = i + 1, len(nums) - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total % 3 == 0:
                return True
            # ❌ Can't efficiently skip to next candidate
            # Must check every combination
            left += 1
    
    return False

# O(n²) but checks many unnecessary combinations

# ✅ BETTER: Hash set for O(1) lookup
def hasTripleSumDivisibleByThree(nums):
    # Use modulo properties
    remainders = [0, 0, 0]
    for num in nums:
        remainders[num % 3] += 1
    
    # Check combinations: (0,0,0), (1,1,1), (2,2,2), (0,1,2)
    if remainders[0] >= 3:
        return True
    if remainders[1] >= 3:
        return True
    if remainders[2] >= 3:
        return True
    if remainders[0] >= 1 and remainders[1] >= 1 and remainders[2] >= 1:
        return True
    
    return False

# O(n) with mathematical insight
\`\`\`

**When this happens:**
- Need to check membership or existence quickly
- Mathematical properties allow smarter grouping
- Checking specific value combinations

**What to use instead:** Hash set/map, mathematical properties, or counting.

---

## ❌ Pattern 4: When Pointers Must Move Independently

**Problem:** Two pointers assumes coordinated movement; doesn't work when pointers need independent decisions.

\`\`\`python
# ❌ BAD: Two pointers for independent pointer movement
def longestMountain(arr):
    # Find longest mountain (increasing then decreasing)
    left, right = 0, 0
    max_length = 0
    
    while right < len(arr):
        # ❌ Can't decide: should left move? should right move?
        # Both need to move independently based on mountain structure
        if arr[right] > arr[right - 1]:
            right += 1
        elif arr[right] < arr[right - 1]:
            # ??? Should left move to start of mountain?
            # Or should right continue down?
            pass
    
    return max_length

# Two pointer logic breaks down - movements aren't coordinated!

# ✅ GOOD: Use iteration with state tracking
def longestMountain(arr):
    n = len(arr)
    max_length = 0
    i = 0
    
    while i < n:
        # Find potential mountain start
        start = i
        
        # Climb up
        while i + 1 < n and arr[i] < arr[i + 1]:
            i += 1
        
        peak = i
        
        # Climb down
        while i + 1 < n and arr[i] > arr[i + 1]:
            i += 1
        
        # Valid mountain?
        if start < peak < i:
            max_length = max(max_length, i - start + 1)
        
        # Move to next potential start
        if i == peak:
            i += 1
    
    return max_length
\`\`\`

**When this happens:**
- Pointers need different strategies
- Complex state transitions (up, down, flat)
- Multi-phase processing

**What to use instead:** Single pointer with state machine, or separate passes.

---

## ❌ Pattern 5: When You Need Subarray Properties Beyond Window

**Problem:** Sliding window (two-pointer variant) only tracks current window, not relationships between windows.

\`\`\`python
# ❌ BAD: Sliding window for maximum sum of non-overlapping subarrays
def maxSumTwoNoOverlap(nums, firstLen, secondLen):
    # Find max sum of two non-overlapping subarrays
    left, right = 0, 0
    window_sum = 0
    max_sum = 0
    
    # ❌ WRONG! Need to track TWO windows simultaneously
    # Can't track both first and second subarray with one window
    while right < len(nums):
        window_sum += nums[right]
        
        if right - left + 1 == firstLen:
            # Found first window, but where's second?
            # Can't track both with two pointers!
            pass
        
        right += 1
    
    return max_sum

# ✅ GOOD: Use prefix sums or DP
def maxSumTwoNoOverlap(nums, firstLen, secondLen):
    def maxSum(L, M):
        # Max sum when L-length subarray comes before M-length
        result = 0
        sum_L = sum(nums[:L])
        max_L = sum_L
        sum_M = sum(nums[L:L + M])
        result = max_L + sum_M
        
        for i in range(L + M, len(nums)):
            sum_M += nums[i] - nums[i - M]
            sum_L += nums[i - M] - nums[i - M - L]
            max_L = max(max_L, sum_L)
            result = max(result, max_L + sum_M)
        
        return result
    
    return max(maxSum(firstLen, secondLen), maxSum(secondLen, firstLen))
\`\`\`

**When this happens:**
- Need multiple windows or subarrays
- Windows have dependencies or ordering
- Need to compare different window positions

**What to use instead:** DP, prefix sums, or separate passes.

---

## ❌ Pattern 6: When Problem Requires Backtracking

**Problem:** Two pointers moves forward; can't backtrack to explore alternate paths.

\`\`\`python
# ❌ BAD: Two pointers for permutations
def permute(nums):
    # Generate all permutations
    left, right = 0, len(nums) - 1
    result = []
    
    # ❌ Can't generate permutations with two pointers
    # Need to explore different orderings, which requires backtracking
    while left <= right:
        # ??? How to generate different permutations?
        left += 1
    
    return result

# ✅ GOOD: Use backtracking
def permute(nums):
    result = []
    
    def backtrack(path, remaining):
        if not remaining:
            result.append(path[:])
            return
        
        for i in range(len(remaining)):
            path.append(remaining[i])
            backtrack(path, remaining[:i] + remaining[i+1:])
            path.pop()  # Backtrack!
    
    backtrack([], nums)
    return result
\`\`\`

**When this happens:**
- Generate all permutations/combinations
- Explore decision trees
- Need to undo choices and try alternatives

**What to use instead:** Backtracking, recursion with state exploration.

---

## ❌ Pattern 7: When You Need Full 2D Matrix Processing

**Problem:** Two pointers is 1D; doesn't naturally extend to 2D problems.

\`\`\`python
# ❌ BAD: Two pointers for 2D matrix search
def searchMatrix(matrix, target):
    # Search in row-wise and column-wise sorted matrix
    left, right = 0, len(matrix[0]) - 1
    
    # ❌ Can't navigate 2D space with just two 1D pointers
    while left < len(matrix) and right >= 0:
        # This works for SOME 2D problems but not general case
        if matrix[left][right] == target:
            return True
        elif matrix[left][right] > target:
            right -= 1
        else:
            left += 1
    
    return False

# This specific case works, but doesn't generalize to:
# - Counting regions
# - Finding paths
# - Traversing in multiple directions

# ✅ GOOD: Use appropriate 2D technique
# For search: Binary search on flattened index
def searchMatrix(matrix, target):
    if not matrix:
        return False
    
    m, n = len(matrix), len(matrix[0])
    left, right = 0, m * n - 1
    
    while left <= right:
        mid = (left + right) // 2
        mid_value = matrix[mid // n][mid % n]  # Convert to 2D
        
        if mid_value == target:
            return True
        elif mid_value < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return False

# For regions: DFS/BFS
# For paths: Dynamic programming
\`\`\`

**When this happens:**
- Working with 2D matrices (grids)
- Need to explore multiple directions
- Problems involving regions or connected components

**What to use instead:** Binary search (flattened), DFS/BFS, or DP.

---

## 🎯 Decision Framework: Should I Use Two Pointers?

### ✅ USE Two Pointers When:

1. **Array is sorted** (or you can sort it without losing info)
2. **Looking for pairs/triplets** with some property
3. **In-place** operation needed (no extra space)
4. **Linear scan** with strategic pointer movement
5. **Window-based** problem with continuous subarray
6. **Partition** operation (separate elements by criteria)

### ❌ DON'T USE Two Pointers When:

1. **Need original indices** → Use hash map
2. **Need O(1) lookup** → Use hash set
3. **Need all combinations** → Use backtracking or nested loops
4. **Complex state** → Use state machine or separate passes
5. **2D problem** → Use DFS/BFS/DP
6. **Multiple independent windows** → Use DP or prefix sums
7. **Backtracking required** → Use recursion

---

## 💡 Quick Recognition Guide

**Two Pointers is RIGHT when you see:**
- "Find pair with sum..."
- "Remove duplicates in-place..."
- "Partition array..."
- "Maximum length subarray with..."
- "Sorted array" + "O(1) space"

**Two Pointers is WRONG when you see:**
- "Return indices of..."
- "Count all pairs..."
- "Find all permutations..."
- "2D matrix"...
- "Need to explore all possibilities..."

---

## 📝 Alternative Techniques Cheat Sheet

| Problem Type | ❌ Not Two Pointers | ✅ Use This Instead |
|--------------|-------------------|-------------------|
| Find indices in unsorted array | Two pointers loses indices | Hash map O(n), O(n) |
| Count all pairs with property | Two pointers finds one pair | Nested loops or hash map |
| Generate permutations | Two pointers is linear | Backtracking O(n!) |
| 2D matrix traversal | Two pointers is 1D | DFS/BFS or DP |
| Multiple independent windows | Single window insufficient | DP or prefix sums |
| Need O(1) existence check | Sequential scan | Hash set O(1) lookup |
| Non-overlapping subarrays | Can't track multiple windows | DP with state tracking |
| Complex state transitions | Two pointers too simple | State machine + iteration |

---

## 🎓 Interview Strategy

**When interviewer says "optimize this":**

1. **Check if two pointers applies:**
   - Is it sorted? ✅
   - Do I need original order? ❌
   - Is it a pair/window problem? ✅
   - Do I need all solutions? ❌

2. **If two pointers doesn't fit:**
   - "Two pointers would lose index information, so I'll use a hash map"
   - "This needs backtracking since we explore multiple paths"
   - "This is 2D, so BFS would be more appropriate"

3. **Show you know the limits:**
   - "Two pointers gets us O(n) time but requires sorting, which changes indices"
   - "If we needed all pairs, I'd use a different approach, but for first pair, two pointers works"

**This demonstrates algorithmic maturity - knowing when NOT to use a technique is as important as knowing when to use it!**`,
};
