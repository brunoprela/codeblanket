/**
 * Two-Sum Patterns Family Section
 */

export const twosumpatternsSection = {
  id: 'two-sum-patterns',
  title: 'Two-Sum Patterns Family',
  content: `**Two-Sum patterns** are the **#1 most asked** pattern family in technical interviews. Mastering these patterns is essential for interview success.

---

## Why Two-Sum Matters

**Interview Frequency:**
- Asked at: Amazon (very frequently), Facebook, Google, Microsoft, Apple
- Appears in ~15-20% of all array/hash table interviews
- Foundation for more complex problems (3Sum, 4Sum, subset sum)
- Tests fundamental hash table optimization thinking

**What Makes It Important:**
- Teaches hash table vs. brute force trade-off
- Demonstrates space-time complexity analysis
- Shows pattern recognition across variations
- Gateway to understanding pair/complement problems

---

## Pattern 1: Two Sum (Hash Table Approach)

**Problem:** Find two numbers in an array that add up to a target.

**Brute Force:** O(n²) - check all pairs  
**Optimized:** O(n) - use hash table to store complements

**Key Insight:** As you iterate, check if \`target - current\` was seen before.

**Implementation:**
\`\`\`python
def two_sum (nums: List[int], target: int) -> List[int]:
    """
    Find indices of two numbers that add up to target.
    
    Time: O(n) - single pass
    Space: O(n) - hash table
    
    Key: Store value → index mapping, look for complement
    """
    seen = {}  # value → index
    
    for i, num in enumerate (nums):
        complement = target - num
        
        if complement in seen:
            return [seen[complement], i]
        
        seen[num] = i
    
    return []  # No solution found
\`\`\`

**Why Hash Table:**
- **Before:** Check if complement exists → O(n) linear search
- **After:** Check if complement exists → O(1) hash lookup
- Trade O(n) space for O(n) time (down from O(n²))

**Example:**
\`\`\`
nums = [2, 7, 11, 15], target = 9

Iteration 1: num=2, complement=7
  - 7 not in seen
  - Add 2 → index 0
  - seen = {2: 0}

Iteration 2: num=7, complement=2
  - 2 IS in seen (index 0)
  - Return [0, 1]
\`\`\`

---

## Pattern 2: Two Sum II - Sorted Array

**Problem:** Same as Two Sum, but array is **sorted**.

**New Approach:** Two pointers (O(1) space!)

**Key Insight:** Sorting enables two-pointer technique without extra space.

**Implementation:**
\`\`\`python
def two_sum_sorted (nums: List[int], target: int) -> List[int]:
    """
    Find indices in sorted array that add up to target.
    
    Time: O(n) - two pointers converge
    Space: O(1) - no extra space needed
    
    Key: Use sorted property to move pointers intelligently
    """
    left, right = 0, len (nums) - 1
    
    while left < right:
        current_sum = nums[left] + nums[right]
        
        if current_sum == target:
            return [left + 1, right + 1]  # 1-indexed
        elif current_sum < target:
            left += 1  # Need larger sum
        else:
            right -= 1  # Need smaller sum
    
    return []
\`\`\`

**Why Two Pointers:**
- If sum too small → increase left (makes sum bigger)
- If sum too large → decrease right (makes sum smaller)
- Converge toward answer without checking all pairs

**Comparison:**
| Approach | Time | Space | When to Use |
|----------|------|-------|-------------|
| Hash Table | O(n) | O(n) | Unsorted array |
| Two Pointers | O(n) | O(1) | Sorted array |
| Sort + Two Pointers | O(n log n) | O(1) | Space constrained |

---

## Pattern 3: 3Sum (Extension to Three Numbers)

**Problem:** Find all **unique triplets** that sum to zero.

**Challenges:**
1. **Three elements** instead of two
2. Need **all unique triplets** (no duplicates)
3. Time limit: must be better than O(n³)

**Approach:** Sort + Two Pointers (O(n²))

**Key Insight:** Fix one element, then do Two Sum II on the rest.

**Implementation:**
\`\`\`python
def three_sum (nums: List[int]) -> List[List[int]]:
    """
    Find all unique triplets that sum to zero.
    
    Time: O(n²) - n iterations × n two-pointer search
    Space: O(1) - excluding output
    
    Strategy:
    1. Sort array: O(n log n)
    2. Fix first element
    3. Two-pointer search for remaining two
    4. Skip duplicates to ensure uniqueness
    """
    nums.sort()
    result = []
    n = len (nums)
    
    for i in range (n - 2):
        # Skip duplicate values for first element
        if i > 0 and nums[i] == nums[i-1]:
            continue
        
        # Two-pointer search for remaining two elements
        left, right = i + 1, n - 1
        target = -nums[i]
        
        while left < right:
            current_sum = nums[left] + nums[right]
            
            if current_sum == target:
                result.append([nums[i], nums[left], nums[right]])
                
                # Skip duplicates for left pointer
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                # Skip duplicates for right pointer
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                
                left += 1
                right -= 1
            elif current_sum < target:
                left += 1
            else:
                right -= 1
    
    return result
\`\`\`

**Example:**
\`\`\`
nums = [-1, 0, 1, 2, -1, -4]
After sort: [-4, -1, -1, 0, 1, 2]

i=0: nums[0]=-4, target=4
  Two-pointer search: no pairs sum to 4

i=1: nums[1]=-1, target=1
  left=2, right=5: nums[2]=-1, nums[5]=2, sum=1 ✓
  Result: [[-1, -1, 2]]
  left=3, right=4: nums[3]=0, nums[4]=1, sum=1 ✓
  Result: [[-1, -1, 2], [-1, 0, 1]]

i=2: Skip (nums[2] == nums[1])

Final: [[-1, -1, 2], [-1, 0, 1]]
\`\`\`

**Duplicate Handling:**
- Skip duplicate first elements: \`if i > 0 and nums[i] == nums[i - 1]: continue\`
- Skip duplicate left/right pointers after finding a match
- This ensures uniqueness without using a set

---

## Pattern 4: 4Sum (Further Extension)

**Problem:** Find all unique quadruplets that sum to a target.

**Approach:** Sort + Two nested loops + Two Pointers (O(n³))

**Key Insight:** Fix two elements, then do Two Sum II on the rest.

**Implementation Sketch:**
\`\`\`python
def four_sum (nums: List[int], target: int) -> List[List[int]]:
    """
    Find all unique quadruplets that sum to target.
    
    Time: O(n³) - n² pairs × n two-pointer search
    Space: O(1) - excluding output
    
    Strategy:
    1. Sort array
    2. Fix first two elements (nested loops)
    3. Two-pointer search for remaining two
    4. Skip duplicates
    """
    nums.sort()
    result = []
    n = len (nums)
    
    for i in range (n - 3):
        # Skip duplicates for first element
        if i > 0 and nums[i] == nums[i-1]:
            continue
        
        for j in range (i + 1, n - 2):
            # Skip duplicates for second element
            if j > i + 1 and nums[j] == nums[j-1]:
                continue
            
            # Two-pointer for remaining elements
            left, right = j + 1, n - 1
            remaining = target - nums[i] - nums[j]
            
            while left < right:
                current_sum = nums[left] + nums[right]
                
                if current_sum == remaining:
                    result.append([nums[i], nums[j], nums[left], nums[right]])
                    # Skip duplicates...
                    while left < right and nums[left] == nums[left+1]:
                        left += 1
                    while left < right and nums[right] == nums[right-1]:
                        right -= 1
                    left += 1
                    right -= 1
                elif current_sum < remaining:
                    left += 1
                else:
                    right -= 1
    
    return result
\`\`\`

---

## Pattern Recognition Guide

**When you see:**
| Keyword | Pattern | Approach |
|---------|---------|----------|
| "Find two numbers that sum to..." | Two Sum | Hash table O(n) |
| "Sorted array, two sum" | Two Sum II | Two pointers O(1) space |
| "Three numbers sum to zero/target" | 3Sum | Sort + fix one + two pointers |
| "Four numbers sum to target" | 4Sum | Sort + fix two + two pointers |
| "K numbers sum to target" | K-Sum | Recursively reduce to 2Sum |
| "Count pairs with sum" | Two Sum variant | Hash table counting |
| "Find all pairs" | Two Sum all | Hash table with list |

---

## Complexity Analysis

| Problem | Best Time | Space | Approach |
|---------|-----------|-------|----------|
| **Two Sum (unsorted)** | O(n) | O(n) | Hash table |
| **Two Sum (sorted)** | O(n) | O(1) | Two pointers |
| **3Sum** | O(n²) | O(1)* | Sort + fixed first + two pointers |
| **4Sum** | O(n³) | O(1)* | Sort + fixed two + two pointers |
| **K-Sum** | O(n^(k-1)) | O(1)* | Generalize to k elements |

*excluding output space

**Pattern Complexity Growth:**
- 2Sum: O(n) with hash table
- 3Sum: O(n²) with sort + two pointers
- 4Sum: O(n³) with sort + two pointers
- K-Sum: O(n^(k-1)) in general

---

## Interview Strategy

**Recognition:**
- "Sum to target" → Think Two-Sum family
- "Pairs" → Two Sum
- "Triplets" → 3Sum  
- "Unique" → Need to skip duplicates

**Approach Selection:**
1. **Unsorted, two elements** → Hash table
2. **Sorted, two elements** → Two pointers
3. **Three+ elements** → Sort + fix first (s) + two pointers
4. **Space constrained** → Sort first, use two pointers

**Communication:**
\`\`\`
"I recognize this as a Two-Sum variant.

For two elements in unsorted array:
- Brute force: O(n²) checking all pairs
- Optimized: O(n) with hash table for complements

For three elements:
- Sort first: O(n log n)
- Fix first element, two-pointer for rest: O(n²)
- Skip duplicates to ensure uniqueness

Time: O(n²), Space: O(1) excluding output"
\`\`\`

**Common Mistakes:**
- ❌ Forgetting to skip duplicates in 3Sum/4Sum
- ❌ Using same element twice (check indices)
- ❌ Not considering sorted vs unsorted
- ❌ Using hash table when two pointers is more efficient

**Practice Progression:**
1. Master Two Sum (hash table)
2. Practice Two Sum II (two pointers)
3. Tackle 3Sum (combining techniques)
4. Attempt 4Sum (harder but same pattern)
5. Try variations (closest sum, count pairs, etc.)`,
};
