/**
 * Detailed Algorithm Walkthrough Section
 */

export const algorithmSection = {
  id: 'algorithm',
  title: 'Detailed Algorithm Walkthrough',
  content: `**Example: Two Sum in Sorted Array**

**Problem:** Given sorted array and target sum, find two numbers that add up to target.

**Brute Force Approach: O(n²)**
\`\`\`python
for i in range(len(nums)):
    for j in range(i + 1, len(nums)):
        if nums[i] + nums[j] == target:
            return [i, j]
# Checks every pair - slow!
\`\`\`

**Two Pointers Approach: O(n)**

**Step-by-Step with Example:**
Array: [1, 3, 5, 7, 9, 11], Target: 14

\`\`\`
Step 1: Initialize
[1, 3, 5, 7, 9, 11]
 L              R
Sum = 1 + 11 = 12 < 14 → Need larger sum → Move L right

Step 2:
[1, 3, 5, 7, 9, 11]
    L           R
Sum = 3 + 11 = 14 == 14 → FOUND!
\`\`\`

**Why This Works:**
1. Array is sorted, so:
   - Moving left pointer RIGHT increases the sum
   - Moving right pointer LEFT decreases the sum
2. We systematically explore valid possibilities
3. We never need to backtrack
4. Each element checked at most once

**Example: Remove Duplicates In-Place**

**Problem:** Remove duplicates from sorted array, return new length.

**Visual Walkthrough:**
\`\`\`
Original: [1, 1, 2, 2, 3, 3, 4]

Step 1:
[1, 1, 2, 2, 3, 3, 4]
 S  F
nums[S] == nums[F], F moves

Step 2:
[1, 1, 2, 2, 3, 3, 4]
 S     F
nums[S] != nums[F], S++, copy nums[F] to nums[S]

Step 3:
[1, 2, 2, 2, 3, 3, 4]
    S     F
nums[S] == nums[F], F moves

Step 4:
[1, 2, 2, 2, 3, 3, 4]
    S        F
nums[S] != nums[F], S++, copy nums[F] to nums[S]

Result:
[1, 2, 3, ...]
        S
Length = S + 1 = 3
\`\`\`

**Key Insight:**
Slow pointer marks the end of unique elements. Fast pointer explores ahead.`,
};
