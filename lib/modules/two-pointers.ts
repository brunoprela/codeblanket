/**
 * Two Pointers module content - Professional & comprehensive guide
 */

import { Module } from '@/lib/types';

export const twoPointersModule: Module = {
  id: 'two-pointers',
  title: 'Two Pointers',
  description:
    'Learn the two-pointer technique for efficiently solving array and string problems.',
  icon: '👉👈',
  sections: [
    {
      id: 'introduction',
      title: 'What is the Two Pointers Technique?',
      content: `The Two Pointers technique is a powerful algorithmic pattern that uses **two pointers** to traverse a data structure (typically an array or string) in a clever way to solve problems efficiently. Instead of using nested loops that lead to O(n²) time complexity, two pointers can often reduce this to O(n).

**The Core Concept:**
Use two references (pointers) to traverse your data structure. The pointers can:
- Start at opposite ends and move toward each other
- Both start at the beginning but move at different speeds
- Define a window that slides through the data

**Why Two Pointers?**
Many problems that seem to require checking all pairs (O(n²)) can be solved more efficiently by maintaining two positions and making smart decisions about which pointer to move.

**Real-World Analogy:**
Imagine two people searching through a sorted list of prices to find two items that sum to your budget. One starts from the cheapest items, the other from the most expensive. If the sum is too high, the person at expensive items moves down. If too low, the person at cheap items moves up. They meet in the middle with the answer - much faster than checking every possible pair!

**When Does Two Pointers Work?**
- When dealing with sorted or sortable data
- When you need to find pairs/triplets with certain properties
- When working with palindromes or symmetric patterns
- When you need to partition or reorder array elements
- When tracking a window of elements with specific constraints`,
    },
    {
      id: 'patterns',
      title: 'The Three Main Patterns',
      content: `Understanding these three patterns will help you recognize when to use two pointers:

**Pattern 1: Opposite Direction (Converging Pointers)**
**Setup:** Start at both ends, move toward center

**Visual:**
\`\`\`
[1, 2, 3, 4, 5, 6, 7, 8]
 L →           ← R
\`\`\`

**When to Use:**
- Working with sorted arrays
- Finding pairs that sum to a target
- Palindrome checking
- Reversing arrays/strings
- Container with most water

**Classic Problem:** Two Sum in Sorted Array
- If sum too small → move left pointer right (increase sum)
- If sum too large → move right pointer left (decrease sum)
- If sum equals target → found it!

**Pattern 2: Same Direction (Fast & Slow Pointers)**
**Setup:** Both start at beginning, move at different speeds

**Visual:**
\`\`\`
[1, 1, 2, 2, 3, 3, 4]
 S
 F →
\`\`\`

**When to Use:**
- Removing duplicates in-place
- Partitioning arrays
- Cycle detection in linked lists
- Finding middle element

**Classic Problem:** Remove Duplicates
- Slow pointer marks position for next unique element
- Fast pointer scans ahead to find different element
- Copy unique elements from fast to slow position

**Pattern 3: Sliding Window**
**Setup:** Two pointers define a window that slides

**Visual:**
\`\`\`
[a, b, c, d, e, f, g]
    L → R
       Window
\`\`\`

**When to Use:**
- Fixed or variable size subarray problems
- Substring problems with constraints
- Maximum/minimum window problems
- Running calculations over ranges

**Classic Problem:** Maximum Sum Subarray of Size K
- Expand window by moving right pointer
- Shrink window by moving left pointer
- Maintain window properties as you slide`,
    },
    {
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
      codeExample: `def two_sum(nums: List[int], target: int) -> List[int]:
    """
    Find two numbers that sum to target in sorted array.
    
    Args:
        nums: Sorted array in ascending order
        target: Target sum
        
    Returns:
        Indices of two numbers that sum to target
        
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    left, right = 0, len(nums) - 1
    
    while left < right:
        current_sum = nums[left] + nums[right]
        
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1  # Need larger sum
        else:
            right -= 1  # Need smaller sum
    
    return []  # No solution

def remove_duplicates(nums: List[int]) -> int:
    """
    Remove duplicates in-place, return new length.
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if not nums:
        return 0
    
    slow = 0  # Position for next unique element
    
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]
    
    return slow + 1  # Length of unique elements`,
    },
    {
      id: 'complexity',
      title: 'Time & Space Complexity Analysis',
      content: `**Time Complexity: O(n)**

**Why Linear Time?**
- Each pointer moves through the array at most once
- In opposite direction pattern: combined movement covers array once
- In same direction pattern: fast pointer covers array once
- No nested loops required!

**Comparison with Brute Force:**

| Approach | Time | Space | Array Size 1000 |
|----------|------|-------|-----------------|
| Brute Force (nested loops) | O(n²) | O(1) | 1,000,000 ops |
| Two Pointers | O(n) | O(1) | 1,000 ops |

**The Difference is Massive:**
- For n = 1,000: 1000x faster
- For n = 10,000: 10,000x faster
- For n = 100,000: 100,000x faster

**Space Complexity: O(1)**
- Only use two pointer variables
- Modify array in-place when needed
- No additional data structures
- Constant extra memory regardless of input size

**Detailed Analysis by Pattern:**

**1. Opposite Direction:**
- Time: O(n) - each element visited once
- Space: O(1) - two pointer variables
- Pointers move total of n times combined

**2. Same Direction:**
- Time: O(n) - fast pointer goes through array once
- Space: O(1) - two pointer variables
- Slow pointer always behind fast pointer

**3. Sliding Window:**
- Time: O(n) - each element enters and leaves window once
- Space: O(1) for basic window, O(k) if storing window elements
- Amortized O(1) per element

**When Complexity Increases:**
- If operation at each step is expensive: O(n × k)
- If storing window elements: Space becomes O(window_size)
- If sorting is needed first: Time becomes O(n log n)

**Practical Performance:**
Beyond Big-O notation:
- Cache-friendly: sequential access pattern
- Low overhead: minimal extra variables
- Parallelizable: in some cases
- In-place: doesn't require extra memory`,
    },
    {
      id: 'templates',
      title: 'Code Templates & Patterns',
      content: `**Template 1: Opposite Direction - Pair with Target Sum**

\`\`\`python
def pair_with_sum(nums: List[int], target: int) -> List[int]:
    """Find pair that sums to target in sorted array."""
    left, right = 0, len(nums) - 1
    
    while left < right:
        current = nums[left] + nums[right]
        
        if current == target:
            return [left, right]
        elif current < target:
            left += 1  # Need larger value
        else:
            right -= 1  # Need smaller value
    
    return [-1, -1]  # Not found
\`\`\`

**Template 2: Same Direction - Remove Elements**

\`\`\`python
def remove_element(nums: List[int], val: int) -> int:
    """Remove all instances of val in-place."""
    slow = 0  # Position for next kept element
    
    for fast in range(len(nums)):
        if nums[fast] != val:
            nums[slow] = nums[fast]
            slow += 1
    
    return slow  # New length
\`\`\`

**Template 3: Sliding Window - Fixed Size**

\`\`\`python
def max_sum_subarray(nums: List[int], k: int) -> int:
    """Find maximum sum of subarray of size k."""
    # Initialize first window
    window_sum = sum(nums[:k])
    max_sum = window_sum
    
    # Slide window
    for right in range(k, len(nums)):
        window_sum += nums[right]  # Add new element
        window_sum -= nums[right - k]  # Remove old element
        max_sum = max(max_sum, window_sum)
    
    return max_sum
\`\`\`

**Template 4: Sliding Window - Variable Size**

\`\`\`python
def longest_substring_k_distinct(s: str, k: int) -> int:
    """Longest substring with at most k distinct characters."""
    left = 0
    max_len = 0
    char_count = {}
    
    for right in range(len(s)):
        # Expand window
        char_count[s[right]] = char_count.get(s[right], 0) + 1
        
        # Shrink window if constraint violated
        while len(char_count) > k:
            char_count[s[left]] -= 1
            if char_count[s[left]] == 0:
                del char_count[s[left]]
            left += 1
        
        # Update max length
        max_len = max(max_len, right - left + 1)
    
    return max_len
\`\`\`

**Template 5: Partition Array**

\`\`\`python
def partition(nums: List[int], pivot: int) -> int:
    """Partition array around pivot value."""
    left = 0  # Next position for elements < pivot
    
    for right in range(len(nums)):
        if nums[right] < pivot:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
    
    return left  # Pivot position
\`\`\`

**Choosing the Right Template:**
1. Sorted array + pair/triplet? → Opposite direction
2. Remove/partition in-place? → Same direction
3. Subarray/substring problems? → Sliding window
4. Need to reorder elements? → Partition template`,
    },
    {
      id: 'advanced',
      title: 'Advanced Techniques & Variations',
      content: `**Three Pointers (3Sum Problem)**

Finding three numbers that sum to target:

\`\`\`python
def three_sum(nums: List[int], target: int) -> List[List[int]]:
    """Find all unique triplets that sum to target."""
    nums.sort()  # O(n log n)
    result = []
    
    for i in range(len(nums) - 2):
        # Skip duplicates for first number
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        
        # Two pointers for remaining two numbers
        left, right = i + 1, len(nums) - 1
        
        while left < right:
            current = nums[i] + nums[left] + nums[right]
            
            if current == target:
                result.append([nums[i], nums[left], nums[right]])
                
                # Skip duplicates
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                    
                left += 1
                right -= 1
            elif current < target:
                left += 1
            else:
                right -= 1
    
    return result
\`\`\`

**Cycle Detection (Floyd's Algorithm)**

Detect cycle in linked list using fast & slow pointers:

\`\`\`python
def has_cycle(head: ListNode) -> bool:
    """Detect if linked list has a cycle."""
    if not head or not head.next:
        return False
    
    slow = head
    fast = head.next
    
    while slow != fast:
        if not fast or not fast.next:
            return False
        slow = slow.next
        fast = fast.next.next
    
    return True
\`\`\`

**Why it works:** If there's a cycle, fast will eventually catch up to slow.

**Dutch National Flag (Three-Way Partitioning)**

Partition array into three parts:

\`\`\`python
def sort_colors(nums: List[int]) -> None:
    """Sort array with values 0, 1, 2 in-place."""
    left = 0  # Next position for 0
    curr = 0  # Current element being examined
    right = len(nums) - 1  # Next position for 2
    
    while curr <= right:
        if nums[curr] == 0:
            nums[left], nums[curr] = nums[curr], nums[left]
            left += 1
            curr += 1
        elif nums[curr] == 2:
            nums[curr], nums[right] = nums[right], nums[curr]
            right -= 1
            # Don't move curr - need to examine swapped element
        else:  # nums[curr] == 1
            curr += 1
\`\`\`

**Container With Most Water**

Find maximum area between two vertical lines:

\`\`\`python
def max_area(height: List[int]) -> int:
    """Find container that holds the most water."""
    left, right = 0, len(height) - 1
    max_area = 0
    
    while left < right:
        width = right - left
        # Height limited by shorter line
        h = min(height[left], height[right])
        max_area = max(max_area, width * h)
        
        # Move pointer at shorter line
        # (might find taller line)
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    
    return max_area
\`\`\`

**Key Insight:** Always move the pointer at the shorter line - moving the taller line can only decrease area.`,
    },
    {
      id: 'strategy',
      title: 'Problem-Solving Strategy & Interview Tips',
      content: `**Recognition Patterns:**

Ask yourself these questions to identify two pointer problems:

**1. "Do I need to find pairs/triplets?"**
- If yes and sorted/sortable → Opposite direction pointers

**2. "Do I need to modify array in-place?"**
- Removing elements → Same direction pointers
- Partitioning → Fast/slow pointers

**3. "Am I looking at subarrays/substrings?"**
- Fixed size window → Sliding window with fixed gap
- Variable size window → Expanding/shrinking window

**4. "Is there a two-pass O(n²) solution?"**
- Often can be optimized to O(n) with two pointers

**Step-by-Step Approach:**

**Step 1: Clarify (30 seconds)**
- "Is the input sorted?" (Crucial!)
- "Can I modify the input?"
- "What should I return?"
- "Are there duplicates?"

**Step 2: Choose Pattern (30 seconds)**
- Sorted + pairs? → Opposite direction
- Modify in-place? → Same direction  
- Window problem? → Sliding window

**Step 3: Explain Approach (1-2 minutes)**
- State which pattern you're using
- Explain why it's better than brute force
- Mention time/space complexity

**Step 4: Code (5-7 minutes)**
- Initialize pointers correctly
- Write clear loop condition
- Handle pointer movement logic
- Consider edge cases

**Step 5: Test (2-3 minutes)**
Test these cases:
- Empty/single element input
- All elements same
- Target at boundaries
- No valid answer
- Multiple valid answers

**Common Mistakes to Avoid:**

**1. Wrong Initialization**
❌ \`left, right = 0, len(nums)\` // right should be len(nums) - 1
✅ \`left, right = 0, len(nums) - 1\`

**2. Infinite Loops**
❌ Forgetting to move pointers
✅ Always ensure at least one pointer moves each iteration

**3. Off-by-One Errors**
❌ \`while left <= right\` when should be \`left < right\`
✅ Think carefully about when pointers can be equal

**4. Not Handling Duplicates**
- For 3Sum, need to skip duplicate values
- For unique pairs, need extra checks

**5. Wrong Pointer Movement**
- In container problem, move the shorter height
- In partition, be careful which pointer to move

**Interview Follow-Ups:**

**Q: "Can you do it without extra space?"**
A: Two pointers is already O(1) space - emphasize this advantage

**Q: "What if array is not sorted?"**
A: "I can sort it first in O(n log n), still better than O(n²)"

**Q: "Can you extend to 4Sum?"**
A: "Yes, fix two elements and use two pointers for other two"

**Q: "What about cycle detection in linked list?"**
A: "Use fast and slow pointers - Floyd's algorithm"

**Practice Roadmap:**

**Week 1: Master Basics**
1. Two Sum (sorted array)
2. Remove duplicates
3. Valid palindrome
4. Container with most water

**Week 2: Intermediate**
5. 3Sum
6. Trapping rain water
7. Longest substring without repeating chars
8. Minimum window substring

**Week 3: Advanced**
9. 4Sum
10. Linked list cycle detection
11. Dutch national flag
12. Merge sorted arrays

**Time Management:**
- Spend 2-3 minutes planning before coding
- Should complete easy problems in 10-15 minutes
- Medium problems in 15-25 minutes
- Don't rush - correct code beats fast wrong code!`,
    },
  ],
  keyTakeaways: [
    'Two pointers reduces O(n²) nested loops to O(n) by strategically moving pointers',
    'Three main patterns: opposite direction (converging), same direction (fast/slow), sliding window',
    'Opposite direction works great for sorted arrays and pair-finding problems',
    'Same direction pattern perfect for in-place modifications and partitioning',
    'Sliding window excels at subarray/substring problems with constraints',
    'Always O(1) space complexity - processes data in-place without extra structures',
    'Key skill: deciding which pointer to move based on current conditions',
    'Can extend to three or more pointers for problems like 3Sum and 4Sum',
    'Common mistakes: wrong initialization, infinite loops, off-by-one errors',
    'Recognition: look for pairs, in-place operations, or window-based problems',
  ],
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  relatedProblems: [
    'valid-palindrome',
    'container-with-most-water',
    'trapping-rain-water',
  ],
};
