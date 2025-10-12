/**
 * Binary Search module content - Professional & comprehensive guide
 */

import { Module } from '@/lib/types';

export const binarySearchModule: Module = {
  id: 'binary-search',
  title: 'Binary Search',
  description:
    'Master the art of efficiently searching in sorted arrays using the divide-and-conquer approach.',
  icon: '🔍',
  sections: [
    {
      id: 'introduction',
      title: 'What is Binary Search?',
      content: `Binary Search is one of the most fundamental and efficient algorithms in computer science. It's a **divide-and-conquer** algorithm that finds the position of a target value within a **sorted array** by repeatedly dividing the search interval in half.

**The Core Insight:**
When dealing with a sorted array, we can determine which half contains our target by comparing it with the middle element. This eliminates half of the remaining elements with each comparison.

**Why "Binary"?**
At each step, we make a binary (yes/no) decision: is our target in the left half or the right half? This binary decision tree is what gives the algorithm its name and its logarithmic efficiency.

**Real-World Analogy:**
Think of finding a word in a dictionary. You don't start from 'A' and flip through every page. You open the dictionary roughly in the middle, check if your word comes before or after that page, then repeat the process with the appropriate half. That's binary search!

**Key Prerequisites:**
- The array MUST be sorted (ascending or descending)
- You need random access to elements (arrays work great, linked lists don't)
- The comparison operation must be well-defined`,
    },
    {
      id: 'algorithm',
      title: 'The Algorithm Step-by-Step',
      content: `**Algorithm Overview:**

1. **Initialize Pointers:**
   - Set \`left = 0\` (start of array)
   - Set \`right = n - 1\` (end of array)

2. **While \`left <= right\`:**
   - Calculate middle: \`mid = left + (right - left) // 2\`
   - Compare \`array[mid]\` with target:
     - **If equal:** Return \`mid\` (found!)
     - **If array[mid] < target:** Search right half (\`left = mid + 1\`)
     - **If array[mid] > target:** Search left half (\`right = mid - 1\`)

3. **If loop ends:** Return -1 (not found)

**Visual Example:**
Searching for 7 in [1, 3, 5, 7, 9, 11, 13, 15, 17]

\`\`\`
Iteration 1:
[1, 3, 5, 7, 9, 11, 13, 15, 17]
 L           M              R
Compare: 9 > 7, search left half

Iteration 2:
[1, 3, 5, 7]
 L     M   R
Compare: 5 < 7, search right half

Iteration 3:
[7]
 L/M/R
Compare: 7 == 7, FOUND at index 3!
\`\`\`

**Why This Works:**
Each comparison eliminates half the search space. After k comparisons, we've eliminated 2^k elements. This is why it's so fast!`,
      codeExample: `def binary_search(nums: List[int], target: int) -> int:
    """
    Classic binary search implementation.
    
    Args:
        nums: Sorted array in ascending order
        target: Value to find
        
    Returns:
        Index of target if found, -1 otherwise
        
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    left, right = 0, len(nums) - 1
    
    while left <= right:
        # Avoid integer overflow: (left + right) // 2
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid  # Found!
        elif nums[mid] < target:
            left = mid + 1  # Target in right half
        else:
            right = mid - 1  # Target in left half
    
    return -1  # Not found`,
    },
    {
      id: 'complexity',
      title: 'Time & Space Complexity Analysis',
      content: `**Time Complexity: O(log n)**

**Why Logarithmic?**
- Start with n elements
- After 1 comparison: n/2 elements remain
- After 2 comparisons: n/4 elements remain
- After 3 comparisons: n/8 elements remain
- After k comparisons: n/2^k elements remain

When n/2^k = 1, we've found our answer: k = log₂(n)

**Concrete Examples:**
- **10 elements:** max 4 comparisons (2^4 = 16)
- **100 elements:** max 7 comparisons (2^7 = 128)
- **1,000 elements:** max 10 comparisons (2^10 = 1,024)
- **1,000,000 elements:** max 20 comparisons (2^20 = 1,048,576)
- **1,000,000,000 elements:** max 30 comparisons!

**Comparison with Linear Search:**

| Array Size | Linear Search | Binary Search | Speedup |
|------------|--------------|---------------|---------|
| 100        | 100          | 7             | 14x     |
| 10,000     | 10,000       | 14            | 714x    |
| 1,000,000  | 1,000,000    | 20            | 50,000x |

**Space Complexity: O(1)**
- Iterative version uses constant space (just a few variables)
- Recursive version uses O(log n) space for the call stack

**Best, Average, Worst Cases:**
- **Best Case:** O(1) - target is at the middle
- **Average Case:** O(log n) - typical scenario
- **Worst Case:** O(log n) - target at an end or not present

The consistency of performance is a major advantage!`,
    },
    {
      id: 'templates',
      title: 'Code Templates & Patterns',
      content: `**Template 1: Classic Binary Search**
Find exact match or return -1

**Template 2: Find First Occurrence**
When duplicates exist, find leftmost occurrence

\`\`\`python
def find_first(nums: List[int], target: int) -> int:
    left, right = 0, len(nums) - 1
    result = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            result = mid
            right = mid - 1  # Continue searching left
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return result
\`\`\`

**Template 3: Find Last Occurrence**
Find rightmost occurrence when duplicates exist

\`\`\`python
def find_last(nums: List[int], target: int) -> int:
    left, right = 0, len(nums) - 1
    result = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            result = mid
            left = mid + 1  # Continue searching right
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return result
\`\`\`

**Template 4: Find Insert Position**
Where to insert target to maintain sorted order

\`\`\`python
def search_insert(nums: List[int], target: int) -> int:
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return left  # Insertion position
\`\`\`

**When to Use Each Template:**
- Use Template 1 for simple existence checks
- Use Template 2 for finding ranges (start boundary)
- Use Template 3 for finding ranges (end boundary)
- Use Template 4 for insertion/floor/ceiling problems`,
    },
    {
      id: 'common-mistakes',
      title: 'Common Pitfalls & How to Avoid Them',
      content: `**1. Integer Overflow (Critical in Some Languages)**

❌ **Wrong:**
\`\`\`python
mid = (left + right) // 2  # Can overflow in Java/C++
\`\`\`

✅ **Correct:**
\`\`\`python
mid = left + (right - left) // 2  # Safe from overflow
\`\`\`

**Why:** In languages with fixed integer sizes, \`left + right\` can exceed max integer value.

**2. Incorrect Loop Condition**

❌ **Wrong:**
\`\`\`python
while left < right:  # Will miss single element case
\`\`\`

✅ **Correct:**
\`\`\`python
while left <= right:  # Handles all cases correctly
\`\`\`

**Why:** When \`left == right\`, we still need to check that element.

**3. Off-by-One Errors in Pointer Updates**

❌ **Wrong:**
\`\`\`python
left = mid  # Can cause infinite loop!
right = mid  # Can cause infinite loop!
\`\`\`

✅ **Correct:**
\`\`\`python
left = mid + 1  # Properly excludes mid
right = mid - 1  # Properly excludes mid
\`\`\`

**Why:** Using \`mid\` directly can create infinite loops when the search space reduces to 2 elements.

**4. Forgetting to Check if Array is Sorted**

Always verify the precondition! If the array isn't sorted, binary search will give incorrect results.

**5. Using Binary Search on Unsorted Data**

Binary search ONLY works on sorted data. For unsorted data:
- Sort first (O(n log n)), then search
- Or use linear search (O(n))
- Or use hash table (O(1) average lookup)

**6. Returning Wrong Value**

Make sure you return:
- The index (not the value) when found
- -1 or appropriate sentinel when not found
- The correct boundary for "find first/last" variants

**Debugging Tips:**
- Print \`left\`, \`mid\`, \`right\` in each iteration
- Verify the search space is shrinking
- Check boundary conditions: empty array, single element, target at ends
- Test with duplicates if applicable`,
    },
    {
      id: 'variations',
      title: 'Advanced Variations & Applications',
      content: `Binary search is incredibly versatile. Once you master the basics, you can apply it to many problems that don't look like traditional search!

**1. Search in Rotated Sorted Array**
**Problem:** Array was sorted, then rotated. Find target.
**Example:** [4,5,6,7,0,1,2], target = 0

**Key Insight:** At least one half is always sorted. Check which half is sorted, then decide where to search.

**2. Find Peak Element**
**Problem:** Find any local maximum in an unsorted array.
**Key Insight:** If mid < mid+1, peak must be on the right. Binary search on the gradient!

**3. Search in 2D Matrix**
**Problem:** Matrix sorted row-wise and column-wise.
**Key Insight:** Treat as 1D array: \`mid = mid // cols, mid % cols\`

**4. Find Minimum in Rotated Sorted Array**
**Problem:** Find the smallest element after rotation.
**Key Insight:** Minimum is at the rotation point. Compare with rightmost element.

**5. Square Root / nth Root**
**Problem:** Find floor(sqrt(x)) without using sqrt function.
**Key Insight:** Binary search on the answer space [0, x].

**6. First Bad Version**
**Problem:** Find first failing version in sequence.
**Key Insight:** Find first True in array of [False, False, ..., True, True].

**Problem-Solving Framework:**
1. **Identify if binary search applies:**
   - Is there a sorted order? (explicit or implicit)
   - Can you check a condition in O(1) or O(log n)?
   - Is the answer monotonic? (if x works, all smaller/larger x work too)

2. **Define search space:**
   - What are the minimum and maximum possible answers?
   - What type: indices, values, or abstract space?

3. **Write the condition:**
   - What makes \`mid\` a valid/invalid answer?
   - How do you decide to go left or right?

4. **Handle edge cases:**
   - Empty input
   - Single element
   - All same elements
   - Target at boundaries

**Interview Tip:**
Binary search problems in interviews often hide the "sorted" aspect. Look for:
- "Find first/last..."
- "Minimum/maximum..."
- "At least/at most..."
- Problems with monotonic properties`,
    },
    {
      id: 'problem-solving',
      title: 'Problem-Solving Strategy & Interview Tips',
      content: `**Step-by-Step Approach:**

**1. Clarify Requirements (30 seconds)**
- Ask: "Is the array sorted?" (Critical!)
- Ask: "Can there be duplicates?"
- Ask: "What should I return if not found?"
- Ask: "What's the expected size?" (helps choose algorithm)

**2. Explain the Approach (1-2 minutes)**
- State: "I'll use binary search because the array is sorted"
- Explain time complexity: "O(log n) instead of O(n) linear search"
- Mention edge cases you'll handle

**3. Code (5-7 minutes)**
- Start with the template
- Clearly label left, right, mid
- Add comments at decision points
- Don't rush! Accuracy > Speed

**4. Test Your Code (2-3 minutes)**
- **Test case 1:** Target found in middle
- **Test case 2:** Target at boundaries (first/last element)
- **Test case 3:** Target not in array
- **Test case 4:** Single element array
- **Test case 5:** Empty array

**5. Analyze Complexity (30 seconds)**
- Time: O(log n) - explain why
- Space: O(1) iterative, O(log n) recursive

**Common Interview Follow-ups:**
1. "What if there are duplicates?" → Find first/last occurrence
2. "What if it's rotated?" → Modified binary search
3. "Can you do it recursively?" → Show recursive version
4. "What about 2D array?" → Treat as 1D or use 2D binary search

**Optimization Tips:**
- For very large arrays, consider cache-friendly modifications
- For repeated searches, consider preprocessing
- For range queries, consider segment trees or other data structures

**Red Flags to Avoid:**
❌ Not checking if array is sorted
❌ Infinite loops from wrong pointer updates
❌ Forgetting edge cases
❌ Integer overflow in mid calculation
❌ Wrong return value (returning value instead of index)

**How to Practice:**
1. Master the basic template first
2. Solve "find first/last occurrence" problems
3. Try rotated array problems
4. Tackle abstract binary search problems
5. Time yourself - aim for 10-15 minutes per problem

**Remember:**
- Binary search is about **eliminating possibilities**
- The search space **must shrink** every iteration
- When in doubt, **trace through with a small example**
- **Practice** until the template becomes second nature`,
    },
  ],
  keyTakeaways: [
    'Binary search reduces O(n) search to O(log n) by eliminating half the search space each iteration',
    'Only works on sorted (or monotonic) data - this is a strict requirement',
    'Use "left + (right - left) // 2" to avoid integer overflow',
    'Three main templates: exact match, find first, find last - master all three',
    'Common mistakes: wrong loop condition (use <=), off-by-one errors (use mid±1)',
    'Can be applied to many problems beyond simple array search - look for monotonic properties',
    'Time complexity: O(log n), Space: O(1) iterative, O(log n) recursive',
    'Always test edge cases: empty array, single element, duplicates, boundaries',
  ],
  timeComplexity: 'O(log n)',
  spaceComplexity: 'O(1)',
  relatedProblems: [
    'binary-search-basic',
    'first-bad-version',
    'search-insert-position',
  ],
};
