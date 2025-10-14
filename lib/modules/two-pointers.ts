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
      quiz: [
        {
          id: 'q1',
          question:
            'Explain why the two pointers technique can reduce O(n²) time complexity to O(n). What makes this possible?',
          hint: 'Think about what nested loops do versus what two pointers accomplish.',
          sampleAnswer:
            'The two pointers technique avoids the need for nested loops by making smart decisions about which pointer to move based on the data structure properties. In a nested loop approach, you check every possible pair which is n × n operations. With two pointers, you traverse the array just once - each pointer moves through the array at most n times total, giving you O(n). The key is that we can eliminate checking certain pairs because we have information - like if the array is sorted, or if we have found duplicates. We make progress with every pointer movement instead of checking all combinations.',
          keyPoints: [
            'Nested loops check all pairs: O(n²)',
            'Two pointers traverse once: O(n)',
            'Each pointer moves at most n times total',
            'Smart decisions eliminate need to check all pairs',
          ],
        },
        {
          id: 'q2',
          question:
            'Describe the two-people-searching-prices analogy in your own words and explain how it relates to the two pointers algorithm.',
          sampleAnswer:
            'Imagine two people looking for items that add up to exactly your budget in a price list that goes from cheap to expensive. One person starts at the cheapest item, the other at the most expensive. If the total is too much, the person at the expensive end moves to a cheaper item. If too little, the person at the cheap end moves to a more expensive item. They work together, meeting in the middle when they find the right combination. This is exactly how two pointers work on a sorted array - start at ends, move toward center based on whether you need a bigger or smaller value. It is efficient because you never backtrack or check the same pair twice.',
          keyPoints: [
            'One starts cheap, one starts expensive',
            'Move based on whether sum is too high or low',
            'Meet in middle when found',
            'Never check same pair twice',
          ],
        },
        {
          id: 'q3',
          question:
            'When would you consider using two pointers over other techniques? Give me some key signals.',
          sampleAnswer:
            'I look for a few signals. First, if the array is sorted or can be sorted, that is a huge hint. Second, if the problem asks about pairs, triplets, or finding elements that satisfy some relationship. Third, palindrome problems are classic two pointer problems. Fourth, if I need to do something in-place like removing duplicates or partitioning. And finally, if I catch myself thinking "I need nested loops to check all combinations" - that is when I pause and ask if two pointers could do it smarter. The key question is: can I make progress by moving pointers based on comparisons rather than checking everything?',
          keyPoints: [
            'Sorted or sortable data',
            'Finding pairs/triplets with properties',
            'Palindrome or symmetry problems',
            'In-place operations needed',
            'Alternative to nested loops',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question:
            'What is the primary benefit of the two pointers technique?',
          options: [
            'It reduces space complexity to O(1)',
            'It reduces time complexity from O(n²) to O(n)',
            'It only works on sorted arrays',
            'It uses recursion',
          ],
          correctAnswer: 1,
          explanation:
            'The two pointers technique typically reduces time complexity from O(n²) (nested loops checking all pairs) to O(n) by making smart decisions about which pointer to move based on the data properties.',
        },
        {
          id: 'mc2',
          question:
            'Which type of data structure property makes two pointers most effective?',
          options: [
            'Unsorted arrays',
            'Sorted or sortable data',
            'Hash tables',
            'Binary trees',
          ],
          correctAnswer: 1,
          explanation:
            'Two pointers works best with sorted or sortable data because the sorted property allows us to make predictable decisions about which pointer to move to progress toward the solution.',
        },
        {
          id: 'mc3',
          question:
            'What are the three main movement patterns for two pointers?',
          options: [
            'Left, right, and center',
            'Fast, slow, and medium',
            'Opposite direction, same direction, and sliding window',
            'Forward, backward, and circular',
          ],
          correctAnswer: 2,
          explanation:
            'The three main patterns are: opposite direction (converging from both ends), same direction (fast & slow pointers), and sliding window (defining an expanding/shrinking window).',
        },
        {
          id: 'mc4',
          question:
            'When should you consider using two pointers instead of nested loops?',
          options: [
            'Only for very small arrays',
            'When you need to find pairs/triplets or work with sorted data',
            'When working with trees',
            'Only for string manipulation',
          ],
          correctAnswer: 1,
          explanation:
            'Two pointers is ideal for finding pairs/triplets with certain properties, working with sorted data, palindromes, in-place operations, or anytime you catch yourself thinking about nested loops.',
        },
        {
          id: 'mc5',
          question:
            'What is the space complexity of most two-pointer solutions?',
          options: ['O(n)', 'O(log n)', 'O(1)', 'O(n²)'],
          correctAnswer: 2,
          explanation:
            'Most two-pointer solutions use O(1) constant space since they only need a few pointer variables and often modify the array in-place without creating new data structures.',
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Walk me through the three main two-pointer patterns. For each one, describe when you would use it and give me an example problem.',
          hint: 'Think about opposite direction, same direction, and sliding window.',
          sampleAnswer:
            'The first pattern is opposite direction where pointers start at both ends and move toward each other. I use this for sorted array problems like two sum - if the sum is too high, move the right pointer left, if too low, move left pointer right. Second is same direction where both pointers start at the beginning but move at different speeds. This is perfect for removing duplicates - slow pointer marks where to write next unique element, fast pointer scans ahead to find it. Third is sliding window where two pointers define a window that expands and shrinks. Use this for subarray problems like finding maximum sum of size k - expand by moving right, shrink by moving left, maintain the window constraint as you go.',
          keyPoints: [
            'Opposite direction: start at ends, meet in middle (two sum)',
            'Same direction: both start at beginning, different speeds (remove duplicates)',
            'Sliding window: define expanding/shrinking window (max sum subarray)',
            'Choose pattern based on problem structure',
          ],
        },
        {
          id: 'q2',
          question:
            'For the opposite direction pattern, explain the decision-making process. How do you decide which pointer to move?',
          sampleAnswer:
            'In opposite direction, the decision is based on comparing your current result with what you want. For two sum, if the current sum is too large, I know I need a smaller number, so I move the right pointer left to get smaller values. If the sum is too small, I need a larger number, so I move the left pointer right. The key insight is that moving the wrong pointer makes things worse - if sum is already too big and I move left pointer right, I am adding an even larger number. The sorted property guarantees that moving a pointer in one direction consistently changes the result in a predictable way.',
          keyPoints: [
            'Compare current result with target',
            'Too high? Move right pointer left for smaller values',
            'Too low? Move left pointer right for larger values',
            'Sorted property makes decisions reliable',
          ],
        },
        {
          id: 'q3',
          question:
            'Describe how the fast and slow pointer approach works for removing duplicates. Why do we need two pointers instead of one?',
          sampleAnswer:
            'With fast and slow pointers for removing duplicates, the slow pointer marks the position where the next unique element should go, while the fast pointer searches for that next unique element. We need two because we are modifying the array in place - we cannot use one pointer to both read ahead and mark where to write. Fast pointer scans through the array looking for elements different from what slow is pointing to. When fast finds something new, we copy it to slow plus one position and move slow forward. This way we build up the deduplicated portion at the start of the array while fast explores the rest.',
          keyPoints: [
            'Slow marks write position for next unique',
            'Fast scans ahead to find different element',
            'Need two for in-place modification',
            'Build deduplicated section at array start',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question:
            'In the opposite direction pattern, when do the pointers stop moving?',
          options: [
            'When they point to the same element or cross',
            'When they reach the middle of the array',
            'After n iterations',
            'When one pointer reaches the end',
          ],
          correctAnswer: 0,
          explanation:
            'In the opposite direction pattern, pointers continue until they meet (point to the same element) or cross each other (left > right). This ensures all elements are considered.',
        },
        {
          id: 'mc2',
          question:
            'What is the key characteristic of the fast and slow pointer pattern?',
          options: [
            'Both pointers move at the same speed',
            'Pointers move from opposite ends',
            'Pointers start at the same position but move at different speeds',
            'One pointer is always twice as fast as the other',
          ],
          correctAnswer: 2,
          explanation:
            'The fast and slow pointer pattern has both pointers starting at the beginning (or same position) but moving at different speeds - often the fast pointer moves every iteration while the slow only moves conditionally.',
        },
        {
          id: 'mc3',
          question:
            'For the two sum problem on a sorted array, if the current sum is too large, which pointer should you move?',
          options: [
            'Move the left pointer right',
            'Move the right pointer left',
            'Move both pointers',
            "It doesn't matter",
          ],
          correctAnswer: 1,
          explanation:
            'If the sum is too large, you need smaller numbers. Since the array is sorted, moving the right pointer left gives you a smaller value, reducing the sum.',
        },
        {
          id: 'mc4',
          question:
            'What problem type is the sliding window pattern best suited for?',
          options: [
            'Finding pairs in sorted arrays',
            'Checking palindromes',
            'Subarray problems with size or property constraints',
            'Removing duplicates',
          ],
          correctAnswer: 2,
          explanation:
            'Sliding window is ideal for subarray problems where you need to maintain a window with certain constraints (fixed size, max sum, unique characters, etc.), expanding or shrinking as needed.',
        },
        {
          id: 'mc5',
          question:
            'In the remove duplicates problem, what does the slow pointer represent?',
          options: [
            'The current element being checked',
            'The position where the next unique element should be written',
            'The last duplicate found',
            'The middle of the array',
          ],
          correctAnswer: 1,
          explanation:
            'The slow pointer marks the end of the unique elements section - it points to the last unique element, so the next unique element will be written at slow + 1.',
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Walk me through the two sum sorted array algorithm step by step using an example. Explain your thought process at each decision point.',
          sampleAnswer:
            'Let me use [1, 3, 5, 7, 9] with target 12. I start with left at 1 and right at 9. Sum is 10, which is less than 12, so I need a bigger number - move left right to 3. Now sum is 12, found it! The key decisions are: after each sum calculation, I compare with target. Too small means I need bigger numbers, so move left pointer right into larger values. Too big means I need smaller numbers, so move right pointer left into smaller values. If they ever cross without finding it, no solution exists. Each move eliminates possibilities - like when sum was 10, I knew anything paired with 1 would be too small, so I never need to check those pairs.',
          keyPoints: [
            'Start at opposite ends of sorted array',
            'Compare sum with target after each check',
            'Too small → move left right for larger values',
            'Too big → move right left for smaller values',
            'Each move eliminates a set of pairs',
          ],
        },
        {
          id: 'q2',
          question:
            'In the remove duplicates algorithm, explain what slow and fast represent at any point during execution.',
          sampleAnswer:
            'At any moment during remove duplicates, slow points to the last position that contains a confirmed unique element in our result. Everything from index 0 to slow is the deduplicated portion we have built so far. Fast points to the element we are currently examining to see if it is different from what slow is pointing at. So slow says "this is where my result ends so far" and fast says "let me check if this new element belongs in the result". When fast finds something new, we put it at slow plus 1 and advance slow. The gap between slow and fast contains elements we have already processed and determined to be duplicates.',
          keyPoints: [
            'Slow: last position of confirmed unique element',
            'Everything from 0 to slow is deduplicated result',
            'Fast: currently examining this element',
            'Gap between them: processed duplicates',
          ],
        },
        {
          id: 'q3',
          question:
            'Why is it important that the array is sorted for two sum with two pointers? What breaks if it is not sorted?',
          sampleAnswer:
            'Sorting is critical because it gives us the monotonic property - moving left pointer right consistently increases values, moving right pointer left consistently decreases values. This lets us make reliable decisions. If the array is not sorted, say [5, 1, 9, 3], and sum is too big, which pointer do I move? I cannot tell! Moving left might give me a smaller or larger number. The algorithm depends on knowing that moving a pointer in one direction predictably changes the sum. Without sorting, we lose this guarantee and the algorithm fails. That is why the brute force O(n²) check-all-pairs approach is needed for unsorted arrays.',
          keyPoints: [
            'Sorting gives monotonic property',
            'Moving pointers predictably changes sum',
            'Without sorting, cannot make reliable decisions',
            'Would need brute force O(n²) for unsorted',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question:
            'What is the time complexity of the two sum algorithm using two pointers on a sorted array?',
          options: ['O(1)', 'O(log n)', 'O(n)', 'O(n²)'],
          correctAnswer: 2,
          explanation:
            'The two pointers algorithm runs in O(n) time because each pointer traverses the array at most once, and combined they cover all elements in a single pass.',
        },
        {
          id: 'mc2',
          question:
            'In the two sum algorithm, when the current sum equals the target, what should you do?',
          options: [
            'Continue searching for more pairs',
            'Return the indices immediately',
            'Move both pointers',
            'Sort the array again',
          ],
          correctAnswer: 1,
          explanation:
            "When you find a pair whose sum equals the target, you can return the indices immediately since you've found the solution.",
        },
        {
          id: 'mc3',
          question:
            'In the remove duplicates algorithm, what is the initial value of the slow pointer?',
          options: ['-1', '0', '1', 'len(array) - 1'],
          correctAnswer: 1,
          explanation:
            'The slow pointer starts at 0 (the first element) since the first element is always unique. The fast pointer typically starts at 1 to begin comparing.',
        },
        {
          id: 'mc4',
          question:
            'What is the return value of the remove duplicates function?',
          options: [
            'The modified array',
            'The number of unique elements',
            'A list of duplicates',
            'True or False',
          ],
          correctAnswer: 1,
          explanation:
            'The function returns slow + 1, which represents the length/count of unique elements. The array is modified in-place, and the first slow + 1 elements contain the unique values.',
        },
        {
          id: 'mc5',
          question:
            'Why is sorting a prerequisite for the two sum two-pointers algorithm?',
          options: [
            'To make the code simpler',
            'To provide predictable behavior when moving pointers',
            'To reduce space complexity',
            'It is not actually required',
          ],
          correctAnswer: 1,
          explanation:
            'Sorting ensures that moving a pointer in one direction consistently increases or decreases values, allowing reliable decisions about which pointer to move based on whether the sum is too large or too small.',
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Explain why two pointers is O(n) time complexity, not O(n²). What is the key insight about how the pointers move?',
          sampleAnswer:
            'Two pointers is O(n) because each pointer traverses the array at most once - they never backtrack. Even though we have two pointers, they move a combined total of at most n steps. Think of it this way: in opposite direction, they start n apart and meet in the middle, covering n positions total. In same direction, fast pointer does at most n moves, and slow pointer does at most n moves, but we process each element once. The key is that we never revisit elements or check the same pairs multiple times. Compare this to nested loops where the inner loop resets for each outer loop iteration, giving n × n checks.',
          keyPoints: [
            'Each pointer moves at most n times',
            'Combined movement is O(n), not O(n) + O(n) = O(2n) = O(n)',
            'Never backtrack or revisit elements',
            'Process each element once',
            'Unlike nested loops that reset inner loop',
          ],
        },
        {
          id: 'q2',
          question:
            'Talk about the space complexity of two pointers. Why is it often better than other approaches?',
          sampleAnswer:
            'Two pointers typically has O(1) space complexity because we only need a few extra variables - the two pointer positions and maybe a couple tracking variables. We are not creating any data structures that grow with input size. This is especially powerful when doing in-place modifications like removing duplicates - we transform the array using just a couple of pointers without needing a separate result array. Compare this to approaches that use hash maps (O(n) space) or store intermediate results. The in-place nature of two pointers makes it very memory efficient, which matters for large datasets or memory-constrained environments.',
          keyPoints: [
            'Usually O(1) space - just pointer variables',
            'No data structures that grow with input',
            'In-place modifications possible',
            'More memory efficient than hash map approaches',
            'Great for large datasets',
          ],
        },
        {
          id: 'q3',
          question:
            'Some problems can be solved with either two pointers or a hash map. How would you decide which approach to use?',
          sampleAnswer:
            'I consider several factors. If the array is already sorted or I can sort it without breaking requirements, two pointers is often cleaner and uses O(1) space versus O(n) for a hash map. Two pointers is also better when I need to modify in-place or when I care about memory. However, hash map wins when sorting would break the problem (like needing to return original indices in unsorted order), or when the problem needs more complex lookups than just comparing two elements. Hash map gives O(1) lookup but uses more space. If space is tight and data is sorted, two pointers. If I need fast arbitrary lookups and space is not an issue, hash map.',
          keyPoints: [
            'Sorted/sortable data → two pointers (O(1) space)',
            'Need original indices/order → hash map',
            'In-place modification needed → two pointers',
            'Complex lookups needed → hash map',
            'Trade-off: space vs convenience',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question:
            'What is the time complexity of the two pointers technique?',
          options: ['O(1)', 'O(log n)', 'O(n)', 'O(n²)'],
          correctAnswer: 2,
          explanation:
            'The two pointers technique is O(n) because each pointer traverses the array at most once. Even with two pointers, the combined movement is linear, not quadratic.',
        },
        {
          id: 'mc2',
          question:
            'What is the space complexity of most two-pointer algorithms?',
          options: ['O(1)', 'O(log n)', 'O(n)', 'O(n²)'],
          correctAnswer: 0,
          explanation:
            'Most two-pointer algorithms use O(1) constant space as they only require a few pointer variables and often modify arrays in-place without additional data structures.',
        },
        {
          id: 'mc3',
          question:
            'How does the time complexity change if you need to sort the array first before using two pointers?',
          options: [
            'Stays O(n)',
            'Becomes O(n log n)',
            'Becomes O(n²)',
            'Becomes O(log n)',
          ],
          correctAnswer: 1,
          explanation:
            'If sorting is required first, the overall time complexity becomes O(n log n) because sorting dominates. The two pointers part is still O(n), but O(n log n) + O(n) = O(n log n).',
        },
        {
          id: 'mc4',
          question:
            'Why is two pointers more cache-friendly than nested loops?',
          options: [
            'It uses less memory',
            'It accesses elements sequentially',
            'It runs faster on all computers',
            'It uses recursion',
          ],
          correctAnswer: 1,
          explanation:
            'Two pointers accesses array elements sequentially, which is cache-friendly because it takes advantage of spatial locality. Nested loops may jump around more, leading to more cache misses.',
        },
        {
          id: 'mc5',
          question:
            'When comparing two pointers vs hash map for the two sum problem, what is the key trade-off?',
          options: [
            'Time vs readability',
            'Space (O(1) vs O(n)) vs need for sorted input',
            'Speed vs accuracy',
            'Complexity vs simplicity',
          ],
          correctAnswer: 1,
          explanation:
            'Two pointers uses O(1) space but requires sorted input, while hash map uses O(n) space but works on unsorted input. The trade-off is space efficiency versus the requirement to sort.',
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Walk me through how you would modify the two sum template to find three numbers that sum to a target (3Sum problem).',
          sampleAnswer:
            'For 3Sum, I would first sort the array, then use one loop plus two pointers. For each element at index i, I treat it as the first number and then use two pointers to find two other numbers that sum to target minus that first number. So outer loop fixes one element, inner two pointers solve "two sum equals target minus first element". The two pointers work just like regular two sum - start at ends, move based on whether sum is too big or small. Key detail: I need to skip duplicates at all three levels to avoid duplicate triplets. This is O(n²) because of the outer loop times the O(n) two pointer search.',
          keyPoints: [
            'Sort array first',
            'Outer loop fixes first element',
            'Two pointers find other two elements',
            'Becomes 2Sum for (target - first element)',
            'Skip duplicates to avoid duplicate triplets',
            'Time: O(n²)',
          ],
        },
        {
          id: 'q2',
          question:
            'Explain the partition template where you separate elements into two groups. How do the pointers work differently than in the other patterns?',
          sampleAnswer:
            'In partition, I use two pointers to separate elements based on some condition - like all evens before all odds. Both pointers usually move inward from the ends. When left pointer finds an element that belongs in the right section, and right pointer finds an element that belongs in the left section, I swap them. Then I move both pointers. This is different from other patterns because I am actively swapping elements to reorder them, not just reading or comparing. The pointers converge toward each other, and when they meet, the array is partitioned. Think of it like organizing a bookshelf - left hand grabs books that should be on right, right hand grabs books that should be on left, swap them.',
          keyPoints: [
            'Separate elements based on condition',
            'Both pointers move inward from ends',
            'Swap elements when both find misplaced items',
            'Actively reordering, not just reading',
            'Done when pointers meet',
          ],
        },
        {
          id: 'q3',
          question:
            'When should you choose the same-direction pattern over the opposite-direction pattern? What is the key difference in what they are suited for?',
          sampleAnswer:
            'I choose same direction when I need to build up a result incrementally in place, like removing duplicates or moving zeros. The key is that slow pointer marks where I am writing my result, while fast pointer scans ahead. Opposite direction is for finding relationships between elements at different positions - like pairs that sum to a target. Same direction reads ahead and writes behind. Opposite direction compares ends and works inward. If the problem says "remove" or "move elements" or "in-place modification", I think same direction. If it says "find pair" or "two numbers" or involves symmetric operations, I think opposite direction.',
          keyPoints: [
            'Same direction: in-place incremental building',
            'Slow writes, fast scans ahead',
            'Opposite: finding relationships between positions',
            'Remove/move → same direction',
            'Find pairs/symmetric → opposite direction',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question:
            'In the "pair with sum" template, what do you return when no pair is found?',
          options: [
            'null',
            'Empty array or [-1, -1]',
            'The closest pair',
            'An error',
          ],
          correctAnswer: 1,
          explanation:
            'When no pair sums to the target, the function typically returns an empty array [] or [-1, -1] to indicate no solution was found.',
        },
        {
          id: 'mc2',
          question:
            'In the "remove element" template, what does the slow pointer represent?',
          options: [
            'Elements to be removed',
            'The next position to write a kept element',
            'The current element being checked',
            'The end of the array',
          ],
          correctAnswer: 1,
          explanation:
            'The slow pointer marks the position where the next element that should be kept (not removed) will be written.',
        },
        {
          id: 'mc3',
          question:
            'For the 3Sum problem, what is the overall time complexity?',
          options: ['O(n)', 'O(n log n)', 'O(n²)', 'O(n³)'],
          correctAnswer: 2,
          explanation:
            '3Sum requires sorting (O(n log n)) plus an outer loop (O(n)) with two pointers inside (O(n)), giving O(n) × O(n) = O(n²). The sorting is dominated by the nested operations.',
        },
        {
          id: 'mc4',
          question: 'In the partition template, when do you swap elements?',
          options: [
            'After every iteration',
            'When left finds an element for right section AND right finds one for left section',
            'Only at the end',
            'When pointers meet',
          ],
          correctAnswer: 1,
          explanation:
            'You swap when the left pointer finds an element that belongs in the right section AND the right pointer finds an element that belongs in the left section - both conditions must be true.',
        },
        {
          id: 'mc5',
          question:
            'What pattern would you use for the "move zeros to end" problem?',
          options: [
            'Opposite direction',
            'Sliding window',
            'Same direction (fast & slow)',
            'Binary search',
          ],
          correctAnswer: 2,
          explanation:
            'Move zeros uses the same direction pattern where slow marks position for next non-zero, and fast scans ahead to find non-zero elements to move.',
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'For the "Container With Most Water" problem, explain the counter-intuitive insight about which pointer to move. Why do we move the pointer at the shorter line?',
          sampleAnswer:
            'The insight is that the area is limited by the shorter of the two heights. If I move the pointer at the taller line, the width decreases and the height can only stay the same or get worse (because it is limited by the shorter side). So moving the tall side can only make things worse. But if I move the pointer at the shorter side, width still decreases, but I have a chance of finding a taller line that could compensate for the lost width. It is the only move that has potential to improve. Think of it like this: the short side is the bottleneck, so we try to fix the bottleneck, not the side that is already good.',
          keyPoints: [
            'Area limited by shorter height',
            'Moving tall side: width down, height same or worse',
            'Moving short side: width down, but height might improve',
            'Short side is the bottleneck',
            'Only move with potential to improve',
          ],
        },
        {
          id: 'q2',
          question:
            'Talk through the 3Sum problem. How does it extend the two pointer technique, and what is the time complexity?',
          sampleAnswer:
            'In 3Sum, I want three numbers that sum to zero. I cannot use just two pointers for three numbers, so I add an outer loop. I fix the first number with the loop, then use two pointers to find the other two numbers that sum to negative of the first number. So it becomes: for each element, solve 2Sum with target equals negative that element. The two pointers part is still O(n), but I do it n times in the outer loop, so overall O(n²). I also need to handle duplicates carefully by skipping over repeated values at all three positions to avoid returning duplicate triplets. Sort the array first to enable the two pointer technique.',
          keyPoints: [
            'Fix first number with loop',
            'Use 2Sum for other two numbers',
            'Target for 2Sum: -(first number)',
            'Time: O(n²) = n iterations × O(n) 2Sum',
            'Handle duplicates by skipping repeats',
            'Sort array first',
          ],
        },
        {
          id: 'q3',
          question:
            'Describe cycle detection in linked lists using fast and slow pointers (Floyd Cycle Detection). How does it work?',
          sampleAnswer:
            'For cycle detection, I use two pointers starting at the head. Slow moves one step at a time, fast moves two steps. If there is no cycle, fast reaches the end and we are done. If there is a cycle, fast will eventually lap slow and they will meet inside the cycle - guaranteed because fast is gaining one step per iteration and the cycle is finite. Once they meet, I know there is a cycle. To find where the cycle starts, I can reset one pointer to head and move both one step at a time - they will meet at the cycle entrance. This works due to mathematical properties of the distances involved.',
          keyPoints: [
            'Slow moves 1 step, fast moves 2 steps',
            'No cycle: fast reaches end',
            'Cycle exists: fast laps slow, they meet',
            'Fast gains 1 step per iteration',
            'To find cycle start: reset one to head, move both 1 step',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question:
            'In the Container With Most Water problem, which pointer should you move?',
          options: [
            'Always move the left pointer',
            'Always move the right pointer',
            'Move the pointer at the shorter line',
            'Move the pointer at the taller line',
          ],
          correctAnswer: 2,
          explanation:
            'Always move the pointer at the shorter line because the area is limited by the shorter height. Moving the taller pointer can only decrease or maintain the area, while moving the shorter pointer gives a chance to find a taller line.',
        },
        {
          id: 'mc2',
          question:
            'What is the time complexity of the 3Sum problem using two pointers?',
          options: ['O(n)', 'O(n log n)', 'O(n²)', 'O(n³)'],
          correctAnswer: 2,
          explanation:
            "3Sum uses an outer loop O(n) with two pointers inside O(n), resulting in O(n²) time. Although sorting takes O(n log n), it's dominated by the O(n²) nested operations.",
        },
        {
          id: 'mc3',
          question:
            "In Floyd's cycle detection, what is the speed ratio between fast and slow pointers?",
          options: [
            'Fast is 3x slow',
            'Fast is 2x slow',
            'They move at same speed',
            'Fast is 4x slow',
          ],
          correctAnswer: 1,
          explanation:
            "In Floyd's cycle detection algorithm, the slow pointer moves 1 step at a time while the fast pointer moves 2 steps at a time - a 2:1 speed ratio.",
        },
        {
          id: 'mc4',
          question:
            'For the Dutch National Flag problem (sort 0s, 1s, 2s), how many pointers are typically used?',
          options: ['One', 'Two', 'Three', 'Four'],
          correctAnswer: 2,
          explanation:
            'The Dutch National Flag problem uses three pointers: one for the boundary of 0s, one for the boundary of 2s, and one for the current element being examined.',
        },
        {
          id: 'mc5',
          question: 'In 3Sum, why must you skip duplicate values?',
          options: [
            'To improve time complexity',
            'To avoid returning duplicate triplets in the result',
            'To reduce space complexity',
            'Because the algorithm breaks otherwise',
          ],
          correctAnswer: 1,
          explanation:
            'Skipping duplicates prevents returning multiple instances of the same triplet. For example, if the array has [1,1,1], you want to return each unique combination only once.',
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'How would you recognize in an interview that a problem can be solved with two pointers? Walk me through your thought process.',
          sampleAnswer:
            'First thing I check: is the input sorted or can I sort it? That is a huge signal. Second, I look at the problem description - words like "pair", "two numbers", "remove", "partition", or "in-place" jump out at me. Third, if I start thinking "I need to check every pair" that is when I pause and ask if two pointers could be smarter. I also think about what information I have - if comparing elements at different positions helps make decisions, maybe two pointers. If the problem involves symmetry like palindromes, definitely two pointers. And if I need O(1) space for in-place modification, that is another hint. The key question: can I make progress by moving pointers based on comparisons?',
          keyPoints: [
            'Is data sorted or sortable?',
            'Keywords: pair, two numbers, remove, partition, in-place',
            'Alternative to checking all pairs',
            'Symmetry or palindrome problems',
            'Need O(1) space?',
            'Can I make decisions by comparing?',
          ],
        },
        {
          id: 'q2',
          question:
            'In an interview, how would you explain your choice of which two pointer pattern to use for a given problem?',
          sampleAnswer:
            'I would explain my reasoning out loud. For opposite direction, I would say "since the array is sorted and I need to find a pair with a specific sum, I will use pointers at both ends so I can increase or decrease the sum by moving the appropriate pointer". For same direction, I would say "I need to remove duplicates in place, so I will use slow pointer to mark where to write unique elements and fast pointer to scan ahead and find them". For sliding window, I would say "this is asking for a subarray with constraints, so I will use two pointers to define a window and slide it while maintaining those constraints". The key is explaining why the pattern fits the problem structure.',
          keyPoints: [
            'Explain reasoning out loud',
            'Connect pattern to problem requirements',
            'Opposite: sorted + pairs/relationships',
            'Same: in-place building/removal',
            'Window: subarray with constraints',
            'Show you understand why it works',
          ],
        },
        {
          id: 'q3',
          question:
            'Walk me through how you would test and debug a two pointer solution during an interview.',
          sampleAnswer:
            'First, I would trace through a small example by hand, writing down the pointer positions at each step. I watch for: do the pointers move as expected? Does the loop terminate? Are edge cases handled? I test with edge cases like empty array, single element, all same values, and target at boundaries. For opposite direction, I make sure pointers do not cross incorrectly. For same direction, I verify slow and fast are doing their jobs. If there is a bug, I add print statements to track pointer values each iteration and check if the logic conditions are right. I also verify I am using correct pointer updates like mid plus one instead of just mid. The key is being systematic and explaining my debugging process out loud.',
          keyPoints: [
            'Trace small example by hand',
            'Check: do pointers move correctly, does loop terminate?',
            'Test edge cases: empty, single element, boundaries',
            'Print pointer values if debugging',
            'Verify loop conditions and pointer updates',
            'Explain process out loud',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question:
            'What is the first thing you should check to recognize a two-pointer problem?',
          options: [
            'If the array is small',
            'If the data is sorted or can be sorted',
            'If recursion is needed',
            'If a hash map would work',
          ],
          correctAnswer: 1,
          explanation:
            'The first signal for two pointers is whether the data is sorted or can be sorted. Sorted data enables predictable pointer movement decisions based on comparisons.',
        },
        {
          id: 'mc2',
          question:
            'Which keywords in a problem statement suggest using two pointers?',
          options: [
            'Recursive, stack, tree',
            'Pair, two numbers, remove, partition, in-place',
            'Hash, frequency, count',
            'Binary, search, logarithmic',
          ],
          correctAnswer: 1,
          explanation:
            'Keywords like "pair", "two numbers", "remove", "partition", and "in-place" strongly suggest two pointers. These indicate pair-finding or in-place modification patterns.',
        },
        {
          id: 'mc3',
          question: 'What is a common mistake when implementing two pointers?',
          options: [
            'Using too much memory',
            'Incorrect loop termination conditions (e.g., left <= right vs left < right)',
            'Making it too fast',
            'Not using recursion',
          ],
          correctAnswer: 1,
          explanation:
            'A common mistake is incorrect loop termination conditions, like using left <= right when you should use left < right, or vice versa. This can cause infinite loops or missed cases.',
        },
        {
          id: 'mc4',
          question:
            'How much time should you spend planning before coding a two-pointer solution in an interview?',
          options: [
            'No time, start coding immediately',
            '2-3 minutes',
            '10-15 minutes',
            'Half the interview time',
          ],
          correctAnswer: 1,
          explanation:
            'You should spend 2-3 minutes planning your approach, choosing the right pattern, and thinking through edge cases before writing code. This prevents costly mistakes.',
        },
        {
          id: 'mc5',
          question:
            'What edge cases should you always test with two-pointer solutions?',
          options: [
            'Only test with large inputs',
            'Empty array, single element, all same values, target at boundaries',
            'Only test the happy path',
            'Only test when there is a bug',
          ],
          correctAnswer: 1,
          explanation:
            "Always test edge cases: empty array (what if there's no input?), single element (do pointers work?), all same values (duplicates), and targets at boundaries (first/last elements).",
        },
      ],
    },
    {
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
      quiz: [
        {
          id: 'q1',
          question:
            'Why does two pointers fail for the classic "Two Sum" problem (return indices)? What should you use instead?',
          hint: 'Think about what happens to indices when you sort.',
          sampleAnswer:
            'Two pointers typically requires sorting the array, but Two Sum asks for the ORIGINAL indices of the two numbers. Once you sort, the indices change and you lose track of original positions. Example: nums = [3,2,4], target = 6. Original indices of 2 and 4 are [1,2]. After sorting to [2,3,4], two pointers would return sorted indices [0,2], which is wrong. Solution: Use hash map. As you iterate, store value->original_index. When you find target-current, lookup gives original index. This is O(n) time, O(n) space, and preserves indices.',
          keyPoints: [
            'Sorting destroys original index information',
            'Two Sum requires original indices, not sorted positions',
            'Hash map preserves value-to-index mapping',
            'Trade space O(n) for ability to keep indices',
            'Still O(n) time like two pointers, but maintains info',
          ],
        },
        {
          id: 'q2',
          question:
            'When would you NOT use sliding window (two-pointer variant) and what would you use instead? Give a specific example.',
          hint: 'Think about problems with multiple windows or non-contiguous elements.',
          sampleAnswer:
            'Do NOT use sliding window when you need multiple non-overlapping subarrays or when elements do not need to be contiguous. Example: "Maximum sum of two non-overlapping subarrays of length L and M." Sliding window tracks ONE contiguous window, but this needs TWO separate windows. If you try, you cannot simultaneously track both windows positions. Solution: Use dynamic programming or prefix sums. Track maximum sum of L-length subarray ending before current position, then for each M-length window, add the best previous L-length sum. This requires DP state tracking, not just two pointers.',
          keyPoints: [
            'Sliding window = one contiguous subarray',
            'Cannot track multiple independent windows',
            'Example: max sum of two non-overlapping subarrays',
            'Use DP to track best previous subarray',
            'Prefix sums help compute window sums efficiently',
          ],
        },
        {
          id: 'q3',
          question:
            'Explain why two pointers does not work for permutation problems. What technique should you use?',
          hint: 'Think about the nature of exploration and the ability to undo choices.',
          sampleAnswer:
            'Two pointers moves linearly forward through the array, possibly backward, but always in a single pass. Permutations require exploring ALL possible orderings, which means making a choice, exploring that path, then UNDOING the choice and trying alternatives. Example: permutations of [1,2,3] include [1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]. Two pointers cannot systematically generate these by just moving pointers - you need to try placing each element in each position. Solution: Use backtracking with recursion. At each step, try placing each remaining element, recurse, then remove it (backtrack) to try the next option.',
          keyPoints: [
            'Two pointers is linear, one-pass technique',
            'Permutations need exploration of decision tree',
            'Must undo choices and try alternatives',
            'Backtracking: choose, explore, unchoose',
            'Two pointers cannot backtrack through choices',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'When does two pointers fail for pair-finding problems?',
          options: [
            'When the array is sorted',
            'When you need the original indices',
            'When there is only one pair',
            'When the array is small',
          ],
          correctAnswer: 1,
          explanation:
            'Two pointers usually requires sorting, which loses original indices. If the problem asks for original indices (like Two Sum), use a hash map instead.',
        },
        {
          id: 'mc2',
          question:
            'Why is two pointers NOT suitable for finding all permutations?',
          options: [
            'Too slow',
            'Requires too much space',
            'Cannot backtrack to explore different orderings',
            'Only works on sorted arrays',
          ],
          correctAnswer: 2,
          explanation:
            'Two pointers moves forward/backward in a single pass. Permutations require backtracking - making choices, exploring, and undoing to try alternatives.',
        },
        {
          id: 'mc3',
          question:
            'What should you use instead of two pointers for "find all pairs with sum = target"?',
          options: [
            'Binary search',
            'Hash map or nested loops',
            'Dynamic programming',
            'Divide and conquer',
          ],
          correctAnswer: 1,
          explanation:
            'Two pointers finds ONE pair efficiently. For ALL pairs (especially with duplicates), use hash map to count occurrences or nested loops if small.',
        },
        {
          id: 'mc4',
          question:
            'When is two pointers inappropriate for 2D matrix problems?',
          options: [
            'When matrix is sorted',
            'When you need to explore regions or traverse in multiple directions',
            'When matrix is small',
            'Never, two pointers always works',
          ],
          correctAnswer: 1,
          explanation:
            'Two pointers is fundamentally 1D. For 2D region exploration, connected components, or multi-directional traversal, use DFS/BFS or DP.',
        },
        {
          id: 'mc5',
          question:
            'You need maximum sum of TWO non-overlapping subarrays. Why not use sliding window?',
          options: [
            'Sliding window is too slow',
            'Sliding window tracks one window, cannot track two independent windows simultaneously',
            'Sliding window only works on sorted arrays',
            'Sliding window uses too much space',
          ],
          correctAnswer: 1,
          explanation:
            'Sliding window (two-pointer technique) maintains one contiguous window. For multiple independent windows, use DP or prefix sums to track best previous windows.',
        },
      ],
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
