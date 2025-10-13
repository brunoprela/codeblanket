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
      quiz: [
        {
          id: 'q1',
          question:
            'Talk through what makes binary search work. What do you absolutely need before you can use this algorithm?',
          hint: 'Think about the array properties that enable the algorithm.',
          sampleAnswer:
            'Binary search requires the array to be sorted, either in ascending or descending order. This is fundamental because the entire algorithm is built on making decisions by comparing the target with the middle element. If the middle is smaller than the target, we know the target must be in the right half (in a sorted array). Without sorting, this decision-making breaks down - we would have no idea which half to search next, and the algorithm falls apart.',
          keyPoints: [
            'The array must be sorted',
            'Sorting enables the comparison-based elimination',
            'Each comparison reliably tells us which half contains the target',
            'Without sorting, we cannot determine direction',
          ],
        },
        {
          id: 'q2',
          question:
            'Walk me through why this algorithm is called "binary" search. What is binary about it?',
          sampleAnswer:
            'The name "binary" comes from the fact that at every step, we are making a binary decision - a two-way choice. When we compare our target with the middle element, we are essentially asking: is the target in the left half or the right half? There are only two possibilities, and we choose one. This happens at every level, creating a binary decision tree as we go deeper. This binary choice at each step is what gives the algorithm its name and also why it is so efficient - we are repeatedly cutting the problem in half.',
          keyPoints: [
            'Binary means two-way decision at each step',
            'Left half or right half - only two choices',
            'Creates a binary decision tree structure',
            'Binary decisions lead to logarithmic efficiency',
          ],
        },
        {
          id: 'q3',
          question:
            'If I give you a sorted linked list, can you use binary search on it? Why or why not?',
          hint: 'Think about what you need to do to find the middle element.',
          sampleAnswer:
            'Even though the linked list is sorted, you cannot efficiently use binary search on it. The problem is random access - to find the middle element in a linked list, you have to start at the head and traverse node by node, which takes O(n) time. Binary search is only fast because it can jump directly to the middle element in O(1) time with arrays. If finding the middle takes O(n), and we do this at every step, we lose all the efficiency gains of binary search. So while technically you could implement it, it would be pointless because the time complexity would still be O(n).',
          keyPoints: [
            'Linked lists lack O(1) random access',
            'Finding middle requires O(n) traversal from head',
            'Binary search needs instant access to middle element',
            'With linked lists, efficiency advantage is lost',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What is the absolute requirement for using binary search?',
          options: [
            'The array must be large',
            'The array must be sorted',
            'The array must have unique elements',
            'The array must be in ascending order only',
          ],
          correctAnswer: 1,
          explanation:
            'Binary search requires the array to be sorted (either ascending or descending) so that comparisons with the middle element can reliably determine which half to search.',
        },
        {
          id: 'mc2',
          question: 'Why is it called "binary" search?',
          options: [
            'It works with binary numbers',
            'At each step, it makes a binary (two-way) decision',
            'It uses binary trees',
            'It divides by 2',
          ],
          correctAnswer: 1,
          explanation:
            'The name comes from the binary (two-way) decision made at each step: is the target in the left half or the right half? This binary choice creates a binary decision tree.',
        },
        {
          id: 'mc3',
          question: 'Can you efficiently use binary search on a sorted linked list?',
          options: [
            'Yes, as long as it is sorted',
            'No, because finding the middle element takes O(N) time',
            'Yes, but only for small lists',
            'No, because linked lists cannot be sorted',
          ],
          correctAnswer: 1,
          explanation:
            'Binary search requires O(1) random access to find the middle element. In a linked list, finding the middle takes O(N) time, eliminating the efficiency advantage.',
        },
        {
          id: 'mc4',
          question: 'What analogy best describes binary search?',
          options: [
            'Reading a book from start to finish',
            'Finding a word in a dictionary by repeatedly opening to the middle',
            'Sorting cards',
            'Counting items one by one',
          ],
          correctAnswer: 1,
          explanation:
            'Finding a word in a dictionary by opening to the middle and deciding which half to search next perfectly illustrates the binary search process.',
        },
        {
          id: 'mc5',
          question: 'What happens if you try binary search on an unsorted array?',
          options: [
            'It works but slower',
            'The algorithm breaks down because comparisons cannot reliably determine direction',
            'It automatically sorts the array first',
            'It still finds the element eventually',
          ],
          correctAnswer: 1,
          explanation:
            'Without sorting, comparing the target with the middle element gives no reliable information about which half contains the target, breaking the core logic of binary search.',
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Say you are at the middle element and it is smaller than your target. Walk me through your next move - what do you update and why?',
          hint: 'Where could the target possibly be if the middle is too small?',
          sampleAnswer:
            'If the middle element is smaller than my target, I know the target has to be somewhere to the right because the array is sorted. So I update left to be mid + 1, which means I am now only looking at the right half of the array. The key is using mid + 1, not just mid, because I have already checked mid and know it is not my target. So I want to exclude it and look at everything to the right of it. This is how I eliminate half the search space in one comparison.',
          keyPoints: [
            'Target must be in the right half',
            'Update left = mid + 1 to search right',
            'Use mid + 1 (not mid) to exclude the checked element',
            'Eliminates left half of search space',
          ],
        },
        {
          id: 'q2',
          question:
            'You might see two different ways to calculate the middle: (left + right) // 2 versus left + (right - left) // 2. Talk about which one you would use and why.',
          sampleAnswer:
            'I would use left + (right - left) // 2, especially if writing code in languages like Java or C++. The reason is integer overflow. When you add two very large integers together with (left + right), you can actually overflow and get a negative number or wrap around, which would break your algorithm completely. By doing left + (right - left) // 2 instead, you keep the numbers smaller during the calculation and avoid that overflow. In Python it does not matter as much because Python handles big integers automatically, but it is a good habit to use the safer formula.',
          keyPoints: [
            'left + (right - left) // 2 prevents integer overflow',
            '(left + right) can overflow in Java/C++ with large numbers',
            'Both give the same mathematical result',
            'Safer formula is a best practice',
          ],
        },
        {
          id: 'q3',
          question:
            'Should your loop condition be "while left < right" or "while left <= right"? Explain your choice.',
          hint: 'What happens when left and right are pointing to the same element?',
          sampleAnswer:
            'It should be "while left <= right" with the equal sign. Here is why: when left equals right, there is still one element left that needs to be checked - they are both pointing to the same index. If I used just "left < right", the loop would stop before checking that final element, and I might miss my target if it happens to be that last one. The equal sign ensures I check every single element in my search space before giving up and returning that the target was not found.',
          keyPoints: [
            'Use "while left <= right" with equal sign',
            'When left == right, one element still needs checking',
            'Without <=, you skip the final element',
            'Ensures complete search space coverage',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'If the middle element is smaller than the target, what should you do next?',
          options: [
            'Set right = mid - 1',
            'Set left = mid + 1',
            'Set left = mid',
            'Return -1',
          ],
          correctAnswer: 1,
          explanation:
            'When the middle element is smaller than the target, the target must be in the right half (since the array is sorted). Update left = mid + 1 to search the right half, excluding the already-checked middle element.',
        },
        {
          id: 'mc2',
          question: 'Why is mid = left + (right - left) // 2 preferred over mid = (left + right) // 2?',
          options: [
            'It is faster',
            'It prevents integer overflow in languages like Java/C++',
            'It gives a different result',
            'It uses less memory',
          ],
          correctAnswer: 1,
          explanation:
            'The formula left + (right - left) // 2 prevents integer overflow that can occur when adding two large integers together. This is important in languages like Java and C++.',
        },
        {
          id: 'mc3',
          question: 'Should the loop condition be "while left < right" or "while left <= right"?',
          options: [
            'while left < right',
            'while left <= right',
            'Both work the same',
            'It doesn\'t matter',
          ],
          correctAnswer: 1,
          explanation:
            'Use "while left <= right" to ensure the final element is checked when left equals right. Without the equal sign, you would skip checking the last element.',
        },
        {
          id: 'mc4',
          question: 'After finding that nums[mid] == target, what should you return?',
          options: [
            'target',
            'mid',
            'nums[mid]',
            'true',
          ],
          correctAnswer: 1,
          explanation:
            'Return mid, which is the index where the target was found. The problem typically asks for the index, not the value itself.',
        },
        {
          id: 'mc5',
          question: 'If the loop exits without finding the target, what should you return?',
          options: [
            '0',
            '-1',
            'null',
            'false',
          ],
          correctAnswer: 1,
          explanation:
            'Return -1 to indicate the target was not found in the array. This is the standard convention in most binary search implementations.',
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Explain in your own words why binary search is O(log n). What is the mathematical reasoning behind that logarithm?',
          hint: 'Think about how many times you can cut something in half.',
          sampleAnswer:
            'Binary search is O(log n) because with each comparison, we cut the problem size in half. If I start with say 1000 elements, after one comparison I have 500 left, then 250, then 125, and so on. The question is: how many times can I divide n by 2 until I get down to 1? That is exactly what a logarithm tells us. If I have n/2^k = 1, solving for k gives me k = log n. So no matter how big the array is, I only need log n comparisons. This is why binary search is so incredibly fast - for a million elements, I only need about 20 comparisons.',
          keyPoints: [
            'Each comparison cuts search space in half',
            'Pattern: n → n/2 → n/4 → n/8 → ... → 1',
            'How many divisions by 2? That is log₂(n)',
            'For 1 million elements, only ~20 comparisons needed',
          ],
        },
        {
          id: 'q2',
          question:
            'If I asked you to implement binary search, would you do it iteratively or recursively? Talk through the space complexity implications of each approach.',
          sampleAnswer:
            'I would go with the iterative approach. The iterative version is O(1) space - I only need three variables: left, right, and mid. These variables do not grow with input size. If I did it recursively, each recursive call adds a frame to the call stack, and since I make about log n calls (one for each level going down the tree), that is O(log n) extra space. So iterative is more space efficient. That said, recursive can be cleaner to read, but in production I would usually prefer iterative to avoid any stack overflow issues with very deep recursion.',
          keyPoints: [
            'Iterative: O(1) space - just a few variables',
            'Recursive: O(log n) space - call stack grows',
            'Each recursive call adds stack frame',
            'Iterative preferred for production code',
          ],
        },
        {
          id: 'q3',
          question:
            'Unlike linear search which can vary a lot, binary search is pretty consistent. Why is that? Talk about best, average, and worst case.',
          sampleAnswer:
            'Binary search is consistent because it does not matter where the target is - we always do roughly the same amount of work. Whether the target is at the beginning, end, or somewhere in the middle, we still divide the array in half each time and keep going. The worst case and average case are both O(log n). The only exception is if we get really lucky and the target is exactly in the middle on the first try - that is O(1), but that is rare. Compare this to linear search where if the target is at the end, you check every element, but if it is at the start, you check just one. Binary search is way more predictable.',
          keyPoints: [
            'Always divides in half regardless of target location',
            'Worst and average cases both O(log n)',
            'Only best case O(1) - lucky first guess',
            'Much more predictable than linear search',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What is the time complexity of binary search?',
          options: [
            'O(N)',
            'O(log N)',
            'O(N log N)',
            'O(1)',
          ],
          correctAnswer: 1,
          explanation:
            'Binary search has O(log N) time complexity because it divides the search space in half with each comparison. For N elements, it takes at most log₂(N) comparisons.',
        },
        {
          id: 'mc2',
          question: 'How many comparisons are needed to search 1 million elements with binary search?',
          options: [
            '1,000',
            'About 20',
            '1,000,000',
            '100',
          ],
          correctAnswer: 1,
          explanation:
            'Binary search needs at most log₂(1,000,000) ≈ 20 comparisons. This is because 2^20 = 1,048,576, which is just over 1 million.',
        },
        {
          id: 'mc3',
          question: 'What is the space complexity of iterative binary search?',
          options: [
            'O(log N)',
            'O(1)',
            'O(N)',
            'O(N log N)',
          ],
          correctAnswer: 1,
          explanation:
            'Iterative binary search uses O(1) constant space - only a few variables (left, right, mid) are needed regardless of input size.',
        },
        {
          id: 'mc4',
          question: 'What is the space complexity of recursive binary search?',
          options: [
            'O(1)',
            'O(log N)',
            'O(N)',
            'O(N²)',
          ],
          correctAnswer: 1,
          explanation:
            'Recursive binary search uses O(log N) space for the call stack. Each recursive call adds a frame to the stack, and there are at most log N calls.',
        },
        {
          id: 'mc5',
          question: 'What is the best case time complexity of binary search?',
          options: [
            'O(log N)',
            'O(1)',
            'O(N)',
            'Best case does not exist',
          ],
          correctAnswer: 1,
          explanation:
            'The best case is O(1) when the target happens to be exactly at the middle position on the first comparison. However, this is rare and average/worst cases are O(log N).',
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Say you have an array with duplicates and you need to find the first occurrence of a target. How does that change your binary search implementation? Walk through what happens when you find a match.',
          hint: 'When you find the target, are you done searching?',
          sampleAnswer:
            'When I find a match at the middle, I cannot just return immediately because there might be earlier occurrences to the left. So I save that index as my result, but then I keep searching left by setting right = mid - 1. This way, if there are earlier copies, I will find them. If there are not, I still have that original match saved. The opposite works for finding the last occurrence - I would continue searching right instead. The key insight is that finding a match is not the end, it is just a candidate answer that might get improved.',
          keyPoints: [
            'Save the match, but keep searching',
            'Find First: continue left (right = mid - 1)',
            'Find Last: continue right (left = mid + 1)',
            'Used when duplicates exist and boundaries matter',
          ],
        },
        {
          id: 'q2',
          question:
            'In the "search insert position" variant, you return left at the end instead of -1. Why does that make sense?',
          sampleAnswer:
            'The left pointer ends up exactly where you need to insert the target to keep the array sorted. Think about what happens: if the target is not in the array, left keeps moving until it finds the spot where the target should go. If the target is smaller than everything, left stays at 0 - insert at the beginning. If it is larger than everything, left ends up at the end of the array. And if it is somewhere in between, left stops at exactly the right insertion point. So left naturally gives you the answer without any extra logic.',
          keyPoints: [
            'left ends at the correct insertion position',
            'Works for all cases: beginning, middle, end',
            'Natural result of how pointers move',
            'Used for insert position, floor/ceiling problems',
          ],
        },
        {
          id: 'q3',
          question:
            'When would you reach for the "find first" or "find last" templates instead of the classic binary search?',
          sampleAnswer:
            'I would use these templates when I am dealing with duplicates and I need specific boundaries. Classic binary search will just give me any match, which is useless if I am trying to count how many times something appears or find a range. For example, if I want to find how many 5s are in a sorted array, I need to find the first 5 and the last 5, then subtract. Or if I am searching for the first element that satisfies some condition, I need the boundary-finding version. Anytime the word "first" or "last" or "range" appears in the problem, I am thinking about these templates.',
          keyPoints: [
            'Use when duplicates exist',
            'Needed for range queries',
            'Required for counting occurrences',
            'Any problem asking for first/last/boundaries',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'When finding the first occurrence of a target in an array with duplicates, what should you do after finding nums[mid] == target?',
          options: [
            'Return mid immediately',
            'Save mid as result and continue searching left (right = mid - 1)',
            'Set left = mid',
            'Break from the loop',
          ],
          correctAnswer: 1,
          explanation:
            'Save the match as a candidate result, but continue searching left by setting right = mid - 1 to find any earlier occurrences. This ensures you find the leftmost match.',
        },
        {
          id: 'mc2',
          question: 'When finding the last occurrence of a target with duplicates, what should you do after finding nums[mid] == target?',
          options: [
            'Return mid immediately',
            'Save mid as result and continue searching right (left = mid + 1)',
            'Set right = mid',
            'Set both pointers to mid',
          ],
          correctAnswer: 1,
          explanation:
            'Save the match and continue searching right by setting left = mid + 1 to find any later occurrences. This ensures you find the rightmost match.',
        },
        {
          id: 'mc3',
          question: 'In the "search insert position" template, what do you return if the target is not found?',
          options: [
            '-1',
            'left (the insertion position)',
            'right',
            'mid',
          ],
          correctAnswer: 1,
          explanation:
            'Return left, which naturally ends up at the correct position where the target should be inserted to maintain sorted order.',
        },
        {
          id: 'mc4',
          question: 'When would you use the "find first" template instead of classic binary search?',
          options: [
            'When the array is unsorted',
            'When duplicates exist and you need the leftmost boundary',
            'When you want faster search',
            'When the array is empty',
          ],
          correctAnswer: 1,
          explanation:
            'Use "find first" when duplicates exist and you need to find the leftmost occurrence, such as for range queries or counting occurrences.',
        },
        {
          id: 'mc5',
          question: 'Which template would you use to count occurrences of a value in a sorted array?',
          options: [
            'Classic binary search',
            'Find first and find last templates',
            'Search insert position',
            'Linear search',
          ],
          correctAnswer: 1,
          explanation:
            'To count occurrences, find the first occurrence and last occurrence, then calculate: last - first + 1. This requires both boundary-finding templates.',
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Walk me through what can go wrong if you use left = mid or right = mid instead of left = mid + 1 or right = mid - 1. Give me a concrete example.',
          hint: 'Imagine you have two elements left: [5, 7], and you are searching for 7.',
          sampleAnswer:
            'Let me give you an example. Say I have [5, 7] and I am searching for 7. My left is 0, right is 1, so mid is 0. nums[mid] is 5, which is less than 7, so I go right. If I set left = mid, then left stays at 0, and next iteration I have the same situation - infinite loop! But if I use left = mid + 1, then left becomes 1, and now left equals right pointing at 7, and I find it. The problem is that mid can equal left or right when you have two elements, so if you do not move past it with +1 or -1, you get stuck.',
          keyPoints: [
            'Using left = mid or right = mid can cause infinite loops',
            'Happens when search space shrinks to 2 elements',
            'Must use mid + 1 or mid - 1 to exclude the checked element',
            'Ensures search space always shrinks',
          ],
        },
        {
          id: 'q2',
          question:
            'Someone hands you an unsorted array and asks you to find an element using binary search. What do you tell them?',
          sampleAnswer:
            'I would explain that binary search fundamentally does not work on unsorted data - it will give you wrong answers or miss elements that are actually there. If they really need to search this array, we have options: we could sort it first (which takes O(n log n)), then do binary search (O(log n)). Or if we only need to search once, just do linear search at O(n), which is simpler. Or if we are going to search many times, build a hash map once at O(n), then get O(1) lookups. The key point is that binary search requires sorted data as a precondition, period.',
          keyPoints: [
            'Binary search does not work on unsorted data',
            'Option 1: Sort first O(n log n), then binary search',
            'Option 2: Use linear search O(n)',
            'Option 3: Hash map for multiple searches O(1) lookup',
          ],
        },
        {
          id: 'q3',
          question:
            'What are the most common bugs you would watch out for when implementing or reviewing binary search code?',
          sampleAnswer:
            'First thing I check is the loop condition - it should be left <= right with the equal sign, not just less than. Second is the mid calculation - make sure it is using left + (right - left) // 2 to avoid overflow, not just (left + right) // 2. Third is the pointer updates - they must be mid + 1 and mid - 1, not just mid. And finally, I make sure the array is actually sorted before using binary search. These are the bugs that bite people over and over again. I also always test edge cases like empty array, single element, and target at boundaries.',
          keyPoints: [
            'Check loop condition (should be <=)',
            'Verify mid calculation (avoid overflow)',
            'Confirm pointer updates (mid + 1, mid - 1)',
            'Ensure array is sorted',
            'Test edge cases',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What is the main problem with using mid = (left + right) // 2 in languages like Java or C++?',
          options: [
            'It is slower',
            'It can cause integer overflow',
            'It gives the wrong result',
            'It uses more memory',
          ],
          correctAnswer: 1,
          explanation:
            'When left and right are both very large, adding them together can exceed the maximum integer value, causing overflow. Use mid = left + (right - left) // 2 instead.',
        },
        {
          id: 'mc2',
          question: 'What can cause an infinite loop in binary search?',
          options: [
            'Using while left <= right',
            'Setting left = mid or right = mid instead of mid ± 1',
            'Calculating mid incorrectly',
            'Having duplicates in the array',
          ],
          correctAnswer: 1,
          explanation:
            'Setting left = mid or right = mid can cause infinite loops when the search space reduces to 2 elements. Always use left = mid + 1 and right = mid - 1 to properly exclude the checked middle element.',
        },
        {
          id: 'mc3',
          question: 'What is the most important precondition for binary search to work correctly?',
          options: [
            'Array must have no duplicates',
            'Array must be sorted',
            'Array must be large',
            'Array must have unique elements',
          ],
          correctAnswer: 1,
          explanation:
            'Binary search absolutely requires the array to be sorted. Without sorting, comparisons with the middle element cannot reliably determine which half contains the target.',
        },
        {
          id: 'mc4',
          question: 'If you must search an unsorted array many times, what is the best approach?',
          options: [
            'Use binary search directly',
            'Build a hash map once for O(1) lookups',
            'Sort before every search',
            'Always use linear search',
          ],
          correctAnswer: 1,
          explanation:
            'For multiple searches on the same data, build a hash map once (O(N)) to get O(1) average lookup time for each subsequent search. Sorting and binary search would be O(N log N) + O(log N) per search.',
        },
        {
          id: 'mc5',
          question: 'What should you return from binary search when the target is found?',
          options: [
            'The value itself',
            'The index where it was found',
            'True',
            'The array',
          ],
          correctAnswer: 1,
          explanation:
            'Return the index (mid) where the target was found, not the value itself. The caller already knows the value (they provided it as the target), they need to know where it is.',
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Explain what a monotonic function is and why it is important for binary search. Give me an example of how you would use binary search on a monotonic function.',
          hint: 'Monotonic means consistently increasing or decreasing.',
          sampleAnswer:
            'A monotonic function is one that is either always increasing or always decreasing - it never changes direction. This is crucial for binary search because it means we can make reliable decisions. If I check a value in the middle and it is too big, I know everything to the right is also too big. For example, say I want to find the square root of 25 without using sqrt. The function f(x) = x squared is monotonic - as x increases, x squared increases. So I can binary search on the range 0 to 25. I pick mid, square it, and if it is too big, I search left. If too small, I search right. The monotonic property guarantees this works. Any problem where you can check a value and know which direction to go can potentially use binary search.',
          keyPoints: [
            'Monotonic: always increasing or always decreasing',
            'Enables reliable decision-making (too big → search left)',
            'Example: x squared is monotonic increasing',
            'Binary search works on any monotonic property',
            'Key insight: can determine direction from any check',
          ],
        },
        {
          id: 'q2',
          question:
            'Describe a scenario where you might binary search on something that does not look sorted at first glance.',
          sampleAnswer:
            'A good example is searching in a rotated sorted array, like [4,5,6,7,0,1,2]. The array is not fully sorted, but it has structure. It is two sorted pieces stuck together. The trick is that at least one half is always properly sorted. When I compare the middle with the edges, I can figure out which half is sorted, then decide if my target is in the sorted half or the other half. So even though it looks messy, there is still enough order to apply binary search logic. Another example is a bitonic array (goes up then down) - not sorted, but still has monotonic regions you can exploit.',
          keyPoints: [
            'Rotated sorted arrays still have structure',
            'At least one half is always sorted',
            'Bitonic arrays (up then down) work too',
            'Key: find the monotonic property',
          ],
        },
        {
          id: 'q3',
          question:
            'How would you recognize in an interview that a problem might be solvable with binary search, even if it does not mention sorted arrays?',
          sampleAnswer:
            'I look for certain keywords and patterns. Words like "minimum", "maximum", "first", "last", or "at least" are red flags. Also if the problem asks about optimizing something or finding a threshold, that is a hint. The key question I ask myself is: if I try a particular value, can I tell whether I need to go higher or lower? That is the hallmark of binary search. Another pattern is when you are searching within a range of possible answers. If I see these signals, I start thinking about whether there is a monotonic property I can exploit with binary search, even if no array is mentioned.',
          keyPoints: [
            'Look for: first, last, minimum, maximum, at least/most',
            'Ask: can I check if a value is too big/small?',
            'Is there a monotonic property?',
            'Searching in a range of possible answers',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What is a monotonic function?',
          options: [
            'A function that is random',
            'A function that is always increasing or always decreasing',
            'A function that has multiple peaks',
            'A function that is constant',
          ],
          correctAnswer: 1,
          explanation:
            'A monotonic function is one that consistently increases or consistently decreases, never changing direction. This property is crucial for binary search because it allows reliable decision-making.',
        },
        {
          id: 'mc2',
          question: 'In a rotated sorted array like [4,5,6,7,0,1,2], what is the key insight for using binary search?',
          options: [
            'Sort it first',
            'At least one half is always properly sorted',
            'Use linear search instead',
            'Find the rotation point first',
          ],
          correctAnswer: 1,
          explanation:
            'Even though the array is rotated, at least one half (left or right of mid) is always properly sorted. You can determine which half is sorted and decide where to search.',
        },
        {
          id: 'mc3',
          question: 'What keywords in a problem statement suggest binary search might apply?',
          options: [
            'Sum, average, total',
            'First, last, minimum, maximum, at least',
            'Count, frequency, duplicate',
            'Random, shuffle, permutation',
          ],
          correctAnswer: 1,
          explanation:
            'Keywords like "first", "last", "minimum", "maximum", or "at least/most" often indicate optimization or threshold problems that can be solved with binary search on a monotonic property.',
        },
        {
          id: 'mc4',
          question: 'How can you use binary search to find square root of 25 without using sqrt()?',
          options: [
            'Binary search from 1 to 25, checking if mid * mid equals, is less than, or greater than 25',
            'Use linear search',
            'Divide 25 by 2 repeatedly',
            'Cannot be done with binary search',
          ],
          correctAnswer: 0,
          explanation:
            'Binary search on the range [0, 25]. The function f(x) = x² is monotonic increasing, so you can check if mid² is too small (search right), too big (search left), or equal (found).',
        },
        {
          id: 'mc5',
          question: 'What is the key question to ask yourself to recognize if binary search applies?',
          options: [
            'Is the array large?',
            'If I check a value, can I tell whether I need to go higher or lower?',
            'Does the array have duplicates?',
            'Is the problem about trees?',
          ],
          correctAnswer: 1,
          explanation:
            'The hallmark of binary search problems is the ability to check a value and determine direction (higher/lower, left/right). This indicates a monotonic property you can exploit.',
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Walk me through how you would approach a binary search problem in an interview, from the moment you read the problem to writing the code.',
          sampleAnswer:
            'First, I would clarify the requirements - is the array sorted? Can there be duplicates? What should I return if not found? Then I would explain my approach: since the array is sorted, I will use binary search for O(log n) time instead of O(n) linear search. I would mention edge cases I will handle - empty array, single element, target at boundaries. Then I would code it carefully, starting with the standard template, clearly naming left, right, mid. After coding, I would walk through test cases: target in middle, at boundaries, not in array, single element. Finally, I would state the complexity: O(log n) time, O(1) space. The key is communicating clearly at every step.',
          keyPoints: [
            'Clarify: sorted? duplicates? return value?',
            'Explain approach and complexity upfront',
            'Code carefully with standard template',
            'Test edge cases',
            'State final complexity',
          ],
        },
        {
          id: 'q2',
          question:
            'An interviewer asks you to search in a rotated sorted array. How would you explain your thought process?',
          sampleAnswer:
            'I would recognize that even though it is rotated, there is still structure I can exploit. The key insight is that one half is always fully sorted. So when I calculate mid, I compare it with left and right to figure out which half is sorted. If the left half is sorted and my target is within that sorted range, I search there. Otherwise, I search the other half. It is still binary search, just with an extra check at each step to determine which half is the good one. The time complexity stays O(log n) because I am still halving the search space each time.',
          keyPoints: [
            'Recognize the structure: one half always sorted',
            'Determine which half is sorted by comparing mid with edges',
            'Check if target is in sorted range',
            'Still O(log n) - search space halves each time',
          ],
        },
        {
          id: 'q3',
          question:
            'What is your strategy for debugging when your binary search is not working correctly?',
          sampleAnswer:
            'My first step is to add print statements for left, mid, and right at each iteration and watch how they move. I verify that the search space is shrinking - if left and right are not getting closer, something is wrong. Then I trace through with a tiny example, like [1, 3, 5] searching for 3. I check my loop condition - is it left <= right? I check my pointer updates - am I using mid + 1 and mid - 1? I also test edge cases: empty array, single element, target at position 0 or at the end. Usually the bug is one of those classic mistakes - wrong loop condition, wrong pointer update, or not handling an edge case.',
          keyPoints: [
            'Print left, mid, right each iteration',
            'Verify search space is shrinking',
            'Trace through small example manually',
            'Check loop condition and pointer updates',
            'Test edge cases',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What is the first question you should ask in a binary search interview problem?',
          options: [
            'What is the array size?',
            'Is the array sorted?',
            'What programming language to use?',
            'Can I use extra space?',
          ],
          correctAnswer: 1,
          explanation:
            'The most critical question is whether the array is sorted, as binary search only works on sorted data. This is a fundamental precondition that must be verified.',
        },
        {
          id: 'mc2',
          question: 'How long should you spend coding a medium binary search problem in an interview?',
          options: [
            '2-3 minutes',
            '5-7 minutes',
            '15-20 minutes',
            '1 minute',
          ],
          correctAnswer: 1,
          explanation:
            'Plan for 5-7 minutes of careful coding. Accuracy is more important than speed. Rush jobs lead to bugs that waste time debugging.',
        },
        {
          id: 'mc3',
          question: 'What is the recommended approach for a rotated sorted array problem?',
          options: [
            'Sort it first',
            'Use linear search',
            'Determine which half is sorted and decide accordingly',
            'Find rotation point first',
          ],
          correctAnswer: 2,
          explanation:
            'In a rotated sorted array, at least one half is always properly sorted. Compare mid with the edges to determine which half is sorted, then decide where to search based on the target.',
        },
        {
          id: 'mc4',
          question: 'What edge cases should you always test for binary search?',
          options: [
            'Only test the middle element',
            'Empty array, single element, boundaries, target not found',
            'Only test when target is found',
            'No need to test edge cases',
          ],
          correctAnswer: 1,
          explanation:
            'Always test: empty array, single element, target at first/last position, target not in array. These edge cases catch most bugs.',
        },
        {
          id: 'mc5',
          question: 'When debugging binary search, what is the first thing to check?',
          options: [
            'The array contents',
            'Print left, mid, right at each iteration to verify search space is shrinking',
            'Run it on larger inputs',
            'Change to linear search',
          ],
          correctAnswer: 1,
          explanation:
            'Print left, mid, right values at each iteration to verify the search space is properly shrinking. If pointers are not converging, you have a logic error.',
        },
      ],
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
