/**
 * Time and Space Complexity module content - Professional & comprehensive guide
 */

import { Module } from '@/lib/types';

export const timeSpaceComplexityModule: Module = {
  id: 'time-space-complexity',
  title: 'Time & Space Complexity',
  description:
    'Master the art of analyzing algorithm efficiency and understanding Big O notation for both time and space.',
  icon: '⏱️',
  sections: [
    {
      id: 'introduction',
      title: 'What is Complexity Analysis?',
      content: `Complexity analysis is the process of determining how the runtime and memory usage of an algorithm scale as the input size grows. It's one of the most fundamental skills in computer science and is crucial for writing efficient code.

**Why Does It Matter?**
- **Scalability:** An algorithm that works for 100 items might fail for 1 million items
- **Resource Management:** Understanding memory usage prevents crashes and optimizes performance
- **Interview Success:** Virtually every technical interview asks about complexity
- **Design Decisions:** Helps you choose the right algorithm and data structure

**The Core Question:**
As the input size \`n\` grows, how does the number of operations (time) or the amount of memory (space) change?

**Real-World Analogy:**
Imagine you need to find a friend's phone number:
- **O(1):** You have them on speed dial - instant access
- **O(log n):** You binary search through a sorted phonebook - very fast
- **O(n):** You scan through your contacts list one by one - grows linearly
- **O(n²):** You check every contact against every other contact - gets slow fast!

**Key Principles:**
- We care about **growth rate**, not exact numbers
- We focus on the **worst case** unless stated otherwise
- We ignore **constants and lower-order terms** (O(2n + 5) → O(n))
- We measure both **time complexity** (operations) and **space complexity** (memory)`,
      quiz: [
        {
          id: 'q1',
          question:
            'Explain in your own words what we mean when we say an algorithm is O(n). What does the "n" represent and why do we care about it?',
          hint: 'Think about what happens when you double the input size.',
          sampleAnswer:
            'When we say an algorithm is O(n), we mean that the number of operations grows linearly with the input size. The "n" represents the size of the input - like the number of elements in an array. If I have 10 elements and it takes 10 operations, then with 100 elements it will take roughly 100 operations, and with 1000 elements it will take roughly 1000 operations. We care about this because it tells us how the algorithm will perform as our data grows. An O(n) algorithm scales much better than O(n²), especially with large datasets. It helps us predict performance and make smart choices about which algorithm to use.',
          keyPoints: [
            'O(n) means operations grow linearly with input size',
            'n represents the input size (e.g., array length)',
            'Doubling input size roughly doubles the work',
            'Helps predict scalability and make design decisions',
          ],
        },
        {
          id: 'q2',
          question:
            'Why do we drop constants in Big O notation? For example, why is O(2n) just written as O(n)?',
          sampleAnswer:
            'We drop constants because Big O is about understanding growth rates, not exact measurements. Whether an algorithm does n operations or 2n operations, both grow linearly - they both scale at the same rate. When n is 1 million, the difference between n and 2n is just a constant factor of 2, but both are vastly better than n² which would be 1 trillion operations. Constants like 2 or 100 can be affected by hardware, programming language, and implementation details. Big O abstracts away these details to focus on the fundamental scaling behavior. So O(2n), O(5n), and O(n + 1000) are all just O(n) because they all grow linearly.',
          keyPoints: [
            'Big O focuses on growth rate, not exact operation counts',
            'Constants become less significant as n grows large',
            'Both O(n) and O(2n) scale linearly',
            'We care about order of magnitude differences',
          ],
        },
        {
          id: 'q3',
          question:
            'What is the difference between time complexity and space complexity? Can an algorithm have different complexities for each?',
          hint: 'Think about what resources an algorithm uses.',
          sampleAnswer:
            'Time complexity measures how many operations an algorithm performs as input size grows, while space complexity measures how much memory it uses. They are independent - you can have an algorithm that is fast but uses a lot of memory, or one that is slow but memory-efficient. For example, recursive algorithms often trade space for elegant code - they might be O(n) time but also O(n) space due to the call stack. Or you might use memoization to speed up an algorithm from O(2^n) to O(n) time, but it costs you O(n) extra space to store the cache. Understanding both helps you make informed tradeoffs based on your constraints.',
          keyPoints: [
            'Time complexity: number of operations',
            'Space complexity: amount of memory used',
            'They are independent - can differ for same algorithm',
            'Often there is a time-space tradeoff',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What does Big O notation describe?',
          options: [
            'The exact number of operations an algorithm performs',
            "How an algorithm's resource requirements grow with input size",
            'The amount of memory an algorithm uses',
            'The programming language used to implement an algorithm',
          ],
          correctAnswer: 1,
          explanation:
            "Big O notation describes how an algorithm's time or space requirements grow as the input size increases. It focuses on growth rate, not exact counts or specific implementations.",
        },
        {
          id: 'mc2',
          question: 'Which complexity is best (fastest) for large inputs?',
          options: ['O(n²)', 'O(n log n)', 'O(n)', 'O(log n)'],
          correctAnswer: 3,
          explanation:
            'O(log n) is the fastest among these options. Logarithmic time complexity grows very slowly as input size increases, making it ideal for large datasets.',
        },
        {
          id: 'mc3',
          question: 'Why do we drop constants in Big O notation?',
          options: [
            'To make the math easier',
            'Because constants are always small',
            'To focus on the growth rate as input size approaches infinity',
            'Because different computers run at different speeds',
          ],
          correctAnswer: 2,
          explanation:
            'We drop constants because Big O notation focuses on how the algorithm scales with input size. As n approaches infinity, constant factors become less significant compared to the growth rate itself.',
        },
        {
          id: 'mc4',
          question:
            'If an algorithm has time complexity O(n) and space complexity O(1), what does this mean?',
          options: [
            'It takes constant time and linear space',
            'It takes linear time and constant space',
            'It takes constant time and constant space',
            'It takes linear time and linear space',
          ],
          correctAnswer: 1,
          explanation:
            "O(n) time means the algorithm's execution time grows linearly with input size. O(1) space means it uses a constant amount of memory regardless of input size.",
        },
        {
          id: 'mc5',
          question:
            'What is the time complexity of accessing an element in an array by index (e.g., arr[5])?',
          options: ['O(1)', 'O(log n)', 'O(n)', 'O(n²)'],
          correctAnswer: 0,
          explanation:
            'Array access by index is O(1) - constant time. The computer can calculate the memory address directly using the base address plus the index, regardless of array size.',
        },
      ],
    },
    {
      id: 'big-o-notation',
      title: 'Understanding Big O Notation',
      content: `**Big O Hierarchy (Best to Worst):**

From fastest growing (best) to slowest growing (worst):

1. **O(1) - Constant Time**
   - Operations: Same regardless of input size
   - Examples: Array access, hash table lookup, simple math
   - \`arr[5]\`, \`hash_map.get(key)\`

2. **O(log n) - Logarithmic Time**
   - Operations: Grows logarithmically
   - Examples: Binary search, balanced tree operations
   - Doubling input adds just one more operation!

3. **O(n) - Linear Time**
   - Operations: Directly proportional to input size
   - Examples: Linear search, iterating through array
   - Most simple loops over data

4. **O(n log n) - Log-Linear Time**
   - Operations: n times log n
   - Examples: Efficient sorting (merge sort, quicksort, heap sort)
   - Sweet spot for sorting algorithms

5. **O(n²) - Quadratic Time**
   - Operations: Square of input size
   - Examples: Nested loops, bubble sort, naive algorithms
   - 1000 items = 1,000,000 operations!

6. **O(n³) - Cubic Time**
   - Operations: Cube of input size
   - Examples: Triple nested loops
   - Gets very slow very fast

7. **O(2ⁿ) - Exponential Time**
   - Operations: Doubles with each additional input
   - Examples: Recursive Fibonacci (naive), generating all subsets
   - Unusable for n > 30 or so

8. **O(n!) - Factorial Time**
   - Operations: n × (n-1) × (n-2) × ... × 1
   - Examples: Generating all permutations
   - Unusable for n > 10

**Practical Comparison:**

| n   | O(1) | O(log n) | O(n) | O(n log n) | O(n²) | O(2ⁿ) |
|-----|------|----------|------|------------|-------|-------|
| 10  | 1    | 3        | 10   | 33         | 100   | 1024  |
| 100 | 1    | 7        | 100  | 664        | 10K   | 10³⁰  |
| 1000| 1    | 10       | 1K   | 10K        | 1M    | ∞     |

**Code Examples:**

\`\`\`python
# O(1) - Constant
def get_first(arr):
    return arr[0]  # Always one operation

# O(n) - Linear
def sum_array(arr):
    total = 0
    for num in arr:  # n iterations
        total += num
    return total

# O(n²) - Quadratic
def has_duplicate(arr):
    for i in range(len(arr)):      # n iterations
        for j in range(len(arr)):  # n iterations each
            if i != j and arr[i] == arr[j]:
                return True
    return False

# O(log n) - Logarithmic
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:  # Halves each time
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
\`\`\``,
      quiz: [
        {
          id: 'q1',
          question:
            'Look at this code: for i in range(n): for j in range(n): print(i, j). What is the time complexity and why?',
          hint: 'Count how many times print() gets called.',
          sampleAnswer:
            'This is O(n²) - quadratic time. The outer loop runs n times, and for each iteration of the outer loop, the inner loop also runs n times. So the print statement executes n × n = n² times total. If n is 10, we print 100 times. If n is 100, we print 10,000 times. The number of operations grows with the square of the input. This is the signature pattern of nested loops where both iterate n times. Nested loops are a red flag for quadratic complexity.',
          keyPoints: [
            'Nested loops over same input size → O(n²)',
            'Outer loop: n times, inner loop: n times each',
            'Total operations: n × n = n²',
            'Quadratic growth - gets slow with large inputs',
          ],
        },
        {
          id: 'q2',
          question:
            'Why is binary search O(log n) and not O(n)? Walk me through the mathematical reasoning.',
          sampleAnswer:
            'Binary search is O(log n) because with each comparison, we eliminate half the remaining elements. If we start with 1000 elements, after one check we have 500 left, then 250, then 125, then 63, and so on. The question is: how many times can we divide n by 2 until we get down to 1? That is exactly what log₂(n) tells us. For 1000 elements, log₂(1000) is about 10, so we need at most 10 comparisons. For 1 million elements, we only need about 20 comparisons. This logarithmic growth is why binary search is so incredibly efficient compared to linear search which would need to check every element.',
          keyPoints: [
            'Each step eliminates half the search space',
            'Pattern: n → n/2 → n/4 → ... → 1',
            'Number of halvings = log₂(n)',
            'For 1 million items, only ~20 operations needed',
          ],
        },
        {
          id: 'q3',
          question:
            'What complexity would you use to describe checking if two strings are anagrams? Walk through your reasoning.',
          sampleAnswer:
            'The best approach is O(n) where n is the length of the strings. You can count the frequency of each character in both strings using hash maps, which takes O(n) time for each string. Then compare the frequency maps, which takes O(1) for each of the 26 letters (constant). So overall it is O(n) + O(n) + O(26) = O(2n + 26) which simplifies to O(n). You could also sort both strings and compare them, but that would be O(n log n) due to sorting. The hash map approach is better because it is linear. The key insight is that the work scales linearly with string length.',
          keyPoints: [
            'Count character frequencies in both strings',
            'Each character count pass: O(n)',
            'Comparing frequencies: O(1) for fixed alphabet',
            'Total: O(n) - linear in string length',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question:
            'What is the time complexity of this code?\n\n```python\nfor i in range(n):\n    for j in range(n):\n        print(i, j)\n```',
          options: ['O(n)', 'O(n²)', 'O(2n)', 'O(n log n)'],
          correctAnswer: 1,
          explanation:
            'This is O(n²) because we have nested loops where both iterate n times. The outer loop runs n times, and for each iteration, the inner loop runs n times, resulting in n × n = n² total operations.',
        },
        {
          id: 'mc2',
          question:
            'Which of these has the best (fastest) time complexity for large inputs?',
          options: ['O(2ⁿ)', 'O(n²)', 'O(n log n)', 'O(n³)'],
          correctAnswer: 2,
          explanation:
            'O(n log n) is the fastest among these options. The hierarchy from best to worst is: O(n log n) < O(n²) < O(n³) < O(2ⁿ). Exponential time O(2ⁿ) grows extremely fast and is the slowest.',
        },
        {
          id: 'mc3',
          question:
            'What is the time complexity of binary search on a sorted array?',
          options: ['O(1)', 'O(log n)', 'O(n)', 'O(n log n)'],
          correctAnswer: 1,
          explanation:
            'Binary search is O(log n) because it halves the search space with each comparison. After k comparisons, we have n/2^k elements left. When n/2^k = 1, we have k = log₂(n) comparisons.',
        },
        {
          id: 'mc4',
          question:
            'If we simplify O(3n² + 2n + 5) using Big O rules, what do we get?',
          options: ['O(3n²)', 'O(n² + n)', 'O(n²)', 'O(n)'],
          correctAnswer: 2,
          explanation:
            'We drop constants (3, 2, 5) and lower-order terms (2n, 5), keeping only the fastest-growing term. So O(3n² + 2n + 5) simplifies to O(n²).',
        },
        {
          id: 'mc5',
          question:
            'What is the time complexity of this code?\n\n```python\nfor i in range(n):\n    print(i)\nfor j in range(n):\n    print(j)\n```',
          options: ['O(n)', 'O(n²)', 'O(2n)', 'O(log n)'],
          correctAnswer: 0,
          explanation:
            'This is O(n). We have two sequential loops (not nested), each running n times. Total operations: n + n = 2n, which simplifies to O(n) after dropping the constant.',
        },
      ],
    },
    {
      id: 'space-complexity',
      title: 'Space Complexity Analysis',
      content: `**What Counts as Space?**

Space complexity measures the total memory used by an algorithm, including:

1. **Input Space:** Memory for input data (usually not counted in analysis)
2. **Auxiliary Space:** Extra memory used by the algorithm
   - Variables and data structures created
   - Recursive call stack
   - Temporary arrays, hash maps, etc.

When we say "space complexity," we typically mean **auxiliary space** - the extra memory beyond the input.

**Common Space Complexities:**

**O(1) - Constant Space:**
- Fixed number of variables regardless of input size
- No dynamic data structures
- Iterative solutions with just pointers/counters

\`\`\`python
# O(1) space - just a few variables
def sum_array(arr):
    total = 0  # One variable
    for num in arr:
        total += num
    return total
\`\`\`

**O(n) - Linear Space:**
- Creating new array/list of size n
- Hash map with n entries
- Recursive call stack of depth n

\`\`\`python
# O(n) space - creating new array
def double_array(arr):
    result = []  # New array of size n
    for num in arr:
        result.append(num * 2)
    return result

# O(n) space - recursive call stack
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)  # n recursive calls
\`\`\`

**O(n²) - Quadratic Space:**
- 2D matrix of size n×n
- Creating all pairs

\`\`\`python
# O(n²) space - 2D matrix
def create_matrix(n):
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    return matrix
\`\`\`

**The Recursive Call Stack:**

Recursive functions use stack space! Each recursive call adds a frame to the call stack.

\`\`\`python
# O(n) space due to call stack
def recursive_sum(arr, index=0):
    if index == len(arr):
        return 0
    return arr[index] + recursive_sum(arr, index + 1)
\`\`\`

**Time-Space Tradeoffs:**

Often you can trade space for time or vice versa:

**Example: Fibonacci**

\`\`\`python
# Naive: O(2ⁿ) time, O(n) space
def fib_naive(n):
    if n <= 1:
        return n
    return fib_naive(n-1) + fib_naive(n-2)

# Memoization: O(n) time, O(n) space
def fib_memo(n, cache={}):
    if n in cache:
        return cache[n]
    if n <= 1:
        return n
    cache[n] = fib_memo(n-1, cache) + fib_memo(n-2, cache)
    return cache[n]

# Iterative: O(n) time, O(1) space
def fib_iterative(n):
    if n <= 1:
        return n
    prev, curr = 0, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr
    return curr
\`\`\`

**Best Practices:**
- Prefer iterative over recursive when space is a concern
- Reuse data structures instead of creating new ones
- Consider in-place algorithms when modifying arrays
- Use generators/streams for large datasets`,
      quiz: [
        {
          id: 'q1',
          question:
            'Explain the difference between auxiliary space and total space. Which one do we typically measure when analyzing space complexity?',
          sampleAnswer:
            'Total space includes everything - the input data plus any extra memory the algorithm uses. Auxiliary space is just the extra memory beyond the input. When we analyze space complexity, we typically measure auxiliary space because we want to know how much additional memory the algorithm needs. For example, if I have an array of n elements and I create a few variables to track sums and indices, my auxiliary space is O(1) even though the total space including the input is O(n). We care about auxiliary space because it tells us about the memory overhead of our algorithm, not just the memory needed to store the data we were given.',
          keyPoints: [
            'Total space = input + extra memory',
            'Auxiliary space = extra memory only',
            'We typically measure auxiliary space',
            'It shows the overhead of the algorithm',
          ],
        },
        {
          id: 'q2',
          question:
            'Walk me through why a recursive function has space complexity related to the depth of recursion. Use factorial as an example.',
          hint: 'Think about the call stack.',
          sampleAnswer:
            'When you call factorial(5), it calls factorial(4), which calls factorial(3), and so on down to factorial(1). Each of these function calls gets added to the call stack and stays there until it can return. So at the deepest point, you have 5 stack frames sitting in memory: factorial(5) waiting for factorial(4), factorial(4) waiting for factorial(3), all the way down. This is O(n) space where n is the input number. The depth of the recursion determines how many stack frames accumulate. This is different from iterative solutions which can use the same small amount of memory repeatedly. The call stack is often overlooked but is crucial for space complexity analysis of recursive algorithms.',
          keyPoints: [
            'Each recursive call adds a frame to the call stack',
            'Stack frames accumulate until base case is reached',
            'Maximum stack depth = space complexity',
            'factorial(n) has n frames → O(n) space',
          ],
        },
        {
          id: 'q3',
          question:
            'What is a time-space tradeoff? Give me an example where you would intentionally use more space to save time.',
          sampleAnswer:
            'A time-space tradeoff is when you sacrifice memory to gain speed, or vice versa. A classic example is memoization - storing previously computed results to avoid recalculating them. In Fibonacci, the naive recursive solution is O(2^n) time because it recalculates the same values over and over. By using a hash map to cache results, we can bring it down to O(n) time, but now we use O(n) extra space for the cache. We are trading space for a massive speed improvement. This tradeoff makes sense when speed is more important than memory, which is often the case. Another example is creating an index on a database - it uses more disk space but makes queries much faster.',
          keyPoints: [
            'Trade memory for speed (or vice versa)',
            'Memoization: use O(n) space to improve time dramatically',
            'Fibonacci: O(2^n) → O(n) time by using O(n) space',
            'Common when speed matters more than memory',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question:
            'What is the space complexity of this function?\n\n```python\ndef sum_array(arr):\n    total = 0\n    for num in arr:\n        total += num\n    return total\n```',
          options: ['O(n)', 'O(1)', 'O(log n)', 'O(n²)'],
          correctAnswer: 1,
          explanation:
            'This is O(1) space. We only use a constant amount of extra space (the variable "total") regardless of the input array size. The input array itself is not counted in auxiliary space complexity.',
        },
        {
          id: 'mc2',
          question:
            'What is the space complexity of a recursive function that has a maximum call stack depth of n?',
          options: ['O(1)', 'O(log n)', 'O(n)', 'O(n²)'],
          correctAnswer: 2,
          explanation:
            'The space complexity is O(n). Each recursive call adds a frame to the call stack, and if the maximum depth is n, we need O(n) space to store all those stack frames in memory.',
        },
        {
          id: 'mc3',
          question: 'Which statement about space complexity is TRUE?',
          options: [
            'Space complexity always equals time complexity',
            'Recursive functions always use O(1) space',
            'Creating a hash map with n entries uses O(n) space',
            'Space complexity is less important than time complexity',
          ],
          correctAnswer: 2,
          explanation:
            'Creating a hash map with n entries uses O(n) space. Space and time complexity are independent, recursive functions typically use O(depth) stack space, and space complexity is often just as important as time complexity.',
        },
        {
          id: 'mc4',
          question:
            'What is the typical space complexity of merge sort (not in-place)?',
          options: ['O(1)', 'O(log n)', 'O(n)', 'O(n log n)'],
          correctAnswer: 2,
          explanation:
            'Merge sort uses O(n) space for the temporary arrays needed during the merge step. While the recursion depth is O(log n), the dominant space factor is the O(n) auxiliary arrays.',
        },
        {
          id: 'mc5',
          question:
            'In a time-space tradeoff, what does it mean to "trade space for time"?',
          options: [
            'Use less memory to run faster',
            'Use more memory to run faster',
            'Use less memory but run slower',
            'Keep both time and space the same',
          ],
          correctAnswer: 1,
          explanation:
            'Trading space for time means using more memory (extra data structures like caches, hash maps, etc.) to achieve faster execution. A classic example is memoization, where we store computed results to avoid recalculating them.',
        },
      ],
    },
    {
      id: 'analyzing-code',
      title: 'How to Analyze Code Complexity',
      content: `**Step-by-Step Process:**

**1. Identify the Input Size**
- What is "n"? Array length? String length? Number of nodes?
- Sometimes multiple variables (n and m)

**2. Count Operations for Time Complexity**

**Simple Rules:**
- **Single loop:** O(n)
- **Nested loops (same size):** O(n²), O(n³), etc.
- **Halving each iteration:** O(log n)
- **Recursive calls:** Analyze recursion tree

**3. Count Memory for Space Complexity**

**What to Count:**
- New arrays, lists, sets, maps created
- Recursive call stack depth
- Variables (usually O(1))

**Common Patterns:**

**Pattern 1: Single Loop**
\`\`\`python
def find_max(arr):
    max_val = arr[0]     # O(1)
    for num in arr:      # O(n)
        if num > max_val:
            max_val = num
    return max_val
# Time: O(n), Space: O(1)
\`\`\`

**Pattern 2: Nested Loops (Different Ranges)**
\`\`\`python
def print_pairs(arr1, arr2):
    for x in arr1:       # n times
        for y in arr2:   # m times
            print(x, y)
# Time: O(n × m), Space: O(1)
\`\`\`

**Pattern 3: Divide and Conquer**
\`\`\`python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:  # Halves each time
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
# Time: O(log n), Space: O(1)
\`\`\`

**Pattern 4: Building New Data Structure**
\`\`\`python
def find_duplicates(arr):
    seen = set()         # O(n) space
    duplicates = []
    for num in arr:      # O(n) time
        if num in seen:
            duplicates.append(num)
        seen.add(num)
    return duplicates
# Time: O(n), Space: O(n)
\`\`\`

**Pattern 5: Recursive with Branching**
\`\`\`python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
# Time: O(2ⁿ) - two branches per call
# Space: O(n) - max call stack depth
\`\`\`

**Analyzing Recursive Algorithms:**

Use the **Recursion Tree Method:**

1. Draw the recursion tree
2. Count nodes (calls) at each level
3. Sum across all levels

**Example: Merge Sort**
\`\`\`python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)
\`\`\`

- **Depth of tree:** log n (halving each time)
- **Work per level:** O(n) (merge operation)
- **Total time:** O(n log n)
- **Space:** O(n) for temporary arrays

**Common Mistakes to Avoid:**

- ❌ **Forgetting about sorting:** \`sorted(arr)\` is O(n log n)
- ❌ **Ignoring built-in operations:** \`list.append()\` is O(1), \`list.insert(0, x)\` is O(n)
- ❌ **Overlooking recursive call stack space**
- ❌ **Confusing amortized with worst-case complexity**
- ❌ **Not considering all nested structures**`,
      quiz: [
        {
          id: 'q1',
          question:
            'Walk me through analyzing this code: def func(arr): for i in range(len(arr)): for j in range(i): print(arr[i], arr[j]). What is the time complexity?',
          hint: 'The inner loop does not run n times for each outer iteration.',
          sampleAnswer:
            'This is still O(n²), but let me explain why. The outer loop runs n times. The inner loop runs a variable number of times - it runs 0 times when i=0, 1 time when i=1, 2 times when i=2, and so on up to n-1 times. So the total number of prints is 0 + 1 + 2 + ... + (n-1), which is the sum of first n-1 numbers. That equals (n-1) × n / 2, which is roughly n² / 2. When we drop constants, n² / 2 becomes O(n²). Even though this does fewer operations than a standard nested loop, it is still quadratic growth. The key insight is that the sum of 1 to n is proportional to n².',
          keyPoints: [
            'Outer loop: n iterations',
            'Inner loop: 0, 1, 2, ..., n-1 iterations',
            'Total operations: 0 + 1 + 2 + ... + (n-1) = n(n-1)/2',
            'Simplifies to O(n²) - still quadratic',
          ],
        },
        {
          id: 'q2',
          question:
            'How would you analyze the space complexity of merge sort? What data structures contribute to it?',
          sampleAnswer:
            'Merge sort has O(n) space complexity. There are two main contributors. First, in the merge step, we create temporary arrays to hold the left and right halves, and these combined are O(n) space. Second, there is the recursive call stack - merge sort divides the array in half each time, so the maximum depth of recursion is log n, which is O(log n) space. Overall, the dominant factor is the O(n) space for the temporary merge arrays. In some implementations, you can reuse the same temporary array, but you still need O(n) auxiliary space. This is different from in-place sorts like heap sort which are O(1) space.',
          keyPoints: [
            'Temporary merge arrays: O(n) space',
            'Recursive call stack: O(log n) depth',
            'Dominant factor: O(n) auxiliary space',
            'Not an in-place sort',
          ],
        },
        {
          id: 'q3',
          question:
            'If you see sorting inside a loop, how does that affect the overall time complexity? Give me an example.',
          hint: 'Sorting is not O(n).',
          sampleAnswer:
            'If you sort inside a loop, you multiply the complexities. For example, if you have a loop that runs n times, and inside that loop you sort an array of size n, the overall complexity is O(n × n log n) = O(n² log n). This is worse than just O(n²). A concrete example would be: for each element in an array, create a sub-array, sort it, and do something with it. The outer loop is O(n), and sorting each sub-array is O(n log n), so total is O(n² log n). This is a common mistake - people forget that sorting is expensive and should be done carefully, not repeatedly inside loops if it can be avoided.',
          keyPoints: [
            'Multiply complexities: loop × sorting',
            'Loop of n × sorting of n → O(n × n log n) = O(n² log n)',
            'Sorting is O(n log n), not O(n)',
            'Avoid sorting inside loops when possible',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question:
            'What is the time complexity of this code?\n\n```python\ndef func(arr):\n    for i in range(len(arr)):\n        for j in range(i):\n            print(arr[i], arr[j])\n```',
          options: ['O(n)', 'O(n log n)', 'O(n²)', 'O(2n)'],
          correctAnswer: 2,
          explanation:
            'This is O(n²). The inner loop runs 0 + 1 + 2 + ... + (n-1) times total, which equals n(n-1)/2. This simplifies to O(n²) after dropping constants and lower-order terms.',
        },
        {
          id: 'mc2',
          question:
            'What is the time complexity of calling sorted() on a Python list?',
          options: ['O(n)', 'O(n log n)', 'O(n²)', 'O(log n)'],
          correctAnswer: 1,
          explanation:
            "Python's sorted() function uses Timsort, which has O(n log n) time complexity in the average and worst case. This is an important built-in operation to remember.",
        },
        {
          id: 'mc3',
          question:
            'If you have a loop that runs n times and inside it you sort an array of size n, what is the total time complexity?',
          options: ['O(n)', 'O(n log n)', 'O(n²)', 'O(n² log n)'],
          correctAnswer: 3,
          explanation:
            'The loop runs n times, and each iteration sorts an array of size n (O(n log n)). We multiply these: O(n) × O(n log n) = O(n² log n). This is worse than O(n²)!',
        },
        {
          id: 'mc4',
          question:
            'What is the time complexity of this recursive Fibonacci?\n\n```python\ndef fib(n):\n    if n <= 1: return n\n    return fib(n-1) + fib(n-2)\n```',
          options: ['O(n)', 'O(n²)', 'O(2ⁿ)', 'O(log n)'],
          correctAnswer: 2,
          explanation:
            'This is O(2ⁿ) exponential time. Each call makes two recursive calls, creating a binary tree of depth n. The total number of calls is approximately 2ⁿ.',
        },
        {
          id: 'mc5',
          question:
            'Which data structure provides O(1) average case lookup time?',
          options: ['Array', 'Linked List', 'Hash Table', 'Binary Search Tree'],
          correctAnswer: 2,
          explanation:
            'Hash tables (dictionaries/hash maps) provide O(1) average case lookup time. Arrays provide O(1) access by index, but O(n) search. Linked lists are O(n) for lookup. BSTs are O(log n) average case.',
        },
      ],
    },
    {
      id: 'best-average-worst',
      title: 'Best, Average, and Worst Case Analysis',
      content: `**Three Types of Analysis:**

**1. Best Case (Ω - Omega Notation)**
- The minimum time/space required
- Occurs under optimal conditions
- **Example:** Quick sort with perfect pivots → O(n log n)

**2. Average Case (Θ - Theta Notation)**
- Expected performance over all possible inputs
- More realistic than best/worst
- **Example:** Quick sort on random data → O(n log n)

**3. Worst Case (O - Big O Notation)**
- Maximum time/space required
- Occurs under pessimal conditions
- **Example:** Quick sort with worst pivots → O(n²)

**Why We Focus on Worst Case:**

Most commonly, we use **Big O** (worst case) because:
- **Safety:** Guarantees performance won't be worse
- **Simplicity:** Easier to analyze and communicate
- **Risk Management:** Better to over-estimate than under-estimate

**Examples:**

**Linear Search:**
\`\`\`python
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1
\`\`\`
- **Best Case:** O(1) - target is first element
- **Average Case:** O(n/2) = O(n) - target in middle on average
- **Worst Case:** O(n) - target is last or not present

**Quick Sort:**
\`\`\`python
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
\`\`\`
- **Best Case:** O(n log n) - pivot always splits evenly
- **Average Case:** O(n log n) - pivot reasonably balanced
- **Worst Case:** O(n²) - pivot is always min or max (sorted array!)

**Insertion Sort:**
- **Best Case:** O(n) - array already sorted
- **Average Case:** O(n²) - random data
- **Worst Case:** O(n²) - array reverse sorted

**Hash Table Operations:**
- **Best/Average Case:** O(1) - good hash function, few collisions
- **Worst Case:** O(n) - all keys collide, degenerates to linked list

**Amortized Analysis:**

Different from average case! Amortized analysis considers **sequences of operations**, not single operations.

**Example: Dynamic Array Append**

\`\`\`python
arr = []
for i in range(n):
    arr.append(i)  # Each append seems O(1)...
\`\`\`

- Most appends are O(1)
- Occasionally need to resize: copy all n elements → O(n)
- **Amortized:** O(1) per operation over the sequence
- Total for n appends: O(n), so average per append is O(n)/n = O(1)

**Interview Tip:**

Always state which case you're analyzing:
- "In the worst case, this runs in O(n²) time"
- "On average, we expect O(n log n)"
- "Best case is O(1) if we find it immediately"

**Practical Implications:**

- **Quick Sort:** Despite O(n²) worst case, it's often faster than Merge Sort (O(n log n) worst case) in practice due to better constants and cache performance
- **Hash Tables:** Despite O(n) worst case, they're widely used because average case is O(1)
- **Insertion Sort:** Despite O(n²) worst case, it's fast for small or nearly-sorted arrays`,
      quiz: [
        {
          id: 'q1',
          question:
            'Explain the difference between worst case and average case complexity. When would you use each in your analysis?',
          sampleAnswer:
            'Worst case is the maximum time or space an algorithm could take under the most pessimal conditions, while average case is what you would expect on typical inputs. Worst case gives you a guarantee - you know performance will never be worse than this. Average case is more realistic but harder to analyze because you need to consider the probability distribution of inputs. In interviews and production code, we typically focus on worst case because it is safer - you want to know your algorithm will not blow up even on adversarial input. But average case is useful for understanding real-world performance, like knowing that hash table lookups are O(1) on average even though they are O(n) worst case.',
          keyPoints: [
            'Worst case: maximum time under pessimal conditions',
            'Average case: expected time on typical inputs',
            'Worst case provides guarantees',
            'Average case more realistic but harder to analyze',
          ],
        },
        {
          id: 'q2',
          question:
            'Why is quicksort O(n²) in the worst case, and when does that worst case actually happen?',
          hint: 'Think about what makes a bad pivot choice.',
          sampleAnswer:
            'Quicksort is O(n²) worst case when the pivot choices are consistently terrible - specifically when the pivot is always the smallest or largest element. This happens when you try to sort an already sorted array using the first or last element as the pivot. In this case, each partition step only removes one element, so you get n levels of recursion instead of log n, and each level does O(n) work. That is n × n = O(n²). You can avoid this by using random pivots or the median-of-three method. This is why quicksort performs poorly on already sorted data unless you add randomization or smarter pivot selection.',
          keyPoints: [
            'Worst case when pivot is always min or max',
            'Happens on sorted/reverse-sorted arrays',
            'Gets n levels of recursion instead of log n',
            'Total: n levels × O(n) work per level = O(n²)',
          ],
        },
        {
          id: 'q3',
          question:
            'What is amortized analysis and how is it different from average case analysis? Use dynamic array appending as an example.',
          sampleAnswer:
            'Amortized analysis looks at the average cost per operation over a sequence of operations, while average case looks at expected cost for a single operation over all possible inputs. For dynamic arrays, when you append an element, most of the time it is O(1) - just add to the end. But occasionally the array is full and you need to resize, which means allocating new memory and copying all n elements - that single operation is O(n). However, this expensive resize happens rarely - roughly every time you double the size. So if you do n appends, you pay O(n) total for resizes spread across n operations, giving O(1) amortized cost per append. It is not average case because we are analyzing a sequence, not averaging over random inputs.',
          keyPoints: [
            'Amortized: average cost over sequence of operations',
            'Average case: expected cost for single operation',
            'Dynamic array: occasional O(n) resize spread over many O(1) appends',
            'Total cost O(n) for n appends → O(1) amortized per append',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What does Big O notation typically represent?',
          options: [
            'Best case complexity',
            'Average case complexity',
            'Worst case complexity',
            'All three cases equally',
          ],
          correctAnswer: 2,
          explanation:
            "By convention, Big O notation represents worst case complexity unless stated otherwise. This provides a guarantee that performance won't be worse than the stated complexity.",
        },
        {
          id: 'mc2',
          question:
            'For linear search, which statement is correct about its complexities?',
          options: [
            'Best: O(1), Worst: O(n)',
            'Best: O(n), Worst: O(n²)',
            'Best: O(log n), Worst: O(n)',
            'Best: O(1), Worst: O(log n)',
          ],
          correctAnswer: 0,
          explanation:
            'Linear search has best case O(1) when the target is the first element, and worst case O(n) when the target is last or not present. We must check every element in the worst case.',
        },
        {
          id: 'mc3',
          question: 'Why is quicksort O(n²) in the worst case?',
          options: [
            'When the array is randomly shuffled',
            'When the pivot always splits the array evenly',
            'When the pivot is always the minimum or maximum element',
            'When using the median-of-three pivot selection',
          ],
          correctAnswer: 2,
          explanation:
            'Quicksort degrades to O(n²) when the pivot is consistently the smallest or largest element, causing unbalanced partitions. This happens with already sorted arrays using first/last element as pivot.',
        },
        {
          id: 'mc4',
          question: 'What is amortized analysis?',
          options: [
            'Analyzing the average case over random inputs',
            'Analyzing the worst case only',
            'Analyzing the average cost per operation over a sequence of operations',
            'Analyzing the best case scenario',
          ],
          correctAnswer: 2,
          explanation:
            'Amortized analysis considers the average cost per operation over a sequence of operations, not individual operations. Example: dynamic array append is O(1) amortized despite occasional O(n) resizing.',
        },
        {
          id: 'mc5',
          question:
            'Hash table lookups are O(1) average case but O(n) worst case. Why do we still use them?',
          options: [
            'The worst case never happens in practice',
            'O(n) is fast enough for any application',
            'With good hash functions, average case O(1) is typical and valuable',
            'They use less memory than other data structures',
          ],
          correctAnswer: 2,
          explanation:
            'We use hash tables because with good hash functions and proper load factors, the average case O(1) performance is what we typically experience. The worst case is rare in well-implemented hash tables.',
        },
      ],
    },
    {
      id: 'optimization',
      title: 'Optimization Strategies & Trade-offs',
      content: `**Common Optimization Techniques:**

**1. Use Better Data Structures**

❌ **Slow:**
\`\`\`python
def has_duplicate(arr):  # O(n²)
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            if arr[i] == arr[j]:
                return True
    return False
\`\`\`

✅ **Fast:**
\`\`\`python
def has_duplicate(arr):  # O(n)
    seen = set()
    for num in arr:
        if num in seen:
            return True
        seen.add(num)
    return False
\`\`\`

**2. Avoid Redundant Work**

❌ **Slow:**
\`\`\`python
def fibonacci(n):  # O(2ⁿ)
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
\`\`\`

✅ **Fast with Memoization:**
\`\`\`python
def fibonacci(n, cache={}):  # O(n)
    if n in cache:
        return cache[n]
    if n <= 1:
        return n
    cache[n] = fibonacci(n-1, cache) + fibonacci(n-2, cache)
    return cache[n]
\`\`\`

**3. Precompute and Cache**

❌ **Slow:**
\`\`\`python
def query_sum(arr, queries):  # O(q × n)
    results = []
    for start, end in queries:
        results.append(sum(arr[start:end]))
    return results
\`\`\`

✅ **Fast with Prefix Sums:**
\`\`\`python
def query_sum(arr, queries):  # O(n + q)
    prefix = [0]
    for num in arr:
        prefix.append(prefix[-1] + num)
    results = []
    for start, end in queries:
        results.append(prefix[end] - prefix[start])
    return results
\`\`\`

**4. Two Pointers Instead of Nested Loops**

❌ **Slow:**
\`\`\`python
def two_sum_sorted(arr, target):  # O(n²)
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            if arr[i] + arr[j] == target:
                return [i, j]
    return None
\`\`\`

✅ **Fast:**
\`\`\`python
def two_sum_sorted(arr, target):  # O(n)
    left, right = 0, len(arr) - 1
    while left < right:
        curr_sum = arr[left] + arr[right]
        if curr_sum == target:
            return [left, right]
        elif curr_sum < target:
            left += 1
        else:
            right -= 1
    return None
\`\`\`

**Common Trade-offs:**

**Time vs. Space:**
- **More Space for Speed:** Memoization, hash tables, prefix sums
- **More Time for Space:** In-place algorithms, iterative instead of recursive

**Preprocessing vs. Query Time:**
- Build index once (O(n)) → Fast queries (O(1))
- Good when many queries expected

**Code Simplicity vs. Performance:**
- Readable but slower
- Optimized but complex
- **Rule:** Optimize only when needed!

**When to Optimize:**

1. **Measure first** - profile your code
2. **Identify bottlenecks** - don't optimize everywhere
3. **Consider trade-offs** - is the improvement worth the complexity?
4. **Keep it readable** - optimize only hot paths

**The Optimization Process:**

1. **Analyze current complexity** - identify the bottleneck
2. **Look for patterns** - nested loops? redundant work?
3. **Consider better data structures** - hash map? heap? tree?
4. **Apply techniques** - memoization? two pointers? binary search?
5. **Verify correctness** - optimization should not break code!
6. **Measure improvement** - did it actually help?

**Red Flags in Code Review:**

- Nested loops over same data → Try hash map or two pointers
- Repeated calculations → Memoize or precompute
- Searching in list → Use set or hash map
- Sorting inside loop → Sort once outside
- Recursive without memoization → Add caching

**Interview Strategy:**

1. Start with brute force - explain it clearly
2. Analyze its complexity
3. Identify what makes it slow
4. Propose optimization
5. Analyze improved complexity
6. Discuss trade-offs`,
      quiz: [
        {
          id: 'q1',
          question:
            'Walk me through how using a hash map can turn an O(n²) algorithm into O(n). Use the two-sum problem as your example.',
          sampleAnswer:
            'In the naive two-sum approach, you use nested loops - for each element, you check every other element to see if they add up to the target. That is n × n = O(n²) comparisons. With a hash map, you can do it in one pass. As you iterate through the array, you check if (target - current_number) exists in the hash map. If it does, you found your pair. If not, you add the current number to the map and continue. Hash map lookups are O(1), so you are doing O(1) work for each of n elements, giving O(n) total. The hash map trades O(n) extra space for a massive speedup from O(n²) to O(n) time.',
          keyPoints: [
            'Naive: nested loops → O(n²)',
            'Optimized: hash map for O(1) lookups',
            'One pass, checking (target - current) in map',
            'Trade O(n) space for O(n²) → O(n) time improvement',
          ],
        },
        {
          id: 'q2',
          question:
            'Explain what memoization is and why it can dramatically improve recursive algorithms. What is the trade-off?',
          sampleAnswer:
            'Memoization is caching the results of function calls so you do not recompute them. In recursive algorithms, you often calculate the same subproblems many times - for example, fibonacci(5) calls fibonacci(3) multiple times. With memoization, the first time you calculate fibonacci(3), you store the result in a cache. Next time you need it, you just look it up in O(1) time instead of recalculating. For Fibonacci, this turns O(2^n) exponential time into O(n) linear time because you only calculate each value once. The trade-off is that you use O(n) extra space for the cache. But going from exponential to linear time is usually worth it - the speed improvement is massive.',
          keyPoints: [
            'Memoization: cache function results to avoid recomputation',
            'Recursive algorithms often recalculate same subproblems',
            'Store result first time, look up next time',
            'Trade-off: O(n) space for exponential → linear time improvement',
          ],
        },
        {
          id: 'q3',
          question:
            'When would you choose an O(n log n) algorithm over an O(n²) algorithm? Is the O(n log n) algorithm always better?',
          sampleAnswer:
            'O(n log n) is asymptotically better - it scales much better as n grows large. For 1000 elements, O(n²) is a million operations while O(n log n) is about 10,000 - a hundred times faster. However, for very small inputs, the O(n²) algorithm might actually be faster because it has lower constant factors. For example, insertion sort (O(n²)) can beat merge sort (O(n log n)) on arrays of size 10-20 because it has less overhead. Also, insertion sort is O(n) on already-sorted data. So you need to consider: 1) How large is n typically? 2) What are the constant factors? 3) What is the distribution of input? For large n, O(n log n) almost always wins.',
          keyPoints: [
            'O(n log n) scales much better for large n',
            'But may have higher constants/overhead',
            'For small n, O(n²) might be faster in practice',
            'Consider input size and characteristics',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question:
            'How can using a hash map improve an O(n²) nested loop algorithm?',
          options: [
            'By reducing space complexity',
            'By providing O(1) lookups to eliminate the inner loop',
            'By sorting the data faster',
            'By using less memory',
          ],
          correctAnswer: 1,
          explanation:
            'A hash map provides O(1) lookups, which can replace an inner O(n) loop with a single O(1) operation. For example, in two-sum, instead of checking every other element (O(n²)), we can check if the complement exists in the hash map (O(n)).',
        },
        {
          id: 'mc2',
          question:
            'What is memoization and what complexity problem does it solve?',
          options: [
            'Sorting data to improve search time',
            'Caching function results to avoid redundant calculations',
            'Using less memory in recursive functions',
            'Converting recursive to iterative solutions',
          ],
          correctAnswer: 1,
          explanation:
            'Memoization caches the results of expensive function calls and returns the cached result when the same inputs occur again. It can dramatically reduce time complexity (e.g., Fibonacci from O(2ⁿ) to O(n)) at the cost of O(n) space.',
        },
        {
          id: 'mc3',
          question:
            'When would you prefer an O(n log n) algorithm over an O(n²) algorithm?',
          options: [
            'Never, O(n²) is always faster',
            'For very small inputs only',
            'For large inputs where better scaling matters',
            'When you need to use less memory',
          ],
          correctAnswer: 2,
          explanation:
            'For large inputs, O(n log n) scales much better than O(n²). While O(n²) might be faster for very small inputs due to lower constants, O(n log n) becomes dramatically faster as n grows (e.g., n=1000: n log n ≈ 10,000 vs n² = 1,000,000).',
        },
        {
          id: 'mc4',
          question:
            'What optimization technique does this demonstrate?\n\n```python\nprefix_sum = [0]\nfor num in arr:\n    prefix_sum.append(prefix_sum[-1] + num)\n```',
          options: [
            'Memoization',
            'Two pointers',
            'Precomputation/preprocessing',
            'Binary search',
          ],
          correctAnswer: 2,
          explanation:
            'This is precomputation - calculating prefix sums upfront (O(n)) to enable O(1) range sum queries later. This trades O(n) space and preprocessing time for much faster subsequent queries.',
        },
        {
          id: 'mc5',
          question: 'Which is a valid time-space tradeoff?',
          options: [
            'Using less memory always makes algorithms faster',
            'Using more memory can sometimes make algorithms faster',
            'Time and space complexity must always be equal',
            'Optimization always improves both time and space',
          ],
          correctAnswer: 1,
          explanation:
            'Time-space tradeoffs often involve using more memory to achieve better time complexity. Examples include hash maps for O(1) lookups, memoization for avoiding recalculation, and prefix sums for fast range queries.',
        },
      ],
    },
  ],
  keyTakeaways: [
    'Big O measures how algorithms scale with input size, focusing on growth rate not exact counts',
    'Common complexities: O(1) < O(log n) < O(n) < O(n log n) < O(n²) < O(2ⁿ) < O(n!)',
    'Space complexity measures auxiliary memory - includes data structures and recursive call stack',
    'Time-space tradeoffs are common - memoization trades space for time',
    'Analyze worst case by default for safety guarantees',
    'Nested loops often indicate quadratic complexity - look for optimizations',
    'Hash tables can turn O(n²) algorithms into O(n) with O(n) space',
    'Always consider both time and space complexity in your analysis',
  ],
  timeComplexity: 'Varies by algorithm',
  spaceComplexity: 'Varies by algorithm',
  relatedProblems: [
    'squares-sorted',
    'pivot-index',
    'first-unique-char',
    'duplicate-number',
    'array-partition',
    'valid-number',
  ],
};
