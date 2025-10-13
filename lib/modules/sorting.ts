/**
 * Sorting module content - Professional & comprehensive guide
 */

import { Module } from '@/lib/types';

export const sortingModule: Module = {
  id: 'sorting',
  title: 'Sorting Algorithms',
  description:
    'Master the fundamental sorting algorithms and understand their time complexity, space usage, and when to use each.',
  icon: '🔄',
  sections: [
    {
      id: 'introduction',
      title: 'Why Sorting Matters',
      content: `Sorting is one of the most fundamental operations in computer science. It's the process of arranging data in a specific order (usually ascending or descending), and it serves as the foundation for many other algorithms.

**Why Learn Sorting?**
- **Ubiquitous:** Used everywhere - databases, search engines, operating systems
- **Prerequisite:** Many algorithms require sorted data (binary search, merge operations)
- **Interview Favorite:** Nearly every coding interview includes sorting questions
- **Algorithmic Thinking:** Teaches divide-and-conquer, recursion, and optimization

**Real-World Applications:**
- **Search Engines:** Ranking search results
- **Databases:** ORDER BY queries, index maintenance
- **E-commerce:** Sorting products by price, rating, popularity
- **Operating Systems:** Process scheduling, memory management
- **Data Analysis:** Finding top-k elements, percentiles

**The Sorting Landscape:**

Sorting algorithms can be categorized by:

1. **Time Complexity:**
   - Simple sorts: O(n²) - Bubble, Selection, Insertion
   - Efficient sorts: O(n log n) - Merge, Quick, Heap
   - Special cases: O(n) - Counting, Radix, Bucket

2. **Space Complexity:**
   - In-place: O(1) extra space - Quick, Heap, Insertion
   - Not in-place: O(n) extra space - Merge

3. **Stability:**
   - Stable: Equal elements maintain relative order - Merge, Insertion
   - Unstable: May change relative order - Quick, Heap

4. **Adaptive:**
   - Adaptive: Faster on nearly-sorted data - Insertion
   - Non-adaptive: Same speed regardless - Selection

**Key Questions to Ask:**
1. How large is the dataset? (Small → Insertion, Large → Quick/Merge)
2. Is it already partially sorted? (Yes → Insertion, Timsort)
3. Do I have extra memory? (Yes → Merge, No → Quick/Heap)
4. Must equal elements stay in order? (Yes → Stable sorts only)
5. What's the nature of the data? (Integers in range → Counting sort)

**The Classic Tradeoff:**
- **Simple algorithms:** Easy to understand, bad performance (O(n²))
- **Advanced algorithms:** Complex to understand, great performance (O(n log n))
- **Specialized algorithms:** Only work for specific data types, can be O(n)`,
      quiz: [
        {
          id: 'q1',
          question:
            'Explain what we mean by a "stable" sorting algorithm. Why would stability matter in practice?',
          hint: 'Think about what happens to equal elements.',
          sampleAnswer:
            'A stable sorting algorithm preserves the relative order of equal elements. If you have two items with the same value, and item A came before item B in the original array, then after a stable sort, A will still come before B. This matters when you are sorting by one field but want to preserve the order of another field. For example, if you have students sorted by name and you want to sort them by grade, a stable sort will keep students with the same grade in alphabetical order. Merge sort and insertion sort are stable, but quicksort and heapsort are not. In production, stability often matters for user-facing features where order needs to be predictable.',
          keyPoints: [
            'Stable: equal elements maintain relative order',
            'Important for multi-level sorting',
            'Example: sort by grade, preserve alphabetical order within same grade',
            'Merge sort, insertion sort are stable; quicksort, heapsort are not',
          ],
        },
        {
          id: 'q2',
          question:
            'Walk me through the difference between in-place and not in-place sorting. What is the tradeoff?',
          sampleAnswer:
            'In-place sorting means the algorithm sorts the array using only O(1) extra space - it rearranges elements within the original array without creating a copy. Not in-place sorting uses O(n) additional space, typically by creating temporary arrays. The tradeoff is space versus implementation simplicity. Quicksort is in-place - it sorts by swapping elements within the array, using minimal extra memory. Merge sort is not in-place - it creates temporary arrays during the merge step, requiring O(n) space. In-place is better when memory is limited, but not-in-place algorithms like merge sort can be easier to implement correctly and are stable. For large datasets, the O(n) space overhead of merge sort can be significant.',
          keyPoints: [
            'In-place: O(1) extra space, sorts within original array',
            'Not in-place: O(n) extra space, creates temporary copies',
            'In-place saves memory but can be more complex',
            'Examples: Quicksort (in-place), Merge sort (not in-place)',
          ],
        },
        {
          id: 'q3',
          question:
            'Why would you ever use a simple O(n²) sorting algorithm like insertion sort when O(n log n) algorithms exist?',
          sampleAnswer:
            "There are actually several good reasons. First, for small arrays (say under 20 elements), insertion sort can be faster than quicksort or mergesort because it has very low overhead - no recursive calls, no complex partitioning. Second, insertion sort is adaptive - it runs in O(n) time on already-sorted or nearly-sorted data. If you know your data is mostly sorted, insertion sort is excellent. Third, it is stable and in-place, which matters for certain use cases. Fourth, it is extremely simple to implement correctly. In fact, many production implementations of quicksort switch to insertion sort for small subarrays. Python's Timsort uses insertion sort as one of its building blocks.",
          keyPoints: [
            'Faster for small arrays due to low overhead',
            'Adaptive: O(n) on nearly-sorted data',
            'Stable and in-place simultaneously',
            'Simple to implement correctly',
            'Used in hybrid algorithms like Timsort',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What is a stable sorting algorithm?',
          options: [
            'Never crashes',
            'Preserves relative order of equal elements',
            'Always O(N log N)',
            'Uses no extra space',
          ],
          correctAnswer: 1,
          explanation:
            'Stable sort preserves the relative order of equal elements. If A comes before B and they\'re equal, A stays before B after sorting. Important for multi-level sorting.',
        },
        {
          id: 'mc2',
          question: 'What does "in-place" sorting mean?',
          options: [
            'Sorts very fast',
            'Uses only O(1) extra space, sorts within original array',
            'Always stable',
            'Never uses recursion',
          ],
          correctAnswer: 1,
          explanation:
            'In-place sorting uses O(1) extra space, modifying the original array without creating copies. Saves memory but may be more complex. Example: Quicksort (in-place), Merge sort (not in-place).',
        },
        {
          id: 'mc3',
          question: 'When would you use O(N²) insertion sort over O(N log N) algorithms?',
          options: [
            'Never',
            'Small arrays, nearly-sorted data, or when stability + in-place both needed',
            'Always',
            'Only for testing',
          ],
          correctAnswer: 1,
          explanation:
            'Insertion sort excels when: 1) Array is small (<20 elements) - low overhead, 2) Data is nearly sorted - adaptive O(N) time, 3) Need both stable and in-place. Used in Timsort and hybrid algorithms.',
        },
        {
          id: 'mc4',
          question: 'What is the theoretical lower bound for comparison-based sorts?',
          options: [
            'O(N)',
            'O(N log N) in average case',
            'O(N²)',
            'O(log N)',
          ],
          correctAnswer: 1,
          explanation:
            'Any comparison-based sort (comparing pairs) must be Ω(N log N) in average case. This is proven using decision tree analysis. Non-comparison sorts like counting sort can beat this.',
        },
        {
          id: 'mc5',
          question: 'Why does sorting matter in computer science?',
          options: [
            'Only for interviews',
            'Foundation for many algorithms (binary search), ubiquitous in real systems, teaches fundamental techniques',
            'Random requirement',
            'Historical reasons',
          ],
          correctAnswer: 1,
          explanation:
            'Sorting is fundamental: 1) Required for binary search and many algorithms, 2) Used everywhere (databases, search engines), 3) Teaches divide-and-conquer, recursion, optimization, 4) Common in interviews.',
        },
      ],
    },
    {
      id: 'comparison-sorts',
      title: 'Comparison-Based Sorting Algorithms',
      content: `**Comparison sorts** work by comparing pairs of elements. There's a theoretical lower bound: any comparison-based sort must be at least **O(n log n)** in the average case.

**1. Bubble Sort - O(n²)**

The simplest (and slowest) sort. Repeatedly steps through the list, compares adjacent elements, and swaps them if they're in the wrong order.

\`\`\`python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:  # Optimization: stop if already sorted
            break
    return arr
\`\`\`

- **Time:** O(n²) average and worst, O(n) best (already sorted)
- **Space:** O(1)
- **Stable:** Yes
- **Use case:** Educational purposes only, not for production

**2. Selection Sort - O(n²)**

Finds the minimum element and places it at the beginning. Repeat for the rest of the array.

\`\`\`python
def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr
\`\`\`

- **Time:** O(n²) always - not adaptive
- **Space:** O(1)
- **Stable:** No (but can be made stable)
- **Use case:** When minimizing number of swaps matters

**3. Insertion Sort - O(n²)**

Builds the final sorted array one item at a time by inserting each element into its correct position.

\`\`\`python
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr
\`\`\`

- **Time:** O(n²) worst, O(n) best (already sorted)
- **Space:** O(1)
- **Stable:** Yes
- **Use case:** Small arrays, nearly-sorted data, online sorting

**4. Merge Sort - O(n log n)**

Divide-and-conquer: split array in half, recursively sort each half, then merge.

\`\`\`python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
\`\`\`

- **Time:** O(n log n) always - consistent performance
- **Space:** O(n) - not in-place
- **Stable:** Yes
- **Use case:** When stability matters, external sorting, linked lists

**5. Quick Sort - O(n log n) average**

Pick a pivot, partition array so smaller elements are left, larger are right, recursively sort partitions.

\`\`\`python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)
\`\`\`

- **Time:** O(n log n) average, O(n²) worst (bad pivots)
- **Space:** O(log n) call stack - in-place version exists
- **Stable:** No (but can be made stable)
- **Use case:** General purpose, fastest in practice, default in many languages

**6. Heap Sort - O(n log n)**

Build a max heap, then repeatedly extract the maximum.

\`\`\`python
def heap_sort(arr):
    import heapq
    # Use min heap to get sorted order
    heapq.heapify(arr)
    return [heapq.heappop(arr) for _ in range(len(arr))]
\`\`\`

- **Time:** O(n log n) always - consistent
- **Space:** O(1) - in-place
- **Stable:** No
- **Use case:** When you need O(n log n) guarantee and can't afford O(n) space`,
      quiz: [
        {
          id: 'q1',
          question:
            'Compare merge sort and quick sort. When would you choose one over the other?',
          hint: 'Think about stability, space, and performance guarantees.',
          sampleAnswer:
            'Both are O(n log n) on average, but they have different tradeoffs. Merge sort is stable and has guaranteed O(n log n) worst-case performance, but requires O(n) extra space. Quick sort is typically faster in practice due to better cache performance and lower constants, is in-place (O(log n) space for recursion), but has O(n²) worst case and is not stable. I would choose merge sort when: 1) Stability is required, 2) I need guaranteed O(n log n) performance, 3) Memory is not a concern, or 4) Sorting linked lists. I would choose quick sort when: 1) Average performance matters more than worst-case, 2) Memory is limited, 3) Stability is not needed. In practice, quicksort with randomized pivots is usually the go-to for general purpose sorting.',
          keyPoints: [
            'Merge sort: stable, O(n log n) guaranteed, O(n) space',
            'Quick sort: faster in practice, in-place, O(n²) worst case, unstable',
            'Choose merge for stability and guaranteed performance',
            'Choose quick for speed and memory efficiency',
          ],
        },
        {
          id: 'q2',
          question:
            'Explain why the worst case of quicksort is O(n²). How can you avoid this?',
          sampleAnswer:
            'Quicksort degrades to O(n²) when the pivot choices are consistently bad - when the pivot is always the smallest or largest element. This creates unbalanced partitions where one side has n-1 elements and the other has 0. You end up with n levels of recursion instead of log n, and each level does O(n) work, giving O(n²) total. This happens when sorting already-sorted data with a naive pivot selection like choosing the first or last element. You can avoid this by: 1) Using randomized pivot selection, 2) Median-of-three pivot selection (choose median of first, middle, last), or 3) Using three-way partitioning for arrays with many duplicates. Randomization essentially guarantees O(n log n) average performance regardless of input.',
          keyPoints: [
            'Worst case when pivot always creates unbalanced partitions',
            'Happens on sorted data with poor pivot selection',
            'Results in n levels × O(n) work = O(n²)',
            'Avoid with: randomized pivots, median-of-three, three-way partitioning',
          ],
        },
        {
          id: 'q3',
          question:
            'Why is insertion sort O(n) on already-sorted data but O(n²) on random data?',
          sampleAnswer:
            'Insertion sort is adaptive - its performance depends on how sorted the input already is. On already-sorted data, each element is already in the correct position. The inner while loop never executes because arr[j] is never greater than key. So we just scan through the array once, doing O(1) work per element, giving O(n) total. On random or reverse-sorted data, each element might need to be compared with and moved past many elements to find its correct position. On average, each element moves halfway back, which is O(n) comparisons and shifts for each of n elements, giving O(n²). This adaptive property makes insertion sort excellent for nearly-sorted data or online sorting where elements arrive one at a time.',
          keyPoints: [
            'Adaptive: performance depends on initial order',
            'Already sorted: inner loop never runs → O(n)',
            'Random data: each element shifts ~n/2 positions → O(n²)',
            'Makes it great for nearly-sorted data and online sorting',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What is the best, average, and worst case complexity of Quicksort?',
          options: [
            'All O(N log N)',
            'Best: O(N log N), Average: O(N log N), Worst: O(N²)',
            'All O(N²)',
            'Best: O(N), Average: O(N log N), Worst: O(N²)',
          ],
          correctAnswer: 1,
          explanation:
            'Quicksort: Best O(N log N) with balanced pivots, Average O(N log N), Worst O(N²) when pivot is always min/max (sorted input). Randomization makes worst case unlikely.',
        },
        {
          id: 'mc2',
          question: 'Why is Merge Sort guaranteed O(N log N) but uses O(N) space?',
          options: [
            'Poor implementation',
            'Divide-and-conquer always splits evenly, but merge step needs temporary arrays',
            'Random',
            'Space can be reduced to O(1)',
          ],
          correctAnswer: 1,
          explanation:
            'Merge sort divides array in half each time (log N levels), each level processes N elements = O(N log N). The merge step creates temporary arrays to combine sorted halves = O(N) space.',
        },
        {
          id: 'mc3',
          question: 'What makes Heap Sort useful despite being slower than Quicksort in practice?',
          options: [
            'It is not useful',
            'Guaranteed O(N log N) worst case, in-place O(1) space',
            'Stable',
            'Adaptive',
          ],
          correctAnswer: 1,
          explanation:
            'Heap sort guarantees O(N log N) worst case (unlike quicksort\'s O(N²)) and is in-place O(1) space (unlike merge sort\'s O(N)). Good when memory is limited and worst-case matters.',
        },
        {
          id: 'mc4',
          question: 'Which sorting algorithm is stable among comparison sorts?',
          options: [
            'Quicksort',
            'Merge sort',
            'Heap sort',
            'All comparison sorts',
          ],
          correctAnswer: 1,
          explanation:
            'Merge sort is stable - equal elements maintain relative order during merge. Quicksort and Heap sort are unstable due to swapping. Insertion and Bubble are also stable.',
        },
        {
          id: 'mc5',
          question: 'Why does Quicksort often outperform Merge Sort in practice despite same average complexity?',
          options: [
            'Better complexity',
            'In-place (cache-friendly), fewer memory operations, lower constant factors',
            'Random',
            'Always slower',
          ],
          correctAnswer: 1,
          explanation:
            'Quicksort is in-place (better cache locality), has fewer memory operations (no array copies), and has lower constant factors. Merge sort creates temporary arrays repeatedly, causing overhead.',
        },
      ],
    },
    {
      id: 'non-comparison-sorts',
      title: 'Non-Comparison Sorting Algorithms',
      content: `**Non-comparison sorts** don't compare elements directly. They exploit properties of the data (like range of integers) to achieve **O(n)** time complexity!

**1. Counting Sort - O(n + k)**

Works for integers in a known range [0, k]. Counts occurrences of each value.

\`\`\`python
def counting_sort(arr):
    if not arr:
        return arr
    
    max_val = max(arr)
    min_val = min(arr)
    range_size = max_val - min_val + 1
    
    # Count occurrences
    count = [0] * range_size
    for num in arr:
        count[num - min_val] += 1
    
    # Reconstruct sorted array
    result = []
    for i in range(range_size):
        result.extend([i + min_val] * count[i])
    
    return result
\`\`\`

- **Time:** O(n + k) where k is the range of values
- **Space:** O(k) for count array
- **Stable:** Yes (with careful implementation)
- **Use case:** Small range of integers, when k = O(n)

**2. Radix Sort - O(d × n)**

Sorts integers digit by digit, from least to most significant.

\`\`\`python
def radix_sort(arr):
    if not arr:
        return arr
    
    max_val = max(arr)
    exp = 1  # Current digit position
    
    while max_val // exp > 0:
        counting_sort_by_digit(arr, exp)
        exp *= 10
    
    return arr

def counting_sort_by_digit(arr, exp):
    n = len(arr)
    output = [0] * n
    count = [0] * 10  # Digits 0-9
    
    # Count occurrences of digits
    for i in range(n):
        digit = (arr[i] // exp) % 10
        count[digit] += 1
    
    # Cumulative count
    for i in range(1, 10):
        count[i] += count[i - 1]
    
    # Build output array
    for i in range(n - 1, -1, -1):
        digit = (arr[i] // exp) % 10
        output[count[digit] - 1] = arr[i]
        count[digit] -= 1
    
    # Copy back
    for i in range(n):
        arr[i] = output[i]
\`\`\`

- **Time:** O(d × n) where d is number of digits
- **Space:** O(n + k) auxiliary space
- **Stable:** Yes (relies on stable counting sort)
- **Use case:** Sorting integers or strings with fixed-length keys

**3. Bucket Sort - O(n + k)**

Distributes elements into buckets, sorts each bucket, then concatenates.

\`\`\`python
def bucket_sort(arr):
    if not arr:
        return arr
    
    # Create buckets
    bucket_count = len(arr)
    max_val = max(arr)
    min_val = min(arr)
    bucket_range = (max_val - min_val) / bucket_count + 1
    
    buckets = [[] for _ in range(bucket_count)]
    
    # Distribute into buckets
    for num in arr:
        index = int((num - min_val) / bucket_range)
        buckets[index].append(num)
    
    # Sort each bucket and concatenate
    result = []
    for bucket in buckets:
        result.extend(sorted(bucket))  # Use insertion sort for small buckets
    
    return result
\`\`\`

- **Time:** O(n + k) average, O(n²) worst if all elements in one bucket
- **Space:** O(n + k)
- **Stable:** Depends on bucket sort used
- **Use case:** Uniformly distributed data, floating-point numbers

**When to Use Non-Comparison Sorts:**

✅ **Use when:**
- Sorting integers in a known, small range (counting sort)
- Data has fixed-length keys or limited digits (radix sort)
- Data is uniformly distributed (bucket sort)
- Need O(n) time and can afford O(n) or O(k) space

❌ **Don't use when:**
- Sorting arbitrary objects with comparison function
- Range k >> n (counting sort wastes space)
- Data distribution is unknown (bucket sort degrades)
- Need truly general-purpose sorting

**Comparison:**

| Algorithm | Time | Space | Stable | Use Case |
|-----------|------|-------|--------|----------|
| Counting  | O(n+k) | O(k) | Yes | Small integer range |
| Radix     | O(d×n) | O(n) | Yes | Fixed-length keys |
| Bucket    | O(n+k) | O(n) | Depends | Uniform distribution |

**Real-World Applications:**
- **Counting Sort:** Sorting grades (0-100), sorting by age
- **Radix Sort:** Sorting IP addresses, sorting strings
- **Bucket Sort:** Sorting floating-point numbers, external sorting`,
      quiz: [
        {
          id: 'q1',
          question:
            'Explain why counting sort is O(n + k) instead of O(n). When does this become a problem?',
          hint: 'Think about what k represents.',
          sampleAnswer:
            'Counting sort is O(n + k) where n is the number of elements and k is the range of values. The n comes from iterating through the input array to count occurrences and then to build the output. The k comes from initializing the count array of size k and iterating through it to reconstruct the sorted array. This becomes a problem when k is much larger than n. For example, if you have 100 numbers ranging from 0 to 1 million, you need a count array of size 1 million, and you have to iterate through all 1 million positions even though only 100 of them have non-zero counts. In this case, k >> n, so O(n + k) ≈ O(k), which is worse than O(n log n) comparison sorts.',
          keyPoints: [
            'O(n) to count occurrences and reconstruct array',
            'O(k) to initialize and iterate through count array',
            'Problem when k >> n (large range, few elements)',
            'Example: 100 numbers in range [0, 1M] wastes space and time',
          ],
        },
        {
          id: 'q2',
          question:
            'How does radix sort achieve O(n) time when the theoretical lower bound for comparison-based sorting is O(n log n)?',
          sampleAnswer:
            'Radix sort gets around the O(n log n) lower bound because it is not a comparison-based sort - it never compares two elements directly. Instead, it exploits the structure of the data by sorting digit by digit. The lower bound of O(n log n) only applies to algorithms that work by comparing elements. Radix sort looks at the individual digits, which is a fundamentally different approach. It is O(d × n) where d is the number of digits. For fixed-length integers, d is constant, so it is effectively O(n). However, this only works for specific types of data - integers, strings, etc. You cannot use radix sort to sort arbitrary objects with a comparison function, which is why comparison-based sorts are still important.',
          keyPoints: [
            'Not comparison-based - never compares two elements',
            'Exploits data structure by processing digits',
            'O(n log n) lower bound only for comparison sorts',
            'O(d × n) where d is number of digits; O(n) for fixed d',
            'Only works for specific data types',
          ],
        },
        {
          id: 'q3',
          question:
            'When would you choose bucket sort over quicksort? What properties of the data make bucket sort effective?',
          sampleAnswer:
            "I would choose bucket sort when the data is uniformly distributed across a known range. Bucket sort works by dividing the range into buckets and distributing elements into those buckets, assuming they will spread out evenly. If the data is uniform, each bucket gets roughly n/k elements, and sorting each bucket takes O(n/k × log(n/k)) time, which averages out to O(n) overall. This is better than quicksort's O(n log n). However, if the distribution is skewed and all elements fall into a few buckets, bucket sort degrades to O(n²). So I would use bucket sort for things like sorting random floating-point numbers between 0 and 1, or sorting uniformly distributed sensor data. For general-purpose sorting without knowing the distribution, quicksort is safer.",
          keyPoints: [
            'Choose bucket sort for uniformly distributed data',
            'Uniform distribution → elements spread evenly across buckets',
            "Achieves O(n) average time vs quicksort's O(n log n)",
            'Degrades to O(n²) if data is skewed',
            'Good for: random floats, uniform data; Bad for: unknown distribution',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'How do non-comparison sorts achieve O(N) time?',
          options: [
            'Magic',
            'Exploit data properties (range, digits) instead of comparing elements',
            'Always faster',
            'Use parallel processing',
          ],
          correctAnswer: 1,
          explanation:
            'Non-comparison sorts like counting sort exploit known properties of data (e.g., integers 0-k, fixed digits). They count/distribute rather than compare, bypassing the O(N log N) comparison lower bound.',
        },
        {
          id: 'mc2',
          question: 'When should you use Counting Sort?',
          options: [
            'Always',
            'When sorting integers in a small known range [0, k] where k is not too large',
            'For any data',
            'Never',
          ],
          correctAnswer: 1,
          explanation:
            'Counting sort is O(N + k) time and space. Use when k (range) is small relative to N. For integers 0-k, count occurrences and reconstruct. If k >> N, space becomes prohibitive.',
        },
        {
          id: 'mc3',
          question: 'What is the time complexity of Radix Sort for d-digit numbers?',
          options: [
            'O(N log N)',
            'O(d × N) - sort by each digit, O(N) for fixed d',
            'O(N²)',
            'O(N)',
          ],
          correctAnswer: 1,
          explanation:
            'Radix sort processes d digits, using counting sort O(N) per digit = O(d × N). For fixed d (like 32-bit integers), this is O(N), beating comparison sorts.',
        },
        {
          id: 'mc4',
          question: 'What makes Bucket Sort effective?',
          options: [
            'Always works',
            'Uniformly distributed data spreads evenly across buckets, each small subset sorts in O(N/k log N/k) ≈ O(N)',
            'Random',
            'Stable',
          ],
          correctAnswer: 1,
          explanation:
            'Bucket sort distributes N elements into k buckets. If uniform, each bucket has ~N/k elements. Sorting each is O(N/k log N/k), totaling O(N) average. Skewed data degrades to O(N²).',
        },
        {
          id: 'mc5',
          question: 'What is the main limitation of non-comparison sorts?',
          options: [
            'Too slow',
            'Only work for specific data types (integers in range, fixed digits) with additional constraints',
            'Always unstable',
            'Use too much space',
          ],
          correctAnswer: 1,
          explanation:
            'Non-comparison sorts are specialized: counting sort needs known range, radix needs digit representation, bucket needs uniform distribution. Can\'t sort arbitrary objects or use custom comparators.',
        },
      ],
    },
    {
      id: 'practical-considerations',
      title: 'Practical Sorting Strategies',
      content: `**Hybrid Algorithms: The Best of Both Worlds**

Real-world sorting implementations use **hybrid algorithms** that combine multiple techniques:

**1. Timsort (Python's sorted() and .sort())**

A hybrid of merge sort and insertion sort, designed for real-world data.

**Key Ideas:**
- Identifies "runs" (already-sorted subsequences) in the data
- Uses insertion sort for small runs (< 64 elements)
- Merges runs using merge sort
- Adaptive: O(n) on already-sorted data, O(n log n) worst case

**Why it's brilliant:**
- Real-world data is often partially sorted
- Exploits natural order in data
- Stable and consistent O(n log n) worst case
- Used in Python, Java, Android, and more

**2. Introsort (C++'s std::sort)**

Hybrid of quicksort, heapsort, and insertion sort.

**Algorithm:**
1. Start with quicksort (fastest average case)
2. If recursion depth exceeds log n, switch to heapsort (prevents O(n²))
3. For small subarrays (< 16), use insertion sort

**Why it's brilliant:**
- Gets quicksort's speed without its O(n²) risk
- Guaranteed O(n log n) worst case
- In-place like quicksort
- Not stable, but very fast

**Choosing the Right Sort:**

\`\`\`
START
  |
  v
Is data size < 20?
  |-- YES --> Insertion Sort
  |
  |-- NO --> Is stability required?
              |
              |-- YES --> Is memory constrained?
              |             |-- YES --> In-place stable sort (harder)
              |             |-- NO --> Merge Sort or Timsort
              |
              |-- NO --> Is O(n log n) guarantee needed?
                          |-- YES --> Heap Sort or Introsort
                          |-- NO --> Quick Sort (with random pivot)
\`\`\`

**Special Cases:**

**Nearly Sorted Data:**
- **Best:** Insertion Sort (O(n)), Timsort
- **Avoid:** Selection Sort (always O(n²))

**Many Duplicate Values:**
- **Best:** Three-way Quicksort
- **Good:** Merge Sort
- **Avoid:** Standard Quicksort (degrades)

**Limited Memory:**
- **Best:** In-place sorts (Quick, Heap, Insertion)
- **Avoid:** Merge Sort (O(n) space)

**Need Stability:**
- **Best:** Merge Sort, Timsort
- **Good:** Insertion Sort for small data
- **Avoid:** Quick Sort, Heap Sort (unstable)

**External Sorting (Data Doesn't Fit in Memory):**
- **Best:** External Merge Sort
- Divide data into chunks that fit in memory
- Sort each chunk
- Merge chunks from disk

**Optimization Tips:**

**1. Use Built-In Sorts:**
\`\`\`python
# Python - Timsort
arr.sort()  # In-place
sorted_arr = sorted(arr)  # Returns new list

# Custom comparison
arr.sort(key=lambda x: x.name)
\`\`\`

**2. Partial Sorting:**
If you only need top-k elements, don't sort everything!
\`\`\`python
import heapq
# Get 10 largest - O(n log k) vs O(n log n)
top_10 = heapq.nlargest(10, arr)
\`\`\`

**3. Check if Already Sorted:**
\`\`\`python
def is_sorted(arr):
    return all(arr[i] <= arr[i+1] for i in range(len(arr)-1))

if not is_sorted(arr):
    arr.sort()  # Skip sorting if already sorted
\`\`\`

**4. Use Counting Sort for Small Ranges:**
\`\`\`python
# If sorting grades 0-100
if all(0 <= x <= 100 for x in grades):
    grades = counting_sort(grades)  # O(n) vs O(n log n)
\`\`\`

**Interview Tips:**

**When asked to sort:**
1. **Clarify requirements:**
   - Input size? (Small → insertion, Large → quick/merge)
   - Already partially sorted? (Adaptive sorts)
   - Stability required?
   - Memory constraints?

2. **Start with built-in:**
   - "In production, I'd use arr.sort()"
   - Then: "But let me implement [algorithm] to show understanding"

3. **Know your complexity:**
   - State time and space complexity
   - Explain best/average/worst cases

4. **Consider alternatives:**
   - "If we only need top-k, a heap would be more efficient"
   - "For integers in small range, counting sort is O(n)"`,
      quiz: [
        {
          id: 'q1',
          question:
            'Explain what makes Timsort adaptive and why this is useful for real-world data.',
          sampleAnswer:
            'Timsort is adaptive because it detects and exploits existing order in the data. Instead of treating all data the same, it looks for "runs" - sequences that are already sorted (either ascending or descending). When it finds these runs, it takes advantage of them rather than breaking them apart and resorting from scratch. For already-sorted data, Timsort runs in O(n) time, just verifying the order. For partially-sorted data, it does less work than algorithms like quicksort which don\'t recognize the existing order. This is incredibly useful in practice because real-world data is often partially sorted - think of adding new records to a database, logging events with timestamps, or updating ranked lists. Timsort handles these cases very efficiently.',
          keyPoints: [
            'Adaptive: performance improves with existing order',
            'Detects and preserves sorted runs',
            'O(n) for already-sorted data',
            'Real-world data often has patterns and partial order',
            'Much faster than non-adaptive algorithms on real data',
          ],
        },
        {
          id: 'q2',
          question:
            'You need to find the top 10 elements from a million-element array. Would you sort the entire array? Why or why not?',
          hint: 'Think about what complexity you actually need.',
          sampleAnswer:
            'No, I would not sort the entire array. Sorting takes O(n log n) time, which for a million elements is about 20 million operations. Instead, I would use a min-heap of size 10. I iterate through the array once, and for each element, if it is larger than the smallest element in my heap (the heap root), I remove the root and add the new element. This takes O(n log k) time where k is 10, so about 1 million × 3 = 3 million operations. That is almost 7 times faster. The key insight is that I do not need a fully sorted array - I just need the top k elements. Using a heap for partial sorting is way more efficient than full sorting. In Python, heapq.nlargest() does exactly this.',
          keyPoints: [
            'Full sort: O(n log n) - wasteful for top-k problem',
            'Min-heap approach: O(n log k) where k=10',
            'For k << n, heap is much faster',
            "Don't do more work than necessary",
            'Use heapq.nlargest() or heapq.nsmallest()',
          ],
        },
        {
          id: 'q3',
          question:
            'Walk me through when you would choose an unstable sort over a stable sort, given that stability seems strictly better.',
          sampleAnswer:
            'Stability is not always free - it can come with performance or implementation complexity costs. I would choose an unstable sort when: 1) Stability does not matter for my use case - if I am sorting primitive values like integers where there is no "secondary" ordering to preserve. 2) Performance is critical and the unstable sort is faster - quicksort is generally faster than merge sort due to better cache locality and lower constants, even though it is unstable. 3) Memory is constrained - heapsort is O(1) space and O(n log n) guaranteed time, which is hard to beat if stability is not needed. In practice, if you are sorting simple values and stability is not a requirement, using the faster unstable sort is the right engineering decision. The key is knowing what you need.',
          keyPoints: [
            'Stability can have performance/space costs',
            'Unstable OK when: sorting primitives, no secondary ordering matters',
            'Quicksort often faster than merge sort',
            'Heapsort: in-place + O(n log n) guarantee',
            'Choose based on requirements, not "stability is always better"',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What is Timsort and why does Python use it?',
          options: [
            'Random algorithm',
            'Hybrid of merge sort + insertion sort, stable, adaptive O(N) on sorted data, O(N log N) worst case',
            'Just quicksort',
            'New algorithm',
          ],
          correctAnswer: 1,
          explanation:
            'Timsort combines merge sort (stable, O(N log N)) with insertion sort (fast on small/sorted data). It\'s adaptive: O(N) on sorted data, O(N log N) worst case. Used in Python and Java for stability and real-world performance.',
        },
        {
          id: 'mc2',
          question: 'When should you use Introsort over Quicksort?',
          options: [
            'Never',
            'Need O(N log N) worst-case guarantee - switches to heapsort if recursion too deep',
            'Always',
            'Random',
          ],
          correctAnswer: 1,
          explanation:
            'Introsort starts with quicksort (fast average case), monitors recursion depth, switches to heapsort if too deep (preventing O(N²) worst case). Used in C++ STL for guaranteed O(N log N).',
        },
        {
          id: 'mc3',
          question: 'Why do production sorts often switch to insertion sort for small subarrays?',
          options: [
            'Random choice',
            'Insertion sort has lower overhead, runs faster than quicksort/mergesort on small N (<10-20)',
            'Always better',
            'Stability',
          ],
          correctAnswer: 1,
          explanation:
            'For small N, insertion sort\'s simplicity (no recursion, minimal operations) beats quicksort/mergesort\'s overhead. Hybrid algorithms use quick/merge for large N, insertion for small subarrays.',
        },
        {
          id: 'mc4',
          question: 'What should you consider when choosing a sorting algorithm?',
          options: [
            'Only speed',
            'Data size, stability requirement, memory constraints, data distribution, worst-case guarantees',
            'Random',
            'Nothing',
          ],
          correctAnswer: 1,
          explanation:
            'Consider: 1) Size (small→insertion, large→quick/merge), 2) Stability needed? (merge), 3) Memory limited? (heap/quick), 4) Nearly sorted? (insertion/timsort), 5) Worst-case matters? (heap/introsort).',
        },
        {
          id: 'mc5',
          question: 'What makes a sorting algorithm "adaptive"?',
          options: [
            'Uses AI',
            'Performance improves on partially sorted data (e.g., insertion O(N) on sorted, O(N²) on random)',
            'Always fast',
            'Random',
          ],
          correctAnswer: 1,
          explanation:
            'Adaptive algorithms exploit existing order. Insertion sort: O(N) on sorted (no shifts), O(N²) on random (many shifts). Timsort detects runs. Non-adaptive (quicksort) takes same time regardless.',
        },
      ],
    },
    {
      id: 'interview-problems',
      title: 'Common Sorting Interview Patterns',
      content: `**Pattern 1: Custom Sorting / Comparators**

Often you need to sort by custom criteria:

\`\`\`python
# Sort by multiple criteria
students = [("Alice", 85), ("Bob", 90), ("Charlie", 85)]
# Sort by grade desc, then name asc
students.sort(key=lambda x: (-x[1], x[0]))
# Result: [("Bob", 90), ("Alice", 85), ("Charlie", 85)]
\`\`\`

**Pattern 2: Sorting as a Preprocessing Step**

Many problems become easier after sorting:

\`\`\`python
# Find if any two numbers sum to target
def two_sum_sorted(arr, target):
    arr.sort()  # O(n log n)
    left, right = 0, len(arr) - 1
    while left < right:  # O(n)
        curr_sum = arr[left] + arr[right]
        if curr_sum == target:
            return True
        elif curr_sum < target:
            left += 1
        else:
            right -= 1
    return False
# Total: O(n log n)
\`\`\`

**Pattern 3: Partial Sorting**

Sometimes you don't need a complete sort:

\`\`\`python
# Kth largest element
def findKthLargest(nums, k):
    import heapq
    return heapq.nlargest(k, nums)[-1]
# O(n log k) vs O(n log n) for full sort
\`\`\`

**Pattern 4: Merge Sorted Arrays/Lists**

\`\`\`python
def merge_sorted_arrays(arr1, arr2):
    result = []
    i = j = 0
    while i < len(arr1) and j < len(arr2):
        if arr1[i] <= arr2[j]:
            result.append(arr1[i])
            i += 1
        else:
            result.append(arr2[j])
            j += 1
    result.extend(arr1[i:])
    result.extend(arr2[j:])
    return result
# O(n + m) - linear time!
\`\`\`

**Pattern 5: In-Place Sorting with Constraints**

\`\`\`python
# Sort colors (Dutch National Flag)
# Array contains only 0s, 1s, 2s - sort in-place in one pass
def sort_colors(nums):
    low = mid = 0
    high = len(nums) - 1
    
    while mid <= high:
        if nums[mid] == 0:
            nums[low], nums[mid] = nums[mid], nums[low]
            low += 1
            mid += 1
        elif nums[mid] == 1:
            mid += 1
        else:  # nums[mid] == 2
            nums[mid], nums[high] = nums[high], nums[mid]
            high -= 1
    return nums
# O(n) time, O(1) space
\`\`\`

**Pattern 6: Finding Duplicates After Sorting**

\`\`\`python
def find_duplicates(arr):
    arr.sort()  # O(n log n)
    duplicates = []
    for i in range(1, len(arr)):  # O(n)
        if arr[i] == arr[i-1] and (not duplicates or duplicates[-1] != arr[i]):
            duplicates.append(arr[i])
    return duplicates
\`\`\`

**Pattern 7: Interval Problems**

\`\`\`python
# Merge overlapping intervals
def merge_intervals(intervals):
    if not intervals:
        return []
    
    intervals.sort(key=lambda x: x[0])  # Sort by start time
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:  # Overlapping
            last[1] = max(last[1], current[1])
        else:
            merged.append(current)
    
    return merged
\`\`\`

**Common Interview Questions:**

1. **"Implement quick sort"** → Know the partition logic
2. **"Find Kth largest element"** → Quick select or heap
3. **"Merge K sorted lists"** → Min heap
4. **"Sort array with limited values (0,1,2)"** → Counting or Dutch flag
5. **"Is array almost sorted?"** → Check inversions, consider insertion sort
6. **"Sort linked list"** → Merge sort (no random access)
7. **"External sort large file"** → External merge sort

**Red Flags to Avoid:**

❌ Using bubble sort in production code
❌ Not considering stability when it matters
❌ Sorting when a heap or partial sort would suffice
❌ Forgetting that built-in sort is often O(n log n)
❌ Not asking about data properties (size, range, distribution)

**Quick Reference:**

| Need | Algorithm | Complexity |
|------|-----------|------------|
| General purpose | Quick Sort / Timsort | O(n log n) |
| Stability required | Merge Sort / Timsort | O(n log n) |
| Memory constrained | Heap Sort | O(n log n), O(1) space |
| Small data | Insertion Sort | O(n²) but fast |
| Nearly sorted | Insertion Sort / Timsort | O(n) best |
| Top-k elements | Heap | O(n log k) |
| Small integer range | Counting Sort | O(n + k) |
| Fixed-length keys | Radix Sort | O(d × n) |`,
      quiz: [
        {
          id: 'q1',
          question:
            'When would sorting actually make a problem easier to solve? Give me a concrete example.',
          sampleAnswer:
            'Sorting can turn an O(n²) problem into O(n log n) + O(n) = O(n log n). A great example is finding duplicate numbers. Without sorting, you need nested loops to compare every pair - O(n²). But if you sort first, duplicates end up adjacent. Then you just scan once checking if arr[i] == arr[i+1] - that is O(n). Total is O(n log n) for sort plus O(n) for scan, which is O(n log n) overall - much better than O(n²). Another example is the two-sum problem on sorted arrays - you can use two pointers to find a pair in O(n) after sorting. The pattern is: if you can solve it in linear time on sorted data, and sorting costs O(n log n), that is often better than the naive approach.',
          keyPoints: [
            'Sorting can reduce O(n²) to O(n log n)',
            'Example: find duplicates via adjacent elements',
            'Sorted data enables two-pointer technique',
            'Pattern: O(n log n) sort + O(n) scan < O(n²) naive',
          ],
        },
        {
          id: 'q2',
          question:
            'Explain the quick select algorithm. How is it related to quicksort, and what is its advantage?',
          hint: 'Think about finding the kth largest element.',
          sampleAnswer:
            "Quick select finds the kth largest (or smallest) element without fully sorting the array. It is based on quicksort's partition step. You partition the array around a pivot, which puts the pivot in its final sorted position. If the pivot is at position k, you found the kth element. If k is less than pivot position, recurse on the left partition. If k is greater, recurse on the right partition. Unlike quicksort which recurses on both sides, quick select only recurses on one side. This gives O(n) average time instead of O(n log n), because n + n/2 + n/4 + ... converges to 2n = O(n). The advantage over sorting is that you get O(n) average time versus O(n log n) - significant for large arrays when you only need one element.",
          keyPoints: [
            'Based on quicksort partition step',
            'Only recurse on one partition (where k is)',
            "Average case: O(n) vs quicksort's O(n log n)",
            'Used for finding kth largest/smallest element',
            'Much faster than full sort when you need one element',
          ],
        },
        {
          id: 'q3',
          question:
            'You need to merge k sorted linked lists. Walk through your approach and complexity.',
          sampleAnswer:
            'The optimal approach is to use a min heap. I would put the first node from each of the k lists into a min heap. Then repeatedly: 1) Extract the minimum from the heap (smallest overall), 2) Add it to the result list, 3) If that node had a next node, add the next node to the heap. The heap always contains at most k elements. Each of the n total nodes goes into the heap once and comes out once, and each heap operation is O(log k), giving O(n log k) total time. Space is O(k) for the heap. This is better than merging lists pairwise which would be O(n × k). The key insight is that I only need to compare the k current candidate nodes, not all n nodes, so a heap of size k is perfect.',
          keyPoints: [
            'Use min heap of size k',
            'Heap contains first unmerged node from each list',
            'Extract min, add to result, add its next to heap',
            'Time: O(n log k) - each of n nodes does O(log k) heap ops',
            'Space: O(k) for heap',
            'Better than pairwise merge: O(n log k) vs O(n × k)',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What is a common sorting interview pattern?',
          options: [
            'Always use quicksort',
            'Custom comparators, kth element, merge operations, frequency sorting, interval sorting',
            'Random',
            'Never sort',
          ],
          correctAnswer: 1,
          explanation:
            'Common patterns: 1) Custom comparators (sort by multiple criteria), 2) Kth largest (quickselect O(N)), 3) Merge k sorted arrays (heap), 4) Frequency sort (count→sort→build), 5) Interval sorting.',
        },
        {
          id: 'mc2',
          question: 'What is Quickselect and when do you use it?',
          options: [
            'Sorting algorithm',
            'Finding kth largest/smallest in O(N) average by partitioning without fully sorting',
            'Binary search',
            'Random algorithm',
          ],
          correctAnswer: 1,
          explanation:
            'Quickselect finds kth element in O(N) average by partitioning like quicksort but only recursing on one side. Beats sorting O(N log N) when you only need kth element, not full sorted array.',
        },
        {
          id: 'mc3',
          question: 'How do you merge k sorted arrays efficiently?',
          options: [
            'Merge one by one',
            'Min heap of k elements, repeatedly extract min and add next from same array - O(N log k)',
            'Sort everything',
            'Cannot do efficiently',
          ],
          correctAnswer: 1,
          explanation:
            'Use min heap with k elements (one from each array). Extract min (O(log k)), add next from that array. Process all N elements with O(log k) operations = O(N log k). Better than merging pairs O(N k).',
        },
        {
          id: 'mc4',
          question: 'What is the typical pattern for frequency-based sorting?',
          options: [
            'Just sort',
            'Count frequencies (hash map) → sort by frequency → build result',
            'Random',
            'Linear scan',
          ],
          correctAnswer: 1,
          explanation:
            'Frequency sort pattern: 1) Count occurrences with hash map O(N), 2) Sort by frequency O(unique log unique), 3) Build output array O(N). Total O(N log N) worst case.',
        },
        {
          id: 'mc5',
          question: 'What should you clarify in a sorting interview?',
          options: [
            'Nothing',
            'Input size, stability needed, memory constraints, data distribution, in-place requirement',
            'Just code fast',
            'Random',
          ],
          correctAnswer: 1,
          explanation:
            'Clarify: 1) Size (affects algorithm choice), 2) Stability (merge vs quick), 3) Memory limits (in-place?), 4) Data properties (nearly sorted? integers in range?), 5) Custom comparator? These determine optimal approach.',
        },
      ],
    },
  ],
  keyTakeaways: [
    'Comparison sorts are at least O(n log n) average case - fundamental lower bound',
    'Simple sorts (bubble, selection, insertion) are O(n²) but useful for small data',
    'Efficient sorts (merge, quick, heap) are O(n log n) with different tradeoffs',
    'Non-comparison sorts (counting, radix, bucket) can achieve O(n) for specific data types',
    'Stability matters when sorting by multiple criteria or preserving order',
    'Real-world implementations use hybrid algorithms like Timsort and Introsort',
    'Quick sort is fastest in practice but has O(n²) worst case; merge sort guarantees O(n log n)',
    'For top-k problems, use heaps (O(n log k)) instead of full sorting (O(n log n))',
  ],
  timeComplexity: 'Varies by algorithm: O(n²) to O(n log n) to O(n)',
  spaceComplexity: 'Varies by algorithm: O(1) to O(n)',
  relatedProblems: [
    'merge-sorted-arrays',
    'sort-array-parity',
    'insertion-sort-list',
    'sort-list',
    'wiggle-sort',
    'count-smaller',
  ],
};
