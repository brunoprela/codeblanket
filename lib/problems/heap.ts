import { Problem } from '../types';

export const heapProblems: Problem[] = [
  {
    id: 'kth-largest-element',
    title: 'Kth Largest Element in an Array',
    difficulty: 'Easy',
    description: `Given an integer array \`nums\` and an integer \`k\`, return **the kth largest element** in the array.

Note that it is the kth largest element in the sorted order, not the kth distinct element.

Can you solve it without sorting?

**LeetCode:** [215. Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array/)
**YouTube:** [NeetCode - Kth Largest Element in an Array](https://www.youtube.com/watch?v=XEmy13g1Qxc)

**Approach:**
Use a min heap of size K. Maintain only the K largest elements in the heap. The root (smallest in heap) is the Kth largest overall.

**Time:** O(N log K) vs O(N log N) for sorting
**Space:** O(K) vs O(1) for in-place sort`,
    examples: [
      {
        input: 'nums = [3,2,1,5,6,4], k = 2',
        output: '5',
        explanation: 'The 2nd largest element is 5.',
      },
      {
        input: 'nums = [3,2,3,1,2,4,5,5,6], k = 4',
        output: '4',
        explanation: 'The 4th largest element is 4.',
      },
    ],
    constraints: ['1 <= k <= nums.length <= 10^5', '-10^4 <= nums[i] <= 10^4'],
    hints: [
      'Use a min heap of size K to track the K largest elements',
      'The smallest element in this heap is the Kth largest overall',
      'For each new element, if heap is full and element > heap top, replace',
      'Time: O(N log K) which is better than O(N log N) sorting when K is small',
      'Alternative: Quickselect algorithm achieves O(N) average time',
    ],
    starterCode: `from typing import List
import heapq

def find_kth_largest(nums: List[int], k: int) -> int:
    """
    Find the kth largest element in an array.
    
    Args:
        nums: Array of integers
        k: Find kth largest
        
    Returns:
        The kth largest element
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[3, 2, 1, 5, 6, 4], 2],
        expected: 5,
      },
      {
        input: [[3, 2, 3, 1, 2, 4, 5, 5, 6], 4],
        expected: 4,
      },
      {
        input: [[1], 1],
        expected: 1,
      },
      {
        input: [[7, 6, 5, 4, 3, 2, 1], 2],
        expected: 6,
      },
    ],
    solution: `from typing import List
import heapq


def find_kth_largest(nums: List[int], k: int) -> int:
    """
    Min heap approach.
    Time: O(N log K), Space: O(K)
    """
    # Maintain min heap of size k
    heap = []
    
    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)  # Remove smallest
    
    return heap[0]  # Smallest in heap = kth largest overall


# Alternative: Max heap (negate values)
def find_kth_largest_max_heap(nums: List[int], k: int) -> int:
    """
    Max heap approach using negation.
    """
    heap = [-num for num in nums]
    heapq.heapify(heap)
    
    # Pop k-1 times
    for _ in range(k - 1):
        heapq.heappop(heap)
    
    return -heap[0]


# Alternative: Using heapq.nlargest
def find_kth_largest_builtin(nums: List[int], k: int) -> int:
    """
    Using Python's built-in function.
    Time: O(N log K), Space: O(K)
    """
    return heapq.nlargest(k, nums)[-1]


# Alternative: Quickselect (optimal average case)
def find_kth_largest_quickselect(nums: List[int], k: int) -> int:
    """
    Quickselect algorithm.
    Time: O(N) average, O(N²) worst
    Space: O(1)
    """
    k = len(nums) - k  # Convert to index (0-based, ascending order)
    
    def quickselect(left, right):
        pivot = nums[right]
        p = left
        
        for i in range(left, right):
            if nums[i] <= pivot:
                nums[p], nums[i] = nums[i], nums[p]
                p += 1
        
        nums[p], nums[right] = nums[right], nums[p]
        
        if p > k:
            return quickselect(left, p - 1)
        elif p < k:
            return quickselect(p + 1, right)
        else:
            return nums[p]
    
    return quickselect(0, len(nums) - 1)`,
    timeComplexity: 'O(N log K) for heap, O(N) average for quickselect',
    spaceComplexity: 'O(K) for heap, O(1) for quickselect',
    order: 1,
    topic: 'Heap / Priority Queue',
    leetcodeUrl:
      'https://leetcode.com/problems/kth-largest-element-in-an-array/',
    youtubeUrl: 'https://www.youtube.com/watch?v=XEmy13g1Qxc',
  },
  {
    id: 'top-k-frequent',
    title: 'Top K Frequent Elements',
    difficulty: 'Medium',
    description: `Given an integer array \`nums\` and an integer \`k\`, return **the k most frequent elements**. You may return the answer in **any order**.

**LeetCode:** [347. Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/)
**YouTube:** [NeetCode - Top K Frequent Elements](https://www.youtube.com/watch?v=YPTqKIgVk-k)

**Approach:**
1. Count frequencies using hash map
2. Use min heap of size K to track K most frequent
3. Store (frequency, element) tuples in heap

**Alternative:** Bucket sort achieves O(N) time.`,
    examples: [
      {
        input: 'nums = [1,1,1,2,2,3], k = 2',
        output: '[1,2]',
        explanation: 'Elements 1 and 2 are the two most frequent elements.',
      },
      {
        input: 'nums = [1], k = 1',
        output: '[1]',
      },
    ],
    constraints: [
      '1 <= nums.length <= 10^5',
      '-10^4 <= nums[i] <= 10^4',
      'k is in the range [1, the number of unique elements in the array]',
      'It is guaranteed that the answer is unique',
    ],
    hints: [
      'First, count the frequency of each element using a hash map',
      'Use a min heap of size K to track the K most frequent elements',
      'Store (frequency, element) tuples in the heap',
      'The heap will automatically keep the K elements with highest frequencies',
      'Alternative: Use bucket sort for O(N) time complexity',
    ],
    starterCode: `from typing import List
import heapq
from collections import Counter

def top_k_frequent(nums: List[int], k: int) -> List[int]:
    """
    Find the k most frequent elements.
    
    Args:
        nums: Array of integers
        k: Number of most frequent elements to return
        
    Returns:
        List of k most frequent elements
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[1, 1, 1, 2, 2, 3], 2],
        expected: [1, 2],
      },
      {
        input: [[1], 1],
        expected: [1],
      },
      {
        input: [[4, 1, -1, 2, -1, 2, 3], 2],
        expected: [-1, 2],
      },
    ],
    solution: `from typing import List
import heapq
from collections import Counter


def top_k_frequent(nums: List[int], k: int) -> List[int]:
    """
    Heap approach.
    Time: O(N log K), Space: O(N)
    """
    # Count frequencies
    count = Counter(nums)
    
    # Min heap of size k: (frequency, element)
    heap = []
    
    for num, freq in count.items():
        heapq.heappush(heap, (freq, num))
        if len(heap) > k:
            heapq.heappop(heap)
    
    return [num for freq, num in heap]


# Alternative: Using heapq.nlargest
def top_k_frequent_builtin(nums: List[int], k: int) -> List[int]:
    """
    Using Python's built-in function.
    Time: O(N log K), Space: O(N)
    """
    count = Counter(nums)
    return [num for num, freq in count.most_common(k)]


# Alternative: Bucket sort (optimal)
def top_k_frequent_bucket(nums: List[int], k: int) -> List[int]:
    """
    Bucket sort approach.
    Time: O(N), Space: O(N)
    """
    # Count frequencies
    count = Counter(nums)
    
    # Create buckets: index = frequency
    buckets = [[] for _ in range(len(nums) + 1)]
    
    for num, freq in count.items():
        buckets[freq].append(num)
    
    # Collect k most frequent
    result = []
    for freq in range(len(buckets) - 1, -1, -1):
        for num in buckets[freq]:
            result.append(num)
            if len(result) == k:
                return result
    
    return result


# Alternative: Max heap (all elements)
def top_k_frequent_max_heap(nums: List[int], k: int) -> List[int]:
    """
    Max heap with all elements.
    Time: O(N log N), Space: O(N)
    """
    count = Counter(nums)
    
    # Max heap: negate frequencies
    heap = [(-freq, num) for num, freq in count.items()]
    heapq.heapify(heap)
    
    # Extract k most frequent
    return [heapq.heappop(heap)[1] for _ in range(k)]`,
    timeComplexity: 'O(N log K) for heap, O(N) for bucket sort',
    spaceComplexity: 'O(N)',
    order: 2,
    topic: 'Heap / Priority Queue',
    leetcodeUrl: 'https://leetcode.com/problems/top-k-frequent-elements/',
    youtubeUrl: 'https://www.youtube.com/watch?v=YPTqKIgVk-k',
  },
  {
    id: 'find-median',
    title: 'Find Median from Data Stream',
    difficulty: 'Hard',
    description: `The **median** is the middle value in an ordered integer list. If the size of the list is even, there is no middle value, and the median is the mean of the two middle values.

Implement the \`MedianFinder\` class:
- \`MedianFinder()\` initializes the \`MedianFinder\` object.
- \`void addNum(int num)\` adds the integer \`num\` from the data stream to the data structure.
- \`double findMedian()\` returns the median of all elements so far.

**LeetCode:** [295. Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/)
**YouTube:** [NeetCode - Find Median from Data Stream](https://www.youtube.com/watch?v=itmhHWaHupI)

**Approach:**
Use two heaps:
- **Max heap** (left half): stores smaller values
- **Min heap** (right half): stores larger values

**Invariants:**
1. Max heap size = Min heap size OR Max heap size = Min heap size + 1
2. All elements in max heap ≤ all elements in min heap
3. Median is either top of max heap or average of both tops

**Example:**
\`\`\`
Numbers: [5, 15, 1, 3]

After 5:  max[5] min[]        → median = 5
After 15: max[5] min[15]      → median = (5+15)/2 = 10
After 1:  max[5,1] min[15]    → median = 5
After 3:  max[3,1] min[5,15]  → median = (3+5)/2 = 4
\`\`\``,
    examples: [
      {
        input:
          '["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]\n[[], [1], [2], [], [3], []]',
        output: '[null, null, null, 1.5, null, 2.0]',
        explanation: `MedianFinder medianFinder = new MedianFinder();
medianFinder.addNum(1);    // arr = [1]
medianFinder.addNum(2);    // arr = [1, 2]
medianFinder.findMedian(); // return 1.5 (i.e., (1 + 2) / 2)
medianFinder.addNum(3);    // arr[1, 2, 3]
medianFinder.findMedian(); // return 2.0`,
      },
    ],
    constraints: [
      '-10^5 <= num <= 10^5',
      'There will be at least one element in the data structure before calling findMedian',
      'At most 5 * 10^4 calls will be made to addNum and findMedian',
    ],
    hints: [
      'Use two heaps: max heap for smaller half, min heap for larger half',
      'Maintain invariant: max_heap.size() = min_heap.size() or max_heap.size() = min_heap.size() + 1',
      'Ensure all elements in max heap ≤ all elements in min heap',
      'If heaps have equal size, median is average of both tops',
      'If max heap is larger, median is its top',
      'Python: use negation for max heap since heapq is min heap by default',
    ],
    starterCode: `import heapq

class MedianFinder:
    """
    Maintain median of a stream of numbers.
    """
    
    def __init__(self):
        """Initialize the data structure."""
        # Write your code here
        pass

    def addNum(self, num: int) -> None:
        """Add a number to the data structure."""
        # Write your code here
        pass

    def findMedian(self) -> float:
        """Return the median of all elements."""
        # Write your code here
        pass
`,
    testCases: [
      {
        input: [
          [
            'MedianFinder',
            'addNum',
            'addNum',
            'findMedian',
            'addNum',
            'findMedian',
          ],
          [[], [1], [2], [], [3], []],
        ],
        expected: [null, null, null, 1.5, null, 2.0],
      },
      {
        input: [
          ['MedianFinder', 'addNum', 'findMedian', 'addNum', 'findMedian'],
          [[], [1], [], [2], []],
        ],
        expected: [null, null, 1.0, null, 1.5],
      },
    ],
    solution: `import heapq


class MedianFinder:
    """
    Two heaps approach.
    Time: O(log N) for addNum, O(1) for findMedian
    Space: O(N)
    """
    
    def __init__(self):
        # Max heap for smaller half (negate for max heap)
        self.small = []
        # Min heap for larger half
        self.large = []
    
    def addNum(self, num: int) -> None:
        # Add to appropriate heap
        if not self.small or num <= -self.small[0]:
            heapq.heappush(self.small, -num)
        else:
            heapq.heappush(self.large, num)
        
        # Rebalance heaps
        # small should have same size or 1 more than large
        if len(self.small) > len(self.large) + 1:
            val = -heapq.heappop(self.small)
            heapq.heappush(self.large, val)
        elif len(self.large) > len(self.small):
            val = heapq.heappop(self.large)
            heapq.heappush(self.small, -val)
    
    def findMedian(self) -> float:
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2


# Alternative: Sorted list (simpler but less efficient)
class MedianFinderSorted:
    """
    Sorted list approach.
    Time: O(N) for addNum (insertion), O(1) for findMedian
    Space: O(N)
    """
    
    def __init__(self):
        self.nums = []
    
    def addNum(self, num: int) -> None:
        # Binary search for insertion position
        left, right = 0, len(self.nums)
        while left < right:
            mid = (left + right) // 2
            if self.nums[mid] < num:
                left = mid + 1
            else:
                right = mid
        self.nums.insert(left, num)
    
    def findMedian(self) -> float:
        n = len(self.nums)
        if n % 2 == 1:
            return self.nums[n // 2]
        return (self.nums[n // 2 - 1] + self.nums[n // 2]) / 2


# Usage example:
# median_finder = MedianFinder()
# median_finder.addNum(1)
# median_finder.addNum(2)
# print(median_finder.findMedian())  # 1.5
# median_finder.addNum(3)
# print(median_finder.findMedian())  # 2.0`,
    timeComplexity: 'O(log N) for addNum, O(1) for findMedian',
    spaceComplexity: 'O(N)',
    order: 3,
    topic: 'Heap / Priority Queue',
    leetcodeUrl: 'https://leetcode.com/problems/find-median-from-data-stream/',
    youtubeUrl: 'https://www.youtube.com/watch?v=itmhHWaHupI',
  },
];
