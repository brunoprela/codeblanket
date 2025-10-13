import { Problem } from '../types';

export const heapProblems: Problem[] = [
  {
    id: 'kth-largest-element',
    title: 'Kth Largest Element in an Array',
    difficulty: 'Easy',
    description: `Given an integer array \`nums\` and an integer \`k\`, return **the kth largest element** in the array.

Note that it is the kth largest element in the sorted order, not the kth distinct element.

Can you solve it without sorting?


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

  // EASY - Last Stone Weight
  {
    id: 'last-stone-weight',
    title: 'Last Stone Weight',
    difficulty: 'Easy',
    topic: 'Heap / Priority Queue',
    description: `You are given an array of integers \`stones\` where \`stones[i]\` is the weight of the \`i-th\` stone.

We are playing a game with the stones. On each turn, we choose the **heaviest two stones** and smash them together. Suppose the heaviest two stones have weights \`x\` and \`y\` with \`x <= y\`. The result of this smash is:

- If \`x == y\`, both stones are destroyed, and
- If \`x != y\`, the stone of weight \`x\` is destroyed, and the stone of weight \`y\` has new weight \`y - x\`.

At the end of the game, there is **at most one** stone left.

Return the weight of the last remaining stone. If there are no stones left, return \`0\`.`,
    examples: [
      {
        input: 'stones = [2,7,4,1,8,1]',
        output: '1',
        explanation:
          'Combine 7 and 8 to get 1, then 4 and 1 to get 3, then 2 and 3 to get 1, then 1 and 1 to get 0, return 1.',
      },
      {
        input: 'stones = [1]',
        output: '1',
      },
    ],
    constraints: ['1 <= stones.length <= 30', '1 <= stones[i] <= 1000'],
    hints: [
      'Use a max heap',
      'Pop two largest stones, push difference if not equal',
    ],
    starterCode: `from typing import List
import heapq

def last_stone_weight(stones: List[int]) -> int:
    """
    Find weight of last remaining stone.
    
    Args:
        stones: Array of stone weights
        
    Returns:
        Weight of last stone or 0
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[2, 7, 4, 1, 8, 1]],
        expected: 1,
      },
      {
        input: [[1]],
        expected: 1,
      },
      {
        input: [[2, 2]],
        expected: 0,
      },
    ],
    timeComplexity: 'O(n log n)',
    spaceComplexity: 'O(n)',
    leetcodeUrl: 'https://leetcode.com/problems/last-stone-weight/',
    youtubeUrl: 'https://www.youtube.com/watch?v=B-QCq79-Vfw',
  },

  // EASY - Kth Largest Element in a Stream
  {
    id: 'kth-largest-element-stream',
    title: 'Kth Largest Element in a Stream',
    difficulty: 'Easy',
    topic: 'Heap / Priority Queue',
    description: `Design a class to find the \`k-th\` largest element in a stream. Note that it is the \`k-th\` largest element in the sorted order, not the \`k-th\` distinct element.

Implement \`KthLargest\` class:
- \`KthLargest(int k, int[] nums)\` Initializes the object with the integer \`k\` and the stream of integers \`nums\`.
- \`int add(int val)\` Appends the integer \`val\` to the stream and returns the element representing the \`k-th\` largest element in the stream.`,
    examples: [
      {
        input:
          '["KthLargest", "add", "add", "add", "add", "add"], [[3, [4, 5, 8, 2]], [3], [5], [10], [9], [4]]',
        output: '[null, 4, 5, 5, 8, 8]',
      },
    ],
    constraints: [
      '1 <= k <= 10^4',
      '0 <= nums.length <= 10^4',
      '-10^4 <= nums[i] <= 10^4',
      '-10^4 <= val <= 10^4',
      'At most 10^4 calls will be made to add',
    ],
    hints: [
      'Use a min heap of size k',
      'Always maintain k largest elements',
      'Top of heap is kth largest',
    ],
    starterCode: `from typing import List
import heapq

class KthLargest:
    def __init__(self, k: int, nums: List[int]):
        """
        Initialize with k and initial stream.
        
        Args:
            k: Position of element to track
            nums: Initial stream
        """
        # Write your code here
        pass
    
    def add(self, val: int) -> int:
        """
        Add value to stream and return kth largest.
        
        Args:
            val: Value to add
            
        Returns:
            Kth largest element
        """
        # Write your code here
        pass
`,
    testCases: [
      {
        input: [[3, [4, 5, 8, 2]], [3], [5], [10], [9], [4]],
        expected: [null, 4, 5, 5, 8, 8],
      },
    ],
    timeComplexity: 'O(log k) for add',
    spaceComplexity: 'O(k)',
    leetcodeUrl:
      'https://leetcode.com/problems/kth-largest-element-in-a-stream/',
    youtubeUrl: 'https://www.youtube.com/watch?v=hOjcdrqMoQ8',
  },

  // EASY - Relative Ranks
  {
    id: 'relative-ranks',
    title: 'Relative Ranks',
    difficulty: 'Easy',
    topic: 'Heap / Priority Queue',
    description: `You are given an integer array \`score\` of size \`n\`, where \`score[i]\` is the score of the \`i-th\` athlete in a competition. All the scores are guaranteed to be **unique**.

The athletes are **placed** based on their scores, where the \`1st\` place athlete has the highest score, the \`2nd\` place athlete has the \`2nd\` highest score, and so on. The placement of each athlete determines their rank:

- The \`1st\` place athlete's rank is \`"Gold Medal"\`.
- The \`2nd\` place athlete's rank is \`"Silver Medal"\`.
- The \`3rd\` place athlete's rank is \`"Bronze Medal"\`.
- For the \`4th\` place to the \`n-th\` place athlete, their rank is their placement number (i.e., the \`x-th\` place athlete's rank is \`"x"\`).

Return an array \`answer\` of size \`n\` where \`answer[i]\` is the **rank** of the \`i-th\` athlete.`,
    examples: [
      {
        input: 'score = [5,4,3,2,1]',
        output: '["Gold Medal","Silver Medal","Bronze Medal","4","5"]',
      },
      {
        input: 'score = [10,3,8,9,4]',
        output: '["Gold Medal","5","Bronze Medal","Silver Medal","4"]',
      },
    ],
    constraints: [
      'n == score.length',
      '1 <= n <= 10^4',
      '0 <= score[i] <= 10^6',
      'All the values in score are unique',
    ],
    hints: [
      'Create pairs of (score, index)',
      'Sort by score descending',
      'Assign ranks based on sorted order',
    ],
    starterCode: `from typing import List

def find_relative_ranks(score: List[int]) -> List[str]:
    """
    Assign ranks to athletes based on scores.
    
    Args:
        score: Array of athlete scores
        
    Returns:
        Array of rank strings
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[5, 4, 3, 2, 1]],
        expected: ['Gold Medal', 'Silver Medal', 'Bronze Medal', '4', '5'],
      },
      {
        input: [[10, 3, 8, 9, 4]],
        expected: ['Gold Medal', '5', 'Bronze Medal', 'Silver Medal', '4'],
      },
    ],
    timeComplexity: 'O(n log n)',
    spaceComplexity: 'O(n)',
    leetcodeUrl: 'https://leetcode.com/problems/relative-ranks/',
    youtubeUrl: 'https://www.youtube.com/watch?v=qFKI9TKXRIs',
  },

  // MEDIUM - Kth Largest Element in an Array
  {
    id: 'kth-largest-element-array',
    title: 'Kth Largest Element in an Array',
    difficulty: 'Medium',
    topic: 'Heap / Priority Queue',
    description: `Given an integer array \`nums\` and an integer \`k\`, return the \`k-th\` largest element in the array.

Note that it is the \`k-th\` largest element in the sorted order, not the \`k-th\` distinct element.

Can you solve it without sorting?`,
    examples: [
      {
        input: 'nums = [3,2,1,5,6,4], k = 2',
        output: '5',
      },
      {
        input: 'nums = [3,2,3,1,2,4,5,5,6], k = 4',
        output: '4',
      },
    ],
    constraints: ['1 <= k <= nums.length <= 10^5', '-10^4 <= nums[i] <= 10^4'],
    hints: [
      'Use a min heap of size k',
      'Maintain k largest elements',
      'Top of heap is kth largest',
    ],
    starterCode: `from typing import List
import heapq

def find_kth_largest(nums: List[int], k: int) -> int:
    """
    Find kth largest element in array.
    
    Args:
        nums: Input array
        k: Position from largest
        
    Returns:
        Kth largest element
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
    ],
    timeComplexity: 'O(n log k)',
    spaceComplexity: 'O(k)',
    leetcodeUrl:
      'https://leetcode.com/problems/kth-largest-element-in-an-array/',
    youtubeUrl: 'https://www.youtube.com/watch?v=XEmy13g1Qxc',
  },

  // MEDIUM - K Closest Points to Origin
  {
    id: 'k-closest-points',
    title: 'K Closest Points to Origin',
    difficulty: 'Medium',
    topic: 'Heap / Priority Queue',
    description: `Given an array of \`points\` where \`points[i] = [xi, yi]\` represents a point on the X-Y plane and an integer \`k\`, return the \`k\` closest points to the origin \`(0, 0)\`.

The distance between two points on the X-Y plane is the Euclidean distance (i.e., \`√(x1 - x2)² + (y1 - y2)²\`).

You may return the answer in **any order**. The answer is **guaranteed** to be **unique** (except for the order that it is in).`,
    examples: [
      {
        input: 'points = [[1,3],[-2,2]], k = 1',
        output: '[[-2,2]]',
        explanation:
          'Distance to origin: (1,3) = sqrt(10), (-2,2) = sqrt(8). Closest is (-2,2).',
      },
      {
        input: 'points = [[3,3],[5,-1],[-2,4]], k = 2',
        output: '[[3,3],[-2,4]]',
      },
    ],
    constraints: ['1 <= k <= points.length <= 10^4', '-10^4 <= xi, yi <= 10^4'],
    hints: [
      'Calculate distances from origin',
      'Use max heap of size k',
      'Maintain k smallest distances',
    ],
    starterCode: `from typing import List
import heapq

def k_closest(points: List[List[int]], k: int) -> List[List[int]]:
    """
    Find k closest points to origin.
    
    Args:
        points: Array of [x, y] coordinates
        k: Number of closest points
        
    Returns:
        K closest points
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [
          [
            [1, 3],
            [-2, 2],
          ],
          1,
        ],
        expected: [[-2, 2]],
      },
      {
        input: [
          [
            [3, 3],
            [5, -1],
            [-2, 4],
          ],
          2,
        ],
        expected: [
          [3, 3],
          [-2, 4],
        ],
      },
    ],
    timeComplexity: 'O(n log k)',
    spaceComplexity: 'O(k)',
    leetcodeUrl: 'https://leetcode.com/problems/k-closest-points-to-origin/',
    youtubeUrl: 'https://www.youtube.com/watch?v=rI2EBUEMfTk',
  },

  // MEDIUM - Task Scheduler
  {
    id: 'task-scheduler',
    title: 'Task Scheduler',
    difficulty: 'Medium',
    topic: 'Heap / Priority Queue',
    description: `You are given an array of CPU \`tasks\`, each represented by letters A to Z, and a cooling time, \`n\`. Each cycle or interval allows the completion of one task. Tasks can be completed in any order, but there is a constraint: **identical** tasks must be separated by at least \`n\` intervals due to cooling time.

Return the **minimum number of intervals** required to complete all tasks.`,
    examples: [
      {
        input: 'tasks = ["A","A","A","B","B","B"], n = 2',
        output: '8',
        explanation:
          'A -> B -> idle -> A -> B -> idle -> A -> B. Only 8 units of time are needed.',
      },
      {
        input: 'tasks = ["A","C","A","B","D","B"], n = 1',
        output: '6',
        explanation: 'A -> B -> C -> D -> A -> B. Only 6 units needed.',
      },
    ],
    constraints: [
      '1 <= tasks.length <= 10^4',
      'tasks[i] is an uppercase English letter',
      '0 <= n <= 100',
    ],
    hints: [
      'Count task frequencies',
      'Use max heap to always pick most frequent task',
      'Track cooling time for each task',
    ],
    starterCode: `from typing import List
import heapq
from collections import Counter

def least_interval(tasks: List[str], n: int) -> int:
    """
    Find minimum intervals to complete all tasks.
    
    Args:
        tasks: Array of task letters
        n: Cooling time between same tasks
        
    Returns:
        Minimum number of intervals
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [['A', 'A', 'A', 'B', 'B', 'B'], 2],
        expected: 8,
      },
      {
        input: [['A', 'C', 'A', 'B', 'D', 'B'], 1],
        expected: 6,
      },
      {
        input: [['A', 'A', 'A', 'B', 'B', 'B'], 0],
        expected: 6,
      },
    ],
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl: 'https://leetcode.com/problems/task-scheduler/',
    youtubeUrl: 'https://www.youtube.com/watch?v=s8p8ukTyA2I',
  },
];
