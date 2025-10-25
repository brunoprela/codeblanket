/**
 * Quiz questions for Non-Comparison Sorting Algorithms section
 */

export const noncomparisonsortsQuiz = [
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
      "I would choose bucket sort when the data is uniformly distributed across a known range. Bucket sort works by dividing the range into buckets and distributing elements into those buckets, assuming they will spread out evenly. If the data is uniform, each bucket gets roughly n/k elements, and sorting each bucket takes O(n/k × log (n/k)) time, which averages out to O(n) overall. This is better than quicksort's O(n log n). However, if the distribution is skewed and all elements fall into a few buckets, bucket sort degrades to O(n²). So I would use bucket sort for things like sorting random floating-point numbers between 0 and 1, or sorting uniformly distributed sensor data. For general-purpose sorting without knowing the distribution, quicksort is safer.",
    keyPoints: [
      'Choose bucket sort for uniformly distributed data',
      'Uniform distribution → elements spread evenly across buckets',
      "Achieves O(n) average time vs quicksort's O(n log n)",
      'Degrades to O(n²) if data is skewed',
      'Good for: random floats, uniform data; Bad for: unknown distribution',
    ],
  },
];
