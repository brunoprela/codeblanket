export default [
  {
    id: 'cp-m1-s1-q1',
    section: 'Building Algorithmic Intuition',
    question:
      'You encounter a problem asking to find the maximum sum of any subarray in an array of integers. What is the most efficient approach?',
    options: [
      'Try all possible subarrays and track the maximum (O(n³) brute force)',
      'Use two nested loops to compute all subarray sums (O(n²))',
      "Use Kadane's algorithm with dynamic programming (O(n))",
      'Sort the array first and take elements from the end (O(n log n))',
    ],
    correctAnswer: 2,
    explanation:
      "Kadane's algorithm solves the maximum subarray sum problem optimally in O(n) time using dynamic programming. The key insight is: at each position, the maximum subarray ending at that position is either the current element alone, or the current element plus the maximum subarray ending at the previous position. Code: `max_ending_here = max(arr[i], max_ending_here + arr[i])`. Brute force approaches work but are inefficient. Sorting destroys the subarray structure and won't give the correct answer.",
    difficulty: 'intermediate',
  },
  {
    id: 'cp-m1-s1-q2',
    section: 'Building Algorithmic Intuition',
    question:
      'Given constraints where n ≤ 10⁵, which time complexity is acceptable for a solution that must run within 1-2 seconds?',
    options: [
      'O(n³) - Always acceptable',
      'O(n²) - Acceptable if constant is small',
      'O(n log n) - Generally safe and optimal',
      'O(2ⁿ) - Fast enough for n ≤ 10⁵',
    ],
    correctAnswer: 2,
    explanation:
      'For n ≤ 10⁵, O(n log n) is generally safe and optimal. This allows roughly 10⁵ × 17 ≈ 1.7 million operations, well within time limits. O(n²) would be 10¹⁰ operations (too slow). O(n³) would be 10¹⁵ operations (way too slow). O(2ⁿ) for n=10⁵ is astronomically large. Rule of thumb: ~10⁸-10⁹ operations per second, so aim for complexity that keeps total operations under 10⁸ for the given constraints.',
    difficulty: 'beginner',
  },
  {
    id: 'cp-m1-s1-q3',
    section: 'Building Algorithmic Intuition',
    question:
      'A problem asks you to determine if a string is a palindrome. What is the most efficient approach?',
    options: [
      'Reverse the string and compare with the original (O(n) time, O(n) space)',
      'Use two pointers from both ends moving toward the center (O(n) time, O(1) space)',
      'Generate all substrings and check each (O(n²) time)',
      'Use a hash map to count character frequencies (O(n) time, O(n) space)',
    ],
    correctAnswer: 1,
    explanation:
      'The two-pointer approach is optimal: O(n) time with O(1) extra space. Start with pointers at the beginning and end, compare characters, and move pointers inward. If any mismatch is found, it\'s not a palindrome. Code: `while(left < right) { if(s[left] != s[right]) return false; left++; right--; }`. Reversing works but uses O(n) extra space. Generating substrings is overkill. Character frequency doesn\'t check order, so won\'t work (e.g., "abc" and "cba" have same frequencies but different structures).',
    difficulty: 'beginner',
  },
  {
    id: 'cp-m1-s1-q4',
    section: 'Building Algorithmic Intuition',
    question:
      'You need to find if a target sum can be formed by adding two numbers from an array. What is the most efficient approach?',
    options: [
      'Use two nested loops to try all pairs (O(n²))',
      'Sort the array and use two pointers (O(n log n))',
      'Use a hash set to track seen numbers (O(n) average)',
      'Use binary search for each element (O(n log n))',
    ],
    correctAnswer: 2,
    explanation:
      'The hash set approach is optimal with O(n) average time complexity. Iterate through the array once: for each element x, check if (target - x) exists in the hash set. If yes, found the pair; if no, add x to the set. Code: `for(int x : arr) { if(set.count(target - x)) return true; set.insert(x); }`. The two-pointer approach after sorting also works (O(n log n)) but is slightly slower. Nested loops are O(n²) which is acceptable but not optimal. Binary search approach is also O(n log n) but more complex to implement.',
    difficulty: 'intermediate',
  },
  {
    id: 'cp-m1-s1-q5',
    section: 'Building Algorithmic Intuition',
    question:
      'When should you consider using a greedy algorithm instead of dynamic programming?',
    options: [
      'When the problem has overlapping subproblems',
      'When locally optimal choices lead to globally optimal solutions',
      'When you need to explore all possible combinations',
      'When the problem requires backtracking',
    ],
    correctAnswer: 1,
    explanation:
      "Greedy algorithms work when locally optimal choices lead to globally optimal solutions (greedy choice property). Examples: activity selection, Huffman coding, Dijkstra's algorithm. The key is proving that making the best local choice at each step will result in the best overall solution. Dynamic programming is needed when there are overlapping subproblems and you need optimal substructure. If you need to explore all combinations or backtrack, greedy won't work. Classic greedy problem: choosing intervals with earliest end time for activity selection. DP problem: 0/1 knapsack (greedy doesn't work because local choice doesn't guarantee global optimum).",
    difficulty: 'intermediate',
  },
] as const;
