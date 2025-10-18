/**
 * Quiz questions for Common Pitfalls & How to Avoid Them section
 */

export const commonmistakesQuiz = [
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
];
