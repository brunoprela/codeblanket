/**
 * Quiz questions for Code Templates & Patterns section
 */

export const templatesQuiz = [
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
];
