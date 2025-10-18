/**
 * Quiz questions for Problem-Solving Strategy & Interview Tips section
 */

export const problemsolvingQuiz = [
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
];
