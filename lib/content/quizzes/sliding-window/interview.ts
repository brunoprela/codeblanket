/**
 * Quiz questions for Interview Strategy section
 */

export const interviewQuiz = [
  {
    id: 'q1',
    question:
      'How do you recognize in an interview that a problem needs sliding window? What keywords or patterns signal this?',
    sampleAnswer:
      'Several signals tell me sliding window. First, keywords: "contiguous", "subarray", "substring", "consecutive" - these scream window. Second, optimization terms: "longest", "shortest", "maximum", "minimum" with constraints - we are seeking optimal window. Third, constraints like "at most k", "at least k" distinct elements - these are window validity conditions. Fourth, if brute force would check all subarrays O(n²) - sliding window likely optimizes to O(n). Fifth, two-pointer vibes with sequential processing. The key question: am I looking for something in a contiguous sequence that can be computed incrementally? If yes, sliding window is probably the answer.',
    keyPoints: [
      'Keywords: contiguous, subarray, substring, consecutive',
      'Optimization: longest, shortest, maximum, minimum',
      'Constraints: at most k, at least k',
      'Brute force: O(n²) checking all subarrays',
      'Incremental computation on contiguous sequence',
    ],
  },
  {
    id: 'q2',
    question:
      'Walk me through explaining a sliding window solution in an interview, from problem recognition to complexity analysis.',
    sampleAnswer:
      'First, I identify: "This is a longest substring problem with constraints, so I am thinking sliding window". Then I explain window type: "Variable-size window - I will expand right to include characters and shrink left when constraints break". I describe the state: "I will track character frequency using a hash map". Then the algorithm: "Expand right each iteration, update frequency. When I have duplicates, shrink left until removing the duplicate. Track maximum window size." I code while explaining each part. After coding, I trace an example: "For \'abcabcbb\', right moves to \'abc\' len 3, hits duplicate \'a\'...". Finally complexity: "O(n) time as each element enters and leaves once, O(k) space for hash map where k is alphabet size." Communication throughout is key.',
    keyPoints: [
      'Identify problem type and pattern',
      'Explain window type and expansion/shrinking logic',
      'Describe state tracking',
      'Explain algorithm step-by-step while coding',
      'Trace example',
      'State time and space complexity',
    ],
  },
  {
    id: 'q3',
    question:
      'What are the most common mistakes in sliding window problems and how do you avoid them?',
    sampleAnswer:
      'First mistake: off-by-one in window size calculation - use right - left + 1, not right - left. I verify with simple case: if left equals right, size is 1. Second: processing order - must add arr[right] before checking conditions, or we check old window. Third: update answer placement - outside while for maximum, inside while for minimum. I remember: shrink invalid for max, shrink valid for min. Fourth: forgetting to update state when moving left - must remove arr[left] from tracking. Fifth: manually incrementing right in for loop - let the for loop handle it. Sixth: not initializing properly - hash map, variables. I avoid these by using templates and testing edge cases early.',
    keyPoints: [
      'Off-by-one: use right - left + 1',
      'Order: add right before checking',
      'Update placement: depends on max vs min',
      'State: update when moving left',
      'Do not manually increment right in for loop',
      'Use templates and test edge cases',
    ],
  },
];
