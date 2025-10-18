/**
 * Quiz questions for Interview Strategy & Tips section
 */

export const interviewQuiz = [
  {
    id: 'q1',
    question:
      'How do you recognize in an interview that a problem can be solved with hash tables? What are the key signals?',
    sampleAnswer:
      'Several signals tell me to consider hash tables. First, if I am thinking about nested loops to check all pairs, that is a red flag - can I use a hash table to remember elements and look them up instead? Second, keywords like "count", "frequency", "group", "unique", "duplicate", "exists" all scream hash table. Third, if I need fast lookup or existence checking and I am okay using extra memory, hash table is perfect. Fourth, if the problem involves finding complements or pairs that satisfy a relationship, like two sum. The fundamental question: am I repeatedly searching for elements? If yes, hash table probably helps by giving O(1) lookups.',
    keyPoints: [
      'Alternative to nested loops',
      'Keywords: count, frequency, group, unique, duplicate',
      'Fast lookup/existence checking needed',
      'Finding pairs/complements',
      'Repeatedly searching for elements',
    ],
  },
  {
    id: 'q2',
    question:
      'Walk me through your approach to an array/hash table problem in an interview, from reading the problem to explaining your solution.',
    sampleAnswer:
      'First, I would clarify: can the array have duplicates? Any constraints on values? Can it be empty? Then I explain my thinking: "I notice I need to find pairs/count frequencies, so I am thinking hash table for O(1) lookups". I state the approach: one pass to build hash map, then use it to answer. I mention complexity: O(n) time, O(k) space where k is unique elements. I write the code carefully, explaining as I go. After coding, I trace through an example: "for [2, 7, 11, 15] with target 9, first iteration checks for 7 in map, not there, adds 2..." Finally, I mention edge cases I am handling: empty array, no solution, all same values. The key is clear communication throughout.',
    keyPoints: [
      'Clarify: duplicates? constraints? empty?',
      'Explain thinking: why hash table?',
      'State approach and complexity upfront',
      'Code carefully with explanations',
      'Trace through example',
      'Mention edge cases',
    ],
  },
  {
    id: 'q3',
    question:
      'What are the most common mistakes people make with hash table problems? How do you avoid them?',
    sampleAnswer:
      'First mistake: not handling collisions or understanding that hash table operations are average case O(1), not guaranteed. I avoid this by saying "O(1) average case" explicitly. Second: using unhashable keys like lists instead of tuples - I always check what I am using as keys. Third: not considering space complexity - I mention that hash tables use O(n) extra space. Fourth: in problems like two sum, accidentally using the same element twice by not checking indices. Fifth: not handling edge cases like empty input or no solution. I avoid these by being deliberate about what I store, how I query it, and testing edge cases. Communication is key - explain what goes in the hash table and why.',
    keyPoints: [
      'Remember O(1) is average case, not guaranteed',
      'Use hashable keys (tuples, not lists)',
      'Consider O(n) space complexity',
      'Check indices to avoid using same element twice',
      'Test edge cases: empty, no solution',
    ],
  },
];
