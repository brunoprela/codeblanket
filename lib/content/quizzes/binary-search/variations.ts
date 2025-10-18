/**
 * Quiz questions for Advanced Variations & Applications section
 */

export const variationsQuiz = [
  {
    id: 'q1',
    question:
      'Explain what a monotonic function is and why it is important for binary search. Give me an example of how you would use binary search on a monotonic function.',
    hint: 'Monotonic means consistently increasing or decreasing.',
    sampleAnswer:
      'A monotonic function is one that is either always increasing or always decreasing - it never changes direction. This is crucial for binary search because it means we can make reliable decisions. If I check a value in the middle and it is too big, I know everything to the right is also too big. For example, say I want to find the square root of 25 without using sqrt. The function f(x) = x squared is monotonic - as x increases, x squared increases. So I can binary search on the range 0 to 25. I pick mid, square it, and if it is too big, I search left. If too small, I search right. The monotonic property guarantees this works. Any problem where you can check a value and know which direction to go can potentially use binary search.',
    keyPoints: [
      'Monotonic: always increasing or always decreasing',
      'Enables reliable decision-making (too big → search left)',
      'Example: x squared is monotonic increasing',
      'Binary search works on any monotonic property',
      'Key insight: can determine direction from any check',
    ],
  },
  {
    id: 'q2',
    question:
      'Describe a scenario where you might binary search on something that does not look sorted at first glance.',
    sampleAnswer:
      'A good example is searching in a rotated sorted array, like [4,5,6,7,0,1,2]. The array is not fully sorted, but it has structure. It is two sorted pieces stuck together. The trick is that at least one half is always properly sorted. When I compare the middle with the edges, I can figure out which half is sorted, then decide if my target is in the sorted half or the other half. So even though it looks messy, there is still enough order to apply binary search logic. Another example is a bitonic array (goes up then down) - not sorted, but still has monotonic regions you can exploit.',
    keyPoints: [
      'Rotated sorted arrays still have structure',
      'At least one half is always sorted',
      'Bitonic arrays (up then down) work too',
      'Key: find the monotonic property',
    ],
  },
  {
    id: 'q3',
    question:
      'How would you recognize in an interview that a problem might be solvable with binary search, even if it does not mention sorted arrays?',
    sampleAnswer:
      'I look for certain keywords and patterns. Words like "minimum", "maximum", "first", "last", or "at least" are red flags. Also if the problem asks about optimizing something or finding a threshold, that is a hint. The key question I ask myself is: if I try a particular value, can I tell whether I need to go higher or lower? That is the hallmark of binary search. Another pattern is when you are searching within a range of possible answers. If I see these signals, I start thinking about whether there is a monotonic property I can exploit with binary search, even if no array is mentioned.',
    keyPoints: [
      'Look for: first, last, minimum, maximum, at least/most',
      'Ask: can I check if a value is too big/small?',
      'Is there a monotonic property?',
      'Searching in a range of possible answers',
    ],
  },
];
