/**
 * Quiz questions for Problem-Solving Patterns section
 */

export const patternsQuiz = [
  {
    id: 'q1',
    question:
      'Describe the frequency counting pattern. Give me an example problem and walk through the solution.',
    sampleAnswer:
      'Frequency counting is where you use a hash map to count how many times each element appears. For example, finding the most common element in an array. I iterate through the array once, and for each element, I increment its count in the hash map. After one pass, I find the key with the maximum value. This is O(n) time and O(k) space where k is unique elements. It appears in problems like "first unique character", "most frequent element", "valid anagram" where you compare counts. The pattern is: iterate once to build frequency map, then use the map to answer the question. Hash maps make counting instant instead of repeatedly scanning the array.',
    keyPoints: [
      'Hash map: element → count',
      'One pass to build frequency map',
      'Use map to answer question',
      'Time: O(n), Space: O(k)',
      'Examples: unique char, most frequent, anagrams',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain the complement pattern used in two sum. How does the hash map store information as you go?',
    sampleAnswer:
      'In two sum with the complement pattern, as I iterate through the array, for each number I calculate what its complement would be (target minus current number). I check if that complement exists in my hash map. If yes, I found the pair. If no, I add the current number and its index to the hash map and continue. The key insight is the hash map stores numbers I have already seen, so when I check for a complement, I am asking "have I seen the other half of this pair already?". This turns checking all pairs O(n²) into a single pass O(n). The hash map accumulates information as we go, remembering past elements for future lookups.',
    keyPoints: [
      'For each number, calculate complement (target - current)',
      'Check if complement in hash map',
      'If yes: found pair. If no: add current to map',
      'Hash map remembers seen elements',
      'Single pass: O(n) instead of O(n²)',
    ],
  },
  {
    id: 'q3',
    question:
      'Talk about the grouping pattern. How do you choose a good key for the hash map?',
    sampleAnswer:
      'The grouping pattern uses a hash map where each key represents a category and the value is a list of items in that category. The critical decision is choosing the right key. For group anagrams, the key is sorted letters or character counts - all anagrams map to the same key. For grouping by sum, the key is the sum value. The key should capture the essential property that defines the group. It must be: hashable (can be a dictionary key), consistent (same input always gives same key), and distinctive (different groups have different keys). Good keys are strings, tuples, or numbers. Lists are not hashable so convert to tuples. A well-chosen key makes the grouping automatic.',
    keyPoints: [
      'Key represents the category/group',
      'Value is list of items in that group',
      'Key must be hashable, consistent, distinctive',
      'Examples: sorted letters for anagrams, sum for grouping by sum',
      'Lists not hashable - use tuples',
    ],
  },
];
