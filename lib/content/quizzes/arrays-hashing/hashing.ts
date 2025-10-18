/**
 * Quiz questions for Hash Tables: Fast Lookups & Storage section
 */

export const hashingQuiz = [
  {
    id: 'q1',
    question:
      'Explain how hash tables achieve O(1) average-case lookup. What is a hash function and why does it matter?',
    sampleAnswer:
      'A hash table uses a hash function to convert keys into array indices. The hash function takes your key and outputs a number, which we modulo by the table size to get an index. Good hash functions distribute keys evenly across the array. So when I want to look up a key, I hash it to get the index and go directly there - O(1). The "average case" qualifier is because of collisions - when two keys hash to the same index. Most hash tables handle this with chaining or open addressing. With a good hash function and proper load factor, collisions are rare, so we get O(1) on average. A bad hash function causes many collisions and degrades to O(n) worst case.',
    keyPoints: [
      'Hash function: key → number → array index',
      'Direct index access: O(1)',
      'Good hash function distributes keys evenly',
      'Collisions handled by chaining or probing',
      'Average O(1), worst O(n) with bad function',
    ],
  },
  {
    id: 'q2',
    question:
      'Walk me through how you would use a hash map to find the first non-repeating character in a string.',
    sampleAnswer:
      'I would make two passes. First pass: iterate through the string and use a hash map to count how many times each character appears. Key is the character, value is the count. This is O(n). Second pass: iterate through the string again (same order) and for each character, check its count in the hash map. The first character with count equals one is my answer. Return it immediately. This is also O(n). Total O(n) time with O(k) space where k is the number of unique characters, at most 26 for lowercase letters. The hash map lets me instantly check counts instead of scanning the string repeatedly for each character.',
    keyPoints: [
      'First pass: count frequency in hash map',
      'Second pass: check counts in original order',
      'Return first character with count 1',
      'Time: O(n), Space: O(k) where k ≤ 26',
      'Avoids repeated scanning',
    ],
  },
  {
    id: 'q3',
    question:
      'Describe the group anagrams problem. How do you use hashing to solve it efficiently?',
    sampleAnswer:
      'Group anagrams means grouping words that contain the same letters in different orders, like "eat", "tea", "ate" are all anagrams. I use a hash map where the key represents the signature of the anagram group. The signature could be the sorted letters, so "eat", "tea", "ate" all become "aet" when sorted. Or I could use character counts like "e1t1a1". All anagrams have the same signature. I iterate through words once, compute each word signature, and append the word to the hash map under that signature key. At the end, the hash map values are the grouped anagrams. This is O(n×m log m) if sorting, or O(n×m) with counting, where n is word count and m is max word length.',
    keyPoints: [
      'Anagrams have same letters, different order',
      'Key: sorted letters or character count signature',
      'Hash map: signature → list of words',
      'One pass: compute signature, append to group',
      'Time: O(n×m log m) sorting or O(n×m) counting',
    ],
  },
];
