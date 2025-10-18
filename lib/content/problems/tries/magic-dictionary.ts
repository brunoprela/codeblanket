/**
 * Implement Magic Dictionary
 * Problem ID: magic-dictionary
 * Order: 5
 */

import { Problem } from '../../../types';

export const magic_dictionaryProblem: Problem = {
  id: 'magic-dictionary',
  title: 'Implement Magic Dictionary',
  difficulty: 'Easy',
  topic: 'Tries',
  description: `Design a data structure that is initialized with a list of **different** words. Provided a string, you should determine if you can change exactly one character in this string to match any word in the data structure.

Implement the \`MagicDictionary\` class:
- \`MagicDictionary()\` Initializes the object.
- \`void buildDict(String[] dictionary)\` Sets the data structure with an array of distinct strings \`dictionary\`.
- \`bool search(String searchWord)\` Returns \`true\` if you can change **exactly one character** in \`searchWord\` to match any string in the data structure, otherwise returns \`false\`.`,
  examples: [
    {
      input:
        '["MagicDictionary", "buildDict", "search", "search", "search", "search"], [[], [["hello", "leetcode"]], ["hello"], ["hhllo"], ["hell"], ["leetcoded"]]',
      output: '[null, null, false, true, false, false]',
    },
  ],
  constraints: [
    '1 <= dictionary.length <= 100',
    '1 <= dictionary[i].length <= 100',
    'dictionary[i] consists of only lower-case English letters',
    'All the strings in dictionary are distinct',
    '1 <= searchWord.length <= 100',
    'searchWord consists of only lower-case English letters',
    'buildDict will be called only once before search',
    'At most 100 calls will be made to search',
  ],
  hints: [
    'For each position, try changing character',
    'Check if resulting word exists in trie',
  ],
  starterCode: `from typing import List

class MagicDictionary:
    def __init__(self):
        """Initialize dictionary."""
        # Write your code here
        pass
    
    def build_dict(self, dictionary: List[str]) -> None:
        """Build dictionary from word list."""
        # Write your code here
        pass
    
    def search(self, search_word: str) -> bool:
        """Search for word with one character change."""
        # Write your code here
        pass
`,
  testCases: [
    {
      input: [[['hello', 'leetcode']], 'hello', 'hhllo', 'hell', 'leetcoded'],
      expected: [null, false, true, false, false],
    },
  ],
  timeComplexity: 'O(m * 26) for search where m is word length',
  spaceComplexity: 'O(n * m) for n words',
  leetcodeUrl: 'https://leetcode.com/problems/implement-magic-dictionary/',
  youtubeUrl: 'https://www.youtube.com/watch?v=9HHuFBx2T_k',
};
