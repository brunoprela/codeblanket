/**
 * Replace Words
 * Problem ID: replace-words
 * Order: 6
 */

import { Problem } from '../../../types';

export const replace_wordsProblem: Problem = {
  id: 'replace-words',
  title: 'Replace Words',
  difficulty: 'Easy',
  topic: 'Tries',
  description: `In English, we have a concept called **root**, which can be followed by some other word to form another longer word - let us call this word **derivative**. For example, when the **root** \`"help"\` is followed by the word \`"ful"\`, we can form a derivative \`"helpful"\`.

Given a \`dictionary\` consisting of many **roots** and a \`sentence\` consisting of words separated by spaces, replace all the derivatives in the sentence with the **root** forming it. If a derivative can be replaced by more than one **root**, replace it with the **root** that has **the shortest length**.

Return the \`sentence\` after the replacement.`,
  examples: [
    {
      input:
        'dictionary = ["cat","bat","rat"], sentence = "the cattle was rattled by the battery"',
      output: '"the cat was rat by the bat"',
    },
    {
      input:
        'dictionary = ["a","b","c"], sentence = "aadsfasf absbs bbab cadsfafs"',
      output: '"a a b c"',
    },
  ],
  constraints: [
    '1 <= dictionary.length <= 1000',
    '1 <= dictionary[i].length <= 100',
    'dictionary[i] consists of only lower-case letters',
    '1 <= sentence.length <= 10^6',
    'sentence consists of only lower-case letters and spaces',
    'The number of words in sentence is in the range [1, 1000]',
    'The length of each word in sentence is in the range [1, 1000]',
    'Every two consecutive words in sentence will be separated by exactly one space',
    'sentence does not have leading or trailing spaces',
  ],
  hints: [
    'Build trie from all roots',
    'For each word, find shortest root prefix',
  ],
  starterCode: `from typing import List

def replace_words(dictionary: List[str], sentence: str) -> str:
    """
    Replace words with shortest root.
    
    Args:
        dictionary: List of roots
        sentence: Input sentence
        
    Returns:
        Sentence with words replaced
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [['cat', 'bat', 'rat'], 'the cattle was rattled by the battery'],
      expected: 'the cat was rat by the bat',
    },
    {
      input: [['a', 'b', 'c'], 'aadsfasf absbs bbab cadsfafs'],
      expected: 'a a b c',
    },
  ],
  timeComplexity:
    'O(N + M) where N is total chars in dictionary, M in sentence',
  spaceComplexity: 'O(N)',
  leetcodeUrl: 'https://leetcode.com/problems/replace-words/',
  youtubeUrl: 'https://www.youtube.com/watch?v=pJJ17U_Qiw8',
};
