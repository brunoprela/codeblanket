/**
 * Word Ladder
 * Problem ID: word-ladder
 * Order: 6
 */

import { Problem } from '../../../types';

export const word_ladderProblem: Problem = {
  id: 'word-ladder',
  title: 'Word Ladder',
  difficulty: 'Medium',
  topic: 'Breadth-First Search (BFS)',
  description: `A **transformation sequence** from word \`beginWord\` to word \`endWord\` using a dictionary \`wordList\` is a sequence of words \`beginWord -> s1 -> s2 -> ... -> sk\` such that:

- Every adjacent pair of words differs by a single letter.
- Every \`si\` for \`1 <= i <= k\` is in \`wordList\`. Note that \`beginWord\` does not need to be in \`wordList\`.
- \`sk == endWord\`

Given two words, \`beginWord\` and \`endWord\`, and a dictionary \`wordList\`, return the **number of words** in the **shortest transformation sequence** from \`beginWord\` to \`endWord\`, or \`0\` if no such sequence exists.`,
  examples: [
    {
      input:
        'beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]',
      output: '5',
      explanation:
        'One shortest transformation is "hit" -> "hot" -> "dot" -> "dog" -> "cog"',
    },
    {
      input:
        'beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log"]',
      output: '0',
      explanation: 'endWord "cog" is not in wordList',
    },
  ],
  constraints: [
    '1 <= beginWord.length <= 10',
    'endWord.length == beginWord.length',
    '1 <= wordList.length <= 5000',
    'wordList[i].length == beginWord.length',
    'beginWord, endWord, and wordList[i] consist of lowercase English letters',
    'beginWord != endWord',
    'All strings in wordList are unique',
  ],
  hints: [
    'Use BFS for shortest path',
    'Try changing each letter',
    'Use set for O(1) word lookup',
  ],
  starterCode: `from typing import List
from collections import deque

def ladder_length(begin_word: str, end_word: str, word_list: List[str]) -> int:
    """
    Find shortest transformation sequence length.
    
    Args:
        begin_word: Starting word
        end_word: Target word
        word_list: Dictionary of valid words
        
    Returns:
        Length of shortest sequence
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: ['hit', 'cog', ['hot', 'dot', 'dog', 'lot', 'log', 'cog']],
      expected: 5,
    },
    {
      input: ['hit', 'cog', ['hot', 'dot', 'dog', 'lot', 'log']],
      expected: 0,
    },
  ],
  timeComplexity: 'O(M^2 * N) where M is word length, N is word list size',
  spaceComplexity: 'O(M * N)',
  leetcodeUrl: 'https://leetcode.com/problems/word-ladder/',
  youtubeUrl: 'https://www.youtube.com/watch?v=h9iTnkgv05E',
};
