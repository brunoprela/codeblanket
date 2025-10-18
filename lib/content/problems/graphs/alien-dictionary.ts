/**
 * Alien Dictionary
 * Problem ID: alien-dictionary
 * Order: 10
 */

import { Problem } from '../../../types';

export const alien_dictionaryProblem: Problem = {
  id: 'alien-dictionary',
  title: 'Alien Dictionary',
  difficulty: 'Hard',
  topic: 'Graphs',
  description: `There is a new alien language that uses the English alphabet. However, the order among the letters is unknown to you.

You are given a list of strings \`words\` from the alien language's dictionary, where the strings in \`words\` are **sorted lexicographically** by the rules of this new language.

Return a string of the unique letters in the new alien language sorted in **lexicographically increasing order** by the new language's rules. If there is no solution, return \`""\`. If there are multiple solutions, return **any of them**.`,
  examples: [
    {
      input: 'words = ["wrt","wrf","er","ett","rftt"]',
      output: '"wertf"',
      explanation:
        'From "wrt" and "wrf", we know \'t\' < \'f\'. From "wrt" and "er", we know \'w\' < \'e\'. From "er" and "ett", we know \'r\' < \'t\'. From "ett" and "rftt", we know \'e\' < \'r\'. Combining these gives one valid order: "wertf".',
    },
    {
      input: 'words = ["z","x"]',
      output: '"zx"',
      explanation: 'From "z" and "x", we know \'z\' < \'x\'.',
    },
    {
      input: 'words = ["z","x","z"]',
      output: '""',
      explanation: 'The order is invalid, so return "".',
    },
  ],
  constraints: [
    '1 <= words.length <= 100',
    '1 <= words[i].length <= 100',
    'words[i] consists of only lowercase English letters',
  ],
  hints: [
    'Build a graph from the word ordering',
    'Compare adjacent words to find character ordering',
    'Use topological sort to find the total ordering',
    'Handle invalid cases: cycle or words not in lexicographic order',
  ],
  starterCode: `from typing import List
from collections import defaultdict, deque

def alien_order(words: List[str]) -> str:
    """
    Find the order of characters in alien language.
    
    Args:
        words: List of words in sorted order by alien language
        
    Returns:
        String of unique characters in alien order, or "" if invalid
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [['wrt', 'wrf', 'er', 'ett', 'rftt']],
      expected: 'wertf',
    },
    {
      input: [['z', 'x']],
      expected: 'zx',
    },
    {
      input: [['z', 'x', 'z']],
      expected: '',
    },
  ],
  timeComplexity: 'O(C)',
  spaceComplexity: 'O(1)',
  solution: `from typing import List
from collections import defaultdict, deque, Counter

def alien_order(words: List[str]) -> str:
    """
    Alien Dictionary - Graph + Topological Sort.
    
    Time: O(C) where C = total characters across all words
    Space: O(1) - at most 26 letters
    
    Key Insight:
    - Build a directed graph from character ordering
    - Compare adjacent words to find edges (char1 -> char2)
    - Use topological sort to find total ordering
    - Invalid if: cycle exists or words not in lexicographic order
    
    Steps:
    1. Initialize graph with all characters
    2. Compare adjacent words to build edges
    3. Topological sort using Kahn's algorithm
    4. Return result if all characters processed
    """
    # Step 1: Initialize graph with all characters
    graph = {c: set() for word in words for c in word}
    in_degree = {c: 0 for c in graph}
    
    # Step 2: Build graph by comparing adjacent words
    for i in range(len(words) - 1):
        word1, word2 = words[i], words[i + 1]
        min_len = min(len(word1), len(word2))
        
        # Check for invalid case: word1 is prefix of word2 but word1 is longer
        # e.g., ["abc", "ab"] is invalid
        if len(word1) > len(word2) and word1[:min_len] == word2[:min_len]:
            return ""
        
        # Find first differing character
        for j in range(min_len):
            if word1[j] != word2[j]:
                # word1[j] comes before word2[j] in alien order
                c1, c2 = word1[j], word2[j]
                
                # Add edge if not already present
                if c2 not in graph[c1]:
                    graph[c1].add(c2)
                    in_degree[c2] += 1
                
                # Only first difference matters
                break
    
    # Step 3: Topological sort using Kahn's algorithm
    queue = deque([c for c in in_degree if in_degree[c] == 0])
    result = []
    
    while queue:
        c = queue.popleft()
        result.append(c)
        
        # Reduce in-degree for neighbors
        for next_c in graph[c]:
            in_degree[next_c] -= 1
            if in_degree[next_c] == 0:
                queue.append(next_c)
    
    # Step 4: Check if all characters processed
    if len(result) < len(in_degree):
        return ""  # Cycle detected
    
    return "".join(result)


# Example walkthrough:
"""
Input: words = ["wrt","wrf","er","ett","rftt"]

Step 1: Extract all characters
{w, r, t, f, e}

Step 2: Build graph by comparing adjacent words

"wrt" vs "wrf":
- w == w, r == r, t != f
- Edge: t → f

"wrf" vs "er":
- w != e
- Edge: w → e

"er" vs "ett":
- e == e, r != t
- Edge: r → t

"ett" vs "rftt":
- e != r
- Edge: e → r

Final graph:
w → e
e → r
r → t
t → f

In-degrees:
w: 0
e: 1
r: 1
t: 1
f: 1

Step 3: Topological sort
queue = [w]
Process w: result = [w], queue = [e]
Process e: result = [w, e], queue = [r]
Process r: result = [w, e, r], queue = [t]
Process t: result = [w, e, r, t], queue = [f]
Process f: result = [w, e, r, t, f]

Result: "wertf"
"""


# Common mistakes:
"""
Mistake 1: Not handling prefix case
- ["abc", "ab"] is invalid (longer word comes first)
- Must check this before comparing characters

Mistake 2: Adding duplicate edges
- Can create wrong in-degrees
- Use set for adjacency list

Mistake 3: Not initializing all characters
- Some characters might not appear in comparisons
- Initialize graph with all characters from all words

Mistake 4: Comparing more than first difference
- Only first differing character matters
- Must break after finding difference
"""`,
  leetcodeUrl: 'https://leetcode.com/problems/alien-dictionary/',
  youtubeUrl: 'https://www.youtube.com/watch?v=6kTZYvNNyps',
};
