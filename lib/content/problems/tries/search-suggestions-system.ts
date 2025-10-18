/**
 * Search Suggestions System
 * Problem ID: search-suggestions-system
 * Order: 7
 */

import { Problem } from '../../../types';

export const search_suggestions_systemProblem: Problem = {
  id: 'search-suggestions-system',
  title: 'Search Suggestions System',
  difficulty: 'Medium',
  topic: 'Tries',
  description: `You are given an array of strings \`products\` and a string \`searchWord\`.

Design a system that suggests at most three product names from \`products\` after each character of \`searchWord\` is typed. Suggested products should have common prefix with \`searchWord\`. If there are more than three products with a common prefix return the three lexicographically minimum products.

Return a list of lists of the suggested products after each character of \`searchWord\` is typed.`,
  examples: [
    {
      input:
        'products = ["mobile","mouse","moneypot","monitor","mousepad"], searchWord = "mouse"',
      output:
        '[["mobile","moneypot","monitor"],["mobile","moneypot","monitor"],["mouse","mousepad"],["mouse","mousepad"],["mouse","mousepad"]]',
    },
    {
      input: 'products = ["havana"], searchWord = "havana"',
      output:
        '[["havana"],["havana"],["havana"],["havana"],["havana"],["havana"]]',
    },
  ],
  constraints: [
    '1 <= products.length <= 1000',
    '1 <= products[i].length <= 3000',
    '1 <= sum(products[i].length) <= 2 * 10^4',
    'All the strings of products are unique',
    'products[i] consists of lowercase English letters',
    '1 <= searchWord.length <= 1000',
    'searchWord consists of lowercase English letters',
  ],
  hints: [
    'Build trie from products',
    'For each prefix, DFS to find up to 3 words',
  ],
  starterCode: `from typing import List

def suggested_products(products: List[str], search_word: str) -> List[List[str]]:
    """
    Find product suggestions for each prefix.
    
    Args:
        products: List of product names
        search_word: Search query
        
    Returns:
        List of suggestions for each prefix
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [['mobile', 'mouse', 'moneypot', 'monitor', 'mousepad'], 'mouse'],
      expected: [
        ['mobile', 'moneypot', 'monitor'],
        ['mobile', 'moneypot', 'monitor'],
        ['mouse', 'mousepad'],
        ['mouse', 'mousepad'],
        ['mouse', 'mousepad'],
      ],
    },
  ],
  timeComplexity:
    'O(N * M + S * M) where N = products, M = avg length, S = searchWord length',
  spaceComplexity: 'O(N * M)',
  leetcodeUrl: 'https://leetcode.com/problems/search-suggestions-system/',
  youtubeUrl: 'https://www.youtube.com/watch?v=D4T2N0yAr20',
};
