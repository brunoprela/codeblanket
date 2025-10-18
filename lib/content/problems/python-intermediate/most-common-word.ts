/**
 * Most Common Word
 * Problem ID: most-common-word
 * Order: 28
 */

import { Problem } from '../../../types';

export const most_common_wordProblem: Problem = {
  id: 'most-common-word',
  title: 'Most Common Word',
  difficulty: 'Easy',
  category: 'python-intermediate',
  description: `Given a string \`paragraph\` and an array of banned words, return the most frequent word that is not banned. Words are case-insensitive and punctuation should be ignored.

Use \`Counter\` for efficient counting.

**Example 1:**
\`\`\`
Input: paragraph = "Bob hit a ball, the hit BALL flew far after it was hit.", banned = ["hit"]
Output: "ball"
Explanation: 
"hit" appears 3 times but is banned.
"ball" appears 2 times and is not banned.
\`\`\`

**Example 2:**
\`\`\`
Input: paragraph = "a.", banned = []
Output: "a"
\`\`\``,
  starterCode: `from collections import Counter
import re

def most_common_word(paragraph, banned):
    """
    Find most common non-banned word.
    
    Args:
        paragraph: String containing words and punctuation
        banned: List of banned words
    
    Returns:
        Most common non-banned word (lowercase)
    """
    pass`,
  testCases: [
    {
      input: [
        'Bob hit a ball, the hit BALL flew far after it was hit.',
        ['hit'],
      ],
      expected: 'ball',
    },
    {
      input: ['a.', []],
      expected: 'a',
    },
    {
      input: ['a, a, a, a, b,b,b,c, c', ['a']],
      expected: 'b',
    },
  ],
  hints: [
    'Use regex to extract words: re.findall(r"\\w+", paragraph)',
    'Convert to lowercase',
    'Use Counter.most_common() after filtering banned words',
  ],
  solution: `from collections import Counter
import re

def most_common_word(paragraph, banned):
    """
    Find most common non-banned word.
    
    Args:
        paragraph: String containing words and punctuation
        banned: List of banned words
    
    Returns:
        Most common non-banned word (lowercase)
    """
    # Extract words using regex, convert to lowercase
    words = re.findall(r'\\w+', paragraph.lower())
    
    # Create set of banned words for O(1) lookup
    banned_set = set(banned)
    
    # Count only non-banned words
    word_counts = Counter(word for word in words if word not in banned_set)
    
    # Return most common
    return word_counts.most_common(1)[0][0]


# Alternative without list comprehension
def most_common_word_alt(paragraph, banned):
    words = re.findall(r'\\w+', paragraph.lower())
    banned_set = set(banned)
    
    # Count all words first
    word_counts = Counter(words)
    
    # Remove banned words
    for word in banned_set:
        word_counts.pop(word, None)
    
    return word_counts.most_common(1)[0][0]`,
  timeComplexity: 'O(n + m) where n is paragraph length, m is banned words',
  spaceComplexity: 'O(n)',
  order: 28,
  topic: 'Python Intermediate',
};
