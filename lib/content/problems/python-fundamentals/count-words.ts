/**
 * Word Counter
 * Problem ID: fundamentals-count-words
 * Order: 3
 */

import { Problem } from '../../../types';

export const count_wordsProblem: Problem = {
  id: 'fundamentals-count-words',
  title: 'Word Counter',
  difficulty: 'Easy',
  description: `Count the frequency of each word in a given text.

**Requirements:**
- Case-insensitive (treat "The" and "the" as the same word)
- Remove punctuation
- Return a dictionary with word counts
- Words are separated by spaces

**Note:** You can use dict, Counter, or defaultdict - all will work!

**Example:** "The quick brown fox jumps over the lazy dog" → {'the': 2, 'quick': 1, ...}`,
  examples: [
    {
      input: '"hello world hello"',
      output: "{'hello': 2, 'world': 1}",
    },
  ],
  constraints: [
    'Text length up to 10^4 characters',
    'Words are separated by spaces',
  ],
  hints: [
    'Use split() to separate words',
    'Use a dictionary to count occurrences',
    'Consider using collections.Counter',
  ],
  starterCode: `from collections import defaultdict

def count_words(text):
    """
    Count frequency of each word in text.
    
    Args:
        text: Input string
        
    Returns:
        Dictionary mapping words to their counts
        
    Examples:
        >>> count_words("hello world hello")
        {'hello': 2, 'world': 1}
    """
    pass


# Test
print(count_words("The quick brown fox jumps over the lazy dog"))
`,
  testCases: [
    {
      input: ['hello world hello'],
      expected: { hello: 2, world: 1 },
    },
    {
      input: ['The quick brown fox'],
      expected: { the: 1, quick: 1, brown: 1, fox: 1 },
    },
  ],
  solution: `def count_words(text):
    # Remove punctuation and convert to lowercase
    import string
    translator = str.maketrans(', ', string.punctuation)
    cleaned = text.translate(translator).lower()
    
    # Split and count
    word_count = {}
    for word in cleaned.split():
        word_count[word] = word_count.get(word, 0) + 1
    
    return word_count


# Using Counter (more Pythonic)
from collections import Counter

def count_words_counter(text):
    import string
    translator = str.maketrans(', ', string.punctuation)
    cleaned = text.translate(translator).lower()
    return dict(Counter(cleaned.split()))`,
  timeComplexity: 'O(n) where n is text length',
  spaceComplexity: 'O(w) where w is number of unique words',
  order: 3,
  topic: 'Python Fundamentals',
};
