/**
 * File Word Frequency Counter
 * Problem ID: intermediate-file-word-frequency
 * Order: 1
 */

import { Problem } from '../../../types';

export const intermediate_file_word_frequencyProblem: Problem = {
  id: 'intermediate-file-word-frequency',
  title: 'File Word Frequency Counter',
  difficulty: 'Medium',
  description: `Read a text file and count the frequency of each word.

**Note:** A virtual test file is created for you in the browser environment using Pyodide's filesystem.

**Requirements:**
- Case-insensitive counting
- Ignore punctuation
- Handle file not found errors
- Return dictionary of word frequencies sorted by count

**Example File Content:**
\`\`\`
The quick brown fox jumps over the lazy dog.
The dog was not that lazy.
\`\`\`

**Expected Output:**
\`\`\`python
{'the': 3, 'dog': 2, 'lazy': 2, 'quick': 1, 'brown': 1, ...}
\`\`\``,
  examples: [
    {
      input: 'filename = "text.txt"',
      output: "{'the': 3, 'dog': 2, 'lazy': 2, ...}",
    },
  ],
  constraints: [
    'Handle FileNotFoundError',
    'Case-insensitive',
    'Remove punctuation',
  ],
  hints: [
    'Use string.punctuation for punctuation',
    'Convert to lowercase before counting',
    'Use try-except for file operations',
  ],
  starterCode: `# Setup: Create virtual test file (for browser environment)
with open('test.txt', 'w') as f:
    f.write("""The quick brown fox jumps over the lazy dog.
The dog was not that lazy.""")

def count_word_frequency(filename):
    """
    Count word frequency in a text file.
    
    Args:
        filename: Path to text file
        
    Returns:
        Dictionary of word -> count, sorted by count (descending)
        
    Raises:
        FileNotFoundError: If file doesn't exist
        
    Examples:
        >>> count_word_frequency("text.txt")
        {'the': 3, 'dog': 2, 'lazy': 2, ...}
    """
    pass


# Test
try:
    result = count_word_frequency("test.txt")
    print(result)
except FileNotFoundError as e:
    print(f"Error: {e}")
`,
  testCases: [
    {
      input: ['test.txt'],
      expected: {
        the: 3,
        dog: 2,
        lazy: 2,
        brown: 1,
        fox: 1,
        jumps: 1,
        not: 1,
        over: 1,
        quick: 1,
        that: 1,
        was: 1,
      },
    },
  ],
  solution: `# Setup: Create virtual test file (for browser environment)
with open('test.txt', 'w') as f:
    f.write("""The quick brown fox jumps over the lazy dog.
The dog was not that lazy.""")

import string
from collections import Counter

def count_word_frequency(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{filename}' not found")
    
    # Remove punctuation and convert to lowercase
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator).lower()
    
    # Count words
    words = text.split()
    word_count = Counter(words)
    
    # Sort by frequency (descending), then alphabetically for ties
    sorted_words = dict(sorted(word_count.items(), 
                               key=lambda x: (-x[1], x[0])))
    
    return sorted_words`,
  timeComplexity: 'O(n) where n is file size',
  spaceComplexity: 'O(w) where w is number of unique words',
  order: 1,
  topic: 'Python Intermediate',
};
