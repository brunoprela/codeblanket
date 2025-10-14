/**
 * Python Fundamentals problems - Testing basic Python concepts
 */

import { Problem } from '../types';
import { pythonFundamentalsBatch1 } from './python-fundamentals-batch1';
import { pythonFundamentalsBatch2 } from './python-fundamentals-batch2';
import { pythonFundamentalsBatch3 } from './python-fundamentals-batch3';
import { pythonFundamentalsBatch4 } from './python-fundamentals-batch4';
import { pythonFundamentalsBatch5 } from './python-fundamentals-batch5';
import { pythonFundamentalsBatch6 } from './python-fundamentals-batch6';
import { pythonFundamentalsBatch7 } from './python-fundamentals-batch7';

export const pythonFundamentalsProblems: Problem[] = [
  {
    id: 'fundamentals-fizzbuzz',
    title: 'FizzBuzz',
    difficulty: 'Easy',
    description: `Write a program that prints numbers from 1 to n with special rules:

- For multiples of 3, print "Fizz" instead of the number
- For multiples of 5, print "Buzz" instead of the number
- For multiples of both 3 and 5, print "FizzBuzz"
- For other numbers, print the number itself

Return a list of strings representing the FizzBuzz sequence.

**Classic Problem:** This is a common programming interview question that tests basic control flow and modulo operations.`,
    examples: [
      {
        input: 'n = 15',
        output:
          '["1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8", "Fizz", "Buzz", "11", "Fizz", "13", "14", "FizzBuzz"]',
      },
    ],
    constraints: ['1 <= n <= 10^4', 'Return a list of strings'],
    hints: [
      'Check for divisibility by 15 first (both 3 and 5)',
      'Use modulo operator (%) to check divisibility',
      'Consider the order of your if conditions',
    ],
    starterCode: `def fizzbuzz(n):
    """
    Generate FizzBuzz sequence up to n.
    
    Args:
        n: Upper limit (inclusive)
        
    Returns:
        List of strings representing FizzBuzz sequence
        
    Examples:
        >>> fizzbuzz(5)
        ['1', '2', 'Fizz', '4', 'Buzz']
    """
    pass


# Test
print(fizzbuzz(15))
`,
    testCases: [
      {
        input: [15],
        expected: [
          '1',
          '2',
          'Fizz',
          '4',
          'Buzz',
          'Fizz',
          '7',
          '8',
          'Fizz',
          'Buzz',
          '11',
          'Fizz',
          '13',
          '14',
          'FizzBuzz',
        ],
      },
      {
        input: [5],
        expected: ['1', '2', 'Fizz', '4', 'Buzz'],
      },
    ],
    solution: `def fizzbuzz(n):
    result = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            result.append("FizzBuzz")
        elif i % 3 == 0:
            result.append("Fizz")
        elif i % 5 == 0:
            result.append("Buzz")
        else:
            result.append(str(i))
    return result


# Alternative using list comprehension
def fizzbuzz_compact(n):
    return [
        "FizzBuzz" if i % 15 == 0
        else "Fizz" if i % 3 == 0
        else "Buzz" if i % 5 == 0
        else str(i)
        for i in range(1, n + 1)
    ]`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(n)',
    order: 1,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-palindrome',
    title: 'Check Palindrome',
    difficulty: 'Easy',
    description: `Determine if a given string is a palindrome. A palindrome reads the same forward and backward.

**Rules:**
- Consider only alphanumeric characters
- Ignore case (uppercase and lowercase are the same)
- Ignore spaces and punctuation

**Example:** "A man, a plan, a canal: Panama" is a palindrome.`,
    examples: [
      {
        input: '"racecar"',
        output: 'True',
      },
      {
        input: '"A man, a plan, a canal: Panama"',
        output: 'True',
      },
      {
        input: '"hello"',
        output: 'False',
      },
    ],
    constraints: [
      'String length can be from 0 to 10^5',
      'Case-insensitive comparison',
    ],
    hints: [
      'Use string slicing [::-1] to reverse',
      'Use isalnum() to check if character is alphanumeric',
      'Convert to lowercase for case-insensitive comparison',
    ],
    starterCode: `def is_palindrome(s):
    """
    Check if string is a palindrome.
    
    Args:
        s: Input string
        
    Returns:
        True if palindrome, False otherwise
        
    Examples:
        >>> is_palindrome("racecar")
        True
        >>> is_palindrome("hello")
        False
    """
    pass


# Test
print(is_palindrome("A man, a plan, a canal: Panama"))
`,
    testCases: [
      {
        input: ['racecar'],
        expected: true,
      },
      {
        input: ['A man, a plan, a canal: Panama'],
        expected: true,
      },
      {
        input: ['hello'],
        expected: false,
      },
    ],
    solution: `def is_palindrome(s):
    # Remove non-alphanumeric and convert to lowercase
    cleaned = ''.join(char.lower() for char in s if char.isalnum())
    return cleaned == cleaned[::-1]


# Alternative: Two pointers
def is_palindrome_two_pointers(s):
    # Clean the string
    cleaned = ''.join(char.lower() for char in s if char.isalnum())
    
    left, right = 0, len(cleaned) - 1
    while left < right:
        if cleaned[left] != cleaned[right]:
            return False
        left += 1
        right -= 1
    return True`,
    timeComplexity: 'O(n) where n is string length',
    spaceComplexity: 'O(n) for cleaned string',
    order: 2,
    topic: 'Python Fundamentals',
  },
  {
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
    translator = str.maketrans('', '', string.punctuation)
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
    translator = str.maketrans('', '', string.punctuation)
    cleaned = text.translate(translator).lower()
    return dict(Counter(cleaned.split()))`,
    timeComplexity: 'O(n) where n is text length',
    spaceComplexity: 'O(w) where w is number of unique words',
    order: 3,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-fibonacci',
    title: 'Fibonacci Sequence',
    difficulty: 'Easy',
    description: `Generate the first n numbers in the Fibonacci sequence.

**Fibonacci Sequence:** Each number is the sum of the two preceding ones:
- F(0) = 0
- F(1) = 1
- F(n) = F(n-1) + F(n-2) for n > 1

**Sequence:** 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...

Return a list of the first n Fibonacci numbers.`,
    examples: [
      {
        input: 'n = 10',
        output: '[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]',
      },
    ],
    constraints: ['0 <= n <= 30', 'Return a list of integers'],
    hints: [
      'Start with [0, 1] for the first two numbers',
      'Use a loop to generate subsequent numbers',
      'Each new number is the sum of the previous two',
    ],
    starterCode: `def fibonacci(n):
    """
    Generate first n Fibonacci numbers.
    
    Args:
        n: Number of Fibonacci numbers to generate
        
    Returns:
        List of first n Fibonacci numbers
        
    Examples:
        >>> fibonacci(5)
        [0, 1, 1, 2, 3]
    """
    pass


# Test
print(fibonacci(10))
`,
    testCases: [
      {
        input: [10],
        expected: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34],
      },
      {
        input: [5],
        expected: [0, 1, 1, 2, 3],
      },
    ],
    solution: `def fibonacci(n):
    if n <= 0:
        return []
    if n == 1:
        return [0]
    
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    
    return fib


# Alternative: Generator
def fibonacci_generator(n):
    a, b = 0, 1
    result = []
    for _ in range(n):
        result.append(a)
        a, b = b, a + b
    return result`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(n)',
    order: 4,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-prime-numbers',
    title: 'Find Prime Numbers',
    difficulty: 'Easy',
    description: `Find all prime numbers up to a given number n using the Sieve of Eratosthenes.

**Prime Number:** A natural number greater than 1 that has no positive divisors other than 1 and itself.

**Sieve of Eratosthenes:**
1. Create a list of consecutive integers from 2 to n
2. Start with the first number (2)
3. Mark all multiples of that number (except the number itself) as composite
4. Move to the next unmarked number and repeat

Return a list of all prime numbers up to and including n.`,
    examples: [
      {
        input: 'n = 10',
        output: '[2, 3, 5, 7]',
      },
    ],
    constraints: ['2 <= n <= 10^6', 'Return sorted list of primes'],
    hints: [
      'Create a boolean array to mark primes',
      'Start marking from 2',
      'Only check up to sqrt(n)',
    ],
    starterCode: `def find_primes(n):
    """
    Find all prime numbers up to n using Sieve of Eratosthenes.
    
    Args:
        n: Upper limit (inclusive)
        
    Returns:
        List of prime numbers up to n
        
    Examples:
        >>> find_primes(10)
        [2, 3, 5, 7]
    """
    pass


# Test
print(find_primes(30))
`,
    testCases: [
      {
        input: [10],
        expected: [2, 3, 5, 7],
      },
      {
        input: [20],
        expected: [2, 3, 5, 7, 11, 13, 17, 19],
      },
    ],
    solution: `def find_primes(n):
    if n < 2:
        return []
    
    # Create boolean array, initially all True
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    
    # Sieve of Eratosthenes
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            # Mark multiples of i as not prime
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    
    # Collect all prime numbers
    return [i for i in range(n + 1) if is_prime[i]]


# Simple approach (less efficient for large n)
def find_primes_simple(n):
    def is_prime(num):
        if num < 2:
            return False
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                return False
        return True
    
    return [i for i in range(2, n + 1) if is_prime(i)]`,
    timeComplexity: 'O(n log log n) - Sieve of Eratosthenes',
    spaceComplexity: 'O(n)',
    order: 5,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-anagram',
    title: 'Check Anagram',
    difficulty: 'Easy',
    description: `Determine if two strings are anagrams of each other.

**Anagram:** Two words are anagrams if they contain the same characters in a different order.

**Rules:**
- Case-insensitive
- Ignore spaces
- Consider only letters

**Examples:**
- "listen" and "silent" are anagrams
- "hello" and "world" are not anagrams`,
    examples: [
      {
        input: 's1 = "listen", s2 = "silent"',
        output: 'True',
      },
      {
        input: 's1 = "hello", s2 = "world"',
        output: 'False',
      },
    ],
    constraints: [
      'String length up to 10^4',
      'Only consider alphabetic characters',
    ],
    hints: [
      'Sort both strings and compare',
      'Or use character frequency counting',
      'Remember to handle case and spaces',
    ],
    starterCode: `def is_anagram(s1, s2):
    """
    Check if two strings are anagrams.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        True if anagrams, False otherwise
        
    Examples:
        >>> is_anagram("listen", "silent")
        True
        >>> is_anagram("hello", "world")
        False
    """
    pass


# Test
print(is_anagram("The Eyes", "They See"))
`,
    testCases: [
      {
        input: ['listen', 'silent'],
        expected: true,
      },
      {
        input: ['hello', 'world'],
        expected: false,
      },
      {
        input: ['The Eyes', 'They See'],
        expected: true,
      },
    ],
    solution: `def is_anagram(s1, s2):
    # Remove spaces and convert to lowercase
    clean1 = ''.join(s1.lower().split())
    clean2 = ''.join(s2.lower().split())
    
    # Sort and compare
    return sorted(clean1) == sorted(clean2)


# Using Counter (more efficient)
from collections import Counter

def is_anagram_counter(s1, s2):
    clean1 = ''.join(s1.lower().split())
    clean2 = ''.join(s2.lower().split())
    return Counter(clean1) == Counter(clean2)


# Using dictionary
def is_anagram_dict(s1, s2):
    clean1 = ''.join(s1.lower().split())
    clean2 = ''.join(s2.lower().split())
    
    if len(clean1) != len(clean2):
        return False
    
    char_count = {}
    for char in clean1:
        char_count[char] = char_count.get(char, 0) + 1
    
    for char in clean2:
        if char not in char_count:
            return False
        char_count[char] -= 1
        if char_count[char] < 0:
            return False
    
    return True`,
    timeComplexity: 'O(n log n) for sorting, O(n) with Counter',
    spaceComplexity: 'O(n)',
    order: 6,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-remove-duplicates',
    title: 'Remove Duplicates from List',
    difficulty: 'Easy',
    description: `Remove duplicates from a list while preserving the original order.

**Requirements:**
- Maintain the order of first occurrence
- Return a new list (don't modify the original)
- Works with any data type

**Example:** [1, 2, 2, 3, 1, 4] → [1, 2, 3, 4]

**Bonus:** Can you solve it in different ways?`,
    examples: [
      {
        input: '[1, 2, 2, 3, 1, 4]',
        output: '[1, 2, 3, 4]',
      },
    ],
    constraints: [
      'List length up to 10^4',
      'Preserve order of first occurrence',
    ],
    hints: [
      'Use a set to track seen elements',
      'Build result list while checking seen set',
      'dict.fromkeys() can also preserve order (Python 3.7+)',
    ],
    starterCode: `def remove_duplicates(items):
    """
    Remove duplicates from list while preserving order.
    
    Args:
        items: List with possible duplicates
        
    Returns:
        New list with duplicates removed
        
    Examples:
        >>> remove_duplicates([1, 2, 2, 3, 1, 4])
        [1, 2, 3, 4]
    """
    pass


# Test
print(remove_duplicates([1, 2, 2, 3, 1, 4]))
print(remove_duplicates(['a', 'b', 'a', 'c', 'b']))
`,
    testCases: [
      {
        input: [[1, 2, 2, 3, 1, 4]],
        expected: [1, 2, 3, 4],
      },
      {
        input: [['a', 'b', 'a', 'c', 'b']],
        expected: ['a', 'b', 'c'],
      },
    ],
    solution: `def remove_duplicates(items):
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


# Using dict.fromkeys (Python 3.7+)
def remove_duplicates_dict(items):
    return list(dict.fromkeys(items))


# Using OrderedDict (older Python versions)
from collections import OrderedDict

def remove_duplicates_ordered(items):
    return list(OrderedDict.fromkeys(items))


# List comprehension with index tracking
def remove_duplicates_index(items):
    return [items[i] for i in range(len(items)) 
            if items[i] not in items[:i]]`,
    timeComplexity: 'O(n) with set, O(n²) with list comprehension',
    spaceComplexity: 'O(n)',
    order: 7,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-reverse-words',
    title: 'Reverse Words in String',
    difficulty: 'Easy',
    description: `Reverse the order of words in a string.

**Requirements:**
- Words are separated by spaces
- Multiple spaces should be reduced to a single space
- Remove leading and trailing spaces
- Preserve individual words (don't reverse characters within words)

**Example:** "  hello   world  " → "world hello"`,
    examples: [
      {
        input: '"the sky is blue"',
        output: '"blue is sky the"',
      },
      {
        input: '"  hello   world  "',
        output: '"world hello"',
      },
    ],
    constraints: [
      'String length up to 10^4',
      'May contain leading/trailing/multiple spaces',
    ],
    hints: [
      'Use split() without arguments to handle multiple spaces',
      'Reverse the list of words',
      'Join with single space',
    ],
    starterCode: `def reverse_words(s):
    """
    Reverse the order of words in a string.
    
    Args:
        s: Input string with words
        
    Returns:
        String with words in reverse order
        
    Examples:
        >>> reverse_words("the sky is blue")
        "blue is sky the"
        >>> reverse_words("  hello   world  ")
        "world hello"
    """
    pass


# Test
print(reverse_words("the sky is blue"))
print(reverse_words("  hello   world  "))
`,
    testCases: [
      {
        input: ['the sky is blue'],
        expected: 'blue is sky the',
      },
      {
        input: ['  hello   world  '],
        expected: 'world hello',
      },
    ],
    solution: `def reverse_words(s):
    # split() without arguments handles multiple spaces
    words = s.split()
    # Reverse the list and join
    return ' '.join(reversed(words))


# Alternative: Using slicing
def reverse_words_slice(s):
    return ' '.join(s.split()[::-1])


# Manual approach
def reverse_words_manual(s):
    words = s.split()
    left, right = 0, len(words) - 1
    while left < right:
        words[left], words[right] = words[right], words[left]
        left += 1
        right -= 1
    return ' '.join(words)`,
    timeComplexity: 'O(n) where n is string length',
    spaceComplexity: 'O(n)',
    order: 8,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-merge-dicts',
    title: 'Merge Dictionaries',
    difficulty: 'Easy',
    description: `Merge multiple dictionaries with conflict resolution.

**Requirements:**
- Take a list of dictionaries and merge them into one
- For duplicate keys, use a strategy:
  - "last": Keep value from the last dictionary (default)
  - "first": Keep value from the first dictionary
  - "sum": Sum all values (assumes numeric values)
  - "list": Collect all values in a list

**Example:**
\`\`\`python
dicts = [{"a": 1, "b": 2}, {"b": 3, "c": 4}]
merge(dicts, "last") → {"a": 1, "b": 3, "c": 4}
merge(dicts, "sum") → {"a": 1, "b": 5, "c": 4}
\`\`\``,
    examples: [
      {
        input:
          'dicts = [{"a": 1, "b": 2}, {"b": 3, "c": 4}], strategy = "last"',
        output: '{"a": 1, "b": 3, "c": 4}',
      },
    ],
    constraints: [
      'List contains 1 to 100 dictionaries',
      'Keys are strings, values are integers',
    ],
    hints: [
      'Iterate through all dictionaries',
      'Use a result dictionary to accumulate values',
      'Handle each strategy differently',
    ],
    starterCode: `def merge_dicts(dicts, strategy="last"):
    """
    Merge multiple dictionaries with conflict resolution.
    
    Args:
        dicts: List of dictionaries to merge
        strategy: How to handle duplicate keys
                  "last", "first", "sum", or "list"
        
    Returns:
        Merged dictionary
        
    Examples:
        >>> merge_dicts([{"a": 1}, {"a": 2}], "last")
        {"a": 2}
        >>> merge_dicts([{"a": 1}, {"a": 2}], "sum")
        {"a": 3}
    """
    pass


# Test
dicts = [{"a": 1, "b": 2}, {"b": 3, "c": 4}, {"a": 5}]
print(merge_dicts(dicts, "last"))
print(merge_dicts(dicts, "sum"))
`,
    testCases: [
      {
        input: [
          [
            { a: 1, b: 2 },
            { b: 3, c: 4 },
          ],
          'last',
        ],
        expected: { a: 1, b: 3, c: 4 },
      },
      {
        input: [
          [
            { a: 1, b: 2 },
            { b: 3, c: 4 },
          ],
          'sum',
        ],
        expected: { a: 1, b: 5, c: 4 },
      },
    ],
    solution: `def merge_dicts(dicts, strategy="last"):
    result = {}
    
    for d in dicts:
        for key, value in d.items():
            if key not in result:
                # First occurrence
                result[key] = value if strategy != "list" else [value]
            else:
                # Duplicate key
                if strategy == "last":
                    result[key] = value
                elif strategy == "first":
                    pass  # Keep existing value
                elif strategy == "sum":
                    result[key] += value
                elif strategy == "list":
                    result[key].append(value)
    
    return result


# Using Python 3.9+ union operator
def merge_dicts_modern(dicts, strategy="last"):
    if strategy == "last":
        result = {}
        for d in dicts:
            result = result | d  # Python 3.9+
        return result
    else:
        return merge_dicts(dicts, strategy)


# Using ChainMap (for reading, preserves separate dicts)
from collections import ChainMap

def merge_dicts_chainmap(dicts):
    return dict(ChainMap(*reversed(dicts)))`,
    timeComplexity: 'O(n*k) where n is number of dicts, k is avg keys per dict',
    spaceComplexity: 'O(k) for result dictionary',
    order: 9,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-list-rotation',
    title: 'Rotate List',
    difficulty: 'Easy',
    description: `Rotate a list to the right by k positions.

**Requirements:**
- Rotate the list k positions to the right
- If k is negative, rotate to the left
- If k > length, rotate k % length positions
- Modify in-place or return new list

**Example:**
- [1, 2, 3, 4, 5] rotated right by 2 → [4, 5, 1, 2, 3]
- [1, 2, 3, 4, 5] rotated left by 2 → [3, 4, 5, 1, 2]

**Visualize:**
\`\`\`
Original:  [1, 2, 3, 4, 5]
Rotate 2:  [4, 5, 1, 2, 3]
           ↑     ↑
           |_____|
\`\`\``,
    examples: [
      {
        input: 'arr = [1, 2, 3, 4, 5], k = 2',
        output: '[4, 5, 1, 2, 3]',
      },
    ],
    constraints: ['1 <= list length <= 10^5', '-10^9 <= k <= 10^9'],
    hints: [
      'Handle k > length using modulo',
      'Use list slicing for simple solution',
      'For in-place: reverse entire array, then reverse two parts',
    ],
    starterCode: `def rotate_list(arr, k):
    """
    Rotate list to the right by k positions.
    
    Args:
        arr: List to rotate
        k: Number of positions (positive = right, negative = left)
        
    Returns:
        Rotated list
        
    Examples:
        >>> rotate_list([1, 2, 3, 4, 5], 2)
        [4, 5, 1, 2, 3]
        >>> rotate_list([1, 2, 3, 4, 5], -2)
        [3, 4, 5, 1, 2]
    """
    pass


# Test
print(rotate_list([1, 2, 3, 4, 5], 2))
print(rotate_list([1, 2, 3, 4, 5], -2))
print(rotate_list([1, 2, 3, 4, 5], 7))  # Same as rotating by 2
`,
    testCases: [
      {
        input: [[1, 2, 3, 4, 5], 2],
        expected: [4, 5, 1, 2, 3],
      },
      {
        input: [[1, 2, 3, 4, 5], -2],
        expected: [3, 4, 5, 1, 2],
      },
      {
        input: [[1, 2, 3], 5],
        expected: [2, 3, 1],
      },
    ],
    solution: `def rotate_list(arr, k):
    if not arr:
        return arr
    
    n = len(arr)
    k = k % n  # Handle k > n
    
    # Slice and concatenate
    return arr[-k:] + arr[:-k] if k else arr


# In-place rotation using reverse
def rotate_list_inplace(arr, k):
    if not arr:
        return arr
    
    n = len(arr)
    k = k % n
    
    def reverse(start, end):
        while start < end:
            arr[start], arr[end] = arr[end], arr[start]
            start += 1
            end -= 1
    
    # Reverse entire array
    reverse(0, n - 1)
    # Reverse first k elements
    reverse(0, k - 1)
    # Reverse remaining elements
    reverse(k, n - 1)
    
    return arr


# Using deque.rotate
from collections import deque

def rotate_list_deque(arr, k):
    d = deque(arr)
    d.rotate(k)
    return list(d)`,
    timeComplexity: 'O(n) with slicing, O(n) in-place',
    spaceComplexity: 'O(n) with slicing, O(1) in-place',
    order: 10,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-sum-multiples',
    title: 'Sum of Multiples',
    difficulty: 'Easy',
    description: `Find the sum of all multiples of 3 or 5 below n.

**Example:** For n=10, multiples are 3, 5, 6, 9. Sum = 23.

This problem tests:
- Loop iteration
- Conditional logic
- Mathematical operations`,
    examples: [
      {
        input: 'n = 10',
        output: '23',
        explanation: '3 + 5 + 6 + 9 = 23',
      },
      {
        input: 'n = 20',
        output: '78',
        explanation: 'Sum of all multiples of 3 or 5 below 20',
      },
    ],
    constraints: ['1 <= n <= 10^6'],
    hints: [
      'Use modulo operator to check divisibility',
      'Add numbers that are divisible by 3 OR 5',
      'Be careful not to double-count multiples of 15',
    ],
    starterCode: `def sum_multiples(n):
    """
    Find sum of all multiples of 3 or 5 below n.
    
    Args:
        n: Upper limit (exclusive)
        
    Returns:
        Sum of multiples
        
    Examples:
        >>> sum_multiples(10)
        23
    """
    pass`,
    testCases: [
      {
        input: [10],
        expected: 23,
      },
      {
        input: [20],
        expected: 78,
      },
      {
        input: [1],
        expected: 0,
      },
    ],
    solution: `def sum_multiples(n):
    total = 0
    for i in range(n):
        if i % 3 == 0 or i % 5 == 0:
            total += i
    return total

# Alternative: Using sum with generator
def sum_multiples_alt(n):
    return sum(i for i in range(n) if i % 3 == 0 or i % 5 == 0)`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    order: 11,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-factorial',
    title: 'Factorial Calculator',
    difficulty: 'Easy',
    description: `Calculate the factorial of a non-negative integer n.

**Factorial** (n!) is the product of all positive integers less than or equal to n.
- 5! = 5 × 4 × 3 × 2 × 1 = 120
- 0! = 1 (by definition)

This problem tests:
- Recursion or iteration
- Base case handling
- Mathematical operations`,
    examples: [
      {
        input: 'n = 5',
        output: '120',
        explanation: '5! = 5 × 4 × 3 × 2 × 1 = 120',
      },
      {
        input: 'n = 0',
        output: '1',
        explanation: '0! = 1 by definition',
      },
    ],
    constraints: ['0 <= n <= 20'],
    hints: [
      'Use a loop to multiply numbers from 1 to n',
      'Or use recursion: n! = n × (n-1)!',
      'Handle the base case: 0! = 1',
    ],
    starterCode: `def factorial(n):
    """
    Calculate factorial of n.
    
    Args:
        n: Non-negative integer
        
    Returns:
        Factorial of n
        
    Examples:
        >>> factorial(5)
        120
        >>> factorial(0)
        1
    """
    pass`,
    testCases: [
      {
        input: [5],
        expected: 120,
      },
      {
        input: [0],
        expected: 1,
      },
      {
        input: [10],
        expected: 3628800,
      },
    ],
    solution: `def factorial(n):
    # Iterative approach
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

# Recursive approach
def factorial_recursive(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial_recursive(n - 1)

# Using math module
import math
def factorial_builtin(n):
    return math.factorial(n)`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1) iterative, O(n) recursive',
    order: 12,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-vowel-count',
    title: 'Count Vowels',
    difficulty: 'Easy',
    description: `Count the number of vowels (a, e, i, o, u) in a string.

Count both uppercase and lowercase vowels.

This problem tests:
- String iteration
- Character checking
- Case handling`,
    examples: [
      {
        input: 's = "Hello World"',
        output: '3',
        explanation: 'e, o, o are vowels',
      },
      {
        input: 's = "Python Programming"',
        output: '4',
        explanation: 'o, a, i, o are vowels',
      },
    ],
    constraints: [
      '1 <= len(s) <= 10^5',
      'String contains only letters and spaces',
    ],
    hints: [
      'Convert to lowercase for easier comparison',
      'Check if each character is in "aeiou"',
      'Use a counter variable',
    ],
    starterCode: `def count_vowels(s):
    """
    Count vowels in a string.
    
    Args:
        s: Input string
        
    Returns:
        Number of vowels
        
    Examples:
        >>> count_vowels("Hello")
        2
    """
    pass`,
    testCases: [
      {
        input: ['Hello World'],
        expected: 3,
      },
      {
        input: ['Python Programming'],
        expected: 4,
      },
      {
        input: ['xyz'],
        expected: 0,
      },
    ],
    solution: `def count_vowels(s):
    vowels = "aeiouAEIOU"
    count = 0
    for char in s:
        if char in vowels:
            count += 1
    return count

# Alternative: Using sum
def count_vowels_alt(s):
    return sum(1 for char in s if char.lower() in 'aeiou')`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    order: 13,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-max-min',
    title: 'Find Max and Min',
    difficulty: 'Easy',
    description: `Find both the maximum and minimum values in a list in a single pass.

Return a tuple (min, max).

This problem tests:
- List traversal
- Comparison operations
- Tuple return values`,
    examples: [
      {
        input: 'nums = [3, 1, 4, 1, 5, 9, 2, 6]',
        output: '(1, 9)',
        explanation: 'Minimum is 1, maximum is 9',
      },
      {
        input: 'nums = [-5, -2, -10, -1]',
        output: '(-10, -1)',
      },
    ],
    constraints: [
      '1 <= len(nums) <= 10^5',
      'Cannot use built-in min() or max()',
    ],
    hints: [
      'Initialize min and max with first element',
      'Iterate through remaining elements',
      'Update min and max as needed',
    ],
    starterCode: `def find_max_min(nums):
    """
    Find max and min in a list.
    
    Args:
        nums: List of numbers
        
    Returns:
        Tuple (min, max)
        
    Examples:
        >>> find_max_min([3, 1, 4, 1, 5])
        (1, 5)
    """
    pass`,
    testCases: [
      {
        input: [[3, 1, 4, 1, 5, 9, 2, 6]],
        expected: [1, 9],
      },
      {
        input: [[-5, -2, -10, -1]],
        expected: [-10, -1],
      },
      {
        input: [[42]],
        expected: [42, 42],
      },
    ],
    solution: `def find_max_min(nums):
    if not nums:
        return None
    
    min_val = nums[0]
    max_val = nums[0]
    
    for num in nums[1:]:
        if num < min_val:
            min_val = num
        if num > max_val:
            max_val = num
    
    return (min_val, max_val)`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    order: 14,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-digit-sum',
    title: 'Sum of Digits',
    difficulty: 'Easy',
    description: `Calculate the sum of all digits in a positive integer.

**Example:** 
- 123 → 1 + 2 + 3 = 6
- 9999 → 9 + 9 + 9 + 9 = 36

This problem tests:
- Number manipulation
- String conversion or modulo operations
- Loop iteration`,
    examples: [
      {
        input: 'n = 123',
        output: '6',
        explanation: '1 + 2 + 3 = 6',
      },
      {
        input: 'n = 9999',
        output: '36',
        explanation: '9 + 9 + 9 + 9 = 36',
      },
    ],
    constraints: ['0 <= n <= 10^9'],
    hints: [
      'Convert to string and iterate through characters',
      'Or use modulo (%) and division (//) to extract digits',
      'Use sum() with a generator for concise solution',
    ],
    starterCode: `def sum_of_digits(n):
    """
    Calculate sum of digits in a number.
    
    Args:
        n: Positive integer
        
    Returns:
        Sum of all digits
        
    Examples:
        >>> sum_of_digits(123)
        6
    """
    pass`,
    testCases: [
      {
        input: [123],
        expected: 6,
      },
      {
        input: [9999],
        expected: 36,
      },
      {
        input: [0],
        expected: 0,
      },
    ],
    solution: `def sum_of_digits(n):
    # String approach
    return sum(int(digit) for digit in str(n))

# Mathematical approach
def sum_of_digits_math(n):
    total = 0
    while n > 0:
        total += n % 10
        n //= 10
    return total`,
    timeComplexity: 'O(log n) - number of digits',
    spaceComplexity: 'O(1)',
    order: 15,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-capitalize-words',
    title: 'Capitalize Words',
    difficulty: 'Easy',
    description: `Capitalize the first letter of each word in a sentence.

**Example:** "hello world" → "Hello World"

This problem tests:
- String manipulation
- String methods
- List operations`,
    examples: [
      {
        input: 's = "hello world"',
        output: '"Hello World"',
      },
      {
        input: 's = "python is awesome"',
        output: '"Python Is Awesome"',
      },
    ],
    constraints: [
      '1 <= len(s) <= 10^4',
      'String contains lowercase letters and spaces',
    ],
    hints: [
      'Split string into words',
      'Capitalize first letter of each word',
      'Join words back together',
    ],
    starterCode: `def capitalize_words(s):
    """
    Capitalize first letter of each word.
    
    Args:
        s: Input string
        
    Returns:
        String with capitalized words
        
    Examples:
        >>> capitalize_words("hello world")
        "Hello World"
    """
    pass`,
    testCases: [
      {
        input: ['hello world'],
        expected: 'Hello World',
      },
      {
        input: ['python is awesome'],
        expected: 'Python Is Awesome',
      },
      {
        input: ['a'],
        expected: 'A',
      },
    ],
    solution: `def capitalize_words(s):
    # Using title() method
    return s.title()

# Manual approach
def capitalize_words_manual(s):
    words = s.split()
    capitalized = [word.capitalize() for word in words]
    return ' '.join(capitalized)

# Without split
def capitalize_words_alt(s):
    return ' '.join(word[0].upper() + word[1:] if len(word) > 0 else '' for word in s.split())`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(n)',
    order: 16,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-common-elements',
    title: 'Common Elements in Lists',
    difficulty: 'Easy',
    description: `Find common elements between two lists.

Return a list of elements that appear in both lists (no duplicates).

This problem tests:
- Set operations
- List comprehension
- Finding intersections`,
    examples: [
      {
        input: 'list1 = [1, 2, 3, 4], list2 = [3, 4, 5, 6]',
        output: '[3, 4]',
        explanation: '3 and 4 appear in both lists',
      },
      {
        input: 'list1 = [1, 1, 2, 3], list2 = [2, 2, 3, 4]',
        output: '[2, 3]',
        explanation: 'Return unique common elements',
      },
    ],
    constraints: ['0 <= len(list1), len(list2) <= 1000'],
    hints: [
      'Convert lists to sets',
      'Use set intersection',
      'Convert back to list',
    ],
    starterCode: `def common_elements(list1, list2):
    """
    Find common elements between two lists.
    
    Args:
        list1: First list
        list2: Second list
        
    Returns:
        List of common elements
        
    Examples:
        >>> common_elements([1, 2, 3], [2, 3, 4])
        [2, 3]
    """
    pass`,
    testCases: [
      {
        input: [
          [1, 2, 3, 4],
          [3, 4, 5, 6],
        ],
        expected: [3, 4],
      },
      {
        input: [
          [1, 1, 2, 3],
          [2, 2, 3, 4],
        ],
        expected: [2, 3],
      },
      {
        input: [
          [1, 2, 3],
          [4, 5, 6],
        ],
        expected: [],
      },
    ],
    solution: `def common_elements(list1, list2):
    # Using set intersection
    return list(set(list1) & set(list2))

# Alternative approaches
def common_elements_alt1(list1, list2):
    return list(set(list1).intersection(set(list2)))

def common_elements_alt2(list1, list2):
    return [x for x in set(list1) if x in set(list2)]`,
    timeComplexity: 'O(n + m)',
    spaceComplexity: 'O(n + m)',
    order: 17,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-second-largest',
    title: 'Second Largest Number',
    difficulty: 'Easy',
    description: `Find the second largest number in a list.

If there is no second largest (all elements are the same), return None.

This problem tests:
- List traversal
- Tracking multiple values
- Edge case handling`,
    examples: [
      {
        input: 'nums = [5, 2, 8, 1, 9]',
        output: '8',
        explanation: 'Largest is 9, second largest is 8',
      },
      {
        input: 'nums = [5, 5, 5]',
        output: 'None',
        explanation: 'All elements are the same',
      },
    ],
    constraints: ['1 <= len(nums) <= 10^4', 'Cannot use sorting'],
    hints: [
      'Track both largest and second largest',
      'Update both values as you iterate',
      'Handle duplicates correctly',
    ],
    starterCode: `def second_largest(nums):
    """
    Find second largest number in a list.
    
    Args:
        nums: List of numbers
        
    Returns:
        Second largest number or None
        
    Examples:
        >>> second_largest([5, 2, 8, 1, 9])
        8
    """
    pass`,
    testCases: [
      {
        input: [[5, 2, 8, 1, 9]],
        expected: 8,
      },
      {
        input: [[10, 20, 5, 15]],
        expected: 15,
      },
      {
        input: [[5, 5, 5]],
        expected: null,
      },
    ],
    solution: `def second_largest(nums):
    if not nums:
        return None
    
    # Remove duplicates and sort
    unique_nums = list(set(nums))
    
    if len(unique_nums) < 2:
        return None
    
    unique_nums.sort()
    return unique_nums[-2]

# Single pass approach
def second_largest_optimized(nums):
    if len(nums) < 2:
        return None
    
    largest = second = float('-inf')
    
    for num in nums:
        if num > largest:
            second = largest
            largest = num
        elif num > second and num != largest:
            second = num
    
    return None if second == float('-inf') else second`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(n) for set approach, O(1) for optimized',
    order: 18,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-list-sum',
    title: 'Sum of List Elements',
    difficulty: 'Easy',
    description: `Calculate the sum of all elements in a list without using the built-in sum() function.

This problem tests:
- Loop iteration
- Accumulator pattern
- Basic arithmetic`,
    examples: [
      {
        input: 'nums = [1, 2, 3, 4, 5]',
        output: '15',
        explanation: '1 + 2 + 3 + 4 + 5 = 15',
      },
      {
        input: 'nums = [-1, -2, 3]',
        output: '0',
        explanation: '-1 + -2 + 3 = 0',
      },
    ],
    constraints: ['0 <= len(nums) <= 10^4', 'Cannot use built-in sum()'],
    hints: [
      'Initialize a total variable to 0',
      'Loop through each element',
      'Add each element to the total',
    ],
    starterCode: `def list_sum(nums):
    """
    Calculate sum of list elements.
    
    Args:
        nums: List of numbers
        
    Returns:
        Sum of all elements
        
    Examples:
        >>> list_sum([1, 2, 3, 4, 5])
        15
    """
    pass`,
    testCases: [
      {
        input: [[1, 2, 3, 4, 5]],
        expected: 15,
      },
      {
        input: [[-1, -2, 3]],
        expected: 0,
      },
      {
        input: [[]],
        expected: 0,
      },
    ],
    solution: `def list_sum(nums):
    total = 0
    for num in nums:
        total += num
    return total

# Using reduce
from functools import reduce
def list_sum_reduce(nums):
    return reduce(lambda x, y: x + y, nums, 0)

# Recursive approach
def list_sum_recursive(nums):
    if not nums:
        return 0
    return nums[0] + list_sum_recursive(nums[1:])`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1) iterative, O(n) recursive',
    order: 19,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-flatten-list',
    title: 'Flatten Nested List',
    difficulty: 'Medium',
    description: `Flatten a nested list structure into a single-level list.

**Example:** [[1, 2], [3, [4, 5]], 6] → [1, 2, 3, 4, 5, 6]

This problem tests:
- Recursion
- Type checking
- List operations`,
    examples: [
      {
        input: 'nested = [[1, 2], [3, 4]]',
        output: '[1, 2, 3, 4]',
      },
      {
        input: 'nested = [[1, 2], [3, [4, 5]], 6]',
        output: '[1, 2, 3, 4, 5, 6]',
      },
    ],
    constraints: ['List can be nested to any depth', 'Elements are integers'],
    hints: [
      'Use recursion to handle nested lists',
      'Check if element is a list using isinstance()',
      'Recursively flatten sub-lists',
    ],
    starterCode: `def flatten_list(nested):
    """
    Flatten a nested list.
    
    Args:
        nested: Nested list structure
        
    Returns:
        Flattened list
        
    Examples:
        >>> flatten_list([[1, 2], [3, 4]])
        [1, 2, 3, 4]
        >>> flatten_list([[1, 2], [3, [4, 5]], 6])
        [1, 2, 3, 4, 5, 6]
    """
    pass`,
    testCases: [
      {
        input: [
          [
            [1, 2],
            [3, 4],
          ],
        ],
        expected: [1, 2, 3, 4],
      },
      {
        input: [[[1, 2], [3, [4, 5]], 6]],
        expected: [1, 2, 3, 4, 5, 6],
      },
      {
        input: [[[1]]],
        expected: [1],
      },
    ],
    solution: `def flatten_list(nested):
    result = []
    for item in nested:
        if isinstance(item, list):
            # Recursively flatten nested lists
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result

# Using list comprehension with recursion
def flatten_list_alt(nested):
    return [item for sublist in nested 
            for item in (flatten_list_alt(sublist) if isinstance(sublist, list) else [sublist])]

# Iterative approach using stack
def flatten_list_iterative(nested):
    stack = list(nested)
    result = []
    
    while stack:
        item = stack.pop(0)
        if isinstance(item, list):
            stack = item + stack
        else:
            result.append(item)
    
    return result`,
    timeComplexity: 'O(n) where n is total number of elements',
    spaceComplexity: 'O(d) where d is maximum nesting depth',
    order: 20,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-is-palindrome-number',
    title: 'Palindrome Number',
    difficulty: 'Easy',
    description: `Determine if an integer is a palindrome without converting it to a string.

A palindrome number reads the same backward as forward.

**Examples:**
- 121 → True
- -121 → False (negative numbers are not palindromes)
- 10 → False

This problem tests:
- Number manipulation
- Mathematical operations
- Logic without string conversion`,
    examples: [
      {
        input: 'x = 121',
        output: 'True',
      },
      {
        input: 'x = -121',
        output: 'False',
        explanation: 'Negative numbers are not palindromes',
      },
      {
        input: 'x = 10',
        output: 'False',
      },
    ],
    constraints: ['-2^31 <= x <= 2^31 - 1'],
    hints: [
      'Extract digits using modulo and division',
      'Build the reversed number',
      'Compare with original',
    ],
    starterCode: `def is_palindrome_number(x):
    """
    Check if number is palindrome without string conversion.
    
    Args:
        x: Integer to check
        
    Returns:
        True if palindrome, False otherwise
        
    Examples:
        >>> is_palindrome_number(121)
        True
        >>> is_palindrome_number(-121)
        False
    """
    pass`,
    testCases: [
      {
        input: [121],
        expected: true,
      },
      {
        input: [-121],
        expected: false,
      },
      {
        input: [10],
        expected: false,
      },
      {
        input: [0],
        expected: true,
      },
    ],
    solution: `def is_palindrome_number(x):
    # Negative numbers are not palindromes
    if x < 0:
        return False
    
    # Single digit numbers are palindromes
    if x < 10:
        return True
    
    # Numbers ending in 0 (except 0 itself) are not palindromes
    if x % 10 == 0:
        return False
    
    # Reverse the number
    original = x
    reversed_num = 0
    
    while x > 0:
        digit = x % 10
        reversed_num = reversed_num * 10 + digit
        x //= 10
    
    return original == reversed_num`,
    timeComplexity: 'O(log n) - number of digits',
    spaceComplexity: 'O(1)',
    order: 21,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-reverse-integer',
    title: 'Reverse Integer',
    difficulty: 'Easy',
    description: `Reverse the digits of a signed integer.

If reversing causes overflow (outside 32-bit range), return 0.

**Examples:**
- 123 → 321
- -123 → -321
- 120 → 21

This problem tests:
- Integer manipulation
- Edge case handling
- Mathematical operations`,
    examples: [
      {
        input: 'x = 123',
        output: '321',
      },
      {
        input: 'x = -123',
        output: '-321',
      },
      {
        input: 'x = 120',
        output: '21',
      },
    ],
    constraints: ['-2^31 <= x <= 2^31 - 1', 'Return 0 if overflow'],
    hints: [
      'Handle negative sign separately',
      'Use modulo to extract digits',
      'Check for 32-bit overflow',
    ],
    starterCode: `def reverse_integer(x):
    """
    Reverse digits of an integer.
    
    Args:
        x: Integer to reverse
        
    Returns:
        Reversed integer, or 0 if overflow
        
    Examples:
        >>> reverse_integer(123)
        321
        >>> reverse_integer(-123)
        -321
    """
    pass`,
    testCases: [
      {
        input: [123],
        expected: 321,
      },
      {
        input: [-123],
        expected: -321,
      },
      {
        input: [120],
        expected: 21,
      },
      {
        input: [0],
        expected: 0,
      },
    ],
    solution: `def reverse_integer(x):
    # Handle sign
    sign = -1 if x < 0 else 1
    x = abs(x)
    
    # Reverse digits
    reversed_num = 0
    while x > 0:
        digit = x % 10
        reversed_num = reversed_num * 10 + digit
        x //= 10
    
    # Apply sign
    result = sign * reversed_num
    
    # Check for 32-bit overflow
    if result < -2**31 or result > 2**31 - 1:
        return 0
    
    return result

# Alternative using string
def reverse_integer_str(x):
    sign = -1 if x < 0 else 1
    reversed_str = str(abs(x))[::-1]
    result = sign * int(reversed_str)
    return result if -2**31 <= result <= 2**31 - 1 else 0`,
    timeComplexity: 'O(log n)',
    spaceComplexity: 'O(1)',
    order: 22,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-perfect-number',
    title: 'Perfect Number',
    difficulty: 'Easy',
    description: `Check if a number is a perfect number.

A perfect number is a positive integer equal to the sum of its proper positive divisors (excluding itself).

**Examples:**
- 6 → True (divisors: 1, 2, 3; sum = 6)
- 28 → True (divisors: 1, 2, 4, 7, 14; sum = 28)
- 12 → False

This problem tests:
- Finding divisors
- Mathematical properties
- Loop optimization`,
    examples: [
      {
        input: 'num = 6',
        output: 'True',
        explanation: '1 + 2 + 3 = 6',
      },
      {
        input: 'num = 28',
        output: 'True',
        explanation: '1 + 2 + 4 + 7 + 14 = 28',
      },
      {
        input: 'num = 12',
        output: 'False',
      },
    ],
    constraints: ['1 <= num <= 10^8'],
    hints: [
      'Find all divisors up to sqrt(n)',
      'Add both i and n/i when found',
      'Compare sum with original number',
    ],
    starterCode: `def is_perfect_number(num):
    """
    Check if number is perfect.
    
    Args:
        num: Positive integer
        
    Returns:
        True if perfect, False otherwise
        
    Examples:
        >>> is_perfect_number(6)
        True
        >>> is_perfect_number(28)
        True
    """
    pass`,
    testCases: [
      {
        input: [6],
        expected: true,
      },
      {
        input: [28],
        expected: true,
      },
      {
        input: [12],
        expected: false,
      },
      {
        input: [1],
        expected: false,
      },
    ],
    solution: `def is_perfect_number(num):
    if num <= 1:
        return False
    
    # Find sum of divisors
    divisor_sum = 1  # 1 is always a divisor
    
    # Only check up to sqrt(num)
    i = 2
    while i * i <= num:
        if num % i == 0:
            divisor_sum += i
            # Add the paired divisor (but not if it's the square root)
            if i * i != num:
                divisor_sum += num // i
        i += 1
    
    return divisor_sum == num`,
    timeComplexity: 'O(sqrt(n))',
    spaceComplexity: 'O(1)',
    order: 23,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-missing-number',
    title: 'Missing Number',
    difficulty: 'Easy',
    description: `Find the missing number in an array containing n distinct numbers from 0 to n.

**Example:** [3, 0, 1] → 2

This problem tests:
- Mathematical formulas
- Set operations
- Array manipulation`,
    examples: [
      {
        input: 'nums = [3, 0, 1]',
        output: '2',
        explanation: 'Range is [0,3], missing number is 2',
      },
      {
        input: 'nums = [0, 1]',
        output: '2',
      },
      {
        input: 'nums = [9,6,4,2,3,5,7,0,1]',
        output: '8',
      },
    ],
    constraints: ['n == len(nums)', '1 <= n <= 10^4', 'All numbers are unique'],
    hints: [
      'Sum of first n numbers: n * (n + 1) / 2',
      'Subtract sum of array from expected sum',
      'Or use XOR properties',
    ],
    starterCode: `def missing_number(nums):
    """
    Find the missing number in range [0, n].
    
    Args:
        nums: List of distinct numbers from 0 to n
        
    Returns:
        The missing number
        
    Examples:
        >>> missing_number([3, 0, 1])
        2
    """
    pass`,
    testCases: [
      {
        input: [[3, 0, 1]],
        expected: 2,
      },
      {
        input: [[0, 1]],
        expected: 2,
      },
      {
        input: [[9, 6, 4, 2, 3, 5, 7, 0, 1]],
        expected: 8,
      },
    ],
    solution: `def missing_number(nums):
    n = len(nums)
    # Expected sum of 0 to n
    expected_sum = n * (n + 1) // 2
    # Actual sum
    actual_sum = sum(nums)
    # Difference is the missing number
    return expected_sum - actual_sum

# Alternative using XOR
def missing_number_xor(nums):
    result = len(nums)
    for i, num in enumerate(nums):
        result ^= i ^ num
    return result

# Alternative using set
def missing_number_set(nums):
    return (set(range(len(nums) + 1)) - set(nums)).pop()`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    order: 24,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-move-zeros',
    title: 'Move Zeros to End',
    difficulty: 'Easy',
    description: `Move all zeros in an array to the end while maintaining the relative order of non-zero elements.

**Must modify the array in-place.**

**Example:** [0, 1, 0, 3, 12] → [1, 3, 12, 0, 0]

This problem tests:
- In-place array manipulation
- Two-pointer technique
- Order preservation`,
    examples: [
      {
        input: 'nums = [0, 1, 0, 3, 12]',
        output: '[1, 3, 12, 0, 0]',
      },
      {
        input: 'nums = [0]',
        output: '[0]',
      },
    ],
    constraints: [
      '1 <= len(nums) <= 10^4',
      'Must be in-place with O(1) extra space',
    ],
    hints: [
      'Use two pointers',
      'One pointer for non-zero position',
      'Swap non-zero elements forward',
    ],
    starterCode: `def move_zeros(nums):
    """
    Move all zeros to end, maintaining order of non-zeros.
    Modify array in-place.
    
    Args:
        nums: List of integers
        
    Returns:
        None (modifies nums in-place)
        
    Examples:
        >>> nums = [0, 1, 0, 3, 12]
        >>> move_zeros(nums)
        >>> nums
        [1, 3, 12, 0, 0]
    """
    pass`,
    testCases: [
      {
        input: [[0, 1, 0, 3, 12]],
        expected: [1, 3, 12, 0, 0],
      },
      {
        input: [[0]],
        expected: [0],
      },
      {
        input: [[1, 2, 3]],
        expected: [1, 2, 3],
      },
    ],
    solution: `def move_zeros(nums):
    # Two pointer approach
    write_pos = 0  # Position to write next non-zero
    
    # Move all non-zeros to the front
    for i in range(len(nums)):
        if nums[i] != 0:
            nums[write_pos] = nums[i]
            write_pos += 1
    
    # Fill remaining positions with zeros
    for i in range(write_pos, len(nums)):
        nums[i] = 0

# Alternative with swapping
def move_zeros_swap(nums):
    write_pos = 0
    for i in range(len(nums)):
        if nums[i] != 0:
            nums[write_pos], nums[i] = nums[i], nums[write_pos]
            write_pos += 1`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    order: 25,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-find-single-number',
    title: 'Single Number',
    difficulty: 'Easy',
    description: `Find the element that appears once in an array where every other element appears twice.

**Must use O(1) extra space.**

**Example:** [2, 2, 1] → 1

This problem tests:
- XOR properties
- Bit manipulation basics
- Space optimization`,
    examples: [
      {
        input: 'nums = [2, 2, 1]',
        output: '1',
      },
      {
        input: 'nums = [4, 1, 2, 1, 2]',
        output: '4',
      },
      {
        input: 'nums = [1]',
        output: '1',
      },
    ],
    constraints: [
      'Each element appears twice except one',
      'Must use O(1) space',
    ],
    hints: [
      'XOR has special properties: a ^ a = 0, a ^ 0 = a',
      'XOR all numbers together',
      'Pairs cancel out, leaving the single number',
    ],
    starterCode: `def single_number(nums):
    """
    Find the number that appears once.
    
    Args:
        nums: List where each element appears twice except one
        
    Returns:
        The single number
        
    Examples:
        >>> single_number([2, 2, 1])
        1
        >>> single_number([4, 1, 2, 1, 2])
        4
    """
    pass`,
    testCases: [
      {
        input: [[2, 2, 1]],
        expected: 1,
      },
      {
        input: [[4, 1, 2, 1, 2]],
        expected: 4,
      },
      {
        input: [[1]],
        expected: 1,
      },
    ],
    solution: `def single_number(nums):
    # XOR all numbers together
    # Numbers appearing twice cancel out (a ^ a = 0)
    # Result is the single number (a ^ 0 = a)
    result = 0
    for num in nums:
        result ^= num
    return result

# Using reduce
from functools import reduce
import operator
def single_number_reduce(nums):
    return reduce(operator.xor, nums)`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    order: 26,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-armstrong-number',
    title: 'Armstrong Number',
    difficulty: 'Easy',
    description: `Check if a number is an Armstrong number (Narcissistic number).

An Armstrong number is equal to the sum of its digits each raised to the power of the number of digits.

**Examples:**
- 153 → True (1³ + 5³ + 3³ = 153)
- 9474 → True (9⁴ + 4⁴ + 7⁴ + 4⁴ = 9474)

This problem tests:
- Digit extraction
- Power operations
- Mathematical properties`,
    examples: [
      {
        input: 'n = 153',
        output: 'True',
        explanation: '1³ + 5³ + 3³ = 1 + 125 + 27 = 153',
      },
      {
        input: 'n = 123',
        output: 'False',
        explanation: '1³ + 2³ + 3³ = 36 ≠ 123',
      },
    ],
    constraints: ['0 <= n <= 10^9'],
    hints: [
      'Count number of digits first',
      'Extract each digit',
      'Sum digit raised to power of digit count',
    ],
    starterCode: `def is_armstrong_number(n):
    """
    Check if number is an Armstrong number.
    
    Args:
        n: Non-negative integer
        
    Returns:
        True if Armstrong number, False otherwise
        
    Examples:
        >>> is_armstrong_number(153)
        True
        >>> is_armstrong_number(123)
        False
    """
    pass`,
    testCases: [
      {
        input: [153],
        expected: true,
      },
      {
        input: [9474],
        expected: true,
      },
      {
        input: [123],
        expected: false,
      },
      {
        input: [0],
        expected: true,
      },
    ],
    solution: `def is_armstrong_number(n):
    # Convert to string to get digits and count
    digits_str = str(n)
    num_digits = len(digits_str)
    
    # Calculate sum of each digit raised to power of digit count
    armstrong_sum = sum(int(digit) ** num_digits for digit in digits_str)
    
    return armstrong_sum == n

# Mathematical approach without string conversion
def is_armstrong_number_math(n):
    if n == 0:
        return True
    
    # Count digits
    temp = n
    num_digits = 0
    while temp > 0:
        num_digits += 1
        temp //= 10
    
    # Calculate sum
    temp = n
    armstrong_sum = 0
    while temp > 0:
        digit = temp % 10
        armstrong_sum += digit ** num_digits
        temp //= 10
    
    return armstrong_sum == n`,
    timeComplexity: 'O(log n) - number of digits',
    spaceComplexity: 'O(1)',
    order: 27,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-list-product',
    title: 'Product Except Self',
    difficulty: 'Medium',
    description: `Return an array where each element is the product of all elements except itself.

**Constraint:** Cannot use division operator.

**Example:** [1, 2, 3, 4] → [24, 12, 8, 6]

This problem tests:
- Array manipulation
- Prefix/suffix products
- Space optimization`,
    examples: [
      {
        input: 'nums = [1, 2, 3, 4]',
        output: '[24, 12, 8, 6]',
        explanation: '[2×3×4, 1×3×4, 1×2×4, 1×2×3]',
      },
      {
        input: 'nums = [-1, 1, 0, -3, 3]',
        output: '[0, 0, 9, 0, 0]',
      },
    ],
    constraints: ['2 <= len(nums) <= 10^5', 'Cannot use division'],
    hints: [
      'Calculate prefix products (left to right)',
      'Calculate suffix products (right to left)',
      'Multiply prefix and suffix for each position',
    ],
    starterCode: `def product_except_self(nums):
    """
    Return array of products except self.
    
    Args:
        nums: List of integers
        
    Returns:
        List where result[i] = product of all except nums[i]
        
    Examples:
        >>> product_except_self([1, 2, 3, 4])
        [24, 12, 8, 6]
    """
    pass`,
    testCases: [
      {
        input: [[1, 2, 3, 4]],
        expected: [24, 12, 8, 6],
      },
      {
        input: [[-1, 1, 0, -3, 3]],
        expected: [0, 0, 9, 0, 0],
      },
      {
        input: [[1, 2]],
        expected: [2, 1],
      },
    ],
    solution: `def product_except_self(nums):
    n = len(nums)
    result = [1] * n
    
    # Calculate prefix products (left to right)
    prefix = 1
    for i in range(n):
        result[i] = prefix
        prefix *= nums[i]
    
    # Calculate suffix products and multiply (right to left)
    suffix = 1
    for i in range(n - 1, -1, -1):
        result[i] *= suffix
        suffix *= nums[i]
    
    return result

# Two-pass approach with extra space
def product_except_self_verbose(nums):
    n = len(nums)
    
    # Build prefix products
    prefix = [1] * n
    for i in range(1, n):
        prefix[i] = prefix[i-1] * nums[i-1]
    
    # Build suffix products
    suffix = [1] * n
    for i in range(n-2, -1, -1):
        suffix[i] = suffix[i+1] * nums[i+1]
    
    # Multiply prefix and suffix
    return [prefix[i] * suffix[i] for i in range(n)]`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1) excluding output array',
    order: 28,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-longest-common-prefix',
    title: 'Longest Common Prefix',
    difficulty: 'Easy',
    description: `Find the longest common prefix string amongst an array of strings.

Return empty string if no common prefix.

**Example:** ["flower", "flow", "flight"] → "fl"

This problem tests:
- String comparison
- Iteration strategies
- Edge case handling`,
    examples: [
      {
        input: 'strs = ["flower", "flow", "flight"]',
        output: '"fl"',
      },
      {
        input: 'strs = ["dog", "racecar", "car"]',
        output: '""',
        explanation: 'No common prefix',
      },
    ],
    constraints: ['1 <= len(strs) <= 200', '0 <= len(strs[i]) <= 200'],
    hints: [
      'Compare characters at same position across all strings',
      'Stop when characters differ',
      'Handle empty strings',
    ],
    starterCode: `def longest_common_prefix(strs):
    """
    Find longest common prefix in array of strings.
    
    Args:
        strs: List of strings
        
    Returns:
        Longest common prefix string
        
    Examples:
        >>> longest_common_prefix(["flower", "flow", "flight"])
        "fl"
    """
    pass`,
    testCases: [
      {
        input: [['flower', 'flow', 'flight']],
        expected: 'fl',
      },
      {
        input: [['dog', 'racecar', 'car']],
        expected: '',
      },
      {
        input: [['']],
        expected: '',
      },
      {
        input: [['a']],
        expected: 'a',
      },
    ],
    solution: `def longest_common_prefix(strs):
    if not strs:
        return ""
    
    # Take first string as reference
    prefix = strs[0]
    
    # Compare with each string
    for string in strs[1:]:
        # Reduce prefix until it matches beginning of string
        while not string.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    
    return prefix

# Vertical scanning approach
def longest_common_prefix_vertical(strs):
    if not strs:
        return ""
    
    # Check each character position
    for i in range(len(strs[0])):
        char = strs[0][i]
        # Check if all strings have same char at position i
        for string in strs[1:]:
            if i >= len(string) or string[i] != char:
                return strs[0][:i]
    
    return strs[0]

# Using zip
def longest_common_prefix_zip(strs):
    if not strs:
        return ""
    
    result = []
    for chars in zip(*strs):
        if len(set(chars)) == 1:
            result.append(chars[0])
        else:
            break
    
    return ''.join(result)`,
    timeComplexity: 'O(S) where S is sum of all characters',
    spaceComplexity: 'O(1)',
    order: 29,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-title-case',
    title: 'Convert to Title Case',
    difficulty: 'Easy',
    description: `Convert a string to title case following these rules:
- First character of each word should be uppercase
- All other characters should be lowercase
- Handle multiple spaces between words

**Example:** "hello  world" → "Hello  World"

This problem tests:
- String manipulation
- Character case operations
- Space handling`,
    examples: [
      {
        input: 's = "hello world"',
        output: '"Hello World"',
      },
      {
        input: 's = "PYTHON programming"',
        output: '"Python Programming"',
      },
      {
        input: 's = "a  b  c"',
        output: '"A  B  C"',
        explanation: 'Preserves multiple spaces',
      },
    ],
    constraints: ['1 <= len(s) <= 10^4', 'Cannot use built-in title() method'],
    hints: [
      'Track if previous character was a space',
      'Capitalize first char and after spaces',
      'Preserve spacing',
    ],
    starterCode: `def to_title_case(s):
    """
    Convert string to title case.
    
    Args:
        s: Input string
        
    Returns:
        String in title case
        
    Examples:
        >>> to_title_case("hello world")
        "Hello World"
    """
    pass`,
    testCases: [
      {
        input: ['hello world'],
        expected: 'Hello World',
      },
      {
        input: ['PYTHON programming'],
        expected: 'Python Programming',
      },
      {
        input: ['a  b  c'],
        expected: 'A  B  C',
      },
      {
        input: ['the quick brown fox'],
        expected: 'The Quick Brown Fox',
      },
    ],
    solution: `def to_title_case(s):
    result = []
    capitalize_next = True
    
    for char in s:
        if char == ' ':
            result.append(char)
            capitalize_next = True
        elif capitalize_next:
            result.append(char.upper())
            capitalize_next = False
        else:
            result.append(char.lower())
    
    return ''.join(result)

# Alternative using enumerate
def to_title_case_alt(s):
    result = list(s.lower())
    
    # Capitalize first character
    if result and result[0] != ' ':
        result[0] = result[0].upper()
    
    # Capitalize after spaces
    for i in range(1, len(result)):
        if result[i-1] == ' ' and result[i] != ' ':
            result[i] = result[i].upper()
    
    return ''.join(result)`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(n)',
    order: 30,
    topic: 'Python Fundamentals',
  },
  // Batch 1: Problems 31-40
  ...pythonFundamentalsBatch1,
  // Batch 2: Problems 41-50
  ...pythonFundamentalsBatch2,
  // Batch 3: Problems 51-60
  ...pythonFundamentalsBatch3,
  // Batch 4: Problems 61-70
  ...pythonFundamentalsBatch4,
  // Batch 5: Problems 71-80
  ...pythonFundamentalsBatch5,
  // Batch 6: Problems 81-90
  ...pythonFundamentalsBatch6,
  // Batch 7: Problems 91-100
  ...pythonFundamentalsBatch7,
];
