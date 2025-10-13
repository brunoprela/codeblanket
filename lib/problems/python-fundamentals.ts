/**
 * Python Fundamentals problems - Testing basic Python concepts
 */

import { Problem } from '../types';

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
    starterCode: `def count_words(text):
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
];
