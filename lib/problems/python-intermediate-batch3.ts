/**
 * Python Intermediate - New Problems Batch 3 (42-50)
 * 9 problems to reach 50 total
 */

import { Problem } from '../types';

export const pythonIntermediateBatch3: Problem[] = [
  {
    id: 'intermediate-dict-get-default',
    title: 'Dict get() with Default Value',
    difficulty: 'Easy',
    description: `Use dict.get() to safely access dictionary values with defaults.

**Syntax:**
\`\`\`python
value = d.get(key, default)
\`\`\`

Safer than d[key] which raises KeyError.

This tests:
- Safe dict access
- Default values
- Avoiding KeyError`,
    examples: [
      {
        input: 'd.get("missing", 0)',
        output: '0 (no KeyError)',
      },
    ],
    constraints: ['Use .get() method', 'Provide defaults'],
    hints: [
      'dict.get(key, default)',
      'Returns default if key missing',
      'No exception raised',
    ],
    starterCode: `def count_items(items):
    """
    Count occurrences of each item.
    
    Args:
        items: List of items
        
    Returns:
        Dict of item: count
        
    Examples:
        >>> count_items(['a', 'b', 'a', 'c', 'b', 'a'])
        {'a': 3, 'b': 2, 'c': 1}
    """
    counts = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return counts


# Test
print(count_items(['x', 'y', 'x', 'z', 'y', 'x']))
`,
    testCases: [
      {
        input: [['x', 'y', 'x', 'z', 'y', 'x']],
        expected: { x: 3, y: 2, z: 1 },
      },
      {
        input: [['a', 'a', 'a']],
        expected: { a: 3 },
      },
    ],
    solution: `def count_items(items):
    counts = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return counts


# Alternative with setdefault
def count_items_setdefault(items):
    counts = {}
    for item in items:
        counts.setdefault(item, 0)
        counts[item] += 1
    return counts`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(k) where k is unique items',
    order: 42,
    topic: 'Python Intermediate',
  },
  {
    id: 'intermediate-string-join',
    title: 'Efficient String Joining',
    difficulty: 'Easy',
    description: `Use str.join() for efficient string concatenation.

**Why join():**
- Strings are immutable
- += creates new string each time (O(n²))
- join() is O(n)

This tests:
- String operations
- Performance awareness
- join() method`,
    examples: [
      {
        input: 'words = ["Hello", "World"]',
        output: '" ".join(words) = "Hello World"',
      },
    ],
    constraints: ['Use join()', 'More efficient than +='],
    hints: [
      'separator.join(list)',
      'Works with any iterable',
      'Faster than +=',
    ],
    starterCode: `def build_sentence(words):
    """
    Join words into sentence.
    
    Args:
        words: List of words
        
    Returns:
        Sentence string
        
    Examples:
        >>> build_sentence(['Hello', 'world', 'today'])
        'Hello world today'
    """
    pass


# Test
print(build_sentence(['Python', 'is', 'awesome']))
`,
    testCases: [
      {
        input: [['Python', 'is', 'awesome']],
        expected: 'Python is awesome',
      },
      {
        input: [['a', 'b', 'c']],
        expected: 'a b c',
      },
    ],
    solution: `def build_sentence(words):
    return ' '.join(words)


# Different separators
def build_csv(values):
    return ','.join(str(v) for v in values)

def build_path(parts):
    return '/'.join(parts)`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(n)',
    order: 43,
    topic: 'Python Intermediate',
  },
  {
    id: 'intermediate-set-operations',
    title: 'Set Operations (Union, Intersection, Difference)',
    difficulty: 'Easy',
    description: `Use set operations for efficient collection comparisons.

**Operations:**
- union: a | b or a.union(b)
- intersection: a & b or a.intersection(b)
- difference: a - b or a.difference(b)
- symmetric_difference: a ^ b

This tests:
- Set operations
- Mathematical sets
- Efficient lookups`,
    examples: [
      {
        input: 'a = {1,2,3}, b = {2,3,4}',
        output: 'a & b = {2, 3}',
      },
    ],
    constraints: ['Use set operations', 'Operators or methods'],
    hints: ['| for union', '& for intersection', '- for difference'],
    starterCode: `def find_common_and_unique(list1, list2):
    """
    Find common and unique elements.
    
    Args:
        list1, list2: Lists of items
        
    Returns:
        Tuple of (common, only_in_list1, only_in_list2)
        
    Examples:
        >>> find_common_and_unique([1,2,3], [2,3,4])
        ([2, 3], [1], [4])
    """
    set1 = set(list1)
    set2 = set(list2)
    
    common = set1 & set2
    only1 = set1 - set2
    only2 = set2 - set1
    
    return (sorted(common), sorted(only1), sorted(only2))


# Test
print(find_common_and_unique([1,2,3,4], [3,4,5,6]))
`,
    testCases: [
      {
        input: [
          [1, 2, 3, 4],
          [3, 4, 5, 6],
        ],
        expected: [
          [3, 4],
          [1, 2],
          [5, 6],
        ],
      },
      {
        input: [
          [1, 2],
          [2, 3],
        ],
        expected: [[2], [1], [3]],
      },
    ],
    solution: `def find_common_and_unique(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    
    common = set1 & set2
    only1 = set1 - set2
    only2 = set2 - set1
    
    return (sorted(common), sorted(only1), sorted(only2))`,
    timeComplexity: 'O(n + m)',
    spaceComplexity: 'O(n + m)',
    order: 44,
    topic: 'Python Intermediate',
  },
  {
    id: 'intermediate-dict-comprehension-filter',
    title: 'Dict Comprehension with Filtering',
    difficulty: 'Easy',
    description: `Use dict comprehensions with conditional filtering.

**Syntax:**
\`\`\`python
{k: v for k, v in items if condition}
\`\`\`

This tests:
- Dict comprehensions
- Filtering
- Key-value transformation`,
    examples: [
      {
        input: 'Filter dict by value > 10',
        output: 'New dict with only matching items',
      },
    ],
    constraints: ['Use dict comprehension', 'Add condition'],
    hints: ['{k: v for ...}', 'Add if condition', 'Can transform k and v'],
    starterCode: `def filter_scores(scores, min_score):
    """
    Filter scores above minimum.
    
    Args:
        scores: Dict of name: score
        min_score: Minimum passing score
        
    Returns:
        Dict with only passing scores
        
    Examples:
        >>> filter_scores({'Alice': 85, 'Bob': 65, 'Charlie': 95}, 70)
        {'Alice': 85, 'Charlie': 95}
    """
    pass


# Test
print(filter_scores({'A': 90, 'B': 60, 'C': 75, 'D': 50}, 70))
`,
    testCases: [
      {
        input: [{ A: 90, B: 60, C: 75, D: 50 }, 70],
        expected: { A: 90, C: 75 },
      },
      {
        input: [{ x: 10, y: 20, z: 5 }, 10],
        expected: { x: 10, y: 20 },
      },
    ],
    solution: `def filter_scores(scores, min_score):
    return {name: score for name, score in scores.items() if score >= min_score}


# Transform keys and values
def uppercase_keys_double_values(d):
    return {k.upper(): v * 2 for k, v in d.items()}`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(n)',
    order: 45,
    topic: 'Python Intermediate',
  },
  {
    id: 'intermediate-lambda-sorting',
    title: 'Lambda Functions for Sorting',
    difficulty: 'Easy',
    description: `Use lambda functions as sort keys.

**Syntax:**
\`\`\`python
sorted(items, key=lambda x: x[1])
\`\`\`

Lambda is useful for simple inline functions.

This tests:
- Lambda expressions
- Sorting with key
- Anonymous functions`,
    examples: [
      {
        input: 'Sort list of tuples by second element',
        output: 'key=lambda x: x[1]',
      },
    ],
    constraints: ['Use lambda', 'Sort by custom key'],
    hints: [
      'lambda args: expression',
      'Use as sort key',
      'Can access tuple/list elements',
    ],
    starterCode: `def sort_by_length_then_alpha(words):
    """
    Sort words by length, then alphabetically.
    
    Args:
        words: List of strings
        
    Returns:
        Sorted list
        
    Examples:
        >>> sort_by_length_then_alpha(['apple', 'pie', 'banana', 'cat'])
        ['cat', 'pie', 'apple', 'banana']
    """
    pass


# Test
print(sort_by_length_then_alpha(['dog', 'cat', 'bird', 'elephant']))
`,
    testCases: [
      {
        input: [['dog', 'cat', 'bird', 'elephant']],
        expected: ['cat', 'dog', 'bird', 'elephant'],
      },
      {
        input: [['xx', 'a', 'z', 'yy']],
        expected: ['a', 'z', 'xx', 'yy'],
      },
    ],
    solution: `def sort_by_length_then_alpha(words):
    return sorted(words, key=lambda w: (len(w), w))


# Other examples
def sort_by_last_char(words):
    return sorted(words, key=lambda w: w[-1])

def sort_tuples_by_second(tuples):
    return sorted(tuples, key=lambda t: t[1])`,
    timeComplexity: 'O(n log n)',
    spaceComplexity: 'O(n)',
    order: 46,
    topic: 'Python Intermediate',
  },
  {
    id: 'intermediate-try-except-multiple',
    title: 'Multiple Exception Types',
    difficulty: 'Easy',
    description: `Handle different exception types separately.

**Syntax:**
\`\`\`python
try:
    code()
except ValueError:
    handle_value_error()
except TypeError:
    handle_type_error()
\`\`\`

Or catch multiple:
\`\`\`python
except (ValueError, TypeError):
    handle_both()
\`\`\`

This tests:
- Exception handling
- Multiple except blocks
- Exception specificity`,
    examples: [
      {
        input: 'Try parsing user input',
        output: 'Different handling per error type',
      },
    ],
    constraints: [
      'Handle multiple exception types',
      'Specific handling per type',
    ],
    hints: [
      'Multiple except blocks',
      'Or tuple of exceptions',
      'More specific first',
    ],
    starterCode: `def safe_divide_and_convert(a, b):
    """
    Divide and convert to int, handling errors.
    
    Args:
        a, b: Values to divide
        
    Returns:
        Result or error message
        
    Examples:
        >>> safe_divide_and_convert(10, 2)
        5
        >>> safe_divide_and_convert(10, 0)
        'Error: Cannot divide by zero'
    """
    try:
        result = a / b
        return int(result)
    except ZeroDivisionError:
        return "Error: Cannot divide by zero"
    except (TypeError, ValueError):
        return "Error: Invalid input types"


# Test
print(safe_divide_and_convert(15, 3))
`,
    testCases: [
      {
        input: [15, 3],
        expected: 5,
      },
      {
        input: [10, 0],
        expected: 'Error: Cannot divide by zero',
      },
    ],
    solution: `def safe_divide_and_convert(a, b):
    try:
        result = a / b
        return int(result)
    except ZeroDivisionError:
        return "Error: Cannot divide by zero"
    except (TypeError, ValueError):
        return "Error: Invalid input types"`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 47,
    topic: 'Python Intermediate',
  },
  {
    id: 'intermediate-unpacking-extended',
    title: 'Extended Unpacking with *',
    difficulty: 'Medium',
    description: `Use * for extended unpacking in assignments.

**Syntax:**
\`\`\`python
first, *middle, last = [1, 2, 3, 4, 5]
# first = 1, middle = [2, 3, 4], last = 5
\`\`\`

This tests:
- Extended unpacking
- * operator
- Variable assignment`,
    examples: [
      {
        input: 'first, *rest = [1, 2, 3, 4]',
        output: 'first = 1, rest = [2, 3, 4]',
      },
    ],
    constraints: ['Use * unpacking', 'Multiple variables'],
    hints: [
      'a, *b, c = list',
      '* captures multiple items',
      'Can be in any position',
    ],
    starterCode: `def split_first_last_middle(items):
    """
    Split list into first, middle, and last.
    
    Args:
        items: List with at least 2 items
        
    Returns:
        Tuple of (first, middle, last)
        
    Examples:
        >>> split_first_last_middle([1, 2, 3, 4, 5])
        (1, [2, 3, 4], 5)
    """
    first, *middle, last = items
    return (first, middle, last)


# Test
print(split_first_last_middle([1, 2, 3, 4, 5, 6]))
`,
    testCases: [
      {
        input: [[1, 2, 3, 4, 5, 6]],
        expected: [1, [2, 3, 4, 5], 6],
      },
      {
        input: [[10, 20, 30]],
        expected: [10, [20], 30],
      },
    ],
    solution: `def split_first_last_middle(items):
    first, *middle, last = items
    return (first, middle, last)


# Other examples
def get_first_and_rest(items):
    first, *rest = items
    return (first, rest)

def get_all_but_last(items):
    *all_but_last, last = items
    return all_but_last`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(n)',
    order: 48,
    topic: 'Python Intermediate',
  },
  {
    id: 'intermediate-reversed-builtin',
    title: 'reversed() Function',
    difficulty: 'Easy',
    description: `Use reversed() to iterate in reverse without copying.

reversed() returns an iterator, not a list.

**Benefits:**
- Memory efficient
- Works with any sequence
- No copying

This tests:
- reversed() function
- Iterator usage
- Reverse iteration`,
    examples: [
      {
        input: 'reversed([1, 2, 3])',
        output: 'Iterator yielding 3, 2, 1',
      },
    ],
    constraints: ['Use reversed()', 'Iterator, not list'],
    hints: [
      'reversed(sequence)',
      'Returns iterator',
      'Convert to list if needed',
    ],
    starterCode: `def reverse_string_words(sentence):
    """
    Reverse order of words in sentence.
    
    Args:
        sentence: String with words
        
    Returns:
        String with reversed word order
        
    Examples:
        >>> reverse_string_words("Hello world Python")
        "Python world Hello"
    """
    words = sentence.split()
    reversed_words = reversed(words)
    return ' '.join(reversed_words)


# Test
print(reverse_string_words("I love Python programming"))
`,
    testCases: [
      {
        input: ['I love Python programming'],
        expected: 'programming Python love I',
      },
      {
        input: ['Hello world'],
        expected: 'world Hello',
      },
    ],
    solution: `def reverse_string_words(sentence):
    words = sentence.split()
    return ' '.join(reversed(words))


# Alternative (more concise)
def reverse_string_words_alt(sentence):
    return ' '.join(sentence.split()[::-1])`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(n)',
    order: 49,
    topic: 'Python Intermediate',
  },
  {
    id: 'intermediate-sum-with-start',
    title: 'sum() with Start Value',
    difficulty: 'Easy',
    description: `Use sum() with custom start value.

**Syntax:**
\`\`\`python
sum(iterable, start=0)
\`\`\`

Start value is added to the sum.

This tests:
- sum() function
- start parameter
- Accumulation`,
    examples: [
      {
        input: 'sum([1, 2, 3], start=10)',
        output: '16',
      },
    ],
    constraints: ['Use sum()', 'Provide start value'],
    hints: ['sum(iterable, start)', 'Default start is 0', 'Useful for offsets'],
    starterCode: `def calculate_total_with_tax(prices, tax_rate):
    """
    Calculate total with tax.
    
    Args:
        prices: List of prices
        tax_rate: Tax rate (e.g., 0.1 for 10%)
        
    Returns:
        Total with tax
        
    Examples:
        >>> calculate_total_with_tax([10, 20, 30], 0.1)
        66.0
    """
    subtotal = sum(prices)
    tax = subtotal * tax_rate
    total = subtotal + tax
    return total


# Test
print(calculate_total_with_tax([100, 200], 0.05))
`,
    testCases: [
      {
        input: [[100, 200], 0.05],
        expected: 315.0,
      },
      {
        input: [[10, 20, 30], 0.1],
        expected: 66.0,
      },
    ],
    solution: `def calculate_total_with_tax(prices, tax_rate):
    subtotal = sum(prices)
    tax = subtotal * tax_rate
    return subtotal + tax


# Using sum with start for different approach
def sum_with_bonus(values, bonus):
    return sum(values, start=bonus)`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    order: 50,
    topic: 'Python Intermediate',
  },
];
