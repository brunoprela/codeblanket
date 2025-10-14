/**
 * Python Fundamentals - Batch 3 (Problems 51-60)
 * New unique problems for Python fundamentals
 */

import { Problem } from '../types';

export const pythonFundamentalsBatch3: Problem[] = [
  {
    id: 'fundamentals-missing-ranges',
    title: 'Missing Ranges',
    difficulty: 'Easy',
    description: `Find missing ranges in a sorted array given a lower and upper bound.

Return list of missing ranges as strings in format "start->end" or just "num" for single number.

**Example:** nums=[0,1,3,50,75], lower=0, upper=99
Missing: ["2", "4->49", "51->74", "76->99"]

This tests:
- Range detection
- String formatting
- Edge case handling`,
    examples: [
      {
        input: 'nums=[0,1,3,50,75], lower=0, upper=99',
        output: '["2", "4->49", "51->74", "76->99"]',
      },
    ],
    constraints: ['-10^9 <= lower <= upper <= 10^9'],
    hints: [
      'Check gaps between consecutive numbers',
      'Handle start and end boundaries',
      'Single number vs range formatting',
    ],
    starterCode: `def find_missing_ranges(nums, lower, upper):
    """
    Find missing ranges in sorted array.
    
    Args:
        nums: Sorted array
        lower: Lower bound
        upper: Upper bound
        
    Returns:
        List of missing range strings
        
    Examples:
        >>> find_missing_ranges([0,1,3,50,75], 0, 99)
        ["2", "4->49", "51->74", "76->99"]
    """
    pass


# Test
print(find_missing_ranges([0,1,3,50,75], 0, 99))
`,
    testCases: [
      {
        input: [[0, 1, 3, 50, 75], 0, 99],
        expected: ['2', '4->49', '51->74', '76->99'],
      },
      {
        input: [[], 1, 1],
        expected: ['1'],
      },
    ],
    solution: `def find_missing_ranges(nums, lower, upper):
    def format_range(start, end):
        return str(start) if start == end else f"{start}->{end}"
    
    result = []
    prev = lower - 1
    
    for num in nums + [upper + 1]:
        if num > prev + 1:
            result.append(format_range(prev + 1, num - 1))
        prev = num
    
    return result`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    order: 51,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-rotate-string',
    title: 'Rotate String',
    difficulty: 'Easy',
    description: `Check if string s can become goal after some rotations.

A rotation shifts characters from left to right:
- "abcde" → "cdeab" (shifted 2)

**Trick:** s can rotate to goal if goal is substring of s+s

**Example:** "abcde" → "cdeab" (yes), "abcde" → "abced" (no)

This tests:
- String manipulation
- Pattern matching
- Clever observations`,
    examples: [
      {
        input: 's = "abcde", goal = "cdeab"',
        output: 'True',
      },
      {
        input: 's = "abcde", goal = "abced"',
        output: 'False',
      },
    ],
    constraints: ['1 <= len(s), len(goal) <= 100', 'Same length'],
    hints: [
      'Check if lengths are equal first',
      'goal in (s + s) checks all rotations',
      'Or manually try each rotation',
    ],
    starterCode: `def rotate_string(s, goal):
    """
    Check if s can rotate to goal.
    
    Args:
        s: Original string
        goal: Target string
        
    Returns:
        True if rotation possible
        
    Examples:
        >>> rotate_string("abcde", "cdeab")
        True
    """
    pass


# Test
print(rotate_string("abcde", "cdeab"))
`,
    testCases: [
      {
        input: ['abcde', 'cdeab'],
        expected: true,
      },
      {
        input: ['abcde', 'abced'],
        expected: false,
      },
    ],
    solution: `def rotate_string(s, goal):
    return len(s) == len(goal) and goal in s + s


# Alternative: try each rotation
def rotate_string_explicit(s, goal):
    if len(s) != len(goal):
        return False
    
    for i in range(len(s)):
        if s[i:] + s[:i] == goal:
            return True
    
    return False`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(n)',
    order: 52,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-reverse-vowels',
    title: 'Reverse Vowels Only',
    difficulty: 'Easy',
    description: `Reverse only the vowels in a string, keeping consonants in place.

Vowels: a, e, i, o, u (both cases)

**Example:** "hello" → "holle"
- Vowels: e, o
- Reversed: o, e
- Result: h + o + l + l + e

This tests:
- Two pointer technique
- Character swapping
- Vowel identification`,
    examples: [
      {
        input: 's = "hello"',
        output: '"holle"',
      },
      {
        input: 's = "leetcode"',
        output: '"leotcede"',
      },
    ],
    constraints: ['1 <= len(s) <= 3*10^5'],
    hints: [
      'Use two pointers from both ends',
      'Move pointers to next vowel',
      'Swap when both point to vowels',
    ],
    starterCode: `def reverse_vowels(s):
    """
    Reverse only the vowels in string.
    
    Args:
        s: Input string
        
    Returns:
        String with vowels reversed
        
    Examples:
        >>> reverse_vowels("hello")
        "holle"
    """
    pass


# Test
print(reverse_vowels("leetcode"))
`,
    testCases: [
      {
        input: ['hello'],
        expected: 'holle',
      },
      {
        input: ['leetcode'],
        expected: 'leotcede',
      },
    ],
    solution: `def reverse_vowels(s):
    vowels = set('aeiouAEIOU')
    chars = list(s)
    left, right = 0, len(s) - 1
    
    while left < right:
        if chars[left] not in vowels:
            left += 1
        elif chars[right] not in vowels:
            right -= 1
        else:
            chars[left], chars[right] = chars[right], chars[left]
            left += 1
            right -= 1
    
    return ''.join(chars)`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(n)',
    order: 53,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-first-unique-char',
    title: 'First Unique Character',
    difficulty: 'Easy',
    description: `Find the index of first non-repeating character in a string.

Return -1 if all characters repeat.

**Example:** "leetcode" → 0 (l appears once)
"loveleetcode" → 2 (v appears once)

This tests:
- Character frequency
- First occurrence tracking
- Two-pass algorithm`,
    examples: [
      {
        input: 's = "leetcode"',
        output: '0',
      },
      {
        input: 's = "loveleetcode"',
        output: '2',
      },
      {
        input: 's = "aabb"',
        output: '-1',
      },
    ],
    constraints: ['1 <= len(s) <= 10^5', 'Only lowercase letters'],
    hints: [
      'Count character frequencies first',
      'Then find first char with count=1',
      'Use Counter or dictionary',
    ],
    starterCode: `def first_uniq_char(s):
    """
    Find index of first unique character.
    
    Args:
        s: Input string
        
    Returns:
        Index of first unique char or -1
        
    Examples:
        >>> first_uniq_char("leetcode")
        0
    """
    pass


# Test
print(first_uniq_char("loveleetcode"))
`,
    testCases: [
      {
        input: ['leetcode'],
        expected: 0,
      },
      {
        input: ['loveleetcode'],
        expected: 2,
      },
      {
        input: ['aabb'],
        expected: -1,
      },
    ],
    solution: `def first_uniq_char(s):
    from collections import Counter
    
    counts = Counter(s)
    
    for i, char in enumerate(s):
        if counts[char] == 1:
            return i
    
    return -1


# Alternative without Counter
def first_uniq_char_dict(s):
    char_count = {}
    
    for char in s:
        char_count[char] = char_count.get(char, 0) + 1
    
    for i, char in enumerate(s):
        if char_count[char] == 1:
            return i
    
    return -1`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1) - at most 26 letters',
    order: 54,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-ransom-note',
    title: 'Ransom Note',
    difficulty: 'Easy',
    description: `Check if ransom note can be constructed from magazine characters.

Each character in magazine can be used only once.

**Example:** 
- ransomNote = "aa", magazine = "aab" → true
- ransomNote = "aa", magazine = "ab" → false

This tests:
- Character counting
- Availability checking
- Counter operations`,
    examples: [
      {
        input: 'ransomNote = "aa", magazine = "aab"',
        output: 'True',
      },
      {
        input: 'ransomNote = "aa", magazine = "ab"',
        output: 'False',
      },
    ],
    constraints: ['1 <= len(ransomNote), len(magazine) <= 10^5'],
    hints: [
      'Count characters in magazine',
      'Check if each ransom char available',
      'Use Counter subtraction',
    ],
    starterCode: `def can_construct(ransom_note, magazine):
    """
    Check if ransom note can be made from magazine.
    
    Args:
        ransom_note: Note to construct
        magazine: Available characters
        
    Returns:
        True if constructible
        
    Examples:
        >>> can_construct("aa", "aab")
        True
    """
    pass


# Test
print(can_construct("aa", "aab"))
`,
    testCases: [
      {
        input: ['aa', 'aab'],
        expected: true,
      },
      {
        input: ['aa', 'ab'],
        expected: false,
      },
    ],
    solution: `def can_construct(ransom_note, magazine):
    from collections import Counter
    
    ransom_count = Counter(ransom_note)
    magazine_count = Counter(magazine)
    
    for char, count in ransom_count.items():
        if magazine_count[char] < count:
            return False
    
    return True


# Alternative using subtraction
def can_construct_subtract(ransom_note, magazine):
    from collections import Counter
    
    ransom_count = Counter(ransom_note)
    magazine_count = Counter(magazine)
    
    return not (ransom_count - magazine_count)`,
    timeComplexity: 'O(m + n)',
    spaceComplexity: 'O(1) - at most 26 letters',
    order: 55,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-length-last-word',
    title: 'Length of Last Word',
    difficulty: 'Easy',
    description: `Return the length of the last word in a string.

A word is a maximal substring of non-space characters.

**Example:** "Hello World" → 5
"   fly me   to   the moon  " → 4

This tests:
- String trimming
- Word splitting
- Edge case handling`,
    examples: [
      {
        input: 's = "Hello World"',
        output: '5',
      },
      {
        input: 's = "   fly me   to   the moon  "',
        output: '4',
      },
    ],
    constraints: ['1 <= len(s) <= 10^4', 'Letters and spaces only'],
    hints: [
      'Strip trailing spaces',
      'Find last space position',
      'Or use split() and get last word',
    ],
    starterCode: `def length_of_last_word(s):
    """
    Find length of last word.
    
    Args:
        s: Input string
        
    Returns:
        Length of last word
        
    Examples:
        >>> length_of_last_word("Hello World")
        5
    """
    pass


# Test
print(length_of_last_word("   fly me   to   the moon  "))
`,
    testCases: [
      {
        input: ['Hello World'],
        expected: 5,
      },
      {
        input: ['   fly me   to   the moon  '],
        expected: 4,
      },
    ],
    solution: `def length_of_last_word(s):
    return len(s.split()[-1])


# Alternative without split
def length_of_last_word_manual(s):
    s = s.rstrip()
    length = 0
    
    for i in range(len(s) - 1, -1, -1):
        if s[i] != ' ':
            length += 1
        else:
            break
    
    return length`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    order: 56,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-is-isomorphic',
    title: 'Isomorphic Strings',
    difficulty: 'Easy',
    description: `Check if two strings are isomorphic.

Strings are isomorphic if characters in s can be replaced to get t, maintaining:
- One-to-one character mapping
- Character order preserved

**Example:** "egg" and "add" → true (e→a, g→d)
"foo" and "bar" → false (o can't map to both a and r)

This tests:
- Character mapping
- Bidirectional checking
- Hash map usage`,
    examples: [
      {
        input: 's = "egg", t = "add"',
        output: 'True',
      },
      {
        input: 's = "foo", t = "bar"',
        output: 'False',
      },
    ],
    constraints: ['1 <= len(s), len(t) <= 5*10^4', 'Same length'],
    hints: [
      'Map each char in s to char in t',
      'Also ensure reverse mapping is unique',
      'Use two dictionaries',
    ],
    starterCode: `def is_isomorphic(s, t):
    """
    Check if strings are isomorphic.
    
    Args:
        s: First string
        t: Second string
        
    Returns:
        True if isomorphic
        
    Examples:
        >>> is_isomorphic("egg", "add")
        True
    """
    pass


# Test
print(is_isomorphic("egg", "add"))
`,
    testCases: [
      {
        input: ['egg', 'add'],
        expected: true,
      },
      {
        input: ['foo', 'bar'],
        expected: false,
      },
    ],
    solution: `def is_isomorphic(s, t):
    if len(s) != len(t):
        return False
    
    map_s_to_t = {}
    map_t_to_s = {}
    
    for char_s, char_t in zip(s, t):
        if char_s in map_s_to_t:
            if map_s_to_t[char_s] != char_t:
                return False
        else:
            map_s_to_t[char_s] = char_t
        
        if char_t in map_t_to_s:
            if map_t_to_s[char_t] != char_s:
                return False
        else:
            map_t_to_s[char_t] = char_s
    
    return True`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1) - at most 256 ASCII chars',
    order: 57,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-word-pattern',
    title: 'Word Pattern',
    difficulty: 'Easy',
    description: `Check if a pattern matches a string following the same pattern.

Each character in pattern maps to a word in str, and vice versa.

**Example:** pattern="abba", str="dog cat cat dog" → true
pattern="abba", str="dog cat cat fish" → false

This tests:
- Bijective mapping
- String splitting
- Pattern matching`,
    examples: [
      {
        input: 'pattern = "abba", s = "dog cat cat dog"',
        output: 'True',
      },
      {
        input: 'pattern = "abba", s = "dog cat cat fish"',
        output: 'False',
      },
    ],
    constraints: ['1 <= len(pattern) <= 300', '1 <= len(s) <= 3000'],
    hints: [
      'Split string into words',
      'Check lengths match',
      'Use bidirectional mapping like isomorphic',
    ],
    starterCode: `def word_pattern(pattern, s):
    """
    Check if pattern matches string.
    
    Args:
        pattern: Pattern string
        s: Space-separated words
        
    Returns:
        True if pattern matches
        
    Examples:
        >>> word_pattern("abba", "dog cat cat dog")
        True
    """
    pass


# Test
print(word_pattern("abba", "dog cat cat dog"))
`,
    testCases: [
      {
        input: ['abba', 'dog cat cat dog'],
        expected: true,
      },
      {
        input: ['abba', 'dog cat cat fish'],
        expected: false,
      },
    ],
    solution: `def word_pattern(pattern, s):
    words = s.split()
    
    if len(pattern) != len(words):
        return False
    
    char_to_word = {}
    word_to_char = {}
    
    for char, word in zip(pattern, words):
        if char in char_to_word:
            if char_to_word[char] != word:
                return False
        else:
            char_to_word[char] = word
        
        if word in word_to_char:
            if word_to_char[word] != char:
                return False
        else:
            word_to_char[word] = char
    
    return True`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(n)',
    order: 58,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-ugly-number',
    title: 'Ugly Number',
    difficulty: 'Easy',
    description: `Check if a number is ugly.

An ugly number is a positive integer whose prime factors are limited to 2, 3, and 5.

**Example:** 
- 6 = 2 × 3 → ugly
- 14 = 2 × 7 → not ugly (contains 7)

This tests:
- Prime factorization
- Division operations
- Number theory`,
    examples: [
      {
        input: 'n = 6',
        output: 'True',
        explanation: '6 = 2 × 3',
      },
      {
        input: 'n = 14',
        output: 'False',
        explanation: 'Contains prime factor 7',
      },
    ],
    constraints: ['-2^31 <= n <= 2^31 - 1'],
    hints: [
      'Divide by 2, 3, 5 repeatedly',
      'If result is 1, number is ugly',
      'Negative numbers are not ugly',
    ],
    starterCode: `def is_ugly(n):
    """
    Check if number is ugly.
    
    Args:
        n: Integer to check
        
    Returns:
        True if ugly number
        
    Examples:
        >>> is_ugly(6)
        True
        >>> is_ugly(14)
        False
    """
    pass


# Test
print(is_ugly(6))
`,
    testCases: [
      {
        input: [6],
        expected: true,
      },
      {
        input: [14],
        expected: false,
      },
      {
        input: [1],
        expected: true,
      },
    ],
    solution: `def is_ugly(n):
    if n <= 0:
        return False
    
    for factor in [2, 3, 5]:
        while n % factor == 0:
            n //= factor
    
    return n == 1`,
    timeComplexity: 'O(log n)',
    spaceComplexity: 'O(1)',
    order: 59,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-add-digits',
    title: 'Add Digits Until Single Digit',
    difficulty: 'Easy',
    description: `Repeatedly add all digits until a single digit remains.

**Example:** 38 → 3+8=11 → 1+1=2

**Follow-up:** Can you do it in O(1) without loops?
**Hint:** Digital root = 1 + ((n-1) % 9)

This tests:
- Digit extraction
- Loop iteration
- Mathematical insight (digital root)`,
    examples: [
      {
        input: 'n = 38',
        output: '2',
        explanation: '3+8=11, 1+1=2',
      },
      {
        input: 'n = 0',
        output: '0',
      },
    ],
    constraints: ['0 <= n <= 2^31 - 1'],
    hints: [
      'Extract and sum digits repeatedly',
      'Stop when result < 10',
      'O(1) solution: digital root formula',
    ],
    starterCode: `def add_digits(n):
    """
    Add digits until single digit remains.
    
    Args:
        n: Non-negative integer
        
    Returns:
        Single digit result
        
    Examples:
        >>> add_digits(38)
        2
    """
    pass


# Test
print(add_digits(38))
`,
    testCases: [
      {
        input: [38],
        expected: 2,
      },
      {
        input: [0],
        expected: 0,
      },
      {
        input: [99],
        expected: 9,
      },
    ],
    solution: `def add_digits(n):
    while n >= 10:
        digit_sum = 0
        while n > 0:
            digit_sum += n % 10
            n //= 10
        n = digit_sum
    
    return n


# O(1) solution using digital root
def add_digits_constant(n):
    if n == 0:
        return 0
    return 1 + ((n - 1) % 9)`,
    timeComplexity: 'O(log n) or O(1) with formula',
    spaceComplexity: 'O(1)',
    order: 60,
    topic: 'Python Fundamentals',
  },
];
