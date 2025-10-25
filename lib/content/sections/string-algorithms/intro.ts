/**
 * Introduction to String Algorithms Section
 */

export const introSection = {
  id: 'intro',
  title: 'Introduction to String Algorithms',
  content: `# String Algorithms

Strings are fundamental in programming and appear in ~30-40% of coding interviews. Mastering string algorithms is essential for:
- **Pattern matching** - Find occurrences of patterns in text
- **String processing** - Manipulate, transform, analyze strings efficiently
- **Text analysis** - Palindromes, anagrams, subsequences
- **Parsing** - Process structured text data

## Why String Algorithms Matter

**1. Ubiquitous in interviews:**
- FAANG companies frequently test string manipulation
- Tests problem-solving, optimization, edge case handling

**2. Real-world applications:**
- Search engines (pattern matching)
- DNA sequencing (longest common subsequence)
- Plagiarism detection (edit distance)
- Auto-complete (tries, prefix matching)

**3. Foundation for advanced topics:**
- Dynamic programming on strings
- Regular expressions
- Compiler design (lexical analysis)

## Core String Operations in Python

### Basic Operations

\`\`\`python
s = "hello world"

# Length
len (s)  # 11

# Indexing (O(1))
s[0]    # 'h'
s[-1]   # 'd'

# Slicing (O(k) where k is slice length)
s[0:5]     # 'hello'
s[6:]      # 'world'
s[::-1]    # 'dlrow olleh' - reverse

# Concatenation (O(n+m))
"hello" + " " + "world"  # Creates new string

# Repetition
"ab" * 3  # 'ababab'
\`\`\`

### String Methods (Important for Interviews)

\`\`\`python
s = "Hello World"

# Case conversion
s.lower()      # 'hello world'
s.upper()      # 'HELLO WORLD'
s.capitalize() # 'Hello world'

# Search
s.find('o')      # 4 (first occurrence, -1 if not found)
s.index('o')     # 4 (raises ValueError if not found)
s.count('o')     # 2

# Checking
s.startswith('He')  # True
s.endswith('ld')    # True
s.isalpha()         # False (has space)
s.isdigit()         # False
s.isalnum()         # False

# Splitting/Joining
s.split()           # ['Hello', 'World']
'-'.join(['a', 'b', 'c'])  # 'a-b-c'

# Strip whitespace
"  hello  ".strip()   # 'hello'
"  hello  ".lstrip()  # 'hello  '
"  hello  ".rstrip()  # '  hello'

# Replace
s.replace('o', '0')   # 'Hell0 W0rld'
\`\`\`

### Important: Strings are Immutable

\`\`\`python
# ❌ BAD: Inefficient string concatenation in loop
result = ""
for char in "hello":
    result += char  # O(n²) - creates new string each time!

# ✅ GOOD: Use list + join
chars = []
for char in "hello":
    chars.append (char)
result = '.join (chars)  # O(n)

# Or use list comprehension
result = '.join([char for char in "hello"])
\`\`\`

## Common String Patterns

### 1. Two Pointers
Moving from both ends toward center.

\`\`\`python
def is_palindrome (s):
    left, right = 0, len (s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True
\`\`\`

### 2. Sliding Window
Fixed or variable window moving through string.

\`\`\`python
def max_vowels_in_substring (s, k):
    """Find max vowels in any substring of length k"""
    vowels = set('aeiou')
    
    # Initial window
    current_vowels = sum(1 for c in s[:k] if c in vowels)
    max_vowels = current_vowels
    
    # Slide window
    for i in range (k, len (s)):
        # Remove left character
        if s[i-k] in vowels:
            current_vowels -= 1
        # Add right character
        if s[i] in vowels:
            current_vowels += 1
        max_vowels = max (max_vowels, current_vowels)
    
    return max_vowels
\`\`\`

### 3. Hash Map / Frequency Count
Track character frequencies.

\`\`\`python
from collections import Counter

def is_anagram (s1, s2):
    return Counter (s1) == Counter (s2)

# Or manually
def char_frequency (s):
    freq = {}
    for char in s:
        freq[char] = freq.get (char, 0) + 1
    return freq
\`\`\`

### 4. Dynamic Programming
Build solutions for subproblems.

\`\`\`python
def longest_palindromic_substring_length (s):
    n = len (s)
    if n == 0:
        return 0
    
    # dp[i][j] = True if s[i:j+1] is palindrome
    dp = [[False] * n for _ in range (n)]
    max_length = 1
    
    # Single characters
    for i in range (n):
        dp[i][i] = True
    
    # Two characters
    for i in range (n - 1):
        if s[i] == s[i + 1]:
            dp[i][i + 1] = True
            max_length = 2
    
    # Longer substrings
    for length in range(3, n + 1):
        for i in range (n - length + 1):
            j = i + length - 1
            if s[i] == s[j] and dp[i + 1][j - 1]:
                dp[i][j] = True
                max_length = length
    
    return max_length
\`\`\`

## Time Complexity Quick Reference

| Operation | Time | Space |
|-----------|------|-------|
| Access s[i] | O(1) | O(1) |
| Slice s[i:j] | O(k) | O(k) |
| Concatenation s1 + s2 | O(n+m) | O(n+m) |
| s.find (sub) | O(n*m) | O(1) |
| s.split() | O(n) | O(n) |
| '.join (list) | O(n) | O(n) |
| s.replace (old, new) | O(n) | O(n) |

## Common Mistakes to Avoid

1. **String concatenation in loops**
   \`\`\`python
   # ❌ O(n²)
   result = ""
   for c in string:
       result += c
   
   # ✅ O(n)
   result = '.join([c for c in string])
   \`\`\`

2. **Not considering edge cases**
   - Empty strings
   - Single character strings
   - All same characters
   - Special characters/spaces

3. **Forgetting strings are immutable**
   \`\`\`python
   # ❌ Won't work
   s = "hello"
   s[0] = 'H'  # TypeError
   
   # ✅ Create new string
   s = 'H' + s[1:]
   \`\`\`

4. **Case sensitivity**
   \`\`\`python
   # Always clarify: is "Abc" == "abc"?
   s1.lower() == s2.lower()  # Case insensitive
   \`\`\``,
};
