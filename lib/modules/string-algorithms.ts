import { Module } from '../types';

export const stringAlgorithmsModule: Module = {
  id: 'string-algorithms',
  title: 'String Algorithms',
  description:
    'Master essential string manipulation algorithms including pattern matching, string processing, and common interview patterns.',
  icon: '📝',
  sections: [
    {
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
len(s)  # 11

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
    chars.append(char)
result = ''.join(chars)  # O(n)

# Or use list comprehension
result = ''.join([char for char in "hello"])
\`\`\`

## Common String Patterns

### 1. Two Pointers
Moving from both ends toward center.

\`\`\`python
def is_palindrome(s):
    left, right = 0, len(s) - 1
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
def max_vowels_in_substring(s, k):
    """Find max vowels in any substring of length k"""
    vowels = set('aeiou')
    
    # Initial window
    current_vowels = sum(1 for c in s[:k] if c in vowels)
    max_vowels = current_vowels
    
    # Slide window
    for i in range(k, len(s)):
        # Remove left character
        if s[i-k] in vowels:
            current_vowels -= 1
        # Add right character
        if s[i] in vowels:
            current_vowels += 1
        max_vowels = max(max_vowels, current_vowels)
    
    return max_vowels
\`\`\`

### 3. Hash Map / Frequency Count
Track character frequencies.

\`\`\`python
from collections import Counter

def is_anagram(s1, s2):
    return Counter(s1) == Counter(s2)

# Or manually
def char_frequency(s):
    freq = {}
    for char in s:
        freq[char] = freq.get(char, 0) + 1
    return freq
\`\`\`

### 4. Dynamic Programming
Build solutions for subproblems.

\`\`\`python
def longest_palindromic_substring_length(s):
    n = len(s)
    if n == 0:
        return 0
    
    # dp[i][j] = True if s[i:j+1] is palindrome
    dp = [[False] * n for _ in range(n)]
    max_length = 1
    
    # Single characters
    for i in range(n):
        dp[i][i] = True
    
    # Two characters
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            dp[i][i + 1] = True
            max_length = 2
    
    # Longer substrings
    for length in range(3, n + 1):
        for i in range(n - length + 1):
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
| s.find(sub) | O(n*m) | O(1) |
| s.split() | O(n) | O(n) |
| ''.join(list) | O(n) | O(n) |
| s.replace(old, new) | O(n) | O(n) |

## Common Mistakes to Avoid

1. **String concatenation in loops**
   \`\`\`python
   # ❌ O(n²)
   result = ""
   for c in string:
       result += c
   
   # ✅ O(n)
   result = ''.join([c for c in string])
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
      quiz: [
        {
          id: 'q1',
          question:
            'Why is repeated string concatenation in a loop O(n²) instead of O(n)?',
          sampleAnswer:
            "Strings in Python are immutable, so each concatenation s += char creates a new string by copying all existing characters plus the new one. For n iterations: 1st copy=1, 2nd=2, 3rd=3, ..., nth=n characters. Total: 1+2+3+...+n = n(n+1)/2 = O(n²). Solution: Build a list and use ''.join(list) at the end, which is O(n) since it only copies characters once.",
          keyPoints: [
            'Strings are immutable in Python',
            'Each += creates new string and copies all chars',
            'n iterations: 1+2+3+...+n = O(n²)',
            "Solution: list.append() + ''.join() = O(n)",
            'join() only copies once',
          ],
        },
        {
          id: 'q2',
          question:
            'What is the difference between find() and index() for strings?',
          sampleAnswer:
            "Both search for substring position, but handle not found differently: find() returns -1 when substring not found (safe, no exception), while index() raises ValueError (need try/except). Use find() when substring might not exist and you want to check: if s.find(sub) != -1. Use index() when you're certain substring exists and want exception on error. Both are O(n*m) time complexity.",
          keyPoints: [
            'find(): returns -1 if not found (safe)',
            'index(): raises ValueError if not found',
            'Both O(n*m) time complexity',
            'find() better for uncertain existence',
            'index() better when expecting to find',
          ],
        },
        {
          id: 'q3',
          question:
            'What are the most common patterns for solving string problems and when should you use each?',
          sampleAnswer:
            'Four main patterns: (1) Two Pointers - use for palindromes, comparing from both ends, O(n) time and O(1) space, (2) Sliding Window - use for substring problems (longest substring without repeating chars, max vowels in k-length substring), maintains window state as it slides, O(n) time, (3) Hash Map / Frequency Count - use for anagrams, character frequency problems, group anagrams, O(n) time and space with Counter, (4) Dynamic Programming - use for longest common subsequence, edit distance, longest palindromic substring, builds table of subproblem solutions, O(n²) time and space. Choose based on problem structure: ends of string → two pointers, contiguous substring → sliding window, character counts → hash map, overlapping subproblems → DP.',
          keyPoints: [
            'Two pointers: palindromes, both ends, O(n) time O(1) space',
            'Sliding window: substring problems, O(n) time',
            'Hash map: anagrams, frequency, O(n) time and space',
            'DP: LCS, edit distance, longest palindrome, O(n²)',
            'Choose by structure: ends/middle/frequency/subproblems',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question:
            'What is the time complexity of string concatenation using + in a loop?',
          options: ['O(n)', 'O(n log n)', 'O(n²)', 'O(1)'],
          correctAnswer: 2,
          explanation:
            "String concatenation with + creates a new string each time, copying all previous characters. For n iterations: 1+2+3+...+n = O(n²). Use list.append() and ''.join() instead for O(n).",
        },
        {
          id: 'mc2',
          question:
            'Which method should you use to check if a substring exists, when you want to avoid exceptions?',
          options: ['index()', 'find()', 'search()', 'locate()'],
          correctAnswer: 1,
          explanation:
            'find() returns -1 if the substring is not found, avoiding exceptions. index() raises ValueError if not found.',
        },
        {
          id: 'mc3',
          question:
            'What is the space complexity of s[::-1] for reversing a string?',
          options: ['O(1)', 'O(log n)', 'O(n)', 'O(n²)'],
          correctAnswer: 2,
          explanation:
            'String slicing creates a new string, so s[::-1] creates a reversed copy with O(n) space complexity.',
        },
        {
          id: 'mc4',
          question:
            'Which operation is most efficient for building a string from many parts?',
          options: [
            's = s + part (repeated concatenation)',
            's += part (in-place concatenation)',
            "parts = []; parts.append(part); ''.join(parts)",
            'All are equally efficient',
          ],
          correctAnswer: 2,
          explanation:
            "Using a list and ''.join() is O(n) total. Repeated concatenation with + or += is O(n²) because strings are immutable and each operation creates a new string.",
        },
        {
          id: 'mc5',
          question: 'What does "hello"[1:4] return?',
          options: ['"hel"', '"ell"', '"ello"', '"hell"'],
          correctAnswer: 1,
          explanation:
            'String slicing [start:end] includes start index but excludes end index. "hello"[1:4] returns characters at indices 1, 2, 3: "ell".',
        },
      ],
    },
    {
      id: 'palindromes',
      title: 'Palindrome Patterns',
      content: `# Palindrome Patterns

A **palindrome** reads the same forward and backward. Palindrome problems are common in interviews and test multiple skills.

## Basic Palindrome Check

### Two Pointers Approach

\`\`\`python
def is_palindrome(s):
    """Check if string is palindrome - O(n) time, O(1) space"""
    left, right = 0, len(s) - 1
    
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    
    return True

print(is_palindrome("racecar"))  # True
print(is_palindrome("hello"))    # False
\`\`\`

### With Preprocessing (Common Interview Twist)

\`\`\`python
def is_palindrome_alphanumeric(s):
    """
    Check if palindrome ignoring case, spaces, punctuation.
    Example: "A man, a plan, a canal: Panama" -> True
    """
    # Clean string: only alphanumeric, lowercase
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    
    left, right = 0, len(cleaned) - 1
    while left < right:
        if cleaned[left] != cleaned[right]:
            return False
        left += 1
        right -= 1
    
    return True

# Or use Python's slicing (less efficient but concise)
def is_palindrome_simple(s):
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]
\`\`\`

## Longest Palindromic Substring

**Problem:** Find the longest palindrome within a string.

### Expand Around Center (O(n²))

\`\`\`python
def longest_palindrome_substring(s):
    """
    Find longest palindromic substring.
    Time: O(n²), Space: O(1)
    """
    if not s:
        return ""
    
    def expand_around_center(left, right):
        """Expand while characters match"""
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left + 1:right]  # Return the palindrome
    
    longest = ""
    for i in range(len(s)):
        # Odd length palindromes (center is single char)
        palindrome1 = expand_around_center(i, i)
        # Even length palindromes (center is between two chars)
        palindrome2 = expand_around_center(i, i + 1)
        
        # Update longest
        longest = max(longest, palindrome1, palindrome2, key=len)
    
    return longest

print(longest_palindrome_substring("babad"))  # "bab" or "aba"
print(longest_palindrome_substring("cbbd"))   # "bb"
\`\`\`

**Why check both odd and even?**
- Odd: "racecar" has center 'e'
- Even: "abba" has center between 'b' and 'b'

### Dynamic Programming Approach (O(n²))

\`\`\`python
def longest_palindrome_dp(s):
    """
    DP approach: dp[i][j] = True if s[i:j+1] is palindrome
    Time: O(n²), Space: O(n²)
    """
    n = len(s)
    if n == 0:
        return ""
    
    dp = [[False] * n for _ in range(n)]
    start, max_length = 0, 1
    
    # Single characters are palindromes
    for i in range(n):
        dp[i][i] = True
    
    # Check 2-character palindromes
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            dp[i][i + 1] = True
            start = i
            max_length = 2
    
    # Check longer palindromes
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            # s[i:j+1] is palindrome if:
            # 1. s[i] == s[j]
            # 2. s[i+1:j] is palindrome
            if s[i] == s[j] and dp[i + 1][j - 1]:
                dp[i][j] = True
                start = i
                max_length = length
    
    return s[start:start + max_length]
\`\`\`

## Palindrome Partitioning

**Problem:** Find all ways to partition string into palindromes.

\`\`\`python
def partition_palindromes(s):
    """
    Return all possible palindrome partitionings.
    Example: "aab" -> [["a","a","b"], ["aa","b"]]
    """
    def is_palindrome(substr):
        return substr == substr[::-1]
    
    def backtrack(start, path):
        if start == len(s):
            result.append(path[:])  # Found valid partition
            return
        
        for end in range(start + 1, len(s) + 1):
            substring = s[start:end]
            if is_palindrome(substring):
                path.append(substring)
                backtrack(end, path)
                path.pop()  # Backtrack
    
    result = []
    backtrack(0, [])
    return result

print(partition_palindromes("aab"))
# [['a', 'a', 'b'], ['aa', 'b']]
\`\`\`

## Minimum Cuts for Palindrome Partitioning

**Problem:** Find minimum cuts needed to partition into all palindromes.

\`\`\`python
def min_cut_palindrome(s):
    """
    Find minimum cuts for palindrome partition.
    Time: O(n²), Space: O(n²)
    """
    n = len(s)
    
    # is_pal[i][j] = True if s[i:j+1] is palindrome
    is_pal = [[False] * n for _ in range(n)]
    
    # Build palindrome table
    for i in range(n):
        is_pal[i][i] = True
    
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                is_pal[i][j] = (length == 2) or is_pal[i + 1][j - 1]
    
    # cuts[i] = minimum cuts for s[0:i+1]
    cuts = [0] * n
    
    for i in range(n):
        if is_pal[0][i]:
            cuts[i] = 0  # Whole substring is palindrome
        else:
            min_cuts = float('inf')
            for j in range(i):
                if is_pal[j + 1][i]:
                    min_cuts = min(min_cuts, cuts[j] + 1)
            cuts[i] = min_cuts
    
    return cuts[n - 1]

print(min_cut_palindrome("aab"))    # 1 (aa|b)
print(min_cut_palindrome("abc"))    # 2 (a|b|c)
\`\`\`

## Palindrome Number

**Problem:** Check if integer is palindrome (without converting to string).

\`\`\`python
def is_palindrome_number(x):
    """Check if number is palindrome without string conversion"""
    if x < 0:
        return False  # Negative numbers aren't palindromes
    
    # Reverse the number
    original = x
    reversed_num = 0
    
    while x > 0:
        reversed_num = reversed_num * 10 + x % 10
        x //= 10
    
    return original == reversed_num

# Optimized: only reverse half
def is_palindrome_number_optimized(x):
    """Only reverse second half"""
    if x < 0 or (x % 10 == 0 and x != 0):
        return False
    
    reversed_half = 0
    while x > reversed_half:
        reversed_half = reversed_half * 10 + x % 10
        x //= 10
    
    # x == reversed_half (odd length) or x == reversed_half // 10 (even)
    return x == reversed_half or x == reversed_half // 10
\`\`\`

## Interview Tips

1. **Clarify requirements:**
   - Case sensitive?
   - Ignore spaces/punctuation?
   - Empty string palindrome?

2. **Choose right approach:**
   - Simple check: Two pointers
   - Longest palindrome: Expand around center
   - Count/DP problems: Dynamic programming

3. **Edge cases:**
   - Empty string: Usually True
   - Single character: Always True
   - Two characters: Check if same
   - All same characters: Always True

4. **Optimization opportunities:**
   - Preprocessing (clean string once)
   - Early termination
   - Only process half (for number palindromes)`,
      quiz: [
        {
          id: 'q-pal1',
          question:
            'Explain the expand-around-center technique for finding palindromic substrings.',
          sampleAnswer:
            'Expand-around-center treats each character (and each pair of characters) as a potential palindrome center, then expands outward while characters match. For each position i, expand from (i,i) for odd-length palindromes and (i,i+1) for even-length. This finds all palindromic substrings in O(n²) time with O(1) space. Example: "aba" - from center \'b\', expand to find the entire palindrome.',
          keyPoints: [
            'Two cases: odd-length (single center) and even-length (two centers)',
            'Expand while left == right',
            'O(n²) time: n centers × O(n) expansion',
            'O(1) space - no extra arrays needed',
            'Better than brute force O(n³)',
          ],
        },
        {
          id: 'q-pal2',
          question:
            'How would you efficiently check if a string is a palindrome ignoring non-alphanumeric characters?',
          sampleAnswer:
            'Use two pointers from both ends, skipping non-alphanumeric characters: left=0, right=len-1. While left < right: skip non-alnum at left, skip at right, compare lower case. If mismatch, return False. This is O(n) time, O(1) space. Better than cleaning the string first (which takes O(n) extra space).',
          keyPoints: [
            'Two pointers: left and right',
            'Skip non-alphanumeric: if not s[left].isalnum(): left += 1',
            'Compare case-insensitive: s[left].lower() == s[right].lower()',
            'O(n) time, O(1) space',
            'More efficient than preprocessing',
          ],
        },
        {
          id: 'q-pal3',
          question:
            'What is the difference between palindromic substring and palindromic subsequence?',
          sampleAnswer:
            'Substring must be contiguous - consecutive characters in order (e.g., "aba" in "xabay"). Subsequence can skip characters but must maintain order (e.g., "aba" from "a__b__a" where underscores are skipped chars). Finding longest palindromic substring is O(n²) with expand-around-center. Finding longest palindromic subsequence needs DP: O(n²) time and space, similar to LCS.',
          keyPoints: [
            'Substring: contiguous characters',
            'Subsequence: can skip characters, maintain order',
            'Substring example: "aba" in "xabay"',
            'Subsequence example: "aba" in "aebfca"',
            'Different algorithms: expand-around vs DP',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc-pal1',
          question:
            'What is the time complexity of checking if a string is a palindrome using two pointers?',
          options: ['O(log n)', 'O(n)', 'O(n log n)', 'O(n²)'],
          correctAnswer: 1,
          explanation:
            'Two pointers traverse the string from both ends meeting in the middle, comparing n/2 pairs of characters, which is O(n) time.',
        },
        {
          id: 'mc-pal2',
          question:
            'For finding all palindromic substrings, why is expand-around-center better than brute force?',
          options: [
            'It uses less space',
            'It reduces time complexity from O(n³) to O(n²)',
            'It only works for even-length palindromes',
            'It requires preprocessing',
          ],
          correctAnswer: 1,
          explanation:
            'Brute force checks all O(n²) substrings, each taking O(n) to verify (total O(n³)). Expand-around-center has O(n) centers and O(n) expansion per center, giving O(n²).',
        },
        {
          id: 'mc-pal3',
          question:
            'How many potential centers should you check for palindromes in a string of length n?',
          options: ['n centers', '2n-1 centers', 'n² centers', 'n/2 centers'],
          correctAnswer: 1,
          explanation:
            'You need to check n centers for odd-length palindromes (between characters) and n-1 centers for even-length (at characters), totaling 2n-1 centers.',
        },
        {
          id: 'mc-pal4',
          question:
            'What is the best approach to find the longest palindromic subsequence?',
          options: [
            'Two pointers',
            'Expand around center',
            'Dynamic Programming',
            'Sliding window',
          ],
          correctAnswer: 2,
          explanation:
            'Longest palindromic subsequence requires DP with O(n²) time and space, similar to Longest Common Subsequence between s and reverse(s). Two pointers and expand-around-center work for substrings, not subsequences.',
        },
        {
          id: 'mc-pal5',
          question:
            'When checking "A man, a plan, a canal: Panama" as a palindrome, what should you do?',
          options: [
            'Compare as-is',
            'Remove spaces only',
            'Remove spaces and punctuation, compare case-insensitive',
            'Reverse and compare',
          ],
          correctAnswer: 2,
          explanation:
            'For phrase palindromes, you must remove non-alphanumeric characters and compare case-insensitive: "amanaplanacanalpanama" reads the same forward and backward.',
        },
      ],
    },
    {
      id: 'anagrams',
      title: 'Anagram Patterns',
      content: `# Anagram Patterns

An **anagram** is formed by rearranging letters of another word using all original letters exactly once.

## Basic Anagram Check

### Using Sorting (O(n log n))

\`\`\`python
def is_anagram_sort(s1, s2):
    """
    Check if two strings are anagrams using sorting.
    Time: O(n log n), Space: O(n)
    """
    if len(s1) != len(s2):
        return False
    return sorted(s1) == sorted(s2)

print(is_anagram_sort("listen", "silent"))  # True
print(is_anagram_sort("hello", "world"))    # False
\`\`\`

### Using Hash Map (O(n))

\`\`\`python
def is_anagram_hash(s1, s2):
    """
    Check using character frequency count.
    Time: O(n), Space: O(1) - at most 26 characters
    """
    if len(s1) != len(s2):
        return False
    
    from collections import Counter
    return Counter(s1) == Counter(s2)

# Manual implementation
def is_anagram_manual(s1, s2):
    if len(s1) != len(s2):
        return False
    
    char_count = {}
    
    # Count characters in s1
    for char in s1:
        char_count[char] = char_count.get(char, 0) + 1
    
    # Decrement for characters in s2
    for char in s2:
        if char not in char_count:
            return False
        char_count[char] -= 1
        if char_count[char] < 0:
            return False
    
    return all(count == 0 for count in char_count.values())
\`\`\`

### Using Array (O(n) - For Lowercase Only)

\`\`\`python
def is_anagram_array(s1, s2):
    """
    Using fixed-size array for lowercase letters.
    Time: O(n), Space: O(1) - fixed 26 slots
    """
    if len(s1) != len(s2):
        return False
    
    # Array for 26 lowercase letters
    counts = [0] * 26
    
    for i in range(len(s1)):
        counts[ord(s1[i]) - ord('a')] += 1
        counts[ord(s2[i]) - ord('a')] -= 1
    
    return all(count == 0 for count in counts)
\`\`\`

## Group Anagrams

**Problem:** Group strings that are anagrams of each other.

\`\`\`python
from collections import defaultdict

def group_anagrams(words):
    """
    Group anagrams together.
    Example: ["eat","tea","tan","ate","nat","bat"]
    -> [["eat","tea","ate"],["tan","nat"],["bat"]]
    
    Time: O(n * k log k) where n = number of words, k = max word length
    Space: O(n * k)
    """
    anagrams = defaultdict(list)
    
    for word in words:
        # Use sorted word as key
        key = ''.join(sorted(word))
        anagrams[key].append(word)
    
    return list(anagrams.values())

words = ["eat", "tea", "tan", "ate", "nat", "bat"]
print(group_anagrams(words))
# [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]
\`\`\`

**Optimization for Lowercase Only:**

\`\`\`python
def group_anagrams_optimized(words):
    """
    Use character count tuple as key (O(n * k) instead of O(n * k log k))
    """
    anagrams = defaultdict(list)
    
    for word in words:
        # Create count signature
        count = [0] * 26
        for char in word:
            count[ord(char) - ord('a')] += 1
        
        # Tuple can be dictionary key
        key = tuple(count)
        anagrams[key].append(word)
    
    return list(anagrams.values())
\`\`\`

## Find All Anagrams in String (Sliding Window)

**Problem:** Find all start indices of anagrams of \`p\` in \`s\`.

\`\`\`python
def find_anagrams(s, p):
    """
    Find all anagram occurrences.
    Example: s = "cbaebabacd", p = "abc" -> [0, 6]
    
    Time: O(n), Space: O(1)
    """
    if len(p) > len(s):
        return []
    
    from collections import Counter
    
    p_count = Counter(p)
    window_count = Counter(s[:len(p)])
    result = []
    
    # Check first window
    if window_count == p_count:
        result.append(0)
    
    # Slide window
    for i in range(len(p), len(s)):
        # Add new character
        window_count[s[i]] += 1
        
        # Remove old character
        left_char = s[i - len(p)]
        window_count[left_char] -= 1
        if window_count[left_char] == 0:
            del window_count[left_char]
        
        # Check if anagram
        if window_count == p_count:
            result.append(i - len(p) + 1)
    
    return result

s = "cbaebabacd"
p = "abc"
print(find_anagrams(s, p))  # [0, 6]
# "cba" at index 0, "bac" at index 6
\`\`\`

## Minimum Window Substring with Anagram

**Problem:** Find minimum window in \`s\` that contains all characters of \`t\`.

\`\`\`python
from collections import Counter

def min_window(s, t):
    """
    Find minimum window substring containing all chars of t.
    Example: s = "ADOBECODEBANC", t = "ABC" -> "BANC"
    
    Time: O(n + m), Space: O(m)
    """
    if not s or not t or len(s) < len(t):
        return ""
    
    t_count = Counter(t)
    required = len(t_count)  # Unique characters in t
    formed = 0  # Unique characters in window matching requirement
    
    window_counts = {}
    min_len = float('inf')
    min_left = 0
    
    left, right = 0, 0
    
    while right < len(s):
        # Expand window
        char = s[right]
        window_counts[char] = window_counts.get(char, 0) + 1
        
        if char in t_count and window_counts[char] == t_count[char]:
            formed += 1
        
        # Contract window
        while left <= right and formed == required:
            # Update result
            if right - left + 1 < min_len:
                min_len = right - left + 1
                min_left = left
            
            # Remove left character
            char = s[left]
            window_counts[char] -= 1
            if char in t_count and window_counts[char] < t_count[char]:
                formed -= 1
            
            left += 1
        
        right += 1
    
    return "" if min_len == float('inf') else s[min_left:min_left + min_len]

s = "ADOBECODEBANC"
t = "ABC"
print(min_window(s, t))  # "BANC"
\`\`\`

## Permutation in String

**Problem:** Check if \`s2\` contains permutation of \`s1\`.

\`\`\`python
def check_inclusion(s1, s2):
    """
    Check if s2 contains permutation of s1.
    Example: s1 = "ab", s2 = "eidbaooo" -> True
    
    Time: O(n), Space: O(1)
    """
    if len(s1) > len(s2):
        return False
    
    from collections import Counter
    
    s1_count = Counter(s1)
    window_count = Counter(s2[:len(s1)])
    
    if window_count == s1_count:
        return True
    
    for i in range(len(s1), len(s2)):
        # Add new char
        window_count[s2[i]] += 1
        
        # Remove old char
        old_char = s2[i - len(s1)]
        window_count[old_char] -= 1
        if window_count[old_char] == 0:
            del window_count[old_char]
        
        if window_count == s1_count:
            return True
    
    return False
\`\`\`

## Key Patterns

1. **Frequency Counting:**
   - Use Counter or hash map
   - O(1) space for fixed alphabet (26 letters)

2. **Sorting as Signature:**
   - Sorted string uniquely identifies anagrams
   - O(k log k) per string

3. **Sliding Window:**
   - For "find all anagrams" problems
   - Maintain frequency count in window
   - O(n) time

4. **Character Array (Lowercase Only):**
   - Fixed 26-slot array
   - Faster than hash map
   - Use ord(char) - ord('a') for index`,
      quiz: [
        {
          id: 'q-ana1',
          question:
            'What is the most efficient way to check if two strings are anagrams?',
          sampleAnswer:
            "Use a frequency counter (hash map or array). Count characters in first string, then decrement for second string. If all counts are zero, they're anagrams. Time: O(n+m), Space: O(1) for fixed alphabet or O(k) for unique chars. This is better than sorting O(n log n). For lowercase only, use 26-element array: counts[ord(c) - ord('a')] for O(1) lookups.",
          keyPoints: [
            'Hash map or array for character frequencies',
            'Count first string, decrement for second',
            'O(n+m) time vs O(n log n) for sorting',
            'O(1) space for fixed alphabet (26 letters)',
            'Check if all counts return to zero',
          ],
        },
        {
          id: 'q-ana2',
          question:
            'How would you find all anagrams of a pattern in a string (sliding window)?',
          sampleAnswer:
            'Use sliding window with character frequency map. Create pattern frequency map. Slide a window of pattern.length through string, maintaining window frequency. When window frequency matches pattern frequency, record start index. Optimization: track "matches" count - when matches == 26 (or unique chars), it\'s an anagram. Time: O(n), Space: O(1) for fixed alphabet.',
          keyPoints: [
            'Sliding window of length = pattern.length',
            'Maintain frequency map for current window',
            'Compare window frequency with pattern frequency',
            'Track matches count for optimization',
            'O(n) time with single pass',
          ],
        },
        {
          id: 'q-ana3',
          question:
            "What's the difference between checking anagrams with sorting vs hash map?",
          sampleAnswer:
            'Sorting: Sort both strings, compare if equal. Time: O(n log n), Space: O(1) or O(n) depending on sort. Hash map: Count character frequencies, compare counts. Time: O(n), Space: O(k) for k unique chars. Hash map is faster for long strings. Sorting is simpler to implement and works for any characters. For interviews, mention both but implement hash map for better complexity.',
          keyPoints: [
            'Sorting: O(n log n) time, simpler code',
            'Hash map: O(n) time, better complexity',
            'Sorting: sort both and compare',
            'Hash map: count and compare frequencies',
            'Trade-off: simplicity vs performance',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc-ana1',
          question:
            'What is the time complexity of checking if two strings are anagrams using a frequency counter?',
          options: ['O(1)', 'O(n)', 'O(n log n)', 'O(n²)'],
          correctAnswer: 1,
          explanation:
            'Using a hash map or array to count character frequencies requires O(n+m) time where n and m are string lengths. This simplifies to O(n).',
        },
        {
          id: 'mc-ana2',
          question:
            'Which data structure is most efficient for grouping anagrams?',
          options: [
            'Array',
            'Hash Map (key = sorted string)',
            'Binary Search Tree',
            'Linked List',
          ],
          correctAnswer: 1,
          explanation:
            'A hash map with sorted string as key groups anagrams in O(n * k log k) time where n is number of strings and k is average string length. All anagrams map to the same key.',
        },
        {
          id: 'mc-ana3',
          question:
            'For finding all anagram starting indices in a string, what is the optimal approach?',
          options: [
            'Check every substring individually',
            'Use a sliding window with frequency counter',
            'Sort all substrings and compare',
            'Use dynamic programming',
          ],
          correctAnswer: 1,
          explanation:
            'Sliding window with frequency counter achieves O(n) time by maintaining character counts as the window slides, avoiding redundant comparisons.',
        },
        {
          id: 'mc-ana4',
          question:
            'What is the space complexity of checking anagrams with a character frequency array for lowercase letters only?',
          options: [
            'O(1)',
            'O(n)',
            'O(k) where k is unique characters',
            'O(26)',
          ],
          correctAnswer: 0,
          explanation:
            'For lowercase letters only, we use a fixed 26-element array regardless of input size, which is O(1) space.',
        },
        {
          id: 'mc-ana5',
          question:
            'When comparing two strings as anagrams, what must be checked first?',
          options: [
            'If they start with the same character',
            'If they have the same length',
            'If they are already sorted',
            'If they contain only lowercase letters',
          ],
          correctAnswer: 1,
          explanation:
            'Strings of different lengths cannot be anagrams. This is an O(1) check that should be done first to avoid unnecessary processing.',
        },
      ],
    },
    {
      id: 'substring-search',
      title: 'Substring Search & Pattern Matching',
      content: `# Substring Search & Pattern Matching

Finding patterns in text is fundamental. We'll cover naive, KMP, and Rabin-Karp algorithms.

## Naive Substring Search

Simple but O(n * m) in worst case.

\`\`\`python
def naive_search(text, pattern):
    """
    Find all occurrences of pattern in text.
    Time: O((n-m+1) * m) = O(n * m) worst case
    Space: O(1)
    """
    n, m = len(text), len(pattern)
    positions = []
    
    for i in range(n - m + 1):
        # Check if pattern matches at position i
        match = True
        for j in range(m):
            if text[i + j] != pattern[j]:
                match = False
                break
        if match:
            positions.append(i)
    
    return positions

text = "AABAACAADAABAABA"
pattern = "AABA"
print(naive_search(text, pattern))  # [0, 9, 12]
\`\`\`

**Worst case:** Text = "AAAA...A", Pattern = "AAA...AB"
- Must check every position
- Each check does m-1 comparisons before failing

## Rabin-Karp (Rolling Hash)

Use hashing to quickly check potential matches.

\`\`\`python
def rabin_karp(text, pattern):
    """
    Find pattern using rolling hash.
    Time: O(n + m) average, O(n * m) worst
    Space: O(1)
    """
    n, m = len(text), len(pattern)
    if m > n:
        return []
    
    BASE = 256  # Number of characters in alphabet
    MOD = 10**9 + 7  # Large prime
    
    positions = []
    pattern_hash = 0
    text_hash = 0
    h = 1  # BASE^(m-1) % MOD
    
    # Calculate h = BASE^(m-1) % MOD
    for i in range(m - 1):
        h = (h * BASE) % MOD
    
    # Calculate initial hashes
    for i in range(m):
        pattern_hash = (BASE * pattern_hash + ord(pattern[i])) % MOD
        text_hash = (BASE * text_hash + ord(text[i])) % MOD
    
    # Slide pattern over text
    for i in range(n - m + 1):
        # If hashes match, verify actual strings (avoid false positives)
        if pattern_hash == text_hash:
            if text[i:i+m] == pattern:
                positions.append(i)
        
        # Calculate next hash (rolling hash)
        if i < n - m:
            # Remove leading digit, add trailing digit
            text_hash = (BASE * (text_hash - ord(text[i]) * h) + ord(text[i + m])) % MOD
            
            # Handle negative values
            if text_hash < 0:
                text_hash += MOD
    
    return positions

text = "AABAACAADAABAABA"
pattern = "AABA"
print(rabin_karp(text, pattern))  # [0, 9, 12]
\`\`\`

**Key insight:** Rolling hash computed in O(1)
\`\`\`
Hash(text[i+1:i+m+1]) = (Hash(text[i:i+m]) - text[i] * BASE^(m-1)) * BASE + text[i+m]
\`\`\`

## KMP (Knuth-Morris-Pratt)

Never re-check matched characters using "failure function".

\`\`\`python
def kmp_search(text, pattern):
    """
    KMP algorithm - never backtracks in text.
    Time: O(n + m), Space: O(m)
    """
    def compute_lps(pattern):
        """
        Compute Longest Proper Prefix which is also Suffix.
        lps[i] = length of longest proper prefix of pattern[0:i+1]
                 which is also a suffix
        """
        m = len(pattern)
        lps = [0] * m
        length = 0  # Length of previous longest prefix suffix
        i = 1
        
        while i < m:
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]  # Try shorter prefix
                else:
                    lps[i] = 0
                    i += 1
        
        return lps
    
    n, m = len(text), len(pattern)
    if m == 0:
        return [0]
    if m > n:
        return []
    
    lps = compute_lps(pattern)
    positions = []
    
    i = 0  # Index for text
    j = 0  # Index for pattern
    
    while i < n:
        if text[i] == pattern[j]:
            i += 1
            j += 1
        
        if j == m:
            # Pattern found
            positions.append(i - j)
            j = lps[j - 1]
        elif i < n and text[i] != pattern[j]:
            if j != 0:
                j = lps[j - 1]  # Don't match lps[0..lps[j-1]] characters
            else:
                i += 1
    
    return positions

text = "AABAACAADAABAABA"
pattern = "AABA"
print(kmp_search(text, pattern))  # [0, 9, 12]

# Example of LPS array:
# Pattern: "AABAABA"
# LPS:     [0,1,0,1,2,3,4]
\`\`\`

**LPS (Longest Proper Prefix which is Suffix) Example:**
\`\`\`
Pattern:  A  A  B  A  A  B  A
Index:    0  1  2  3  4  5  6
LPS:      0  1  0  1  2  3  4

Explanation:
Index 0: "" has no proper prefix
Index 1: "AA" -> longest is "A" (length 1)
Index 2: "AAB" -> no match (length 0)
Index 3: "AABA" -> longest is "A" (length 1)
Index 4: "AABAA" -> longest is "AA" (length 2)
Index 5: "AABAAB" -> longest is "AAB" (length 3)
Index 6: "AABAABA" -> longest is "AABA" (length 4)
\`\`\`

**Why KMP is Faster:**
- Never backtracks in text (i never decreases)
- Uses LPS to skip unnecessary comparisons
- Guarantees O(n + m) time

## Algorithm Comparison

| Algorithm | Time (Avg) | Time (Worst) | Space | Use Case |
|-----------|------------|--------------|-------|----------|
| Naive | O(n * m) | O(n * m) | O(1) | Short patterns, simple implementation |
| Rabin-Karp | O(n + m) | O(n * m) | O(1) | Multiple pattern search |
| KMP | O(n + m) | O(n + m) | O(m) | Single pattern, guaranteed performance |

## Built-in Python Methods

\`\`\`python
text = "hello world"

# Find first occurrence
pos = text.find("world")  # 6 (returns -1 if not found)
pos = text.index("world") # 6 (raises ValueError if not found)

# Count occurrences
count = text.count("l")  # 3

# Check existence
exists = "world" in text  # True

# Replace
new_text = text.replace("world", "python")  # "hello python"
\`\`\`

**When to use each:**
- **Naive:** Simple, short patterns, or when built-in methods suffice
- **Rabin-Karp:** Multiple patterns, or preprocessing is expensive
- **KMP:** Guaranteed linear time, or pattern has repeating structure`,
      quiz: [
        {
          id: 'q-sub1',
          question:
            'Explain the Rabin-Karp rolling hash technique for pattern matching.',
          sampleAnswer:
            'Rabin-Karp uses a rolling hash function to compute hash values for all substrings of length m in O(1) per substring after initial O(m) computation. Hash formula: hash = (c1*d^(m-1) + c2*d^(m-2) + ... + cm) % q, where d is base (e.g., 256) and q is prime. To "roll" from position i to i+1: remove first character contribution, shift, add new character. If hashes match, verify with character-by-character check (avoid hash collisions). Average O(n+m), worst O(nm).',
          keyPoints: [
            'Rolling hash: O(1) to compute next substring hash',
            'Hash formula: polynomial with base d, modulo prime q',
            'Remove old char: hash = (hash - c1*d^(m-1)) / d',
            'Add new char: hash = hash * d + new_c',
            'Verify matches to handle collisions',
          ],
        },
        {
          id: 'q-sub2',
          question:
            'What is the LPS (Longest Proper Prefix which is also Suffix) array in KMP and how is it used?',
          sampleAnswer:
            'LPS[i] stores length of longest proper prefix of pattern[0...i] that is also a suffix. Used to avoid re-comparing characters after mismatch. When mismatch at pattern[j], we know pattern[0...j-1] matched, so we can skip to position LPS[j-1] instead of starting over. This is because pattern[0...LPS[j-1]] equals pattern[j-LPS[j-1]...j-1]. Example: pattern="ABABC", LPS=[0,0,1,2,0]. Guarantees O(n+m) time.',
          keyPoints: [
            'LPS[i] = length of longest proper prefix = suffix',
            'Proper prefix: excludes full string',
            'Used to skip redundant comparisons',
            'On mismatch at j, jump to LPS[j-1]',
            'Preprocessing: O(m), Matching: O(n)',
          ],
        },
        {
          id: 'q-sub3',
          question:
            'Why is naive substring search O(nm) and when is it acceptable to use?',
          sampleAnswer:
            'Naive search checks every position in text (n positions), and for each position compares up to m characters of the pattern. In worst case (e.g., text="aaaaa", pattern="aaab"), all m characters are compared at each of n positions, giving O(nm). It\'s acceptable when: pattern and text are short, pattern has no repeating structure (making worst case rare), or simplicity is prioritized over performance. For most real-world text (English), average case is much better than worst case.',
          keyPoints: [
            'Checks n positions in text',
            'Compares up to m chars per position',
            'Worst case: O(nm) with repeating characters',
            'Average case better for random text',
            'Use when simplicity matters or inputs are small',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc-sub1',
          question:
            'What is the time complexity of KMP pattern matching algorithm?',
          options: ['O(n)', 'O(m)', 'O(n + m)', 'O(nm)'],
          correctAnswer: 2,
          explanation:
            'KMP guarantees O(n+m) time: O(m) to build LPS array and O(n) to search the text. Each character in text is examined at most once.',
        },
        {
          id: 'mc-sub2',
          question:
            'What advantage does Rabin-Karp have over naive string matching?',
          options: [
            'Guaranteed O(n+m) worst case',
            'Can search for multiple patterns simultaneously',
            'Uses less space',
            'Simpler to implement',
          ],
          correctAnswer: 1,
          explanation:
            'Rabin-Karp can search for multiple patterns by computing hashes for all patterns and checking against each substring hash. This is efficient for multi-pattern matching.',
        },
        {
          id: 'mc-sub3',
          question:
            'In Rabin-Karp, why do we need to verify matches after hash collision?',
          options: [
            'To reduce time complexity',
            'Because different strings can have the same hash value',
            'To save memory',
            'Because rolling hash is inaccurate',
          ],
          correctAnswer: 1,
          explanation:
            'Hash collisions occur when different strings produce the same hash value. We must verify character-by-character to confirm a true match. This is called "spurious hit" handling.',
        },
        {
          id: 'mc-sub4',
          question: 'What is the purpose of the LPS array in KMP algorithm?',
          options: [
            'To store all pattern occurrences',
            'To skip redundant comparisons after a mismatch',
            'To calculate pattern hash',
            'To store text positions',
          ],
          correctAnswer: 1,
          explanation:
            'The LPS (Longest Proper Prefix which is also Suffix) array tells us how many characters to skip when a mismatch occurs, avoiding the need to restart pattern matching from the beginning.',
        },
        {
          id: 'mc-sub5',
          question:
            'Which substring search algorithm is best for searching multiple patterns in the same text?',
          options: ['Naive search', 'KMP', 'Rabin-Karp', 'Binary search'],
          correctAnswer: 2,
          explanation:
            'Rabin-Karp is most efficient for multiple patterns because you can compute all pattern hashes once, then check each text substring hash against all pattern hashes in one pass.',
        },
      ],
    },
  ],
  keyTakeaways: [
    'Strings are immutable—avoid repeated concatenation in loops, use list + join instead',
    'Two pointers effective for palindromes and comparisons',
    'Hash maps (Counter) essential for anagrams and character frequency problems',
    'Sliding window for substring problems: maintains state while moving through string',
    'Rabin-Karp uses rolling hash for O(n+m) average pattern matching',
    'KMP guarantees O(n+m) by using LPS array to avoid backtracking',
    'Always clarify: case sensitivity, spaces/punctuation, empty string handling',
    'For anagrams: sorted string as key (O(k log k)) or count tuple (O(k))',
  ],
  relatedProblems: [
    'longest-palindrome',
    'group-anagrams',
    'min-window-substring',
  ],
};
