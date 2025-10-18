/**
 * Anagram Patterns Section
 */

export const anagramsSection = {
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
};
