/**
 * Palindrome Patterns Section
 */

export const palindromesSection = {
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
    cleaned = '.join(c.lower() for c in s if c.isalnum())
    
    left, right = 0, len(cleaned) - 1
    while left < right:
        if cleaned[left] != cleaned[right]:
            return False
        left += 1
        right -= 1
    
    return True

# Or use Python's slicing (less efficient but concise)
def is_palindrome_simple(s):
    cleaned = '.join(c.lower() for c in s if c.isalnum())
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
};
