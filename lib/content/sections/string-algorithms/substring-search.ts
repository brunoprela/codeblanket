/**
 * Substring Search & Pattern Matching Section
 */

export const substringsearchSection = {
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
};
